import type { FastifyInstance } from 'fastify'
import { randomUUID } from 'node:crypto'
import { z } from 'zod'
import '../fastify-types.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiParameterRef,
  openApiSchemaRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import { zodRequestValidation } from './utils.js'
import {
  buildHealthExportSnapshot,
  buildHealthSnapshot,
  buildReadinessSnapshot,
  formatHealthReport,
  now,
} from '../health-support.js'
import {
  buildSseResponseHeaders,
  registerSseDisconnectHandler,
} from '../sse-response.js'

const healthComponentStatusSchema = z.object({
  status: z.string(),
  details: z.string().optional(),
}).passthrough()

const readinessCheckSchema = healthComponentStatusSchema.extend({
  critical: z.boolean(),
})

const healthMemorySchema = z.object({
  rss: z.number(),
  heap: z.number(),
})

const healthNowSchema = z.object({
  version: z.string(),
  apiVersion: z.number().int().positive(),
  uptime: z.number(),
  timestamp: z.string(),
})

const healthExportQuerySchema = z.object({
  format: z.enum(['markdown', 'json']).optional(),
})

const healthExportSnapshotSchema = z.object({
  generatedAt: z.string(),
  health: healthNowSchema.extend({
    status: z.string(),
    components: z.record(healthComponentStatusSchema),
    memory: healthMemorySchema,
  }),
  readiness: healthNowSchema.extend({
    status: z.string(),
    checks: z.record(readinessCheckSchema),
  }),
})

const healthWatchPayloadSchema = z.union([
  z.object({
    type: z.literal('snapshot'),
    health: healthExportSnapshotSchema.shape.health,
    report: healthExportSnapshotSchema,
  }),
  z.object({
    type: z.literal('heartbeat'),
    timestamp: z.string(),
  }),
])

export const healthOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    HealthComponentStatus: healthComponentStatusSchema,
    ReadinessCheck: readinessCheckSchema,
    HealthMemory: healthMemorySchema,
    HealthExportSnapshot: healthExportSnapshotSchema,
    HealthExportResponse: z.object({
      data: healthExportSnapshotSchema,
    }),
    HealthWatchPayload: healthWatchPayloadSchema,
    HealthResponse: z.object({
      data: healthNowSchema.extend({
        status: z.string(),
        components: z.record(healthComponentStatusSchema),
        memory: healthMemorySchema,
      }),
    }),
    LivenessResponse: z.object({
      data: healthNowSchema.extend({
        status: z.literal('ok'),
      }),
    }),
    ReadinessResponse: z.object({
      data: healthNowSchema.extend({
        status: z.string(),
        checks: z.record(readinessCheckSchema),
      }),
    }),
  },
  parameters: {
    HealthExportFormatParam: {
      name: 'format',
      in: 'query',
      schema: healthExportQuerySchema.shape.format,
    },
  },
})

export const healthOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/health': {
    get: {
      summary: 'Health check',
      tags: ['System'],
      responses: { 200: openApiJsonResponseRef('HealthResponse') },
      security: [],
    },
  },
  '/api/v1/health/live': {
    get: {
      summary: 'Liveness check',
      tags: ['System'],
      responses: { 200: openApiJsonResponseRef('LivenessResponse', 'Process is live') },
      security: [],
    },
  },
  '/api/v1/health/watch': {
    get: {
      summary: 'Watch health snapshots',
      tags: ['System'],
      responses: {
        200: {
          description: 'Server-sent health snapshots',
          content: {
            'text/event-stream': {
              schema: openApiSchemaRef('HealthWatchPayload'),
            },
          },
        },
      },
    },
  },
  '/api/v1/health/export': {
    get: {
      summary: 'Export health snapshot',
      tags: ['System'],
      parameters: [openApiParameterRef('HealthExportFormatParam')],
      responses: {
        200: {
          description: 'Health snapshot export',
          content: {
            'application/json': {
              schema: openApiSchemaRef('HealthExportResponse'),
            },
            'text/markdown': {
              schema: {
                type: 'string',
              },
            },
          },
        },
      },
    },
  },
  '/api/v1/health/ready': {
    get: {
      summary: 'Readiness check',
      tags: ['System'],
      responses: {
        200: openApiJsonResponseRef('ReadinessResponse', 'Runtime is ready'),
        503: openApiJsonResponseRef('ReadinessResponse', 'Runtime is not ready'),
      },
      security: [],
    },
  },
}

export async function healthRoutes(app: FastifyInstance) {
  app.get('/health', async () => {
    return {
      data: await buildHealthSnapshot(app),
    }
  })

  app.get('/health/watch', async (req, reply) => {
    reply.hijack()
    reply.raw.writeHead(200, buildSseResponseHeaders(req, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Request-ID': req.requestId ?? randomUUID(),
    }))

    let closed = false
    const send = (payload: z.infer<typeof healthWatchPayloadSchema>) => {
      if (closed) return
      reply.raw.write(`event: health\ndata: ${JSON.stringify(payload)}\n\n`)
    }
    const sendSnapshot = async () => {
      const snapshot = await buildHealthExportSnapshot(app)
      send({
        type: 'snapshot',
        health: snapshot.health,
        report: snapshot,
      })
    }
    const snapshotTimer = setInterval(() => {
      void sendSnapshot().catch(() => {
        send({
          type: 'heartbeat',
          timestamp: new Date().toISOString(),
        })
      })
    }, 15_000)
    snapshotTimer.unref?.()
    const heartbeatTimer = setInterval(() => {
      send({
        type: 'heartbeat',
        timestamp: new Date().toISOString(),
      })
    }, 5_000)
    heartbeatTimer.unref?.()
    const close = () => {
      if (closed) return
      closed = true
      clearInterval(snapshotTimer)
      clearInterval(heartbeatTimer)
      if (!reply.raw.destroyed && !reply.raw.writableEnded) {
        reply.raw.end()
      }
    }

    registerSseDisconnectHandler(req, reply, close)
    void sendSnapshot().catch(() => {
      send({
        type: 'heartbeat',
        timestamp: new Date().toISOString(),
      })
    })
  })

  app.get('/health/live', async () => ({
    data: {
      status: 'ok',
      ...now(),
    },
  }))

  app.get<{ Querystring: z.infer<typeof healthExportQuerySchema> }>('/health/export', {
    preValidation: zodRequestValidation({
      query: {
        schema: healthExportQuerySchema,
        message: 'Invalid health export format',
      },
    }),
  }, async (request, reply) => {
    const snapshot = await buildHealthExportSnapshot(app)
    const query = request.query
    const format = query.format ?? 'markdown'

    if (format === 'json') {
      return {
        data: snapshot,
      }
    }

    reply.header('Content-Type', 'text/markdown')
    return formatHealthReport(snapshot)
  })

  app.get('/health/ready', async (_request, reply) => {
    const snapshot = await buildReadinessSnapshot(app)
    if (snapshot.status !== 'ok') {
      reply.status(503)
    }

    return {
      data: snapshot,
    }
  })
}
