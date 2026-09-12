import { randomUUID } from 'node:crypto'
import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import type { ToolResult } from '../../tools/registry.js'
import { zodRequestValidation } from './utils.js'

const serviceIdParamsSchema = z.object({
  id: z.string().trim().min(1),
})

const numberQueryValue = z.preprocess(
  (value) => value === '' ? undefined : value,
  z.coerce.number().finite().optional(),
)

const booleanQueryValue = z.preprocess((value) => {
  if (value === undefined || value === '') return undefined
  if (value === true || value === 'true') return true
  if (value === false || value === 'false') return false
  return value
}, z.boolean().optional())

const serviceStartBodySchema = z.object({}).passthrough()

const serviceLogsQuerySchema = z.object({
  stdoutOffset: numberQueryValue,
  stderrOffset: numberQueryValue,
  limitBytes: numberQueryValue,
  tailBytes: numberQueryValue,
  followMs: numberQueryValue,
  pollIntervalMs: numberQueryValue,
})

const serviceStopBodySchema = z.preprocess(
  (value) => value ?? {},
  z.object({
    signal: z.string().trim().min(1).optional(),
    timeoutMs: z.number().finite().optional(),
  }),
)

const serviceRemoveQuerySchema = z.object({
  force: booleanQueryValue,
  deleteLogs: booleanQueryValue,
})

const nativeInstallBodySchema = z.object({}).passthrough()

const nativeControlBodySchema = z.preprocess(
  (value) => value ?? {},
  z.object({
    start: z.boolean().optional(),
    stop: z.boolean().optional(),
    reload: z.boolean().optional(),
  }),
)

type ServiceIdParams = z.infer<typeof serviceIdParamsSchema>
type ServiceStartBody = z.infer<typeof serviceStartBodySchema>
type ServiceLogsQuery = z.infer<typeof serviceLogsQuerySchema>
type ServiceStopBody = z.infer<typeof serviceStopBodySchema>
type ServiceRemoveQuery = z.infer<typeof serviceRemoveQuerySchema>
type NativeInstallBody = z.infer<typeof nativeInstallBodySchema>
type NativeControlBody = z.infer<typeof nativeControlBodySchema>

async function runServiceTool(
  app: FastifyInstance,
  toolName: string,
  input: Record<string, unknown> = {},
): Promise<ToolResult | { status: 'error'; output: string; durationMs: number; code: string }> {
  const runtime = app.runtime
  if (!runtime) {
    return {
      status: 'error',
      output: 'Runtime not initialized',
      durationMs: 0,
      code: 'SERVICE_UNAVAILABLE',
    }
  }
  const tool = runtime.toolRegistry.get(toolName)
  if (!tool) {
    return {
      status: 'error',
      output: `Tool not available: ${toolName}`,
      durationMs: 0,
      code: 'TOOL_UNAVAILABLE',
    }
  }
  return tool.execute(input, {
    executionId: randomUUID(),
    sessionId: 'services-api',
    startedAt: new Date().toISOString(),
  })
}

function serviceToolResponse(
  reply: { status(code: number): { send(payload: unknown): unknown } },
  result: Awaited<ReturnType<typeof runServiceTool>>,
) {
  if (result.code === 'SERVICE_UNAVAILABLE' || result.code === 'TOOL_UNAVAILABLE') {
    return reply.status(503).send({ error: { code: result.code, message: result.output } })
  }
  return { data: result }
}

function withoutUndefined(input: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(input).filter((entry) => entry[1] !== undefined),
  )
}

export async function servicesRoutes(app: FastifyInstance) {
  app.get('/services', async (_request, reply) => {
    const result = await runServiceTool(app, 'service.list')
    return serviceToolResponse(reply, result)
  })

  app.post<{ Body: ServiceStartBody }>('/services', {
    preValidation: zodRequestValidation({
      body: {
        schema: serviceStartBodySchema,
        message: 'Invalid service start request body',
      },
    }),
  }, async (request, reply) => {
    const body = request.body
    const result = await runServiceTool(app, 'service.start', body)
    return serviceToolResponse(reply, result)
  })

  app.post<{ Body: NativeInstallBody }>('/native-services', {
    preValidation: zodRequestValidation({
      body: {
        schema: nativeInstallBodySchema,
        message: 'Invalid native service install request body',
      },
    }),
  }, async (request, reply) => {
    const body = request.body
    const result = await runServiceTool(app, 'service.install', body)
    return serviceToolResponse(reply, result)
  })

  app.get<{ Params: ServiceIdParams }>('/native-services/:id', {
    preValidation: zodRequestValidation({
      params: {
        schema: serviceIdParamsSchema,
        message: 'Invalid native service id',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const result = await runServiceTool(app, 'service.native.status', { id: params.id })
    return serviceToolResponse(reply, result)
  })

  app.get<{ Params: ServiceIdParams; Querystring: ServiceLogsQuery }>('/native-services/:id/logs', {
    preValidation: zodRequestValidation({
      params: {
        schema: serviceIdParamsSchema,
        message: 'Invalid native service id',
      },
      query: {
        schema: serviceLogsQuerySchema,
        message: 'Invalid native service logs query',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const query = request.query
    const result = await runServiceTool(app, 'service.native.logs', withoutUndefined({
      id: params.id,
      ...query,
    }))
    return serviceToolResponse(reply, result)
  })

  app.post<{ Params: ServiceIdParams; Body: NativeControlBody }>('/native-services/:id/enable', {
    preValidation: zodRequestValidation({
      params: {
        schema: serviceIdParamsSchema,
        message: 'Invalid native service id',
      },
      body: {
        schema: nativeControlBodySchema,
        message: 'Invalid native service enable request body',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const body = request.body
    const result = await runServiceTool(app, 'service.enable', withoutUndefined({ id: params.id, ...body }))
    return serviceToolResponse(reply, result)
  })

  app.post<{ Params: ServiceIdParams; Body: NativeControlBody }>('/native-services/:id/disable', {
    preValidation: zodRequestValidation({
      params: {
        schema: serviceIdParamsSchema,
        message: 'Invalid native service id',
      },
      body: {
        schema: nativeControlBodySchema,
        message: 'Invalid native service disable request body',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const body = request.body
    const result = await runServiceTool(app, 'service.disable', withoutUndefined({ id: params.id, ...body }))
    return serviceToolResponse(reply, result)
  })

  app.post<{ Params: ServiceIdParams; Body: NativeControlBody }>('/native-services/:id/uninstall', {
    preValidation: zodRequestValidation({
      params: {
        schema: serviceIdParamsSchema,
        message: 'Invalid native service id',
      },
      body: {
        schema: nativeControlBodySchema,
        message: 'Invalid native service uninstall request body',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const body = request.body
    const result = await runServiceTool(app, 'service.uninstall', withoutUndefined({ id: params.id, ...body }))
    return serviceToolResponse(reply, result)
  })

  app.get<{ Params: ServiceIdParams }>('/services/:id', {
    preValidation: zodRequestValidation({
      params: {
        schema: serviceIdParamsSchema,
        message: 'Invalid service id',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const result = await runServiceTool(app, 'service.status', { id: params.id })
    return serviceToolResponse(reply, result)
  })

  app.get<{ Params: ServiceIdParams; Querystring: ServiceLogsQuery }>('/services/:id/logs', {
    preValidation: zodRequestValidation({
      params: {
        schema: serviceIdParamsSchema,
        message: 'Invalid service id',
      },
      query: {
        schema: serviceLogsQuerySchema,
        message: 'Invalid service logs query',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const query = request.query
    const result = await runServiceTool(app, 'service.logs', withoutUndefined({
      id: params.id,
      ...query,
    }))
    return serviceToolResponse(reply, result)
  })

  app.post<{ Params: ServiceIdParams }>('/services/:id/healthcheck', {
    preValidation: zodRequestValidation({
      params: {
        schema: serviceIdParamsSchema,
        message: 'Invalid service id',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const result = await runServiceTool(app, 'service.healthcheck', { id: params.id })
    return serviceToolResponse(reply, result)
  })

  app.post<{ Params: ServiceIdParams; Body: ServiceStopBody }>('/services/:id/stop', {
    preValidation: zodRequestValidation({
      params: {
        schema: serviceIdParamsSchema,
        message: 'Invalid service id',
      },
      body: {
        schema: serviceStopBodySchema,
        message: 'Invalid service stop request body',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const body = request.body
    const result = await runServiceTool(app, 'service.stop', withoutUndefined({ id: params.id, ...body }))
    return serviceToolResponse(reply, result)
  })

  app.post<{ Params: ServiceIdParams }>('/services/:id/restart', {
    preValidation: zodRequestValidation({
      params: {
        schema: serviceIdParamsSchema,
        message: 'Invalid service id',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const result = await runServiceTool(app, 'service.restart', { id: params.id })
    return serviceToolResponse(reply, result)
  })

  app.delete<{ Params: ServiceIdParams; Querystring: ServiceRemoveQuery }>('/services/:id', {
    preValidation: zodRequestValidation({
      params: {
        schema: serviceIdParamsSchema,
        message: 'Invalid service id',
      },
      query: {
        schema: serviceRemoveQuerySchema,
        message: 'Invalid service remove query',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const query = request.query
    const result = await runServiceTool(app, 'service.remove', withoutUndefined({
      id: params.id,
      force: query.force,
      deleteLogs: query.deleteLogs,
    }))
    return serviceToolResponse(reply, result)
  })
}
