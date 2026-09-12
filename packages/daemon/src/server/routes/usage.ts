import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { fastifySchemaFromZod } from './utils.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiParameterRef,
  type OpenApiOverrideMap,
} from '../openapi.js'

const usageSummarySchema = z.object({
  inputTokens: z.number().int(),
  outputTokens: z.number().int(),
  costUsd: z.number(),
  requestCount: z.number().int(),
})

const dailyUsageSummarySchema = z.object({
  date: z.string(),
  provider: z.string(),
  model: z.string(),
  totalInputTokens: z.number().int(),
  totalOutputTokens: z.number().int(),
  totalCostUsd: z.number(),
  requestCount: z.number().int(),
})

const usageSnapshotSchema = z.object({
  totalSessions: z.number().int(),
  totalMessages: z.number().int(),
  totalToolCalls: z.number().int(),
})

const usageSummaryResponseSchema = z.object({
  data: usageSummarySchema,
})

const dailyUsageResponseSchema = z.object({
  data: z.array(dailyUsageSummarySchema),
})

const usageSnapshotResponseSchema = z.object({
  data: usageSnapshotSchema,
})

const usageDailyQuerySchema = z.object({
  days: z.coerce.number().int().min(1).optional(),
})

const usageSessionParamsSchema = z.object({
  id: z.string().min(1),
})

type UsageDailyQuery = z.infer<typeof usageDailyQuerySchema>
type UsageSessionParams = z.infer<typeof usageSessionParamsSchema>

export const usageOpenApiComponents = openApiComponentsFromZod({
  schemas: {
    UsageSummary: usageSummarySchema,
    DailyUsageSummary: dailyUsageSummarySchema,
    UsageSnapshot: usageSnapshotSchema,
    UsageSummaryResponse: usageSummaryResponseSchema,
    DailyUsageResponse: dailyUsageResponseSchema,
    UsageSnapshotResponse: usageSnapshotResponseSchema,
  },
  parameters: {
    UsageDaysParam: {
      name: 'days',
      in: 'query',
      schema: z.number().int().min(1),
    },
  },
})

export const usageOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/usage': {
    get: {
      summary: 'Usage summary',
      tags: ['Usage'],
      responses: { 200: openApiJsonResponseRef('UsageSummaryResponse') },
    },
  },
  '/api/v1/usage/session/{id}': {
    get: {
      summary: 'Session usage',
      tags: ['Usage'],
      responses: { 200: openApiJsonResponseRef('UsageSummaryResponse') },
    },
  },
  '/api/v1/usage/daily': {
    get: {
      summary: 'Daily usage',
      tags: ['Usage'],
      parameters: [openApiParameterRef('UsageDaysParam')],
      responses: { 200: openApiJsonResponseRef('DailyUsageResponse') },
    },
  },
  '/api/v1/usage/snapshot': {
    get: {
      summary: 'Usage snapshot',
      tags: ['Usage'],
      responses: { 200: openApiJsonResponseRef('UsageSnapshotResponse') },
    },
  },
}

export async function usageRoutes(app: FastifyInstance) {
  const runtime = app.runtime

  // GET /usage — total usage summary
  app.get('/usage', async (_request, reply) => {
    if (!runtime?.usageTracker) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const total = runtime.usageTracker.getTotalUsage()
    return { data: total }
  })

  // GET /usage/daily — daily usage breakdown
  app.get<{ Querystring: UsageDailyQuery }>('/usage/daily', {
    schema: fastifySchemaFromZod({ querystring: usageDailyQuerySchema }),
  }, async (request, reply) => {
    if (!runtime?.usageTracker) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const query = request.query
    const summaries = runtime.usageTracker.getDailySummaries(query.days ?? 30)
    return { data: summaries }
  })

  // GET /usage/snapshot — session/message/tool-call totals
  app.get('/usage/snapshot', async (_request, reply) => {
    if (!runtime?.usageTracker) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const snapshot = runtime.usageTracker.getSnapshot()
    return { data: snapshot }
  })

  // GET /usage/session/:id — usage for specific session
  app.get<{ Params: UsageSessionParams }>('/usage/session/:id', {
    schema: fastifySchemaFromZod({ params: usageSessionParamsSchema }),
  }, async (request, reply) => {
    if (!runtime?.usageTracker) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const usage = runtime.usageTracker.getSessionUsage(params.id)
    return { data: usage }
  })
}
