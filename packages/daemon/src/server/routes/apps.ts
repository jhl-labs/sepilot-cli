import { randomUUID } from 'node:crypto'
import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import type { ToolResult } from '../../tools/registry.js'
import { zodRequestValidation } from './utils.js'

const appIdParamsSchema = z.object({
  id: z.string().trim().min(1),
})

const appListQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(50).optional(),
  offset: z.coerce.number().int().min(0).optional(),
  reference: z.string().trim().min(1).max(80).optional(),
})

const appReadQuerySchema = z.object({
  includeHtml: z.preprocess((value) => {
    if (value === undefined || value === '') return undefined
    if (value === true || value === 'true') return true
    if (value === false || value === 'false') return false
    return value
  }, z.boolean().optional()),
})

const appSearchBodySchema = z.object({
  query: z.string().trim().min(1),
  type: z.enum(['hybrid', 'keyword', 'semantic']).optional(),
  limit: z.number().int().min(1).max(50).optional(),
  includeHtml: z.boolean().optional(),
  interAppReadableOnly: z.boolean().optional(),
})

const appMutateBodySchema = z.object({
  mutations: z.array(z.record(z.unknown())).min(1),
  dryRun: z.boolean().optional(),
})

const appWriteBodySchema = z
  .object({
    mode: z.enum(['patch', 'replace']).optional(),
    data: z.record(z.unknown()).optional(),
    patch: z.record(z.unknown()).optional(),
  })
  .refine((value) => value.data || value.patch, {
    message: 'data or patch is required',
  })

type AppIdParams = z.infer<typeof appIdParamsSchema>
type AppReadQuery = z.infer<typeof appReadQuerySchema>
type AppSearchBody = z.infer<typeof appSearchBodySchema>
type AppMutateBody = z.infer<typeof appMutateBodySchema>
type AppWriteBody = z.infer<typeof appWriteBodySchema>

async function runAppsTool(
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
    sessionId: 'apps-api',
    startedAt: new Date().toISOString(),
  })
}

function appsToolResponse(
  reply: { status(code: number): { send(payload: unknown): unknown } },
  result: Awaited<ReturnType<typeof runAppsTool>>,
) {
  if (result.code === 'SERVICE_UNAVAILABLE' || result.code === 'TOOL_UNAVAILABLE') {
    return reply.status(503).send({ error: { code: result.code, message: result.output } })
  }
  return { data: result }
}

export async function appsRoutes(app: FastifyInstance) {
  app.get<{ Querystring: z.infer<typeof appListQuerySchema> }>('/apps', {
    preValidation: zodRequestValidation({ query: { schema: appListQuerySchema, message: 'Invalid apps list query' } }),
  }, async (request, reply) => {
    const result = await runAppsTool(app, 'apps.list', request.query)
    return appsToolResponse(reply, result)
  })

  app.get<{ Params: AppIdParams; Querystring: AppReadQuery }>(
    '/apps/:id',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: appIdParamsSchema,
          message: 'Invalid app id',
        },
        query: {
          schema: appReadQuerySchema,
          message: 'Invalid app read query',
        },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const query = request.query
      const result = await runAppsTool(app, 'apps.read', {
        id: params.id,
        includeHtml: query.includeHtml,
      })
      return appsToolResponse(reply, result)
    },
  )

  app.post<{ Body: AppSearchBody }>(
    '/apps/search',
    {
      preValidation: zodRequestValidation({
        body: {
          schema: appSearchBodySchema,
          message: 'Invalid apps search request body',
        },
      }),
    },
    async (request, reply) => {
      const body = request.body
      const result = await runAppsTool(app, 'apps.search', body)
      return appsToolResponse(reply, result)
    },
  )

  app.post<{ Params: AppIdParams; Body: AppMutateBody }>(
    '/apps/:id/mutate',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: appIdParamsSchema,
          message: 'Invalid app id',
        },
        body: {
          schema: appMutateBodySchema,
          message: 'Invalid app mutation request body',
        },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const body = request.body
      const result = await runAppsTool(app, 'apps.mutate', {
        ...body,
        id: params.id,
      })
      return appsToolResponse(reply, result)
    },
  )

  app.put<{ Params: AppIdParams; Body: AppWriteBody }>(
    '/apps/:id/data',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: appIdParamsSchema,
          message: 'Invalid app id',
        },
        body: {
          schema: appWriteBodySchema,
          message: 'Invalid app data write request body',
        },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const body = request.body
      const result = await runAppsTool(app, 'apps.write', {
        ...body,
        id: params.id,
      })
      return appsToolResponse(reply, result)
    },
  )
}
