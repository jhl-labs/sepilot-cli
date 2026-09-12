import { randomUUID } from 'node:crypto'
import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import type { ToolResult } from '../../tools/registry.js'
import { fastifySchemaFromZod, zodRequestValidation } from './utils.js'

const pagesScaffoldBodySchema = z.object({
  repoPath: z.string().trim().min(1).optional(),
  sitePath: z.string().trim().min(1).optional(),
  siteName: z.string().trim().min(1).optional(),
  template: z.literal('astro-mdx').optional(),
  defaultBranch: z.string().trim().min(1).optional(),
  includeWorkflow: z.boolean().optional(),
  overwrite: z.boolean().optional(),
  cwd: z.string().trim().min(1).optional(),
})

const pagesScanBodySchema = z.object({
  path: z.string().trim().min(1).optional(),
  maxFiles: z.number().finite().optional(),
  maxBytesPerFile: z.number().finite().optional(),
  cwd: z.string().trim().min(1).optional(),
})

const pagesStatusQuerySchema = z.object({
  repoPath: z.string().trim().min(1).optional(),
  repo: z.string().trim().min(1).optional(),
  workflowFile: z.string().trim().min(1).optional(),
  branch: z.string().trim().min(1).optional(),
  cwd: z.string().trim().min(1).optional(),
})

type PagesScaffoldBody = z.infer<typeof pagesScaffoldBodySchema>
type PagesScanBody = z.infer<typeof pagesScanBodySchema>
type PagesStatusQuery = z.infer<typeof pagesStatusQuerySchema>

async function runPagesTool(
  app: FastifyInstance,
  toolName: string,
  input: Record<string, unknown>,
  cwd?: string,
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
    sessionId: 'pages-api',
    startedAt: new Date().toISOString(),
    cwd,
  })
}

export async function pagesRoutes(app: FastifyInstance) {
  app.post<{ Body: PagesScaffoldBody }>('/pages/scaffold', {
    preValidation: zodRequestValidation({
      body: {
        schema: pagesScaffoldBodySchema,
        message: 'Invalid pages scaffold request body',
      },
    }),
  }, async (request, reply) => {
    const body = request.body
    const result = await runPagesTool(app, 'pages.scaffold', body, body.cwd)
    if (result.code === 'SERVICE_UNAVAILABLE' || result.code === 'TOOL_UNAVAILABLE') {
      return reply.status(503).send({ error: { code: result.code, message: result.output } })
    }
    return { data: result }
  })

  app.post<{ Body: PagesScanBody }>('/pages/scan', {
    preValidation: zodRequestValidation({
      body: {
        schema: pagesScanBodySchema,
        message: 'Invalid pages scan request body',
      },
    }),
  }, async (request, reply) => {
    const body = request.body
    const result = await runPagesTool(app, 'pages.scan', body, body.cwd)
    if (result.code === 'SERVICE_UNAVAILABLE' || result.code === 'TOOL_UNAVAILABLE') {
      return reply.status(503).send({ error: { code: result.code, message: result.output } })
    }
    return { data: result }
  })

  app.get<{ Querystring: PagesStatusQuery }>('/pages/status', {
    schema: fastifySchemaFromZod({ querystring: pagesStatusQuerySchema }),
  }, async (request, reply) => {
    const query = request.query
    const result = await runPagesTool(app, 'pages.status', query, query.cwd)
    if (result.code === 'SERVICE_UNAVAILABLE' || result.code === 'TOOL_UNAVAILABLE') {
      return reply.status(503).send({ error: { code: result.code, message: result.output } })
    }
    return { data: result }
  })
}
