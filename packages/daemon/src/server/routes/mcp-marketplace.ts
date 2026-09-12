import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { zodRequestValidation } from './utils.js'
import { installMcpServer } from '../../mcp/mcp-install.js'

const searchQuerySchema = z.object({
  q: z.string().trim().min(1),
})

const installBodySchema = z.object({
  name: z.string().trim().min(1),
  marketplace: z.string().trim().min(1).optional(),
  variables: z.record(z.string()).optional(),
  allowUnverified: z.boolean().optional().default(false),
})

const marketplaceCreateBodySchema = z.object({
  name: z.string().trim().min(1),
  url: z.string().trim().url(),
  publisher: z.string().trim().min(1).optional(),
  publicKey: z.string().trim().min(1).optional(),
})

const marketplaceParamsSchema = z.object({
  name: z.string().trim().min(1),
})

type SearchQuery = z.infer<typeof searchQuerySchema>
type InstallBody = z.infer<typeof installBodySchema>
type MarketplaceCreateBody = z.infer<typeof marketplaceCreateBodySchema>
type MarketplaceParams = z.infer<typeof marketplaceParamsSchema>

export async function mcpMarketplaceRoutes(app: FastifyInstance) {
  const runtime = app.runtime as Record<string, unknown> | undefined

  app.get<{ Querystring: SearchQuery }>(
    '/mcp/marketplace/search',
    {
      preValidation: zodRequestValidation({
        query: {
          schema: searchQuerySchema,
          message: 'Missing or invalid query parameter q',
        },
      }),
    },
    async (request, _reply) => {
      const query = request.query
      const source = (runtime as Record<string, unknown>)?.mcpMarketplaceSource as {
        search: (q: string) => Promise<unknown[]>
      }
      const results = await source.search(query.q)
      return { data: results }
    },
  )

  app.post<{ Body: InstallBody }>(
    '/mcp/install',
    {
      preValidation: zodRequestValidation({
        body: {
          schema: installBodySchema,
          message: 'Invalid install request body',
        },
      }),
    },
    async (request, reply) => {
      const body = request.body
      const source = (runtime as Record<string, unknown>)?.mcpMarketplaceSource as {
        get: (marketplace: string | null, name: string) => Promise<unknown | null>
      }
      const configWriter = (runtime as Record<string, unknown>)?.mcpConfigWriter as {
        addMcpServer: (entry: Record<string, unknown>) => Promise<void>
      }
      const configPath = (runtime as Record<string, unknown>)?.configPath as string | undefined

      const template = await source.get(body.marketplace ?? null, body.name)
      if (!template) {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: `MCP server "${body.name}" not found in any marketplace`,
          },
        })
      }

      try {
        await installMcpServer({
          template: template as import('../../mcp/marketplace-source.js').McpServerTemplate,
          configPath: configPath ?? '',
          configWriter: configWriter as import('../../mcp/config-writer.js').ConfigWriter,
          variables: body.variables,
          allowUnverified: body.allowUnverified,
        })
        return { data: { installed: true, serverName: body.name } }
      } catch (error: unknown) {
        const message =
          error instanceof Error ? error.message : 'install failed'
        if (/already exists/i.test(message)) {
          return reply.status(409).send({
            error: { code: 'CONFLICT', message },
          })
        }
        if (/not verified|digest mismatch/i.test(message)) {
          return reply.status(400).send({
            error: { code: 'MCP_TEMPLATE_UNVERIFIED', message },
          })
        }
        throw error
      }
    },
  )

  app.get('/mcp/marketplaces', async (_request, _reply) => {
    const catalog = (runtime as Record<string, unknown>)?.mcpMarketplaceCatalog as {
      list: () => Promise<unknown[]>
    }
    const data = await catalog.list()
    return { data }
  })

  app.post<{ Body: MarketplaceCreateBody }>(
    '/mcp/marketplaces',
    {
      preValidation: zodRequestValidation({
        body: {
          schema: marketplaceCreateBodySchema,
          message: 'Invalid marketplace request body',
        },
      }),
    },
    async (request, reply) => {
      const body = request.body
      const catalog = (runtime as Record<string, unknown>)?.mcpMarketplaceCatalog as {
        add: (
          name: string,
          url: string,
          options?: { publisher?: string; publicKey?: string },
        ) => Promise<unknown>
      }
      try {
        const entry = await catalog.add(body.name, body.url, {
          publisher: body.publisher,
          publicKey: body.publicKey,
        })
        return { data: entry }
      } catch (error: unknown) {
        return reply.status(409).send({
          error: {
            code: 'CONFLICT',
            message:
              error instanceof Error
                ? error.message
                : 'marketplace already exists',
          },
        })
      }
    },
  )

  app.delete<{ Params: MarketplaceParams }>(
    '/mcp/marketplaces/:name',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: marketplaceParamsSchema,
          message: 'Invalid marketplace name',
        },
      }),
    },
    async (request, _reply) => {
      const params = request.params
      const catalog = (runtime as Record<string, unknown>)?.mcpMarketplaceCatalog as {
        remove: (name: string) => Promise<boolean>
      }
      const removed = await catalog.remove(params.name)
      return { data: { removed } }
    },
  )
}
