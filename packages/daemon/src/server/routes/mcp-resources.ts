import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { zodRequestValidation } from './utils.js'
import {
  getConnectedMcpClient,
  sendMcpUpstreamError,
  type ConnectedMcpClient,
  type McpRouteRuntime,
} from './mcp-route-utils.js'

interface ResourceClient extends ConnectedMcpClient {
  discoverResources: () => Promise<unknown[]>
  discoverResourceTemplates: () => Promise<unknown[]>
  readResource: (uri: string) => Promise<unknown>
  subscribeResource: (uri: string) => Promise<unknown>
  unsubscribeResource: (uri: string) => Promise<unknown>
  listResourceSubscriptions: () => string[]
  listResourceUpdates: (limit?: number) => unknown[]
}
type ResourceRouteRuntime = McpRouteRuntime<ResourceClient>

const serverParamsSchema = z.object({
  name: z.string().trim().min(1),
})

const resourceReadBodySchema = z.object({
  uri: z.string().trim().min(1),
})

const resourceUpdatesQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(200).optional().default(50),
})

type ServerParams = z.infer<typeof serverParamsSchema>
type ResourceReadBody = z.infer<typeof resourceReadBodySchema>
type ResourceUpdatesQuery = z.infer<typeof resourceUpdatesQuerySchema>

export async function mcpResourcesRoutes(app: FastifyInstance) {
  const runtime = app.runtime as ResourceRouteRuntime | undefined

  app.get<{ Params: ServerParams }>(
    '/mcp/servers/:name/resources',
    {
      preValidation: zodRequestValidation({
        params: { schema: serverParamsSchema, message: 'Invalid server name' },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const client = await getConnectedMcpClient(runtime, reply, params.name)
      if (!client) return reply
      try {
        const resources = await client.discoverResources()
        return { data: resources }
      } catch (err) {
        return sendMcpUpstreamError(reply, err)
      }
    },
  )

  app.get<{ Params: ServerParams }>(
    '/mcp/servers/:name/resources/templates',
    {
      preValidation: zodRequestValidation({
        params: { schema: serverParamsSchema, message: 'Invalid server name' },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const client = await getConnectedMcpClient(runtime, reply, params.name)
      if (!client) return reply
      try {
        const templates = await client.discoverResourceTemplates()
        return { data: templates }
      } catch (err) {
        return sendMcpUpstreamError(reply, err)
      }
    },
  )

  app.post<{ Params: ServerParams; Body: ResourceReadBody }>(
    '/mcp/servers/:name/resources/read',
    {
      preValidation: zodRequestValidation({
        params: { schema: serverParamsSchema, message: 'Invalid server name' },
        body: { schema: resourceReadBodySchema, message: 'Invalid resource read request' },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const body = request.body
      const client = await getConnectedMcpClient(runtime, reply, params.name)
      if (!client) return reply
      try {
        const result = await client.readResource(body.uri)
        return { data: result }
      } catch (err) {
        return sendMcpUpstreamError(reply, err)
      }
    },
  )

  app.get<{ Params: ServerParams }>(
    '/mcp/servers/:name/resources/subscriptions',
    {
      preValidation: zodRequestValidation({
        params: { schema: serverParamsSchema, message: 'Invalid server name' },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const client = await getConnectedMcpClient(runtime, reply, params.name)
      if (!client) return reply
      return { data: client.listResourceSubscriptions() }
    },
  )

  app.post<{ Params: ServerParams; Body: ResourceReadBody }>(
    '/mcp/servers/:name/resources/subscribe',
    {
      preValidation: zodRequestValidation({
        params: { schema: serverParamsSchema, message: 'Invalid server name' },
        body: { schema: resourceReadBodySchema, message: 'Invalid resource subscription request' },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const body = request.body
      const client = await getConnectedMcpClient(runtime, reply, params.name)
      if (!client) return reply
      try {
        const result = await client.subscribeResource(body.uri)
        return { data: result }
      } catch (err) {
        return sendMcpUpstreamError(reply, err)
      }
    },
  )

  app.post<{ Params: ServerParams; Body: ResourceReadBody }>(
    '/mcp/servers/:name/resources/unsubscribe',
    {
      preValidation: zodRequestValidation({
        params: { schema: serverParamsSchema, message: 'Invalid server name' },
        body: { schema: resourceReadBodySchema, message: 'Invalid resource subscription request' },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const body = request.body
      const client = await getConnectedMcpClient(runtime, reply, params.name)
      if (!client) return reply
      try {
        const result = await client.unsubscribeResource(body.uri)
        return { data: result }
      } catch (err) {
        return sendMcpUpstreamError(reply, err)
      }
    },
  )

  app.get<{ Params: ServerParams; Querystring: ResourceUpdatesQuery }>(
    '/mcp/servers/:name/resources/updates',
    {
      preValidation: zodRequestValidation({
        params: { schema: serverParamsSchema, message: 'Invalid server name' },
        query: { schema: resourceUpdatesQuerySchema, message: 'Invalid resource updates query' },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const query = request.query
      const client = await getConnectedMcpClient(runtime, reply, params.name)
      if (!client) return reply
      return { data: client.listResourceUpdates(query.limit) }
    },
  )
}
