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

interface UtilityClient extends ConnectedMcpClient {
  complete: (input: CompletionBody) => Promise<unknown>
  setLoggingLevel: (level: LoggingLevel) => Promise<unknown>
  getLoggingState: () => unknown
}

type UtilityRouteRuntime = McpRouteRuntime<UtilityClient>

const loggingLevels = [
  'debug',
  'info',
  'notice',
  'warning',
  'error',
  'critical',
  'alert',
  'emergency',
] as const

const serverParamsSchema = z.object({
  name: z.string().trim().min(1),
})

const completionBodySchema = z.object({
  ref: z.union([
    z.object({ type: z.literal('ref/prompt'), name: z.string().trim().min(1) }),
    z.object({ type: z.literal('ref/resource'), uri: z.string().trim().min(1) }),
  ]),
  argument: z.object({
    name: z.string().trim().min(1),
    value: z.string(),
  }),
  context: z.object({
    arguments: z.record(z.string()).optional(),
  }).optional(),
})

const loggingLevelBodySchema = z.object({
  level: z.enum(loggingLevels),
})

type ServerParams = z.infer<typeof serverParamsSchema>
type CompletionBody = z.infer<typeof completionBodySchema>
type LoggingLevel = z.infer<typeof loggingLevelBodySchema>['level']

export async function mcpUtilitiesRoutes(app: FastifyInstance) {
  const runtime = app.runtime as UtilityRouteRuntime | undefined

  app.post<{ Params: ServerParams; Body: CompletionBody }>(
    '/mcp/servers/:name/completion',
    {
      preValidation: zodRequestValidation({
        params: { schema: serverParamsSchema, message: 'Invalid server name' },
        body: { schema: completionBodySchema, message: 'Invalid completion request' },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const body = request.body
      const client = await getConnectedMcpClient(runtime, reply, params.name)
      if (!client) return reply
      try {
        const result = await client.complete(body)
        return { data: result }
      } catch (err) {
        return sendMcpUpstreamError(reply, err)
      }
    },
  )

  app.post<{ Params: ServerParams; Body: { level: LoggingLevel } }>(
    '/mcp/servers/:name/logging/level',
    {
      preValidation: zodRequestValidation({
        params: { schema: serverParamsSchema, message: 'Invalid server name' },
        body: { schema: loggingLevelBodySchema, message: 'Invalid logging level request' },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const body = request.body
      const client = await getConnectedMcpClient(runtime, reply, params.name)
      if (!client) return reply
      try {
        const result = await client.setLoggingLevel(body.level)
        return { data: result }
      } catch (err) {
        return sendMcpUpstreamError(reply, err)
      }
    },
  )

  app.get<{ Params: ServerParams }>(
    '/mcp/servers/:name/logs',
    {
      preValidation: zodRequestValidation({
        params: { schema: serverParamsSchema, message: 'Invalid server name' },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const client = await getConnectedMcpClient(runtime, reply, params.name)
      if (!client) return reply
      return { data: client.getLoggingState() }
    },
  )
}
