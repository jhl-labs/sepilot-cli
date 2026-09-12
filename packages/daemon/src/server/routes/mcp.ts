import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiSchemaRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'

const mcpServerStatusSchema = z.object({
  name: z.string(),
  enabled: z.boolean(),
  status: z.enum(['disabled', 'connected', 'error']),
  connected: z.boolean(),
  transport: z.string(),
  command: z.string().optional(),
  args: z.array(z.string()).optional(),
  url: z.string().optional(),
  toolCount: z.number().int(),
  tools: z.array(z.string()),
  allTools: z.array(z.string()).optional(),
  disabledTools: z.array(z.string()).optional(),
  error: z.string().optional(),
})

const mcpOpenApiZodComponents = openApiComponentsFromZod({
  schemas: {
    McpServerStatus: mcpServerStatusSchema,
  },
})

export const mcpOpenApiComponents: OpenApiComponentOverrides = {
  schemas: {
    ...(mcpOpenApiZodComponents.schemas ?? {}),
    McpServersResponse: {
      type: 'object',
      properties: {
        data: {
          type: 'array',
          items: openApiSchemaRef('McpServerStatus'),
        },
      },
      required: ['data'],
    },
  },
}

export const mcpOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/mcp/servers': {
    get: {
      summary: 'List MCP servers',
      tags: ['MCP'],
      responses: { 200: openApiJsonResponseRef('McpServersResponse') },
    },
  },
}

export async function mcpRoutes(app: FastifyInstance) {
  app.get('/mcp/servers', async (_request, reply) => {
    const runtime = app.runtime
    if (!runtime?.mcpManager) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Runtime not initialized',
        },
      })
    }

    return { data: runtime.mcpManager.listServers() }
  })
}
