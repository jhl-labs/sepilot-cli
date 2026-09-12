import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import type { SepilotdConfig } from '../../config/schema.js'
import type { ToolDefinitionRuntime, ToolResult } from '../../tools/registry.js'
import { applyAndPersistRuntimeUpdate } from '../runtime/config-runtime.js'
import { zodRequestValidation } from './utils.js'

const serverParamsSchema = z.object({
  name: z.string().trim().min(1),
})

const toolParamsSchema = z.object({
  name: z.string().trim().min(1),
  tool: z.string().trim().min(1),
})

const toolCallBodySchema = z.object({
  arguments: z.record(z.unknown()).optional().default({}),
})

type ServerParams = z.infer<typeof serverParamsSchema>
type ToolParams = z.infer<typeof toolParamsSchema>
type ToolCallBody = z.infer<typeof toolCallBodySchema>

type McpToolsRuntime = {
  config?: SepilotdConfig
  configMutationService?: {
    apply: <T>(event: string, operation: () => T | Promise<T>) => Promise<T>
  }
  mcpManager?: {
    listServers: () => Array<{
      name: string
      tools: string[]
      allTools: string[]
      disabledTools: string[]
      status: 'disabled' | 'connected' | 'error'
      toolManifestStatus?: 'trusted' | 'changed' | 'untrusted-baseline'
    }>
  }
  toolRegistry?: {
    get: (name: string) => ToolDefinitionRuntime | undefined
  }
  mcpConfigWriter?: {
    read: () => Promise<Record<string, unknown>>
    setDisabledTools: (serverName: string, tools: string[]) => Promise<void>
  }
}

type ConfigRuntime = Parameters<typeof applyAndPersistRuntimeUpdate>[0]

async function updateDisabledTools(
  runtime: McpToolsRuntime | undefined,
  serverName: string,
  update: (current: string[]) => string[],
): Promise<'ok' | 'not_found'> {
  if (runtime?.config && runtime.configMutationService) {
    return runtime.configMutationService.apply(
      'config.mcp.servers.disabled_tools',
      async () => {
        const server = runtime.config?.mcp.servers.find(
          (entry) => entry.name === serverName,
        )
        if (!server) return 'not_found' as const

        server.disabledTools = Array.from(new Set(update(server.disabledTools ?? [])))
        await applyAndPersistRuntimeUpdate(
          runtime as ConfigRuntime,
          new Set(['mcp.servers']),
          { 'mcp.servers': runtime.config?.mcp.servers },
        )
        return 'ok' as const
      },
    )
  }

  const configWriter = runtime?.mcpConfigWriter
  if (!configWriter) return 'not_found'

  const config = await configWriter.read()
  const mcp = (config.mcp ?? { servers: [] }) as {
    servers: Array<{ name: string; disabledTools?: string[] }>
  }
  const server = mcp.servers.find((s) => s.name === serverName)
  if (!server) return 'not_found'

  await configWriter.setDisabledTools(
    serverName,
    Array.from(new Set(update(server.disabledTools ?? []))),
  )
  return 'ok'
}

function resolveToolName(serverName: string, toolName: string): string {
  const prefix = `mcp.${serverName}.`
  return toolName.startsWith(prefix) ? toolName : `${prefix}${toolName}`
}

function errorToolResult(error: unknown, startedAt: number): ToolResult {
  return {
    output: error instanceof Error ? error.message : String(error),
    status: 'error',
    durationMs: Date.now() - startedAt,
  }
}

export async function mcpToolsRoutes(app: FastifyInstance) {
  const runtime = app.runtime as McpToolsRuntime | undefined

  app.get<{ Params: ServerParams }>(
    '/mcp/servers/:name/tools',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: serverParamsSchema,
          message: 'Invalid server name',
        },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const mcpManager = runtime?.mcpManager
      const server = mcpManager?.listServers().find((s) => s.name === params.name)
      if (!server) {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: `MCP server "${params.name}" not found`,
          },
        })
      }
      return {
        data: {
          enabled: server.tools,
          disabled: server.disabledTools,
          advertised: server.allTools,
          quarantined: server.toolManifestStatus === 'changed',
        },
      }
    },
  )

  app.post<{ Params: ToolParams }>(
    '/mcp/servers/:name/tools/:tool/disable',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: toolParamsSchema,
          message: 'Invalid params',
        },
      }),
    },
    async (request, reply) => {
      const params = request.params

      const outcome = await updateDisabledTools(
        runtime,
        params.name,
        (current) => current.includes(params.tool) ? current : [...current, params.tool],
      )

      if (outcome === 'not_found') {
        return reply.status(404).send({
          error: { code: 'NOT_FOUND', message: `MCP server "${params.name}" not found` },
        })
      }

      return { data: { ok: true } }
    },
  )

  app.post<{ Params: ToolParams }>(
    '/mcp/servers/:name/tools/:tool/enable',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: toolParamsSchema,
          message: 'Invalid params',
        },
      }),
    },
    async (request, reply) => {
      const params = request.params

      const outcome = await updateDisabledTools(
        runtime,
        params.name,
        (current) => current.filter((tool) => tool !== params.tool),
      )

      if (outcome === 'not_found') {
        return reply.status(404).send({
          error: { code: 'NOT_FOUND', message: `MCP server "${params.name}" not found` },
        })
      }

      return { data: { ok: true } }
    },
  )

  app.post<{ Params: ToolParams; Body: ToolCallBody }>(
    '/mcp/servers/:name/tools/:tool/call',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: toolParamsSchema,
          message: 'Invalid params',
        },
        body: {
          schema: toolCallBodySchema,
          message: 'Invalid tool call input',
        },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const server = runtime?.mcpManager?.listServers().find((s) => s.name === params.name)
      if (!server) {
        return reply.status(404).send({
          error: { code: 'NOT_FOUND', message: `MCP server "${params.name}" not found` },
        })
      }
      if (server.toolManifestStatus === 'changed') {
        return reply.status(409).send({
          error: {
            code: 'MCP_TOOL_MANIFEST_CHANGED',
            message: `MCP server "${params.name}" tools are quarantined because its manifest changed. Review the advertised tools and trust the new manifest before calling them.`,
          },
        })
      }
      if (server.status !== 'connected') {
        return reply.status(503).send({
          error: {
            code: 'SERVICE_UNAVAILABLE',
            message: `MCP server "${params.name}" is ${server.status}`,
          },
        })
      }

      const qualifiedName = resolveToolName(params.name, params.tool)
      const tool = runtime?.toolRegistry?.get(qualifiedName)
      if (!tool) {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: `MCP tool "${params.tool}" not found on server "${params.name}"`,
          },
        })
      }

      const startedAt = Date.now()
      try {
        const { arguments: args } = request.body
        return { data: await tool.execute(args) }
      } catch (error) {
        return { data: errorToolResult(error, startedAt) }
      }
    },
  )
}
