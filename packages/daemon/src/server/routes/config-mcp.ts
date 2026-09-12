import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import { mcpServerSchema } from '../../config/schema.js'
import {
  applyAndPersistRuntimeUpdate,
} from '../runtime/config-runtime.js'
import { restoreMcpServerEnv } from '../runtime/config-mutations.js'
import { zodRequestValidation } from './utils.js'
import {
  mcpServerDisabledToolsRequestSchema,
  mcpServerNameParamsSchema,
  mcpServerToggleRequestSchema,
  type McpServerBody,
  type McpServerDisabledToolsBody,
  type McpServerNameParams,
  type McpServerToggleBody,
} from './config-schema.js'

export function registerConfigMcpRoutes(app: FastifyInstance): void {
  const runtime = app.runtime

  app.post<{ Body: McpServerBody }>('/config/mcp/servers', {
    preValidation: zodRequestValidation({
      body: {
        schema: mcpServerSchema,
        message: 'Invalid MCP server request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    await runtime.configMutationService.apply(
      'config.mcp.servers.upsert',
      async () => {
        const server = request.body
        const normalizedServer = restoreMcpServerEnv(
          runtime.config.mcp.servers,
          server,
        )

        const existingIndex = runtime.config.mcp.servers.findIndex(
          (entry) => entry.name === normalizedServer.name,
        )

        if (existingIndex >= 0) {
          runtime.config.mcp.servers[existingIndex] = normalizedServer
        } else {
          runtime.config.mcp.servers.push(normalizedServer)
        }

        await applyAndPersistRuntimeUpdate(runtime, new Set(['mcp.servers']))
      },
    )
    return { data: { updated: ['mcp.servers'] } }
  })

  app.post<{ Params: McpServerNameParams }>(
    '/config/mcp/servers/:name/trust-manifest',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: mcpServerNameParamsSchema,
          message: 'Invalid MCP server params',
        },
      }),
    },
    async (request, reply) => {
      if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
      const params = request.params

      const outcome = await runtime.configMutationService.apply(
        'config.mcp.servers.trust_manifest',
        async () => {
          const server = runtime.config.mcp.servers.find(
            (entry) => entry.name === params.name,
          )
          const status = runtime.mcpManager.listServers().find(
            (entry) => entry.name === params.name,
          )
          if (!server || !status) return { kind: 'not_found' as const }
          if (status.toolManifestStatus !== 'changed' || !status.toolManifest) {
            return { kind: 'not_changed' as const }
          }

          server.toolManifest = status.toolManifest
          await applyAndPersistRuntimeUpdate(runtime, new Set(['mcp.servers']))
          return { kind: 'trusted' as const, digest: status.toolManifest.digest }
        },
      )

      if (outcome.kind === 'not_found') {
        return reply.status(404).send({
          error: { code: 'NOT_FOUND', message: `MCP server not found: ${params.name}` },
        })
      }
      if (outcome.kind === 'not_changed') {
        return reply.status(409).send({
          error: {
            code: 'MCP_MANIFEST_NOT_CHANGED',
            message: `MCP server does not have a quarantined manifest change: ${params.name}`,
          },
        })
      }
      return { data: { name: params.name, digest: outcome.digest, trusted: true } }
    },
  )

  app.delete<{ Params: McpServerNameParams }>(
    '/config/mcp/servers/:name',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: mcpServerNameParamsSchema,
          message: 'Invalid MCP server params',
        },
      }),
    },
    async (request, reply) => {
      if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

      const params = request.params

      const outcome = await runtime.configMutationService.apply(
        'config.mcp.servers.delete',
        async () => {
          const nextServers = runtime.config.mcp.servers.filter(
            (server) => server.name !== params.name,
          )
          if (nextServers.length === runtime.config.mcp.servers.length) {
            return 'not_found' as const
          }

          runtime.config.mcp.servers = nextServers
          await applyAndPersistRuntimeUpdate(runtime, new Set(['mcp.servers']))
          return 'ok' as const
        },
      )

      if (outcome === 'not_found') {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: `MCP server not found: ${params.name}`,
          },
        })
      }
      return { data: { updated: ['mcp.servers'] } }
    },
  )

  app.post<{
    Params: McpServerNameParams
    Body: McpServerToggleBody
  }>('/config/mcp/servers/:name/enable', {
    preValidation: zodRequestValidation({
      params: {
        schema: mcpServerNameParamsSchema,
        message: 'Invalid MCP server params',
      },
      body: {
        schema: mcpServerToggleRequestSchema,
        message: 'Invalid MCP server toggle request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const params = request.params
    const body = request.body

    const outcome = await runtime.configMutationService.apply(
      'config.mcp.servers.enable',
      async () => {
        const server = runtime.config.mcp.servers.find(
          (entry) => entry.name === params.name,
        )
        if (!server) {
          return 'not_found' as const
        }

        server.enabled = body.enabled
        await applyAndPersistRuntimeUpdate(runtime, new Set(['mcp.servers']))
        return 'ok' as const
      },
    )

    if (outcome === 'not_found') {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: `MCP server not found: ${params.name}`,
        },
      })
    }
    return { data: { updated: ['mcp.servers'] } }
  })

  app.post<{
    Params: McpServerNameParams
    Body: McpServerDisabledToolsBody
  }>('/config/mcp/servers/:name/disabled-tools', {
    preValidation: zodRequestValidation({
      params: {
        schema: mcpServerNameParamsSchema,
        message: 'Invalid MCP server params',
      },
      body: {
        schema: mcpServerDisabledToolsRequestSchema,
        message: 'Invalid MCP server disabled-tools request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const params = request.params
    const body = request.body

    const outcome = await runtime.configMutationService.apply(
      'config.mcp.servers.disabled_tools',
      async () => {
        const server = runtime.config.mcp.servers.find(
          (entry) => entry.name === params.name,
        )
        if (!server) {
          return 'not_found' as const
        }

        // Dedupe and persist; trimming is already enforced by the schema.
        // The config schema makes `disabledTools` a required array with a
        // default of [], so an empty list still has to be an array.
        server.disabledTools = Array.from(new Set(body.disabledTools))
        await applyAndPersistRuntimeUpdate(runtime, new Set(['mcp.servers']))
        return 'ok' as const
      },
    )

    if (outcome === 'not_found') {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: `MCP server not found: ${params.name}`,
        },
      })
    }
    return { data: { updated: ['mcp.servers'] } }
  })
}
