import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import {
  isDefaultPersonalTool,
  isLeanMemoryTool,
  toolExposureGroupForTool,
} from '../../tools/role-filter.js'

export async function toolsIntrospectRoutes(app: FastifyInstance) {
  app.get('/tools', async (_request, reply) => {
    const registry = app.runtime?.toolRegistry
    if (!registry) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    return registry.listRegistered().map((tool) => ({
      name: tool.name,
      description: tool.description,
      inputSchema: tool.inputSchema,
      autonomyRequired: undefined,
      enabled: registry.isEnabled(tool.name),
      source: registry.registrationSource(tool.name) ?? 'builtin',
      recommended: !tool.name.startsWith('memory.') || isLeanMemoryTool(tool.name),
      defaultExposure: isDefaultPersonalTool(tool.name),
      exposureGroup: toolExposureGroupForTool(tool.name),
    }))
  })

  app.get<{ Querystring: { sessionId?: string } }>('/tools/stats', async (request, reply) => {
    const statsStore = app.runtime?.toolStatsStore
    if (!statsStore) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    const query = request.query
    const sessionId = query.sessionId?.trim()
    const records = sessionId ? statsStore.list(sessionId) : []
    return records.map((record) => ({
      name: record.tool,
      success: record.successCount,
      failure: record.errorCount,
      total: record.totalCount,
      successRate: record.successRate,
      avgLatencyMs: record.avgDurationMs,
      recentErrorOutputs: record.recentErrorOutputs,
    }))
  })
}
