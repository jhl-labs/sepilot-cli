import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'

export async function mcpMetricsRoutes(app: FastifyInstance) {
  const runtime = app.runtime

  app.get('/mcp/metrics', async (_req, reply) => {
    if (!runtime?.mcpManager) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    return { data: runtime.mcpManager.getMetrics() }
  })

  app.get<{ Params: { server: string } }>('/mcp/metrics/:server', async (request, reply) => {
    if (!runtime?.mcpManager) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const snap = runtime.mcpManager.getMetrics(params.server)
    const entry = snap.servers[params.server]
    if (!entry) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Server not found' } })
    return { data: entry }
  })
}
