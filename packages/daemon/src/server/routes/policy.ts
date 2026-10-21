import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'

export async function policyRoutes(app: FastifyInstance) {
  app.get('/policy', async (_request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    return { data: runtime.policyEngine.describe() }
  })
}
