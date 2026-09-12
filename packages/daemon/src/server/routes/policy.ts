import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'

/**
 * GET /policy — return a read-only snapshot of the active tool policy so
 * cli/desktop surfaces can show *which* tools are gated, blocked, or
 * autonomous without operators needing to read the YAML file directly.
 *
 * This is a deliberately small surface: only the rule table, not the
 * approval-decisions store, audit history, or any other security-adjacent
 * state. Surfaces that need decisions/audit go through the dedicated
 * approvals/audit routes.
 */
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
