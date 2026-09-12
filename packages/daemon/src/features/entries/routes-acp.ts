import type { FastifyInstance } from 'fastify'
import { externalAcpRoutes } from '../../server/routes/acp-agents.js'
import type { FeatureRouteDeps } from '../types.js'

export async function registerRoutes(app: FastifyInstance, deps: FeatureRouteDeps): Promise<void> {
  await app.register(externalAcpRoutes, { prefix: deps.prefix })
}
