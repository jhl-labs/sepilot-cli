import type { FastifyInstance } from 'fastify'
import { swarmRoutes } from '../../server/routes/swarm.js'
import type { FeatureRouteDeps } from '../types.js'

export async function registerRoutes(app: FastifyInstance, _deps: FeatureRouteDeps): Promise<void> {
  await app.register(swarmRoutes)
}
