import type { FastifyInstance } from 'fastify'
import { eventRoutingRoutes } from '../../server/routes/event-routing.js'
import type { FeatureRouteDeps } from '../types.js'

export async function registerRoutes(app: FastifyInstance, deps: FeatureRouteDeps): Promise<void> {
  await app.register(eventRoutingRoutes, { prefix: deps.prefix })
}
