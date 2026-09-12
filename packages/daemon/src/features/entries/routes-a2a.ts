import type { FastifyInstance } from 'fastify'
import { a2aRoutes } from '../../server/routes/a2a.js'
import type { FeatureRouteDeps } from '../types.js'

export async function registerRoutes(app: FastifyInstance, _deps: FeatureRouteDeps): Promise<void> {
  await app.register(a2aRoutes)
}
