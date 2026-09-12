import type { FastifyInstance } from 'fastify'
import { appsRoutes } from '../../server/routes/apps.js'
import type { FeatureRouteDeps } from '../types.js'

export async function registerRoutes(app: FastifyInstance, deps: FeatureRouteDeps): Promise<void> {
  await app.register(appsRoutes, { prefix: deps.prefix })
}
