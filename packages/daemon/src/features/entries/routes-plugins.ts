import type { FastifyInstance } from 'fastify'
import { pluginRoutes } from '../../server/routes/plugins.js'
import type { FeatureRouteDeps } from '../types.js'

export async function registerRoutes(app: FastifyInstance, deps: FeatureRouteDeps): Promise<void> {
  await app.register(pluginRoutes, { prefix: deps.prefix })
}
