import type { FastifyInstance } from 'fastify'
import { pagesRoutes } from '../../server/routes/pages.js'
import type { FeatureRouteDeps } from '../types.js'

export async function registerRoutes(app: FastifyInstance, deps: FeatureRouteDeps): Promise<void> {
  await app.register(pagesRoutes, { prefix: deps.prefix })
}
