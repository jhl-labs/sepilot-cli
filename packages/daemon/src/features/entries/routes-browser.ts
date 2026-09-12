import type { FastifyInstance } from 'fastify'
import { remoteBrowserRoutes } from '../../server/routes/browser-remote.js'
import type { FeatureRouteDeps } from '../types.js'

export async function registerRoutes(app: FastifyInstance, deps: FeatureRouteDeps): Promise<void> {
  await app.register(remoteBrowserRoutes, { prefix: deps.prefix })
}
