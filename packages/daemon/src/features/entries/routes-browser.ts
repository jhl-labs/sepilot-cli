import type { FastifyInstance } from 'fastify'
import { remoteBrowserRoutes } from '../../server/routes/browser-remote.js'
import { browserCdpRoutes } from '../../server/routes/browser-cdp.js'
import { browserObservationRoutes } from '../../server/routes/browser-observation.js'
import { viewsRoutes } from '../../server/routes/views.js'
import type { FeatureRouteDeps } from '../types.js'

export async function registerRoutes(app: FastifyInstance, deps: FeatureRouteDeps): Promise<void> {
  await app.register(remoteBrowserRoutes, { prefix: deps.prefix })
  await app.register(browserCdpRoutes, { prefix: deps.prefix })
  await app.register(browserObservationRoutes, { prefix: deps.prefix })
  await app.register(viewsRoutes, { prefix: deps.prefix })
}
