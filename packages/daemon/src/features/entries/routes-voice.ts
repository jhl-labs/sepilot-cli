import type { FastifyInstance } from 'fastify'
import { voiceRoutes } from '../../server/routes/voice.js'
import type { FeatureRouteDeps } from '../types.js'

export async function registerRoutes(app: FastifyInstance, deps: FeatureRouteDeps): Promise<void> {
  await app.register(voiceRoutes, { prefix: deps.prefix })
}
