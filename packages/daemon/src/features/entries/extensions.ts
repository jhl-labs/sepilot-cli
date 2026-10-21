import type { FastifyInstance } from 'fastify'
import { registerExtensionsCapabilityRoutes } from '../../server/routes/extensions-capability.js'

export async function registerRoutes(app: FastifyInstance): Promise<void> {
  await registerExtensionsCapabilityRoutes(app)
}
