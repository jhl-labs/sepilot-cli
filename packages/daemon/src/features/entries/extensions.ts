import type { FastifyInstance } from 'fastify'
import { registerExtensionsCapabilityRoutes } from '../../server/routes/extensions-capability.js'

/**
 * Registers the extensions capability routes. Only imported (and thus only
 * bundled) when the `extensions` feature is enabled.
 */
export async function registerRoutes(app: FastifyInstance): Promise<void> {
  await registerExtensionsCapabilityRoutes(app)
}
