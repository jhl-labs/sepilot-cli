import type { FastifyInstance } from 'fastify'
import { registerWikiRoutes } from '../../wiki/routes.js'

/**
 * Registers the wiki capability routes. Only imported (and thus only bundled)
 * when the `wiki` feature is enabled.
 */
export async function registerRoutes(app: FastifyInstance): Promise<void> {
  await registerWikiRoutes(app)
}
