import type { FastifyInstance } from 'fastify'
import { registerWikiRoutes } from '../../wiki/routes.js'

export async function registerRoutes(app: FastifyInstance): Promise<void> {
  await registerWikiRoutes(app)
}
