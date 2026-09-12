import type { FastifyInstance } from 'fastify'
import { listCapabilities } from '../capabilities/registry.js'

export async function registerCapabilitiesRoute(
  app: FastifyInstance,
): Promise<void> {
  app.get('/capabilities', async () => ({
    capabilities: listCapabilities(),
  }))
}
