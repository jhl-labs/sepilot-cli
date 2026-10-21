import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'

export async function pluginRoutes(app: FastifyInstance) {
  app.get('/plugins', async (_request, reply) => {
    const runtime = app.runtime
    if (!runtime?.pluginLoader)
      return reply
        .status(503)
        .send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const plugins = runtime.pluginLoader.list()
    return {
      data: plugins.map((plugin) => ({
        ...plugin.manifest,
        status: plugin.status,
        error: plugin.error,
        security: plugin.security,
      })),
      meta: {
        providerFactoryTypes: runtime.providerFactoryRegistry?.listTypes?.() ?? [],
        channelFactoryTypes: runtime.channelFactoryRegistry?.listTypes?.() ?? [],
        hookHandlers: runtime.hookRegistry?.listHandlers?.() ?? [],
      },
    }
  })
}
