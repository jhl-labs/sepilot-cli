import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'

// OpenAPI components/overrides for the plugin routes live in
// `plugins-openapi.ts` so `server/app.ts` can register them (via the generated
// manifest) without statically importing this route-handler module — that keeps
// the `List plugins` marker and this handler out of the binary when the
// `plugins` feature is disabled.

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
