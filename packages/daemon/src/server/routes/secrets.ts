import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'

export async function secretsRoutes(app: FastifyInstance) {
  const runtime = app.runtime

  app.get('/secrets', async (_req, reply) => {
    if (!runtime?.secretVault)
      return reply
        .status(503)
        .send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    return { data: { keys: runtime.secretVault.list() } }
  })

  app.post<{ Params: { key: string }; Body: { value: string } }>(
    '/secrets/:key',
    async (request, reply) => {
      if (!runtime?.secretVault)
        return reply
          .status(503)
          .send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
      const params = request.params
      const body = request.body
      if (typeof body?.value !== 'string') {
        return reply
          .status(400)
          .send({ error: { code: 'BAD_REQUEST', message: 'value is required' } })
      }
      await runtime.secretVault.set(params.key, body.value)
      return { data: { ok: true } }
    },
  )

  app.delete<{ Params: { key: string } }>('/secrets/:key', async (request, reply) => {
    if (!runtime?.secretVault)
      return reply
        .status(503)
        .send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const removed = await runtime.secretVault.remove(params.key)
    return { data: { removed } }
  })
}
