import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { buildSharedSessionPayload } from '../../share/payload.js'
import { renderSharedSessionHtml } from '../../share/renderer.js'
import { verifyShareToken } from '../../share/token.js'
import { zodRequestValidation } from './utils.js'

const sharePublicParamsSchema = z.object({
  token: z.string().min(1),
})

const sharePublicQuerySchema = z.object({
  format: z.string().optional(),
})

type SharePublicParams = z.output<typeof sharePublicParamsSchema>
type SharePublicQuery = z.output<typeof sharePublicQuerySchema>

function getPublicShareConfig(app: FastifyInstance): {
  enabled: boolean
  secret: string
} {
  const config = app.runtime?.config.share?.public
  return {
    enabled: config?.enabled === true,
    secret: config?.secret ?? '',
  }
}

export async function sharePublicRoutes(app: FastifyInstance): Promise<void> {
  app.get<{ Params: SharePublicParams; Querystring: SharePublicQuery }>('/share/:token', {
    preValidation: zodRequestValidation({
      params: {
        schema: sharePublicParamsSchema,
        message: 'Invalid share token',
      },
      query: {
        schema: sharePublicQuerySchema,
        message: 'Invalid share query',
      },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).type('text/plain; charset=utf-8').send('runtime unavailable')
    }

    const { enabled, secret } = getPublicShareConfig(app)
    if (!enabled || !secret) {
      return reply.status(404).type('text/plain; charset=utf-8').send('share disabled')
    }

    const params = request.params
    const result = verifyShareToken(params.token, secret)
    if (!result.valid || !result.sessionId) {
      const reason = result.reason ? `: ${result.reason}` : ''
      return reply.status(403).type('text/plain; charset=utf-8').send(`invalid share token${reason}`)
    }

    const session = await runtime.sessions.get(result.sessionId)
    if (!session) {
      return reply.status(404).type('text/plain; charset=utf-8').send('session not found')
    }

    const events = await runtime.sessions.getEvents(result.sessionId)
    const query = request.query
    const format = typeof query.format === 'string'
      ? query.format
      : undefined

    if (format === 'json') {
      return {
        data: buildSharedSessionPayload(session, events),
      }
    }

    return reply
      .header('content-type', 'text/html; charset=utf-8')
      .send(renderSharedSessionHtml(session, events))
  })
}
