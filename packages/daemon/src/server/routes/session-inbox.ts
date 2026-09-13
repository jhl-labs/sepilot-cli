import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { SessionInbox } from '../runtime/session-inbox.js'
import { zodRequestValidation } from './utils.js'

export function registerSessionInboxRoutes(app: FastifyInstance): void {
  const params = z.object({ id: z.string().min(1) })
  const query = z.object({ after: z.coerce.number().int().min(0).optional(), limit: z.coerce.number().int().min(1).max(100).optional(), unreadOnly: z.enum(['true', 'false']).optional() }).strict()
  app.get<{ Params: { id: string }; Querystring: z.infer<typeof query> }>('/sessions/:id/inbox', {
    preValidation: zodRequestValidation({ params: { schema: params, message: 'Invalid session id' }, query: { schema: query, message: 'Invalid inbox query' } }),
  }, async (request, reply) => {
    const params = request.params
    const query = request.query
    if (!await app.runtime?.sessions.get(params.id)) return reply.code(404).send({ error: { code: 'NOT_FOUND', message: 'Session not found' } })
    return { data: new SessionInbox().list(params.id, { ...query, unreadOnly: query.unreadOnly === 'true' }) }
  })
  app.post<{ Params: { id: string }; Body: { seq: number } }>('/sessions/:id/inbox/ack', {
    preValidation: zodRequestValidation({ params: { schema: params, message: 'Invalid session id' }, body: { schema: z.object({ seq: z.number().int().positive() }).strict(), message: 'Invalid inbox acknowledgement' } }),
  }, async (request, reply) => {
    const params = request.params
    const body = request.body
    if (!await app.runtime?.sessions.get(params.id)) return reply.code(404).send({ error: { code: 'NOT_FOUND', message: 'Session not found' } })
    if (!new SessionInbox().acknowledge(params.id, body.seq)) return reply.code(404).send({ error: { code: 'NOT_FOUND', message: 'Receipt not found in this session' } })
    return { data: { acknowledged: true } }
  })
}
