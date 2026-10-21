import type { FastifyInstance, FastifyRequest } from 'fastify'
import { z } from 'zod'
import type { RequestAuthContext } from '../auth.js'
import { browserObservations } from '../../tools/browser-observation.js'
import { zodRequestValidation } from './utils.js'

const startSchema = z.object({ sessionId: z.string().min(1).max(200), screenSharingAcknowledged: z.literal(true) }).strict()
const idSchema = z.object({ id: z.string().uuid() })
const querySchema = z.object({ after: z.coerce.number().int().nonnegative().safe().default(0) })

export async function browserObservationRoutes(app: FastifyInstance): Promise<void> {
  app.addHook('preHandler', async (request, reply) => {
    const auth = (request as FastifyRequest & { authContext?: RequestAuthContext }).authContext
    if (!auth) return reply.code(401).send({ error: { message: 'Browser observation requires authentication' } })
    if (auth.kind !== 'master') return reply.code(403).send({ error: { message: 'Daemon owner authorization required' } })
  })
  app.post('/browser/observations', { preValidation: zodRequestValidation({ body: { schema: startSchema, message: 'Explicit screen privacy acknowledgement required' } }) }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.sessions || !runtime.toolRegistry) return reply.code(503).send({ error: { message: 'Runtime not initialized' } })
    const input = startSchema.parse(request.body)
    if (!await runtime.sessions.get(input.sessionId)) return reply.code(404).send({ error: { code: 'SESSION_NOT_FOUND', message: 'Session not found' } })
    const observations = browserObservations(runtime.toolRegistry)
    observations.setLineageResolver(async sourceId => {
      const lineage = [sourceId]
      let child = sourceId
      for (let depth = 0; depth < 8; depth++) {
        const session = await runtime.sessions.get(child)
        if (!session?.tags.includes('subagent')) break
        const parents = session.tags.filter(tag => tag.startsWith('parent:'))
        if (parents.length !== 1) break
        const parent = parents[0]!.slice('parent:'.length)
        if (!parent || lineage.includes(parent) || !runtime.subagentDispatcher) break
        const children = await runtime.subagentDispatcher.listChildren(parent, 64)
        if (!children.some(candidate => candidate.sessionId === child)) break
        lineage.push(parent)
        child = parent
      }
      return lineage
    })
    return reply.header('cache-control', 'private, no-store').send(observations.start(input.sessionId))
  })
  app.get<{ Params: { id: string }; Querystring: { after?: string } }>('/browser/observations/:id', {
    preValidation: zodRequestValidation({ params: { schema: idSchema, message: 'Invalid observation id' }, query: { schema: querySchema, message: 'Invalid frame sequence' } }),
  }, async (request, reply) => {
    if (!app.runtime?.toolRegistry) return reply.code(503).send({ error: { message: 'Runtime not initialized' } })
    const { id } = idSchema.parse(request.params), { after } = querySchema.parse(request.query)
    const view = browserObservations(app.runtime.toolRegistry).read(id, after)
    return view ? reply.header('cache-control', 'private, no-store').send(view) : reply.code(404).send({ error: { message: 'Observation expired or stopped' } })
  })
  app.delete<{ Params: { id: string } }>('/browser/observations/:id', { preValidation: zodRequestValidation({ params: { schema: idSchema, message: 'Invalid observation id' } }) }, async (request) => {
    const { id } = idSchema.parse(request.params)
    if (app.runtime?.toolRegistry) browserObservations(app.runtime.toolRegistry).stop(id)
    return { ok: true }
  })
  app.addHook('onClose', async () => { if (app.runtime?.toolRegistry) await browserObservations(app.runtime.toolRegistry).close() })
}
