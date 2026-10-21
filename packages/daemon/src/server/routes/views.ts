import type { FastifyInstance, FastifyRequest } from 'fastify'
import { randomUUID } from 'node:crypto'
import { z } from 'zod'
import type { RequestAuthContext } from '../auth.js'
import { liveViewRegistry } from '../../views/live-view-registry.js'
import {
  buildSseResponseHeaders,
  registerSseDisconnectHandler,
  trackSseConnection,
} from '../sse-response.js'

const idSchema = z.object({ id: z.string().min(1).max(100) })
const listSchema = z.object({ sessionId: z.string().min(1).max(200).optional() })

function principal(request: FastifyRequest): RequestAuthContext | undefined {
  return (request as FastifyRequest & { authContext?: RequestAuthContext }).authContext
}

export async function viewsRoutes(app: FastifyInstance) {
  const runtime = () => {
    if (!app.runtime) throw new Error('Runtime not initialized')
    return app.runtime
  }
  const registry = () => liveViewRegistry(runtime().toolRegistry)
  app.addHook('preHandler', async (request, reply) => {
    const auth = principal(request)
    if (!auth) return reply.code(401).send({ error: { message: 'Authentication required' } })
    if (auth.kind !== 'master')
      return reply.code(403).send({ error: { message: 'Live views require desktop authorization' } })
  })

  app.get<{ Querystring: z.infer<typeof listSchema> }>('/views', async (request) => {
    const { sessionId } = listSchema.parse(request.query)
    return registry().list(sessionId)
  })

  app.get<{ Params: { id: string } }>('/views/:id/still', async (request, reply) => {
    const { id } = idSchema.parse(request.params)
    const still = await registry().still(id, runtime().config.views.stills.maxWidth, runtime().config.views.stills.format)
    if (!still) return reply.code(404).send({ error: { message: 'View not found or has no frame yet' } })
    return still
  })

  app.get<{ Params: { id: string } }>('/views/:id/frames', async (request, reply) => {
    const { id } = idSchema.parse(request.params)
    const view = registry().get(id)
    if (!view) return reply.code(404).send({ error: { message: 'View not found' } })
    reply.hijack()
    reply.raw.writeHead(
      200,
      buildSseResponseHeaders(request, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        'X-Request-ID': request.requestId ?? randomUUID(),
      }),
    )
    let closed = false
    let seq = 0
    const send = (event: string, data: unknown) => {
      if (closed) return
      seq += 1
      reply.raw.write(`id: ${seq}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
    }
    const heartbeat = setInterval(() => send('heartbeat', { timestamp: new Date().toISOString() }), 15_000)
    heartbeat.unref?.()
    let unsubscribeFrames = () => {}
    let unsubscribeEvents = () => {}
    const close = () => {
      if (closed) return
      closed = true
      unsubscribeFrames()
      unsubscribeEvents()
      clearInterval(heartbeat)
      if (!reply.raw.destroyed && !reply.raw.writableEnded) reply.raw.end()
    }
    try {
      unsubscribeFrames = registry().subscribeFrames(id, (frame) => send('frame', frame))
    } catch {
      close()
      return
    }
    unsubscribeEvents = registry().onEvent((event) => {
      if (event.type === 'changed' && event.view.id === id) send('change', event.view)
      if (event.type === 'detached' && event.viewId === id) close()
    })
    registerSseDisconnectHandler(request, reply, close)
    trackSseConnection(app, request, reply, { label: 'view-frames' })
    send('ready', { view })
    reply.raw.flushHeaders?.()
  })
}
