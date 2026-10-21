import type { FastifyInstance, FastifyRequest } from 'fastify'
import { z } from 'zod'
import type { RequestAuthContext } from '../auth.js'
import { randomUUID } from 'node:crypto'
import { remoteBrowserBridge, remoteFrameSchema } from '../../tools/browser-remote.js'
import { zodRequestValidation } from './utils.js'
import {
  buildSseResponseHeaders,
  registerSseDisconnectHandler,
  trackSseConnection,
} from '../sse-response.js'

const connectSchema = z.object({ label: z.string().trim().min(1).max(100) })
const exchangeSchema = z.object({
  id: z.string().uuid(),
  frame: remoteFrameSchema.optional(),
  paused: z.boolean().optional(),
  result: z
    .object({
      id: z.string().uuid(),
      output: z.string().max(50000).optional(),
      error: z.string().max(1000).optional(),
    })
    .optional(),
})
const controlSchema = z.object({
  id: z.string().uuid(),
  sessionId: z.string().min(1).max(200),
  mode: z.enum(['human', 'agent']),
})
const idSchema = z.object({ id: z.string().uuid() })
function principal(request: FastifyRequest) {
  return (request as FastifyRequest & { authContext?: RequestAuthContext }).authContext
}

export async function remoteBrowserRoutes(app: FastifyInstance) {
  app.setErrorHandler((error, _request, reply) => {
    const detail = error as { statusCode?: number; message?: string }
    reply
      .code(detail.statusCode ?? 409)
      .send({ error: { message: detail.message ?? 'Browser request failed' } })
  })
  const bridge = () => {
    if (!app.runtime?.toolRegistry) throw new Error('Runtime not initialized')
    return remoteBrowserBridge(app.runtime.toolRegistry)
  }
  app.addHook('onClose', async () => {
    if (app.runtime?.toolRegistry) bridge().close()
  })
  app.addHook('preHandler', async (request, reply) => {
    const auth = principal(request)
    if (!auth)
      return reply.code(401).send({ error: { message: 'Browser control requires authentication' } })
    if (
      !request.url.split('?')[0]!.startsWith('/api/v1/browser/extension/') &&
      auth.kind !== 'master'
    ) {
      return reply.code(403).send({ error: { message: 'Desktop authorization required' } })
    }
  })
  const owner = (request: FastifyRequest) => {
    const auth = principal(request)
    return auth?.kind === 'extension' ? auth.tokenId : 'master'
  }
  const validate = (schema: z.ZodTypeAny) =>
    zodRequestValidation({ body: { schema, message: 'Invalid browser request' } })
  app.post<{ Body: z.infer<typeof connectSchema> }>(
    '/browser/extension/connect',
    { preValidation: validate(connectSchema) },
    async (request) => {
      const body = connectSchema.parse(request.body)
      return bridge().connect(owner(request), body.label)
    },
  )
  app.post<{ Body: z.infer<typeof exchangeSchema> }>(
    '/browser/extension/exchange',
    { bodyLimit: 2_100_000, preValidation: validate(exchangeSchema) },
    async (request, reply) => {
      const body = request.body
      if (!bridge().owns(body.id, owner(request)))
        return reply
          .code(403)
          .send({ error: { message: 'Browser connection belongs to another extension' } })
      return bridge().exchange(body.id, body)
    },
  )
  app.post<{ Body: z.infer<typeof idSchema> }>(
    '/browser/extension/disconnect',
    { preValidation: validate(idSchema) },
    async (request, reply) => {
      const body = request.body
      if (!bridge().owns(body.id, owner(request)))
        return reply
          .code(403)
          .send({ error: { message: 'Browser connection belongs to another extension' } })
      bridge().disconnect(body.id)
      return { ok: true }
    },
  )
  app.get('/browser/connections', async () => bridge().list())
  app.get<{ Params: { id: string } }>('/browser/connections/:id/frames', async (request, reply) => {
    const { id } = idSchema.parse(request.params)
    const connection = bridge().list().find((item) => item.id === id)
    if (!connection) return reply.code(404).send({ error: { message: 'Browser connection not found' } })
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
    const send = (event: string, data: unknown) => {
      if (closed) return
      reply.raw.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
    }
    const heartbeat = setInterval(() => send('heartbeat', { timestamp: new Date().toISOString() }), 15_000)
    heartbeat.unref?.()
    let unsubscribe = () => {}
    const close = () => {
      if (closed) return
      closed = true
      unsubscribe()
      clearInterval(heartbeat)
      if (!reply.raw.destroyed && !reply.raw.writableEnded) reply.raw.end()
    }
    try {
      unsubscribe = bridge().subscribeFrames(id, (frame) => send('frame', frame))
    } catch {
      close()
      return
    }
    registerSseDisconnectHandler(request, reply, close)
    trackSseConnection(app, request, reply, { label: 'browser-frames' })
    send('ready', { id, transport: connection.transport })
    reply.raw.flushHeaders?.()
    if (connection.frame) send('frame', connection.frame)
  })
  app.post<{ Body: z.infer<typeof controlSchema> }>(
    '/browser/control',
    { preValidation: validate(controlSchema) },
    async (request) => {
      const body = controlSchema.parse(request.body)
      return bridge().control(body.id, body.sessionId, body.mode)
    },
  )
  app.post<{ Body: z.infer<typeof idSchema> }>(
    '/browser/disconnect',
    { preValidation: validate(idSchema) },
    async (request) => {
      const body = request.body
      bridge().disconnect(body.id)
      return { ok: true }
    },
  )
}
