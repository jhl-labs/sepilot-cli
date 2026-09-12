import { randomUUID } from 'node:crypto'
import type { FastifyReply, FastifyRequest } from 'fastify'
import { buildSseResponseHeaders, registerSseDisconnectHandler } from '../sse-response.js'

export interface JsonWatchOptions<TPayload> {
  eventName: string
  subscribers: Set<(payload: TPayload) => void>
  buildSnapshot: () => TPayload
  buildHeartbeat: () => TPayload
}

export function openJsonWatch<TPayload>(
  req: FastifyRequest,
  reply: FastifyReply,
  options: JsonWatchOptions<TPayload>,
): void {
  reply.hijack()
  reply.raw.writeHead(
    200,
    buildSseResponseHeaders(req, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Request-ID': req.requestId ?? randomUUID(),
    }),
  )

  let closed = false
  const send = (payload: TPayload) => {
    if (closed) return
    reply.raw.write(`event: ${options.eventName}\ndata: ${JSON.stringify(payload)}\n\n`)
  }
  const heartbeat = setInterval(() => {
    send(options.buildHeartbeat())
  }, 15_000)
  heartbeat.unref?.()
  const snapshotTimer = setInterval(() => {
    send(options.buildSnapshot())
  }, 20_000)
  snapshotTimer.unref?.()
  const close = () => {
    if (closed) return
    closed = true
    options.subscribers.delete(send)
    clearInterval(heartbeat)
    clearInterval(snapshotTimer)
    if (!reply.raw.destroyed && !reply.raw.writableEnded) {
      reply.raw.end()
    }
  }

  options.subscribers.add(send)
  registerSseDisconnectHandler(req, reply, close)
  send(options.buildSnapshot())
}
