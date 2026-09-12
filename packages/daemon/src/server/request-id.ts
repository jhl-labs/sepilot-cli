import type { FastifyInstance, FastifyRequest } from 'fastify'
import { randomUUID } from 'node:crypto'
import { skipOverride } from './skip-override.js'

export async function requestIdPlugin(app: FastifyInstance) {
  app.addHook('onRequest', async (request, reply) => {
    const requestId = (request.headers['x-request-id'] as string) ?? randomUUID()
    request.requestId = requestId
    reply.header('x-request-id', requestId)
  })
}

skipOverride(requestIdPlugin)

/** Get request ID from Fastify request */
export function getRequestId(request: FastifyRequest): string {
  return request.requestId ?? 'unknown'
}
