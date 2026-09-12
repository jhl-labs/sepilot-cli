import type { FastifyInstance } from 'fastify'
import { createLogger } from '../logger.js'
import { skipOverride } from './skip-override.js'

const logger = createLogger('http')

export async function requestLoggerPlugin(app: FastifyInstance) {
  app.addHook('onResponse', async (request, reply) => {
    const duration = reply.elapsedTime?.toFixed(0) ?? '?'
    const method = request.method
    const url = request.url
    const status = reply.statusCode
    const requestId = request.requestId ?? '-'

    if (url === '/api/v1/health') return // Skip health spam

    const level = status >= 500 ? 'error' : status >= 400 ? 'warn' : 'info'
    logger[level](`${method} ${url} ${status} ${duration}ms`, { requestId })
  })
}

skipOverride(requestLoggerPlugin)
