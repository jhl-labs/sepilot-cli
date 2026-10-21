import type { FastifyInstance, FastifyError } from 'fastify'
import { createLogger } from '../logger.js'
import { ConfigRevisionConflictError } from '../server/runtime/config-runtime.js'
import { redactUrlForLog } from './log-url.js'
import { skipOverride } from './skip-override.js'

const logger = createLogger('error')

export async function errorHandlerPlugin(app: FastifyInstance) {
  app.setErrorHandler((error: FastifyError, request, reply) => {
    const requestId = request.requestId ?? 'unknown'

    if (error instanceof ConfigRevisionConflictError) {
      logger.warn(`${request.method} ${redactUrlForLog(request.url)}: ${error.message}`, { requestId })
      reply.status(409).send({
        error: {
          code: 'CONFIG_REVISION_CONFLICT',
          message: error.message,
          requestId,
        },
      })
      return
    }

    logger.error(`${request.method} ${redactUrlForLog(request.url)}: ${error.message}`, { requestId, statusCode: error.statusCode })

    const statusCode = error.statusCode ?? 500
    const code = statusCode === 400 ? 'INVALID_REQUEST'
      : statusCode === 401 ? 'UNAUTHORIZED'
      : statusCode === 404 ? 'NOT_FOUND'
      : statusCode === 429 ? 'RATE_LIMITED'
      : 'INTERNAL_ERROR'

    reply.status(statusCode).send({
      error: {
        code,
        message: statusCode >= 500 ? 'Internal server error' : error.message,
        requestId,
      },
    })
  })

  app.setNotFoundHandler((request, reply) => {
    reply.status(404).send({
      error: {
        code: 'NOT_FOUND',
        message: `Route ${request.method} ${request.url} not found`,
        requestId: request.requestId,
      },
    })
  })
}

skipOverride(errorHandlerPlugin)
