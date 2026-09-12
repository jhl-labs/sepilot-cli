import type {
  FastifyError,
  FastifyInstance,
  FastifyReply,
  FastifyRequest,
} from 'fastify'

export class DaemonError extends Error {
  code: string
  retriable: boolean
  details?: unknown
  statusCode: number
  constructor(input: {
    code: string
    message: string
    statusCode?: number
    retriable?: boolean
    details?: unknown
  }) {
    super(input.message)
    this.code = input.code
    this.retriable = input.retriable ?? false
    this.details = input.details
    this.statusCode = input.statusCode ?? 500
  }
}

export interface MappedError {
  statusCode: number
  body: {
    code: string
    message: string
    retriable: boolean
    details?: unknown
  }
}

export function mapDaemonError(err: Error | FastifyError): MappedError {
  if (err instanceof DaemonError) {
    return {
      statusCode: err.statusCode,
      body: {
        code: err.code,
        message: err.message,
        retriable: err.retriable,
        details: err.details,
      },
    }
  }
  const fe = err as FastifyError
  if (
    typeof fe.statusCode === 'number' &&
    fe.statusCode >= 400 &&
    fe.statusCode < 600
  ) {
    return {
      statusCode: fe.statusCode,
      body: {
        code: fe.code ?? 'INVALID_REQUEST',
        message: fe.message,
        retriable: false,
      },
    }
  }
  return {
    statusCode: 500,
    body: {
      code: 'INTERNAL',
      message: err.message || 'internal error',
      retriable: false,
    },
  }
}

export function attachDaemonErrorHandler(app: FastifyInstance): void {
  app.setErrorHandler(
    (err: FastifyError, _req: FastifyRequest, reply: FastifyReply) => {
      const { statusCode, body } = mapDaemonError(err)
      void reply.status(statusCode).send(body)
    },
  )
}
