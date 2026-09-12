import type { FastifyInstance } from 'fastify'
import type { ExternalAcpAgentDispatcher } from '../../acp/external-agent.js'
import { externalAcpDispatchRequestSchema } from './acp-agents-schema.js'
import {
  InvalidCwdError,
  invalidCwdResponse,
  resolveRequestCwd,
} from './request-cwd.js'
import '../fastify-types.js'

export interface ExternalAcpRoutesDeps {
  dispatcher: () => ExternalAcpAgentDispatcher | null
}

export function registerExternalAcpRoutes(
  app: FastifyInstance,
  deps: ExternalAcpRoutesDeps,
): void {
  app.post('/acp/agents/run', async (request, reply) => {
    const parsed = externalAcpDispatchRequestSchema.safeParse(request.body)
    if (!parsed.success) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: parsed.error.issues
            .map((issue) => `${issue.path.join('.') || 'body'}: ${issue.message}`)
            .join('; '),
        },
      })
    }

    const dispatcher = deps.dispatcher()
    if (!dispatcher) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'External ACP agent dispatcher not initialized',
        },
      })
    }

    let cwd: string | undefined
    try {
      cwd = await resolveRequestCwd(parsed.data.cwd)
    } catch (error) {
      if (error instanceof InvalidCwdError) {
        return reply.status(400).send(invalidCwdResponse(error))
      }
      throw error
    }

    try {
      const result = await dispatcher.dispatch({ ...parsed.data, cwd })
      return reply.status(200).send(result)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      const code = message.startsWith('EXTERNAL_ACP_INVALID_REQUEST')
        ? 'INVALID_REQUEST'
        : message.startsWith('EXTERNAL_ACP_UNKNOWN_AGENT')
          ? 'UNKNOWN_AGENT'
          : 'INTERNAL_ERROR'
      return reply.status(code === 'INTERNAL_ERROR' ? 500 : 400).send({
        error: { code, message },
      })
    }
  })
}

export async function externalAcpRoutes(app: FastifyInstance) {
  registerExternalAcpRoutes(app, {
    dispatcher: () => app.runtime?.externalAcpAgentDispatcher ?? null,
  })
}
