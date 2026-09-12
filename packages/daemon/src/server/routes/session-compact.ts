import { randomUUID } from 'node:crypto'
import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import {
  zodRequestValidation,
} from './utils.js'
import {
  sessionIdParamsSchema,
  type SessionIdParams,
} from './sessions-schema.js'
import { loadSessionContext } from '../../agent/context-loader.js'
import { compactSessionMessages } from '../../agent/session-compaction.js'
import { tokenCalibration } from '../../providers/token-calibration.js'

export function registerSessionCompactRoute(app: FastifyInstance): void {
  const runtime = app.runtime

  app.post<{ Params: SessionIdParams }>('/sessions/:id/compact', {
    preValidation: zodRequestValidation({
      params: {
        schema: sessionIdParamsSchema,
        message: 'Invalid session id',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    const params = request.params
    const session = await runtime.sessions.get(params.id)
    if (!session) {
      return reply.status(404).send({
        error: { code: 'NOT_FOUND', message: 'Session not found' },
      })
    }

    const sessionMessages = await loadSessionContext(
      runtime.sessions,
      params.id,
      Number.MAX_SAFE_INTEGER,
    )

    if (sessionMessages.length < 4) {
      return reply.status(400).send({
        error: { code: 'INVALID_REQUEST', message: 'Session too short to compact' },
      })
    }

    // Summarize with the session's own provider/model. `models[0]` is whatever
    // order the provider returned — alphabetical for Ollama, so an embedding
    // model can sort first and a manual /compact would try to summarize with a
    // model that cannot chat. Fall back to the default provider only when the
    // session's provider is gone.
    const sessionProvider = session.provider
      ? runtime.providerRegistry.get(session.provider)
      : undefined
    const provider = sessionProvider ?? runtime.providerRegistry.getDefault()
    const model = (sessionProvider && session.model)
      ? session.model
      : provider?.models.find((candidate) => candidate.id === session.model)?.id
        ?? provider?.models[0]?.id
    const compaction = await compactSessionMessages({
      messages: sessionMessages,
      provider: provider ?? undefined,
      model,
      hooks: runtime.hookRegistry,
      sessionId: params.id,
      charsPerToken: tokenCalibration.charsPerToken(provider?.id, model),
    })

    if (!compaction || compaction.removedMessageCount === 0) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: 'Session does not have enough older context to compact safely',
        },
      })
    }

    await runtime.sessions.appendEvent(params.id, {
      type: 'context_compact',
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      beforeTokens: compaction.originalTokens,
      afterTokens: compaction.compactedTokens,
      summary: compaction.summary,
      strategy: compaction.strategy,
      removedMessageCount: compaction.removedMessageCount,
      preservedMessageCount: compaction.preservedMessageCount,
      preservedMessages: compaction.preservedMessages,
    })

    return {
      data: {
        sessionId: params.id,
        originalTokens: compaction.originalTokens,
        compactedTokens: compaction.compactedTokens,
        savedTokens: compaction.savedTokens,
        summary: compaction.summary,
        strategy: compaction.strategy,
        removedMessageCount: compaction.removedMessageCount,
        preservedMessageCount: compaction.preservedMessageCount,
      },
    }
  })
}
