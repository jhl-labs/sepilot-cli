import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import {
  loadAutoCompactedSessionContext,
  resolveSessionContextMaxMessages,
} from '../../agent/auto-compaction.js'
import { buildSystemPrompt } from '../../agent/system-prompt.js'
import {
  createTraceRedactionContext,
  redactSecretKeys,
} from '../../observability/trace-redaction.js'

export async function sessionPromptRoutes(app: FastifyInstance) {
  app.get<{
    Params: { id: string }
    Querystring: { turn?: string }
  }>('/sessions/:id/prompt', async (request, reply) => {
    // This endpoint returns the fully assembled system prompt and session
    // history. Extension session access must not implicitly grant access to
    // daemon-wide instructions or legacy global memory embedded in that
    // prompt. An unauthenticated loopback daemon has no authContext and keeps
    // the existing local-operator behavior; authenticated extensions fail
    // closed here.
    if (request.authContext?.kind === 'extension') {
      return reply.status(403).send({
        error: {
          code: 'SESSION_PROMPT_ADMIN_DENIED',
          message: 'Session prompt introspection requires the daemon master token.',
        },
      })
    }

    const runtime = app.runtime
    if (
      !runtime?.sessions ||
      !runtime.config ||
      !runtime.toolRegistry ||
      !runtime.skillRegistry
    ) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    const query = request.query
    const params = request.params
    const turn = query.turn ?? 'last'
    if (turn !== 'last') {
      return reply.status(400).send({
        error: { code: 'BAD_REQUEST', message: 'Only turn=last is supported' },
      })
    }

    const session = await runtime.sessions.get(params.id)
    if (!session) {
      return reply.status(404).send({
        error: { code: 'NOT_FOUND', message: 'Session not found' },
      })
    }

    const { messages } = await loadAutoCompactedSessionContext({
      sessionStore: runtime.sessions,
      sessionId: session.id,
      provider: runtime.providerRegistry?.get(session.provider),
      model: session.model,
      hooks: runtime.hookRegistry,
      maxMessages: resolveSessionContextMaxMessages(),
    })
    const systemPrompt = await buildSystemPrompt({
      config: runtime.config,
      tools: runtime.toolRegistry,
      skills: runtime.skillRegistry,
      fileMemory: runtime.fileMemory,
      cwd: session.cwd,
      sessionId: session.id,
    })
    const payload = {
      sessionId: session.id,
      provider: session.provider,
      model: session.model,
      turn,
      systemPrompt,
      messages,
      tools: runtime.toolRegistry.toToolDefinitions(),
    }
    return redactSecretKeys(
      payload,
      createTraceRedactionContext({
        cwd: session.cwd,
        stateDir: runtime.dataDir,
      }),
      '',
      0,
      { maxStringLength: Number.MAX_SAFE_INTEGER },
    )
  })
}
