import type { FastifyInstance } from 'fastify'
import type { SubagentDispatcher } from '../../agent/subagent-dispatcher.js'
import { listSubagentDelegationCategories } from '../../agent/subagent-categories.js'
import { subagentDispatchRequestSchema } from './subagents-schema.js'
import '../fastify-types.js'

export interface SubagentRoutesDeps {
  /**
   * Resolver for the dispatcher instance. A function (not a direct
   * reference) so the route stays valid when the runtime swaps the
   * dispatcher during config reloads, and so tests can inject a
   * stub without rebuilding the entire runtime object.
   */
  dispatcher: () => SubagentDispatcher | null
}

/**
 * `POST /api/v1/subagents/dispatch` — synchronous HTTP entry point for the
 * isolated subagent dispatcher. The dispatcher itself is also surfaced as
 * the `subagent.dispatch` LLM tool; this route exists so cli/desktop
 * surfaces (and external automation) can invoke a subagent without going
 * through the chat loop.
 *
 * Privilege-escalation attempts (asking for tools the parent does not
 * have) are rejected as 400 with `code: 'SUBAGENT_TOOL_ESCALATION'`.
 * Empty prompts → 400 `INVALID_REQUEST`. Anything else surfaces as
 * 500 `INTERNAL_ERROR`. The success body is the raw
 * `SubagentDispatchResult` (output, sessionId, iterations, usage,
 * truncated, status).
 */
export function registerSubagentRoutes(app: FastifyInstance, deps: SubagentRoutesDeps): void {
  app.get('/subagents/categories', async (_request, reply) => {
    return reply.status(200).send({
      categories: listSubagentDelegationCategories().map((category) => ({
        id: category.id,
        label: category.label,
        description: category.description,
        defaultMaxIterations: category.defaultMaxIterations,
        toolHints: category.toolHints,
      })),
    })
  })

  app.post('/subagents/dispatch', async (request, reply) => {
    const parsed = subagentDispatchRequestSchema.safeParse(request.body)
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
          message: 'Subagent dispatcher not initialized',
        },
      })
    }

    const controller = new AbortController()
    const abort = () => controller.abort(new Error('subagent client disconnected'))
    request.raw.once('aborted', abort)
    reply.raw.once('close', abort)
    try {
      if (request.raw.aborted || reply.raw.destroyed) abort()
      const result = await dispatcher.dispatch({ ...parsed.data, signal: controller.signal })
      return reply.status(200).send(result)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      const code = (err as { code?: string }).code
      if (code === 'SUBAGENT_TOOL_ESCALATION' || message.startsWith('SUBAGENT_TOOL_ESCALATION')) {
        return reply.status(400).send({
          error: { code: 'SUBAGENT_TOOL_ESCALATION', message },
        })
      }
      if (code === 'SUBAGENT_CATEGORY_UNKNOWN' || message.startsWith('SUBAGENT_CATEGORY_UNKNOWN')) {
        return reply.status(400).send({
          error: { code: 'SUBAGENT_CATEGORY_UNKNOWN', message },
        })
      }
      if (code === 'SUBAGENT_AGENT_UNKNOWN' || message.startsWith('SUBAGENT_AGENT_UNKNOWN')) {
        return reply.status(400).send({
          error: { code: 'SUBAGENT_AGENT_UNKNOWN', message },
        })
      }
      return reply.status(500).send({
        error: { code: 'INTERNAL_ERROR', message },
      })
    } finally {
      request.raw.off('aborted', abort)
      reply.raw.off('close', abort)
    }
  })
}

export async function subagentRoutes(app: FastifyInstance) {
  registerSubagentRoutes(app, {
    dispatcher: () => app.runtime?.subagentDispatcher ?? null,
  })
}
