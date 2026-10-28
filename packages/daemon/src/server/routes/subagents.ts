import type { FastifyInstance, FastifyReply } from 'fastify'
import type { SubagentDispatcher } from '../../agent/subagent-dispatcher.js'
import type { BackgroundSubagents } from '../../jobs/subagent.js'
import { resolveAutonomy } from '../runtime/providers.js'
import { listSubagentDelegationCategories } from '../../agent/subagent-categories.js'
import { admitHttpSubagent, backgroundAdmissionError, prepareSubagentRequest, type SubagentRequestAuthority } from '../runtime/subagent-request.js'
import { subagentDispatchRequestSchema } from './subagents-schema.js'
import '../fastify-types.js'

export interface SubagentRoutesDeps extends SubagentRequestAuthority {
  dispatcher: () => SubagentDispatcher | null
  backgroundSubagents?: () => BackgroundSubagents | null | undefined

}

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

    let inherited: Awaited<ReturnType<typeof prepareSubagentRequest>>
    try { inherited = await prepareSubagentRequest(deps, parsed.data) }
    catch (err) { return sendDispatchError(reply, err) }
    if (parsed.data.background === true) {
      try { return reply.status(202).send(admitHttpSubagent(deps.backgroundSubagents?.(), inherited)) }
      catch (err) { return sendDispatchError(reply, err) }
    }

    const controller = new AbortController()
    const abort = () => controller.abort(new Error('subagent client disconnected'))
    request.raw.once('aborted', abort)
    reply.raw.once('close', abort)
    try {
      if (request.raw.aborted || reply.raw.destroyed) abort()
      const result = await dispatcher.dispatch({ ...inherited, signal: controller.signal })
      return reply.status(200).send(result)
    } catch (err) {
      return sendDispatchError(reply, err)
    } finally {
      request.raw.off('aborted', abort)
      reply.raw.off('close', abort)
    }
  })
}

function sendDispatchError(reply: FastifyReply, err: unknown) {
  const admission = backgroundAdmissionError(err)
  if (admission) return reply.status(admission.statusCode).send({ error: admission.error })
  const message = err instanceof Error ? err.message : String(err)
  const code = (err as { code?: string }).code
  for (const known of [
    'SUBAGENT_TOOL_ESCALATION',
    'SUBAGENT_CATEGORY_UNKNOWN',
    'SUBAGENT_AGENT_UNKNOWN',
    'SUBAGENT_FORK_UNAVAILABLE',
    'SUBAGENT_MODEL_UNAVAILABLE',
    'SUBAGENT_NOT_FOUND',
    'SUBAGENT_ISOLATION_UNAVAILABLE',
  ]) {
    if (code === known || message.startsWith(known)) {
      return reply.status(400).send({ error: { code: known, message } })
    }
  }
  if (code === 'SUBAGENT_PROVIDER_UNAVAILABLE' || message.startsWith('SUBAGENT_PROVIDER_UNAVAILABLE:')) {
    return reply.status(503).send({ error: { code: 'SUBAGENT_PROVIDER_UNAVAILABLE', message } })
  }
  return reply.status(500).send({
    error: { code: 'INTERNAL_ERROR', message },
  })
}

export async function subagentRoutes(app: FastifyInstance) {
  registerSubagentRoutes(app, {
    dispatcher: () => app.runtime?.subagentDispatcher ?? null,
    backgroundSubagents: () => app.runtime?.backgroundSubagents,
    parentSelection: async (sessionId) => app.runtime?.sessions.get(sessionId) ?? null,
    callerExecutionPolicy: () => {
      const runtime = app.runtime
      if (!runtime) return null
      return {
        autonomy: resolveAutonomy(runtime.config),
        requireToolApproval: false,
        allowedToolNames: runtime.toolRegistry.list().map((tool) => tool.name),
      }
    },
  })
}
