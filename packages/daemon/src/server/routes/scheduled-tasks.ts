import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify'
import { z } from 'zod'
import {
  MAX_SCHEDULED_AGENT_SKILL_ID_CHARS,
  scheduledAgentSkillRefsFromMetadata,
  withScheduledAgentSkillRefs,
} from '@sepilotd/api-client'
import '../fastify-types.js'
import { SchedulerParseError, type ParsedSchedule } from '../../scheduler/time-parser.js'
import type { JobStatus, JobStore, ScheduledJob } from '../../scheduler/job-store.js'
import { deliveryRouteError } from '../../scheduler/delivery-route.js'
import {
  isSchedulerSurfaceEnabled,
  sendSchedulerSurfaceDisabled,
  type SchedulerSurface,
} from '../../scheduler/surface-access.js'
import {
  describeSchedulerNotificationSubscriptions,
  setSchedulerNotificationSubscribers,
  updateSchedulerNotificationSubscription,
} from '../../scheduler/notification-subscriptions.js'
import { schedulerSessionIdForJob } from '../../scheduler/session-id.js'
import { normalizeSurfaceLabel, resolveRequestSurface } from '../request-surface.js'
import { validateScheduledSkillRefs } from '../../scheduler/skill-selection.js'
import { scriptMonitorConfigFromMetadata } from '../../scheduler/script-monitor.js'
import { createSchedulerDeliveryOutbox } from '../../scheduler/delivery-outbox.js'

function taskExecutionError(instruction: string, metadata: Record<string, unknown> | null): string | null {
  try {
    const monitor = scriptMonitorConfigFromMetadata(metadata)
    if (monitor) {
      if (scheduledAgentSkillRefsFromMetadata(metadata).length > 0) {
        return 'Script monitors execute without an LLM and cannot select agent skills'
      }
      return null
    }
    return instruction.trim() ? null : 'An agent task requires an instruction'
  } catch (error) {
    return error instanceof Error ? error.message : String(error)
  }
}

const scheduledSkillRefsSchema = z.array(z.object({
  name: z.string().min(1).max(MAX_SCHEDULED_AGENT_SKILL_ID_CHARS),
})).max(64)

const createSchema = z.object({
  when: z.string().min(1),
  instruction: z.string(),
  name: z.string().optional(),
  timezone: z.string().optional(),
  maxAttempts: z.number().int().min(1).max(20).optional(),
  retryBackoffMs: z.number().int().min(1000).max(86_400_000).optional(),
  unattended: z.boolean().optional(),
  channelType: z.string().min(1).optional(),
  channelTarget: z.string().min(1).optional(),
  replyToMessageId: z.string().min(1).optional(),
  parentSessionId: z.string().min(1).nullable().optional(),
  skillRefs: scheduledSkillRefsSchema.optional(),
  metadata: z.record(z.unknown()).nullable().optional(),
})

const updateSchema = z
  .object({
    when: z.string().min(1).optional(),
    instruction: z.string().optional(),
    name: z.string().min(1).optional(),
    timezone: z.string().min(1).optional(),
    maxAttempts: z.number().int().min(1).max(20).optional(),
    retryBackoffMs: z.number().int().min(1000).max(86_400_000).optional(),
    enabled: z.boolean().optional(),
    unattended: z.boolean().optional(),
    channelType: z.string().min(1).nullable().optional(),
    channelTarget: z.string().min(1).nullable().optional(),
    replyToMessageId: z.string().min(1).nullable().optional(),
    parentSessionId: z.string().min(1).nullable().optional(),
    skillRefs: scheduledSkillRefsSchema.optional(),
    metadata: z.record(z.unknown()).nullable().optional(),
  })
  .refine((body) => Object.values(body).some((value) => value !== undefined), {
    message: 'At least one field is required',
  })

const listQuerySchema = z.object({
  status: z.enum(['pending', 'all']).optional(),
  channel: z.string().optional(),
})

const runsQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(200).optional(),
})

const notificationSubscribersSchema = z.object({
  subscribers: z.array(z.string().min(1)).nullable(),
})

const notificationSubscriptionSchema = z.object({
  surface: z.string().min(1).optional(),
  subscribed: z.boolean(),
})

const manualRunSchema = z.object({
  suppressDelivery: z.boolean().optional(),
  waitForCompletion: z.boolean().optional(),
}).strict()

type IdParams = { id: string }
type RunParams = IdParams & { runId: string }

type ResolvedJob =
  | { ok: true; job: ScheduledJob }
  | { ok: false; status: 404 | 409; code: 'NOT_FOUND' | 'AMBIGUOUS_ID'; message: string }

function schedulerSurfaceFromRequest(request: FastifyRequest): SchedulerSurface {
  const surface = resolveRequestSurface(request)
  if (surface === 'mobile') return 'mobile'
  if (surface === 'desktop' || surface === 'web') return 'desktop'
  return 'cli'
}

export async function scheduledTasksRoutes(app: FastifyInstance) {
  const requireStore = (reply: FastifyReply) => {
    const store = app.runtime?.jobStore
    if (!store) {
      reply
        .status(503)
        .send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Scheduler not initialized' } })
      return null
    }
    return store
  }
  const requireRequestSurface = (request: FastifyRequest, reply: FastifyReply) => {
    const surface = schedulerSurfaceFromRequest(request)
    if (isSchedulerSurfaceEnabled(app.runtime?.config, surface)) return true
    sendSchedulerSurfaceDisabled(reply, surface)
    return false
  }
  const resolveJob = (store: JobStore, idOrPrefix: string): ResolvedJob => {
    const exact = store.get(idOrPrefix)
    if (exact) return { ok: true, job: exact }
    if (idOrPrefix.length < 4) {
      return { ok: false, status: 404, code: 'NOT_FOUND', message: 'job not found' }
    }
    const matches = store.list().filter((job) => job.id.startsWith(idOrPrefix))
    if (matches.length === 1) return { ok: true, job: matches[0] }
    if (matches.length > 1) {
      return {
        ok: false,
        status: 409,
        code: 'AMBIGUOUS_ID',
        message: `job id prefix is ambiguous: ${idOrPrefix}`,
      }
    }
    return { ok: false, status: 404, code: 'NOT_FOUND', message: 'job not found' }
  }
  const sendResolveError = (reply: FastifyReply, resolved: Extract<ResolvedJob, { ok: false }>) =>
    reply.status(resolved.status).send({
      error: { code: resolved.code, message: resolved.message },
    })

  // GET /api/v1/scheduled-tasks?status=pending|all&channel=<target>
  app.get('/scheduled-tasks', async (request, reply) => {
    if (!requireRequestSurface(request, reply)) return reply
    const store = requireStore(reply)
    if (!store) return reply
    const query = listQuerySchema.parse(request.query)
    const statusFilter: JobStatus[] | undefined = query.status === 'all' ? undefined : ['pending']
    return { data: store.list({ status: statusFilter, channelTarget: query.channel }) }
  })

  // GET /api/v1/scheduled-tasks/delivery-status
  // Aggregate counts only: delivery targets, message bodies, and errors remain private.
  app.get('/scheduled-tasks/delivery-status', async (request, reply) => {
    if (!requireRequestSurface(request, reply)) return reply
    const store = requireStore(reply)
    if (!store) return reply
    return { data: createSchedulerDeliveryOutbox().summary() }
  })

  // POST /api/v1/scheduled-tasks
  app.post('/scheduled-tasks', async (request, reply) => {
    if (!requireRequestSurface(request, reply)) return reply
    const store = requireStore(reply)
    if (!store) return reply
    const parser = app.runtime?.parseWhen
    if (!parser) {
      return reply
        .status(503)
        .send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Scheduler not initialized' } })
    }

    let body: z.infer<typeof createSchema>
    try {
      body = createSchema.parse(request.body)
    } catch {
      return reply
        .status(400)
        .send({ error: { code: 'INVALID_BODY', message: 'Invalid request body' } })
    }

    const routeError = deliveryRouteError(body)
    if (routeError) {
      return reply
        .status(400)
        .send({ error: { code: 'INVALID_DELIVERY_ROUTE', message: routeError } })
    }

    let metadata: Record<string, unknown> | null
    try {
      metadata = body.skillRefs === undefined
        ? (body.metadata ?? null)
        : withScheduledAgentSkillRefs(body.metadata, body.skillRefs)
      const refs = scheduledAgentSkillRefsFromMetadata(metadata)
      await validateScheduledSkillRefs(refs, {
        skillRegistry: app.runtime?.skillRegistry,
        toolRegistry: app.runtime!.toolRegistry,
        autonomy: app.runtime!.autonomy,
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      return reply.status(400).send({
        error: { code: 'INVALID_SKILL_REFS', message },
      })
    }

    const executionError = taskExecutionError(body.instruction, metadata)
    if (executionError) return reply.status(400).send({ error: { code: 'INVALID_TASK_EXECUTION', message: executionError } })

    const timezone = body.timezone ?? app.runtime?.schedulerDefaultTimezone

    let parsed
    try {
      parsed = parser(body.when, { timezone })
    } catch (err) {
      const message = err instanceof SchedulerParseError ? err.message : (err as Error).message
      return reply.status(400).send({ error: { code: 'PARSE_FAILED', message } })
    }

    const job = store.create({
      name: body.name || body.instruction.slice(0, 60) || scriptMonitorConfigFromMetadata(metadata)!.monitorId,
      kind: parsed.kind,
      cron: parsed.kind === 'recurring' ? parsed.cron : null,
      runAt: parsed.kind === 'oneshot' ? parsed.runAt : null,
      nextRunAt: parsed.kind === 'oneshot' ? parsed.runAt : parsed.nextRunAt,
      timezone: parsed.kind === 'recurring' ? (timezone ?? null) : null,
      instruction: body.instruction,
      channelType: body.channelType ?? null,
      channelTarget: body.channelTarget ?? null,
      replyToMessageId: body.replyToMessageId ?? null,
      parentSessionId: body.parentSessionId ?? null,
      enabled: true,
      createdBy: 'rest',
      maxAttempts: body.maxAttempts,
      retryBackoffMs: body.retryBackoffMs,
      unattended: body.unattended,
      metadata,
    })
    if (body.unattended) {
      app.runtime?.approvalRegistry?.grantSessionAutoApproval(
        schedulerSessionIdForJob(job.id),
        'unattended scheduled job',
      )
    }
    return { data: job }
  })

  // PATCH /api/v1/scheduled-tasks/:id
  app.patch<{ Params: IdParams }>('/scheduled-tasks/:id', async (request, reply) => {
    if (!requireRequestSurface(request, reply)) return reply
    const store = requireStore(reply)
    if (!store) return reply

    let body: z.infer<typeof updateSchema>
    try {
      body = updateSchema.parse(request.body)
    } catch {
      return reply
        .status(400)
        .send({ error: { code: 'INVALID_BODY', message: 'Invalid request body' } })
    }

    const { id } = request.params
    const resolved = resolveJob(store, id)
    if (!resolved.ok) return sendResolveError(reply, resolved)
    const existing = resolved.job

    let parsed: ParsedSchedule | undefined
    if (body.when !== undefined) {
      const parser = app.runtime?.parseWhen
      if (!parser) {
        return reply
          .status(503)
          .send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Scheduler not initialized' } })
      }

      try {
        parsed = parser(body.when, {
          timezone: body.timezone ?? existing.timezone ?? app.runtime?.schedulerDefaultTimezone,
        })
      } catch (err) {
        const message = err instanceof SchedulerParseError ? err.message : (err as Error).message
        return reply.status(400).send({ error: { code: 'PARSE_FAILED', message } })
      }

      if (parsed.kind !== existing.kind) {
        return reply.status(400).send({
          error: {
            code: 'INVALID_KIND_CHANGE',
            message: `Cannot change scheduled task kind from ${existing.kind} to ${parsed.kind}`,
          },
        })
      }
    }

    const name = body.name ?? existing.name
    const instruction = body.instruction ?? existing.instruction
    const enabled = body.enabled ?? existing.enabled
    const maxAttempts = body.maxAttempts ?? existing.maxAttempts
    const retryBackoffMs = body.retryBackoffMs ?? existing.retryBackoffMs
    const channelType = body.channelType === undefined ? existing.channelType : body.channelType
    const channelTarget = body.channelTarget === undefined
      ? existing.channelTarget
      : body.channelTarget
    const replyToMessageId = body.replyToMessageId === undefined
      ? existing.replyToMessageId
      : body.replyToMessageId
    const routeError = deliveryRouteError({ channelType, channelTarget, replyToMessageId })
    if (routeError) {
      return reply
        .status(400)
        .send({ error: { code: 'INVALID_DELIVERY_ROUTE', message: routeError } })
    }
    const parentSessionId =
      body.parentSessionId === undefined ? existing.parentSessionId : body.parentSessionId
    let metadata: Record<string, unknown> | null
    try {
      const metadataBase = body.metadata === undefined ? existing.metadata : body.metadata
      metadata = body.skillRefs === undefined
        ? metadataBase
        : withScheduledAgentSkillRefs(metadataBase, body.skillRefs)
      const refs = scheduledAgentSkillRefsFromMetadata(metadata)
      await validateScheduledSkillRefs(refs, {
        skillRegistry: app.runtime?.skillRegistry,
        toolRegistry: app.runtime!.toolRegistry,
        autonomy: app.runtime!.autonomy,
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      return reply.status(400).send({
        error: { code: 'INVALID_SKILL_REFS', message },
      })
    }
    const executionError = taskExecutionError(instruction, metadata)
    if (executionError) return reply.status(400).send({ error: { code: 'INVALID_TASK_EXECUTION', message: executionError } })

    // Display, delivery, authority, and retry-policy edits must not erase the
    // current execution evidence or silently re-arm a terminal task. Only a
    // replacement schedule/timezone/instruction changes what will execute.
    const preserveRunState = body.when === undefined
      && body.timezone === undefined
      && body.instruction === undefined
    const updated =
      existing.kind === 'recurring'
        ? store.updateRecurringJob({
            id: existing.id,
            name,
            cron: parsed?.kind === 'recurring' ? parsed.cron : existing.cron!,
            nextRunAt: parsed?.kind === 'recurring' ? parsed.nextRunAt : existing.nextRunAt,
            timezone: body.timezone ?? existing.timezone ?? null,
            instruction,
            channelType,
            channelTarget,
            replyToMessageId,
            parentSessionId,
            enabled,
            maxAttempts,
            retryBackoffMs,
            metadata,
            preserveRunState,
          })
        : store.updateOneShotJob({
            id: existing.id,
            name,
            runAt:
              parsed?.kind === 'oneshot' ? parsed.runAt : (existing.runAt ?? existing.nextRunAt),
            nextRunAt: parsed?.kind === 'oneshot' ? parsed.runAt : existing.nextRunAt,
            instruction,
            channelType,
            channelTarget,
            replyToMessageId,
            parentSessionId,
            enabled,
            maxAttempts,
            retryBackoffMs,
            metadata,
            preserveRunState,
          })

    if (!updated)
      return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'job not found' } })
    if (body.unattended !== undefined) {
      store.setUnattended(existing.id, body.unattended)
      const sessionId = schedulerSessionIdForJob(existing.id)
      if (body.unattended) {
        app.runtime?.approvalRegistry?.grantSessionAutoApproval(
          sessionId,
          'unattended scheduled job',
        )
      } else {
        app.runtime?.approvalRegistry?.revokeSessionAutoApproval(sessionId)
      }
    }
    return { data: store.get(existing.id) ?? updated }
  })

  // GET /api/v1/scheduled-tasks/:id
  app.get<{ Params: IdParams }>('/scheduled-tasks/:id', async (request, reply) => {
    if (!requireRequestSurface(request, reply)) return reply
    const store = requireStore(reply)
    if (!store) return reply
    const { id } = request.params
    const resolved = resolveJob(store, id)
    if (!resolved.ok) return sendResolveError(reply, resolved)
    return { data: resolved.job }
  })

  // GET /api/v1/scheduled-tasks/:id/notifications
  app.get<{ Params: IdParams }>('/scheduled-tasks/:id/notifications', async (request, reply) => {
    if (!requireRequestSurface(request, reply)) return reply
    const store = requireStore(reply)
    if (!store) return reply
    const { id } = request.params
    const resolved = resolveJob(store, id)
    if (!resolved.ok) return sendResolveError(reply, resolved)
    return { data: describeSchedulerNotificationSubscriptions(resolved.job) }
  })

  // PUT /api/v1/scheduled-tasks/:id/notifications
  app.put<{ Params: IdParams }>('/scheduled-tasks/:id/notifications', async (request, reply) => {
    if (!requireRequestSurface(request, reply)) return reply
    const store = requireStore(reply)
    if (!store) return reply
    const parsed = notificationSubscribersSchema.safeParse(request.body)
    if (!parsed.success) {
      return reply.status(400).send({
        error: { code: 'INVALID_BODY', message: 'Invalid request body' },
      })
    }
    const { id } = request.params
    const resolved = resolveJob(store, id)
    if (!resolved.ok) return sendResolveError(reply, resolved)
    const updated = store.updateMetadata(
      resolved.job.id,
      setSchedulerNotificationSubscribers(resolved.job.metadata, parsed.data.subscribers),
    )
    if (!updated) {
      return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'job not found' } })
    }
    return { data: describeSchedulerNotificationSubscriptions(updated) }
  })

  // POST /api/v1/scheduled-tasks/:id/notifications/subscription
  app.post<{ Params: IdParams }>(
    '/scheduled-tasks/:id/notifications/subscription',
    async (request, reply) => {
      if (!requireRequestSurface(request, reply)) return reply
      const store = requireStore(reply)
      if (!store) return reply
      const parsed = notificationSubscriptionSchema.safeParse(request.body ?? {})
      if (!parsed.success) {
        return reply.status(400).send({
          error: { code: 'INVALID_BODY', message: 'Invalid request body' },
        })
      }
      const surface = normalizeSurfaceLabel(parsed.data.surface) ?? resolveRequestSurface(request)
      if (!surface) {
        return reply.status(400).send({
          error: { code: 'INVALID_BODY', message: 'surface is required' },
        })
      }
      const { id } = request.params
      const resolved = resolveJob(store, id)
      if (!resolved.ok) return sendResolveError(reply, resolved)
      const updated = store.updateMetadata(
        resolved.job.id,
        updateSchedulerNotificationSubscription(
          resolved.job.metadata,
          surface,
          parsed.data.subscribed,
        ),
      )
      if (!updated) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'job not found' } })
      }
      return { data: describeSchedulerNotificationSubscriptions(updated) }
    },
  )

  // GET /api/v1/scheduled-tasks/:id/runs?limit=N
  app.get<{ Params: IdParams }>('/scheduled-tasks/:id/runs', async (request, reply) => {
    if (!requireRequestSurface(request, reply)) return reply
    const store = requireStore(reply)
    if (!store) return reply
    const { id } = request.params
    const resolved = resolveJob(store, id)
    if (!resolved.ok) return sendResolveError(reply, resolved)
    const { limit } = runsQuerySchema.parse(request.query ?? {})
    return { data: store.listRuns(resolved.job.id, limit ?? 20) }
  })

  // GET /api/v1/scheduled-tasks/:id/runs/:runId
  app.get<{ Params: RunParams }>('/scheduled-tasks/:id/runs/:runId', async (request, reply) => {
    if (!requireRequestSurface(request, reply)) return reply
    const store = requireStore(reply)
    if (!store) return reply
    const { id, runId } = request.params
    const resolved = resolveJob(store, id)
    if (!resolved.ok) return sendResolveError(reply, resolved)
    const run = store.getRun(resolved.job.id, runId)
    if (!run) {
      return reply.status(404).send({
        error: { code: 'NOT_FOUND', message: 'run not found for job' },
      })
    }
    return { data: run }
  })

  // DELETE /api/v1/scheduled-tasks/:id  (soft-cancel — keeps the record + history)
  app.delete<{ Params: IdParams }>('/scheduled-tasks/:id', async (request, reply) => {
    if (!requireRequestSurface(request, reply)) return reply
    const store = requireStore(reply)
    if (!store) return reply
    const { id } = request.params
    const resolved = resolveJob(store, id)
    if (!resolved.ok) return sendResolveError(reply, resolved)
    store.cancel(resolved.job.id)
    return reply.status(204).send()
  })

  // POST /api/v1/scheduled-tasks/:id/pause
  app.post<{ Params: IdParams }>('/scheduled-tasks/:id/pause', async (request, reply) => {
    if (!requireRequestSurface(request, reply)) return reply
    const store = requireStore(reply)
    if (!store) return reply
    const { id } = request.params
    const resolved = resolveJob(store, id)
    if (!resolved.ok) return sendResolveError(reply, resolved)
    store.setEnabled(resolved.job.id, false)
    return { data: store.get(resolved.job.id) }
  })

  // POST /api/v1/scheduled-tasks/:id/resume
  app.post<{ Params: IdParams }>('/scheduled-tasks/:id/resume', async (request, reply) => {
    if (!requireRequestSurface(request, reply)) return reply
    const store = requireStore(reply)
    if (!store) return reply
    const { id } = request.params
    const resolved = resolveJob(store, id)
    if (!resolved.ok) return sendResolveError(reply, resolved)
    store.setEnabled(resolved.job.id, true)
    return { data: store.get(resolved.job.id) }
  })

  // POST /api/v1/scheduled-tasks/:id/run  (manual fire — recorded in run history)
  app.post<{ Params: IdParams }>('/scheduled-tasks/:id/run', async (request, reply) => {
    if (!requireRequestSurface(request, reply)) return reply
    const store = requireStore(reply)
    if (!store) return reply
    const trigger = app.runtime?.triggerSchedulerJob
    if (!trigger) {
      return reply
        .status(503)
        .send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Scheduler not initialized' } })
    }
    const { id } = request.params
    const resolved = resolveJob(store, id)
    if (!resolved.ok) return sendResolveError(reply, resolved)
    const parsed = manualRunSchema.safeParse(request.body ?? {})
    if (!parsed.success) {
      return reply
        .status(400)
        .send({ error: { code: 'INVALID_BODY', message: 'Invalid request body' } })
    }
    const result = await trigger(resolved.job.id, parsed.data)
    if (!result.started) {
      // Reporting ok for a run that never started makes the surface's
      // "Run now" look like it worked.
      return reply.status(409).send({
        error: {
          code: 'CONFLICT',
          message: `Job is not runnable (enabled: ${store.get(resolved.job.id)?.enabled === true}, status: ${store.get(resolved.job.id)?.status ?? 'unknown'}); manual run was not started`,
        },
      })
    }
    return { data: result }
  })
}
