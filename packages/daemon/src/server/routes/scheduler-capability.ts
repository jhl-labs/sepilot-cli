import type { FastifyInstance } from 'fastify'
import { randomUUID } from 'node:crypto'
import { z } from 'zod'
import {
  MAX_SCHEDULED_AGENT_SKILL_ID_CHARS,
  scheduledAgentSkillRefsFromMetadata,
  withScheduledAgentSkillRefs,
} from '@sepilotd/api-client'
import { bindCapability } from '../capabilities/bind.js'
import {
  buildSseResponseHeaders,
  registerSseDisconnectHandler,
  trackSseConnection,
} from '../sse-response.js'
import type { JobKind, ScheduledJob } from '../../scheduler/job-store.js'
import { nextRecurringRun, SchedulerParseError } from '../../scheduler/time-parser.js'
import { schedulerSessionIdForJob } from '../../scheduler/session-id.js'
import {
  isSchedulerSurfaceEnabled,
  sendSchedulerCapabilityDisabled,
} from '../../scheduler/surface-access.js'
import {
  describeSchedulerNotificationSubscriptions,
  setSchedulerNotificationSubscribers,
  updateSchedulerNotificationSubscription,
} from '../../scheduler/notification-subscriptions.js'
import { normalizeSurfaceLabel, resolveRequestSurface } from '../request-surface.js'
import '../fastify-types.js'
import { validateScheduledSkillRefs } from '../../scheduler/skill-selection.js'
import { scriptMonitorConfigFromMetadata } from '../../scheduler/script-monitor.js'

type SchedulerWatchPayload =
  | { type: 'snapshot'; jobs: ScheduledJob[] }
  | { type: 'heartbeat'; timestamp: string }

interface SchedulerJobRequest {
  id?: string
  name: string
  cron?: string
  when?: string
  runAt?: number
  instruction?: string
  timezone?: string
  nextRunAt?: number
  enabled?: boolean
  maxAttempts?: number
  retryBackoffMs?: number
  parentSessionId?: string | null
  skillRefs?: Array<{ name: string }>
  metadata?: Record<string, unknown> | null
}

const schedulerNotificationSubscribersSchema = z.object({
  subscribers: z.array(z.string().min(1)).nullable(),
})

const schedulerNotificationSubscriptionSchema = z.object({
  surface: z.string().min(1).optional(),
  subscribed: z.boolean(),
})

const schedulerManualRunSchema = z.object({
  suppressDelivery: z.boolean().optional(),
  waitForCompletion: z.boolean().optional(),
}).strict()

interface ResolvedSchedulerInput {
  kind: JobKind
  cron: string | null
  runAt: number | null
  nextRunAt: number
  timezone: string | null
}

export async function registerSchedulerCapabilityRoutes(
  app: FastifyInstance,
): Promise<void> {
  const watchSubscribers = new Set<(payload: SchedulerWatchPayload) => void>()

  function getStore() {
    return app.runtime?.jobStore
  }

  function buildWatchSnapshot(): SchedulerWatchPayload {
    const store = getStore()
    return {
      type: 'snapshot',
      jobs: store ? store.list() : [],
    }
  }

  function publishWatchSnapshot(): void {
    if (watchSubscribers.size === 0) return
    const payload = buildWatchSnapshot()
    for (const subscriber of watchSubscribers) subscriber(payload)
  }

  function watchFingerprint(): string {
    const store = getStore()
    if (!store) return '[]'
    return JSON.stringify(
      store.list().map((job) => [
        job.id,
        job.status,
        job.enabled,
        job.nextRunAt,
        job.lastRunAt,
        job.updatedAt,
        job.lastError,
        job.metadata,
      ]),
    )
  }

  await bindCapability(
    app,
    {
      name: 'scheduler',
      version: '1',
      methods: [
        { method: 'GET', path: '/scheduler/jobs' },
        { method: 'GET', path: '/scheduler/jobs/:id' },
        { method: 'GET', path: '/scheduler/jobs/watch' },
        { method: 'GET', path: '/scheduler/jobs/:id/runs' },
        { method: 'GET', path: '/scheduler/jobs/:id/runs/:runId' },
        { method: 'GET', path: '/scheduler/jobs/:id/notifications' },
        { method: 'PUT', path: '/scheduler/jobs/:id/notifications' },
        { method: 'POST', path: '/scheduler/jobs/:id/notifications/subscription' },
        { method: 'POST', path: '/scheduler/jobs' },
        { method: 'POST', path: '/scheduler/jobs/:id/run' },
        { method: 'POST', path: '/scheduler/jobs/:id/pause' },
        { method: 'POST', path: '/scheduler/jobs/:id/resume' },
        { method: 'POST', path: '/scheduler/jobs/:id/unattended' },
        { method: 'DELETE', path: '/scheduler/jobs/:id' },
      ],
    },
    async (a) => {
      const notReady = (reply: import('fastify').FastifyReply) =>
        reply.status(503).send({ code: 'SERVICE_UNAVAILABLE', message: 'Scheduler not ready', retriable: true })
      const notFound = (reply: import('fastify').FastifyReply) =>
        reply.status(404).send({ code: 'NOT_FOUND', message: 'job not found', retriable: false })
      const runNotFound = (reply: import('fastify').FastifyReply) =>
        reply.status(404).send({ code: 'NOT_FOUND', message: 'run not found for job', retriable: false })
      const requireDesktopSurface = (reply: import('fastify').FastifyReply) => {
        if (isSchedulerSurfaceEnabled(app.runtime?.config, 'desktop')) return true
        sendSchedulerCapabilityDisabled(reply, 'desktop')
        return false
      }

      a.get('/scheduler/jobs', async (_req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        const store = getStore()
        if (!store) return notReady(reply)
        return store.list()
      })
      a.get('/scheduler/jobs/:id', async (req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        const store = getStore()
        if (!store) return notReady(reply)
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        return store.get(id) ?? notFound(reply)
      })
      a.get('/scheduler/jobs/:id/runs', async (req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        const store = getStore()
        if (!store) return notReady(reply)
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        if (!store.get(id)) return notFound(reply)
        const { limit } = z.object({ limit: z.coerce.number().int().min(1).max(200).optional() }).parse(req.query ?? {})
        return store.listRuns(id, limit ?? 20)
      })
      a.get('/scheduler/jobs/:id/runs/:runId', async (req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        const store = getStore()
        if (!store) return notReady(reply)
        const { id, runId } = z.object({ id: z.string().min(1), runId: z.string().min(1) }).parse(req.params)
        if (!store.get(id)) return notFound(reply)
        return store.getRun(id, runId) ?? runNotFound(reply)
      })
      a.get('/scheduler/jobs/:id/notifications', async (req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        const store = getStore()
        if (!store) return notReady(reply)
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        const job = store.get(id)
        if (!job) return notFound(reply)
        return describeSchedulerNotificationSubscriptions(job)
      })
      a.put('/scheduler/jobs/:id/notifications', async (req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        const store = getStore()
        if (!store) return notReady(reply)
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        const parsed = schedulerNotificationSubscribersSchema.safeParse(req.body)
        if (!parsed.success) {
          void reply.status(400).send({ code: 'INVALID_REQUEST', message: parsed.error.message, retriable: false })
          return reply
        }
        const job = store.get(id)
        if (!job) return notFound(reply)
        const updated = store.updateMetadata(
          job.id,
          setSchedulerNotificationSubscribers(job.metadata, parsed.data.subscribers),
        )
        if (!updated) return notFound(reply)
        publishWatchSnapshot()
        return describeSchedulerNotificationSubscriptions(updated)
      })
      a.post('/scheduler/jobs/:id/notifications/subscription', async (req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        const store = getStore()
        if (!store) return notReady(reply)
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        const parsed = schedulerNotificationSubscriptionSchema.safeParse(req.body ?? {})
        if (!parsed.success) {
          void reply.status(400).send({ code: 'INVALID_REQUEST', message: parsed.error.message, retriable: false })
          return reply
        }
        const surface = normalizeSurfaceLabel(parsed.data.surface) ?? resolveRequestSurface(req)
        if (!surface) {
          void reply.status(400).send({ code: 'INVALID_REQUEST', message: 'surface is required', retriable: false })
          return reply
        }
        const job = store.get(id)
        if (!job) return notFound(reply)
        const updated = store.updateMetadata(
          job.id,
          updateSchedulerNotificationSubscription(
            job.metadata,
            surface,
            parsed.data.subscribed,
          ),
        )
        if (!updated) return notFound(reply)
        publishWatchSnapshot()
        return describeSchedulerNotificationSubscriptions(updated)
      })
      a.get('/scheduler/jobs/watch', async (req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        reply.hijack()
        reply.raw.writeHead(200, buildSseResponseHeaders(req, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
          'X-Request-ID': req.requestId ?? randomUUID(),
        }))

        let closed = false
        const send = (payload: SchedulerWatchPayload) => {
          if (closed) return
          reply.raw.write(
            `event: scheduler\ndata: ${JSON.stringify(payload)}\n\n`,
          )
        }
        const heartbeat = setInterval(() => {
          send({ type: 'heartbeat', timestamp: new Date().toISOString() })
        }, 15_000)
        heartbeat.unref?.()
        let lastFingerprint = watchFingerprint()
        const snapshotPoll = setInterval(() => {
          const nextFingerprint = watchFingerprint()
          if (nextFingerprint === lastFingerprint) return
          lastFingerprint = nextFingerprint
          send(buildWatchSnapshot())
        }, 1_000)
        snapshotPoll.unref?.()
        const close = () => {
          if (closed) return
          closed = true
          watchSubscribers.delete(send)
          clearInterval(heartbeat)
          clearInterval(snapshotPoll)
          if (!reply.raw.destroyed && !reply.raw.writableEnded) {
            reply.raw.end()
          }
        }

        watchSubscribers.add(send)
        registerSseDisconnectHandler(req, reply, close)
        trackSseConnection(app, req, reply, { label: 'scheduler-watch' })
        send(buildWatchSnapshot())
      })
      a.post('/scheduler/jobs', async (req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        const store = getStore()
        if (!store) return notReady(reply)
        const parsed = z
          .object({
            id: z.string().optional(),
            name: z.string().min(1),
            cron: z.string().min(1).optional(),
            when: z.string().min(1).optional(),
            runAt: z.number().int().positive().optional(),
            instruction: z.string().optional(),
            timezone: z.string().optional(),
            nextRunAt: z.number().int().optional(),
            enabled: z.boolean().optional(),
            maxAttempts: z.number().int().min(1).max(20).optional(),
            retryBackoffMs: z.number().int().min(1000).max(86_400_000).optional(),
            parentSessionId: z.string().min(1).nullable().optional(),
            skillRefs: z.array(z.object({
              name: z.string().min(1).max(MAX_SCHEDULED_AGENT_SKILL_ID_CHARS),
            })).max(64).optional(),
            metadata: z.record(z.unknown()).nullable().optional(),
          })
          .refine((value) => [value.cron, value.when, value.runAt].filter((v) => v !== undefined).length === 1, {
            message: 'exactly one of cron, when, or runAt is required',
          })
          .safeParse(req.body)
        if (!parsed.success) {
          void reply.status(400).send({ code: 'INVALID_REQUEST', message: parsed.error.message, retriable: false })
          return reply
        }
        const {
          id,
          name,
          instruction,
          enabled,
          maxAttempts,
          retryBackoffMs,
          parentSessionId,
          metadata,
          skillRefs,
        } = parsed.data
        const existing = id ? store.get(id) : null
        let resolvedMetadata: Record<string, unknown> | null
        try {
          const metadataBase = metadata === undefined ? (existing?.metadata ?? null) : metadata
          resolvedMetadata = skillRefs === undefined
            ? metadataBase
            : withScheduledAgentSkillRefs(metadataBase, skillRefs)
          const refs = scheduledAgentSkillRefsFromMetadata(resolvedMetadata)
          const scriptMonitor = scriptMonitorConfigFromMetadata(resolvedMetadata)
          if (scriptMonitor && refs.length > 0) {
            throw new Error('script monitors execute without an LLM and cannot select agent skills')
          }
          await validateScheduledSkillRefs(refs, {
            skillRegistry: app.runtime?.skillRegistry,
            toolRegistry: app.runtime!.toolRegistry,
            autonomy: app.runtime!.autonomy,
          })
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err)
          const requestedMetadata = metadata === undefined ? existing?.metadata : metadata
          const hasScriptMonitor = Boolean(
            requestedMetadata
            && Object.prototype.hasOwnProperty.call(requestedMetadata, 'scriptMonitor'),
          )
          void reply.status(400).send({
            code: hasScriptMonitor ? 'INVALID_SCRIPT_MONITOR' : 'INVALID_SKILL_REFS',
            message,
            retriable: false,
          })
          return reply
        }
        let resolved: ResolvedSchedulerInput
        try {
          resolved = resolveSchedulerInput(parsed.data, existing)
        } catch (err) {
          const message = err instanceof SchedulerParseError
            ? err.message
            : err instanceof Error
              ? err.message
              : String(err)
          void reply.status(400).send({ code: 'INVALID_REQUEST', message, retriable: false })
          return reply
        }
        if (existing && existing.kind !== resolved.kind) {
          void reply.status(400).send({ code: 'INVALID_REQUEST', message: 'changing scheduler job kind is not supported; create a new job instead', retriable: false })
          return reply
        }
        const job = existing
          ? resolved.kind === 'recurring'
            ? store.updateRecurringJob({
                id: existing.id,
                name,
                cron: resolved.cron ?? '',
                nextRunAt: resolved.nextRunAt,
                timezone: resolved.timezone,
                instruction: instruction ?? existing.instruction,
                parentSessionId,
                enabled: enabled ?? existing.enabled,
                maxAttempts,
                retryBackoffMs,
                metadata: metadata !== undefined || skillRefs !== undefined
                  ? resolvedMetadata
                  : undefined,
              })
            : store.updateOneShotJob({
                id: existing.id,
                name,
                runAt: resolved.runAt ?? resolved.nextRunAt,
                nextRunAt: resolved.nextRunAt,
                instruction: instruction ?? existing.instruction,
                parentSessionId,
                enabled: enabled ?? existing.enabled,
                maxAttempts,
                retryBackoffMs,
                metadata: metadata !== undefined || skillRefs !== undefined
                  ? resolvedMetadata
                  : undefined,
              })
          : store.create({
              id,
              name,
              kind: resolved.kind,
              cron: resolved.cron,
              runAt: resolved.runAt,
              nextRunAt: resolved.nextRunAt,
              timezone: resolved.timezone,
              instruction: instruction ?? '',
              channelType: null,
              channelTarget: null,
              replyToMessageId: null,
              parentSessionId: parentSessionId ?? null,
              enabled: enabled !== false,
              createdBy: 'rest',
              maxAttempts,
              retryBackoffMs,
              metadata: resolvedMetadata,
            })
        if (!job) return notFound(reply)
        publishWatchSnapshot()
        return job
      })
      a.post('/scheduler/jobs/:id/run', async (req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        const store = getStore()
        if (!store) return notReady(reply)
        const trigger = app.runtime?.triggerSchedulerJob
        if (!trigger) return notReady(reply)
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        if (!store.get(id)) return notFound(reply)
        const parsed = schedulerManualRunSchema.safeParse(req.body ?? {})
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        const result = await trigger(id, parsed.data)
        publishWatchSnapshot()
        if (!result.started) {
          // Reporting ok for a run that never started makes the surface's
          // "Run now" look like it worked.
          void reply.status(409).send({
            code: 'CONFLICT',
            message: `job is not runnable (enabled: ${store.get(id)?.enabled === true}, status: ${store.get(id)?.status ?? 'unknown'}); manual run was not started`,
            retriable: true,
          })
          return reply
        }
        return result
      })
      a.post('/scheduler/jobs/:id/pause', async (req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        const store = getStore()
        if (!store) return notReady(reply)
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        if (!store.get(id)) return notFound(reply)
        store.setEnabled(id, false)
        publishWatchSnapshot()
        return store.get(id)
      })
      a.post('/scheduler/jobs/:id/resume', async (req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        const store = getStore()
        if (!store) return notReady(reply)
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        if (!store.get(id)) return notFound(reply)
        store.setEnabled(id, true)
        publishWatchSnapshot()
        return store.get(id)
      })
      // A job that fires while nobody is at the screen can't answer an approval
      // prompt — the prompt just times out and the run accomplishes nothing.
      // Turning this on grants that job's own session a standing approval; the
      // flag is persisted and the executor replays the grant on every fire.
      a.post('/scheduler/jobs/:id/unattended', async (req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        const store = getStore()
        if (!store) return notReady(reply)
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        const { unattended } = z.object({ unattended: z.boolean() }).parse(req.body ?? {})
        if (!store.get(id)) return notFound(reply)
        store.setUnattended(id, unattended)
        const sessionId = schedulerSessionIdForJob(id)
        if (unattended) app.runtime?.approvalRegistry?.grantSessionAutoApproval(sessionId, 'unattended scheduled job')
        else app.runtime?.approvalRegistry?.revokeSessionAutoApproval(sessionId)
        publishWatchSnapshot()
        return store.get(id)
      })
      a.delete('/scheduler/jobs/:id', async (req, reply) => {
        if (!requireDesktopSurface(reply)) return reply
        const store = getStore()
        if (!store) return notReady(reply)
        const { id } = z.object({ id: z.string().min(1) }).parse(req.params)
        if (!store.get(id)) return notFound(reply)
        store.delete(id)
        publishWatchSnapshot()
        return { ok: true }
      })
    },
  )

  function resolveSchedulerInput(
    input: SchedulerJobRequest,
    existing: ScheduledJob | null,
  ): ResolvedSchedulerInput {
    const tz = input.timezone ?? existing?.timezone ?? app.runtime?.schedulerDefaultTimezone
    if (input.cron) {
      const nextRunAt = input.nextRunAt ?? nextRecurringRun(input.cron, Date.now(), tz)
      return {
        kind: 'recurring',
        cron: input.cron,
        runAt: null,
        nextRunAt,
        timezone: tz ?? null,
      }
    }

    if (input.runAt !== undefined) {
      if (!Number.isFinite(input.runAt) || input.runAt <= Date.now()) {
        throw new SchedulerParseError('runAt must be a future epoch millisecond timestamp')
      }
      return {
        kind: 'oneshot',
        cron: null,
        runAt: input.runAt,
        nextRunAt: input.runAt,
        timezone: null,
      }
    }

    const parser = app.runtime?.parseWhen
    if (!parser) {
      throw new Error('Scheduler parser not ready')
    }
    const parsed = parser(input.when ?? '', { timezone: tz })
    if (parsed.kind === 'recurring') {
      return {
        kind: 'recurring',
        cron: parsed.cron,
        runAt: null,
        nextRunAt: parsed.nextRunAt,
        timezone: tz ?? null,
      }
    }
    return {
      kind: 'oneshot',
      cron: null,
      runAt: parsed.runAt,
      nextRunAt: parsed.runAt,
      timezone: null,
    }
  }
}
