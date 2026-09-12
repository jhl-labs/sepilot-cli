// packages/daemon/src/tools/scheduler.ts
import { createHash } from 'node:crypto'
import type { ManualJobRunOptions, ManualJobRunResult } from '@sepilotd/core'
import {
  normalizeScheduledAgentSkillRefs,
  scheduledAgentSkillRefsFromMetadata,
  withScheduledAgentSkillRefs,
  type ScheduledAgentSkillRef,
  normalizeScheduledAppSources,
  scheduledAppSourcesFromMetadata,
  scheduledAppSourceMatches,
  withScheduledAppSources,
  SCHEDULED_APP_SOURCE_SCHEMA,
  normalizeSchedulerMisfirePolicy,
  schedulerMisfirePolicyFromMetadata,
  schedulerMissedEvidence,
} from '@sepilotd/api-client'
import { redactSensitive } from '../memory/sensitive.js'
import { deliveryRouteError, type SchedulerDeliveryRoute } from '../scheduler/delivery-route.js'
import { parseWhen, formatWhen, SchedulerParseError } from '../scheduler/time-parser.js'
import type { JobStore, JobRun, ScheduledJob } from '../scheduler/job-store.js'
import { projectStoredJobRunOutcome } from '../scheduler/agent-outcome.js'
import { schedulerSessionIdForJob } from '../scheduler/session-id.js'
import type { ToolDefinitionRuntime, ToolResult, ToolExecutionContext } from './registry.js'

const success = (output: string, durationMs: number): ToolResult => ({ output, status: 'success', durationMs })
const failure = (output: string, durationMs: number, code?: string): ToolResult => ({
  output, status: 'error', durationMs, code,
})

export interface SchedulerToolDeps {
  store: JobStore
  /** Optional: enables `schedule_run_now` to fire a job immediately (recorded in run history). */
  triggerSchedulerJob?: (id: string, options?: ManualJobRunOptions) => Promise<ManualJobRunResult>
  /** Default IANA timezone applied when `schedule_create` doesn't specify one. */
  defaultTimezone?: string
  /**
   * Pre-authorise the tool calls a job's future runs will make, scoped to that
   * job alone. Wired to ApprovalRegistry.grantSessionAutoApproval. Only invoked
   * when the caller passes `unattended: true`, which the surface must obtain
   * from the user. This authority is independent of whether the scheduler fires
   * the job; without it, only a policy-gated tool call may stall on approval.
   */
  grantUnattendedApproval?: (schedulerSessionId: string) => void
  /** Drop that pre-authorisation when a job is cancelled. */
  revokeUnattendedApproval?: (schedulerSessionId: string) => void
  /** Validate selected skills against the live registry and autonomy ceiling. */
  validateSkillRefs?: (refs: readonly ScheduledAgentSkillRef[]) => Promise<void>
}

export { schedulerSessionIdForJob }

const numOpt = (value: unknown, min: number, max?: number): number | undefined => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined
  const clamped = max != null ? Math.min(value, max) : value
  return Math.max(min, Math.round(clamped))
}

const currentChatKey = (context?: ToolExecutionContext): string | undefined => {
  const chatKey = context?.channelContext?.chatKey?.trim()
  return chatKey || undefined
}

function jobVisibleInContext(job: ScheduledJob, context?: ToolExecutionContext): boolean {
  const chatKey = currentChatKey(context)
  return !chatKey || job.channelTarget === chatKey
}

function resolveScopedJob(store: JobStore, rawId: string, context?: ToolExecutionContext):
  | { job: ScheduledJob }
  | { error: string } {
  const id = rawId.trim()
  if (!id) return { error: 'job id is required' }

  const exact = store.get(id)
  if (exact) {
    return jobVisibleInContext(exact, context)
      ? { job: exact }
      : { error: `job not found in current scope: ${id}` }
  }

  if (id.length < 4) {
    return { error: 'job id prefix must be at least 4 characters' }
  }

  const chatKey = currentChatKey(context)
  const matches = store
    .list(chatKey ? { channelTarget: chatKey } : undefined)
    .filter((job) => job.id.startsWith(id))

  if (matches.length === 0) return { error: `job not found: ${id}` }
  if (matches.length > 1) {
    return { error: `job id prefix is ambiguous: ${matches.slice(0, 5).map((job) => job.id.slice(0, 8)).join(', ')}` }
  }
  return { job: matches[0]! }
}

function serializeJob(job: ScheduledJob): Record<string, unknown> {
  const notificationPriority = job.metadata?.notificationPriority
  let skillRefs: ScheduledAgentSkillRef[] = []
  let skillProfileStatus: 'valid' | 'invalid' = 'valid'
  try {
    skillRefs = scheduledAgentSkillRefsFromMetadata(job.metadata)
  } catch {
    skillProfileStatus = 'invalid'
  }
  let sourceRefs: ReturnType<typeof scheduledAppSourcesFromMetadata> = []
  let misfirePolicy: ReturnType<typeof schedulerMisfirePolicyFromMetadata> = null
  let assistantContractStatus = 'valid'
  try {
    sourceRefs = scheduledAppSourcesFromMetadata(job.metadata)
    misfirePolicy = schedulerMisfirePolicyFromMetadata(job.metadata)
  } catch { assistantContractStatus = 'invalid' }
  return {
    id: job.id,
    name: safeSchedulerEvidenceText(job.name),
    instruction: safeSchedulerEvidenceText(job.instruction),
    kind: job.kind,
    parentSessionId: job.parentSessionId,
    when_human: formatWhen(job),
    timezone: job.timezone,
    status: job.status,
    enabled: job.enabled,
    nextRunAt: job.nextRunAt,
    lastRunAt: job.lastRunAt,
    attempt: job.attempt,
    maxAttempts: job.maxAttempts,
    retryBackoffMs: job.retryBackoffMs,
    lastError: job.lastError ? safeSchedulerEvidenceText(job.lastError) : null,
    channelType: job.channelType,
    channelTarget: job.channelTarget ? safeSchedulerEvidenceText(job.channelTarget) : null,
    replyToMessageId: job.replyToMessageId ? safeSchedulerEvidenceText(job.replyToMessageId) : null,
    unattended: job.unattended,
    approvalMode: job.unattended === true ? 'job-scoped-standing' : 'interactive-if-required',
    notificationPriority:
      notificationPriority === 'high' || notificationPriority === 'critical'
        ? notificationPriority
        : 'normal',
    skillRefs,
    skillProfileStatus,
    source_refs: sourceRefs,
    missed_policy: misfirePolicy?.policy ?? 'skip',
    max_lateness_ms: misfirePolicy?.policy === 'run_once' ? misfirePolicy.maxLatenessMs : null,
    missed: schedulerMissedEvidence(job),
    assistantContractStatus,
  }
}

const sourceRefsSchema = {
  type: 'array', maxItems: 16, items: SCHEDULED_APP_SOURCE_SCHEMA,
  description: 'App/collection/record identities this task depends on, obtained by reading the real apps. Link reminders to their item and briefings to their source apps. Execution must re-read these sources. Empty array clears links on update; omission preserves them.',
}
const missedPolicySchema = {
  type: 'string', enum: ['skip', 'run_once'],
  description: 'One-shot offline recovery: skip (default) cancels after the normal grace; run_once permits one late execution within max_lateness_ms. Choose run_once only when the user wants catch-up. Never enable it for time-sensitive actions without that agreement.',
}
const maxLatenessSchema = {
  type: 'integer', minimum: 60_000, maximum: 30 * 86400_000,
  description: 'Maximum delay for run_once recovery, in milliseconds (default 24 hours). One-shot tasks only.',
}

function assistantMetadata(input: Record<string, unknown>, existing?: Record<string, unknown> | null): Record<string, unknown> {
  let metadata = { ...existing }
  if (input.source_refs !== undefined) metadata = withScheduledAppSources(metadata, input.source_refs)
  if (input.missed_policy !== undefined || input.max_lateness_ms !== undefined) {
    const previous = schedulerMisfirePolicyFromMetadata(metadata)
    const policy = input.missed_policy ?? previous?.policy ?? 'skip'
    if (input.max_lateness_ms !== undefined && policy !== 'run_once') {
      throw new Error('max_lateness_ms requires missed_policy=run_once')
    }
    metadata.schedulerMisfire = normalizeSchedulerMisfirePolicy({
      version: 1, policy, maxLatenessMs: input.max_lateness_ms ?? previous?.maxLatenessMs ?? 86400_000,
    })
  }
  return metadata
}

function parseSkillRefsInput(value: unknown):
  | { refs: ScheduledAgentSkillRef[] }
  | { error: string } {
  if (!Array.isArray(value)) return { error: 'skill_refs must be an array' }
  try {
    return { refs: normalizeScheduledAgentSkillRefs(value as ScheduledAgentSkillRef[]) }
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) }
  }
}

type RouteInputField = 'channel_type' | 'channel_target' | 'reply_to_message_id'

function routeField(
  input: Record<string, unknown>,
  name: RouteInputField,
): { present: false; value: null } | { present: true; value: string } | { error: string } {
  if (input[name] === undefined) return { present: false, value: null }
  if (typeof input[name] !== 'string' || !input[name].trim()) {
    return { error: `${name} must be a non-empty string` }
  }
  return { present: true, value: input[name].trim() }
}

function resolveToolDeliveryRoute(
  input: Record<string, unknown>,
  context: ToolExecutionContext | undefined,
  options: { allowClear: boolean; existing?: ScheduledJob },
): { route: SchedulerDeliveryRoute } | { error: string } {
  const type = routeField(input, 'channel_type')
  const target = routeField(input, 'channel_target')
  const reply = routeField(input, 'reply_to_message_id')
  if ('error' in type) return type
  if ('error' in target) return target
  if ('error' in reply) return reply

  const clearValue = input.clear_delivery_route
  if (clearValue !== undefined && typeof clearValue !== 'boolean') {
    return { error: 'clear_delivery_route must be a boolean' }
  }
  const clear = clearValue === true
  const hasExplicitRoute = type.present || target.present || reply.present
  if (clear && !options.allowClear) return { error: 'clear_delivery_route is not supported here' }
  if (clear && hasExplicitRoute) {
    return { error: 'clear_delivery_route cannot be combined with delivery route fields' }
  }
  if (clear) {
    return { route: { channelType: null, channelTarget: null, replyToMessageId: null } }
  }

  const channelContext = context?.channelContext
  if (channelContext) {
    const expected: SchedulerDeliveryRoute = {
      channelType: channelContext.channel,
      channelTarget: channelContext.chatKey,
      replyToMessageId: channelContext.triggerMessageId ?? null,
    }
    if (!hasExplicitRoute) return { route: expected }
    if (!type.present || !target.present) {
      return { error: 'channel_type and channel_target must be provided together' }
    }
    if (type.value !== expected.channelType || target.value !== expected.channelTarget) {
      return { error: 'explicit delivery route must match the current channel context' }
    }
    if (reply.present && reply.value !== expected.replyToMessageId) {
      return { error: 'reply_to_message_id must match the current channel context' }
    }
    return { route: expected }
  }

  if (!hasExplicitRoute) {
    return {
      route: options.existing
        ? {
            channelType: options.existing.channelType,
            channelTarget: options.existing.channelTarget,
            replyToMessageId: options.existing.replyToMessageId,
          }
        : { channelType: null, channelTarget: null, replyToMessageId: null },
    }
  }
  if (type.present !== target.present) {
    return { error: 'channel_type and channel_target must be provided together' }
  }

  const route: SchedulerDeliveryRoute = {
    channelType: type.present ? type.value : (options.existing?.channelType ?? null),
    channelTarget: target.present ? target.value : (options.existing?.channelTarget ?? null),
    replyToMessageId: reply.present
      ? reply.value
      : type.present
        ? null
        : (options.existing?.replyToMessageId ?? null),
  }
  const routeError = deliveryRouteError(route)
  return routeError ? { error: routeError } : { route }
}

function serializeRun(run: JobRun): Record<string, unknown> {
  const outcome = projectStoredJobRunOutcome(run)
  return {
    id: run.id,
    jobId: run.jobId,
    agentSessionId: run.agentSessionId ?? null,
    startedAt: run.startedAt,
    finishedAt: run.finishedAt,
    status: run.status,
    taskOutcome: run.taskOutcome ?? outcome.taskOutcome,
    statusIntegrity: run.statusIntegrity ?? outcome.statusIntegrity,
    attempt: run.attempt,
    durationMs: run.durationMs,
    error: run.error ? safeSchedulerEvidenceText(run.error) : null,
    output: run.outputExcerpt
      ? safeSchedulerEvidenceText(run.outputExcerpt.slice(0, 500))
      : null,
  }
}

function safeSchedulerEvidenceText(value: string): string {
  return redactSensitive(value).redacted
}

function notificationPriority(value: unknown): 'normal' | 'high' | 'critical' {
  return value === 'high' || value === 'critical' ? value : 'normal'
}

export function createSchedulerTools(deps: SchedulerToolDeps): ToolDefinitionRuntime[] {
  const { store, triggerSchedulerJob, defaultTimezone } = deps

  const create: ToolDefinitionRuntime = {
    name: 'schedule_create',
    description:
      'Create a durable background, future, or recurring agent task. Use `when: "@now"` for an explicit request to work asynchronously now; it queues a one-shot and returns without waiting for execution. Otherwise `when` accepts natural language ("in 2 minutes", ' +
      '"tomorrow 9am", "every Monday 9am", "every 30 minutes", "2분 후", "매일 오전 9시", ' +
      '"30분마다", "30분 간격으로", "30분 단위로"), 5-field cron ("*/5 * * * *"), ' +
      'cron nicknames ("@daily", "@hourly"), or intervals ("@every 30s", "@every 30m", "@every 1h"). The agent ' +
      'is invoked at the scheduled time with `instruction`; if created from a chat channel or with ' +
      'an explicit delivery route, the result is sent there automatically. Optionally pass `timezone` (IANA, e.g. ' +
      '"Asia/Seoul") for cron, and `max_attempts`/`retry_backoff_ms` for automatic retries on ' +
      'failure. Use only when the user explicitly asks for background, future, or recurring work. ' +
      'When the scheduled workflow depends on a skill explicitly loaded in the current turn, ' +
      'copy that stable `[Skill: id]` identity into `skill_refs`; omit it for unrelated jobs. ' +
      'This is the only supported way to reload skill instructions at execution time — prose in ' +
      '`instruction` is never interpreted as a skill selection. ' +
      'For an anomaly-check job that intentionally emits output only when action is needed, ' +
      'set `notification_priority` to high or critical; ordinary reports stay normal. ' +
      'Use source_refs for app-dependent work; before changing an app item, find its existing jobs with schedule_list source_ref and update/cancel them within the user request. A link identifies a dependency; it does not itself move the scheduled time. ' +
      'When this tool returns success, this action is fully scheduled. Finish any other authorized parts of the request, then confirm the saved result. Do NOT additionally call process.start / terminal.run with ' +
      '`sleep && echo`, `sleep && cat`, `at`, or `wait` as a backup; that second call cannot ' +
      'deliver any output to the user and just burns an approval round-trip.',
    inputSchema: {
      type: 'object',
      properties: {
        when: { type: 'string', description: 'Schedule expression: @now for immediate background work, natural language, cron, @daily/@hourly, or @every 30s / @every 30m. Korean recurring intervals such as "30분 단위로" are accepted.' },
        instruction: { type: 'string', description: 'What to do at the scheduled time' },
        name: { type: 'string', description: 'Optional human-readable name' },
        request_key: { type: 'string', minLength: 1, maxLength: 128, description: 'Stable unique key for this logical creation in the current conversation. Reuse exactly on a retry to avoid duplicate jobs; use schedule_update for changes. A repeated key returns the original job, including its current paused/cancelled/completed state.' },
        source_refs: sourceRefsSchema,
        missed_policy: missedPolicySchema,
        max_lateness_ms: maxLatenessSchema,
        timezone: { type: 'string', description: 'IANA timezone for cron interpretation (e.g. "Asia/Seoul"); omit for the daemon default' },
        max_attempts: { type: 'integer', description: 'Total attempts per fire (1 = no retry, default 1)', minimum: 1, maximum: 20 },
        retry_backoff_ms: { type: 'integer', description: 'Base retry backoff in ms (exponential, default 30000)', minimum: 1000 },
        notification_priority: {
          type: 'string',
          enum: ['normal', 'high', 'critical'],
          description:
            'Priority used for external delivery when this job produces output. Use high/critical only for action-worthy alert jobs, not routine reports.',
        },
        skill_refs: {
          type: 'array',
          maxItems: 64,
          items: {
            type: 'object',
            properties: {
              name: { type: 'string', minLength: 1, maxLength: 256 },
            },
            required: ['name'],
            additionalProperties: false,
          },
          description:
            'Optional stable skill ids to load again at every scheduled execution. Use structured ids from the skill registry; never infer them from instruction prose.',
        },
        unattended: {
          type: 'boolean',
          description:
            'Pre-approve the tool calls this job\'s future runs make, for this job only. '
            + 'Set it only when the user explicitly agreed to let the job run without '
            + 'supervision. This setting does not enable, disable, pause, or resume scheduled firing; '
            + 'it only supplies job-scoped standing approval for policy-gated tool calls. Without it, '
            + 'autonomously allowed calls can still complete, while a call needing approval may time out. '
            + 'Never assume it; ask if the user did not say. '
            + 'The authorization is persisted on this job and restored after daemon restart; '
            + 'it remains scoped to the derived scheduler session and is revoked when the job is cancelled.',
        },
        channel_type: {
          type: 'string',
          minLength: 1,
          description:
            'Optional explicit delivery connector (for example mattermost) when scheduling outside a channel. Must be paired with channel_target. Channel-originated calls cannot override their current channel.',
        },
        channel_target: {
          type: 'string',
          minLength: 1,
          description:
            'Optional connector-specific destination when scheduling outside a channel. Must be paired with channel_type.',
        },
        reply_to_message_id: {
          type: 'string',
          minLength: 1,
          description: 'Optional thread/message id; requires a complete channel_type/channel_target route.',
        },
      },
      required: ['when', 'instruction'],
    },
    async execute(input, context) {
      const start = Date.now()
      try {
        const when = String(input.when ?? '').trim()
        const instruction = String(input.instruction ?? '').trim()
        if (!when || !instruction) return failure('when and instruction are required', Date.now() - start, 'INVALID_INPUT_PERMANENT')

        const timezone =
          typeof input.timezone === 'string' && input.timezone.trim() ? input.timezone.trim() : defaultTimezone
        const linkedMetadata = assistantMetadata(input)
        const requestKey = input.request_key
        if (requestKey !== undefined && (typeof requestKey !== 'string' || !requestKey.trim() || requestKey.length > 128)) {
          return failure('request_key must be a non-empty string of at most 128 characters', Date.now() - start, 'INVALID_INPUT_PERMANENT')
        }
        const tctx = (context ?? {}) as ToolExecutionContext
        const route = resolveToolDeliveryRoute(input, tctx, { allowClear: false })
        if ('error' in route) {
          return failure(route.error, Date.now() - start, 'INVALID_DELIVERY_ROUTE_PERMANENT')
        }
        const parsedSkillRefs = input.skill_refs === undefined
          ? { refs: [] as ScheduledAgentSkillRef[] }
          : parseSkillRefsInput(input.skill_refs)
        if ('error' in parsedSkillRefs) {
          return failure(parsedSkillRefs.error, Date.now() - start, 'INVALID_SKILL_REFS_PERMANENT')
        }
        if (parsedSkillRefs.refs.length > 0) {
          if (!deps.validateSkillRefs) {
            return failure(
              'scheduled skill validation is unavailable',
              Date.now() - start,
              'INVALID_SKILL_REFS_PERMANENT',
            )
          }
          try {
            await deps.validateSkillRefs(parsedSkillRefs.refs)
          } catch (err) {
            return failure(
              err instanceof Error ? err.message : String(err),
              Date.now() - start,
              'INVALID_SKILL_REFS_PERMANENT',
            )
          }
        }
        if (typeof requestKey === 'string') {
          const fingerprint = createHash('sha256').update(JSON.stringify({
            when, instruction, name: input.name, timezone, route: route.route,
            unattended: input.unattended, maxAttempts: input.max_attempts, retryBackoff: input.retry_backoff_ms,
            notificationPriority: input.notification_priority, skills: parsedSkillRefs.refs, linkedMetadata,
          })).digest('hex')
          const previous = store.list().find(candidate => candidate.parentSessionId === (tctx.sessionId ?? null)
            && candidate.channelTarget === route.route.channelTarget
            && candidate.channelType === route.route.channelType
            && (candidate.metadata?.schedulerRequest as Record<string, unknown> | undefined)?.key === requestKey.trim())
          if (previous) {
            const request = previous.metadata?.schedulerRequest as Record<string, unknown>
            if (request.fingerprint !== fingerprint) {
              return failure('request_key was already used with different input; inspect the existing job and use schedule_update', Date.now() - start, 'IDEMPOTENCY_CONFLICT_PERMANENT')
            }
            return success(JSON.stringify({ ...serializeJob(previous), deduplicated: true }), Date.now() - start)
          }
          linkedMetadata.schedulerRequest = { version: 1, key: requestKey.trim(), fingerprint }
        }
        const parsed = parseWhen(when, { timezone })
        if (parsed.kind !== 'oneshot' && (input.missed_policy !== undefined || input.max_lateness_ms !== undefined)) {
          return failure('missed_policy applies only to one-shot tasks', Date.now() - start, 'INVALID_INPUT_PERMANENT')
        }
        const job = store.create({
          name: String(input.name ?? instruction.slice(0, 60)),
          kind: parsed.kind,
          cron: parsed.kind === 'recurring' ? parsed.cron : null,
          runAt: parsed.kind === 'oneshot' ? parsed.runAt : null,
          nextRunAt: parsed.kind === 'oneshot' ? parsed.runAt : parsed.nextRunAt,
          timezone: parsed.kind === 'recurring' ? (timezone ?? null) : null,
          instruction,
          channelType: route.route.channelType,
          channelTarget: route.route.channelTarget,
          replyToMessageId: route.route.replyToMessageId,
          parentSessionId: tctx.sessionId ?? null,
          enabled: true,
          createdBy: 'agent',
          maxAttempts: numOpt(input.max_attempts, 1, 20),
          unattended: input.unattended === true,
          retryBackoffMs: numOpt(input.retry_backoff_ms, 1000),
          metadata: withScheduledAgentSkillRefs(
            { ...linkedMetadata, notificationPriority: notificationPriority(input.notification_priority) },
            parsedSkillRefs.refs,
          ),
        })
        const unattended = input.unattended === true
        if (unattended) {
          deps.grantUnattendedApproval?.(schedulerSessionIdForJob(job.id))
        }
        // Whether the user was actually asked is the thing that matters here.
        // An omitted field means the supervision question is still open, which
        // is materially different from the user having declined it.
        const approvalDecisionPending = input.unattended === undefined
        return success(JSON.stringify({
          id: job.id, name: safeSchedulerEvidenceText(job.name), kind: job.kind,
          when_human: formatWhen(job),
          timezone: job.timezone,
          max_attempts: job.maxAttempts,
          notification_priority: notificationPriority(job.metadata?.notificationPriority),
          skill_refs: parsedSkillRefs.refs,
          source_refs: scheduledAppSourcesFromMetadata(job.metadata),
          missed_policy: schedulerMisfirePolicyFromMetadata(job.metadata)?.policy ?? 'skip',
          max_lateness_ms: schedulerMisfirePolicyFromMetadata(job.metadata)?.policy === 'run_once' ? schedulerMisfirePolicyFromMetadata(job.metadata)!.maxLatenessMs : null,
          unattended,
          approval_mode: unattended ? 'job-scoped-standing' : 'interactive-if-required',
          channel_type: job.channelType,
          channel_target: job.channelTarget,
          reply_to_message_id: job.replyToMessageId,
          status: job.status,
          nextRunAt: job.nextRunAt,
          parentSessionId: job.parentSessionId,
          done: true,
          approval_decision_pending: approvalDecisionPending,
          next_step:
            'Complete any other authorized parts of the current request, then briefly report this saved reservation (e.g. "예약했어요 — ' +
            formatWhen(job) +
            '").'
            + (approvalDecisionPending
              // A scheduled run fires with nobody watching, so an unresolved
              // supervision question is not a detail to omit: without standing
              // approval the first policy-gated call stalls until it times out,
              // and the user only learns of it from a failed job. Ending the
              // turn silently here is what produced exactly that.
              ? ' Then, in the same reply, tell the user this job has no standing approval,'
                + ' so any tool call needing approval will stall while the job runs unattended,'
                + ' and ask whether to grant approval scoped to this job.'
                + ' Apply their answer with schedule_update `unattended`. Do not assume it.'
              : ' Then end your turn.')
            + ' Do NOT call process.start / terminal.run with sleep+echo or '
            + 'a similar shell wait-and-print pattern after this; schedule_create alone delivers '
            + 'the scheduled work through the configured scheduler and delivery route.',
        }), Date.now() - start)
      } catch (err) {
        if (err instanceof SchedulerParseError) {
          return failure(err.message, Date.now() - start, 'PARSE_FAILED_PERMANENT')
        }
        return failure(err instanceof Error ? err.message : String(err), Date.now() - start)
      }
    },
  }

  const list: ToolDefinitionRuntime = {
    name: 'schedule_list',
    description:
      'List scheduled tasks for the current chat (or all when no chat context). ' +
      'Pass status=all to include completed/cancelled/failed jobs. The result includes ' +
      'each job\'s instruction/prompt, next-run time, last error (if any), enabled flag, retry settings, ' +
      'and approvalMode. The `unattended` setting/approvalMode describes only job-scoped standing approval and ' +
      'does not control whether the schedule fires; use enabled, status, nextRunAt, and run history for that. ' +
      'Use status=all when the user asks whether a schedule disappeared, whether a one-shot already ran, or what was actually registered. Filter by source_ref to find jobs linked to an app/collection/item before rescheduling, completing, or deleting it. Page with offset/limit; an empty result ends pagination.',
    inputSchema: {
      type: 'object',
      properties: {
        status: { type: 'string', enum: ['pending', 'all'] },
        source_ref: SCHEDULED_APP_SOURCE_SCHEMA,
        offset: { type: 'integer', minimum: 0 },
        limit: { type: 'integer', minimum: 1, maximum: 100 },
      },
    },
    async execute(input, context) {
      const start = Date.now()
      const tctx = (context ?? {}) as ToolExecutionContext
      const all = input.status === 'all'
      let query: ReturnType<typeof normalizeScheduledAppSources>[number] | undefined
      try {
        query = input.source_ref === undefined ? undefined : normalizeScheduledAppSources([input.source_ref])[0]
      } catch (error) {
        return failure(String(error), Date.now() - start, 'INVALID_INPUT_PERMANENT')
      }
      const jobs = store.list({
        status: all ? undefined : ['pending', 'running'],
        channelTarget: tctx.channelContext?.chatKey,
      })
      const matches = query ? jobs.filter(job => {
        try { return scheduledAppSourcesFromMetadata(job.metadata).some(ref => scheduledAppSourceMatches(ref, query!)) }
        catch { return false }
      }) : jobs
      const offset = numOpt(input.offset, 0) ?? 0
      const limit = numOpt(input.limit, 1, 100) ?? 50
      const out = matches.slice(offset, offset + limit).map(j => ({
        ...serializeJob(j),
        createdBy: j.createdBy,
      }))
      return success(JSON.stringify(out), Date.now() - start)
    },
  }

  const get: ToolDefinitionRuntime = {
    name: 'schedule_get',
    description:
      'Show one scheduled task by id or id prefix. This is the tool equivalent of `/schedule show <id>`: ' +
      'in chat/channel context it is scoped to that chat, so it will not reveal another chat\'s schedules. ' +
      'Optionally include recent run history. The `unattended` setting/approvalMode describes only job-scoped standing ' +
      'approval and does not control whether the schedule fires; use enabled, status, nextRunAt, and run history for that.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'scheduler' },
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Full job id or at least 4-character id prefix from schedule_list' },
        include_runs: { type: 'boolean', description: 'Include recent execution history' },
        run_limit: { type: 'integer', minimum: 1, maximum: 100 },
      },
      required: ['id'],
    },
    async execute(input, context) {
      const start = Date.now()
      const resolved = resolveScopedJob(store, String(input.id ?? ''), context as ToolExecutionContext | undefined)
      if ('error' in resolved) return failure(resolved.error, Date.now() - start, 'NOT_FOUND_PERMANENT')

      const payload: Record<string, unknown> = {
        scope: currentChatKey(context as ToolExecutionContext | undefined) ? 'current_channel' : 'all_visible',
        job: serializeJob(resolved.job),
      }
      if (input.include_runs === true) {
        const limit = numOpt(input.run_limit, 1, 100) ?? 20
        payload.runs = store.listRuns(resolved.job.id, limit).map(serializeRun)
      }
      return success(JSON.stringify(payload), Date.now() - start)
    },
  }

  const runs: ToolDefinitionRuntime = {
    name: 'schedule_runs',
    description:
      'Show recent execution history for a scheduled task (id from schedule_list). Use this to ' +
      'check whether a recurring job is failing, or to see what a job actually produced. ' +
      'When run_id is provided, return only that exact run under the resolved job.',
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string' },
        run_id: {
          type: 'string',
          minLength: 1,
          description: 'Optional exact persisted run id from schedule_run_now or prior history',
        },
        limit: { type: 'integer', minimum: 1, maximum: 100 },
      },
      required: ['id'],
    },
    async execute(input, context) {
      const start = Date.now()
      const resolved = resolveScopedJob(store, String(input.id ?? ''), context as ToolExecutionContext | undefined)
      if ('error' in resolved) return failure(resolved.error, Date.now() - start, 'NOT_FOUND_PERMANENT')
      const hasRunId = input.run_id !== undefined
      const runId = typeof input.run_id === 'string' ? input.run_id.trim() : ''
      if (hasRunId && !runId) {
        return failure('run_id must be a non-empty string', Date.now() - start)
      }
      if (runId) {
        const run = store.getRun(resolved.job.id, runId)
        if (!run) {
          return failure('run not found for scheduled job', Date.now() - start, 'NOT_FOUND_PERMANENT')
        }
        return success(JSON.stringify([serializeRun(run)]), Date.now() - start)
      }
      const limit = numOpt(input.limit, 1, 100) ?? 20
      const out = store.listRuns(resolved.job.id, limit).map(serializeRun)
      return success(JSON.stringify(out), Date.now() - start)
    },
  }

  const makeToggle = (name: string, enabled: boolean, verb: string): ToolDefinitionRuntime => ({
    name,
    description:
      `${verb} a scheduled task by id (from schedule_list). ` +
      (enabled
        ? 'A resumed job fires again on its next scheduled time.'
        : 'A paused job stays registered but does not fire until resumed. An already-running execution may still finish.'),
    inputSchema: {
      type: 'object',
      properties: { id: { type: 'string' } },
      required: ['id'],
    },
    async execute(input, context) {
      const start = Date.now()
      const resolved = resolveScopedJob(store, String(input.id ?? ''), context as ToolExecutionContext | undefined)
      if ('error' in resolved) return failure(resolved.error, Date.now() - start, 'NOT_FOUND_PERMANENT')
      store.setEnabled(resolved.job.id, enabled)
      return success(JSON.stringify({
        ok: true,
        id: resolved.job.id,
        name: safeSchedulerEvidenceText(resolved.job.name),
        enabled,
      }), Date.now() - start)
    },
  })

  const cancel: ToolDefinitionRuntime = {
    name: 'schedule_cancel',
    description:
      'Cancel a scheduled task by id or id prefix (from schedule_list). This is permanent — use ' +
      'schedule_pause to temporarily stop a recurring job. In chat/channel context this is scoped ' +
      'to that chat, matching `/schedule cancel <id>`. Cancelling prevents future firing and retries; an already-running execution may still finish. Do not claim its active tools were interrupted.',
    inputSchema: {
      type: 'object',
      properties: { id: { type: 'string' } },
      required: ['id'],
    },
    async execute(input, context) {
      const start = Date.now()
      const resolved = resolveScopedJob(store, String(input.id ?? ''), context as ToolExecutionContext | undefined)
      if ('error' in resolved) return failure(resolved.error, Date.now() - start, 'NOT_FOUND_PERMANENT')
      store.cancel(resolved.job.id)
      // A cancelled job must not leave its pre-authorisation behind: the
      // session id is derived from the job id, so a later job could not reuse
      // it, but the grant should still not outlive what it was granted for.
      deps.revokeUnattendedApproval?.(schedulerSessionIdForJob(resolved.job.id))
      return success(JSON.stringify({
        ok: true,
        id: resolved.job.id,
        name: safeSchedulerEvidenceText(resolved.job.name),
      }), Date.now() - start)
    },
  }

  const cancelAll: ToolDefinitionRuntime = {
    name: 'schedule_cancel_all',
    description:
      'Cancel all pending scheduled tasks in the current chat/channel. This is the built-in tool ' +
      'equivalent of `/schedule cancel all confirm`; use it when the user explicitly asks to cancel, ' +
      'delete, remove, or clear all scheduled tasks. In chat/channel context it is scoped to that ' +
      'chat and must not affect other chats. Outside channel context it cancels all pending jobs ' +
      'visible to this runtime.',
    inputSchema: {
      type: 'object',
      properties: {},
    },
    async execute(_input, context) {
      const start = Date.now()
      const chatKey = currentChatKey(context as ToolExecutionContext | undefined)
      const jobs = store.list({
        status: ['pending'],
        ...(chatKey ? { channelTarget: chatKey } : {}),
      })
      for (const job of jobs) {
        store.cancel(job.id)
        deps.revokeUnattendedApproval?.(schedulerSessionIdForJob(job.id))
      }
      return success(JSON.stringify({
        ok: true,
        scope: chatKey ? 'current_channel' : 'all_visible',
        cancelled: jobs.length,
        ids: jobs.map((job) => job.id),
        done: true,
        next_step:
          jobs.length === 0
            ? 'Reply that there were no pending scheduled tasks to cancel and end the turn.'
            : `Reply that ${jobs.length} scheduled task(s) were cancelled and end the turn.`,
      }), Date.now() - start)
    },
  }

  const update: ToolDefinitionRuntime = {
    name: 'schedule_update',
    description:
      'Update one scheduled task by id or id prefix. This is the tool equivalent of `/schedule edit` ' +
      'and `/schedule reschedule`: change `name`, `when`, `instruction`, retry settings, notification priority, selected skill refs, explicit unattended authority, or delivery route without losing run ' +
      'history. Route type/target are atomic, and clear_delivery_route removes type/target/reply together. In chat/channel context it is scoped to that chat and cannot route outside that current channel. It cannot change a one-shot task ' +
      'into a recurring task or vice versa; cancel and recreate for that.',
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Full job id or at least 4-character id prefix from schedule_list' },
        name: { type: 'string', minLength: 1, description: 'Optional replacement human-readable name; omitted preserves the current name' },
        source_refs: sourceRefsSchema,
        missed_policy: missedPolicySchema,
        max_lateness_ms: maxLatenessSchema,
        when: { type: 'string', description: 'Optional replacement schedule expression with the same kind as the existing job' },
        instruction: { type: 'string', description: 'Optional replacement instruction/prompt for the scheduled agent run' },
        timezone: { type: 'string', description: 'Optional IANA timezone for recurring schedule interpretation' },
        max_attempts: { type: 'integer', description: 'Optional total attempts per fire', minimum: 1, maximum: 20 },
        retry_backoff_ms: { type: 'integer', description: 'Optional base retry backoff in ms', minimum: 1000 },
        notification_priority: {
          type: 'string',
          enum: ['normal', 'high', 'critical'],
          description: 'Optional replacement external-delivery priority.',
        },
        skill_refs: {
          type: 'array',
          maxItems: 64,
          items: {
            type: 'object',
            properties: {
              name: { type: 'string', minLength: 1, maxLength: 256 },
            },
            required: ['name'],
            additionalProperties: false,
          },
          description:
            'Optional replacement skill selection. An empty array clears selected skills; omission preserves them.',
        },
        unattended: {
          type: 'boolean',
          description:
            'Optional explicit standing authorization for future headless runs. Set true only when the user explicitly authorizes it; false revokes the job-scoped grant.',
        },
        channel_type: {
          type: 'string',
          minLength: 1,
          description: 'Optional replacement delivery connector; provide together with channel_target.',
        },
        channel_target: {
          type: 'string',
          minLength: 1,
          description: 'Optional replacement connector destination; provide together with channel_type.',
        },
        reply_to_message_id: {
          type: 'string',
          minLength: 1,
          description: 'Optional replacement thread/message id; requires an existing or supplied complete route.',
        },
        clear_delivery_route: {
          type: 'boolean',
          description: 'Set true to atomically remove channel type, target, and reply id.',
        },
      },
      required: ['id'],
    },
    async execute(input, context) {
      const start = Date.now()
      const resolved = resolveScopedJob(store, String(input.id ?? ''), context as ToolExecutionContext | undefined)
      if ('error' in resolved) return failure(resolved.error, Date.now() - start, 'NOT_FOUND_PERMANENT')

      const hasWhen = typeof input.when === 'string' && input.when.trim().length > 0
      const hasInstruction = typeof input.instruction === 'string'
      const hasName = input.name !== undefined
      const hasTimezone = input.timezone !== undefined
      if (hasName && (typeof input.name !== 'string' || !input.name.trim())) {
        return failure('name must be a non-empty string', Date.now() - start, 'INVALID_INPUT_PERMANENT')
      }
      if (hasTimezone && (typeof input.timezone !== 'string' || !input.timezone.trim())) {
        return failure('timezone must be a non-empty IANA timezone', Date.now() - start, 'INVALID_INPUT_PERMANENT')
      }
      const maxAttempts = numOpt(input.max_attempts, 1, 20)
      const retryBackoffMs = numOpt(input.retry_backoff_ms, 1000)
      const hasNotificationPriority = input.notification_priority !== undefined
      const hasSkillRefs = input.skill_refs !== undefined
      const hasAssistantContract = input.source_refs !== undefined || input.missed_policy !== undefined || input.max_lateness_ms !== undefined
      let linkedMetadata: Record<string, unknown> | undefined
      if (hasAssistantContract) {
        try {
          if (resolved.job.kind !== 'oneshot' && (input.missed_policy !== undefined || input.max_lateness_ms !== undefined)) {
            throw new Error('missed_policy applies only to one-shot tasks')
          }
          linkedMetadata = assistantMetadata(input, resolved.job.metadata)
        } catch (error) {
          return failure(String(error), Date.now() - start, 'INVALID_INPUT_PERMANENT')
        }
      }
      const parsedSkillRefs = hasSkillRefs ? parseSkillRefsInput(input.skill_refs) : undefined
      if (parsedSkillRefs && 'error' in parsedSkillRefs) {
        return failure(parsedSkillRefs.error, Date.now() - start, 'INVALID_SKILL_REFS_PERMANENT')
      }
      if (parsedSkillRefs && parsedSkillRefs.refs.length > 0) {
        if (!deps.validateSkillRefs) {
          return failure(
            'scheduled skill validation is unavailable',
            Date.now() - start,
            'INVALID_SKILL_REFS_PERMANENT',
          )
        }
        try {
          await deps.validateSkillRefs(parsedSkillRefs.refs)
        } catch (err) {
          return failure(
            err instanceof Error ? err.message : String(err),
            Date.now() - start,
            'INVALID_SKILL_REFS_PERMANENT',
          )
        }
      }
      const hasUnattended = input.unattended !== undefined
      if (hasUnattended && typeof input.unattended !== 'boolean') {
        return failure('unattended must be a boolean', Date.now() - start, 'INVALID_INPUT_PERMANENT')
      }
      const hasDeliveryRouteUpdate = input.clear_delivery_route === true
        || input.channel_type !== undefined
        || input.channel_target !== undefined
        || input.reply_to_message_id !== undefined
      if (!hasName && !hasWhen && !hasInstruction && !hasTimezone && maxAttempts == null && retryBackoffMs == null && !hasNotificationPriority && !hasSkillRefs && !hasUnattended && !hasDeliveryRouteUpdate && !hasAssistantContract) {
        return failure('provide at least one schedule, timezone, instruction, retry, priority, skill, unattended, or delivery-route change', Date.now() - start, 'INVALID_INPUT_PERMANENT')
      }

      const job = resolved.job
      const routeUpdate = hasDeliveryRouteUpdate
        ? resolveToolDeliveryRoute(input, context as ToolExecutionContext | undefined, {
            allowClear: true,
            existing: job,
          })
        : undefined
      if (routeUpdate && 'error' in routeUpdate) {
        return failure(routeUpdate.error, Date.now() - start, 'INVALID_DELIVERY_ROUTE_PERMANENT')
      }

      if (!hasName && !hasWhen && !hasInstruction && !hasTimezone && maxAttempts == null && retryBackoffMs == null && !hasNotificationPriority && !hasSkillRefs && !hasDeliveryRouteUpdate && !hasAssistantContract && hasUnattended) {
        const unattended = input.unattended === true
        store.setUnattended(job.id, unattended)
        const schedulerSessionId = schedulerSessionIdForJob(job.id)
        if (unattended) deps.grantUnattendedApproval?.(schedulerSessionId)
        else deps.revokeUnattendedApproval?.(schedulerSessionId)
        return success(JSON.stringify({ ok: true, job: serializeJob(store.get(job.id)!) }), Date.now() - start)
      }

      const instruction = hasInstruction ? String(input.instruction ?? '').trim() : job.instruction
      if (!instruction) return failure('instruction cannot be empty', Date.now() - start, 'INVALID_INPUT_PERMANENT')
      const name = hasName ? String(input.name).trim() : job.name
      let nextMetadata: Record<string, unknown> | null | undefined
      if (hasNotificationPriority || parsedSkillRefs || linkedMetadata) {
        const priorityMetadata = hasNotificationPriority
          ? { ...(linkedMetadata ?? job.metadata), notificationPriority: notificationPriority(input.notification_priority) }
          : linkedMetadata ?? job.metadata
        nextMetadata = parsedSkillRefs
          ? withScheduledAgentSkillRefs(priorityMetadata, parsedSkillRefs.refs)
          : priorityMetadata
      }

      if (hasTimezone && job.kind !== 'recurring') {
        return failure(
          'timezone applies only to recurring tasks',
          Date.now() - start,
          'INVALID_INPUT_PERMANENT',
        )
      }
      const timezone = hasTimezone
        ? String(input.timezone).trim()
        : job.timezone ?? defaultTimezone
      let parsed: ReturnType<typeof parseWhen> | null = null
      if (hasWhen || hasTimezone) {
        try {
          const expression = hasWhen ? String(input.when).trim() : job.cron
          if (!expression) {
            return failure(
              'existing recurring job has no schedule; provide when to replace it',
              Date.now() - start,
              'INVALID_INPUT_PERMANENT',
            )
          }
          parsed = parseWhen(expression, { timezone })
        } catch (err) {
          if (err instanceof SchedulerParseError) {
            return failure(err.message, Date.now() - start, 'PARSE_FAILED_PERMANENT')
          }
          return failure(err instanceof Error ? err.message : String(err), Date.now() - start)
        }
        if (parsed.kind !== job.kind) {
          return failure(
            `cannot change schedule kind from ${job.kind} to ${parsed.kind}; cancel and recreate the task instead`,
            Date.now() - start,
            'INVALID_INPUT_PERMANENT',
          )
        }
      }

      let updated: ScheduledJob | null
      const preserveRunState = !hasWhen && !hasInstruction && !hasTimezone
      if (job.kind === 'recurring') {
        const cron = parsed?.kind === 'recurring' ? parsed.cron : job.cron
        if (!cron) {
          return failure('existing recurring job has no schedule; provide when to replace it', Date.now() - start, 'INVALID_INPUT_PERMANENT')
        }
        updated = store.updateRecurringJob({
          id: job.id,
          name,
          cron,
          nextRunAt: parsed?.kind === 'recurring' ? parsed.nextRunAt : job.nextRunAt,
          timezone: parsed ? (timezone ?? null) : job.timezone,
          instruction,
          channelType: routeUpdate?.route.channelType,
          channelTarget: routeUpdate?.route.channelTarget,
          replyToMessageId: routeUpdate?.route.replyToMessageId,
          enabled: job.enabled,
          maxAttempts,
          retryBackoffMs,
          metadata: nextMetadata,
          preserveRunState,
        })
      } else {
        const runAt = parsed?.kind === 'oneshot' ? parsed.runAt : job.runAt ?? job.nextRunAt
        if (!Number.isFinite(runAt)) {
          return failure('existing one-shot job has no run time; provide when to replace it', Date.now() - start, 'INVALID_INPUT_PERMANENT')
        }
        updated = store.updateOneShotJob({
          id: job.id,
          name,
          runAt,
          nextRunAt: parsed?.kind === 'oneshot' ? parsed.runAt : job.nextRunAt,
          instruction,
          channelType: routeUpdate?.route.channelType,
          channelTarget: routeUpdate?.route.channelTarget,
          replyToMessageId: routeUpdate?.route.replyToMessageId,
          enabled: job.enabled,
          maxAttempts,
          retryBackoffMs,
          metadata: nextMetadata,
          preserveRunState,
        })
      }

      if (!updated) return failure(`job could not be updated: ${job.id}`, Date.now() - start)
      if (hasUnattended) {
        const unattended = input.unattended === true
        store.setUnattended(job.id, unattended)
        const schedulerSessionId = schedulerSessionIdForJob(job.id)
        if (unattended) deps.grantUnattendedApproval?.(schedulerSessionId)
        else deps.revokeUnattendedApproval?.(schedulerSessionId)
      }
      return success(JSON.stringify({ ok: true, job: serializeJob(store.get(job.id) ?? updated) }), Date.now() - start)
    },
  }

  const tools: ToolDefinitionRuntime[] = [
    create, list, get, runs,
    makeToggle('schedule_pause', false, 'Pause'),
    makeToggle('schedule_resume', true, 'Resume'),
    cancel,
    cancelAll,
    update,
  ]

  if (triggerSchedulerJob) {
    tools.push({
      name: 'schedule_run_now',
      description:
        'Trigger a scheduled task to run immediately, in addition to its normal schedule. ' +
        'Returns immediately with a persisted jobId/runId and running status; this acknowledges a start, not completion. ' +
        'The run continues after this chat turn. Inspect schedule_runs with that exact run_id on a later status request; do not poll in a tight loop. ' +
        'The run is recorded in run history (visible via schedule_runs). Use when the user asks ' +
        'to "run it now" / "지금 실행해줘".',
      inputSchema: {
        type: 'object',
        properties: { id: { type: 'string' } },
        required: ['id'],
      },
      async execute(input, context) {
        const start = Date.now()
        const resolved = resolveScopedJob(store, String(input.id ?? ''), context as ToolExecutionContext | undefined)
        if ('error' in resolved) return failure(resolved.error, Date.now() - start, 'NOT_FOUND_PERMANENT')
        try {
          const result = await triggerSchedulerJob(resolved.job.id, { waitForCompletion: false })
          if (!result.started) {
            // Reporting ok for a run that never started would let the agent
            // tell the user the task ran.
            return failure(
              `job is not runnable (enabled: ${store.get(resolved.job.id)?.enabled === true}, status: ${store.get(resolved.job.id)?.status ?? 'unknown'}); manual run was not started`,
              Date.now() - start,
            )
          }
          if (result.status === 'failed') {
            return failure(
              JSON.stringify({ ok: false, ...result }),
              Date.now() - start,
            )
          }
          return success(
            JSON.stringify({
              ok: true,
              ...result,
              name: safeSchedulerEvidenceText(resolved.job.name),
            }),
            Date.now() - start,
          )
        } catch (err) {
          return failure(err instanceof Error ? err.message : String(err), Date.now() - start)
        }
      },
    })
  }

  return tools
}
