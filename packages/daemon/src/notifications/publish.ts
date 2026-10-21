import {
  createTraceRedactionContext,
  redactSensitiveText,
} from '../observability/trace-redaction.js'
import { createLogger } from '../logger.js'
import type { SepilotdConfig } from '../config/schema.js'
import type { PendingApproval } from '../server/runtime/approvals.js'
import {
  ALL_NOTIFICATION_AUDIENCES,
  normalizeNotificationAudience,
  type NotificationAudience,
} from './audience.js'
import { publishNotification } from './broker.js'
import {
  notifyRelayFromEnv,
  type NotifyRelayNotification,
} from './push-notify-relay.js'
import { ntfyRelayFromEnv, type NtfyNotification } from './push-ntfy.js'
import {
  createNotificationsRepo,
  notificationRelayProviderStatusIsTerminal,
  type NotificationCorrelation,
  type NotificationItem,
} from './repo.js'
import {
  createNotificationRelayOutbox,
  type NotificationRelayOutboxRecord,
} from './relay-outbox.js'

const logger = createLogger('notification-publish')
const NOTIFICATION_TITLE_MAX = 200
const NOTIFICATION_BODY_MAX = 8_000
const NOTIFICATION_RELAY_RETRY_BASE_MS = 30_000
const NOTIFICATION_RELAY_RETRY_MAX_MS = 60 * 60_000
const NOTIFICATION_RELAY_INVALID_RESPONSE_MAX_ATTEMPTS = 3
const NOTIFICATION_RELAY_STATUS_UNCONFIRMED_MAX_ATTEMPTS = 3
const NOTIFICATION_RELAY_POLL_INTERVAL_MS = 30_000
const durableNotificationRedactionContext = createTraceRedactionContext({
  env: {} as NodeJS.ProcessEnv,
  cwd: '',
  stateDir: '',
})

function redactDurableNotificationText(value: string): string {
  return redactSensitiveText(value, durableNotificationRedactionContext, {
    maxStringLength: Math.max(1, value.length),
  })
}

export interface StoredNotificationInput {
  id?: string
  title: string
  body?: string
  url?: string | null
  topic?: string | null
  audience?: string[] | null
  correlation?: NotificationCorrelation | null
  createdAt?: number
}

const ntfyRelay = ntfyRelayFromEnv(process.env)
export function externalNotificationRelayConfigured(): boolean {
  return notifyRelayFromEnv(process.env) !== null
}

function isPushEligibleAudience(audience: NotificationAudience | undefined): boolean {
  if (audience == null) return true
  return audience.includes(ALL_NOTIFICATION_AUDIENCES)
}

function isApprovalTopic(topic: string | null | undefined): boolean {
  return typeof topic === 'string' && topic.startsWith('approval:')
}

export function buildNtfyRelayPayload(item: NotificationItem): NtfyNotification | null {
  if (!isPushEligibleAudience(item.audience)) return null
  const relayBody = isApprovalTopic(item.topic) ? undefined : item.body
  return {
    title: redactSensitiveText(item.title, createTraceRedactionContext(), {
      maxStringLength: NOTIFICATION_TITLE_MAX,
    }),
    body: relayBody
      ? redactSensitiveText(relayBody, createTraceRedactionContext(), {
          maxStringLength: NOTIFICATION_BODY_MAX,
        })
      : undefined,
    url: item.url,
  }
}

export function buildNotifyRelayPayload(
  item: NotificationItem,
): NotifyRelayNotification | null {
  const externalPayload = buildNtfyRelayPayload(item)
  if (!externalPayload) return null

  const isScheduler = item.topic?.startsWith('scheduler') === true
  const schedulerPriority = item.topic === 'scheduler:critical'
    || item.topic?.startsWith('scheduler:critical:')
    ? 'critical'
    : item.topic === 'scheduler:high'
      || item.topic?.startsWith('scheduler:high:')
      ? 'high'
      : 'normal'
  const agentParts = item.topic?.split(':') ?? []
  const isAgentNotification = agentParts[0] === 'agent-notification'
  const agentType = agentParts[1]
  const agentPriority = agentParts[2]
  const isApproval = isApprovalTopic(item.topic)
  return {
    id: item.id,
    type: isAgentNotification && (
      agentType === 'alert'
      || agentType === 'incident'
      || agentType === 'report'
      || agentType === 'custom'
    )
      ? agentType
      : isScheduler && schedulerPriority !== 'normal'
      ? 'alert'
      : isScheduler
        ? 'report'
        : isApproval
          ? 'alert'
          : 'custom',
    priority: isAgentNotification && (
      agentPriority === 'low'
      || agentPriority === 'high'
      || agentPriority === 'critical'
    )
      ? agentPriority
      : isApproval
        ? 'high'
        : schedulerPriority,
    title: externalPayload.title,
    text: externalPayload.body?.trim() || 'Open Sepilot to review this notification.',
  }
}

function relayToNtfy(item: NotificationItem): void {
  if (!ntfyRelay) return
  const payload = buildNtfyRelayPayload(item)
  if (!payload) return
  void ntfyRelay.publish(payload)
}

function relayRetryDelayMs(attempt: number): number {
  return Math.min(
    NOTIFICATION_RELAY_RETRY_MAX_MS,
    NOTIFICATION_RELAY_RETRY_BASE_MS * 2 ** Math.max(0, attempt - 1),
  )
}

function rejectedRelayStatusIsRetryable(httpStatus: number | null): boolean {
  return httpStatus === 408
    || httpStatus === 425
    || httpStatus === 429
    || (httpStatus !== null && httpStatus >= 500)
}

function retryRelayDelivery(
  record: NotificationRelayOutboxRecord,
  error: string,
  completedAt: number,
): void {
  createNotificationRelayOutbox().markFailed(
    record.notificationId,
    error,
    completedAt + relayRetryDelayMs(record.attempt),
  )
}

function relayStatusLookupShouldKeepRetrying(
  outcome: 'observed' | 'rejected' | 'unreachable' | 'invalid_response',
  httpStatus: number | null,
  attempt: number,
): boolean {
  if (outcome === 'observed' || outcome === 'unreachable') return true
  if (
    outcome === 'rejected'
    && (
      httpStatus === 401
      || httpStatus === 403
      || rejectedRelayStatusIsRetryable(httpStatus)
    )
  ) {
    return true
  }
  return attempt < NOTIFICATION_RELAY_STATUS_UNCONFIRMED_MAX_ATTEMPTS
}

async function runNotificationRelayOutboxDrain(
  env: NodeJS.ProcessEnv,
  fetchImpl?: typeof fetch,
): Promise<void> {
  const notifyRelay = notifyRelayFromEnv(env, fetchImpl)
  if (!notifyRelay) return

  const repo = createNotificationsRepo()
  const outbox = createNotificationRelayOutbox()
  while (true) {
    const claimed = outbox.claimDue(Date.now())
    if (claimed.length === 0) break

    for (const record of claimed) {
      try {
        const item = repo.get(record.notificationId)
        if (!item) {
          outbox.markDead(record.notificationId, 'notification no longer exists')
          continue
        }
        const payload = buildNotifyRelayPayload(item)
        if (!payload) {
          outbox.markDead(record.notificationId, 'notification is not externally eligible')
          continue
        }

        const result = await notifyRelay.publish(payload)
        const updated = repo.recordRelayDelivery(item.id, result)
        if (!updated) {
          outbox.markDead(record.notificationId, 'notification disappeared during delivery')
          continue
        }
        publishNotification(updated)

        if (result.status === 'accepted' || result.status === 'pending_review') {
          outbox.markDelivered(record.notificationId, result.completedAt)
          continue
        }
        if (result.status === 'unreachable') {
          retryRelayDelivery(record, 'notify relay unreachable', result.completedAt)
          continue
        }
        if (result.status === 'rejected') {
          const error = result.httpStatus === null
            ? 'notify relay rejected notification'
            : `notify relay rejected notification with HTTP ${result.httpStatus}${result.errorCode ? ` (${result.errorCode})` : ''}`
          if (rejectedRelayStatusIsRetryable(result.httpStatus)) {
            retryRelayDelivery(record, error, result.completedAt)
          } else {
            outbox.markDead(record.notificationId, error)
          }
          continue
        }
        if (record.attempt < NOTIFICATION_RELAY_INVALID_RESPONSE_MAX_ATTEMPTS) {
          retryRelayDelivery(record, 'notify relay returned invalid acceptance evidence', result.completedAt)
        } else {
          outbox.markDead(
            record.notificationId,
            'notify relay repeatedly returned invalid acceptance evidence',
          )
        }
      } catch (error) {
        try {
          retryRelayDelivery(
            record,
            error instanceof Error ? error.message : String(error),
            Date.now(),
          )
        } catch (recordError) {
          logger.warn('failed to retain notify relay retry state', {
            error: recordError instanceof Error ? recordError.message : String(recordError),
          })
        }
      }
    }
  }
}

async function runNotificationRelayProviderReconciliation(
  env: NodeJS.ProcessEnv,
  fetchImpl?: typeof fetch,
): Promise<void> {
  const notifyRelay = notifyRelayFromEnv(env, fetchImpl)
  if (!notifyRelay) return

  const repo = createNotificationsRepo()

  while (true) {
    const claimed = repo.claimRelayProviderChecks(Date.now())
    if (claimed.length === 0) return

    for (const record of claimed) {
      try {
        const result = await notifyRelay.status(record.messageId)
        const terminal = result.outcome === 'observed' && result.messageStatus !== null
          ? notificationRelayProviderStatusIsTerminal(result.messageStatus)
          : !relayStatusLookupShouldKeepRetrying(
              result.outcome,
              result.httpStatus,
              record.attempt,
            )
        const nextCheckAt = terminal
          ? null
          : result.completedAt + (
              result.outcome === 'observed'
                ? NOTIFICATION_RELAY_POLL_INTERVAL_MS
                : relayRetryDelayMs(record.attempt)
            )
        const updated = repo.recordRelayProviderStatus(record.notificationId, result, {
          terminal,
          nextCheckAt,
        })
        if (updated) publishNotification(updated)
      } catch (error) {
        logger.warn('failed to reconcile notify relay provider status', {
          error: error instanceof Error ? error.message : String(error),
        })
      }
    }
  }
}

let notificationRelayOutboxDrainInFlight: Promise<void> | null = null
let notificationRelayProviderReconciliationInFlight: Promise<void> | null = null

function drainNotificationRelayOutbox(
  env: NodeJS.ProcessEnv,
  fetchImpl?: typeof fetch,
): Promise<void> {
  if (notificationRelayOutboxDrainInFlight) return notificationRelayOutboxDrainInFlight
  const draining = runNotificationRelayOutboxDrain(env, fetchImpl).finally(() => {
    if (notificationRelayOutboxDrainInFlight === draining) {
      notificationRelayOutboxDrainInFlight = null
    }
  })
  notificationRelayOutboxDrainInFlight = draining
  return draining
}

function reconcileNotificationRelayProviderStatuses(
  env: NodeJS.ProcessEnv,
  fetchImpl?: typeof fetch,
): Promise<void> {
  if (notificationRelayProviderReconciliationInFlight) {
    return notificationRelayProviderReconciliationInFlight
  }
  const reconciling = runNotificationRelayProviderReconciliation(env, fetchImpl).finally(() => {
    if (notificationRelayProviderReconciliationInFlight === reconciling) {
      notificationRelayProviderReconciliationInFlight = null
    }
  })
  notificationRelayProviderReconciliationInFlight = reconciling
  return reconciling
}

export function drainNotificationRelayDeliveries(options?: {
  env?: NodeJS.ProcessEnv
  fetchImpl?: typeof fetch
}): Promise<void> {
  const env = options?.env ?? process.env
  return Promise.all([
    drainNotificationRelayOutbox(env, options?.fetchImpl),
    reconcileNotificationRelayProviderStatuses(env, options?.fetchImpl),
  ]).then(() => undefined)
}

export interface NotificationRelayDeliveryWorker {
  start(): void
  stop(): Promise<void>
}

export function createNotificationRelayDeliveryWorker(options?: {
  intervalMs?: number
}): NotificationRelayDeliveryWorker {
  let timer: ReturnType<typeof setInterval> | null = null
  const wake = () => {
    void drainNotificationRelayDeliveries().catch((error) => {
      logger.warn('notify relay outbox drain failed', {
        error: error instanceof Error ? error.message : String(error),
      })
    })
  }
  return {
    start() {
      if (timer) return
      wake()
      timer = setInterval(wake, options?.intervalMs ?? NOTIFICATION_RELAY_POLL_INTERVAL_MS)
      timer.unref()
    },
    async stop() {
      if (timer) {
        clearInterval(timer)
        timer = null
      }
      await Promise.all([
        notificationRelayOutboxDrainInFlight,
        notificationRelayProviderReconciliationInFlight,
      ])
    },
  }
}

export function publishStoredNotification(input: StoredNotificationInput): NotificationItem {
  const repo = createNotificationsRepo()
  const normalizedAudience = normalizeNotificationAudience(input.audience)
  const relayDeliveryPending = externalNotificationRelayConfigured()
    && isPushEligibleAudience(normalizedAudience)
  const item = repo.upsert({
    id: input.id,
    title: redactDurableNotificationText(input.title),
    body: redactDurableNotificationText(input.body ?? ''),
    url: input.url ?? null,
    topic: input.topic ?? null,
    audience: normalizedAudience,
    correlation: input.correlation ?? null,
    createdAt: input.createdAt,
    relayDeliveryPending,
  })
  publishNotification(item)
  relayToNtfy(item)
  if (relayDeliveryPending) void drainNotificationRelayDeliveries().catch((error) => {
    logger.warn('notify relay outbox wake failed', {
      error: error instanceof Error ? error.message : String(error),
    })
  })
  return item
}

export function publishChatCompletionNotification(
  config: Pick<SepilotdConfig, 'notifications'> | undefined,
  input: {
    outcome: 'completed' | 'failed'
    sessionId: string
    sessionTitle?: string | null
    durationMs: number
    surface?: string | null
    errorMessage?: string
  },
): NotificationItem | null {
  const cfg = config?.notifications?.chatCompletion
  if (cfg?.enabled === false) return null
  if (input.outcome === 'failed' && cfg?.includeFailures === false) return null

  const seconds = Math.max(1, Math.round(input.durationMs / 1000))
  const sessionTitle = input.sessionTitle?.trim() || input.sessionId
  const title =
    input.outcome === 'completed'
      ? `Chat completed: ${sessionTitle}`
      : `Chat failed: ${sessionTitle}`
  const surface = input.surface ? ` from ${input.surface}` : ''
  const body =
    input.outcome === 'completed'
      ? `Finished${surface} in ${seconds}s.`
      : `Failed${surface} after ${seconds}s.${input.errorMessage ? ` ${input.errorMessage.slice(0, 200)}` : ''}`

  return publishStoredNotification({
    title,
    body,
    url: `sepilotd://chat?sessionId=${encodeURIComponent(input.sessionId)}`,
    topic: `chat:${input.sessionId}`,
  })
}

export function publishSchedulerRunNotification(input: {
  id?: string
  title: string
  body: string
  jobId?: string
  runId?: string
  audience?: string[] | null
  priority?: 'normal' | 'high' | 'critical'
}): NotificationItem {
  const priority = input.priority ?? 'normal'
  return publishStoredNotification({
    id: input.id,
    title: input.title,
    body: input.body,
    url: input.jobId
      ? `sepilotd://tasks?jobId=${encodeURIComponent(input.jobId)}`
      : 'sepilotd://tasks',
    topic: input.jobId
      ? priority === 'normal'
        ? `scheduler:job:${input.jobId}`
        : `scheduler:${priority}:job:${input.jobId}`
      : priority === 'normal'
        ? 'scheduler'
        : `scheduler:${priority}`,
    audience: input.audience ?? null,
    correlation: input.jobId
      ? {
          kind: 'scheduler',
          jobId: input.jobId,
          runId: input.runId?.trim() || null,
        }
      : null,
  })
}

function summarizeApprovalInput(approval: PendingApproval): string {
  const input = approval.input ?? {}
  const executable = typeof input.executable === 'string' ? input.executable : ''
  const args = Array.isArray(input.args)
    ? input.args.filter((arg): arg is string => typeof arg === 'string')
    : []
  if (executable) return [executable, ...args].join(' ').slice(0, 180)
  if (typeof input.path === 'string') return input.path.slice(0, 180)
  if (typeof input.url === 'string') return input.url.slice(0, 180)
  return approval.tool
}

export function publishApprovalPendingNotification(
  _config: Pick<SepilotdConfig, 'notifications'> | undefined,
  input: {
    approval: PendingApproval
    outcome?: 'pending' | 'parked' | 'expired'
  },
): NotificationItem {
  const outcome = input.outcome === 'expired' ? 'parked' : (input.outcome ?? 'pending')
  const approval = input.approval
  const title =
    outcome === 'parked'
      ? `Approval still pending: ${approval.tool}`
      : `Approval needed: ${approval.tool}`
  const summary = summarizeApprovalInput(approval)
  const body =
    outcome === 'parked'
      ? `The run is paused and will continue when you answer (stop reason approval_timeout): ${summary}.`
      : `Waiting for approval until ${approval.expiresAt}: ${summary}.`
  return publishStoredNotification({
    title,
    body,
    url: `sepilotd://chat?sessionId=${encodeURIComponent(approval.sessionId)}`,
    topic: `approval:${approval.sessionId}:${approval.requestId}`,
    audience: null,
  })
}
