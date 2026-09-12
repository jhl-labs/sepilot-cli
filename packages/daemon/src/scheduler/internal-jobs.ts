import { createHash } from 'node:crypto'
import type { ScheduledJob, JobStore } from './job-store.js'
import type { DreamingEngine } from '../memory/dreaming.js'
import type { SemanticMemoryStore } from '../memory/types.js'
import { nextRecurringRun } from './time-parser.js'

export interface InternalJobDeps {
  dreaming: DreamingEngine
  semanticIndex: SemanticMemoryStore
  notifications?: {
    publish(input: { id?: string; title: string; body: string; url: string | null }): void
  }
  memoryMaintenance: {
    maxAgeDays: number
    maxImportance: number
    dryRun: boolean
  }
  // Periodic retention sweep for the observability + notification stores, which
  // otherwise grow unbounded (pruneExpired was only ever triggered manually and
  // notifications had no delete path at all).
  retention?: {
    prune(): Promise<void> | void
  }
}

export function createInternalJobRunner(deps: InternalJobDeps) {
  return async (kind: string, _job: ScheduledJob, context?: { runId?: string }): Promise<void> => {
    if (kind === 'dreaming') {
      await deps.dreaming.consolidate()
      return
    }
    if (kind === 'memory-maintenance') {
      await deps.semanticIndex.runMaintenance({
        staleAfterDays: deps.memoryMaintenance.maxAgeDays,
        lowImportance: deps.memoryMaintenance.maxImportance,
        dryRun: deps.memoryMaintenance.dryRun,
        actor: 'scheduler',
        reason: 'Scheduled memory lifecycle maintenance',
      })
      return
    }
    if (kind === 'notification') {
      if (!deps.notifications) throw new Error('notifications capability is not initialized')
      const notification = notificationInputFromJob(_job, context?.runId)
      deps.notifications.publish(notification)
      return
    }
    if (kind === 'retention') {
      if (!deps.retention) throw new Error('retention capability is not initialized')
      await deps.retention.prune()
      return
    }
    throw new Error(`unknown internal job: ${kind}`)
  }
}

function notificationInputFromJob(job: ScheduledJob, runId?: string): {
  id?: string
  title: string
  body: string
  url: string | null
} {
  const notification = recordValue(job.metadata?.notification)
  const title = stringValue(notification.title) || job.name || 'Reminder'
  const body = stringValue(notification.body)
  const url = stringValue(notification.url)
  return {
    // A job can fire repeatedly. Deduplicate the same run, never all future
    // occurrences of the job (including reminders that were already read).
    id: runId
      ? `scheduled-notification-${createHash('sha256').update(JSON.stringify([job.id, stringValue(notification.id), runId])).digest('hex')}`
      : stringValue(notification.id) || job.id,
    title,
    body,
    url: url || null,
  }
}

function recordValue(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

/**
 * Builds the self-contained instruction for the opt-in user-profile
 * refresh run. It runs headless (no channel) so it must spell out which
 * tools to use — there is no system-prompt guidance in that context.
 * LLM-driven only: the agent does the reflection and decides what to
 * keep/drop; no regex/heuristic extraction here.
 */
export function userProfileRefreshInstruction(section: string): string {
  return [
    `You are running an unattended maintenance pass to refresh the "${section}" section of long-term memory (MEMORY.md).`,
    'Steps:',
    `1. Call memory.context.snapshot to load the current "${section}" section and recent daily notes.`,
    '2. Call memory.daily.read("today") and memory.daily.read("yesterday"); optionally memory.search for durable facts the user has stated about themselves (role, preferences, working style, recurring projects, tools/stack, communication style).',
    `3. Synthesise an updated "${section}": concise bulleted durable facts about the user. Merge duplicates, drop anything that turned out to be wrong or stale, and NEVER include volatile data (prices, schedules, secrets, one-off task state).`,
    `4. Write it back with memory.section.replace({ section: "${section}", content: <markdown bullets> }). If you learned nothing new and the section is already accurate, make no changes.`,
    'Keep it short — this is a profile, not a log. End with a one-line summary of what changed (or "no change").',
  ].join('\n')
}

/**
 * Builds the self-contained instruction for the opt-in proactive digest
 * run. Channel-bound, so the agent's reply is delivered to the chat by
 * the scheduler executor's dispatch step. LLM-driven; the agent decides
 * what (if anything) is worth surfacing.
 */
export function proactiveDigestInstruction(): string {
  return [
    'You are sending an unprompted, proactive check-in to the user. Keep it short and skippable — a glance, not a report.',
    'Gather context first:',
    '1. memory.daily.read("today") and memory.daily.read("yesterday") — look for open-loop / backlog items still pending.',
    '2. memory.usage (or memory.reminders.list) — any reminders coming due soon.',
    '3. schedule_list — any pending scheduled tasks the user set up.',
    'Then write 1-2 short paragraphs surfacing only what is actually actionable or noteworthy (pending items, things due, anything that looks stuck). If nothing is pending, say so in one line — do not pad it. Never invent items; never include secrets or volatile data. Do not run any write/state-changing tools.',
  ].join('\n')
}

export function ensureInternalJobs(
  store: JobStore,
  schedules: {
    dreaming: { schedule: string; enabled: boolean }
    memoryMaintenance: { schedule: string; enabled: boolean }
    userProfile?: { schedule: string; enabled: boolean; section: string }
    digest?: { schedule: string; enabled: boolean; channelType?: string; channelTarget?: string }
    retention?: { schedule: string; enabled: boolean }
  },
): void {
  const retention = schedules.retention ?? { schedule: 'daily', enabled: true }
  const specs: InternalRecurringJobSpec[] = [
    {
      name: 'dreaming-consolidation',
      schedule: schedules.dreaming.schedule,
      enabled: schedules.dreaming.enabled,
      instruction: '__internal:dreaming',
    },
    {
      name: 'observability-retention',
      schedule: retention.schedule,
      enabled: retention.enabled,
      instruction: '__internal:retention',
    },
    {
      name: 'memory-maintenance',
      schedule: schedules.memoryMaintenance.schedule,
      enabled: schedules.memoryMaintenance.enabled,
      instruction: '__internal:memory-maintenance',
    },
  ]

  const userProfile = schedules.userProfile
  specs.push({
    name: 'user-profile-refresh',
    schedule: userProfile?.schedule ?? '1d',
    enabled: userProfile?.enabled ?? false,
    instruction: userProfileRefreshInstruction(userProfile?.section ?? 'User Profile'),
  })

  const digest = schedules.digest
  specs.push({
    name: 'proactive-digest',
    schedule: digest?.schedule ?? '0 9 * * *',
    enabled: Boolean(digest?.enabled && digest.channelType && digest.channelTarget),
    instruction: proactiveDigestInstruction(),
    channelType: digest?.channelType ?? null,
    channelTarget: digest?.channelTarget ?? null,
  })

  for (const spec of specs) {
    reconcileInternalRecurringJob(store, spec)
  }
}

interface InternalRecurringJobSpec {
  name: string
  schedule: string
  enabled: boolean
  instruction: string
  channelType?: string | null
  channelTarget?: string | null
}

/**
 * Reconcile scheduler rows to the configured internal-task state. Internal
 * jobs are configuration-owned: disabling an option cancels a stale row,
 * while re-enabling it reuses the same row and run history.
 */
function reconcileInternalRecurringJob(store: JobStore, spec: InternalRecurringJobSpec): void {
  const matches = store.list().filter(
    (job) => job.createdBy === 'internal' && job.name === spec.name,
  )
  const existing = matches.find((job) => job.kind === 'recurring')

  for (const duplicate of matches) {
    if (duplicate.id === existing?.id) continue
    if (duplicate.enabled || !['completed', 'cancelled'].includes(duplicate.status)) {
      store.cancel(duplicate.id)
    }
  }

  if (!spec.enabled) {
    if (existing && (existing.enabled || !['completed', 'cancelled'].includes(existing.status))) {
      store.cancel(existing.id)
    }
    return
  }

  const cron = scheduleToCron(spec.schedule)
  const channelType = spec.channelType ?? null
  const channelTarget = spec.channelTarget ?? null
  if (!existing) {
    store.create({
      name: spec.name,
      kind: 'recurring',
      cron,
      runAt: null,
      nextRunAt: Date.now() + 60_000,
      instruction: spec.instruction,
      channelType,
      channelTarget,
      replyToMessageId: null,
      parentSessionId: null,
      enabled: true,
      createdBy: 'internal',
    })
    return
  }

  const definitionChanged = existing.cron !== cron
    || existing.instruction !== spec.instruction
    || existing.channelType !== channelType
    || existing.channelTarget !== channelTarget
  const needsReactivation = !existing.enabled
    || existing.status === 'completed'
    || existing.status === 'cancelled'
  if (!definitionChanged && !needsReactivation) return

  try {
    store.updateRecurringJob({
      id: existing.id,
      name: spec.name,
      cron,
      nextRunAt: needsReactivation
        ? Date.now() + 60_000
        : nextRecurringRun(cron, Date.now()),
      timezone: existing.timezone,
      instruction: spec.instruction,
      channelType,
      channelTarget,
      replyToMessageId: null,
      parentSessionId: null,
      enabled: true,
    })
  } catch {
    // Keep the last known-good definition when a configured schedule is invalid.
  }
}

export function scheduleToCron(s: string): string {
  const interval = s.match(/^(\d+)(s|m|h|d)$/)
  if (interval) {
    const [, n, unit] = interval
    if (unit === 'd') return `0 0 */${n} * *`
    if (unit === 'h') return `0 */${n} * * *`
    if (unit === 'm') return `*/${n} * * * *`
    return s
  }
  return s
}
