// packages/daemon/src/scheduler/engine.ts
import { nextRetryAt, DEFAULT_RETRY_POLICY, type RunStopReason, type TokenUsage } from '@sepilotd/core'
import { schedulerMisfirePolicyFromMetadata } from '@sepilotd/api-client'
import { createLogger } from '../logger.js'
import type { JobStore, ScheduledJob } from './job-store.js'
import { nextRecurringRun, SchedulerParseError } from './time-parser.js'

const log = createLogger('scheduler:engine')
const ZERO_USAGE: TokenUsage = { inputTokens: 0, outputTokens: 0 }

export interface JobExecutionResult {
  output?: string
  usage?: TokenUsage | null
  /** Explicit task-level outcome. Omitted means complete for backward compatibility. */
  outcome?: 'complete' | 'incomplete'
  /** Bounded user-safe reason when outcome is incomplete. */
  failureReason?: string
  /**
   * Structured termination cause from the agent's done event. Lets run history
   * separate "stopped at a deliberate budget with resumable state" from a hard
   * failure instead of leaving both as an opaque error string.
   */
  stopReason?: RunStopReason
}

export interface JobExecutionContext {
  runId: string
  /** Aborted when bounded graceful shutdown has to release this run. */
  signal?: AbortSignal
  /** Suppress scheduler-owned delivery while preserving a real execution. */
  suppressDelivery?: boolean
}

/** Executor returns output text (for run-history excerpts) and optional token usage. */
export type JobExecutor = (
  job: ScheduledJob,
  context?: JobExecutionContext,
) => Promise<string | void | JobExecutionResult | null>

export interface SchedulerEngineOptions {
  store: JobStore
  executor: JobExecutor
  /** Live manual runs share the store but execute outside this engine. */
  activeManualJobIds?: () => readonly string[]
  tickMs?: number
  maxConcurrent?: number
  missedGraceMs?: number
  staleRunningMs?: number
  gcAfterDays?: number
  /** Upper bound for a single retry backoff wait (default 1h). */
  maxBackoffMs?: number
  /** Default IANA timezone for cron jobs that don't carry their own. */
  defaultTimezone?: string
  /** Auto-disable a recurring job after this many consecutive failed fires (default 5). */
  maxConsecutiveFailures?: number
  /** Global automatic-run kill switch. Manual runs are handled by the runtime trigger path. */
  enabled?: () => boolean
  /** Daily automatic-run token budget. Undefined means unlimited. */
  dailyTokenBudget?: () => number | null | undefined
  onEvent?: (event: SchedulerEngineEvent) => void
}

export type SchedulerEngineEvent =
  | { type: 'picked'; jobId: string; runId: string; latencyMs: number }
  | { type: 'completed'; jobId: string; runId: string; durationMs: number }
  | { type: 'failed'; jobId: string; runId: string; reason: string; durationMs: number; willRetry: boolean }
  | { type: 'missed'; jobId: string; missedByMs: number }
  | { type: 'auto-disabled'; jobId: string; runId: string; reason: string; consecutiveFailures: number; threshold: number }
  | { type: 'budget-exhausted'; dateKey: string; usedTokens: number; budgetTokens: number }
  | { type: 'interrupted'; jobId: string; runId: string; reason: string; retryAt: number }

export interface SchedulerShutdownResult {
  drained: number
  interrupted: number
}

interface ActiveSchedulerRun {
  job: ScheduledJob
  runId: string
  controller: AbortController
  promise: Promise<void>
  interrupted: boolean
}

export const SCHEDULER_SHUTDOWN_INTERRUPTED =
  'daemon graceful shutdown interrupted the running job; it will retry after restart'

export function schedulerLocalDateKey(now = Date.now()): string {
  const date = new Date(now)
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function normalizeTokenUsage(usage: TokenUsage | null | undefined): TokenUsage {
  const rawInputTokens = usage?.inputTokens
  const rawOutputTokens = usage?.outputTokens
  const inputTokens = typeof rawInputTokens === 'number' && Number.isFinite(rawInputTokens)
    ? Math.max(0, Math.floor(rawInputTokens))
    : 0
  const outputTokens = typeof rawOutputTokens === 'number' && Number.isFinite(rawOutputTokens)
    ? Math.max(0, Math.floor(rawOutputTokens))
    : 0
  return { inputTokens, outputTokens }
}

function tokenTotal(usage: TokenUsage): number {
  return usage.inputTokens + usage.outputTokens
}

function normalizeJobExecutionResult(
  result: string | void | JobExecutionResult | null,
): {
  output: string | void
  usage: TokenUsage
  outcome: 'complete' | 'incomplete'
  failureReason?: string
  stopReason?: RunStopReason
} {
  if (typeof result === 'string' || result == null) {
    return {
      output: result ?? undefined,
      usage: { ...ZERO_USAGE },
      outcome: 'complete',
    }
  }
  return {
    output: typeof result.output === 'string' ? result.output : undefined,
    usage: normalizeTokenUsage(result.usage),
    outcome: result.outcome === 'incomplete' ? 'incomplete' : 'complete',
    ...(result.outcome === 'incomplete' && result.failureReason?.trim()
      ? { failureReason: result.failureReason.trim() }
      : {}),
    ...(result.stopReason ? { stopReason: result.stopReason } : {}),
  }
}

export class SchedulerEngine {
  private timer: NodeJS.Timeout | null = null
  private gcTimer: NodeJS.Timeout | null = null
  private staleTimer: NodeJS.Timeout | null = null
  private acceptingRuns = false
  private readonly activeRuns = new Map<string, ActiveSchedulerRun>()
  private shutdownPromise: Promise<SchedulerShutdownResult> | null = null
  private readonly tickMs: number
  private readonly maxConcurrent: number
  private readonly missedGraceMs: number
  private readonly staleRunningMs: number
  private readonly gcAfterDays: number
  private readonly maxBackoffMs: number
  private readonly maxConsecutiveFailures: number
  private readonly defaultTimezone: string | undefined

  constructor(private readonly opts: SchedulerEngineOptions) {
    this.tickMs = opts.tickMs ?? 1000
    this.maxConcurrent = opts.maxConcurrent ?? 4
    this.missedGraceMs = opts.missedGraceMs ?? 3600_000
    this.staleRunningMs = opts.staleRunningMs ?? 5 * 60_000
    this.gcAfterDays = opts.gcAfterDays ?? 7
    this.maxBackoffMs = opts.maxBackoffMs ?? DEFAULT_RETRY_POLICY.maxBackoffMs
    this.maxConsecutiveFailures = Math.max(1, opts.maxConsecutiveFailures ?? 5)
    this.defaultTimezone = opts.defaultTimezone
  }

  /** Number of jobs currently executing. Used by the idle reaper work-check. */
  get activeRunCount(): number {
    return this.activeRuns.size
  }

  start(): void {
    if (this.timer) return
    this.acceptingRuns = true
    this.shutdownPromise = null
    this.reapStaleRunning(false)
    this.reanchorFailedRecurring()
    this.gcExpiredHistory()
    this.timer = setInterval(() => {
      void this.tick().catch((error) => {
        log.error(
          `scheduler tick failed; the next interval will retry: ${error instanceof Error ? error.message : String(error)}`,
        )
      })
    }, this.tickMs)
    this.timer.unref?.()
    this.staleTimer = setInterval(() => {
      try {
        this.reapStaleRunning(true)
      } catch (error) {
        log.error(
          `scheduler stale-run maintenance failed; the next interval will retry: ${error instanceof Error ? error.message : String(error)}`,
        )
      }
    }, this.staleRunningMs)
    this.staleTimer.unref?.()
    this.gcTimer = setInterval(() => {
      try {
        this.gcExpiredHistory()
      } catch (error) {
        log.error(
          `scheduler history GC failed; the next interval will retry: ${error instanceof Error ? error.message : String(error)}`,
        )
      }
    }, 86_400_000)
    this.gcTimer.unref?.()
  }

  stop(): void {
    this.acceptingRuns = false
    if (this.timer) clearInterval(this.timer)
    if (this.gcTimer) clearInterval(this.gcTimer)
    if (this.staleTimer) clearInterval(this.staleTimer)
    this.timer = null
    this.gcTimer = null
    this.staleTimer = null
  }

  /**
   * Stop admission, then give already-claimed jobs a bounded window to finish.
   * Anything still running at the boundary is explicitly persisted and
   * released for retry before its executor is aborted. Late executor results
   * are ignored, so they cannot overwrite the interruption evidence.
   */
  shutdown(graceMs = 10_000): Promise<SchedulerShutdownResult> {
    if (this.shutdownPromise) return this.shutdownPromise
    this.stop()
    this.shutdownPromise = this.drainActiveRuns(Math.max(0, Math.floor(graceMs)))
    return this.shutdownPromise
  }

  private async drainActiveRuns(graceMs: number): Promise<SchedulerShutdownResult> {
    const snapshot = [...this.activeRuns.values()]
    if (snapshot.length === 0) return { drained: 0, interrupted: 0 }

    let timer: NodeJS.Timeout | undefined
    const drained = Promise.allSettled(snapshot.map((run) => run.promise)).then(() => true)
    const deadline = new Promise<false>((resolve) => {
      timer = setTimeout(() => resolve(false), graceMs)
    })
    const completed = graceMs > 0 ? await Promise.race([drained, deadline]) : false
    if (timer) clearTimeout(timer)
    if (completed) return { drained: snapshot.length, interrupted: 0 }

    let interrupted = 0
    for (const run of snapshot) {
      if (this.activeRuns.get(run.runId) !== run) continue
      run.interrupted = true
      const retryAt = Date.now()
      if (
        this.opts.store.releaseInterruptedRun(
          run.job.id,
          run.runId,
          retryAt,
          SCHEDULER_SHUTDOWN_INTERRUPTED,
        )
      ) {
        interrupted++
        this.emit({
          type: 'interrupted',
          jobId: run.job.id,
          runId: run.runId,
          reason: SCHEDULER_SHUTDOWN_INTERRUPTED,
          retryAt,
        })
      }
      run.controller.abort(new Error(SCHEDULER_SHUTDOWN_INTERRUPTED))
    }
    if (interrupted > 0) {
      log.warn(`interrupted ${interrupted} scheduler job(s) after ${graceMs}ms graceful drain`)
    }
    return { drained: snapshot.length - interrupted, interrupted }
  }

  private emit(event: SchedulerEngineEvent): void {
    if (!this.opts.onEvent) return
    try {
      this.opts.onEvent(event)
    } catch (err) {
      log.error(`scheduler onEvent subscriber failed: ${err instanceof Error ? err.message : err}`)
    }
  }

  private gcExpiredHistory(): void {
    const removed = this.opts.store.gc(this.gcAfterDays * 86_400_000)
    const removedRuns = this.opts.store.gcRuns(this.gcAfterDays * 86_400_000)
    if (removed > 0 || removedRuns > 0) {
      log.info(`gc removed ${removed} terminal jobs, ${removedRuns} run records`)
    }
  }

  private timezoneFor(job: ScheduledJob): string | undefined {
    return job.timezone ?? this.defaultTimezone
  }

  private reapStaleRunning(reanchorRecurring: boolean): void {
    const recovered = this.opts.store.cleanupStaleRunning(this.staleRunningMs, [
      ...new Set([
        ...[...this.activeRuns.values()].map(run => run.job.id),
        ...(this.opts.activeManualJobIds?.() ?? []),
      ]),
    ])
    if (recovered <= 0) return
    log.warn(`recovered ${recovered} stale-running jobs`)
    if (reanchorRecurring) this.reanchorFailedRecurring()
  }

  private maybeAutoDisableRecurring(jobId: string, runId: string, reason: string): void {
    const job = this.opts.store.get(jobId)
    if (!job || job.kind !== 'recurring' || !job.enabled) return
    if (job.consecutiveFailures < this.maxConsecutiveFailures) return

    this.opts.store.setEnabled(job.id, false)
    this.emit({
      type: 'auto-disabled',
      jobId: job.id,
      runId,
      reason,
      consecutiveFailures: job.consecutiveFailures,
      threshold: this.maxConsecutiveFailures,
    })
    log.error(`auto-disabled recurring job ${job.id} after ${job.consecutiveFailures} consecutive failures: ${reason}`)
  }

  private currentDailyTokenBudget(): number | undefined {
    const configured = this.opts.dailyTokenBudget?.()
    if (configured == null) return undefined
    if (!Number.isFinite(configured) || configured <= 0) return undefined
    return Math.floor(configured)
  }

  private dailyBudgetAllowsRun(): boolean {
    const budgetTokens = this.currentDailyTokenBudget()
    if (budgetTokens == null) return true

    const dateKey = schedulerLocalDateKey()
    const usage = this.opts.store.getSchedulerDailyUsage(dateKey)
    const usedTokens = tokenTotal(usage)
    if (usedTokens < budgetTokens) return true

    if (this.opts.store.markSchedulerDailyBudgetNotified(dateKey)) {
      this.emit({
        type: 'budget-exhausted',
        dateKey,
        usedTokens,
        budgetTokens,
      })
      log.warn(`scheduler daily token budget exhausted for ${dateKey}: ${usedTokens}/${budgetTokens}`)
    }
    return false
  }

  private recordSchedulerUsage(usage: TokenUsage): void {
    if (tokenTotal(usage) <= 0) return
    try {
      this.opts.store.addSchedulerDailyUsage(schedulerLocalDateKey(), usage)
    } catch (err) {
      log.warn(`failed to record scheduler token usage: ${err instanceof Error ? err.message : err}`)
    }
  }

  /**
   * A recurring job is meant to keep firing on its cron forever. If one ended
   * up `failed` (executor threw and retries were exhausted, a bad cron once
   * threw during advance, an old daemon left it that way, …) it would sit
   * dead because `pickReady` only takes `pending` jobs. On startup, re-arm any
   * still-enabled recurring job at its next cron occurrence; `lastError` is
   * kept so the prior failure is still visible via `schedule_runs` / list.
   */
  private reanchorFailedRecurring(): void {
    const now = Date.now()
    let rearmed = 0
    for (const job of this.opts.store.list({ status: ['failed'] })) {
      if (job.kind !== 'recurring' || !job.enabled || !job.cron) continue
      try {
        const next = nextRecurringRun(job.cron, now, this.timezoneFor(job))
        this.opts.store.reanchorRecurring(job.id, next)
        rearmed++
      } catch (err) {
        log.warn(`could not re-arm recurring job ${job.id} (${job.cron}): ${err instanceof Error ? err.message : err}`)
      }
    }
    if (rearmed > 0) log.info(`re-armed ${rearmed} recurring job(s) that were left in a failed state`)
  }

  private async tick(): Promise<void> {
    if (!this.acceptingRuns) return
    if (this.opts.enabled?.() === false) return
    const slots = this.maxConcurrent - this.activeRuns.size
    if (slots <= 0) return
    if (!this.dailyBudgetAllowsRun()) return
    const now = Date.now()
    const ready = this.opts.store.pickReady(now, slots)
    for (const job of ready) {
      let graceMs = this.missedGraceMs
      if (job.kind === 'oneshot' && job.attempt === 0) {
        try {
          const policy = schedulerMisfirePolicyFromMetadata(job.metadata)
          if (policy?.policy === 'run_once') graceMs = policy.maxLatenessMs
        } catch {
          this.opts.store.updateStatus(job.id, 'failed', { lastError: 'Invalid scheduler offline recovery policy; review this task before retrying.' })
          continue
        }
      }
      if (job.kind === 'oneshot' && job.attempt === 0 && job.runAt !== null && job.nextRunAt === job.runAt && now - job.runAt > graceMs) {
        this.opts.store.updateMetadata(job.id, {
          ...job.metadata,
          schedulerMissed: { version: 1, reason: 'late', scheduledAt: job.runAt, detectedAt: now },
        })
        this.opts.store.updateStatus(job.id, 'cancelled', { lastError: `missed by ${now - job.runAt}ms` })
        this.emit({ type: 'missed', jobId: job.id, missedByMs: now - job.runAt })
        log.warn(`missed oneshot job ${job.id} (${now - job.runAt}ms past grace)`)
        continue
      }
      this.runJob(job)
    }
  }

  private runJob(job: ScheduledJob): void {
    const startedAt = Date.now()
    // job.attempt is 0 on a fresh fire; the executing attempt is attempt+1 (1-indexed).
    const attemptNo = job.attempt + 1
    const runId = this.opts.store.recordRunStart(job.id, attemptNo, startedAt)
    const controller = new AbortController()
    const activeRun: ActiveSchedulerRun = {
      job,
      runId,
      controller,
      promise: Promise.resolve(),
      interrupted: false,
    }
    this.emit({ type: 'picked', jobId: job.id, runId, latencyMs: startedAt - job.nextRunAt })

    const execution = Promise.resolve()
      .then(() => this.opts.executor(job, { runId, signal: controller.signal }))
      .then((result) => {
        if (activeRun.interrupted) return
        const { output, usage, outcome, failureReason, stopReason } = normalizeJobExecutionResult(result)
        const duration = Date.now() - startedAt
        const excerpt = typeof output === 'string' && output.trim().length > 0 ? output : null
        if (outcome === 'incomplete') {
          this.finishFailedRun({
            job,
            runId,
            attemptNo,
            startedAt,
            duration,
            reason: failureReason ?? 'Scheduled agent reported an incomplete result.',
            outputExcerpt: excerpt,
            usage,
            stopReason,
          })
          return
        }
        this.opts.store.recordRunFinish(runId, 'success', {
          durationMs: duration,
          outputExcerpt: excerpt,
          ...(stopReason ? { stopReason } : {}),
        })
        this.recordSchedulerUsage(usage)
        if (this.opts.store.get(job.id)?.status !== 'running') {
          // Preserve a cancellation or re-armed definition made during execution.
          // The exact run above still retains its real completion evidence.
        } else if (job.kind === 'oneshot') {
          this.opts.store.updateStatus(job.id, 'completed', { lastRunAt: startedAt })
        } else {
          this.advanceRecurring(job, runId, startedAt, null)
        }
        this.emit({ type: 'completed', jobId: job.id, runId, durationMs: duration })
      })
      .catch((err) => {
        if (activeRun.interrupted) return
        const duration = Date.now() - startedAt
        const reason = err instanceof Error ? err.message : String(err)
        this.finishFailedRun({ job, runId, attemptNo, startedAt, duration, reason })
      })
      .finally(() => {
        if (this.activeRuns.get(runId) === activeRun) this.activeRuns.delete(runId)
      })
    activeRun.promise = execution
    this.activeRuns.set(runId, activeRun)
  }

  private finishFailedRun(input: {
    job: ScheduledJob
    runId: string
    attemptNo: number
    startedAt: number
    duration: number
    reason: string
    outputExcerpt?: string | null
    usage?: TokenUsage
    stopReason?: RunStopReason
  }): void {
    this.opts.store.recordRunFinish(input.runId, 'failed', {
      durationMs: input.duration,
      error: input.reason,
      outputExcerpt: input.outputExcerpt ?? null,
      ...(input.stopReason ? { stopReason: input.stopReason } : {}),
    })
    if (input.usage) this.recordSchedulerUsage(input.usage)

    if (this.opts.store.get(input.job.id)?.status !== 'running') {
      this.emit({ type: 'failed', jobId: input.job.id, runId: input.runId,
        reason: input.reason, durationMs: input.duration, willRetry: false })
      return
    }

    const retryAt = nextRetryAt(Date.now(), input.attemptNo, {
      maxAttempts: input.job.maxAttempts,
      backoffMs: input.job.retryBackoffMs,
      maxBackoffMs: this.maxBackoffMs,
    })
    if (retryAt != null) {
      this.opts.store.markRetry(input.job.id, retryAt, input.attemptNo, input.reason)
      this.emit({
        type: 'failed',
        jobId: input.job.id,
        runId: input.runId,
        reason: input.reason,
        durationMs: input.duration,
        willRetry: true,
      })
      log.warn(`job ${input.job.id} failed (attempt ${input.attemptNo}/${input.job.maxAttempts}); retrying at ${new Date(retryAt).toISOString()}: ${input.reason}`)
      return
    }

    if (input.job.kind === 'oneshot') {
      this.opts.store.updateStatus(input.job.id, 'failed', {
        lastRunAt: input.startedAt,
        lastError: input.reason,
      })
    } else {
      this.advanceRecurring(input.job, input.runId, input.startedAt, input.reason)
    }
    this.emit({
      type: 'failed',
      jobId: input.job.id,
      runId: input.runId,
      reason: input.reason,
      durationMs: input.duration,
      willRetry: false,
    })
    log.error(`job ${input.job.id} failed permanently (attempt ${input.attemptNo}/${input.job.maxAttempts}): ${input.reason}`)
  }

  /** Move a recurring job to its next scheduled fire. `error` non-null = the just-finished fire failed. */
  private advanceRecurring(
    job: ScheduledJob,
    runId: string,
    startedAt: number,
    error: string | null,
  ): void {
    if (!job.cron) {
      this.opts.store.updateStatus(job.id, 'failed', { lastRunAt: startedAt, lastError: 'recurring job has no cron expression' })
      return
    }
    try {
      const next = nextRecurringRun(job.cron, Date.now(), this.timezoneFor(job))
      if (error) {
        this.opts.store.rescheduleAfterFailure(job.id, next, startedAt, error)
        this.maybeAutoDisableRecurring(job.id, runId, error)
      } else {
        this.opts.store.updateNextRun(job.id, next, startedAt)
      }
    } catch (err) {
      const reason = err instanceof SchedulerParseError ? err.message : (err as Error).message
      log.error(`recurring next-run computation failed for ${job.id} (${job.cron}): ${reason}`)
      this.opts.store.updateStatus(job.id, 'failed', { lastRunAt: startedAt, lastError: reason })
    }
  }
}
