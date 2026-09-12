// packages/daemon/src/scheduler/job-store.ts
import { randomUUID } from 'node:crypto'
import type { SqliteDatabase } from '../db/sqlite.js'
import type {
  ScheduledJob,
  JobRun,
  JobKind,
  JobCreatedBy,
  JobRunStatus,
  RunStopKind,
  RunStopReason,
  TokenUsage,
  ScheduledJobStatus as JobStatus,
} from '@sepilotd/core'
import { openDomainDb } from '../storage/domain-db.js'
import { projectStoredJobRunOutcome } from './agent-outcome.js'

export type { JobKind, JobCreatedBy, ScheduledJob, JobRun, JobRunStatus } from '@sepilotd/core'
export type { ScheduledJobStatus as JobStatus } from '@sepilotd/core'

/** Only an enabled pending job currently owns a future scheduler claim. */
export function hasPendingScheduledRun(
  job: Pick<ScheduledJob, 'enabled' | 'nextRunAt' | 'status'>,
): boolean {
  return job.enabled
    && job.status === 'pending'
    && Number.isFinite(job.nextRunAt)
}

const DEFAULT_MAX_ATTEMPTS = 1
const DEFAULT_RETRY_BACKOFF_MS = 30_000
const OUTPUT_EXCERPT_MAX = 2000

export interface NewJobInput {
  id?: string
  name: string
  kind: JobKind
  cron: string | null
  runAt: number | null
  nextRunAt: number
  timezone?: string | null
  instruction: string
  channelType: string | null
  channelTarget: string | null
  replyToMessageId: string | null
  parentSessionId: string | null
  enabled: boolean
  createdBy: JobCreatedBy
  maxAttempts?: number
  /** Pre-approve this job's own runs; persisted so it survives a restart. */
  unattended?: boolean
  retryBackoffMs?: number
  metadata?: Record<string, unknown> | null
}

export interface UpdateRecurringJobInput {
  id: string
  name: string
  cron: string
  nextRunAt: number
  timezone: string | null
  instruction: string
  channelType?: string | null
  channelTarget?: string | null
  replyToMessageId?: string | null
  parentSessionId?: string | null
  enabled: boolean
  maxAttempts?: number
  retryBackoffMs?: number
  metadata?: Record<string, unknown> | null
  /** Preserve current execution status/error counters for control-plane-only edits. */
  preserveRunState?: boolean
}

export interface UpdateOneShotJobInput {
  id: string
  name: string
  runAt: number
  nextRunAt: number
  instruction: string
  channelType?: string | null
  channelTarget?: string | null
  replyToMessageId?: string | null
  parentSessionId?: string | null
  enabled: boolean
  maxAttempts?: number
  retryBackoffMs?: number
  metadata?: Record<string, unknown> | null
  /** Preserve current execution status/error counters for control-plane-only edits. */
  preserveRunState?: boolean
}

export interface UpdateStatusOpts {
  lastRunAt?: number
  lastError?: string | null
}

export interface RecordRunFinishOpts {
  finishedAt?: number
  durationMs?: number
  error?: string | null
  outputExcerpt?: string | null
  /**
   * Structured termination cause. Persisted so run history can separate a
   * deliberate, resumable budget stop from a hard failure without parsing the
   * human-readable error text.
   */
  stopReason?: RunStopReason
}

export interface SchedulerDailyUsage {
  dateKey: string
  inputTokens: number
  outputTokens: number
  budgetNotifiedAt: number | null
}

export interface JobStore {
  list(filter?: { status?: JobStatus[]; channelTarget?: string }): ScheduledJob[]
  get(id: string): ScheduledJob | null
  create(input: NewJobInput): ScheduledJob
  /** Set a terminal/transient status. For 'failed', records lastError; for 'completed'/'cancelled' clears it and disables the job. */
  updateStatus(id: string, status: JobStatus, opts?: UpdateStatusOpts): void
  /** Advance a recurring job to its next fire after a successful run: status=pending, attempt=0, lastError cleared. */
  updateNextRun(id: string, nextRunAt: number, lastRunAt: number): void
  /** Advance a recurring job to its next fire after exhausting retries: status=pending, attempt=0, lastError kept. */
  rescheduleAfterFailure(id: string, nextRunAt: number, lastRunAt: number, lastError: string): void
  /** Hold a failed attempt for retry: status=pending, attempt=N, next_run_at=backoff target, lastError kept. */
  markRetry(id: string, nextRunAt: number, attempt: number, lastError: string): void
  /**
   * Finish one in-flight run as a graceful-shutdown interruption and release
   * its job back to pending without consuming retry/failure state. Returns
   * false when the run was already terminal, so a late shutdown cannot
   * overwrite a real executor outcome.
   */
  releaseInterruptedRun(
    jobId: string,
    runId: string,
    retryAt: number,
    reason: string,
    finishedAt?: number,
  ): boolean
  /** Re-arm a recurring job that was left in a terminal state: status=pending, attempt=0, next_run_at advanced. Keeps lastError/lastRunAt for visibility. */
  reanchorRecurring(id: string, nextRunAt: number): void
  /** Point a recurring job at a new cron and re-arm it: cron updated, status=pending, attempt=0, next_run_at advanced. */
  updateRecurringSchedule(id: string, cron: string, nextRunAt: number): void
  /** Update a recurring job definition without losing run history. */
  updateRecurringJob(input: UpdateRecurringJobInput): ScheduledJob | null
  /** Update a one-shot job definition without losing run history. */
  updateOneShotJob(input: UpdateOneShotJobInput): ScheduledJob | null
  /** Update only the structured metadata payload. */
  updateMetadata(id: string, metadata: Record<string, unknown> | null): ScheduledJob | null
  cancel(id: string): void
  delete(id: string): void
  setEnabled(id: string, enabled: boolean): void
  /**
   * Mark a job as running without a human present, so its tool calls skip
   * approval prompts nobody would answer. Persisted, unlike the grant itself.
   */
  setUnattended(id: string, unattended: boolean): void
  /** Claim an enabled pending job for execution. Returns false when paused or already claimed. */
  tryLockForRun(id: string): boolean
  pickReady(now: number, limit: number): ScheduledJob[]
  gc(olderThanMs: number): number
  cleanupStaleRunning(staleMs: number, activeJobIds?: readonly string[]): number
  // run history
  recordRunStart(jobId: string, attempt: number, startedAt?: number): string
  /** Link one run to its immutable agent evidence journal before execution. */
  linkRunAgentSession(jobId: string, runId: string, agentSessionId: string): boolean
  recordRunFinish(runId: string, status: JobRunStatus, opts?: RecordRunFinishOpts): void
  /** Return one run only when both sides of its durable composite identity match. */
  getRun(jobId: string, runId: string): JobRun | null
  listRuns(jobId: string, limit?: number): JobRun[]
  listRecentRuns(limit?: number): JobRun[]
  gcRuns(olderThanMs: number): number
  getSchedulerDailyUsage(dateKey: string): SchedulerDailyUsage
  addSchedulerDailyUsage(dateKey: string, usage: TokenUsage): SchedulerDailyUsage
  markSchedulerDailyBudgetNotified(dateKey: string, notifiedAt?: number): boolean
  touchCreatedAtForTesting(id: string, ts: number): void
  touchUpdatedAtForTesting(id: string, ts: number): void
}

interface Row {
  id: string
  name: string
  kind: JobKind
  cron: string | null
  run_at: number | null
  next_run_at: number
  last_run_at: number | null
  timezone: string | null
  instruction: string
  channel_type: string | null
  channel_target: string | null
  reply_to_message_id: string | null
  parent_session_id: string | null
  enabled: number
  status: JobStatus
  attempt: number
  max_attempts: number
  retry_backoff_ms: number
  last_error: string | null
  consecutive_failures: number
  metadata_json: string | null
  created_at: number
  updated_at: number
  created_by: JobCreatedBy
  unattended: number
}

interface RunRow {
  id: string
  job_id: string
  agent_session_id: string | null
  started_at: number
  finished_at: number | null
  status: JobRunStatus
  attempt: number
  duration_ms: number | null
  error: string | null
  output_excerpt: string | null
  stop_kind: string | null
  stop_code: string | null
  stop_resumable: number | null
}

interface DailyUsageRow {
  date_key: string
  input_tokens: number
  output_tokens: number
  budget_notified_at: number | null
}

const SELECT_COLS =
  'id, name, kind, cron, run_at, next_run_at, last_run_at, timezone, instruction, channel_type, channel_target, reply_to_message_id, parent_session_id, enabled, status, attempt, max_attempts, retry_backoff_ms, last_error, consecutive_failures, metadata_json, created_at, updated_at, created_by, unattended'

const RUN_SELECT_COLS =
  'id, job_id, agent_session_id, started_at, finished_at, status, attempt, duration_ms, error, output_excerpt,'
  + ' stop_kind, stop_code, stop_resumable'

const toJob = (r: Row): ScheduledJob => ({
  id: r.id,
  name: r.name,
  kind: r.kind,
  cron: r.cron,
  runAt: r.run_at,
  nextRunAt: r.next_run_at,
  lastRunAt: r.last_run_at,
  timezone: r.timezone,
  instruction: r.instruction,
  channelType: r.channel_type,
  channelTarget: r.channel_target,
  replyToMessageId: r.reply_to_message_id,
  parentSessionId: r.parent_session_id,
  enabled: !!r.enabled,
  unattended: !!r.unattended,
  status: r.status,
  attempt: r.attempt,
  maxAttempts: r.max_attempts,
  retryBackoffMs: r.retry_backoff_ms,
  lastError: r.last_error,
  consecutiveFailures: r.consecutive_failures,
  metadata: parseMetadata(r.metadata_json),
  createdAt: r.created_at,
  updatedAt: r.updated_at,
  createdBy: r.created_by,
})

function parseMetadata(value: string | null): Record<string, unknown> | null {
  if (!value) return null
  try {
    const parsed = JSON.parse(value) as unknown
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : null
  } catch {
    return null
  }
}

function stringifyMetadata(value: Record<string, unknown> | null | undefined): string | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  return JSON.stringify(value)
}

const toRun = (r: RunRow): JobRun => {
  const outcome = projectStoredJobRunOutcome({
    status: r.status,
    error: r.error,
    outputExcerpt: r.output_excerpt,
    stopKind: (r.stop_kind as RunStopKind | null) ?? null,
  })
  return {
    id: r.id,
    jobId: r.job_id,
    agentSessionId: r.agent_session_id,
    startedAt: r.started_at,
    finishedAt: r.finished_at,
    status: r.status,
    ...outcome,
    attempt: r.attempt,
    durationMs: r.duration_ms,
    error: r.error,
    outputExcerpt: r.output_excerpt,
    ...(r.stop_kind
      ? {
          stopKind: r.stop_kind as RunStopKind,
          stopCode: r.stop_code,
          stopResumable: r.stop_resumable === 1,
        }
      : {}),
  }
}

const toSchedulerDailyUsage = (r: DailyUsageRow): SchedulerDailyUsage => ({
  dateKey: r.date_key,
  inputTokens: r.input_tokens,
  outputTokens: r.output_tokens,
  budgetNotifiedAt: r.budget_notified_at,
})

const clampExcerpt = (text: string | null | undefined): string | null => {
  if (text == null) return null
  const t = String(text)
  return t.length > OUTPUT_EXCERPT_MAX ? t.slice(0, OUTPUT_EXCERPT_MAX) + '…' : t
}

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS scheduler_jobs (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      cron TEXT,
      command TEXT NOT NULL DEFAULT '',
      last_run_at INTEGER,
      next_run_at INTEGER NOT NULL,
      enabled INTEGER NOT NULL DEFAULT 1
    )`,
  ).run()

  const cols = new Set(
    (db.prepare(`PRAGMA table_info(scheduler_jobs)`).all() as Array<{ name: string }>).map(c => c.name),
  )
  const addColumn = (name: string, ddl: string) => {
    if (!cols.has(name)) db.prepare(`ALTER TABLE scheduler_jobs ADD COLUMN ${ddl}`).run()
  }
  addColumn('kind', `kind TEXT NOT NULL DEFAULT 'recurring'`)
  addColumn('run_at', `run_at INTEGER`)
  addColumn('timezone', `timezone TEXT`)
  addColumn('instruction', `instruction TEXT NOT NULL DEFAULT ''`)
  addColumn('channel_type', `channel_type TEXT`)
  addColumn('channel_target', `channel_target TEXT`)
  addColumn('reply_to_message_id', `reply_to_message_id TEXT`)
  addColumn('parent_session_id', `parent_session_id TEXT`)
  addColumn('status', `status TEXT NOT NULL DEFAULT 'pending'`)
  addColumn('attempt', `attempt INTEGER NOT NULL DEFAULT 0`)
  addColumn('max_attempts', `max_attempts INTEGER NOT NULL DEFAULT ${DEFAULT_MAX_ATTEMPTS}`)
  addColumn('retry_backoff_ms', `retry_backoff_ms INTEGER NOT NULL DEFAULT ${DEFAULT_RETRY_BACKOFF_MS}`)
  addColumn('last_error', `last_error TEXT`)
  addColumn('consecutive_failures', `consecutive_failures INTEGER NOT NULL DEFAULT 0`)
  addColumn('metadata_json', `metadata_json TEXT`)
  addColumn('created_at', `created_at INTEGER NOT NULL DEFAULT 0`)
  addColumn('updated_at', `updated_at INTEGER NOT NULL DEFAULT 0`)
  addColumn('created_by', `created_by TEXT NOT NULL DEFAULT 'internal'`)
  // Persisted so an unattended job survives a daemon restart: the in-memory
  // approval grant does not, and a job outlives the process that created it.
  addColumn('unattended', `unattended INTEGER NOT NULL DEFAULT 0`)

  db.prepare(
    `UPDATE scheduler_jobs SET instruction = command WHERE instruction = '' AND command IS NOT NULL AND command != ''`,
  ).run()
  db.prepare(
    `UPDATE scheduler_jobs SET created_at = unixepoch() * 1000 WHERE created_at = 0`,
  ).run()
  db.prepare(
    `UPDATE scheduler_jobs SET updated_at = created_at WHERE updated_at = 0`,
  ).run()

  relaxLegacyCronNotNull(db)

  db.prepare(`CREATE INDEX IF NOT EXISTS idx_scheduler_jobs_next_run ON scheduler_jobs(next_run_at)`).run()
  db.prepare(`CREATE INDEX IF NOT EXISTS idx_scheduler_jobs_status_updated ON scheduler_jobs(status, updated_at)`).run()

  db.prepare(
    `CREATE TABLE IF NOT EXISTS scheduler_job_runs (
      id TEXT PRIMARY KEY,
      job_id TEXT NOT NULL,
      started_at INTEGER NOT NULL,
      finished_at INTEGER,
      status TEXT NOT NULL,
      attempt INTEGER NOT NULL DEFAULT 0,
      duration_ms INTEGER,
      error TEXT,
      output_excerpt TEXT,
      agent_session_id TEXT,
      stop_kind TEXT,
      stop_code TEXT,
      stop_resumable INTEGER
    )`,
  ).run()
  const runCols = new Set(
    (db.prepare(`PRAGMA table_info(scheduler_job_runs)`).all() as Array<{ name: string }>).map(c => c.name),
  )
  if (!runCols.has('agent_session_id')) {
    db.prepare(`ALTER TABLE scheduler_job_runs ADD COLUMN agent_session_id TEXT`).run()
  }
  // Structured termination cause. Rows written before this stay null and fall
  // back to the error-text probe in projectStoredJobRunOutcome.
  if (!runCols.has('stop_kind')) {
    db.prepare(`ALTER TABLE scheduler_job_runs ADD COLUMN stop_kind TEXT`).run()
  }
  if (!runCols.has('stop_code')) {
    db.prepare(`ALTER TABLE scheduler_job_runs ADD COLUMN stop_code TEXT`).run()
  }
  if (!runCols.has('stop_resumable')) {
    db.prepare(`ALTER TABLE scheduler_job_runs ADD COLUMN stop_resumable INTEGER`).run()
  }
  db.prepare(
    `CREATE INDEX IF NOT EXISTS idx_scheduler_job_runs_job ON scheduler_job_runs(job_id, started_at DESC)`,
  ).run()

  db.prepare(
    `CREATE TABLE IF NOT EXISTS scheduler_daily_usage (
      date_key TEXT PRIMARY KEY,
      input_tokens INTEGER NOT NULL DEFAULT 0,
      output_tokens INTEGER NOT NULL DEFAULT 0,
      budget_notified_at INTEGER,
      updated_at INTEGER NOT NULL
    )`,
  ).run()
}

// Columns in the canonical `scheduler_jobs` shape, used when rebuilding a
// legacy table. Order is irrelevant — the copy lists names on both sides.
const SCHEDULER_JOBS_COLUMNS = [
  'id', 'name', 'cron', 'command', 'last_run_at', 'next_run_at', 'enabled',
  'kind', 'run_at', 'timezone', 'instruction', 'channel_type', 'channel_target',
  'reply_to_message_id', 'parent_session_id', 'status', 'attempt', 'max_attempts',
  'retry_backoff_ms', 'last_error', 'consecutive_failures', 'metadata_json', 'created_at', 'updated_at', 'created_by',
  'unattended',
] as const

/**
 * Early `scheduler_jobs` revisions declared `cron TEXT NOT NULL` (every job was
 * a cron job). One-shot jobs pass `cron = null`, which the stale constraint
 * rejects — and `CREATE TABLE IF NOT EXISTS` never rewrites an existing table,
 * so the daemon was stuck failing every `schedule_create "in 3 minutes"` with
 * `NOT NULL constraint failed: scheduler_jobs.cron`. SQLite can't drop a column
 * constraint in place, so rebuild the table when we detect the legacy shape.
 *
 * Must run after the `ADD COLUMN` migrations above, so every canonical column
 * exists on the legacy table before we copy it across.
 */
function relaxLegacyCronNotNull(db: SqliteDatabase): void {
  const cronCol = (db.prepare(`PRAGMA table_info(scheduler_jobs)`).all() as Array<{
    name: string
    notnull: number
  }>).find((c) => c.name === 'cron')
  if (!cronCol || cronCol.notnull === 0) return

  const existing = new Set(
    (db.prepare(`PRAGMA table_info(scheduler_jobs)`).all() as Array<{ name: string }>).map((c) => c.name),
  )
  const copyCols = SCHEDULER_JOBS_COLUMNS.filter((c) => existing.has(c))
  const colList = copyCols.join(', ')

  const rebuild = db.transaction(() => {
    db.prepare(`DROP TABLE IF EXISTS scheduler_jobs__rebuild`).run()
    db.prepare(
      `CREATE TABLE scheduler_jobs__rebuild (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        cron TEXT,
        command TEXT NOT NULL DEFAULT '',
        last_run_at INTEGER,
        next_run_at INTEGER NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        kind TEXT NOT NULL DEFAULT 'recurring',
        run_at INTEGER,
        timezone TEXT,
        instruction TEXT NOT NULL DEFAULT '',
        channel_type TEXT,
        channel_target TEXT,
        reply_to_message_id TEXT,
        parent_session_id TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        attempt INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL DEFAULT ${DEFAULT_MAX_ATTEMPTS},
        retry_backoff_ms INTEGER NOT NULL DEFAULT ${DEFAULT_RETRY_BACKOFF_MS},
        last_error TEXT,
        consecutive_failures INTEGER NOT NULL DEFAULT 0,
        metadata_json TEXT,
        created_at INTEGER NOT NULL DEFAULT 0,
        updated_at INTEGER NOT NULL DEFAULT 0,
        created_by TEXT NOT NULL DEFAULT 'internal',
        unattended INTEGER NOT NULL DEFAULT 0
      )`,
    ).run()
    db.prepare(`INSERT INTO scheduler_jobs__rebuild (${colList}) SELECT ${colList} FROM scheduler_jobs`).run()
    db.prepare(`DROP TABLE scheduler_jobs`).run()
    db.prepare(`ALTER TABLE scheduler_jobs__rebuild RENAME TO scheduler_jobs`).run()
  })
  rebuild()
}

export function createJobStore(): JobStore {
  const db = openDomainDb({ name: 'scheduler' })
  ensureSchema(db)

  const insertStmt = db.prepare(
    `INSERT INTO scheduler_jobs
       (id, name, kind, cron, run_at, next_run_at, last_run_at, timezone, instruction,
        channel_type, channel_target, reply_to_message_id, parent_session_id,
        enabled, status, attempt, max_attempts, retry_backoff_ms, last_error, metadata_json,
        created_at, updated_at, created_by, command, unattended)
     VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, NULL, ?, ?, ?, ?, '', ?)`,
  )

  const insertRunStmt = db.prepare(
    `INSERT INTO scheduler_job_runs (id, job_id, agent_session_id, started_at, finished_at, status, attempt, duration_ms, error, output_excerpt)
     VALUES (?, ?, NULL, ?, NULL, 'running', ?, NULL, NULL, NULL)`,
  )

  return {
    list(filter) {
      const where: string[] = []
      const params: unknown[] = []
      if (filter?.status?.length) {
        where.push(`status IN (${filter.status.map(() => '?').join(',')})`)
        params.push(...filter.status)
      }
      if (filter?.channelTarget) {
        where.push('channel_target = ?')
        params.push(filter.channelTarget)
      }
      const sql = `SELECT ${SELECT_COLS} FROM scheduler_jobs ${
        where.length ? 'WHERE ' + where.join(' AND ') : ''
      } ORDER BY next_run_at`
      return (db.prepare(sql).all(...params) as Row[]).map(toJob)
    },
    get(id) {
      const row = db.prepare(`SELECT ${SELECT_COLS} FROM scheduler_jobs WHERE id = ?`).get(id) as Row | undefined
      return row ? toJob(row) : null
    },
    create(input) {
      const id = input.id ?? randomUUID()
      const now = Date.now()
      insertStmt.run(
        id, input.name, input.kind, input.cron, input.runAt, input.nextRunAt,
        input.timezone ?? null, input.instruction, input.channelType, input.channelTarget,
        input.replyToMessageId, input.parentSessionId,
        input.enabled ? 1 : 0,
        input.maxAttempts ?? DEFAULT_MAX_ATTEMPTS,
        input.retryBackoffMs ?? DEFAULT_RETRY_BACKOFF_MS,
        stringifyMetadata(input.metadata),
        now, now, input.createdBy,
        input.unattended ? 1 : 0,
      )
      const row = db.prepare(`SELECT ${SELECT_COLS} FROM scheduler_jobs WHERE id = ?`).get(id) as Row
      return toJob(row)
    },
    updateStatus(id, status, opts) {
      const now = Date.now()
      const disable = status === 'completed' || status === 'cancelled' ? ', enabled = 0' : ''
      const lastError = status === 'failed' ? (opts?.lastError ?? null) : null
      const resetFailures = status === 'completed' || status === 'cancelled' ? ', consecutive_failures = 0' : ''
      db.prepare(
        `UPDATE scheduler_jobs SET status = ?, last_run_at = COALESCE(?, last_run_at), last_error = ?, updated_at = ? ${disable}${resetFailures} WHERE id = ?`,
      ).run(status, opts?.lastRunAt ?? null, lastError, now, id)
    },
    updateNextRun(id, nextRunAt, lastRunAt) {
      db.prepare(
        `UPDATE scheduler_jobs SET next_run_at = ?, last_run_at = ?, status = 'pending', attempt = 0, last_error = NULL, consecutive_failures = 0, updated_at = ? WHERE id = ?`,
      ).run(nextRunAt, lastRunAt, Date.now(), id)
    },
    rescheduleAfterFailure(id, nextRunAt, lastRunAt, lastError) {
      db.prepare(
        `UPDATE scheduler_jobs SET next_run_at = ?, last_run_at = ?, status = 'pending', attempt = 0, last_error = ?, consecutive_failures = consecutive_failures + 1, updated_at = ? WHERE id = ?`,
      ).run(nextRunAt, lastRunAt, lastError, Date.now(), id)
    },
    markRetry(id, nextRunAt, attempt, lastError) {
      db.prepare(
        `UPDATE scheduler_jobs SET next_run_at = ?, status = 'pending', attempt = ?, last_error = ?, updated_at = ? WHERE id = ?`,
      ).run(nextRunAt, attempt, lastError, Date.now(), id)
    },
    releaseInterruptedRun(jobId, runId, retryAt, reason, finishedAt) {
      const completedAt = finishedAt ?? Date.now()
      const tx = db.transaction(() => {
        const run = db.prepare(
          `UPDATE scheduler_job_runs
           SET status = 'failed', finished_at = ?, duration_ms = MAX(0, ? - started_at), error = ?, output_excerpt = NULL
           WHERE id = ? AND job_id = ? AND status = 'running'`,
        ).run(completedAt, completedAt, reason, runId, jobId)
        if (run.changes !== 1) return false

        // A concurrent pause leaves enabled=0 while status is still running;
        // releasing the claim must preserve that authority. A concurrent
        // cancel/terminal edit has already changed status, so leave it intact.
        db.prepare(
          `UPDATE scheduler_jobs
           SET next_run_at = ?, status = 'pending', updated_at = ?
           WHERE id = ? AND status = 'running'`,
        ).run(retryAt, completedAt, jobId)
        return true
      })
      return tx()
    },
    reanchorRecurring(id, nextRunAt) {
      db.prepare(
        `UPDATE scheduler_jobs SET next_run_at = ?, status = 'pending', attempt = 0, updated_at = ? WHERE id = ? AND kind = 'recurring'`,
      ).run(nextRunAt, Date.now(), id)
    },
    updateRecurringSchedule(id, cron, nextRunAt) {
      db.prepare(
        `UPDATE scheduler_jobs SET cron = ?, next_run_at = ?, status = 'pending', attempt = 0, consecutive_failures = 0, updated_at = ? WHERE id = ? AND kind = 'recurring'`,
      ).run(cron, nextRunAt, Date.now(), id)
    },
    updateRecurringJob(input) {
      const now = Date.now()
      const existing = db.prepare(
        `SELECT ${SELECT_COLS} FROM scheduler_jobs WHERE id = ? AND kind = 'recurring'`,
      ).get(input.id) as Row | undefined
      if (!existing) return null
      const status = input.preserveRunState ? existing.status : 'pending'
      const attempt = input.preserveRunState ? existing.attempt : 0
      const lastError = input.preserveRunState ? existing.last_error : null
      const consecutiveFailures = input.preserveRunState ? existing.consecutive_failures : 0

      db.prepare(
        `UPDATE scheduler_jobs
         SET name = ?,
             cron = ?,
             run_at = NULL,
             next_run_at = ?,
             timezone = ?,
             instruction = ?,
             channel_type = ?,
             channel_target = ?,
             reply_to_message_id = ?,
             parent_session_id = ?,
             enabled = ?,
             status = ?,
             attempt = ?,
             max_attempts = ?,
             retry_backoff_ms = ?,
             last_error = ?,
             consecutive_failures = ?,
             metadata_json = ?,
             updated_at = ?
         WHERE id = ? AND kind = 'recurring'`,
      ).run(
        input.name,
        input.cron,
        input.nextRunAt,
        input.timezone,
        input.instruction,
        input.channelType === undefined ? existing.channel_type : input.channelType,
        input.channelTarget === undefined ? existing.channel_target : input.channelTarget,
        input.replyToMessageId === undefined ? existing.reply_to_message_id : input.replyToMessageId,
        input.parentSessionId === undefined ? existing.parent_session_id : input.parentSessionId,
        input.enabled ? 1 : 0,
        status,
        attempt,
        input.maxAttempts ?? existing.max_attempts,
        input.retryBackoffMs ?? existing.retry_backoff_ms,
        lastError,
        consecutiveFailures,
        input.metadata === undefined ? existing.metadata_json : stringifyMetadata(input.metadata),
        now,
        input.id,
      )
      const row = db.prepare(`SELECT ${SELECT_COLS} FROM scheduler_jobs WHERE id = ?`).get(input.id) as Row
      return toJob(row)
    },
    updateOneShotJob(input) {
      const now = Date.now()
      const existing = db.prepare(
        `SELECT ${SELECT_COLS} FROM scheduler_jobs WHERE id = ? AND kind = 'oneshot'`,
      ).get(input.id) as Row | undefined
      if (!existing) return null
      const status = input.preserveRunState ? existing.status : 'pending'
      const attempt = input.preserveRunState ? existing.attempt : 0
      const lastError = input.preserveRunState ? existing.last_error : null
      const consecutiveFailures = input.preserveRunState ? existing.consecutive_failures : 0

      db.prepare(
        `UPDATE scheduler_jobs
         SET name = ?,
             kind = 'oneshot',
             cron = NULL,
             run_at = ?,
             next_run_at = ?,
             timezone = NULL,
             instruction = ?,
             channel_type = ?,
             channel_target = ?,
             reply_to_message_id = ?,
             parent_session_id = ?,
             enabled = ?,
             status = ?,
             attempt = ?,
             max_attempts = ?,
             retry_backoff_ms = ?,
             last_error = ?,
             consecutive_failures = ?,
             metadata_json = ?,
             updated_at = ?
         WHERE id = ? AND kind = 'oneshot'`,
      ).run(
        input.name,
        input.runAt,
        input.nextRunAt,
        input.instruction,
        input.channelType === undefined ? existing.channel_type : input.channelType,
        input.channelTarget === undefined ? existing.channel_target : input.channelTarget,
        input.replyToMessageId === undefined ? existing.reply_to_message_id : input.replyToMessageId,
        input.parentSessionId === undefined ? existing.parent_session_id : input.parentSessionId,
        input.enabled ? 1 : 0,
        status,
        attempt,
        input.maxAttempts ?? existing.max_attempts,
        input.retryBackoffMs ?? existing.retry_backoff_ms,
        lastError,
        consecutiveFailures,
        input.metadata === undefined ? existing.metadata_json : stringifyMetadata(input.metadata),
        now,
        input.id,
      )
      const row = db.prepare(`SELECT ${SELECT_COLS} FROM scheduler_jobs WHERE id = ?`).get(input.id) as Row
      return toJob(row)
    },
    updateMetadata(id, metadata) {
      db.prepare(
        `UPDATE scheduler_jobs SET metadata_json = ?, updated_at = ? WHERE id = ?`,
      ).run(stringifyMetadata(metadata), Date.now(), id)
      const row = db.prepare(`SELECT ${SELECT_COLS} FROM scheduler_jobs WHERE id = ?`).get(id) as Row | undefined
      return row ? toJob(row) : null
    },
    cancel(id) {
      db.prepare(
        `UPDATE scheduler_jobs SET status = 'cancelled', enabled = 0, consecutive_failures = 0, updated_at = ? WHERE id = ?`,
      ).run(Date.now(), id)
    },
    delete(id) {
      const tx = db.transaction((jobId: string) => {
        db.prepare(`DELETE FROM scheduler_job_runs WHERE job_id = ?`).run(jobId)
        db.prepare(`DELETE FROM scheduler_jobs WHERE id = ?`).run(jobId)
      })
      tx(id)
    },
    setEnabled(id, enabled) {
      db.prepare(`UPDATE scheduler_jobs SET enabled = ?, updated_at = ? WHERE id = ?`).run(
        enabled ? 1 : 0, Date.now(), id,
      )
    },
    setUnattended(id, unattended) {
      db.prepare(`UPDATE scheduler_jobs SET unattended = ?, updated_at = ? WHERE id = ?`).run(
        unattended ? 1 : 0, Date.now(), id,
      )
    },
    tryLockForRun(id) {
      const result = db.prepare(
        `UPDATE scheduler_jobs SET status = 'running', updated_at = ? WHERE id = ? AND status = 'pending' AND enabled = 1
         AND NOT EXISTS (SELECT 1 FROM scheduler_job_runs r WHERE r.job_id = scheduler_jobs.id AND r.status = 'running')`,
      ).run(Date.now(), id)
      return result.changes === 1
    },
    pickReady(now, limit) {
      const tx = db.transaction((cutoff: number, lim: number) => {
        const rows = db
          .prepare(
            `SELECT ${SELECT_COLS} FROM scheduler_jobs
             WHERE enabled = 1 AND status = 'pending' AND next_run_at <= ?
             AND NOT EXISTS (SELECT 1 FROM scheduler_job_runs r WHERE r.job_id = scheduler_jobs.id AND r.status = 'running')
             ORDER BY next_run_at LIMIT ?`,
          )
          .all(cutoff, lim) as Row[]
        const update = db.prepare(
          `UPDATE scheduler_jobs SET status = 'running', updated_at = ? WHERE id = ?`,
        )
        const ts = Date.now()
        for (const r of rows) {
          update.run(ts, r.id)
          r.status = 'running'
          r.updated_at = ts
        }
        return rows
      })
      return tx(now, limit).map(toJob)
    },
    gc(olderThanMs) {
      const cutoff = Date.now() - olderThanMs
      const tx = db.transaction((c: number) => {
        const removed = db.prepare(
          `SELECT id FROM scheduler_jobs WHERE status IN ('completed', 'cancelled', 'failed') AND updated_at < ?
           AND NOT EXISTS (SELECT 1 FROM scheduler_job_runs r WHERE r.job_id = scheduler_jobs.id AND r.status = 'running')`,
        ).all(c) as Array<{ id: string }>
        for (const r of removed) {
          db.prepare(`DELETE FROM scheduler_job_runs WHERE job_id = ?`).run(r.id)
        }
        const result = db.prepare(
          `DELETE FROM scheduler_jobs WHERE status IN ('completed', 'cancelled', 'failed') AND updated_at < ?
           AND NOT EXISTS (SELECT 1 FROM scheduler_job_runs r WHERE r.job_id = scheduler_jobs.id AND r.status = 'running')`,
        ).run(c)
        return result.changes
      })
      return tx(cutoff)
    },
    cleanupStaleRunning(staleMs, activeJobIds = []) {
      // Age identifies abandoned work only when no live executor owns the job.
      const placeholders = activeJobIds.map(() => '?').join(', ')
      const excludeRuns = activeJobIds.length ? ` AND job_id NOT IN (${placeholders})` : ''
      const excludeJobs = activeJobIds.length ? ` AND id NOT IN (${placeholders})` : ''
      const cutoff = Date.now() - staleMs
      const now = Date.now()
      const tx = db.transaction(() => {
        db.prepare(
          `UPDATE scheduler_job_runs SET status = 'failed', finished_at = ?, error = 'daemon restarted while job was running'
           WHERE status = 'running' AND started_at < ?${excludeRuns}`,
        ).run(now, cutoff, ...activeJobIds)
        const result = db.prepare(
          `UPDATE scheduler_jobs SET status = 'failed', last_error = 'daemon restarted while job was running', consecutive_failures = consecutive_failures + 1, updated_at = ?
           WHERE status = 'running' AND updated_at < ?${excludeJobs}`,
        ).run(now, cutoff, ...activeJobIds)
        return result.changes
      })
      return tx()
    },
    recordRunStart(jobId, attempt, startedAt) {
      const id = randomUUID()
      insertRunStmt.run(id, jobId, startedAt ?? Date.now(), attempt)
      return id
    },
    linkRunAgentSession(jobId, runId, agentSessionId) {
      const existing = db.prepare(
        `SELECT agent_session_id FROM scheduler_job_runs WHERE job_id = ? AND id = ?`,
      ).get(jobId, runId) as { agent_session_id: string | null } | undefined
      if (!existing) return false
      if (existing.agent_session_id !== null) return existing.agent_session_id === agentSessionId
      const result = db.prepare(
        `UPDATE scheduler_job_runs SET agent_session_id = ? WHERE job_id = ? AND id = ? AND agent_session_id IS NULL`,
      ).run(agentSessionId, jobId, runId)
      return result.changes === 1
    },
    recordRunFinish(runId, status, opts) {
      const finishedAt = opts?.finishedAt ?? Date.now()
      db.prepare(
        `UPDATE scheduler_job_runs SET status = ?, finished_at = ?, duration_ms = ?, error = ?, output_excerpt = ?,
           stop_kind = ?, stop_code = ?, stop_resumable = ? WHERE id = ?`,
      ).run(
        status,
        finishedAt,
        opts?.durationMs ?? null,
        status === 'failed' ? (opts?.error ?? null) : null,
        clampExcerpt(opts?.outputExcerpt),
        opts?.stopReason?.kind ?? null,
        opts?.stopReason?.code ?? null,
        opts?.stopReason ? (opts.stopReason.resumable ? 1 : 0) : null,
        runId,
      )
    },
    getRun(jobId, runId) {
      const row = db.prepare(
        `SELECT ${RUN_SELECT_COLS} FROM scheduler_job_runs WHERE job_id = ? AND id = ?`,
      ).get(jobId, runId) as RunRow | undefined
      return row ? toRun(row) : null
    },
    listRuns(jobId, limit) {
      const lim = Math.max(1, Math.min(limit ?? 20, 200))
      return (
        db.prepare(
          `SELECT ${RUN_SELECT_COLS} FROM scheduler_job_runs WHERE job_id = ? ORDER BY started_at DESC LIMIT ?`,
        ).all(jobId, lim) as RunRow[]
      ).map(toRun)
    },
    listRecentRuns(limit) {
      const lim = Math.max(1, Math.min(limit ?? 50, 500))
      return (
        db.prepare(
          `SELECT ${RUN_SELECT_COLS} FROM scheduler_job_runs ORDER BY started_at DESC LIMIT ?`,
        ).all(lim) as RunRow[]
      ).map(toRun)
    },
    gcRuns(olderThanMs) {
      const cutoff = Date.now() - olderThanMs
      const result = db.prepare(
        `DELETE FROM scheduler_job_runs WHERE finished_at IS NOT NULL AND finished_at < ?`,
      ).run(cutoff)
      return result.changes
    },
    getSchedulerDailyUsage(dateKey) {
      const row = db.prepare(
        `SELECT date_key, input_tokens, output_tokens, budget_notified_at FROM scheduler_daily_usage WHERE date_key = ?`,
      ).get(dateKey) as DailyUsageRow | undefined
      return row
        ? toSchedulerDailyUsage(row)
        : { dateKey, inputTokens: 0, outputTokens: 0, budgetNotifiedAt: null }
    },
    addSchedulerDailyUsage(dateKey, usage) {
      const inputTokens = Math.max(0, Math.floor(usage.inputTokens))
      const outputTokens = Math.max(0, Math.floor(usage.outputTokens))
      const now = Date.now()
      db.prepare(
        `INSERT INTO scheduler_daily_usage (date_key, input_tokens, output_tokens, budget_notified_at, updated_at)
         VALUES (?, ?, ?, NULL, ?)
         ON CONFLICT(date_key) DO UPDATE SET
           input_tokens = input_tokens + excluded.input_tokens,
           output_tokens = output_tokens + excluded.output_tokens,
           updated_at = excluded.updated_at`,
      ).run(dateKey, inputTokens, outputTokens, now)
      return this.getSchedulerDailyUsage(dateKey)
    },
    markSchedulerDailyBudgetNotified(dateKey, notifiedAt) {
      const now = notifiedAt ?? Date.now()
      const result = db.prepare(
        `INSERT INTO scheduler_daily_usage (date_key, input_tokens, output_tokens, budget_notified_at, updated_at)
         VALUES (?, 0, 0, ?, ?)
         ON CONFLICT(date_key) DO UPDATE SET
           budget_notified_at = COALESCE(budget_notified_at, excluded.budget_notified_at),
           updated_at = excluded.updated_at
         WHERE budget_notified_at IS NULL`,
      ).run(dateKey, now, now)
      return result.changes > 0
    },
    touchCreatedAtForTesting(id, ts) {
      db.prepare(`UPDATE scheduler_jobs SET created_at = ?, updated_at = ? WHERE id = ?`).run(ts, ts, id)
    },
    touchUpdatedAtForTesting(id, ts) {
      db.prepare(`UPDATE scheduler_jobs SET updated_at = ? WHERE id = ?`).run(ts, id)
    },
  }
}
