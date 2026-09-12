import type { RunStopKind } from '../agent/types.js'

export type JobKind = 'oneshot' | 'recurring'

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export type JobRunStatus = 'running' | 'success' | 'failed'

/** Task-level interpretation of a run, independent of its legacy stored status. */
export type JobRunTaskOutcome = 'running' | 'complete' | 'incomplete' | 'failed'

/** Whether the stored run status agrees with the current task-outcome contract. */
export type JobRunStatusIntegrity = 'consistent' | 'legacy-incomplete-conflict'

/**
 * Options for a real manual scheduler execution.
 *
 * suppressDelivery prevents only scheduler-owned progress, parent-session,
 * configured-channel/outbox, and notification/Relay delivery. The agent still
 * runs with the job's normal tool authority and the job/run state still
 * advances exactly like any other manual execution.
 */
export interface ManualJobRunOptions {
  suppressDelivery?: boolean
  /**
   * Wait for the terminal run result. Set false to return the persisted running
   * identity immediately and observe completion through run history.
   * Defaults to true for compatibility.
   */
  waitForCompletion?: boolean
}

/**
 * Terminal evidence returned by a synchronous manual scheduler trigger.
 * A rejected trigger has no run identity; every started trigger returns the
 * canonical persisted run id and its terminal status.
 */
export type ManualJobRunResult =
  | {
      started: false
      jobId: string
      runId: null
      status: null
    }
  | {
      started: true
      jobId: string
      runId: string
      status: 'running' | 'success' | 'failed'
      /** Present only when scheduler-owned delivery was explicitly suppressed. */
      deliverySuppressed?: true
    }

export type JobCreatedBy = 'agent' | 'rest' | 'cli' | 'channel-cmd' | 'internal'

/** Retry behavior applied to a failed execution of a scheduled job. */
export interface RetryPolicy {
  /** Total attempts allowed per scheduled fire. 1 means no retry. */
  maxAttempts: number
  /** Base backoff in ms. Attempt N (1-indexed) waits backoffMs * 2^(N-1), capped at maxBackoffMs. */
  backoffMs: number
  /** Upper bound for a single backoff wait. */
  maxBackoffMs: number
}

export const DEFAULT_RETRY_POLICY: RetryPolicy = {
  maxAttempts: 1,
  backoffMs: 30_000,
  maxBackoffMs: 3_600_000,
}

/** A persisted scheduled job. */
export interface ScheduledJob {
  id: string
  name: string
  kind: JobKind
  /**
   * Schedule expression for recurring jobs. Supports 5-field cron,
   * cron nicknames (`@daily`, `@hourly`, …) and interval form `@every 30s`.
   * null for oneshot jobs.
   */
  cron: string | null
  /** Absolute fire time (epoch ms) for oneshot jobs; null for recurring. */
  runAt: number | null
  /** Next fire time (epoch ms). Recomputed after every run for recurring jobs. */
  nextRunAt: number
  /** Time the job last started executing (epoch ms); null if it never ran. */
  lastRunAt: number | null
  /** IANA timezone used to interpret the cron expression (e.g. "Asia/Seoul"); null = daemon default. */
  timezone: string | null
  /** Instruction handed to the agent, empty for metadata-defined script monitors, or `__internal:<kind>` for built-in jobs. */
  instruction: string
  channelType: string | null
  channelTarget: string | null
  replyToMessageId: string | null
  parentSessionId: string | null
  enabled: boolean
  status: JobStatus
  /** Retry attempt for the in-progress fire. 0 = first attempt. */
  attempt: number
  /**
   * Pre-approve the tool calls this job's runs make, scoped to this job.
   * Persisted because the approval grant itself is in-memory and a job
   * outlives the daemon process that created it.
   */
  unattended: boolean
  /** Total attempts allowed per fire (mirrors RetryPolicy.maxAttempts). */
  maxAttempts: number
  /** Base retry backoff in ms (mirrors RetryPolicy.backoffMs). */
  retryBackoffMs: number
  /** Message from the most recent failure, if any. */
  lastError: string | null
  /** Consecutive scheduled-fire failures since the last successful run. */
  consecutiveFailures: number
  /** App/domain-owned structured context used by surfaces. Never used for secrets. */
  metadata: Record<string, unknown> | null
  /** Epoch ms. */
  createdAt: number
  /** Epoch ms. */
  updatedAt: number
  createdBy: JobCreatedBy
}

/** One execution record of a scheduled job. */
export interface JobRun {
  id: string
  jobId: string
  /**
   * Durable session journal containing this run's agent/tool evidence.
   * null means the run did not invoke an agent (for example an internal
   * maintenance job) or predates scheduler audit-session linkage. Optional
   * for compatibility with older daemons.
   */
  agentSessionId?: string | null
  startedAt: number
  finishedAt: number | null
  status: JobRunStatus
  /** Derived by current daemons; optional for compatibility with older servers. */
  taskOutcome?: JobRunTaskOutcome
  /**
   * Structured termination cause recorded by current daemons. Absent on runs
   * written before it was captured, and on runs that never invoked an agent.
   */
  stopKind?: RunStopKind
  stopCode?: string | null
  /** Whether the run retained resumable state when it stopped. */
  stopResumable?: boolean
  /** Flags legacy records whose success status conflicts with a structural incomplete result. */
  statusIntegrity?: JobRunStatusIntegrity
  attempt: number
  durationMs: number | null
  error: string | null
  /** Truncated excerpt of the agent's output text, for quick inspection. */
  outputExcerpt: string | null
}

/** Input for creating a scheduled job via REST/CLI. */
export interface CreateScheduledJobInput {
  /** Natural-language ("매일 오전 9시", "in 2 minutes") or a cron/interval expression. */
  when: string
  instruction: string
  name?: string
  timezone?: string
  maxAttempts?: number
  retryBackoffMs?: number
  /** Explicit standing authorization for future headless runs. */
  unattended?: boolean
  channelType?: string
  channelTarget?: string
  replyToMessageId?: string
  parentSessionId?: string | null
  /** Stable skill ids loaded again whenever the scheduled agent executes. */
  skillRefs?: Array<{ name: string }>
  metadata?: Record<string, unknown> | null
}

export interface ScheduledJobListFilter {
  status?: JobStatus[]
  channelTarget?: string
}

/**
 * Compute the next nextRunAt after a failed attempt, or null when retries are exhausted.
 * Pure helper shared by the daemon engine and tests.
 */
export function nextRetryAt(
  now: number,
  attempt: number,
  policy: Pick<RetryPolicy, 'maxAttempts' | 'backoffMs' | 'maxBackoffMs'>,
): number | null {
  if (attempt >= policy.maxAttempts) return null
  const wait = Math.min(policy.backoffMs * 2 ** (attempt - 1), policy.maxBackoffMs)
  return now + Math.max(wait, 0)
}
