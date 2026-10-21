import type { RunStopKind } from '../agent/types.js'

export type JobKind = 'oneshot' | 'recurring'

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export type JobRunStatus = 'running' | 'success' | 'failed'

export type JobRunTaskOutcome = 'running' | 'complete' | 'incomplete' | 'failed'

export type JobRunStatusIntegrity = 'consistent' | 'legacy-incomplete-conflict'

export interface ManualJobRunOptions {
  suppressDelivery?: boolean
  waitForCompletion?: boolean
}

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
      deliverySuppressed?: true
    }

export type JobCreatedBy = 'agent' | 'rest' | 'cli' | 'channel-cmd' | 'internal'

export interface RetryPolicy {
  maxAttempts: number
  backoffMs: number
  maxBackoffMs: number
}

export const DEFAULT_RETRY_POLICY: RetryPolicy = {
  maxAttempts: 1,
  backoffMs: 30_000,
  maxBackoffMs: 3_600_000,
}

export interface ScheduledJob {
  id: string
  name: string
  kind: JobKind
  cron: string | null
  runAt: number | null
  nextRunAt: number
  lastRunAt: number | null
  timezone: string | null
  instruction: string
  channelType: string | null
  channelTarget: string | null
  replyToMessageId: string | null
  parentSessionId: string | null
  enabled: boolean
  status: JobStatus
  attempt: number
  unattended: boolean
  maxAttempts: number
  retryBackoffMs: number
  lastError: string | null
  consecutiveFailures: number
  metadata: Record<string, unknown> | null
  createdAt: number
  updatedAt: number
  createdBy: JobCreatedBy
}

export interface JobRun {
  id: string
  jobId: string
  agentSessionId?: string | null
  startedAt: number
  finishedAt: number | null
  status: JobRunStatus
  taskOutcome?: JobRunTaskOutcome
  stopKind?: RunStopKind
  stopCode?: string | null
  stopResumable?: boolean
  statusIntegrity?: JobRunStatusIntegrity
  attempt: number
  durationMs: number | null
  error: string | null
  outputExcerpt: string | null
}

export interface CreateScheduledJobInput {
  when: string
  instruction: string
  name?: string
  timezone?: string
  maxAttempts?: number
  retryBackoffMs?: number
  unattended?: boolean
  channelType?: string
  channelTarget?: string
  replyToMessageId?: string
  parentSessionId?: string | null
  skillRefs?: Array<{ name: string }>
  metadata?: Record<string, unknown> | null
}

export interface ScheduledJobListFilter {
  status?: JobStatus[]
  channelTarget?: string
}

export function nextRetryAt(
  now: number,
  attempt: number,
  policy: Pick<RetryPolicy, 'maxAttempts' | 'backoffMs' | 'maxBackoffMs'>,
): number | null {
  if (attempt >= policy.maxAttempts) return null
  const wait = Math.min(policy.backoffMs * 2 ** (attempt - 1), policy.maxBackoffMs)
  return now + Math.max(wait, 0)
}
