export type JobKind = 'batch' | 'migration' | 'subagent' | 'meeting_voice' | 'hook'
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'canceled'
export type ItemStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled'

export interface JobProgress {
  sessionId: string
  phase: string
  toolName?: string
  approvalRequestId?: string
  updatedAt: number
}

export interface Job {
  id: string
  kind: JobKind
  status: JobStatus
  total: number
  succeeded: number
  failed: number
  canceled: number
  concurrency: number
  createdAt: number
  startedAt: number | null
  finishedAt: number | null
  error: string | null
}

export interface JobItem {
  progress?: JobProgress | null
  jobId: string
  idx: number
  status: ItemStatus
  request: unknown
  result: unknown | null
  error: string | null
  sessionId: string | null
  attempts: number
  maxAttempts: number
  createdAt: number
  startedAt: number | null
  finishedAt: number | null
}

export interface PartitionedQueueStatus {
  sessionId: string
  queued: number
  running: number
  succeeded: number
  failed: number
  total: number
  position: number | null
  items: Array<{
    itemId: string
    jobId: string
    idx: number
    status: ItemStatus
    attempts: number
    maxAttempts: number
    error: string | null
    createdAt: number
    startedAt: number | null
    finishedAt: number | null
  }>
}
