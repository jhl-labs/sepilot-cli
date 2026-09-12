import type { Timestamp, DeviceId } from '../types/common.js'
import type { PaginationParams, PaginatedResult } from '../types/pagination.js'

export interface JobInput { workflow: string; inputs: Record<string, unknown>; targetDevice?: DeviceId; timeoutSec?: number }
export interface JobHandle { jobId: string; statusUrl?: string }
export type JobStatusValue = 'queued' | 'in_progress' | 'completed' | 'failed' | 'cancelled'
export interface JobStatus { jobId: string; status: JobStatusValue; progress?: number; currentStep?: string; startedAt?: Timestamp; completedAt?: Timestamp }
export interface JobResult { jobId: string; status: 'completed' | 'failed'; output?: Record<string, unknown>; logs?: string; artifacts?: Artifact[]; duration: number }
export interface Artifact { name: string; url: string; size: number; contentType: string }
export interface Job { jobId: string; workflow: string; status: JobStatusValue; targetDevice?: DeviceId; createdAt: Timestamp; completedAt?: Timestamp }
export interface JobFilter { status?: JobStatusValue[]; workflow?: string; targetDevice?: DeviceId; pagination?: PaginationParams }

export interface IComputeService {
  dispatch(job: JobInput): Promise<JobHandle>
  getJobStatus(jobId: string): Promise<JobStatus>
  getJobResult(jobId: string): Promise<JobResult>
  cancelJob(jobId: string): Promise<void>
  listJobs(filter?: JobFilter): Promise<PaginatedResult<Job>>
}
