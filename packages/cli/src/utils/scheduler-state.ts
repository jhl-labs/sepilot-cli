import type { DaemonSchedulerJob } from '@sepilotd/api-client'

export function hasPendingScheduledRun(
  job: Pick<DaemonSchedulerJob, 'enabled' | 'nextRunAt' | 'status'>,
): boolean {
  return job.enabled
    && job.status === 'pending'
    && Number.isFinite(job.nextRunAt)
}
