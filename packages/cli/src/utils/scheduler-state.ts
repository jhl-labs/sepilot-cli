import type { DaemonSchedulerJob } from '@sepilotd/api-client'

/**
 * A persisted nextRunAt is an audit/reschedule anchor on terminal and paused
 * jobs. Only an enabled pending job currently owns a future scheduler claim.
 */
export function hasPendingScheduledRun(
  job: Pick<DaemonSchedulerJob, 'enabled' | 'nextRunAt' | 'status'>,
): boolean {
  return job.enabled
    && job.status === 'pending'
    && Number.isFinite(job.nextRunAt)
}
