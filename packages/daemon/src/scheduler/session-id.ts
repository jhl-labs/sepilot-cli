/**
 * The session a scheduled job runs under.
 *
 * Both the scheduler executor (which runs the job) and `schedule_create`
 * (which pre-authorises unattended jobs) have to name the same session, or the
 * approval is granted to a session that never runs. Keep the one definition.
 */
export function schedulerSessionIdForJob(jobId: string): string {
  return `scheduler-${jobId}`
}

/**
 * The durable audit session for one persisted scheduler run.
 *
 * Approval authority remains on the stable per-job session above, while
 * execution state, tool evidence, and integration read/write receipts are
 * isolated per run. Run ids are daemon-generated UUIDs, so this identity is
 * both filesystem-safe for the JSONL session store and collision-resistant.
 */
export function schedulerSessionIdForRun(runId: string): string {
  return `scheduler-run-${runId}`
}
