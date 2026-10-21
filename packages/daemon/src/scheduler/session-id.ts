export function schedulerSessionIdForJob(jobId: string): string {
  return `scheduler-${jobId}`
}

export function schedulerSessionIdForRun(runId: string): string {
  return `scheduler-run-${runId}`
}
