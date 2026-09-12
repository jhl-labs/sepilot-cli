import type { DaemonSchedulerJobRun } from './types.js'

/**
 * Returns the daemon-issued audit-session identity for a scheduler run.
 * Legacy, non-agent, and malformed blank values all mean no linked evidence.
 */
export function daemonSchedulerRunAgentSessionId(
  run: Pick<DaemonSchedulerJobRun, 'agentSessionId'>,
): string | null {
  return run.agentSessionId?.trim() || null
}
