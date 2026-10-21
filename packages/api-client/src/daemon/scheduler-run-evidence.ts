import type { DaemonSchedulerJobRun } from './types.js'

export function daemonSchedulerRunAgentSessionId(
  run: Pick<DaemonSchedulerJobRun, 'agentSessionId'>,
): string | null {
  return run.agentSessionId?.trim() || null
}
