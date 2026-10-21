import type { RunStopReason } from '@sepilotd/api-client'

export interface TurnOutcome {
  kind: 'completed' | 'cancelled' | 'timeout' | 'error' | 'stopped'
  reason?: RunStopReason
  errorCode?: string
}

export function terminalTurnOutcome(type: 'done' | 'error', reason?: RunStopReason, errorCode?: string): TurnOutcome {
  if (reason?.code === 'user_abort') return { kind: 'cancelled', reason, errorCode }
  if (reason?.code === 'wall_clock' || reason?.code === 'inactivity' || errorCode === 'TIMEOUT') {
    return { kind: 'timeout', reason, errorCode }
  }
  if (type === 'error' || reason?.code === 'provider_error') return { kind: 'error', reason, errorCode }
  if (reason && reason.code !== 'completed') return { kind: 'stopped', reason }
  return { kind: 'completed', reason }
}
