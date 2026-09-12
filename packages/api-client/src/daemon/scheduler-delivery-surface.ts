import type { DaemonSchedulerDeliveryOutboxSummary } from './types.js'

export type DaemonSchedulerDeliveryOutboxState = 'idle' | 'pending' | 'retrying'

export interface DaemonSchedulerDeliveryOutboxPresentation {
  state: DaemonSchedulerDeliveryOutboxState
  label: string
  active: number
  pending: number
  delivering: number
  retrying: number
  delivered: number
  nextAttemptAt: number | null
  nextRetryAt: number | null
  requiresAttention: boolean
}

function deliveryLabel(count: number, state: 'pending' | 'retrying'): string {
  return `${count} channel ${count === 1 ? 'delivery' : 'deliveries'} ${state}`
}

/**
 * Gives every user-facing surface the same content-free interpretation of the
 * durable scheduler delivery queue. Failed rows remain retryable work, not
 * failed scheduler jobs, and therefore take precedence over ordinary pending
 * rows when mixed states are present.
 */
export function daemonSchedulerDeliveryOutboxEvidence(
  summary: DaemonSchedulerDeliveryOutboxSummary | null | undefined,
): DaemonSchedulerDeliveryOutboxPresentation | null {
  if (!summary) return null

  const pending = summary.pending + summary.delivering
  const active = pending + summary.failed
  const state: DaemonSchedulerDeliveryOutboxState = summary.failed > 0
    ? 'retrying'
    : pending > 0
      ? 'pending'
      : 'idle'

  return {
    state,
    label: state === 'idle'
      ? 'Channel delivery queue clear'
      : deliveryLabel(state === 'retrying' ? summary.failed : pending, state),
    active,
    pending: summary.pending,
    delivering: summary.delivering,
    retrying: summary.failed,
    delivered: summary.delivered,
    nextAttemptAt: summary.nextAttemptAt,
    nextRetryAt: summary.nextRetryAt,
    requiresAttention: state !== 'idle',
  }
}
