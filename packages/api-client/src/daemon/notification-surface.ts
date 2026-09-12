import type {
  DaemonNotificationItem,
  DaemonNotifyRelayMessageStatus,
} from './types.js'

export type DaemonNotifyRelayEvidenceState =
  | 'accepted'
  | 'pending_review'
  | 'provider_pending'
  | 'provider_delivery_retrying'
  | 'provider_status_retrying'
  | 'delivered'
  | 'provider_failed'
  | 'provider_unconfirmed'
  | 'handoff_failed'

export interface DaemonNotifyRelayEvidencePresentation {
  state: DaemonNotifyRelayEvidenceState
  label: string
  terminal: boolean
}

const TERMINAL_PROVIDER_FAILURES = new Set<DaemonNotifyRelayMessageStatus>([
  'rejected',
  'denied',
  'dead_letter',
  'expired',
])

function statusLabel(status: string): string {
  return status.replaceAll('_', ' ')
}

function errorCodeSuffix(errorCode: string | null | undefined): string {
  return errorCode ? ` · ${statusLabel(errorCode)}` : ''
}

/**
 * Classifies Relay evidence from one notification only. Acceptance is never
 * promoted to final delivery, and callers cannot accidentally combine the
 * acceptance receipt from one notification with provider evidence from another.
 */
export function daemonNotifyRelayEvidence(
  notification: Pick<DaemonNotificationItem, 'relayDelivery' | 'relayProviderDelivery'>,
): DaemonNotifyRelayEvidencePresentation | null {
  const acceptance = notification.relayDelivery
  if (!acceptance) return null

  if (
    acceptance.status === 'rejected'
    || acceptance.status === 'unreachable'
    || acceptance.status === 'invalid_response'
  ) {
    return {
      state: 'handoff_failed',
      label: `Relay handoff failed · ${statusLabel(acceptance.status)}${errorCodeSuffix(acceptance.errorCode)}`,
      terminal: true,
    }
  }

  const provider = notification.relayProviderDelivery
  if (!provider) {
    return acceptance.status === 'pending_review'
      ? { state: 'pending_review', label: 'Relay pending review', terminal: false }
      : {
          state: 'accepted',
          label: 'Relay accepted · provider pending',
          terminal: false,
        }
  }

  if (provider.status === 'delivered') {
    return {
      state: 'delivered',
      label: 'Relay provider delivered',
      terminal: true,
    }
  }

  if (provider.status && TERMINAL_PROVIDER_FAILURES.has(provider.status)) {
    return {
      state: 'provider_failed',
      label: `Relay provider failed · ${statusLabel(provider.status)}`,
      terminal: true,
    }
  }

  if (provider.status === 'delivery_failed') {
    return {
      state: 'provider_delivery_retrying',
      label: 'Relay provider delivery retrying',
      terminal: false,
    }
  }

  if (provider.status) {
    return provider.status === 'pending_review'
      ? {
          state: 'pending_review',
          label: 'Relay pending review',
          terminal: false,
        }
      : {
          state: 'provider_pending',
          label: `Relay provider · ${statusLabel(provider.status)}`,
          terminal: false,
        }
  }

  if (provider.completedAt != null) {
    return {
      state: 'provider_unconfirmed',
      label: `Relay provider status unconfirmed${errorCodeSuffix(provider.errorCode)}`,
      terminal: true,
    }
  }

  return {
    state: 'provider_status_retrying',
    label: `Relay provider status retrying${errorCodeSuffix(provider.errorCode)}`,
    terminal: false,
  }
}
