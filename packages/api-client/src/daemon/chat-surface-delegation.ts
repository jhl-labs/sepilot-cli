import type { DaemonSessionDetail } from './types.js'
import type { ActivityStatus } from './chat-surface-types.js'

export function delegationHealthLabel(
  delegation?: DaemonSessionDetail['delegation'],
): string {
  switch (delegation?.claimHealth) {
    case 'degraded':
      return 'Lease degraded'
    case 'lost':
      return 'Lease lost'
    case 'healthy':
      return 'Lease healthy'
    default:
      return 'Local run'
  }
}

export function delegationHealthDetail(
  delegation?: DaemonSessionDetail['delegation'],
): string {
  if (!delegation) {
    return 'This session is running locally on the current daemon.'
  }

  switch (delegation.claimHealth) {
    case 'degraded':
      return `Lease renewals are failing on ${delegation.targetDevice}, but claim verification still holds. The worker will fence if this does not recover.`
    case 'lost':
      return `The worker lost its delegation lease on ${delegation.targetDevice} and fenced the run.`
    default:
      return `Delegated execution is actively renewing its claim on ${delegation.targetDevice}.`
  }
}

export function delegationHealthStatus(
  delegation?: DaemonSessionDetail['delegation'],
): ActivityStatus {
  switch (delegation?.claimHealth) {
    case 'degraded':
      return 'pending'
    case 'lost':
      return 'error'
    default:
      return 'neutral'
  }
}
