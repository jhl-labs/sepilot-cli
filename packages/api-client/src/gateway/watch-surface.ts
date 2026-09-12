import type { DaemonHealth } from '../daemon/types.js'
import type { ActivityItem } from '../daemon/chat-surface-types.js'
import type { GatewayTicketWatchRevalidateHealth } from './http.js'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function readGatewayWatchStatus(value: unknown): 'ok' | 'degraded' | 'error' | null {
  return value === 'ok' || value === 'degraded' || value === 'error'
    ? value
    : null
}

function readNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function readBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

function readNullableString(value: unknown): string | null {
  return typeof value === 'string' || value === null ? value : null
}

export function getGatewayWatchHealthFromDaemonHealth(
  daemonHealth: DaemonHealth | null | undefined,
): GatewayTicketWatchRevalidateHealth | null {
  const gatewayComponent = daemonHealth?.components?.gateway
  if (!isRecord(gatewayComponent)) {
    return null
  }

  const hasWatchDetails = [
    'watchStatus',
    'staleHeartbeatTakeovers',
    'recentStaleHeartbeatTakeover',
    'lastStaleHeartbeatTakeoverAt',
    'lastStaleHeartbeatTakeoverKey',
    'leaseTtlMs',
    'leaseRetryMs',
    'staleHeartbeatMs',
  ].some((key) => key in gatewayComponent)

  if (!hasWatchDetails) {
    return null
  }

  const status = readGatewayWatchStatus(gatewayComponent.watchStatus)
    ?? readGatewayWatchStatus(gatewayComponent.status)
  if (!status) {
    return null
  }

  return {
    status,
    staleHeartbeatTakeovers:
      readNumber(gatewayComponent.staleHeartbeatTakeovers) ?? 0,
    recentStaleHeartbeatTakeover:
      readBoolean(gatewayComponent.recentStaleHeartbeatTakeover) ?? false,
    lastStaleHeartbeatTakeoverAt:
      readNullableString(gatewayComponent.lastStaleHeartbeatTakeoverAt),
    lastStaleHeartbeatTakeoverKey:
      readNullableString(gatewayComponent.lastStaleHeartbeatTakeoverKey),
    leaseTtlMs: readNumber(gatewayComponent.leaseTtlMs) ?? 0,
    leaseRetryMs: readNumber(gatewayComponent.leaseRetryMs) ?? 0,
    staleHeartbeatMs: readNumber(gatewayComponent.staleHeartbeatMs) ?? 0,
  }
}

export function hasGatewayWatchAlert(
  gatewayWatchHealth: GatewayTicketWatchRevalidateHealth | null,
): boolean {
  return Boolean(
    gatewayWatchHealth
    && (
      gatewayWatchHealth.recentStaleHeartbeatTakeover
      || gatewayWatchHealth.staleHeartbeatTakeovers > 0
    ),
  )
}

export function getGatewayWatchAlertMeta(
  gatewayWatchHealth: GatewayTicketWatchRevalidateHealth | null,
): string | null {
  if (!gatewayWatchHealth || gatewayWatchHealth.staleHeartbeatTakeovers <= 0) {
    return null
  }

  const parts = [
    `${gatewayWatchHealth.staleHeartbeatTakeovers} takeover${
      gatewayWatchHealth.staleHeartbeatTakeovers === 1 ? '' : 's'
    }`,
  ]

  if (gatewayWatchHealth.lastStaleHeartbeatTakeoverAt) {
    parts.push(
      `last ${new Date(
        gatewayWatchHealth.lastStaleHeartbeatTakeoverAt,
      ).toLocaleTimeString()}`,
    )
  }

  return parts.join(' · ')
}

export function getGatewayWatchMonitorLabel(
  gatewayWatchHealth: GatewayTicketWatchRevalidateHealth | null,
): string {
  if (!gatewayWatchHealth) {
    return 'Unavailable'
  }

  return gatewayWatchHealth.recentStaleHeartbeatTakeover
    ? 'Degraded'
    : gatewayWatchHealth.status === 'degraded'
      ? 'Recovering'
      : 'Stable'
}

export function getGatewayWatchMonitorDetail(
  gatewayWatchHealth: GatewayTicketWatchRevalidateHealth | null,
): string {
  if (!gatewayWatchHealth) {
    return 'Gateway health details are unavailable.'
  }

  if (gatewayWatchHealth.recentStaleHeartbeatTakeover) {
    return 'Recent stale lease takeover recovered.'
  }

  if (gatewayWatchHealth.staleHeartbeatTakeovers > 0) {
    return 'Previous failovers recorded, watch currently stable.'
  }

  return 'No stale lease takeover recorded.'
}

export function createGatewayWatchActivity(
  gatewayWatchHealth: GatewayTicketWatchRevalidateHealth | null,
): ActivityItem | null {
  if (!gatewayWatchHealth || gatewayWatchHealth.staleHeartbeatTakeovers <= 0) {
    return null
  }

  return {
    id: `gateway-watch-${
      gatewayWatchHealth.lastStaleHeartbeatTakeoverAt
      ?? gatewayWatchHealth.staleHeartbeatTakeovers
    }`,
    kind: 'state',
    label: gatewayWatchHealth.recentStaleHeartbeatTakeover
      ? 'Gateway watch failover'
      : 'Gateway watch history',
    detail: gatewayWatchHealth.recentStaleHeartbeatTakeover
      ? 'Recovered from a stale ticket-watch lease takeover.'
      : 'Previous stale ticket-watch failovers were recorded.',
    status: gatewayWatchHealth.recentStaleHeartbeatTakeover ? 'pending' : 'neutral',
    meta: getGatewayWatchAlertMeta(gatewayWatchHealth) ?? undefined,
  }
}
