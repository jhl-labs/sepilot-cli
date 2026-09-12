import type {
  DaemonHealth,
  DaemonProviderInfo,
  DaemonUsageSummary,
} from '@sepilotd/api-client'

interface BuildDoctorSummaryOptions {
  health: DaemonHealth
  providers: DaemonProviderInfo[]
  usage: DaemonUsageSummary | null
  currentSessionId?: string | null
}

interface BuildDoctorUnavailableSummaryOptions {
  error: unknown
  baseUrl?: string
}

function shortId(value: string | null | undefined): string {
  return value ? value.slice(0, 8) : 'none'
}

function formatUptime(seconds: number | undefined): string {
  if (!seconds) return 'unknown'
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  return `${hours}h ${minutes}m`
}

function formatProviderHealthSummary(
  provider: Pick<DaemonProviderInfo, 'health'>,
): string {
  if (provider.health.status === 'env_missing') {
    return provider.health.missingEnvVars.length > 0
      ? `env missing: ${provider.health.missingEnvVars.join(', ')}`
      : 'env missing'
  }

  if (provider.health.status === 'unavailable') {
    return provider.health.message
      ? `unavailable: ${provider.health.message}`
      : 'unavailable'
  }

  return 'ready'
}

function formatUsage(usage: DaemonUsageSummary | null): string {
  if (!usage) {
    return 'unavailable'
  }

  const totalTokens = usage.inputTokens + usage.outputTokens
  return [
    `${usage.requestCount.toLocaleString()} req`,
    `${usage.inputTokens.toLocaleString()} in / ${usage.outputTokens.toLocaleString()} out`,
    `${totalTokens.toLocaleString()} total`,
    `$${usage.costUsd.toFixed(4)}`,
  ].join(' · ')
}

function isProblemComponent(status: string): boolean {
  return status !== 'ok' && status !== 'disabled'
}

function isOptionalComponent(
  name: string,
  component: { optional?: boolean },
): boolean {
  return name === 'gateway' || component.optional === true
}

function isCoreComponent(
  name: string,
  component: { optional?: boolean, core?: boolean },
): boolean {
  return !isOptionalComponent(name, component) && component.core !== false
}

function deriveDaemonCoreStatus(health: DaemonHealth): string {
  const statuses = Object.entries(health.components ?? {})
    .filter(([name, component]) => isCoreComponent(name, component))
    .map(([, component]) => component.status)
    .filter(isProblemComponent)

  if (statuses.length === 0) return 'ok'
  if (statuses.some((status) => status === 'error' || status === 'unreachable')) {
    return 'error'
  }
  return 'degraded'
}

function formatGateway(health: DaemonHealth): string | null {
  const gateway = health.components?.gateway
  if (!gateway) return null
  return `Gateway  optional · ${gateway.status}${gateway.details ? ` · ${gateway.details}` : ''}`
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message
  return String(error || 'unknown error')
}

export function buildDoctorSummary({
  health,
  providers,
  usage,
  currentSessionId,
}: BuildDoctorSummaryOptions): string {
  const degradedComponents = Object.entries(health.components ?? {})
    .filter(([name, component]) => isCoreComponent(name, component))
    .filter(([, component]) => isProblemComponent(component.status))
    .slice(0, 4)
  const optionalComponents = Object.entries(health.components ?? {})
    .filter(([name, component]) => name !== 'gateway' && isOptionalComponent(name, component))
    .filter(([, component]) => isProblemComponent(component.status))
    .slice(0, 4)
  const featureComponents = Object.entries(health.components ?? {})
    .filter(([name, component]) => !isOptionalComponent(name, component) && component.core === false)
    .filter(([, component]) => isProblemComponent(component.status))
    .slice(0, 4)
  const unhealthyProviders = providers
    .filter((provider) => provider.health.status !== 'ready')
    .slice(0, 3)
  const readyProviders = providers.filter((provider) => provider.health.status === 'ready').length

  const nextActions: string[] = []
  if (providers.length <= 0) {
    nextActions.push('/provider setup')
  } else if (unhealthyProviders.length > 0) {
    nextActions.push(`/provider edit ${unhealthyProviders[0].id}`)
  }
  if (currentSessionId) {
    nextActions.push('/context map')
  } else {
    nextActions.push('/resume')
  }
  nextActions.push('/help')

  return [
    'Doctor',
    `Daemon   ${deriveDaemonCoreStatus(health)} · v${health.version} · uptime ${formatUptime(health.uptime)}`,
    formatGateway(health),
    `Session  ${shortId(currentSessionId)} ${currentSessionId ? 'active' : 'no active session'}`,
    `Providers ${providers.length} configured · ${readyProviders} ready${unhealthyProviders.length > 0 ? ` · ${unhealthyProviders.length} need attention` : ''}`,
    `Usage    ${formatUsage(usage)}`,
    degradedComponents.length > 0
      ? 'Components'
      : 'Components  all core health checks are ok',
    ...degradedComponents.map(([name, component]) => (
      `  ${name} ${component.status}${component.details ? ` · ${component.details}` : ''}`
    )),
    ...(optionalComponents.length > 0
      ? [
          'Optional  non-core components need attention',
          ...optionalComponents.map(([name, component]) => (
            `  ${name} ${component.status}${component.details ? ` · ${component.details}` : ''}`
          )),
        ]
      : []),
    ...(featureComponents.length > 0
      ? [
          'Features  non-core features need attention',
          ...featureComponents.map(([name, component]) => (
            `  ${name} ${component.status}${component.details ? ` · ${component.details}` : ''}`
          )),
        ]
      : []),
    ...(unhealthyProviders.length > 0
      ? [
          'Providers needing attention',
          ...unhealthyProviders.map((provider) => (
            `  ${provider.id} ${formatProviderHealthSummary(provider)}`
          )),
        ]
      : []),
    `Next     ${nextActions.join('  ·  ')}`,
  ].filter((line): line is string => Boolean(line)).join('\n')
}

export function buildDoctorUnavailableSummary({
  error,
  baseUrl,
}: BuildDoctorUnavailableSummaryOptions): string {
  const endpoint = baseUrl ?? 'configured daemon URL'
  return [
    'Doctor',
    `Daemon   unreachable · ${endpoint}`,
    `Issue    ${errorMessage(error)}`,
    'Meaning  The TUI could not fetch the daemon health snapshot, so provider and usage checks were skipped.',
    'Next     /status  ·  sepilot daemon status  ·  sepilot daemon restart',
  ].join('\n')
}
