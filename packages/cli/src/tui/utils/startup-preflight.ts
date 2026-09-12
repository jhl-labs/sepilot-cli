import type {
  DaemonHealth,
  DaemonProviderInfo,
} from '@sepilotd/api-client'

interface BuildStartupPreflightSummaryOptions {
  health: DaemonHealth
  providers: DaemonProviderInfo[]
  daemonDefaultProvider?: string | null
  daemonDefaultModel?: string | null
  providersLoaded?: boolean
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
  const components = Object.entries(health.components ?? {})
    .filter(([name, component]) => isCoreComponent(name, component))
    .map(([, component]) => component.status)
    .filter(isProblemComponent)

  if (components.length === 0) return 'ok'
  if (components.some((status) => status === 'error' || status === 'unreachable')) {
    return 'error'
  }
  return 'degraded'
}

function formatOptionalComponentLine(
  name: string,
  status: string,
  details?: string,
): string {
  return `${name.padEnd(8)} optional · ${status}${details ? ` · ${details}` : ''}`
}

export function buildStartupPreflightSummary({
  health,
  providers,
  daemonDefaultProvider,
  daemonDefaultModel,
  providersLoaded = true,
}: BuildStartupPreflightSummaryOptions): string | null {
  const unhealthyProviders = providers.filter((provider) => provider.health.status !== 'ready')
  const defaultProvider = daemonDefaultProvider
    ? providers.find((provider) => provider.id === daemonDefaultProvider) ?? null
    : null

  const issues: string[] = []
  const nextActions: string[] = []
  const reportedProviderIds = new Set<string>()
  const daemonCoreStatus = deriveDaemonCoreStatus(health)
  const gateway = health.components?.gateway

  if (daemonCoreStatus !== 'ok') {
    issues.push(`Daemon core health is ${daemonCoreStatus}. Use /doctor for the current health summary.`)
    nextActions.push('/doctor')
  }

  if (!providersLoaded) {
    issues.push('Provider snapshot is unavailable right now. Use /doctor or /provider to retry the daemon state check.')
    nextActions.push('/doctor', '/provider')
  } else if (providers.length === 0) {
    issues.push('No providers are configured on the daemon yet.')
    nextActions.push('/provider setup', 'Ctrl+T')
  } else {
    if (!defaultProvider && daemonDefaultProvider) {
      issues.push(`Daemon default provider ${daemonDefaultProvider} is not configured anymore.`)
      nextActions.push('/provider default', '/provider setup')
    } else if (defaultProvider && defaultProvider.health.status !== 'ready') {
      issues.push(
        `Daemon default ${daemonDefaultProvider}/${daemonDefaultModel || 'default'} needs attention: ${formatProviderHealthSummary(defaultProvider)}.`,
      )
      nextActions.push(`/provider edit ${defaultProvider.id}`, '/doctor')
      reportedProviderIds.add(defaultProvider.id)
    } else if (
      defaultProvider
      && daemonDefaultModel
      && defaultProvider.modelCatalogAuthority === 'endpoint'
      && !defaultProvider.models.some((model) => model.id === daemonDefaultModel)
    ) {
      issues.push(
        `Daemon default model ${daemonDefaultProvider}/${daemonDefaultModel} is absent from the current endpoint catalog; new turns will use an available fallback.`,
      )
      nextActions.push('/provider default', '/provider')
    }

    const remainingUnhealthyProviders = unhealthyProviders
      .filter((provider) => !reportedProviderIds.has(provider.id))

    if (remainingUnhealthyProviders.length > 0) {
      const topIssues = remainingUnhealthyProviders
        .slice(0, 2)
        .map((provider) => `${provider.id} ${formatProviderHealthSummary(provider)}`)
      issues.push(`Providers needing attention: ${topIssues.join(' · ')}.`)
      nextActions.push(`/provider edit ${remainingUnhealthyProviders[0].id}`, '/provider')
    }
  }

  if (issues.length === 0) {
    return null
  }

  const providerSummary = providersLoaded
    ? `${providers.length} configured`
    : 'snapshot unavailable'

  return [
    'Startup Check',
    `Daemon   ${daemonCoreStatus} · v${health.version}`,
    ...(gateway ? [formatOptionalComponentLine('Gateway', gateway.status, gateway.details)] : []),
    `Providers ${providerSummary}${daemonDefaultProvider ? ` · default ${daemonDefaultProvider}/${daemonDefaultModel || 'default'}` : ''}`,
    ...issues.map((issue, index) => `${index === 0 ? 'Issue' : 'Issue'.padEnd(5)}   ${issue}`),
    `Next     ${Array.from(new Set(nextActions)).join('  ·  ')}`,
  ].join('\n')
}
