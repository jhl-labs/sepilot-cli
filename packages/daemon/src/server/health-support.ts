import type { FastifyInstance } from 'fastify'
import type { GatewayTicketWatchRevalidateHealth } from '@sepilotd/api-client'
import {
  resolveChannelPipelineHealthConfig,
  resolveChannelPipelineHealthThresholds,
} from './runtime/channel-pipeline-health.js'
import { getLatestChannelPipelineHealthPolicyChange } from './runtime/channel-pipeline-health-audit.js'
import { resolveWebhookSecurityHealthConfig } from './runtime/webhook-security-health.js'
import { summarizeWebhookSecurity } from './runtime/webhook-endpoints.js'
import { getLatestWebhookSecurityPolicyChange } from './runtime/webhook-security-policy-audit.js'
import { DAEMON_VERSION } from '../version.js'

export interface HealthComponentStatus {
  status: string
  details?: string
  optional?: boolean
  core?: boolean
  [key: string]: unknown
}

export interface ReadinessCheck extends HealthComponentStatus {
  critical: boolean
}

export interface HealthSnapshot {
  status: string
  version: string
  apiVersion: number
  uptime: number
  timestamp: string
  components: Record<string, HealthComponentStatus>
  memory: {
    rss: number
    heap: number
  }
}

export interface ReadinessSnapshot {
  status: string
  version: string
  apiVersion: number
  uptime: number
  timestamp: string
  checks: Record<string, ReadinessCheck>
}

export interface HealthExportSnapshot {
  generatedAt: string
  health: HealthSnapshot
  readiness: ReadinessSnapshot
}

export function now() {
  return {
    version: DAEMON_VERSION,
    apiVersion: 1,
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
  }
}

function isDegradedStatus(status: string): boolean {
  return status === 'error'
    || status === 'degraded'
    || status === 'reindex_required'
}

function isOptionalComponent(component: HealthComponentStatus): boolean {
  return component.optional === true
}

function isCoreComponent(component: HealthComponentStatus): boolean {
  return component.core !== false && !isOptionalComponent(component)
}

function shouldDegradeChannelPipeline(
  totalEvents: number,
  failureRate: number,
  avgAgentLatencyMs: number,
  sampleCount: number,
  thresholds: {
    minRecentEvents: number
    degradeFailureRate: number
    minAgentSamples: number
    degradeAgentAvgLatencyMs: number
  },
): boolean {
  if (
    totalEvents >= thresholds.minRecentEvents
    && failureRate >= thresholds.degradeFailureRate
  ) {
    return true
  }
  if (
    sampleCount >= thresholds.minAgentSamples
    && avgAgentLatencyMs >= thresholds.degradeAgentAvgLatencyMs
  ) {
    return true
  }
  return false
}

function describeDegradedChannelPipelineTypes(
  config: Parameters<typeof resolveChannelPipelineHealthConfig>[0],
  recentByChannelType: Array<{
    channelType: string
    totalEvents: number
    failureRate: number
    byStage: Array<{ stage: string, avgDurationMs: number, count: number }>
  }>,
): string[] {
  const policy = resolveChannelPipelineHealthConfig(config)
  return recentByChannelType
    .filter((entry) => {
      const executeStage = entry.byStage.find((stage) => stage.stage === 'agent_execute')
      const thresholds = resolveChannelPipelineHealthThresholds(
        config,
        entry.channelType,
      )
      return shouldDegradeChannelPipeline(
        entry.totalEvents,
        entry.failureRate,
        executeStage?.avgDurationMs ?? 0,
        executeStage?.count ?? 0,
        thresholds,
      )
    })
    .sort((left, right) => {
      if (right.failureRate !== left.failureRate) {
        return right.failureRate - left.failureRate
      }
      return right.totalEvents - left.totalEvents
    })
    .slice(0, policy.hotChannelTopN)
    .map((entry) => {
      const executeStage = entry.byStage.find((stage) => stage.stage === 'agent_execute')
      const parts = [`${entry.channelType} ${Math.round(entry.failureRate * 100)}%`]
      if (executeStage) {
        parts.push(`${executeStage.avgDurationMs}ms`)
      }
      return parts.join(' ')
    })
}

function describeWebhookSecurityDetails(
  summary: ReturnType<typeof summarizeWebhookSecurity>,
  options: ReturnType<typeof resolveWebhookSecurityHealthConfig>,
  latestPolicyChange: Awaited<
    ReturnType<typeof getLatestWebhookSecurityPolicyChange>
  >,
): string {
  if (summary.totalEndpoints === 0) {
    return latestPolicyChange
      ? `policy ${latestPolicyChange.timestamp} via ${latestPolicyChange.route} on ${latestPolicyChange.device}`
      : 'no policy changes recorded'
  }

  const detailParts = [
    `${summary.verificationReadyEndpoints}/${summary.totalEndpoints} verification-ready`,
  ]
  if (summary.verificationNotReadyEndpoints > 0) {
    detailParts.push(`${summary.verificationNotReadyEndpoints} unready`)
  }
  const topMissingRequirements = summary.unreadySummary.byMissingRequirement
    .slice(0, options.detailTopMissingRequirements)
    .map((entry) => `${entry.requirement}(${entry.endpointCount})`)
  if (topMissingRequirements.length > 0) {
    detailParts.push(`missing ${topMissingRequirements.join('; ')}`)
  }
  if (latestPolicyChange) {
    detailParts.push(
      `policy ${latestPolicyChange.timestamp} via ${latestPolicyChange.route} on ${latestPolicyChange.device}`,
    )
  }
  return detailParts.join(', ')
}

function describeGatewayWatchHealth(
  gatewayWatch: GatewayTicketWatchRevalidateHealth,
): string {
  const detailParts: string[] = []

  if (gatewayWatch.recentStaleHeartbeatTakeover) {
    detailParts.push('ticket watch degraded')
  } else if (gatewayWatch.staleHeartbeatTakeovers > 0) {
    detailParts.push('ticket watch stable')
  }

  if (gatewayWatch.staleHeartbeatTakeovers > 0) {
    detailParts.push(`${gatewayWatch.staleHeartbeatTakeovers} stale takeover${
      gatewayWatch.staleHeartbeatTakeovers === 1 ? '' : 's'
    }`)
  }

  if (gatewayWatch.lastStaleHeartbeatTakeoverAt) {
    detailParts.push(`last ${gatewayWatch.lastStaleHeartbeatTakeoverAt}`)
  }

  return detailParts.join(', ')
}

function describeWebhookSecurityReadinessDetails(
  summary: ReturnType<typeof summarizeWebhookSecurity>,
  options: ReturnType<typeof resolveWebhookSecurityHealthConfig>,
): string {
  if (summary.totalEndpoints === 0) {
    return '0 configured'
  }

  const detailParts = [
    `${summary.verificationReadyEndpoints}/${summary.totalEndpoints} verification-ready`,
  ]
  const unreadyRoutes = summary.unreadyEndpoints
    .map((endpoint) =>
      endpoint.verificationMissing.length > 0
        ? `${endpoint.route} (${endpoint.verificationMissing.join(',')})`
        : endpoint.route,
    )
    .slice(0, options.detailTopUnreadyRoutes)
  if (unreadyRoutes.length > 0) {
    detailParts.push(`unready: ${unreadyRoutes.join(', ')}`)
  }
  const topMissingRequirements = summary.unreadySummary.byMissingRequirement
    .slice(0, options.detailTopMissingRequirements)
    .map((entry) => `${entry.requirement}(${entry.endpointCount})`)
  if (topMissingRequirements.length > 0) {
    detailParts.push(`missing ${topMissingRequirements.join('; ')}`)
  }
  return detailParts.join(', ')
}

async function buildDetailedComponents(
  app: FastifyInstance,
): Promise<Record<string, HealthComponentStatus>> {
  const runtime = app.runtime
  const components: Record<string, HealthComponentStatus> = {}

  if (!runtime) {
    return components
  }

  const providers = runtime.providerRegistry.list()
  components.providers = {
    status: providers.length > 0 ? 'ok' : 'degraded',
    details: `${providers.length} configured`,
  }

  const runLimiterStats = runtime.runLimiter?.getStats()
  if (runLimiterStats) {
    components.run_limiter = {
      status: runLimiterStats.accepting ? 'ok' : 'degraded',
      details: `${runLimiterStats.active}/${runLimiterStats.maxActive} active, ${runLimiterStats.queued}/${runLimiterStats.maxQueued} queued`,
    }
  }

  if (runtime.storageDegraded && runtime.storageDegraded.length > 0) {
    components.storage = {
      status: 'degraded',
      details: runtime.storageDegraded
        .map((entry) => `${entry.store}: ${entry.error}`)
        .join('; '),
    }
  }

  const breakerSummary = runtime.providerCircuitBreaker?.getSummary()
  if (breakerSummary) {
    components.provider_circuit_breaker = {
      status: breakerSummary.openCircuits > 0 ? 'degraded' : 'ok',
      details: `${breakerSummary.openCircuits} open, ${breakerSummary.halfOpenCircuits} half-open, ${breakerSummary.trackedCircuits} tracked`,
    }
  }

  try {
    await runtime.sessions.list({ perPage: 1 })
    components.sessions = { status: 'ok' }
  } catch {
    components.sessions = {
      status: 'error',
      details: 'Session store unavailable',
    }
  }

  if (runtime.semanticIndex && typeof runtime.semanticIndex.getStatus === 'function') {
    const semantic = runtime.semanticIndex.getStatus()
    const detailParts = []
    if (semantic.configuredProviderId && semantic.configuredModel) {
      detailParts.push(`${semantic.configuredProviderId}/${semantic.configuredModel}`)
    }
    if (semantic.vectorBackend) {
      detailParts.push(`backend=${semantic.vectorBackend}`)
    }
    if (semantic.dimensions) {
      detailParts.push(`${semantic.dimensions}d`)
    }
    if (semantic.pendingCount > 0) {
      detailParts.push(`${semantic.pendingCount} pending`)
    }
    if (semantic.failedCount > 0) {
      detailParts.push(`${semantic.failedCount} failed`)
    }
    if (semantic.lastError) {
      detailParts.push(semantic.lastError)
    }

    components.memory_semantic = {
      status: semantic.status,
      details: detailParts.join(', ') || undefined,
    }
  }

  const activeChannels = runtime.channels.filter((channel) => channel.getStatus() === 'connected')
  components.channels = {
    status: 'ok',
    details: `${activeChannels.length}/${runtime.channels.length} connected`,
  }

  if (runtime.jobStore) {
    try {
      const jobs = runtime.jobStore.list()
      const pendingJobs = jobs.filter((job) => job.status === 'pending')
      const recentRuns = typeof runtime.jobStore.listRecentRuns === 'function'
        ? runtime.jobStore.listRecentRuns(50)
        : []
      const recentDeliveryFailures = recentRuns.filter((run) =>
        (run.outputExcerpt ?? '').startsWith('Delivery failed after the scheduled agent produced output:'),
      )
      const detailParts = [
        `${pendingJobs.length} pending / ${jobs.length} total`,
      ]
      if (recentDeliveryFailures.length > 0) {
        detailParts.push(`${recentDeliveryFailures.length} recent delivery failure${recentDeliveryFailures.length === 1 ? '' : 's'}`)
      }
      components.scheduler = {
        status: recentDeliveryFailures.length > 0 ? 'degraded' : 'ok',
        details: detailParts.join(', '),
        core: false,
      }
    } catch {
      components.scheduler = {
        status: 'degraded',
        details: 'scheduler store unavailable',
        core: false,
      }
    }
  }

  const webhookSecuritySummary = summarizeWebhookSecurity(runtime)
  const webhookSecurityHealth = resolveWebhookSecurityHealthConfig(runtime.config)
  const latestWebhookSecurityPolicyChange =
    await getLatestWebhookSecurityPolicyChange(runtime)
  components.webhook_security = {
    status:
      webhookSecuritySummary.verificationNotReadyEndpoints
      >= webhookSecurityHealth.degradeWhenUnreadyEndpointsAtLeast
      ? 'degraded'
      : 'ok',
    details: describeWebhookSecurityDetails(
      webhookSecuritySummary,
      webhookSecurityHealth,
      latestWebhookSecurityPolicyChange,
    ),
  }

  if (runtime.channelReplayStore) {
    const replayStats = await runtime.channelReplayStore.getStats()
    components.channel_replays = {
      status: replayStats.staleProcessingRecords > 0 ? 'degraded' : 'ok',
      details: `${replayStats.totalRecords} tracked, ${replayStats.processingRecords} processing, ${replayStats.staleProcessingRecords} stale`,
    }
  }

  if (runtime.channelSessionStore) {
    const sessionStats = await runtime.channelSessionStore.getStats()
    components.channel_sessions = {
      status: sessionStats.staleBindings > 0 ? 'degraded' : 'ok',
      details: `${sessionStats.totalBindings} bound, ${sessionStats.staleBindings} stale`,
    }
  }

  const pipelineStats = runtime.channelPipelineMonitor?.getStats?.()
  if (pipelineStats) {
    const latestHealthPolicyChange =
      await getLatestChannelPipelineHealthPolicyChange(runtime)
    const overallThresholds = resolveChannelPipelineHealthThresholds(runtime.config)
    const executeStage = pipelineStats.recent.byStage.find((entry) => entry.stage === 'agent_execute')
    const failurePercent = Math.round(pipelineStats.recent.failureRate * 100)
    const degradedChannelTypes = describeDegradedChannelPipelineTypes(
      runtime.config,
      pipelineStats.recent.byChannelType,
    )
    const detailParts = [
      `${pipelineStats.inFlight} in-flight`,
      `${pipelineStats.recent.totalEvents} recent`,
      `${pipelineStats.recent.failureEvents} failures (${failurePercent}%)`,
    ]
    if (executeStage) {
      detailParts.push(`agent avg ${executeStage.avgDurationMs}ms`)
    }
    if (latestHealthPolicyChange) {
      detailParts.push(
        `policy ${latestHealthPolicyChange.timestamp} via ${latestHealthPolicyChange.route} on ${latestHealthPolicyChange.device}`,
      )
    }
    const degradedOverall = shouldDegradeChannelPipeline(
      pipelineStats.recent.totalEvents,
      pipelineStats.recent.failureRate,
      executeStage?.avgDurationMs ?? 0,
      executeStage?.count ?? 0,
      overallThresholds,
    )
    if (!degradedOverall && degradedChannelTypes.length > 0) {
      detailParts.push(`hot ${degradedChannelTypes.join('; ')}`)
    }
    components.channel_pipeline = {
      status: degradedOverall || degradedChannelTypes.length > 0
        ? 'degraded'
        : 'ok',
      details: detailParts.join(', '),
    }
  }

  const plugins = runtime.pluginLoader?.list?.() ?? []
  const registeredPlugins = plugins.filter((plugin) => plugin.status === 'registered').length
  const failedPlugins = plugins.filter((plugin) => plugin.status === 'failed').length
  components.plugins = {
    status: failedPlugins > 0 ? 'degraded' : 'ok',
    details: `${registeredPlugins}/${plugins.length} registered`,
  }

  const mcpServers = runtime.mcpManager?.listServers?.() ?? []
  const enabledMcpServers = mcpServers.filter((server) => server.enabled)
  const connectedMcpServers = enabledMcpServers.filter((server) => server.status === 'connected').length
  const failedMcpServers = enabledMcpServers.filter((server) => server.status === 'error').length
  components.mcp = {
    status: failedMcpServers > 0 ? 'degraded' : 'ok',
    details: `${connectedMcpServers}/${enabledMcpServers.length} connected`,
  }

  try {
    const gatewayClient = runtime.gatewayClient as typeof runtime.gatewayClient & {
      healthInfo?: () => Promise<{
        status: string
        components: {
          ticketWatchRevalidate?: GatewayTicketWatchRevalidateHealth
        }
      } | null>
    }
    const gatewayHealthInfo = typeof gatewayClient.healthInfo === 'function'
      ? await gatewayClient.healthInfo()
      : null

    if (gatewayHealthInfo) {
      const gatewayWatch = gatewayHealthInfo.components.ticketWatchRevalidate
      const gatewayComponent: HealthComponentStatus = {
        status:
          gatewayWatch?.recentStaleHeartbeatTakeover
          || gatewayWatch?.status === 'degraded'
            ? 'degraded'
            : gatewayHealthInfo.status,
        details: gatewayWatch
          ? describeGatewayWatchHealth(gatewayWatch)
          : undefined,
        optional: true,
        core: false,
        gatewayStatus: gatewayHealthInfo.status,
        ...(gatewayWatch
          ? {
              watchStatus: gatewayWatch.status,
              staleHeartbeatTakeovers: gatewayWatch.staleHeartbeatTakeovers,
              recentStaleHeartbeatTakeover:
                gatewayWatch.recentStaleHeartbeatTakeover,
              lastStaleHeartbeatTakeoverAt:
                gatewayWatch.lastStaleHeartbeatTakeoverAt,
              lastStaleHeartbeatTakeoverKey:
                gatewayWatch.lastStaleHeartbeatTakeoverKey,
              leaseTtlMs: gatewayWatch.leaseTtlMs,
              leaseRetryMs: gatewayWatch.leaseRetryMs,
              staleHeartbeatMs: gatewayWatch.staleHeartbeatMs,
            }
          : {}),
      }
      components.gateway = gatewayComponent
    } else {
      const gatewayHealthy = await runtime.gatewayClient.health()
      components.gateway = {
        status: gatewayHealthy ? 'ok' : 'unreachable',
        details: gatewayHealthy ? undefined : 'optional sidecar unavailable',
        optional: true,
        core: false,
      }
    }
  } catch {
    components.gateway = {
      status: 'unreachable',
      details: 'optional sidecar unavailable',
      optional: true,
      core: false,
    }
  }

  return components
}

async function buildReadinessChecks(
  app: FastifyInstance,
): Promise<Record<string, ReadinessCheck>> {
  const runtime = app.runtime
  const checks: Record<string, ReadinessCheck> = {}

  checks.runtime = {
    status: runtime ? 'ok' : 'error',
    details: runtime ? undefined : 'Runtime not initialized',
    critical: true,
  }
  if (!runtime) {
    return checks
  }

  const providers = runtime.providerRegistry.list()
  checks.providers = {
    status: providers.length > 0 ? 'ok' : 'error',
    details: `${providers.length} configured`,
    critical: true,
  }

  const runLimiterStats = runtime.runLimiter?.getStats()
  checks.run_limiter = {
    status: !runLimiterStats || runLimiterStats.accepting ? 'ok' : 'error',
    details: runLimiterStats
      ? `${runLimiterStats.active}/${runLimiterStats.maxActive} active, ${runLimiterStats.queued}/${runLimiterStats.maxQueued} queued`
      : undefined,
    critical: true,
  }

  try {
    await runtime.sessions.list({ perPage: 1 })
    checks.sessions = {
      status: 'ok',
      critical: true,
    }
  } catch {
    checks.sessions = {
      status: 'error',
      details: 'Session store unavailable',
      critical: true,
    }
  }

  checks.gateway = {
    status: 'ok',
    details: 'Optional dependency',
    critical: false,
  }

  const webhookSecuritySummary = summarizeWebhookSecurity(runtime)
  const webhookSecurityHealth = resolveWebhookSecurityHealthConfig(runtime.config)
  checks.webhook_security = {
    status:
      webhookSecuritySummary.verificationNotReadyEndpoints
      >= webhookSecurityHealth.degradeWhenUnreadyEndpointsAtLeast
      ? 'error'
      : 'ok',
    details: describeWebhookSecurityReadinessDetails(
      webhookSecuritySummary,
      webhookSecurityHealth,
    ),
    critical: webhookSecuritySummary.totalEndpoints > 0,
  }

  return checks
}

function isReady(checks: Record<string, ReadinessCheck>): boolean {
  return Object.values(checks).every((check) => !check.critical || check.status === 'ok')
}

export async function buildHealthSnapshot(app: FastifyInstance): Promise<HealthSnapshot> {
  const components = await buildDetailedComponents(app)
  const overallStatus = Object.values(components).some((component) =>
    isCoreComponent(component) && isDegradedStatus(component.status),
  )
    ? 'degraded'
    : 'ok'

  return {
    status: overallStatus,
    ...now(),
    components,
    memory: {
      rss: Math.round(process.memoryUsage().rss / 1024 / 1024),
      heap: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
    },
  }
}

export async function buildReadinessSnapshot(app: FastifyInstance): Promise<ReadinessSnapshot> {
  const checks = await buildReadinessChecks(app)
  const ready = isReady(checks)

  return {
    status: ready ? 'ok' : 'not_ready',
    ...now(),
    checks,
  }
}

export async function buildHealthExportSnapshot(
  app: FastifyInstance,
): Promise<HealthExportSnapshot> {
  const [health, readiness] = await Promise.all([
    buildHealthSnapshot(app),
    buildReadinessSnapshot(app),
  ])

  return {
    generatedAt: new Date().toISOString(),
    health,
    readiness,
  }
}

export function formatHealthReport(snapshot: HealthExportSnapshot): string {
  const lines = [
    '# sepilotd Health Report',
    '',
    `Generated: ${snapshot.generatedAt}`,
    `Status: ${snapshot.health.status}`,
    `Version: ${snapshot.health.version}`,
    `Uptime: ${Math.round(snapshot.health.uptime)}s`,
    `Memory: ${snapshot.health.memory.rss}MB RSS / ${snapshot.health.memory.heap}MB Heap`,
    '',
    '## Components',
  ]

  for (const [name, component] of Object.entries(snapshot.health.components)) {
    const label = component.optional
      ? `${name} [optional]`
      : component.core === false
        ? `${name} [feature]`
        : name
    lines.push(`- ${label}: ${component.status}${component.details ? ` — ${component.details}` : ''}`)
    for (const [key, value] of Object.entries(component)) {
      if (
        key === 'status'
        || key === 'details'
        || key === 'optional'
        || key === 'core'
        || value === undefined
      ) {
        continue
      }
      lines.push(`  - ${key}: ${String(value)}`)
    }
  }

  lines.push('', '## Readiness')
  for (const [name, check] of Object.entries(snapshot.readiness.checks)) {
    lines.push(
      `- ${name}: ${check.status}${check.critical ? ' [critical]' : ''}${check.details ? ` — ${check.details}` : ''}`,
    )
  }

  return `${lines.join('\n')}\n`
}
