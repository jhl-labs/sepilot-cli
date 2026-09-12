import chalk from 'chalk'
import {
  ApiHttpError,
  daemonSchedulerDeliveryOutboxEvidence,
  type DaemonAgentDescriptor,
  type DaemonAgentGraphSnapshot,
  type DaemonAssistantJpadPublicationCorrelation,
  type DaemonAssistantSchedulerRunCorrelation,
  type DaemonAssistantRuntimeStatus,
  type DaemonChannelConfig,
  type DaemonChatBackgroundListItem,
  type DaemonConfig,
  type DaemonHealth,
  type DaemonLatestNotifyRelayDelivery,
  type DaemonNotifyRelayOutboxSummary,
  type DaemonNotifyRelayProviderSummary,
  type DaemonNotificationItem,
  type DaemonSchedulerDeliveryOutboxSummary,
  type DaemonSchedulerJob,
  type DaemonSessionManagementSnapshot,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { ensureDaemon } from '../client/ensure-daemon.js'
import {
  type DaemonEndpointScope,
  resolveDaemonEndpointScope,
} from '../client/token.js'
import { output, outputError } from '../output/formatter.js'
import { friendlyErrorMessage } from '../utils/error-message.js'
import { hasPendingScheduledRun } from '../utils/scheduler-state.js'

type AssistantOverallStatus = 'ready' | 'attention' | 'degraded'
type OptionalFailureKind = 'authentication' | 'authorization' | 'other'

function notifyRelayFailureAction(delivery: DaemonLatestNotifyRelayDelivery): string {
  if (delivery.errorCode === 'forbidden_scope') {
    return 'HTTP 403 confirms Relay connectivity. Verify message:create, destination, and message-type scopes plus the client-to-destination mapping.'
  }
  if (delivery.errorCode === 'source_ip_not_allowed') {
    return 'HTTP 403 confirms Relay connectivity. Verify the client source and trusted-proxy CIDR policy.'
  }
  if (
    delivery.errorCode === 'client_not_found'
    || delivery.errorCode === 'client_paused'
    || delivery.errorCode === 'tenant_paused'
  ) {
    return 'HTTP 403 confirms Relay connectivity. Restore the Relay client and tenant to an active administrative state before retrying.'
  }
  if (delivery.httpStatus === 403) {
    return 'HTTP 403 confirms Relay connectivity. Verify Relay scopes, client-to-destination mapping, source CIDR policy, and client/tenant/destination state.'
  }
  if (
    delivery.httpStatus === 401
    || delivery.errorCode === 'invalid_token'
    || delivery.errorCode === 'token_expired'
    || delivery.errorCode === 'token_revoked'
  ) {
    return 'Replace or rotate the runtime-managed Relay token; do not copy it into config, logs, or command arguments.'
  }
  if (delivery.status === 'unreachable') {
    return 'Inspect daemon-to-Relay DNS, TLS, and network reachability; the durable outbox will retry.'
  }
  if (delivery.status === 'invalid_response') {
    return 'Verify Notification Relay response compatibility; Sepilot requires HTTP 202 with a message ID and accepted or pending_review status.'
  }
  if (delivery.httpStatus === 429 || (delivery.httpStatus !== null && delivery.httpStatus >= 500)) {
    return 'Inspect Relay availability and rate limits; the durable outbox will retry with the original idempotency key.'
  }
  return 'Inspect the structured Relay rejection code and runtime-managed integration policy.'
}

export type AssistantAgentRouting =
  | { kind: 'dynamic'; detail: 'per-request' }
  | { kind: 'direct'; detail: 'react-loop' | 'instant-react-loop' }
  | { kind: 'graph'; detail: 'registered-graph' }
  | {
    kind: 'unavailable'
    detail: 'mode-unavailable' | 'graph-catalog-unavailable' | 'configured-mode-missing'
  }

interface OptionalResult<T> {
  section: string
  value: T | null
  available: boolean
  failureKind: OptionalFailureKind | null
}

export interface AssistantStatusSnapshot {
  generatedAt: string
  overall: AssistantOverallStatus
  daemon: {
    status: string
    version: string
  }
  agent: {
    mode: string | null
    autonomy: string | null
    provider: string | null
    model: string | null
    descriptor: DaemonAgentDescriptor | null
    graph: Pick<DaemonAgentGraphSnapshot, 'id' | 'nodeCount' | 'edgeCount'> | null
    routing: AssistantAgentRouting
  }
  workspace: {
    defaultRoot: string | null
  }
  capabilities: {
    requiredTools: Array<{ name: string; enabled: boolean }>
    enabledRequiredTools: number
    requiredSkills: Array<{ id: string; available: boolean; enabled: boolean }>
  }
  channels: Array<{
    type: string
    enabled: boolean
    status: string
  }>
  scheduler: {
    configured: boolean | null
    accessible: boolean | null
    total: number
    enabled: number
    running: number
    failed: number
    deliveryRoutes: {
      routed: number
      invalid: number
      unavailable: number
      unknown: number
    }
    channelOutbox: DaemonSchedulerDeliveryOutboxSummary | null
    next: Array<{ id: string; name: string; nextRunAt: number; status: string }>
    recentRuns: DaemonAssistantSchedulerRunCorrelation[]
  }
  background: {
    active: number
    waitingApproval: number
    waitingQuestion: number
    jobs: Array<{
      jobId: string
      sessionId: string
      action: 'running' | 'approval' | 'question'
      label: string | null
    }>
  }
  approvals: {
    pending: number | null
    orphaned: number | null
    pendingQuestions: number | null
  }
  notifications: {
    total: number | null
    unread: number | null
    listed: number
    countsExact: boolean
    truncated: boolean | null
    relayConfigured: boolean | null
    deliveryMode: 'durable_outbox' | 'acceptance_receipts' | 'best_effort_untracked' | null
    latestRelayDelivery: DaemonLatestNotifyRelayDelivery | null
    relayOutbox: DaemonNotifyRelayOutboxSummary | null
    relayProviderDelivery: DaemonNotifyRelayProviderSummary | null
    recent: Array<{
      id: string
      title: string
      topic: string | null
      createdAt: number
      read: boolean
    }>
  }
  research: {
    aiSearchConfigured: boolean | null
    aiSearchSelected: boolean | null
    searchProvider: string | null
    jpadConfigured: boolean | null
    jpadPublicationEvidence: 'available' | 'partial' | 'unavailable' | null
    recentJpadPublications: DaemonAssistantJpadPublicationCorrelation[]
  }
  attention: Array<{
    code: string
    message: string
    action?: string
  }>
  partialFailures: string[]
}

type ChannelRuntimeSummary = {
  enabled: boolean
  status: string
} | null

type DeliveryRouteReadiness = 'available' | 'unavailable' | 'unknown'

const REQUIRED_SKILL_IDS = ['monitor-infrastructure', 'research-to-jpad'] as const

function optionalFailureKind(error: unknown): OptionalFailureKind {
  let status: number | null = null
  if (error instanceof ApiHttpError) {
    status = error.status
  } else if (error && typeof error === 'object') {
    const record = error as Record<string, unknown>
    const candidate = typeof record.status === 'number'
      ? record.status
      : typeof record.statusCode === 'number'
        ? record.statusCode
        : null
    if (Number.isInteger(candidate)) status = candidate
    if (status === null && error instanceof Error) {
      const legacyStatus = /^(\d{3})(?:\s|:)/.exec(error.message)?.[1]
      if (legacyStatus) status = Number.parseInt(legacyStatus, 10)
    }
  }

  if (status === 401) return 'authentication'
  if (status === 403) return 'authorization'
  return 'other'
}

async function optional<T>(section: string, promise: Promise<T>): Promise<OptionalResult<T>> {
  try {
    return { section, value: await promise, available: true, failureKind: null }
  } catch (error) {
    // Do not copy raw remote errors into the snapshot: provider errors may
    // contain endpoints or credential-bearing diagnostic text. The section
    // name is enough to make partial availability explicit.
    return {
      section,
      value: null,
      available: false,
      failureKind: optionalFailureKind(error),
    }
  }
}

function parseLimit(value: string | undefined): number {
  const parsed = Number.parseInt(value ?? '5', 10)
  if (!Number.isFinite(parsed)) return 5
  return Math.min(20, Math.max(1, parsed))
}

function backgroundAction(job: DaemonChatBackgroundListItem): 'running' | 'approval' | 'question' {
  const action = job.progress?.action?.type
  return action === 'approval' || action === 'question' ? action : 'running'
}

function channelStatus(
  channel: DaemonChannelConfig,
  summaries: ReadonlyMap<string, ChannelRuntimeSummary>,
): { type: string; enabled: boolean; status: string } {
  const summary = summaries.get(channel.type)
  const enabled = summary?.enabled ?? channel.enabled !== false
  return {
    type: channel.type,
    enabled,
    status: summary?.status ?? (enabled ? 'configured' : 'disabled'),
  }
}

function runtimeEnvironmentRecoveryAction(
  endpointScope: DaemonEndpointScope,
  names: readonly string[],
): string {
  const joined = names.join(', ')
  if (endpointScope !== 'loopback') {
    return `Configure ${joined} through the daemon runtime secret manager (OpenBao/ExternalSecret), then roll or restart the daemon.`
  }
  const flags = names.map((name) => `--forward-env ${name}`).join(' ')
  return `Set ${joined} in the current shell, then run: sepilot restart ${flags}`
}

function nonEmpty(value: string | null | undefined): boolean {
  return typeof value === 'string' && value.trim().length > 0
}

function assistantAgentRouting(
  mode: string | null,
  graphs: readonly DaemonAgentGraphSnapshot[] | null,
  graph: DaemonAgentGraphSnapshot | null,
): AssistantAgentRouting {
  if (mode === null) return { kind: 'unavailable', detail: 'mode-unavailable' }
  if (mode === 'auto') return { kind: 'dynamic', detail: 'per-request' }
  if (mode === 'react') return { kind: 'direct', detail: 'react-loop' }
  if (mode === 'instant') return { kind: 'direct', detail: 'instant-react-loop' }
  if (graphs === null) return { kind: 'unavailable', detail: 'graph-catalog-unavailable' }
  if (graph) return { kind: 'graph', detail: 'registered-graph' }
  return { kind: 'unavailable', detail: 'configured-mode-missing' }
}

function deliveryRouteReadiness(
  channelType: string,
  configuredChannels: readonly DaemonChannelConfig[],
  summaries: ReadonlyMap<string, ChannelRuntimeSummary>,
  partialFailures: ReadonlySet<string>,
): DeliveryRouteReadiness {
  const configured = configuredChannels.find((channel) => channel.type === channelType)
  if (!configured || configured.enabled === false) return 'unavailable'

  // Only channel types with a daemon runtime probe can prove connection
  // state. Other configured channel types remain usable under the existing
  // config contract instead of being mislabeled unavailable.
  if (!summaries.has(channelType)) return 'available'
  if (partialFailures.has(`channel.${channelType}`)) return 'unknown'

  const summary = summaries.get(channelType)
  if (!summary) return 'unavailable'
  return summary.enabled && summary.status === 'connected' ? 'available' : 'unavailable'
}

export function buildAssistantStatusSnapshot(input: {
  health: DaemonHealth
  runtime: DaemonAssistantRuntimeStatus | null
  config: DaemonConfig | null
  agents: DaemonAgentDescriptor[] | null
  graphs: DaemonAgentGraphSnapshot[] | null
  schedules: DaemonSchedulerJob[] | null
  backgroundJobs: DaemonChatBackgroundListItem[] | null
  sessionManagement: DaemonSessionManagementSnapshot | null
  notifications: DaemonNotificationItem[] | null
  channelSummaries: ReadonlyMap<string, ChannelRuntimeSummary>
  partialFailures: string[]
  partialFailureKinds?: ReadonlyMap<string, OptionalFailureKind>
  daemonEndpointScope: DaemonEndpointScope
  limit?: number
}): AssistantStatusSnapshot {
  const limit = input.limit ?? 5
  const mode = input.config?.agent.mode ?? null
  const descriptor = input.agents?.find((agent) => agent.id === mode) ?? null
  const graph = input.graphs?.find((candidate) => candidate.id === mode) ?? null
  const routing = assistantAgentRouting(mode, input.graphs, graph)
  const requiredSkills = input.runtime?.skills
    ?? REQUIRED_SKILL_IDS.map((id) => ({ id, available: false, enabled: false }))
  const requiredTools = input.runtime?.tools ?? []
  const schedules = input.schedules ?? []
  const activeFailedSchedules = input.schedules?.filter(
    (job) => job.enabled && (job.status === 'failed' || Boolean(job.lastError)),
  ) ?? []
  const attendedFailedSchedules = activeFailedSchedules.filter((job) => job.unattended !== true)
  const backgroundJobs = (input.backgroundJobs ?? [])
    .filter((job) => job.status === 'running')
  const notifications = input.notifications ?? []
  const recentNotifications = notifications.slice(0, limit)
  const notificationInventory = input.runtime?.operations?.notifications
  const notificationsTruncated = notificationInventory
    ? notificationInventory.total > recentNotifications.length
    : notifications.length > recentNotifications.length
      ? true
      : null
  const interactions = input.runtime?.operations?.interactions
  const schedulerDeliveryOutbox = input.runtime?.operations?.schedulerDeliveryOutbox ?? null
  const schedulerDeliveryEvidence = daemonSchedulerDeliveryOutboxEvidence(schedulerDeliveryOutbox)
  const channels = (input.config?.channels ?? [])
    .map((channel) => channelStatus(channel, input.channelSummaries))
  const activeRouteCandidates = input.schedules?.filter((job) => (
    job.enabled
    && (
      nonEmpty(job.channelType)
      || nonEmpty(job.channelTarget)
      || nonEmpty(job.replyToMessageId)
    )
  )) ?? []
  const invalidDeliveryRoutes = activeRouteCandidates.filter((job) => (
    !nonEmpty(job.channelType)
    || !nonEmpty(job.channelTarget)
  ))
  const routedSchedules = activeRouteCandidates.filter((job) => (
    nonEmpty(job.channelType)
    && nonEmpty(job.channelTarget)
  ))
  const deliveryRouteReadinessCounts = {
    available: 0,
    unavailable: 0,
    unknown: 0,
  }
  if (input.config) {
    const partialFailures = new Set(input.partialFailures)
    for (const job of routedSchedules) {
      const readiness = deliveryRouteReadiness(
        job.channelType!.trim(),
        input.config.channels ?? [],
        input.channelSummaries,
        partialFailures,
      )
      deliveryRouteReadinessCounts[readiness] += 1
    }
  }
  const attention: AssistantStatusSnapshot['attention'] = []
  const authenticationFailures = input.partialFailures.filter(
    (section) => input.partialFailureKinds?.get(section) === 'authentication',
  )
  const authorizationFailures = input.partialFailures.filter(
    (section) => input.partialFailureKinds?.get(section) === 'authorization',
  )
  const otherPartialFailures = input.partialFailures.filter((section) => {
    const kind = input.partialFailureKinds?.get(section)
    return kind === undefined || kind === 'other'
  })

  if (input.health.status !== 'ok') {
    attention.push({
      code: 'DAEMON_DEGRADED',
      message: `Daemon health is ${input.health.status}.`,
      action: 'sepilot status --report',
    })
  }
  if (routing.detail === 'configured-mode-missing') {
    attention.push({
      code: 'AGENT_MODE_NOT_REGISTERED',
      message: `Configured agent mode ${mode} is not present in the available graph catalog.`,
      action: 'sepilot config-set agent.mode auto',
    })
  }
  if (input.config && !input.config.channelPipeline?.defaultWorkspaceRoot) {
    attention.push({
      code: 'WORKSPACE_ROOT_MISSING',
      message: 'Channel development requests do not have a default workspace collection root.',
      action: 'sepilot config-set channelPipeline.defaultWorkspaceRoot ~/workspace',
    })
  }
  if (input.config) {
    const mattermost = channels.find((channel) => channel.type === 'mattermost')
    if (!mattermost) {
      attention.push({
        code: 'MATTERMOST_NOT_CONFIGURED',
        message: 'Mattermost is not configured.',
        action: input.daemonEndpointScope === 'loopback'
          ? 'Set MATTERMOST_SERVER_URL, MATTERMOST_BOT_TOKEN, and MATTERMOST_WEBHOOK_TOKEN in the CLI process, then run: sepilot channel add mattermost --from-env'
          : 'Configure SEPILOTD_MATTERMOST_FROM_ENV=1, MATTERMOST_SERVER_URL, MATTERMOST_BOT_TOKEN, MATTERMOST_WEBHOOK_TOKEN, and a SEPILOTD_MATTERMOST_ALLOWED_* allowlist at daemon runtime, then restart the daemon.',
      })
    } else if (!mattermost.enabled || mattermost.status !== 'connected') {
      attention.push({
        code: 'MATTERMOST_NOT_CONNECTED',
        message: `Mattermost is ${mattermost.enabled ? mattermost.status : 'disabled'}.`,
        action: 'sepilot channel list',
      })
    }
  }
  if (input.runtime?.integrations.scheduler.enabled === false) {
    attention.push({
      code: 'SCHEDULER_DISABLED',
      message: 'The internal scheduler is disabled.',
      action: 'sepilot config-set scheduler.enabled true',
    })
  } else if (input.runtime?.integrations.scheduler.cliSurfaceEnabled === false) {
    attention.push({
      code: 'SCHEDULER_CLI_DISABLED',
      message: 'The scheduler is running, but its CLI surface is disabled.',
      action: 'sepilot config-set scheduler.surfaces.cli true',
    })
  }
  if (activeFailedSchedules.length > 0) {
    attention.push({
      code: 'SCHEDULER_JOB_FAILURES',
      message: `${activeFailedSchedules.length} enabled scheduled ${activeFailedSchedules.length === 1 ? 'task has' : 'tasks have'} a failed status or recorded error.`,
      action: 'sepilot schedule list --all',
    })
  }
  if (attendedFailedSchedules.length > 0) {
    attention.push({
      code: 'SCHEDULER_FAILED_JOBS_ATTENDED',
      message: `${attendedFailedSchedules.length} failed enabled scheduled ${attendedFailedSchedules.length === 1 ? 'task is' : 'tasks are'} configured for attended execution; this does not prove approval caused the failure.`,
      action: 'Inspect with: sepilot schedule show <id>. Only after explicit future-run authorization: sepilot schedule unattended <id> on',
    })
  }
  if (invalidDeliveryRoutes.length > 0) {
    attention.push({
      code: 'SCHEDULER_DELIVERY_ROUTE_INVALID',
      message: `${invalidDeliveryRoutes.length} enabled scheduled ${invalidDeliveryRoutes.length === 1 ? 'task has' : 'tasks have'} an incomplete durable delivery route.`,
      action: 'Inspect with: sepilot schedule list --all. Repair with: sepilot schedule route <id> <channel-type> <channel-target>, or clear with: sepilot schedule route <id> clear',
    })
  }
  if (deliveryRouteReadinessCounts.unavailable > 0) {
    attention.push({
      code: 'SCHEDULER_DELIVERY_CHANNEL_UNAVAILABLE',
      message: `${deliveryRouteReadinessCounts.unavailable} enabled channel-routed scheduled ${deliveryRouteReadinessCounts.unavailable === 1 ? 'task has' : 'tasks have'} no available configured connector.`,
      action: 'Inspect with: sepilot channel list and sepilot schedule list --all.',
    })
  }
  if (schedulerDeliveryEvidence?.state === 'retrying' && schedulerDeliveryOutbox) {
    attention.push({
      code: 'SCHEDULER_CHANNEL_DELIVERIES_RETRYING',
      message: `${schedulerDeliveryOutbox.failed} scheduled channel ${schedulerDeliveryOutbox.failed === 1 ? 'delivery is' : 'deliveries are'} waiting for retry${schedulerDeliveryOutbox.nextRetryAt == null ? '' : ` at or after ${new Date(schedulerDeliveryOutbox.nextRetryAt).toISOString()}`}.`,
      action: 'Inspect channel connectivity and daemon logs; the completed scheduler run will not be rerun.',
    })
  } else if (schedulerDeliveryEvidence?.state === 'pending') {
    const active = schedulerDeliveryEvidence.active
    attention.push({
      code: 'SCHEDULER_CHANNEL_DELIVERIES_PENDING',
      message: `${active} scheduled channel ${active === 1 ? 'delivery is' : 'deliveries are'} pending or in progress.`,
      action: 'Re-run assistant status to confirm delivery progress; the completed scheduler run will not be rerun.',
    })
  }
  if (input.runtime) {
    for (const skill of requiredSkills) {
      if (!skill.enabled) {
        attention.push({
          code: skill.available ? 'ASSISTANT_SKILL_DISABLED' : 'ASSISTANT_SKILL_MISSING',
          message: `${skill.id} is ${skill.available ? 'disabled' : 'unavailable'}.`,
          action: skill.available ? `sepilot skills enable ${skill.id}` : undefined,
        })
      }
    }
  }
  const disabledTools = requiredTools.filter((tool) => !tool.enabled)
  if (disabledTools.length > 0) {
    attention.push({
      code: 'ASSISTANT_TOOLS_DISABLED',
      message: `${disabledTools.length} required assistant tools are disabled: ${disabledTools.map((tool) => tool.name).join(', ')}.`,
      action: 'Review agent.disabledTools and the build feature manifest.',
    })
  }
  if (input.runtime) {
    const notifyRelay = input.runtime.integrations.notifyRelay
    const latestDelivery = notifyRelay.latestDelivery ?? null
    const relayOutbox = notifyRelay.outbox ?? null
    const providerDelivery = notifyRelay.providerDelivery ?? null
    if (!notifyRelay.configured) {
      const relayEnvNames = [
        'NOTIFY_RELAY_BASE_URL',
        'NOTIFY_RELAY_DESTINATION_ID',
        'NOTIFY_RELAY_TOKEN',
      ] as const
      attention.push({
        code: 'NOTIFY_RELAY_NOT_CONFIGURED',
        message: 'External notification relay delivery is not configured.',
        action: runtimeEnvironmentRecoveryAction(input.daemonEndpointScope, relayEnvNames),
      })
    } else if (notifyRelay.deliveryMode === 'best_effort_untracked') {
      attention.push({
        code: 'NOTIFY_RELAY_RECEIPTS_UNAVAILABLE',
        message: 'The daemon attempts relay delivery but does not expose acceptance evidence.',
        action: 'Upgrade and restart the daemon, then publish a test notification.',
      })
    } else if (!latestDelivery) {
      attention.push({
        code: 'NOTIFY_RELAY_NO_ACCEPTANCE_EVIDENCE',
        message: 'Notification Relay is configured, but no completed relay attempt is recorded.',
        action: 'Publish an approved test notification and run assistant status again.',
      })
    } else if (
      latestDelivery.status === 'pending_review'
      && (
        latestDelivery.providerDelivery == null
        || latestDelivery.providerDelivery.status === 'pending_review'
      )
    ) {
      attention.push({
        code: 'NOTIFY_RELAY_PENDING_REVIEW',
        message: 'The latest relay request is pending review.',
        action: 'Review the request in Notification Relay using the recorded message ID.',
      })
    } else if (
      latestDelivery.status === 'rejected'
      || latestDelivery.status === 'unreachable'
      || latestDelivery.status === 'invalid_response'
    ) {
      attention.push({
        code: 'NOTIFY_RELAY_LAST_ATTEMPT_FAILED',
        message: `The latest relay attempt ended with ${latestDelivery.status}${latestDelivery.httpStatus == null ? '' : ` (HTTP ${latestDelivery.httpStatus})`}${latestDelivery.errorCode ? `, code=${latestDelivery.errorCode}` : ''}.`,
        action: notifyRelayFailureAction(latestDelivery),
      })
    }
    if (latestDelivery?.providerDelivery?.status === 'delivery_failed') {
      attention.push({
        code: 'NOTIFY_RELAY_PROVIDER_DELIVERY_RETRYING',
        message: 'Notification Relay reports that the latest provider attempt failed and remains non-terminal.',
        action: 'Inspect Notification Relay delivery health; its own outbox may still retry the message.',
      })
    }
    if (relayOutbox && relayOutbox.failed > 0) {
      attention.push({
        code: 'NOTIFY_RELAY_DELIVERIES_RETRYING',
        message: `${relayOutbox.failed} durable relay ${relayOutbox.failed === 1 ? 'delivery is' : 'deliveries are'} waiting for retry${relayOutbox.nextRetryAt == null ? '' : ` at or after ${new Date(relayOutbox.nextRetryAt).toISOString()}`}.`,
        action: 'Inspect daemon and Notification Relay connectivity; the daemon will retry with the original idempotency key.',
      })
    }
    if (relayOutbox && relayOutbox.dead > 0) {
      attention.push({
        code: 'NOTIFY_RELAY_DELIVERIES_DEAD',
        message: `${relayOutbox.dead} durable relay ${relayOutbox.dead === 1 ? 'delivery has' : 'deliveries have'} reached a terminal state.`,
        action: 'Inspect daemon and Notification Relay logs and runtime-managed credentials before publishing a newly authorized event.',
      })
    }
    if (providerDelivery && providerDelivery.retrying > 0) {
      attention.push({
        code: 'NOTIFY_RELAY_PROVIDER_STATUS_RETRYING',
        message: `${providerDelivery.retrying} Relay provider status ${providerDelivery.retrying === 1 ? 'check is' : 'checks are'} waiting for retry${providerDelivery.nextCheckAt == null ? '' : ` at or after ${new Date(providerDelivery.nextCheckAt).toISOString()}`}.`,
        action: 'Inspect daemon-to-Relay connectivity and runtime-managed credentials; status checks resume automatically.',
      })
    }
    if (providerDelivery && providerDelivery.failed > 0) {
      attention.push({
        code: 'NOTIFY_RELAY_PROVIDER_DELIVERIES_FAILED',
        message: `${providerDelivery.failed} Relay ${providerDelivery.failed === 1 ? 'message has' : 'messages have'} a confirmed terminal provider failure.`,
        action: 'Inspect the recorded Relay message status and destination health before publishing a newly authorized event.',
      })
    }
    if (providerDelivery && providerDelivery.unconfirmed > 0) {
      attention.push({
        code: 'NOTIFY_RELAY_PROVIDER_STATUS_UNCONFIRMED',
        message: `${providerDelivery.unconfirmed} Relay ${providerDelivery.unconfirmed === 1 ? 'message ended' : 'messages ended'} without confirmable provider status.`,
        action: 'Inspect Notification Relay retention, response compatibility, and client-token ownership.',
      })
    }
  }
  if (input.runtime && !input.runtime.integrations.aiSearch.configured) {
    attention.push({
      code: 'AI_SEARCH_NOT_CONFIGURED',
      message: 'The internal AI Search endpoint is not configured.',
      action: 'Configure webSearch.endpoint or AI_SEARCH_URL at daemon runtime.',
    })
  }
  if (input.runtime && !input.runtime.integrations.jpad.configured) {
    attention.push({
      code: 'JPAD_NOT_CONFIGURED',
      message: 'JPAD publishing is not configured.',
      action: runtimeEnvironmentRecoveryAction(
        input.daemonEndpointScope,
        ['JPAD_PERSONAL_API_TOKEN'],
      ),
    })
  } else if (
    input.runtime?.operations?.jpadPublications?.status === 'partial'
    || input.runtime?.operations?.jpadPublications?.status === 'unavailable'
  ) {
    attention.push({
      code: 'JPAD_PUBLICATION_EVIDENCE_INCOMPLETE',
      message: `Recent JPAD publication evidence is ${input.runtime.operations.jpadPublications.status}.`,
      action: 'Inspect session storage health and retry assistant status.',
    })
  }
  if (authenticationFailures.length > 0) {
    attention.push({
      code: 'DAEMON_AUTHENTICATION_FAILED',
      message: `The daemon rejected the current CLI credentials for: ${authenticationFailures.join(', ')}.`,
      action: input.daemonEndpointScope === 'loopback'
        ? 'Verify SEPILOTD_URL and SEPILOTD_DATA_DIR select the same local daemon profile, then stop the conflicting profile with its matching launcher and retry. Do not copy token contents between profiles.'
        : 'Verify the daemon URL and its runtime-managed bearer token. Do not replace it with a token from a local daemon profile.',
    })
  }
  if (authorizationFailures.length > 0) {
    attention.push({
      code: 'DAEMON_AUTHORIZATION_FAILED',
      message: `The current daemon credentials do not authorize: ${authorizationFailures.join(', ')}.`,
      action: 'Verify the configured daemon token scopes and endpoint permissions without exposing or copying the token into diagnostics.',
    })
  }
  if (otherPartialFailures.length > 0) {
    attention.push({
      code: 'SNAPSHOT_PARTIAL',
      message: `Some assistant status sections were unavailable: ${otherPartialFailures.join(', ')}.`,
      action: 'Upgrade/restart the daemon and retry assistant status.',
    })
  }

  const toolFailure = disabledTools.length > 0
  const accessFailure = authenticationFailures.length > 0 || authorizationFailures.length > 0
  const overall: AssistantOverallStatus = input.health.status !== 'ok' || toolFailure || accessFailure
    ? 'degraded'
    : attention.length > 0
      ? 'attention'
      : 'ready'

  return {
    generatedAt: new Date().toISOString(),
    overall,
    daemon: {
      status: input.health.status,
      version: input.health.version,
    },
    agent: {
      mode,
      autonomy: input.config?.agent.autonomy ?? null,
      provider: input.config?.agent.defaultProvider ?? null,
      model: input.config?.agent.defaultModel ?? null,
      descriptor,
      graph: graph
        ? { id: graph.id, nodeCount: graph.nodeCount, edgeCount: graph.edgeCount }
        : null,
      routing,
    },
    workspace: {
      defaultRoot: input.config?.channelPipeline?.defaultWorkspaceRoot ?? null,
    },
    capabilities: {
      requiredTools,
      enabledRequiredTools: requiredTools.filter((tool) => tool.enabled).length,
      requiredSkills,
    },
    channels,
    scheduler: {
      configured: input.runtime?.integrations.scheduler.enabled ?? null,
      accessible: input.runtime?.integrations.scheduler.cliSurfaceEnabled ?? null,
      total: schedules.length,
      enabled: schedules.filter((job) => job.enabled).length,
      running: schedules.filter((job) => job.status === 'running').length,
      failed: activeFailedSchedules.length,
      deliveryRoutes: {
        routed: routedSchedules.length,
        invalid: invalidDeliveryRoutes.length,
        unavailable: deliveryRouteReadinessCounts.unavailable,
        unknown: input.config ? deliveryRouteReadinessCounts.unknown : routedSchedules.length,
      },
      channelOutbox: schedulerDeliveryOutbox,
      next: schedules
        .filter(hasPendingScheduledRun)
        .sort((left, right) => left.nextRunAt - right.nextRunAt)
        .slice(0, limit)
        .map((job) => ({
          id: job.id,
          name: job.name,
          nextRunAt: job.nextRunAt,
          status: job.status,
        })),
      recentRuns: input.runtime?.operations?.schedulerRuns.slice(0, limit) ?? [],
    },
    background: {
      active: backgroundJobs.length,
      waitingApproval: backgroundJobs.filter((job) => backgroundAction(job) === 'approval').length,
      waitingQuestion: backgroundJobs.filter((job) => backgroundAction(job) === 'question').length,
      jobs: backgroundJobs.slice(0, limit).map((job) => ({
        jobId: job.jobId,
        sessionId: job.sessionId,
        action: backgroundAction(job),
        label: job.progress?.label ?? null,
      })),
    },
    approvals: {
      pending: interactions?.pendingApprovals.total
        ?? input.sessionManagement?.pendingApprovals.total
        ?? null,
      orphaned: interactions?.pendingApprovals.orphaned
        ?? input.sessionManagement?.pendingApprovals.orphaned
        ?? null,
      pendingQuestions: interactions?.pendingQuestions.total
        ?? input.sessionManagement?.pendingQuestions.total
        ?? null,
    },
    notifications: {
      total: notificationInventory?.total ?? null,
      unread: notificationInventory?.unread ?? null,
      listed: recentNotifications.length,
      countsExact: notificationInventory !== undefined,
      truncated: notificationsTruncated,
      relayConfigured: input.runtime?.integrations.notifyRelay.configured ?? null,
      deliveryMode: input.runtime?.integrations.notifyRelay.deliveryMode ?? null,
      latestRelayDelivery: input.runtime?.integrations.notifyRelay.latestDelivery ?? null,
      relayOutbox: input.runtime?.integrations.notifyRelay.outbox ?? null,
      relayProviderDelivery: input.runtime?.integrations.notifyRelay.providerDelivery ?? null,
      recent: recentNotifications.map((item) => ({
        id: item.id,
        title: item.title,
        topic: item.topic ?? null,
        createdAt: item.createdAt,
        read: item.readAt != null,
      })),
    },
    research: {
      aiSearchConfigured: input.runtime?.integrations.aiSearch.configured ?? null,
      aiSearchSelected: input.runtime?.integrations.aiSearch.selected ?? null,
      searchProvider: input.runtime?.integrations.aiSearch.provider ?? null,
      jpadConfigured: input.runtime?.integrations.jpad.configured ?? null,
      jpadPublicationEvidence: input.runtime?.operations?.jpadPublications?.status ?? null,
      recentJpadPublications:
        input.runtime?.operations?.jpadPublications?.items.slice(0, limit) ?? [],
    },
    attention,
    partialFailures: input.partialFailures,
  }
}

function value<T>(result: OptionalResult<T>): T | null {
  return result.available ? result.value : null
}

export function formatAssistantStatusSnapshot(snapshot: AssistantStatusSnapshot): string {
  const status = snapshot.overall === 'ready'
    ? chalk.green(snapshot.overall)
    : snapshot.overall === 'degraded'
      ? chalk.red(snapshot.overall)
      : chalk.yellow(snapshot.overall)
  const mode = snapshot.agent.descriptor
    ? `${snapshot.agent.mode} (${snapshot.agent.descriptor.name})`
    : snapshot.agent.mode ?? 'unavailable'
  const routing = snapshot.agent.routing.kind === 'graph'
    ? snapshot.agent.graph
      ? `graph=${snapshot.agent.graph.nodeCount} nodes / ${snapshot.agent.graph.edgeCount} edges`
      : 'routing=unavailable (graph details unavailable)'
    : snapshot.agent.routing.kind === 'dynamic'
      ? 'routing=dynamic (per request)'
      : snapshot.agent.routing.kind === 'direct'
        ? snapshot.agent.routing.detail === 'instant-react-loop'
          ? 'routing=direct (instant ReAct loop)'
          : 'routing=direct (ReAct loop)'
        : `routing=unavailable (${snapshot.agent.routing.detail.replaceAll('-', ' ')})`
  const notificationCounts = snapshot.notifications.countsExact
    ? `${snapshot.notifications.unread}/${snapshot.notifications.total} unread${snapshot.notifications.truncated ? ` · ${snapshot.notifications.listed} listed` : ''}`
    : `${snapshot.notifications.listed} listed · exact unread/total unavailable`
  const relayOutbox = snapshot.notifications.relayOutbox
  const relayOutboxStatus = relayOutbox
    ? ` · outbox=${relayOutbox.pending + relayOutbox.delivering + relayOutbox.failed} active/${relayOutbox.dead} dead/${relayOutbox.delivered} handed-off`
    : ''
  const latestProviderDelivery = snapshot.notifications.latestRelayDelivery?.providerDelivery
  const providerStatus = latestProviderDelivery?.status
    ?? (latestProviderDelivery
      ? `lookup-${latestProviderDelivery.lookupStatus}`
      : snapshot.notifications.relayProviderDelivery
        && snapshot.notifications.latestRelayDelivery?.messageId
        ? 'pending'
        : 'none')
  const providerSummary = snapshot.notifications.relayProviderDelivery
  const providerSummaryStatus = providerSummary
    ? ` · provider-tracking=${providerSummary.pending + providerSummary.checking} pending/${providerSummary.retrying} retrying/${providerSummary.delivered} delivered/${providerSummary.failed} failed/${providerSummary.unconfirmed} unconfirmed`
    : ''
  const schedulerChannelOutbox = snapshot.scheduler.channelOutbox
  const schedulerChannelOutboxStatus = schedulerChannelOutbox
    ? ` · channel-outbox=${schedulerChannelOutbox.pending + schedulerChannelOutbox.delivering + schedulerChannelOutbox.failed} active/${schedulerChannelOutbox.delivered} delivered`
    : ' · channel-outbox=unavailable'
  const lines = [
    `${chalk.bold('Personal assistant')} — ${status}`,
    `Daemon: ${snapshot.daemon.version} · ${snapshot.daemon.status}`,
    `Agent: mode=${mode} · ${routing} · autonomy=${snapshot.agent.autonomy ?? 'unavailable'}`,
    `Workspace: ${snapshot.workspace.defaultRoot ?? chalk.yellow('not configured')}`,
    `Capabilities: tools ${snapshot.capabilities.enabledRequiredTools}/${snapshot.capabilities.requiredTools.length} · skills ${snapshot.capabilities.requiredSkills.filter((skill) => skill.enabled).length}/${snapshot.capabilities.requiredSkills.length}`,
    '',
    chalk.bold('Channels'),
    ...(snapshot.channels.length > 0
      ? snapshot.channels.map((channel) => `  ${channel.type}: ${channel.enabled ? channel.status : 'disabled'}`)
      : [chalk.gray('  none configured')]),
    '',
    chalk.bold('Operations'),
    `  schedules: ${snapshot.scheduler.enabled}/${snapshot.scheduler.total} enabled · ${snapshot.scheduler.running} running · ${snapshot.scheduler.failed} failed · routes=${snapshot.scheduler.deliveryRoutes.routed} routed/${snapshot.scheduler.deliveryRoutes.invalid} invalid/${snapshot.scheduler.deliveryRoutes.unavailable} unavailable/${snapshot.scheduler.deliveryRoutes.unknown} unknown · CLI=${snapshot.scheduler.accessible == null ? 'unavailable' : snapshot.scheduler.accessible ? 'enabled' : 'disabled'}${schedulerChannelOutboxStatus}`,
    `  background: ${snapshot.background.active} active · ${snapshot.background.waitingApproval} approvals · ${snapshot.background.waitingQuestion} questions`,
    `  pending interactions: ${snapshot.approvals.pending ?? 'unavailable'} approvals · ${snapshot.approvals.pendingQuestions ?? 'unavailable'} questions · ${snapshot.approvals.orphaned ?? 'unavailable'} orphaned approvals`,
    `  notifications: ${notificationCounts} · relay=${snapshot.notifications.relayConfigured == null ? 'unavailable' : snapshot.notifications.relayConfigured ? 'configured' : 'local-only'} · mode=${snapshot.notifications.deliveryMode ?? 'unavailable'} · acceptance=${snapshot.notifications.latestRelayDelivery?.status ?? (snapshot.notifications.deliveryMode === 'best_effort_untracked' ? 'untracked' : 'none')} · provider=${providerStatus}${snapshot.notifications.latestRelayDelivery?.messageId ? ` · relay-message=${snapshot.notifications.latestRelayDelivery.messageId}` : ''}${relayOutboxStatus}${providerSummaryStatus}`,
    '',
    chalk.bold('Research'),
    `  AI Search: ${snapshot.research.aiSearchConfigured == null ? 'unavailable' : snapshot.research.aiSearchConfigured ? snapshot.research.aiSearchSelected ? 'configured/selected' : 'configured/not selected' : 'not configured'}${snapshot.research.searchProvider ? ` (${snapshot.research.searchProvider})` : ''}`,
    `  JPAD: ${snapshot.research.jpadConfigured == null ? 'unavailable' : snapshot.research.jpadConfigured ? 'configured' : 'not configured'} · publication-evidence=${snapshot.research.jpadPublicationEvidence ?? 'unavailable'} · recent=${snapshot.research.recentJpadPublications.length}`,
  ]

  if (snapshot.scheduler.next.length > 0) {
    lines.push('', chalk.bold('Next schedules'))
    for (const job of snapshot.scheduler.next) {
      lines.push(`  ${job.name} (${job.id.slice(0, 8)}): ${new Date(job.nextRunAt).toISOString()} · ${job.status}`)
    }
  }
  if (snapshot.scheduler.recentRuns.length > 0) {
    lines.push('', chalk.bold('Recent scheduler delivery evidence'))
    for (const correlation of snapshot.scheduler.recentRuns) {
      const relayNotification = correlation.notifications
        .filter((notification) => notification.relayDelivery !== null)
        .sort((left, right) => (
          right.relayDelivery!.completedAt - left.relayDelivery!.completedAt
        ))[0]
      const relayReceipt = relayNotification?.relayDelivery
      const channelEvidence = correlation.channelDeliveries.length > 0
        ? correlation.channelDeliveries
            .map((delivery) => `${delivery.channelType}:${delivery.status}`)
            .join(',')
        : 'none'
      const relayEvidence = relayReceipt
        ? `${relayReceipt.status}${relayReceipt.messageId ? `:${relayReceipt.messageId}` : ''}`
        : 'none'
      const providerEvidence = relayNotification?.relayProviderDelivery
      lines.push(
        `  ${correlation.jobName ?? correlation.jobId} · run=${correlation.run.id} · ${correlation.run.status} · notification=${correlation.notifications.length} · channel=${channelEvidence} · relay=${relayEvidence} · provider=${providerEvidence?.status ?? (providerEvidence ? `lookup-${providerEvidence.lookupStatus}` : 'none')}`,
      )
    }
  }
  if (snapshot.background.jobs.length > 0) {
    lines.push('', chalk.bold('Active background work'))
    for (const job of snapshot.background.jobs) {
      lines.push(`  ${job.jobId}: ${job.action} · session=${job.sessionId}${job.label ? ` · ${job.label}` : ''}`)
    }
  }
  if (snapshot.notifications.recent.length > 0) {
    lines.push('', chalk.bold('Recent notifications'))
    for (const notification of snapshot.notifications.recent) {
      lines.push(`  ${notification.read ? '○' : '●'} ${notification.title}${notification.topic ? ` · ${notification.topic}` : ''}`)
    }
  }
  if (snapshot.research.recentJpadPublications.length > 0) {
    lines.push('', chalk.bold('Recent JPAD publication evidence'))
    for (const publication of snapshot.research.recentJpadPublications) {
      lines.push(
        `  session=${publication.sessionId} · call=${publication.toolCallId} · ${publication.operation} · ${publication.outcome} · page=${publication.pageId ?? 'unconfirmed'}`,
      )
    }
  }
  if (snapshot.notifications.deliveryMode === 'best_effort_untracked') {
    lines.push('', chalk.gray('External relay acceptance/delivery receipts are not persisted yet; configured means send attempts are enabled, not confirmed delivered.'))
  } else if (
    snapshot.notifications.deliveryMode === 'acceptance_receipts'
    || snapshot.notifications.deliveryMode === 'durable_outbox'
  ) {
    lines.push('', chalk.gray(
      snapshot.notifications.relayProviderDelivery
        ? 'Relay acceptance and provider status are tracked separately; only provider=delivered proves final provider delivery.'
        : 'accepted/pending_review records Notification Relay API acceptance only; confirm final provider delivery in Notification Relay using the message ID.',
    ))
  }
  if (snapshot.attention.length > 0) {
    lines.push('', chalk.bold('Attention'))
    for (const item of snapshot.attention) {
      lines.push(chalk.yellow(`  [${item.code}] ${item.message}`))
      if (item.action) lines.push(chalk.gray(`    ${item.action}`))
    }
  }
  return lines.join('\n')
  }
export async function assistantStatusCommand(options: {
  url?: string
  limit?: string
}): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    await ensureDaemon(client, { url: options.url, quiet: true })
    const health = await client.health()
    // Runtime status and the remaining readiness reads are independent. Start
    // them together so a slow journal-backed runtime projection does not add
    // its latency to another slow optional endpoint. Scheduler remains gated
    // on the runtime capability result below, preserving disabled-surface
    // semantics instead of turning an intentional 403 into a partial failure.
    const runtimePromise = optional('assistant-runtime', client.assistantRuntimeStatus())
    const independentResultsPromise = Promise.all([
      optional('config', client.config()),
      optional('agents', client.agents()),
      optional('agent-graphs', client.agentGraphs()),
      optional('background', client.backgroundChatJobs()),
      optional('notifications', client.notifications()),
      optional('channel.telegram', client.telegramChannel()),
      optional('channel.mattermost', client.mattermostChannel()),
      optional('channel.discord', client.discordChannel()),
      optional('channel.slack', client.slackChannel()),
    ] as const)
    const runtime = await runtimePromise
    const runtimeStatus = value(runtime)
    const schedulerAvailable = runtimeStatus === null
      || (runtimeStatus.integrations.scheduler.enabled
        && runtimeStatus.integrations.scheduler.cliSurfaceEnabled)
    const [independentResults, schedules, sessionManagement] = await Promise.all([
      independentResultsPromise,
      optional(
        'scheduler',
        schedulerAvailable ? client.listScheduledTasks({ all: true }) : Promise.resolve([]),
      ),
      runtimeStatus?.operations?.interactions
        ? Promise.resolve({
            section: 'session-management',
            value: null,
            available: true,
            failureKind: null,
          } satisfies OptionalResult<DaemonSessionManagementSnapshot>)
        : optional('session-management', client.sessionManagement()),
    ] as const)
    const [
      config,
      agents,
      graphs,
      background,
      notifications,
      telegram,
      mattermost,
      discord,
      slack,
    ] = independentResults
    const channelSummaries = new Map<string, ChannelRuntimeSummary>([
      ['telegram', value(telegram)],
      ['mattermost', value(mattermost)],
      ['discord', value(discord)],
      ['slack', value(slack)],
    ])
    // Keep the historical section ordering stable even though acquisition is
    // now concurrent; JSON/text automation may compare the partial list.
    const optionalResults = [
      runtime,
      config,
      agents,
      graphs,
      schedules,
      background,
      sessionManagement,
      notifications,
      telegram,
      mattermost,
      discord,
      slack,
    ]
    const unavailableResults = optionalResults.filter((result) => !result.available)
    const snapshot = buildAssistantStatusSnapshot({
      health,
      runtime: runtimeStatus,
      config: value(config),
      agents: value(agents),
      graphs: value(graphs),
      schedules: value(schedules),
      backgroundJobs: value(background)?.jobs ?? null,
      sessionManagement: value(sessionManagement),
      notifications: value(notifications),
      channelSummaries,
      partialFailures: unavailableResults.map((result) => result.section),
      partialFailureKinds: new Map(
        unavailableResults.map((result) => [result.section, result.failureKind ?? 'other']),
      ),
      daemonEndpointScope: resolveDaemonEndpointScope(options.url),
      limit: parseLimit(options.limit),
    })
    output(snapshot, formatAssistantStatusSnapshot)
  } catch (error) {
    outputError(
      { ok: false, error: 'assistant-status-failed' },
      () => chalk.red(`Failed to build assistant status: ${friendlyErrorMessage(error)}`),
    )
    process.exit(1)
  }
}

export const __testables = {
  parseLimit,
  backgroundAction,
  channelStatus,
  deliveryRouteReadiness,
  optionalFailureKind,
}
