import type {
  ActivityItem,
  Message,
} from './chat-surface-types.js'
import type {
  DaemonSessionDetail,
  DaemonUsage,
} from './types.js'
import type { GatewayTicketWatchRevalidateHealth } from '../gateway/http.js'
import { createGatewayWatchActivity } from '../gateway/watch-surface.js'

export interface ChatMetricsInput {
  messages: Message[]
  activities: ActivityItem[]
  sessionDetail: DaemonSessionDetail | null
  selectedProvider: string
  selectedModel: string
  gatewayWatchHealth?: GatewayTicketWatchRevalidateHealth | null
}

export interface ChatMetrics {
  toolMessages: Message[]
  lastUsage: DaemonUsage | Message['usage'] | undefined
  pendingApprovals: number
  sessionTokenCount: number
  recentActivities: ActivityItem[]
  activityCount: number
  canResumeRun: boolean
  runContract: DaemonSessionDetail['runContract'] | null
  contextEngine: DaemonSessionDetail['contextEngine'] | null
  evidenceManifest: DaemonSessionDetail['evidenceManifest'] | null
  evaluationGate: DaemonSessionDetail['evaluationGate'] | null
  providerFallbackCount: number
  providerAttemptCount: number
  delegation: DaemonSessionDetail['delegation'] | undefined
  delegationMeta: string | null
  nonToolMessageCount: number
  lastUsageTokenCount: number | undefined
  costLabel: string
  updatedLabel: string
  providerLabel: string
  modelLabel: string
}

export function computeChatMetrics({
  messages,
  activities,
  sessionDetail,
  selectedProvider,
  selectedModel,
  gatewayWatchHealth = null,
}: ChatMetricsInput): ChatMetrics {
  const toolMessages = messages.filter((message) => message.role === 'tool')
  const lastUsage = [...messages].reverse().find((message) => message.usage)?.usage
  const pendingApprovals = toolMessages.filter(
    (message) => message.toolNeedsApproval && message.toolStatus === 'pending',
  ).length
  const sessionTokenCount = lastUsage
    ? lastUsage.inputTokens + lastUsage.outputTokens
    : (sessionDetail?.totalTokens.input ?? 0) + (sessionDetail?.totalTokens.output ?? 0)
  const gatewayWatchActivity = createGatewayWatchActivity(gatewayWatchHealth)
  const baseActivities = activities.slice(-8).reverse()
  const recentActivities = gatewayWatchActivity
    ? [
        gatewayWatchActivity,
        ...baseActivities.filter((activity) => activity.id !== gatewayWatchActivity.id),
      ].slice(0, 8)
    : baseActivities
  const activityCount = activities.length + (gatewayWatchActivity ? 1 : 0)
  const canResumeRun = Boolean(sessionDetail?.resumableRun) && pendingApprovals === 0
  const runContract = sessionDetail?.runContract ?? null
  const contextEngine = sessionDetail?.contextEngine ?? null
  const evidenceManifest = sessionDetail?.evidenceManifest ?? null
  const evaluationGate = sessionDetail?.evaluationGate ?? null
  const traceMetrics = sessionDetail?.traceMetrics
  const providerFallbackCount = traceMetrics?.providerFallbacks ?? 0
  const providerAttemptCount = traceMetrics?.providerAttemptsStarted ?? 0
  const delegation = sessionDetail?.delegation
  const delegationMeta = delegation?.degradedSince
    ? `Since ${new Date(delegation.degradedSince).toLocaleTimeString()}`
    : delegation?.lastHeartbeatAt
      ? `Last renew ${new Date(delegation.lastHeartbeatAt).toLocaleTimeString()}`
      : null
  const nonToolMessageCount = messages.filter((message) => message.role !== 'tool').length
  const lastUsageTokenCount = lastUsage
    ? lastUsage.inputTokens + lastUsage.outputTokens
    : undefined
  const costLabel = lastUsage?.estimatedCost !== undefined
    ? `$${lastUsage.estimatedCost.toFixed(4)}`
    : `$${(sessionDetail?.totalCost ?? 0).toFixed(4)}`
  const updatedLabel = sessionDetail?.updatedAt
    ? new Date(sessionDetail.updatedAt).toLocaleTimeString()
    : 'Live session'
  const providerLabel = traceMetrics?.finalProvider || sessionDetail?.provider || selectedProvider || 'auto'
  const modelLabel = traceMetrics?.finalModel || sessionDetail?.model || selectedModel || 'default'

  return {
    toolMessages,
    lastUsage,
    pendingApprovals,
    sessionTokenCount,
    recentActivities,
    activityCount,
    canResumeRun,
    runContract,
    contextEngine,
    evidenceManifest,
    evaluationGate,
    providerFallbackCount,
    providerAttemptCount,
    delegation,
    delegationMeta,
    nonToolMessageCount,
    lastUsageTokenCount,
    costLabel,
    updatedLabel,
    providerLabel,
    modelLabel,
  }
}
