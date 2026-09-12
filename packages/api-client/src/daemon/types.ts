import type {
  ApprovalDecisionStatus,
  AgentRunContract,
  AgentContextUsage,
  AgentEvent,
  ApiError,
  ContentPart,
  DebateRoundSummary as CoreDebateRoundSummary,
  DocumentIngestInput,
  Device,
  EditCheckpointSummary,
  JobRun as CoreJobRun,
  ManualJobRunOptions as CoreManualJobRunOptions,
  ManualJobRunResult as CoreManualJobRunResult,
  MemoryEntry,
  MemoryPinnedEntry,
  MemoryContextItem,
  MemoryDocument,
  MemoryDocumentChunk,
  ModelInfo,
  PaginatedResult,
  ScheduledJob as CoreScheduledJob,
  SessionEvent,
  SessionMeta,
  SkillMetadata,
  Ticket,
  TokenUsage,
  ToolCall,
  PlannerWorkingMemory,
  ImageCanvasOperation,
  ImageCanvasOutputKind,
  ImageCanvasRecommendedModel,
  SessionEvidenceArtifactKind as CoreSessionEvidenceArtifactKind,
  RunStopReason,
} from '@sepilotd/core'

// Surface packages can't import from `@sepilotd/core` directly. Re-export
// the agent-event payload types they need for chat stream wiring here.
export type {
  AgentAcceptanceCriterion,
  AgentRunContract,
  AgentFailedAttempt,
  AgentOpenQuestion,
  AgentStateBoardPlanStep,
  AgentStateBoardSnapshot,
  SessionAttachmentRef,
  TodoItem,
  EditCheckpointFile,
  EditCheckpointSummary,
  DebateRole,
  DebateRoundEntry,
  DebateDecision,
  DebateRoundSummary,
  PlannerStepStatus,
  PlannerHierarchicalStep,
  PlannerDecision,
  PlannerRisk,
  PlannerWorkingMemory,
  CreateScheduledJobInput as DaemonScheduledTaskInput,
} from '@sepilotd/core'

/**
 * Response of `GET /api/v1/runs/:sessionId/state` (D4 canonical board endpoint).
 * `source` reports where the board came from: `live` (a run is currently
 * active), `journal` (persisted only), or `none` (no board exists yet).
 */
export interface DaemonStateBoardResponse {
  board: import('@sepilotd/core').AgentStateBoardSnapshot | null
  source: 'live' | 'journal' | 'none'
  updatedAt: number | null
}

export interface ApiEnvelope<T> {
  data: T
}

export type DaemonAgentMode = 'react' | 'enhanced' | 'auto' | (string & {})

export interface DaemonHealthComponent {
  status: string
  details?: string
  optional?: boolean
  core?: boolean
  [key: string]: unknown
}

export interface DaemonHealth {
  status: string
  version: string
  /** Version of the stable HTTP API contract, independent of product releases. */
  apiVersion?: number
  uptime?: number
  memory?: {
    rss: number
    heap: number
  }
  components?: Record<string, DaemonHealthComponent>
}

export type DaemonNotifyRelayDeliveryStatus =
  | 'accepted'
  | 'pending_review'
  | 'rejected'
  | 'unreachable'
  | 'invalid_response'

export interface DaemonNotifyRelayDeliveryReceipt {
  status: DaemonNotifyRelayDeliveryStatus
  messageId: string | null
  httpStatus: number | null
  /** Safe structured Relay rejection code; absent on older daemons. */
  errorCode?: string | null
  attemptedAt: number
  completedAt: number
}

export type DaemonNotifyRelayMessageStatus =
  | 'received'
  | 'accepted'
  | 'pending_review'
  | 'approved'
  | 'rejected'
  | 'denied'
  | 'queued'
  | 'delivering'
  | 'delivered'
  | 'delivery_failed'
  | 'dead_letter'
  | 'expired'

export interface DaemonNotifyRelayProviderDeliveryReceipt {
  lookupStatus: 'observed' | 'rejected' | 'unreachable' | 'invalid_response'
  status: DaemonNotifyRelayMessageStatus | null
  httpStatus: number | null
  /** Safe structured Relay status-lookup rejection code; absent on older daemons. */
  errorCode?: string | null
  attempt: number
  checkedAt: number
  completedAt: number | null
}

export interface DaemonLatestNotifyRelayDelivery
  extends DaemonNotifyRelayDeliveryReceipt {
  notificationId: string
  topic: string | null
  /** Final provider evidence when status reconciliation has run. */
  providerDelivery?: DaemonNotifyRelayProviderDeliveryReceipt | null
}

export interface DaemonAssistantSchedulerRunCorrelation {
  jobId: string
  jobName: string | null
  run: Pick<
    CoreJobRun,
    'id' | 'status' | 'attempt' | 'startedAt' | 'finishedAt' | 'durationMs'
  >
  notifications: Array<{
    id: string
    createdAt: number
    /** Relay API acceptance evidence, not final provider delivery. */
    relayDelivery: DaemonNotifyRelayDeliveryReceipt | null
    relayProviderDelivery?: DaemonNotifyRelayProviderDeliveryReceipt | null
  }>
  channelDeliveries: Array<{
    id: string
    status: 'pending' | 'delivering' | 'delivered' | 'failed'
    channelType: string
    attempt: number
    deliveredAt: number | null
  }>
}

export interface DaemonSchedulerDeliveryOutboxSummary {
  total: number
  pending: number
  delivering: number
  failed: number
  delivered: number
  nextAttemptAt: number | null
  nextRetryAt: number | null
}

export interface DaemonAssistantJpadPublicationCorrelation {
  sessionId: string
  sessionUpdatedAt: string
  toolCallId: string
  operation: 'create' | 'update'
  outcome: 'confirmed' | 'unconfirmed'
  pageId: string | null
  completedAt: number
}

export interface DaemonAssistantJpadPublicationEvidence {
  status: 'available' | 'partial' | 'unavailable'
  items: DaemonAssistantJpadPublicationCorrelation[]
}

export interface DaemonNotifyRelayOutboxSummary {
  total: number
  pending: number
  delivering: number
  failed: number
  /** Relay API handoffs completed; not final provider delivery. */
  delivered: number
  dead: number
  nextAttemptAt: number | null
  nextRetryAt: number | null
}

export interface DaemonNotifyRelayProviderSummary {
  total: number
  pending: number
  checking: number
  retrying: number
  delivered: number
  failed: number
  unconfirmed: number
  nextCheckAt: number | null
  lastCheckedAt: number | null
}

export interface DaemonAssistantRuntimeStatus {
  generatedAt: string
  integrations: {
    notifyRelay: {
      configured: boolean
      deliveryMode: 'durable_outbox' | 'acceptance_receipts' | 'best_effort_untracked'
      /** Absent on older daemons; null means no relay attempt has completed yet. */
      latestDelivery?: DaemonLatestNotifyRelayDelivery | null
      /** Absent on older daemons. Contains counts/timing only, never notification content. */
      outbox?: DaemonNotifyRelayOutboxSummary
      /** Absent on older daemons. Final provider tracking counts/timing only. */
      providerDelivery?: DaemonNotifyRelayProviderSummary
    }
    jpad: {
      configured: boolean
    }
    aiSearch: {
      configured: boolean
      selected: boolean
      provider: DaemonWebSearchProviderId
    }
    scheduler: {
      enabled: boolean
      cliSurfaceEnabled: boolean
    }
  }
  /** Absent on older daemons. Contains identifiers/status only, never content or destinations. */
  operations?: {
    /** Exact retained counts for the requesting surface; list responses remain bounded. */
    notifications?: {
      total: number
      unread: number
      listed: number
      truncated: boolean
    }
    /** Content-free pending interaction counts; avoids a full history-management scan. */
    interactions?: {
      pendingApprovals: {
        total: number
        orphaned: number
      }
      pendingQuestions: {
        total: number
        orphaned: number
      }
    }
    /** Non-secret configured/live channel counts. Missing requested types are not implied here. */
    channels?: Array<{
      type: string
      configured: boolean
      configuredCount: number
      enabled: boolean
      enabledCount: number
      activeCount: number
      status: 'connected' | 'disconnected' | 'connecting' | 'error' | 'not_configured'
    }>
    /** Aggregate job state only; excludes names, instructions, destinations, and errors. */
    schedulerJobs?: {
      total: number
      enabled: number
      disabled: number
      running: number
      failed: number
      enabledFailed: number
    }
    /** Counts/timing only; never scheduled output, destinations, or connector errors. */
    schedulerDeliveryOutbox?: DaemonSchedulerDeliveryOutboxSummary
    schedulerRuns: DaemonAssistantSchedulerRunCorrelation[]
    /** Built-in JPAD mutation receipts from durable session journals. */
    jpadPublications?: DaemonAssistantJpadPublicationEvidence
  }
  skills: Array<{
    id: string
    available: boolean
    enabled: boolean
  }>
  tools: Array<{
    name: string
    enabled: boolean
  }>
}

export interface DaemonScheduledTaskUpdateInput {
  when?: string
  instruction?: string
  name?: string
  timezone?: string
  maxAttempts?: number
  retryBackoffMs?: number
  enabled?: boolean
  unattended?: boolean
  channelType?: string | null
  channelTarget?: string | null
  replyToMessageId?: string | null
  parentSessionId?: string | null
  /** Stable skill ids loaded again whenever the scheduled agent executes. Empty clears them. */
  skillRefs?: Array<{ name: string }>
  metadata?: Record<string, unknown> | null
}

export interface DaemonHealthReadinessCheck extends DaemonHealthComponent {
  critical: boolean
}

export interface DaemonHealthReadinessSnapshot {
  status: string
  version: string
  uptime: number
  timestamp: string
  checks: Record<string, DaemonHealthReadinessCheck>
}

export interface DaemonHealthSnapshot extends DaemonHealth {
  uptime: number
  timestamp: string
  components: Record<string, DaemonHealthComponent>
  memory: {
    rss: number
    heap: number
  }
}

export interface DaemonHealthReportSnapshot {
  generatedAt: string
  health: DaemonHealthSnapshot
  readiness: DaemonHealthReadinessSnapshot
}

export type DaemonServiceBackend = 'process' | 'container'
export type DaemonServiceStatus =
  | 'running'
  | 'stopping'
  | 'stopped'
  | 'exited'
  | 'failed'
  | 'unknown'
export type DaemonServiceRestartMode = 'never' | 'on-failure' | 'always'
export type DaemonServiceHealthStatus = 'unknown' | 'starting' | 'healthy' | 'unhealthy'
export type DaemonNativeServiceProvider = 'systemd-user' | 'launchd-user'
export type DaemonNativeServiceRestart = 'no' | 'on-failure' | 'always'

export interface DaemonServiceRestartPolicy {
  mode: DaemonServiceRestartMode
  maxRestarts?: number
  backoffMs?: number
}

export interface DaemonServiceHealthState {
  status: DaemonServiceHealthStatus
  checkedAt?: string
  message?: string
  consecutiveFailures: number
  nextCheckAt?: string
}

export type DaemonServiceHealthCheck =
  | { type: 'process'; intervalMs?: number; timeoutMs?: number; graceMs?: number }
  | {
      type: 'http'
      url: string
      expectedStatus?: number
      intervalMs?: number
      timeoutMs?: number
      graceMs?: number
    }
  | {
      type: 'tcp'
      host: string
      port: number
      intervalMs?: number
      timeoutMs?: number
      graceMs?: number
    }

export interface DaemonServiceContainerPort {
  containerPort: number
  hostPort?: number
  protocol?: 'tcp' | 'udp'
}

export interface DaemonServiceContainerVolume {
  source: string
  target: string
  readonly?: boolean
}

export interface DaemonServiceContainerSpec {
  image: string
  runtime?: string
  name?: string
  command?: string[]
  ports?: DaemonServiceContainerPort[]
  volumes?: DaemonServiceContainerVolume[]
}

export interface DaemonServiceStartInput {
  id?: string
  name?: string
  backend?: DaemonServiceBackend
  executable?: string
  args?: string[]
  cwd?: string
  env?: Record<string, string>
  container?: DaemonServiceContainerSpec
  image?: string
  containerRuntime?: string
  containerName?: string
  command?: string[]
  ports?: DaemonServiceContainerPort[]
  volumes?: DaemonServiceContainerVolume[]
  restart?: DaemonServiceRestartPolicy
  health?: DaemonServiceHealthCheck
}

export interface DaemonServiceSnapshot {
  id: string
  name: string
  backend: DaemonServiceBackend
  executable?: string
  args: string[]
  cwd?: string
  envKeys: string[]
  container?: DaemonServiceContainerSpec
  containerId?: string
  restart: DaemonServiceRestartPolicy
  health?: DaemonServiceHealthCheck
  healthState: DaemonServiceHealthState
  pid: number | null
  status: DaemonServiceStatus
  createdAt: string
  updatedAt: string
  startedAt?: string
  stoppedAt?: string
  exitedAt?: string
  exitCode?: number | null
  signal?: string | null
  lastError?: string
  nextRestartAt?: string
  restartCount: number
  logs: {
    stdout: string
    stderr: string
    events: string
  }
}

export interface DaemonServiceLogsOptions {
  stdoutOffset?: number
  stderrOffset?: number
  limitBytes?: number
  tailBytes?: number
  followMs?: number
  pollIntervalMs?: number
}

export interface DaemonServiceLogChunk {
  service: DaemonServiceSnapshot
  stdout: string
  stderr: string
  nextStdoutOffset: number
  nextStderrOffset: number
}

export interface DaemonServiceStopInput {
  signal?: string
  timeoutMs?: number
}

export interface DaemonServiceRemoveInput {
  force?: boolean
  deleteLogs?: boolean
}

export interface DaemonServiceRemoveResult {
  removed: DaemonServiceSnapshot
}

export interface DaemonNativeServiceInstallInput {
  id: string
  name?: string
  description?: string
  provider?: DaemonNativeServiceProvider
  executable: string
  args?: string[]
  cwd?: string
  env?: Record<string, string>
  restart?: DaemonNativeServiceRestart
  enable?: boolean
  start?: boolean
  reload?: boolean
}

export interface DaemonNativeServiceSnapshot {
  id: string
  name: string
  provider: DaemonNativeServiceProvider
  unitName: string
  unitPath: string
  installed: boolean
  enabled: boolean | null
  running: boolean | null
  enableState?: string
  activeState?: string
  executable?: string
  args: string[]
  cwd?: string
  envKeys: string[]
  restart: DaemonNativeServiceRestart
  logs: {
    stdout: string
    stderr: string
  }
  createdAt?: string
  updatedAt?: string
}

export interface DaemonNativeServiceLogChunk {
  service: DaemonNativeServiceSnapshot
  stdout: string
  stderr: string
  nextStdoutOffset: number
  nextStderrOffset: number
}

export interface DaemonNativeServiceControlInput {
  start?: boolean
  stop?: boolean
  reload?: boolean
}

export interface DaemonNativeServiceUninstallResult {
  removed: DaemonNativeServiceSnapshot
}

export interface DaemonHealthWatchSnapshot {
  type: 'snapshot'
  health: DaemonHealthSnapshot
  report: DaemonHealthReportSnapshot
}

export interface DaemonHealthWatchHeartbeat {
  type: 'heartbeat'
  timestamp: string
}

export type DaemonHealthWatchPayload = DaemonHealthWatchSnapshot | DaemonHealthWatchHeartbeat

export type DaemonDoctorCheckStatus = 'PASS' | 'WARN' | 'FAIL'

export interface DaemonDoctorCheck {
  name: string
  category: string
  status: DaemonDoctorCheckStatus
  message: string
  recommendation?: string
}

export interface DaemonDoctorSummary {
  score: number
  grade: 'Excellent' | 'Good' | 'Fair' | 'Poor' | 'Critical'
  warnings: number
  errors: number
  generatedAt: string
}

export interface DaemonDoctorReport {
  data: DaemonDoctorCheck[]
  summary: DaemonDoctorSummary
}

export interface DaemonUsage extends TokenUsage {
  costUsd?: number
}

export interface DaemonChatAttachment {
  type: string
  path?: string
  url?: string
}

export interface DaemonChatResult {
  sessionId: string
  messageId: string
  content: string
  toolCalls?: ToolCall[]
  usage: DaemonUsage
  stopReason?: RunStopReason
}

export type DaemonChatBackgroundStatusValue = 'running' | 'completed' | 'failed' | 'cancelled'

export interface DaemonChatBackgroundStartResult {
  jobId: string
  sessionId: string
  status: DaemonChatBackgroundStatusValue
}

export interface DaemonChatBackgroundProgress {
  eventType: string
  label: string
  detail?: string
  action?: DaemonChatBackgroundProgressAction
  eventCount: number
  updatedAt: string
  partialContent?: string
}

export type DaemonChatBackgroundProgressAction =
  | { type: 'approval'; requestId: string; toolName: string; preview?: string }
  | { type: 'question'; questionId: string; choices?: string[] }

export interface DaemonChatBackgroundStatusResult extends DaemonChatBackgroundStartResult {
  messageId?: string
  content?: string
  progress?: DaemonChatBackgroundProgress
  error?: {
    code?: string
    message: string
  }
  createdAt: string
  updatedAt: string
}

export type DaemonChatBackgroundListItem = Omit<
  DaemonChatBackgroundStatusResult,
  'content' | 'progress'
> & {
  progress?: Omit<DaemonChatBackgroundProgress, 'partialContent'>
}

export interface DaemonChatBackgroundListResult {
  jobs: DaemonChatBackgroundListItem[]
}

export interface DaemonArtifact {
  id: string
  key?: string
  version?: number
  type: 'code' | 'html' | 'document' | 'mermaid' | 'svg' | 'image'
  title?: string
  language?: string
  content: string
}

export interface DaemonRecentArtifact {
  sessionId: string
  artifact: DaemonArtifact
  modifiedAt: number
}

export interface DaemonSessionApprovalCounters {
  approvalsRequested: number
  approvalsApproved: number
  approvalsDenied: number
  /** Tools auto-approved by a remembered session/always rule. */
  autoApprovalsApproved: number
}

export interface DaemonSessionMeta extends SessionMeta {
  primaryAgentId?: string
  /** Whether this session currently has an agent run in progress. */
  isRunning?: boolean
  /** Present only when caller passed metrics=true to /sessions. */
  approvalCounters?: DaemonSessionApprovalCounters
}

export type DaemonSessionList = PaginatedResult<DaemonSessionMeta>

export interface DaemonSessionsWatchSnapshot {
  type: 'snapshot'
  sessions: DaemonSessionList
}

export interface DaemonSessionsWatchHeartbeat {
  type: 'heartbeat'
  timestamp: string
}

export type DaemonSessionsWatchPayload = DaemonSessionsWatchSnapshot | DaemonSessionsWatchHeartbeat

export interface DaemonPendingApproval {
  requestId: string
  sessionId: string
  toolCallId: string
  tool: string
  input: Record<string, unknown>
  requestedAt: string
  expiresAt: string
  state: 'live' | 'stale'
  resumeAvailable?: boolean
}

export interface DaemonPendingQuestion {
  id: string
  sessionId: string
  prompt: string
  choices?: string[]
}

export interface DaemonResumableRun {
  stage: 'thinking' | 'acting' | 'observing'
  checkpointedAt: string
  mode: 'exact' | 'replay-safe' | 'replay-risky'
  forceRequired: boolean
  currentTool?: string
  currentToolCount?: number
  currentTools?: string[]
  journaledResultAvailable?: boolean
  recoveryProbeAvailable?: boolean
}

export interface DaemonResumableRunIssue {
  status: 'corrupt' | 'unreadable'
  message: string
}

export interface DaemonSessionDelegation {
  delegationId: string
  targetDevice: string
  claimHealth: 'healthy' | 'degraded' | 'lost'
  startedAt: string
  updatedAt: string
  degradedSince?: string
  lastHeartbeatAt?: string
  lastError?: string
  leaseLossSource?: 'gateway' | 'comments' | 'transport'
}

export interface DaemonSessionTraceMetrics {
  startedAt: string
  lastEventAt: string
  runDurationMs: number
  totalEvents: number
  userMessages: number
  assistantMessages: number
  toolCalls: number
  toolResults: number
  toolSuccesses: number
  toolFailures: number
  approvalRequests: number
  approvalApproved: number
  approvalFeedback: number
  approvalDenied: number
  /** Tools auto-approved by a remembered session/always rule. */
  autoApprovalsApproved: number
  /** Tools auto-denied by a remembered session/always rule (rare). */
  autoApprovalsDenied: number
  contextCompactions: number
  contextCompactionTokensBefore: number
  contextCompactionTokensAfter: number
  contextCompactionTokensSaved: number
  lastContextCompactedAt?: string
  assistantMessagesSinceLastContextCompaction?: number
  memoryContextEvents: number
  memoryContextItems: number
  memorySummaryEvents: number
  memorySummaryLightCaptures: number
  memorySummarySemanticExtractions: number
  memorySummaryRagPromotions: number
  lastMemorySummarizedAt?: string
  assistantMessagesSinceLastMemorySummary?: number
  coworkTasksStarted: number
  coworkTasksCompleted: number
  coworkTasksFailed: number
  coworkDiscussRequests: number
  coworkDiscussResponses: number
  todoUpdates: number
  providerAttemptEvents?: number
  providerAttemptsStarted?: number
  providerAttemptFailures?: number
  providerAttemptSuccesses?: number
  providerFallbacks?: number
  lastProviderFallbackAt?: string
  finalProvider?: string
  finalModel?: string
  timeToFirstAssistantMessageMs?: number
  timeToFirstToolCallMs?: number
  timeToFirstApprovalRequestMs?: number
  timeToFirstCoworkTaskMs?: number
}

export interface DaemonSessionContextEngine {
  schemaVersion: 1
  status: 'empty' | 'warming' | 'versioned'
  revision: string
  fingerprint: string
  eventCount: number
  lastEventId?: string
  lastEventAt: string
  sources: {
    memoryContext: {
      events: number
      items: number
      lastAt?: string
    }
    compaction: {
      events: number
      tokensSaved: number
      lastAt?: string
    }
    memorySummary: {
      events: number
      semanticExtractions: number
      ragPromotions: number
      lastAt?: string
    }
    workingMemory: {
      decisions: number
      fileChanges: number
      openQuestions: number
      lastUpdatedAt?: string
    }
    runContract: {
      present: boolean
      source?: 'planner' | 'fallback'
      acceptanceCriteria: number
    }
  }
}

export interface DaemonSessionHistoryManagementRisk {
  code:
    | 'semantic_index_unavailable'
    | 'semantic_recall_not_observed'
    | 'compaction_not_observed'
    | 'compaction_stale'
    | 'memory_summary_not_observed'
    | 'memory_summary_stale'
    | 'dreaming_provider_missing'
    | 'memory_lifecycle_attention'
  severity: 'info' | 'warning'
  message: string
}

export interface DaemonSessionHistoryManagement {
  status: 'empty' | 'warming' | 'managed' | 'attention_needed'
  compact: {
    compactions: number
    tokensBefore: number
    tokensAfter: number
    tokensSaved: number
    lastCompactedAt?: string
    assistantMessagesSinceLastCompaction: number
    lastStrategy?: 'preserve_tail' | 'summary_only'
    lastRemovedMessageCount?: number
    lastPreservedMessageCount?: number
  }
  semanticRecall: {
    contextEvents: number
    contextItems: number
    memoryItems: number
    documentItems: number
    lastContextAt?: string
    lastMemoryItemAt?: string
    lastDocumentContextAt?: string
  }
  memorySummary: {
    events: number
    lightCaptures: number
    semanticExtractions: number
    ragPromotions: number
    lastSummarizedAt?: string
    assistantMessagesSinceLastSummary: number
    sources: Record<string, number>
  }
  runtime: {
    semanticIndex?: {
      status: string
      pendingCount?: number
      failedCount?: number
      vecAvailable?: boolean
      backendAvailable?: boolean
      vectorBackend?: string
      lastError?: string
    }
    dreaming?: {
      enabled: boolean
      running: boolean
      providerConfigured: boolean
      model?: string
      fileMemoryEnabled: boolean
    }
    memoryLifecycle?: {
      totalMemories: number
      staleConversationMemories: number
      lowImportanceConversationMemories: number
      pruneCandidateMemories: number
      pendingEmbeddings: number
      failedEmbeddings: number
      lastAuditAt?: string
    }
  }
  risks: DaemonSessionHistoryManagementRisk[]
  lastUpdatedAt: string
}

export interface DaemonSessionRuntimeCleanupResult {
  pendingApprovals: number
  pendingQuestions: number
  approvalCheckpoints: number
  unavailableApprovalCheckpoints: number
  runCheckpoints: number
  toolExecutions: number
}

export interface DaemonSessionManagementSnapshot {
  totalSessions: number
  byStatus: Record<'active' | 'completed' | 'abandoned', number>
  pendingApprovals: {
    total: number
    orphaned: number
  }
  pendingQuestions: {
    total: number
    orphaned: number
  }
  runCheckpoints: {
    total: number
    unavailable: number
    locked: number
    orphaned: number
  }
  approvalCheckpoints: {
    total: number
    unavailable: number
    orphaned: number
  }
  toolExecutions: {
    total: number
    running: number
    completed: number
    unavailable: number
    orphaned: number
  }
  history: {
    totalSessions: number
    compactedSessions: number
    semanticRecallSessions: number
    documentRecallSessions: number
    memorySummarySessions: number
    attentionNeededSessions: number
    historyReadFailures: number
    totalCompactions: number
    totalTokensSaved: number
    totalSemanticContextEvents: number
    totalSemanticContextItems: number
    totalDocumentContextItems: number
    totalMemorySummaryEvents: number
    totalSemanticExtractions: number
    totalRagPromotions: number
    risksByCode: Partial<Record<DaemonSessionHistoryManagementRisk['code'], number>>
    attentionSessionIds: string[]
    failedHistorySessionIds: string[]
    semanticIndex?: DaemonSessionHistoryManagement['runtime']['semanticIndex']
    dreaming?: DaemonSessionHistoryManagement['runtime']['dreaming']
    memoryLifecycle?: DaemonSessionHistoryManagement['runtime']['memoryLifecycle']
  }
  orphaned: {
    pendingApprovalRequestIds: string[]
    pendingApprovalSessionIds: string[]
    pendingQuestionIds: string[]
    pendingQuestionSessionIds: string[]
    runCheckpointSessionIds: string[]
    approvalCheckpointRequestIds: string[]
    toolExecutionSessionIds: string[]
  }
}

export interface DaemonSessionManagementCleanupResult {
  dryRun: boolean
  candidates: DaemonSessionManagementSnapshot['orphaned']
  deleted: DaemonSessionRuntimeCleanupResult
  before: DaemonSessionManagementSnapshot
  after: DaemonSessionManagementSnapshot
}

export interface DaemonSessionChecklistItem {
  label: string
  status: 'pending' | 'in_progress' | 'completed' | 'blocked' | 'skipped'
  detail?: string
  source?: 'todo' | 'tool' | 'approval' | 'question' | 'session'
}

export interface DaemonSessionCompletionChecklist {
  status: 'not_started' | 'in_progress' | 'completed' | 'blocked'
  items: DaemonSessionChecklistItem[]
  lastUpdatedAt?: string
}

export interface DaemonSessionWorkingMemoryDecision {
  type: 'approval' | 'question' | 'delegation'
  summary: string
  timestamp: string
}

export interface DaemonSessionWorkingMemoryToolOutcome {
  tool: string
  status: 'success' | 'error'
  output: string
  timestamp: string
}

export interface DaemonSessionWorkingMemoryFileChange {
  path: string
  tool: string
  kind: 'write' | 'edit' | 'patch' | 'unknown'
  timestamp: string
}

export interface DaemonSessionWorkingMemory {
  taskSummary: string
  latestPlanStep?: string
  activeTodo?: string
  keyDecisions: DaemonSessionWorkingMemoryDecision[]
  recentToolOutcomes: DaemonSessionWorkingMemoryToolOutcome[]
  fileChanges: DaemonSessionWorkingMemoryFileChange[]
  openQuestions: string[]
  lastUpdatedAt?: string
}

export type DaemonSessionContractLedgerStatus = 'empty' | 'ready' | 'needs_attention' | 'blocked'

export type DaemonSessionContractLedgerSectionName =
  | 'goal'
  | 'scope'
  | 'acceptance_criteria'
  | 'verification_plan'
  | 'runtime_context'
  | 'blockers'

export type DaemonSessionContractLedgerEntrySource =
  | 'user_goal'
  | 'run_contract'
  | 'session_runtime'
  | 'repo_fact'
  | 'evidence'
  | 'conservative_default'
  | 'blocker'

export type DaemonSessionContractLedgerEntryStatus =
  | 'missing'
  | 'weak'
  | 'defaulted'
  | 'inferred'
  | 'confirmed'
  | 'blocked'

export type DaemonSessionContractLedgerBlockerCode =
  | 'credential_or_secret'
  | 'destructive_production_action'
  | 'external_side_effect'
  | 'billing_authority'
  | 'legal_or_medical_judgment'

export interface DaemonSessionContractLedgerEntry {
  id: string
  key: string
  value: string
  source: DaemonSessionContractLedgerEntrySource
  status: DaemonSessionContractLedgerEntryStatus
  confidence: number
  reversible: boolean
  rationale: string
  eventIds?: string[]
  evidenceIds?: string[]
}

export interface DaemonSessionContractLedgerSection {
  name: DaemonSessionContractLedgerSectionName
  status: DaemonSessionContractLedgerEntryStatus
  summary: string
  entries: DaemonSessionContractLedgerEntry[]
}

export interface DaemonSessionContractLedgerBlocker {
  code: DaemonSessionContractLedgerBlockerCode
  severity: 'warning' | 'blocker'
  message: string
  eventIds: string[]
}

export interface DaemonSessionContractLedger {
  schemaVersion: 1
  status: DaemonSessionContractLedgerStatus
  revision: string
  fingerprint: string
  eventCount: number
  lastEventId?: string
  lastEventAt: string
  summary: {
    sections: number
    confirmedSections: number
    defaultedSections: number
    inferredSections: number
    missingSections: number
    blockedSections: number
    blockers: number
    safeDefaults: number
  }
  sections: DaemonSessionContractLedgerSection[]
  blockers: DaemonSessionContractLedgerBlocker[]
  lastUpdatedAt: string
}

export type DaemonSessionEvidenceManifestStatus =
  | 'empty'
  | 'collecting'
  | 'ready'
  | 'attention_needed'

export type DaemonSessionEvidenceArtifactKind = CoreSessionEvidenceArtifactKind

export type DaemonSessionEvidenceArtifactStatus =
  | 'pending'
  | 'success'
  | 'error'
  | 'warning'
  | 'info'

export interface DaemonSessionEvidenceArtifact {
  id: string
  kind: DaemonSessionEvidenceArtifactKind
  label: string
  summary: string
  status: DaemonSessionEvidenceArtifactStatus
  timestamp: string
  sourceEventIds: string[]
  toolCallId?: string
  tool?: string
  path?: string
  hash?: string
  relatedAcceptanceCriteriaIds?: string[]
}

export interface DaemonSessionAcceptanceEvidence {
  id: string
  text: string
  status: 'supported' | 'failed' | 'blocked' | 'unverified'
  evidenceIds: string[]
  reason: string
}

export interface DaemonSessionEvidenceRisk {
  code:
    | 'no_run_contract'
    | 'acceptance_criteria_unverified'
    | 'validation_failed'
    | 'changes_without_validation'
    | 'pending_tool_results'
    | 'failed_tool_results'
    | 'post_edit_findings'
    | 'assistant_marked_unverified'
  severity: 'info' | 'warning'
  message: string
  evidenceIds?: string[]
}

export interface DaemonSessionEvidenceManifest {
  schemaVersion: 1
  status: DaemonSessionEvidenceManifestStatus
  revision: string
  fingerprint: string
  eventCount: number
  lastEventId?: string
  lastEventAt: string
  summary: {
    artifacts: number
    toolCalls: number
    toolResults: number
    validationRuns: number
    validationFailures: number
    filesChanged: number
    approvals: number
    acceptanceCriteria: number
    supportedAcceptanceCriteria: number
    failedAcceptanceCriteria: number
    blockedAcceptanceCriteria: number
    unverifiedAcceptanceCriteria: number
    risks: number
  }
  acceptanceCriteria: DaemonSessionAcceptanceEvidence[]
  artifacts: DaemonSessionEvidenceArtifact[]
  risks: DaemonSessionEvidenceRisk[]
  lastUpdatedAt: string
}

export type DaemonSessionEvaluationGateStatus =
  | 'empty'
  | 'not_started'
  | 'running'
  | 'passed'
  | 'failed'
  | 'blocked'
  | 'unverified'

export type DaemonSessionEvaluationStageStatus =
  | 'not_started'
  | 'running'
  | 'passed'
  | 'failed'
  | 'blocked'
  | 'skipped'
  | 'unverified'

export interface DaemonSessionEvaluationStage {
  status: DaemonSessionEvaluationStageStatus
  summary: string
  evidenceIds: string[]
}

export type DaemonSessionEvaluationArtifactBundleStatus =
  | 'empty'
  | 'metadata_only'
  | 'ready'
  | 'partial'

export type DaemonSessionEvaluationArtifactContentState = 'hashed' | 'metadata_only'

export type DaemonSessionEvaluationArtifactSkipReason =
  | 'missing_path'
  | 'path_traversal'
  | 'absolute_path_without_cwd'
  | 'outside_cwd'
  | 'generated_or_vendor'
  | 'duplicate_path'
  | 'max_files_exceeded'
  | 'file_missing'
  | 'not_file'
  | 'file_too_large'
  | 'total_budget_exceeded'
  | 'read_error'

export interface DaemonSessionEvaluationArtifactFile {
  id: string
  path: string
  artifactId: string
  status: DaemonSessionEvidenceArtifactStatus
  operation: string
  contentState: DaemonSessionEvaluationArtifactContentState
  sourceEventIds: string[]
  tool?: string
  sizeBytes?: number
  contentHash?: string
}

export interface DaemonSessionEvaluationArtifactSkip {
  reason: DaemonSessionEvaluationArtifactSkipReason
  message: string
  artifactId?: string
  path?: string
  sourceEventIds?: string[]
}

export interface DaemonSessionEvaluationArtifactBundle {
  schemaVersion: 1
  status: DaemonSessionEvaluationArtifactBundleStatus
  revision: string
  fingerprint: string
  eventCount: number
  cwd?: string
  limits: {
    maxFiles: number
    maxFileBytes: number
    maxTotalBytes: number
  }
  summary: {
    files: number
    hashedFiles: number
    metadataOnlyFiles: number
    skippedFiles: number
    totalBytes: number
    validationArtifacts: number
    acceptanceCriteria: number
  }
  files: DaemonSessionEvaluationArtifactFile[]
  skipped: DaemonSessionEvaluationArtifactSkip[]
  validationEvidenceIds: string[]
  acceptanceEvidenceIds: string[]
  lastUpdatedAt: string
}

export type DaemonSessionAcceptanceAssertionTier =
  | 'constant'
  | 'structural'
  | 'behavioral'
  | 'subjective'

export type DaemonSessionAcceptanceAssertionKind =
  | 'file_exists'
  | 'symbol_exists'
  | 'text_match'
  | 'validation_required'
  | 'human_review'

export type DaemonSessionAcceptanceAssertionStatus =
  | 'verified'
  | 'failed'
  | 'unverified'
  | 'skipped'

export type DaemonSessionAcceptanceVerificationStatus =
  | 'empty'
  | 'passed'
  | 'failed'
  | 'unverified'
  | 'skipped'

export interface DaemonSessionAcceptanceAssertion {
  id: string
  acceptanceCriterionId: string
  tier: DaemonSessionAcceptanceAssertionTier
  kind: DaemonSessionAcceptanceAssertionKind
  description: string
  pattern?: string
  expectedValue?: string
  fileHint?: string
  confidence: number
}

export interface DaemonSessionAcceptanceAssertionResult {
  assertion: DaemonSessionAcceptanceAssertion
  status: DaemonSessionAcceptanceAssertionStatus
  detail: string
  evidenceIds: string[]
  path?: string
  actualValue?: string
}

export interface DaemonSessionAcceptanceVerificationReport {
  acceptanceCriterionId: string
  acceptanceCriterionText: string
  status: DaemonSessionAcceptanceAssertionStatus
  results: DaemonSessionAcceptanceAssertionResult[]
  evidenceIds: string[]
  reason: string
}

export interface DaemonSessionAcceptanceVerification {
  schemaVersion: 1
  status: DaemonSessionAcceptanceVerificationStatus
  revision: string
  fingerprint: string
  eventCount: number
  summary: {
    acceptanceCriteria: number
    assertions: number
    verifiedAssertions: number
    failedAssertions: number
    unverifiedAssertions: number
    skippedAssertions: number
    constantAssertions: number
    structuralAssertions: number
    behavioralAssertions: number
    subjectiveAssertions: number
  }
  reports: DaemonSessionAcceptanceVerificationReport[]
  lastUpdatedAt: string
}

export type DaemonSessionConsensusTriggerCode =
  | 'validation_failed'
  | 'post_edit_findings'
  | 'assistant_marked_unverified'
  | 'mechanical_validation_missing'
  | 'semantic_acceptance_unverified'
  | 'artifact_bundle_partial'
  | 'large_change_set'
  | 'missing_run_contract'

export interface DaemonSessionConsensusTrigger {
  code: DaemonSessionConsensusTriggerCode
  priority: number
  fired: boolean
  severity: 'info' | 'warning'
  message: string
  evidenceIds?: string[]
}

export interface DaemonSessionConsensusTriggerMatrix {
  required: boolean
  primaryTrigger?: DaemonSessionConsensusTrigger
  triggers: DaemonSessionConsensusTrigger[]
}

export interface DaemonSessionEvaluationGateRisk {
  code:
    | 'execution_complete_without_evaluation'
    | 'mechanical_validation_missing'
    | 'mechanical_validation_failed'
    | 'semantic_acceptance_unverified'
    | 'semantic_acceptance_failed'
    | 'pending_tool_results'
    | 'consensus_required'
  severity: 'info' | 'warning'
  message: string
  evidenceIds?: string[]
}

export interface DaemonSessionEvaluationGate {
  schemaVersion: 1
  status: DaemonSessionEvaluationGateStatus
  revision: string
  fingerprint: string
  eventCount: number
  lastEventId?: string
  lastEventAt: string
  stages: {
    mechanical: DaemonSessionEvaluationStage
    semantic: DaemonSessionEvaluationStage
    consensus: DaemonSessionEvaluationStage
  }
  artifactBundle: DaemonSessionEvaluationArtifactBundle
  acceptanceVerification: DaemonSessionAcceptanceVerification
  consensusTriggers: DaemonSessionConsensusTriggerMatrix
  signals: {
    executionComplete: boolean
    runContractPresent: boolean
    fileChanges: number
    validationRuns: number
    validationFailures: number
    acceptanceCriteria: number
    supportedAcceptanceCriteria: number
    failedAcceptanceCriteria: number
    blockedAcceptanceCriteria: number
    unverifiedAcceptanceCriteria: number
    pendingToolResults: number
    assistantMarkedVerified: boolean
    assistantMarkedUnverified: boolean
    consensusRequired: boolean
  }
  verdict: {
    approved: boolean
    reason: string
  }
  risks: DaemonSessionEvaluationGateRisk[]
  lastUpdatedAt: string
}

export type DaemonSessionRunbookStatus = 'ready' | 'needs_attention' | 'blocked' | 'recoverable'

export type DaemonSessionRunbookSeverity = 'info' | 'warning' | 'critical'
export type DaemonSessionRunbookActionPriority = 'now' | 'next' | 'optional'
export type DaemonSessionRunbookSignalSource =
  | 'evidence'
  | 'runtime'
  | 'history'
  | 'checklist'
  | 'session'
export type DaemonSessionRunbookMechanicalValidationStatus =
  | 'not_needed'
  | 'already_validated'
  | 'ready'
  | 'missing_project_context'
  | 'unsupported_project'
export type DaemonSessionRunbookMechanicalValidationToolchain =
  | 'node-pnpm-turbo'
  | 'node-pnpm'
  | 'node-npm'
  | 'node-yarn'
  | 'node-bun'
  | 'python-uv'
  | 'python'
  | 'rust'
  | 'go'
  | 'zig'
export type DaemonSessionRunbookMechanicalValidationCommandKind =
  | 'lint'
  | 'typecheck'
  | 'build'
  | 'test'
  | 'coverage'
  | 'static'

export interface DaemonSessionRunbookSignal {
  id: string
  severity: DaemonSessionRunbookSeverity
  title: string
  detail: string
  source: DaemonSessionRunbookSignalSource
  evidenceIds?: string[]
}

export interface DaemonSessionRunbookAction {
  id: string
  priority: DaemonSessionRunbookActionPriority
  title: string
  command: string
  description: string
  destructive: boolean
  requiresReview: boolean
}

export interface DaemonSessionRunbookMechanicalValidationCommand {
  kind: DaemonSessionRunbookMechanicalValidationCommandKind
  label: string
  command: string
  reason: string
}

export interface DaemonSessionRunbookMechanicalValidationPlan {
  status: DaemonSessionRunbookMechanicalValidationStatus
  reason: string
  cwd?: string
  toolchain?: DaemonSessionRunbookMechanicalValidationToolchain
  commands: DaemonSessionRunbookMechanicalValidationCommand[]
}

export interface DaemonSessionRunbook {
  schemaVersion: 1
  generatedAt: string
  status: DaemonSessionRunbookStatus
  headline: string
  session: {
    id: string
    title: string
    status: DaemonSessionMeta['status']
    provider: string
    model: string
    createdAt: string
    updatedAt: string
    eventCount: number
  }
  summary: {
    signals: number
    criticalSignals: number
    warningSignals: number
    pendingApprovals: number
    pendingQuestions: number
    evidenceRisks: number
    validationRuns: number
    validationFailures: number
    filesChanged: number
    acceptanceCriteria: number
    supportedAcceptanceCriteria: number
    failedAcceptanceCriteria: number
    blockedAcceptanceCriteria: number
    unverifiedAcceptanceCriteria: number
    validationSuggestions: number
  }
  signals: DaemonSessionRunbookSignal[]
  actions: DaemonSessionRunbookAction[]
  mechanicalValidation: DaemonSessionRunbookMechanicalValidationPlan
  evidence: {
    manifestRevision?: string
    manifestStatus?: DaemonSessionEvidenceManifestStatus
    acceptanceCriteria: DaemonSessionAcceptanceEvidence[]
    risks: DaemonSessionEvidenceRisk[]
    artifacts: DaemonSessionEvidenceArtifact[]
  }
  supportBundle: {
    recommended: boolean
    command: string
    reason: string
  }
}

export interface DaemonSessionDetail extends DaemonSessionMeta {
  events: SessionEvent[]
  pendingQuestions?: DaemonPendingQuestion[]
  pendingApprovals?: DaemonPendingApproval[]
  traceMetrics?: DaemonSessionTraceMetrics
  contextEngine?: DaemonSessionContextEngine
  historyManagement?: DaemonSessionHistoryManagement
  completionChecklist?: DaemonSessionCompletionChecklist
  workingMemory?: DaemonSessionWorkingMemory
  runContract?: AgentRunContract | null
  contractLedger?: DaemonSessionContractLedger
  evidenceManifest?: DaemonSessionEvidenceManifest
  evaluationGate?: DaemonSessionEvaluationGate
  resumableRun?: DaemonResumableRun
  resumableRunIssue?: DaemonResumableRunIssue
  delegation?: DaemonSessionDelegation
  /** Raw checkpoints/rounds/working-memory extracted from persisted session
   * events — same shapes the live SSE onEditCheckpointResolved /
   * onDebateRound / onPlannerWorkingMemoryUpdated callbacks send per-turn,
   * so history replay can feed the same client-side store actions. */
  editRollbacks?: EditCheckpointSummary[]
  debateRounds?: CoreDebateRoundSummary[]
  plannerWorkingMemory?: PlannerWorkingMemory | null
}

export interface DaemonSessionExportSnapshot {
  session: DaemonSessionDetail
  events: SessionEvent[]
  health?: DaemonHealthReportSnapshot
}

export type DaemonSessionShareMode = 'knowledge' | 'public'

export interface DaemonSessionShareResult {
  shared: boolean
  mode?: DaemonSessionShareMode
  path?: string
  shareUrl?: string
  expiresAt?: string
  sessionId?: string
  reason?: string
}

export interface DaemonSessionImportResult {
  sessionId: string
  importedEvents: number
  title: string
  sourceTitle: string
}

export interface DaemonSessionBranchResult {
  branchId: string
  sourceId: string
  copiedEvents: number
}

export interface DaemonSessionCompactResult {
  sessionId: string
  originalTokens: number
  compactedTokens: number
  savedTokens: number
  summary: string
  strategy?: 'preserve_tail' | 'summary_only'
  removedMessageCount?: number
  preservedMessageCount?: number
}

export interface DaemonApprovalResponseResult {
  requestId: string
  decision: ApprovalDecisionStatus
  approved: boolean
  note?: string
  resolved: boolean
  state: 'live' | 'stale'
  /**
   * Set when the respond persisted or enabled a scoped auto decision —
   * carries the derived rule (tool + pattern, e.g. "ls /tmp/foo *")
   * the daemon will match future invocations against. Surfaces in
   * the cli approve success line so operators can predict
   * short-circuit behaviour without grepping `decisions list`.
   */
  rule?: { tool: string; pattern: string }
}

export interface DaemonRememberedApproval {
  tool: string
  pattern: string
  scope: 'session' | 'always'
  approved: boolean
  sessionId?: string
  createdAt: string
  /** Times this rule short-circuited a tool prompt. Surfaces in
   * `decisions list` so operators can tell active rules from stale
   * ones at a glance. */
  hitCount?: number
  /** ISO timestamp of the most recent match (undefined = never used). */
  lastHitAt?: string
}

export interface DaemonRememberedApprovalInput {
  tool: string
  pattern: string
  scope: 'session' | 'always'
  approved: boolean
  sessionId?: string
}

export interface DaemonRememberedApprovalMatch {
  tool: string
  pattern: string
  scope: 'session' | 'always'
  sessionId?: string
}

export interface DaemonRememberedApprovalRule {
  tool: string
  pattern: string
}

export interface DaemonRememberedApprovalDescribeInput {
  tool: string
  input: Record<string, unknown>
}

export interface DaemonConfigProvider {
  id: string
  /** Built-in or plugin-contributed provider type. */
  type: string
  apiKey?: string
  baseUrl?: string
  headers?: Record<string, string>
  models: string[]
  default?: boolean
  defaultContextWindow?: number
  defaultMaxOutputTokens?: number
  capabilities?: Partial<ModelInfo['capabilities']>
  modelOverrides?: Array<{
    id: string
    contextWindow?: number
    maxOutputTokens?: number
    capabilities?: Partial<ModelInfo['capabilities']>
    compatibility?: ModelInfo['compatibility']
  }>
  [key: string]: unknown
}

export interface DaemonChannelConfig {
  type:
    | 'github-issue'
    | 'telegram'
    | 'slack'
    | 'discord'
    | 'mattermost'
    | 'webhook'
    | 'webchat'
    | 'whatsapp'
    | 'teams'
    | 'line'
  enabled?: boolean
  config?: Record<string, unknown>
}

export interface DaemonWebhookEndpointConfig {
  enabled?: boolean
  path: string
  secretHeader: string
  secretValue: string
  allowedIps?: string[]
  allowedEvents?: string[]
}

export interface DaemonWebhookEndpointSummary {
  id: string
  enabled: boolean
  path: string
  publicRoute: string
  secretHeader: string
  hasSecretValue: boolean
  allowedIps: string[]
  allowedEvents: string[]
}

export interface DaemonTelegramChannelConfig {
  enabled?: boolean
  botToken: string
  allowedUsers?: string[]
  pairingRequired?: boolean
  pairingCodeTtl?: number
  rateLimitPerMinute?: number
}

export interface DaemonTelegramChannelSummary {
  enabled: boolean
  status: 'connected' | 'disconnected' | 'connecting' | 'error'
  hasBotToken: boolean
  allowedUsers: string[]
  pairingRequired: boolean
  pairingCodeTtl: number
  rateLimitPerMinute: number
}

export interface DaemonDiscordChannelConfig {
  enabled?: boolean
  botToken: string
  applicationId: string
  publicKey: string
  allowedGuilds?: string[]
  allowedChannels?: string[]
  allowedUsers?: string[]
}

export interface DaemonDiscordChannelSummary {
  enabled: boolean
  status: 'connected' | 'disconnected' | 'connecting' | 'error'
  hasBotToken: boolean
  hasApplicationId: boolean
  hasPublicKey: boolean
  allowedGuilds: string[]
  allowedChannels: string[]
  allowedUsers: string[]
}

export interface DaemonMattermostChannelConfig {
  enabled?: boolean
  serverUrl: string
  botToken: string
  webhookToken: string
  allowedTeams?: string[]
  allowedChannels?: string[]
  allowedUsers?: string[]
}

export interface DaemonMattermostChannelSummary {
  enabled: boolean
  status: 'connected' | 'disconnected' | 'connecting' | 'error'
  hasServerUrl: boolean
  hasBotToken: boolean
  hasWebhookToken: boolean
  serverUrl: string
  allowedTeams: string[]
  allowedChannels: string[]
  allowedUsers: string[]
}

export interface DaemonSlackChannelConfig {
  enabled?: boolean
  botToken: string
  signingSecret: string
  allowedChannels?: string[]
  allowedUsers?: string[]
}

export interface DaemonSlackChannelSummary {
  enabled: boolean
  status: 'connected' | 'disconnected' | 'connecting' | 'error'
  hasBotToken: boolean
  hasSigningSecret: boolean
  allowedChannels: string[]
  allowedUsers: string[]
}

export interface DaemonTelegramPairingCode {
  code: string
  expiresAt: string
}

export interface DaemonTeamDocsConfigInput {
  id?: string
  name: string
  description?: string
  serverType?: 'github.com' | 'ghes'
  ghesUrl?: string
  token: string
  owner: string
  repo: string
  branch?: string
  docsPath?: string
  enabled?: boolean
  autoSync?: boolean
  syncInterval?: number
}

export interface DaemonTeamDocsConfig {
  id: string
  name: string
  description: string
  serverType: 'github.com' | 'ghes'
  ghesUrl: string
  token: string
  owner: string
  repo: string
  branch: string
  docsPath: string
  enabled: boolean
  autoSync: boolean
  syncInterval: number
  lastTestedAt: number | null
  lastSyncAt: number | null
  lastSyncStatus: 'success' | 'error' | null
  lastSyncError: string | null
  syncedDocuments: number
  updatedAt: number
}

export interface DaemonTeamDocsDocument {
  path: string
  sha: string | null
  size: number
  syncedAt: number
}

export interface DaemonTeamDocsDocumentContent extends DaemonTeamDocsDocument {
  content: string
}

export interface DaemonTeamDocsActionResult {
  success: boolean
  message: string
  config: DaemonTeamDocsConfig | null
}

export interface DaemonTeamDocsSyncAllResult {
  success: boolean
  total: number
  succeeded: number
  failed: number
  items: DaemonTeamDocsConfig[]
}

export interface DaemonTeamDocsWatchSnapshot {
  type: 'snapshot'
  items: DaemonTeamDocsConfig[]
  documentsByConfigId: Record<string, DaemonTeamDocsDocument[]>
}

export interface DaemonTeamDocsWatchHeartbeat {
  type: 'heartbeat'
  timestamp: string
}

export type DaemonTeamDocsWatchPayload = DaemonTeamDocsWatchSnapshot | DaemonTeamDocsWatchHeartbeat

export interface DaemonPersonalDoc {
  id: string
  path: string
  updatedAt: number
}

export interface DaemonPersonalDocContent extends DaemonPersonalDoc {
  content: string
}

export interface DaemonPersonalDocInput {
  id?: string
  path: string
  content: string
}

export interface DaemonPromptTemplate {
  id: string
  title: string
  body: string
}

export interface DaemonPromptTemplateInput {
  id?: string
  title: string
  body: string
}

export interface DaemonBackupItem {
  id: string
  createdAt: number
  sizeBytes: number
  path: string
  sha256: string
}

export interface DaemonQuickQuestion {
  id: string
  name: string
  prompt: string
  shortcut: string
  enabled: boolean
}

export interface DaemonQuickInputSettings {
  hotkey: string
  prefix: string
  /** User-defined hotkeys that combine prompt + clipboard into a new chat. */
  quickQuestions?: DaemonQuickQuestion[]
}

export interface DaemonQuickInputWatchSnapshot {
  type: 'snapshot'
  settings: DaemonQuickInputSettings
}

export interface DaemonQuickInputWatchHeartbeat {
  type: 'heartbeat'
  timestamp: string
}

export type DaemonQuickInputWatchPayload =
  | DaemonQuickInputWatchSnapshot
  | DaemonQuickInputWatchHeartbeat

export interface DaemonQuickInputPublishResult {
  ok: boolean
}

export interface DaemonSettingsJsonDocument {
  [key: string]: unknown
}

export interface DaemonSettingsJsonWatchSnapshot {
  type: 'snapshot'
  document: DaemonSettingsJsonDocument
}

export interface DaemonSettingsJsonWatchHeartbeat {
  type: 'heartbeat'
  timestamp: string
}

export type DaemonSettingsJsonWatchPayload =
  | DaemonSettingsJsonWatchSnapshot
  | DaemonSettingsJsonWatchHeartbeat

export interface DaemonRagFolder {
  id: string
  name: string
  documents: number
  sourceType: 'manual' | 'git' | 'web'
  tlsVerify?: boolean
  caCert?: string
  path?: string
  include?: string[]
  exclude?: string[]
  lastSyncedAt: number | null
  lastSyncStatus: 'success' | 'error' | null
  lastSyncError: string | null
}

export interface DaemonRagDocument {
  id: string
  folderId: string
  title: string
  path?: string
  sourceFileId?: string
  size?: number
  updatedAt: number
}

export interface DaemonRagDocumentContent extends DaemonRagDocument {
  body: string
}

export interface DaemonRagFolderInput {
  id?: string
  name: string
  sourceType?: 'manual' | 'git' | 'web'
  tlsVerify?: boolean
  caCert?: string
  path?: string
  include?: string[]
  exclude?: string[]
}

export interface DaemonRagDocumentInput {
  id?: string
  folderId: string
  title: string
  body: string
  path?: string
  sourceFileId?: string
  size?: number
}

export interface DaemonRagSearchHit {
  documentId: string
  folderId: string
  title: string
  score: number
  snippet: string
  path?: string
}

export interface DaemonRagVectorDbInfo {
  engine:
    | 'memory'
    | 'sqlite-vec'
    | 'sqlite-scan'
    | 'qdrant'
    | 'opensearch'
    | 'elasticsearch'
    | 'meilisearch'
    | 'custom-api'
  vectorBackend?:
    | 'sqlite-vec'
    | 'sqlite-scan'
    | 'qdrant'
    | 'opensearch'
    | 'elasticsearch'
    | 'meilisearch'
    | 'custom-api'
  backendAvailable?: boolean
  vecAvailable?: boolean
  status?: 'disabled' | 'ready' | 'backfilling' | 'degraded' | 'reindex_required'
  dimension: number
  documents: number
  lastError?: string
}

export interface DaemonRagSyncResult {
  ok: boolean
  folders: number
  indexed: number
  deleted: number
  skipped: number
  errors: Array<{ folderId: string; path?: string; error: string }>
}

export interface DaemonRagConnectionTestResult {
  ok: boolean
  target: 'vector-backend' | 'rerank'
  backend?: string
  url?: string
  status?: number
  durationMs: number
  message: string
}

export interface DaemonMemoryHttpAuth {
  type?: 'none' | 'bearer' | 'api-key' | 'basic'
  headerName?: string
  username?: string
  password?: string
}

export interface DaemonNetworkConfig {
  proxyMode: 'environment' | 'direct' | 'manual'
  proxyUrl: string | null
  noProxy: string | null
  customCaPath: string | null
  tlsRejectUnauthorized: boolean
  timeoutMs: number
  maxConcurrency: number
}

export interface DaemonNetworkStatus {
  active: boolean
  effective: {
    proxyMode: DaemonNetworkConfig['proxyMode']
    useProxy: boolean
    timeoutMs: number
    customCaPath: string | null
    tlsRejectUnauthorized: boolean
    degradedReason: string | null
  }
  overrides: {
    tlsRejectUnauthorized:
      | 'SEPILOTD_TLS_REJECT_UNAUTHORIZED'
      | 'NODE_TLS_REJECT_UNAUTHORIZED'
      | null
    timeoutMs: 'SEPILOTD_PROVIDER_HTTP_TIMEOUT_MS' | null
    customCaPath: 'SEPILOTD_EXTRA_CA_CERTS' | 'NODE_EXTRA_CA_CERTS' | null
  }
  /** Present on daemons that distinguish declared environment values which
   * were rejected and therefore did not override the saved policy. */
  ignoredOverrides?: {
    tlsRejectUnauthorized:
      | 'SEPILOTD_TLS_REJECT_UNAUTHORIZED'
      | 'NODE_TLS_REJECT_UNAUTHORIZED'
      | null
    timeoutMs: 'SEPILOTD_PROVIDER_HTTP_TIMEOUT_MS' | null
    customCaPath: 'SEPILOTD_EXTRA_CA_CERTS' | 'NODE_EXTRA_CA_CERTS' | null
  }
}

export interface DaemonNetworkWatchSnapshot {
  type: 'snapshot'
  config: DaemonNetworkConfig
}

export interface DaemonNetworkWatchHeartbeat {
  type: 'heartbeat'
  timestamp: string
}

export type DaemonNetworkWatchPayload = DaemonNetworkWatchSnapshot | DaemonNetworkWatchHeartbeat

export interface DaemonNetworkProbeResult {
  ok: boolean
  reachable?: boolean
  latencyMs?: number
  status?: number
  reason?: string
}

export interface DaemonGitHubOAuthStatus {
  connected: boolean
  login: string | null
}

export interface DaemonGitHubOAuthStartResult {
  url: string
  warning?: string
}

export interface DaemonGitHubSyncRepo {
  fullName: string
  enabled: boolean
  lastSyncedAt: number | null
}

export interface DaemonGitHubSyncPolicy {
  intervalMin: number
  pullRequests: boolean
  issues: boolean
  releases: boolean
}

export interface DaemonGitHubChatSyncConfig {
  enabled: boolean
  repoFullName: string | null
  branch: string
  lastSyncedAt: number | null
  lastSyncStatus: 'success' | 'error' | null
  lastSyncError: string | null
  lastCommit: string | null
  syncedSessions: number
  updatedAt: number | null
}

export interface DaemonGitHubChatSyncConfigUpdate {
  enabled?: boolean
  repoFullName?: string | null
  branch?: string
}

export interface DaemonGitHubChatSyncResult {
  startedAt: number
  finishedAt: number
  repoFullName: string
  branch: string
  worktree: string
  syncedSessions: number
  changedFiles: number
  pushed: boolean
  commitSha: string | null
}

export interface DaemonMessageSubscriptionConfig {
  enabled: boolean
  connectionType: 'polling' | 'websocket' | 'nats'
  pollingUrl: string
  websocketUrl: string
  pollingInterval: number
  authToken: string
  customHeaders: Record<string, string>
  natsUrl: string
  natsConsumerId: string
  natsConsumerSecret: string
  natsStreamName: string
  natsSubject: string
  natsBatchSize: number
  natsFetchTimeout: number
  maxQueueSize: number
  retentionDays: number
  autoProcess: boolean
  retryAttempts: number
  retryDelay: number
  useAIProcessing: boolean
  aiPromptTemplate: string
  thinkingMode: 'instant' | 'sequential'
  showNotification: boolean
}

export interface DaemonMessageQueueStatus {
  pending: number
  processing: number
  completed: number
  failed: number
  totalProcessed: number
  lastPolled: number | null
  lastProcessed: number | null
}

export interface DaemonMessageSubscriptionStatus {
  isConnected: boolean
  lastPolled: number | null
  lastError: string | null
}

export interface DaemonMessageSubscriptionItem {
  hash: string
  id: string | null
  type: 'github_webhook' | 'community_post' | 'custom'
  source: string
  title: string
  body: string
  content: string
  metadata: Record<string, unknown>
  timestamp: number
  queuedAt: number
  status: 'pending' | 'processing' | 'completed' | 'failed'
  processedAt: number | null
  error: string | null
  retryCount: number
  conversationId: string | null
}

export interface DaemonMessageSubscriptionOverview {
  config: DaemonMessageSubscriptionConfig
  queueStatus: DaemonMessageQueueStatus
  status: DaemonMessageSubscriptionStatus
  recentMessages: DaemonMessageSubscriptionItem[]
}

export interface DaemonMessageSubscriptionRefreshResult {
  success: boolean
  count: number
  processed?: number
  error?: string
  overview: DaemonMessageSubscriptionOverview
}

export interface DaemonMessageSubscriptionProcessResult {
  success: boolean
  processed: number
  overview: DaemonMessageSubscriptionOverview
}

export interface DaemonMessageSubscriptionWatchSnapshot {
  type: 'snapshot'
  overview: DaemonMessageSubscriptionOverview
  messages: DaemonMessageSubscriptionItem[]
  groups: {
    pending: DaemonMessageSubscriptionItem[]
    processing: DaemonMessageSubscriptionItem[]
    completed: DaemonMessageSubscriptionItem[]
    failed: DaemonMessageSubscriptionItem[]
  }
}

export interface DaemonMessageSubscriptionWatchHeartbeat {
  type: 'heartbeat'
  timestamp: string
}

export type DaemonMessageSubscriptionWatchPayload =
  | DaemonMessageSubscriptionWatchSnapshot
  | DaemonMessageSubscriptionWatchHeartbeat

export interface DaemonNotificationItem {
  id: string
  title: string
  body: string
  url: string | null
  topic?: string | null
  audience?: string[] | null
  createdAt: number
  readAt: number | null
  correlation?: {
    kind: 'scheduler'
    jobId: string
    runId: string | null
  } | null
  /** Relay API acceptance evidence, not proof of final provider delivery. */
  relayDelivery?: DaemonNotifyRelayDeliveryReceipt | null
  /** Relay status evidence; only status=delivered proves provider delivery. */
  relayProviderDelivery?: DaemonNotifyRelayProviderDeliveryReceipt | null
}

export interface DaemonNotificationDraft {
  id?: string
  title: string
  body?: string
  url?: string | null
  topic?: string | null
  audience?: string[] | null
}

export interface DaemonNotificationChannelSetting {
  id: string
  enabled: boolean
}

export interface DaemonNotificationSettings {
  channels: DaemonNotificationChannelSetting[]
}

export interface DaemonNotificationsWatchSnapshot {
  type: 'snapshot'
  items: DaemonNotificationItem[]
}

export interface DaemonNotificationsWatchItem {
  type: 'item'
  item: DaemonNotificationItem
}

export interface DaemonNotificationsWatchHeartbeat {
  type: 'heartbeat'
  timestamp: string
}

export type DaemonNotificationsWatchPayload =
  | DaemonNotificationsWatchSnapshot
  | DaemonNotificationsWatchItem
  | DaemonNotificationsWatchHeartbeat

/** Full scheduled-job record as exposed by the daemon scheduler capability. */
export type DaemonSchedulerJob = CoreScheduledJob

/** A single execution record of a scheduled job. */
export type DaemonSchedulerJobRun = CoreJobRun

/** Terminal evidence returned after an immediate scheduler run request. */
export type DaemonSchedulerManualRunResult = CoreManualJobRunResult

/** Options for a real manual scheduler execution. */
export type DaemonSchedulerManualRunOptions = CoreManualJobRunOptions

export interface DaemonSchedulerWatchSnapshot {
  type: 'snapshot'
  jobs: DaemonSchedulerJob[]
}

export interface DaemonSchedulerWatchHeartbeat {
  type: 'heartbeat'
  timestamp: string
}

export type DaemonSchedulerWatchPayload =
  | DaemonSchedulerWatchSnapshot
  | DaemonSchedulerWatchHeartbeat

export interface DaemonSchedulerNotificationSubscriptions {
  jobId: string
  subscribers: string[]
}

/**
 * Search backend the daemon queries for `web.search`. `auto` uses a keyed
 * provider when its credential is present in the daemon environment and
 * otherwise falls back to keyless DuckDuckGo.
 */
export type DaemonWebSearchProviderId =
  | 'auto'
  | 'duckduckgo'
  | 'brave'
  | 'tavily'
  | 'searxng'
  | 'ai-search'

export interface DaemonWebSearchProviderUpdate {
  provider: DaemonWebSearchProviderId
  /** Send `***redacted***` to keep the stored key. */
  apiKey?: string
  /** SearXNG or compatible AI Search instance URL; empty string clears it. */
  endpoint?: string
}

export interface DaemonSchedulerJobInput {
  id?: string
  name: string
  /** 5-field cron, cron nickname (@daily …) or interval form (@every 30s). */
  cron?: string
  /** Natural-language one-shot time such as "in 2 hours" or "tomorrow at 9am". */
  when?: string
  /** Absolute one-shot fire time as epoch milliseconds. */
  runAt?: number
  instruction?: string
  timezone?: string
  nextRunAt?: number
  enabled?: boolean
  maxAttempts?: number
  retryBackoffMs?: number
  parentSessionId?: string | null
  /** Stable skill ids loaded again whenever the scheduled agent executes. Empty clears them. */
  skillRefs?: Array<{ name: string }>
  metadata?: Record<string, unknown> | null
}

export interface DaemonWebhookSecurityPolicyThreshold {
  signatureMaxSkewSeconds: number
  verificationUnavailableStatus: 'service_unavailable' | 'not_found'
}

export interface DaemonWebhookSecurityPolicyConfig extends DaemonWebhookSecurityPolicyThreshold {
  byChannelType: Record<string, Partial<DaemonWebhookSecurityPolicyThreshold>>
}

export interface DaemonWebhookSecurityHealthConfig {
  degradeWhenUnreadyEndpointsAtLeast: number
  detailTopMissingRequirements: number
  detailTopUnreadyRoutes: number
}

export interface DaemonChannelPipelineHealthThreshold {
  minRecentEvents: number
  degradeFailureRate: number
  minAgentSamples: number
  degradeAgentAvgLatencyMs: number
}

export interface DaemonChannelPipelineHealthConfig extends DaemonChannelPipelineHealthThreshold {
  hotChannelTopN: number
  byChannelType: Record<string, Partial<DaemonChannelPipelineHealthThreshold>>
}

export interface DaemonSkillSourceSecurityConfig {
  enforceUrlAllowlist: boolean
  allowedHosts: string[]
  allowedUrlPrefixes: string[]
  allowLocalPaths: boolean
}

export type DaemonExtensionTokenScope =
  | 'all'
  | 'inspect'
  | 'chat'
  | 'ws'
  | 'sessions'
  | 'memory'
  | 'files'
  | 'skills'
  | 'projects'
  | 'approvals'
  | 'extensions'
  | 'personas'
  | 'artifacts'

export interface DaemonExtensionTokenSummary {
  id: string
  label: string
  scopes: DaemonExtensionTokenScope[]
  createdAt: string
  expiresAt?: string
  revokedAt?: string
  active: boolean
}

export interface DaemonIssueExtensionTokenInput {
  label: string
  scopes: DaemonExtensionTokenScope[]
  expiresAt?: string
}

export interface DaemonIssuedExtensionToken extends DaemonExtensionTokenSummary {
  token: string
}

export interface DaemonImageGenProvider {
  id: string
  label: string
  enabled: boolean
  operations?: ImageCanvasOperation[]
  recommendedModels?: ImageCanvasRecommendedModel[]
  hardware?: {
    supportsDeviceSelection?: boolean
    devicesEndpoint?: string
  }
}

export type DaemonImageGenJobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'

export interface DaemonImageGenJobOutput {
  id: string
  mime: string
  path?: string
  kind?: ImageCanvasOutputKind
}

export interface DaemonImageGenJob {
  id: string
  providerId: string
  prompt: string
  status: DaemonImageGenJobStatus
  progress: number
  outputs: DaemonImageGenJobOutput[]
  error: string | null
  createdAt: number
}

export interface DaemonImageGenJobInput {
  providerId: string
  prompt: string
  params?: Record<string, unknown>
}

export interface DaemonSnippetGistLink {
  id: string
  file: string
  htmlUrl: string | null
  updatedAt: string | null
  syncedAt: number | null
}

export interface DaemonSnippet {
  id: string
  title: string
  language: string
  body: string
  tags: string[]
  gist: DaemonSnippetGistLink | null
}

export interface DaemonSnippetInput {
  id?: string
  title: string
  language: string
  body: string
  tags?: string[]
}

export interface DaemonWikiNode {
  id: string
  parentId: string | null
  title: string
  icon: string | null
  group: string | null
  order: number
  body: string
  updatedAt: number
}

export interface DaemonWikiSearchHit extends DaemonWikiNode {
  snippet?: string
}

export interface DaemonWikiNodeInput {
  id?: string
  parentId?: string | null
  title: string
  icon?: string | null
  group?: string | null
  body?: string
}

export interface DaemonWikiMoveInput {
  parentId: string | null
  order: number
}

export type DaemonOutboundWebhookEvent =
  | 'post:agent:run'
  | 'post:tool:execute'
  | 'post:process:start'
  | 'post:process:exit'
  | 'post:llm:call'
  | 'post:channel:msg'

export interface DaemonOutboundWebhookRetryPolicy {
  maxAttempts?: number
  backoffMs?: number
}

export interface DaemonOutboundWebhookConfig {
  enabled?: boolean
  url: string
  events: DaemonOutboundWebhookEvent[]
  secret?: string
  headers?: Record<string, string>
  retry?: DaemonOutboundWebhookRetryPolicy
}

export interface DaemonOutboundWebhookSummary {
  id: string
  enabled: boolean
  url: string
  events: DaemonOutboundWebhookEvent[]
  hasSecret: boolean
  headerKeys: string[]
  retry: {
    maxAttempts: number
    backoffMs: number
  }
}

export interface DaemonOutboundWebhookDelivery {
  timestamp: string
  deliveryId: string
  webhookId: string
  url: string
  hookEvent: DaemonOutboundWebhookEvent
  deliveryStatus: 'success' | 'error'
  attemptCount: number
  statusCode?: number
  durationMs: number
  error?: string
  sessionId?: string
  replayedFromDeliveryId?: string
}

export interface DaemonListPageMeta {
  limit: number
  returned: number
  nextCursor: string | null
}

export interface DaemonListPage<T> {
  data: T[]
  meta: DaemonListPageMeta
}

export interface DaemonOutboundWebhookDeadLetter {
  rootDeliveryId: string
  latestDeliveryId: string
  webhookId: string
  url: string
  hookEvent: DaemonOutboundWebhookEvent
  firstFailedAt: string
  lastAttemptAt: string
  replayCount: number
  attemptCount: number
  statusCode?: number
  durationMs: number
  error?: string
  sessionId?: string
  state: 'open' | 'acknowledged'
  acknowledgedAt?: string
  acknowledgedByDevice?: string
  acknowledgmentNote?: string
}

export type DaemonMcpServerConfig =
  | {
      name: string
      enabled?: boolean
      transport?: 'stdio'
      command: string
      args?: string[]
      env?: Record<string, string>
      disabledTools?: string[]
    }
  | {
      name: string
      enabled?: boolean
      transport: 'sse' | 'http'
      url: string
      headers?: Record<string, string>
      disabledTools?: string[]
    }

export interface DaemonMcpClientRootConfig {
  uri: string
  name?: string
  _meta?: Record<string, unknown>
}

export interface DaemonMcpClientConfig {
  roots?: {
    enabled?: boolean
    listChanged?: boolean
    entries?: DaemonMcpClientRootConfig[]
  }
  sampling?: {
    enabled?: boolean
    provider?: string
    model?: string
    maxTokens?: number
    temperature?: number
  }
  elicitation?: {
    enabled?: boolean
    mode?: 'decline' | 'accept-defaults'
    applyDefaults?: boolean
  }
}

export interface DaemonMcpServerStatus {
  name: string
  enabled: boolean
  status: 'disabled' | 'connected' | 'error'
  connected: boolean
  transport: string
  command?: string
  args?: string[]
  url?: string
  toolCount: number
  tools: string[]
  allTools?: string[]
  disabledTools?: string[]
  toolManifest?: {
    version: 1
    generatedAt: string
    digest: string
    tools: Array<{ name: string; digest: string }>
  }
  toolManifestStatus?: 'trusted' | 'changed' | 'untrusted-baseline'
  securityAlerts?: Array<{
    code: string
    severity: 'warning' | 'critical'
    message: string
    detectedAt: string
    details?: Record<string, unknown>
  }>
  error?: string
}

export const DAEMON_CONFIG_UPDATE_KEYS = [
  'agent.mode',
  'agent.defaultProvider',
  'agent.defaultModel',
  'agent.auxModel',
  'agent.autonomy',
  'agent.thinkingLevel',
  'agent.disabledTools',
  'agent.graphNodeModelOverrides',
  'daemon.resumeArtifactRetentionDays',
  'device.name',
  'network',
  'webSearch.trustedDomains',
  'webSearch.provider',
  'channels',
  'channels.sharedGroupContext',
  'channels.maxGlobalRuns',
  'channels.maxGlobalQueuedRuns',
  'channelPipeline.sharedGroupContext',
  'channelPipeline.maxGlobalRuns',
  'channelPipeline.maxGlobalQueuedRuns',
  'channelPipeline.defaultWorkspaceRoot',
  'providers',
  'mcp.servers',
  'mcp.client',
  'hooks.outboundWebhooks',
  'hooks.commandHooks',
  'security.skillSources',
  'security.webhooks',
  'observability.channelPipelineHealth',
  'observability.webhookSecurityHealth',
  // Vector DB / embedding settings — surfaced in the desktop's
  // VectorDBSettings tab; daemon stores them under config.memory.*.
  'memory.vectorBackend',
  'memory.embeddingProvider',
  'memory.embeddingModel',
  'memory.qdrant',
  'memory.opensearch',
  'memory.elasticsearch',
  'memory.meilisearch',
  'memory.customApi',
  'memory.rag',
  'memory.maintenance',
  'scheduler',
  'scheduler.enabled',
  'scheduler.timezone',
  'scheduler.dailyTokenBudget',
  'scheduler.maxConsecutiveFailures',
  'scheduler.surfaces',
  'scheduler.surfaces.cli',
  'scheduler.surfaces.desktop',
  'scheduler.surfaces.mobile',
] as const

export type DaemonConfigUpdateKey = (typeof DAEMON_CONFIG_UPDATE_KEYS)[number]

export function isDaemonConfigUpdateKey(value: string): value is DaemonConfigUpdateKey {
  return (DAEMON_CONFIG_UPDATE_KEYS as readonly string[]).includes(value)
}

export interface DaemonConfigUpdateInput {
  'agent.mode'?: DaemonAgentMode
  'agent.defaultProvider'?: string
  'agent.defaultModel'?: string
  'agent.autonomy'?: 'readonly' | 'accept-edits' | 'workspace-write' | 'supervised' | 'autonomous'
  'agent.thinkingLevel'?: 'off' | 'low' | 'medium' | 'high' | 'max'
  'agent.disabledTools'?: string[]
  'agent.graphNodeModelOverrides'?: DaemonGraphNodeModelOverrides
  'daemon.resumeArtifactRetentionDays'?: number
  'device.name'?: string
  network?: DaemonNetworkConfig
  'webSearch.trustedDomains'?: string[]
  'webSearch.provider'?: DaemonWebSearchProviderUpdate
  'channels.sharedGroupContext'?: boolean
  'channels.maxGlobalRuns'?: number
  'channels.maxGlobalQueuedRuns'?: number
  'channelPipeline.sharedGroupContext'?: boolean
  'channelPipeline.maxGlobalRuns'?: number
  'channelPipeline.maxGlobalQueuedRuns'?: number
  'channelPipeline.defaultWorkspaceRoot'?: string | null
  providers?: DaemonConfigProvider[]
  'mcp.servers'?: DaemonMcpServerConfig[]
  'mcp.client'?: DaemonMcpClientConfig
  'hooks.outboundWebhooks'?: DaemonOutboundWebhookConfig[]
  'security.skillSources'?: Partial<DaemonSkillSourceSecurityConfig>
  'security.webhooks'?: DaemonWebhookSecurityPolicyConfig
  'observability.channelPipelineHealth'?: Partial<DaemonChannelPipelineHealthConfig>
  'observability.webhookSecurityHealth'?: DaemonWebhookSecurityHealthConfig
  'memory.maintenance'?: DaemonConfig['memory']['maintenance']
  scheduler?: DaemonConfig['scheduler']
  'scheduler.enabled'?: boolean
  'scheduler.timezone'?: string | null
  'scheduler.dailyTokenBudget'?: number | null
  'scheduler.maxConsecutiveFailures'?: number
  'scheduler.surfaces'?: NonNullable<DaemonConfig['scheduler']>['surfaces']
  'scheduler.surfaces.cli'?: boolean
  'scheduler.surfaces.desktop'?: boolean
  'scheduler.surfaces.mobile'?: boolean
  [key: string]: unknown
}

export type DaemonToolPolicyMode = 'autonomous' | 'supervised' | 'ask' | 'blocked'

export interface DaemonPolicyRule {
  mode: DaemonToolPolicyMode
  deny_patterns?: string[]
  deny_paths?: string[]
  deny_urls?: string[]
  deny_executables?: string[]
  allow_patterns?: string[]
  allow_paths?: string[]
  max_timeout_ms?: number
  max_output_bytes?: number
}

export interface DaemonPolicySnapshot {
  version: number
  defaults: {
    mode: DaemonToolPolicyMode
    unmatched_policy: 'allow' | 'deny'
    max_timeout_ms: number
    max_output_bytes: number
  }
  tools: Record<string, DaemonPolicyRule>
  elevated?: Record<string, Partial<DaemonPolicyRule>>
}

export interface DaemonConfig {
  version: 1
  device: {
    id: string
    name: string
    role: 'desktop' | 'server' | 'edge'
  }
  daemon: {
    port: number
    host: string
    resumeArtifactRetentionDays: number
  }
  gateway: {
    url: string
  }
  network: DaemonNetworkConfig
  webSearch?: {
    trustedDomains: string[]
    provider: DaemonWebSearchProviderId
    /** Redacted to `***redacted***` in GET /config responses. */
    apiKey?: string
    endpoint?: string
  }
  providers: DaemonConfigProvider[]
  agent: {
    autonomy: 'readonly' | 'accept-edits' | 'workspace-write' | 'supervised' | 'autonomous'
    thinkingLevel: 'off' | 'low' | 'medium' | 'high' | 'max'
    defaultProvider?: string
    defaultModel?: string
    mode: DaemonAgentMode
    intentRouter?: {
      enabled: boolean
      provider?: string
      model?: string
      timeoutMs: number
      maxPreviousMessages: number
      perMessageCharLimit: number
      reasonMaxChars: number
    }
    capabilities: {
      hostSystemInfo: boolean
    }
    disabledTools?: string[]
    graphNodeModelOverrides?: DaemonGraphNodeModelOverrides
  }
  channels: DaemonChannelConfig[]
  channelPipeline?: {
    sharedGroupContext: boolean
    maxGlobalRuns: number
    maxGlobalQueuedRuns: number
    defaultWorkspaceRoot?: string
  }
  hooks: {
    outboundWebhooks: DaemonOutboundWebhookConfig[]
  }
  mcp: {
    servers: DaemonMcpServerConfig[]
    client: DaemonMcpClientConfig
  }
  memory: {
    encryption: boolean
    directApiToolset?: 'lean' | 'full' | 'none'
    vectorBackend:
      | 'auto'
      | 'sqlite-vec'
      | 'sqlite-scan'
      | 'qdrant'
      | 'opensearch'
      | 'elasticsearch'
      | 'meilisearch'
      | 'custom-api'
    qdrant?: {
      url: string
      apiKey?: string
      collection?: string
    }
    opensearch?: {
      url: string
      index?: string
      apiKey?: string
      username?: string
      password?: string
    }
    elasticsearch?: {
      url: string
      index?: string
      apiKey?: string
      username?: string
      password?: string
    }
    meilisearch?: {
      url: string
      index?: string
      apiKey?: string
      embedder?: string
    }
    customApi?: {
      url: string
      apiKey?: string
      auth?: DaemonMemoryHttpAuth
      headers?: Record<string, string>
      healthPath?: string
      configurePath?: string
      upsertPath?: string
      deletePath?: string
      searchPath?: string
      clearPath?: string
      timeoutMs?: number
    }
    rag?: {
      defaultLimit?: number
      candidateMultiplier?: number
      scoreThreshold?: number
      rerank?: {
        enabled?: boolean
        provider?: 'local' | 'custom-api'
        model?: string
        endpoint?: string
        apiKey?: string
        auth?: DaemonMemoryHttpAuth
        headers?: Record<string, string>
        healthPath?: string
        timeoutMs?: number
        weight?: number
      }
    }
    maintenance?: {
      enabled: boolean
      schedule: string
      maxAgeDays: number
      maxImportance: number
      dryRun: boolean
    }
    userProfile?: {
      enabled: boolean
      schedule: string
      section: string
    }
    embeddingProvider?: string
    embeddingModel?: string
  }
  proactivity?: {
    digest?: {
      enabled: boolean
      schedule: string
      channelType?: string
      channelTarget?: string
    }
  }
  scheduler?: {
    enabled: boolean
    timezone?: string
    dailyTokenBudget?: number
    maxConsecutiveFailures: number
    surfaces: {
      cli: boolean
      desktop: boolean
      mobile: boolean
    }
  }
  notifications?: {
    chatCompletion?: {
      enabled: boolean
      includeFailures: boolean
    }
  }
  share?: {
    public: {
      enabled: boolean
      secret: string
      ttlSeconds: number
      baseUrl?: string
    }
  }
  security: {
    toolPolicy: string
    auditLog: boolean
    sandbox: 'local' | 'docker'
    sandboxDocker?: {
      image: string
      networkMode: 'none' | 'bridge'
      mountMode: 'rw' | 'ro'
      cpuLimit: string
      memoryLimit: string
      pidsLimit: number
      readOnlyRootfs: boolean
      noNewPrivileges: boolean
      capDrop: string[]
    }
    skillSources?: DaemonSkillSourceSecurityConfig
    webhooks?: DaemonWebhookSecurityPolicyConfig
  }
  observability: {
    telemetry: boolean
    otlpEndpoint: string
    channelPipelineHealth?: DaemonChannelPipelineHealthConfig
    webhookSecurityHealth?: DaemonWebhookSecurityHealthConfig
  }
}

export interface DaemonConfigUpdateResult {
  updated: string[]
}

export interface DaemonConfigEnvUpdateInput {
  updates: Record<string, string | null>
}

export interface DaemonConfigEnvUpdateResult {
  updated: string[]
  removed: string[]
  path: string
}

export interface DaemonProviderValidationInput {
  provider: DaemonConfigProvider
  model?: string
  timeoutMs?: number
  env?: Record<string, string>
}

export interface DaemonProviderValidationResult {
  ok: true
  providerId: string
  providerType: string
  model: string
  latencyMs: number
  message: string
}

export interface DaemonProviderDiscoverModelsInput {
  type: string
  baseUrl?: string
  apiKey?: string
  headers?: Record<string, string>
  timeoutMs?: number
}

export interface DaemonProviderDiscoverModelsResult {
  models: string[]
  source: 'remote' | 'static'
  endpoint?: string
  compatibility?: DaemonProviderCompatibilityProfile
}

export interface DaemonProviderCompatibilityProfile {
  profile: 'ollama-openai'
  capabilities: {
    thinkingControl: 'reasoning-effort'
  }
}

export interface DaemonProviderRefreshModelsResult {
  providerId: string
  models: string[]
  source: 'remote' | 'static'
  endpoint?: string
  compatibility?: DaemonProviderCompatibilityProfile
  updated: boolean
}

export interface DaemonProviderMutationResult {
  providerId: string
  action: 'created' | 'updated'
}

export interface DaemonModelTarget {
  providerId: string
  modelId: string
}

export interface DaemonAvailableProviderModels {
  providerId: string
  providerName: string
  providerType?: string
  models: ModelInfo[]
  configuredModelIds: string[]
  defaultModelId?: string
  ready: boolean
}

export interface DaemonModelSnapshot {
  current: DaemonModelTarget | null
  providers: DaemonAvailableProviderModels[]
  lines: string[]
}

export interface DaemonModelSwitchInput {
  target: string
  timeoutMs?: number
}

export type DaemonModelSwitchResult =
  | {
      ok: true
      status: 'switched' | 'no_change'
      providerId: string
      model: string
      previousProviderId?: string
      previousModel?: string
      latencyMs?: number
      message: string
    }
  | {
      ok: false
      status: 'self_test_failed_rolled_back' | 'self_test_failed_no_previous' | 'rollback_failed'
      providerId: string
      model: string
      previousProviderId?: string
      previousModel?: string
      latencyMs?: number
      reason: string
      rollbackError?: string
      message: string
    }

export interface DaemonModelPullInput {
  model: string
  providerId?: string
  timeoutMs?: number
  switch?: boolean
}

export interface DaemonModelPullResult {
  ok: true
  providerId: string
  model: string
  alreadyAvailable: boolean
  alreadyConfigured: boolean
  latencyMs: number
  message: string
}

export interface DaemonModelPullResponse {
  pull: DaemonModelPullResult
  switch?: DaemonModelSwitchResult
}

export interface DaemonProviderHealth {
  status: 'ready' | 'env_missing' | 'unavailable'
  message?: string
  missingEnvVars: string[]
}

export interface DaemonProviderInfo {
  id: string
  name: string
  models: ModelInfo[]
  supportsEmbedding: boolean
  embeddingModelIds: string[]
  configuredModelIds: string[]
  modelCatalogAuthority?: 'configured' | 'endpoint'
  unavailableConfiguredModelIds?: string[]
  health: DaemonProviderHealth
}

export interface DaemonPersona {
  id: string
  name: string
  description: string
  systemPromptAddition: string
}

export type DaemonSkill = SkillMetadata

export interface DaemonSkillDetail {
  metadata: SkillMetadata
  content: string
}

export interface DaemonSkillCreateInput {
  metadata: SkillMetadata
  content: string
  force?: boolean
}

export interface DaemonSkillCreateResult {
  id: string
  version: string
}

export interface InstallSkillRequest {
  source: string
  force?: boolean
  expectedDigest?: string
}
export interface InstallSkillPreviewRequest {
  source: string
}
export interface InstallSkillPreviewResponse {
  digest: string
  candidates: Array<{
    metadata: SkillMetadata
    source: SkillMetadata['source']
  }>
}
export interface InstallSkillResponse {
  installed: SkillMetadata[]
}
export interface MarketplaceSkillSearchResult {
  marketplace: string
  source: string
  metadata: SkillMetadata
  installed: boolean
}
export interface UpdateSkillResponse {
  changed: boolean
  from?: string
  to?: string
  reason?: string
}
export interface Marketplace {
  name: string
  url: string
  addedAt: string
  lastSync: string | null
}
export interface AddMarketplaceRequest {
  name: string
  url: string
}

export type DaemonMemoryEntry = MemoryEntry
export type DaemonMemoryPinnedEntry = MemoryPinnedEntry
export interface DaemonMemoryRecentEntry extends Omit<MemoryEntry, 'score'> {
  createdAt?: string
  updatedAt?: string
}
export type DaemonMemoryDocument = MemoryDocument
export type DaemonMemoryDocumentChunk = MemoryDocumentChunk
export type DaemonDevice = Device

export type DaemonMemoryGraphNodeKind =
  | 'person'
  | 'project'
  | 'preference'
  | 'tool'
  | 'topic'
  | 'decision'
  | 'constraint'
  | 'fact'
  | 'place'
  | 'organization'
  | 'other'

export interface DaemonMemoryGraphNode {
  id: string
  label: string
  kind: DaemonMemoryGraphNodeKind
  aliases: string[]
  tags: string[]
  evidenceMemoryIds: string[]
  confidence: number
  createdAt: string
  updatedAt: string
  lastSeenAt: string
}

export interface DaemonMemoryGraphEdge {
  id: string
  fromNodeId: string
  toNodeId: string
  relation: string
  tags: string[]
  evidenceMemoryIds: string[]
  confidence: number
  createdAt: string
  updatedAt: string
  lastSeenAt: string
}

export interface DaemonMemoryGraphEvidenceEntry {
  id: string
  content: string
  source: MemoryEntry['source']
  tags: string[]
}

export type DaemonMemoryGraphQualitySeverity = 'info' | 'warning' | 'critical'

export type DaemonMemoryGraphQualitySignalCode =
  | 'low_confidence'
  | 'thin_evidence'
  | 'missing_evidence'
  | 'inactive_evidence'
  | 'contradiction'
  | 'stale'
  | 'orphan'

export interface DaemonMemoryGraphQualitySignal {
  code: DaemonMemoryGraphQualitySignalCode
  severity: DaemonMemoryGraphQualitySeverity
  subjectType: 'page' | 'node' | 'edge'
  subjectId?: string
  message: string
  evidenceMemoryIds?: string[]
}

export interface DaemonMemoryGraphWikiPageRelationship {
  direction: 'in' | 'out'
  edge: DaemonMemoryGraphEdge
  node: DaemonMemoryGraphNode
}

export interface DaemonMemoryGraphWikiPageQuality {
  score: number
  confidence: number
  evidenceCount: number
  activeEvidenceCount: number
  inactiveEvidenceCount: number
  missingEvidenceCount: number
  relationshipCount: number
  contradictionCount: number
  staleRelationshipCount: number
  signals: DaemonMemoryGraphQualitySignal[]
}

export interface DaemonMemoryGraphWikiPage {
  query?: string
  node: DaemonMemoryGraphNode | null
  relationships: DaemonMemoryGraphWikiPageRelationship[]
  evidence: DaemonMemoryGraphEvidenceEntry[]
  quality?: DaemonMemoryGraphWikiPageQuality
}

export interface DaemonMemoryGraphQualityReport {
  generatedAt: string
  stats: {
    nodes: number
    edges: number
    lastUpdatedAt?: string
  }
  scannedNodes: number
  scannedEdges: number
  lowConfidenceNodes: number
  lowConfidenceEdges: number
  thinEvidenceNodes: number
  thinEvidenceEdges: number
  missingEvidenceNodes: number
  missingEvidenceEdges: number
  inactiveEvidenceNodes: number
  inactiveEvidenceEdges: number
  staleNodes: number
  staleEdges: number
  orphanNodes: number
  contradictions: number
  signals: DaemonMemoryGraphQualitySignal[]
}

export type DaemonMemoryGraphRepairAction =
  | 'prune_unbacked_graph_entry'
  | 'relink_or_add_evidence'
  | 'refresh_graph_entry'
  | 'review_contradiction'

export interface DaemonMemoryGraphRepairProposal {
  id: string
  action: DaemonMemoryGraphRepairAction
  subjectType: 'node' | 'edge' | 'page'
  subjectId?: string
  severity: DaemonMemoryGraphQualitySeverity
  safeToApply: boolean
  reason: string
  signalCodes: DaemonMemoryGraphQualitySignalCode[]
  evidenceMemoryIds: string[]
}

export interface DaemonMemoryGraphRepairResult {
  dryRun: boolean
  generatedAt: string
  report: DaemonMemoryGraphQualityReport
  proposals: DaemonMemoryGraphRepairProposal[]
  applied: {
    checkedNodes: number
    checkedEdges: number
    prunedNodes: number
    prunedEdges: number
    skippedUnsafe: number
  }
}

export type DaemonMemoryGraphRepairDecisionAction = 'supersede_memories'

export interface DaemonMemoryGraphRepairDecisionInput {
  action?: DaemonMemoryGraphRepairDecisionAction
  winnerMemoryId: string
  supersededMemoryIds: string[]
  proposalId?: string
  dryRun?: boolean
  includeAllScopes?: boolean
  reason?: string
}

export interface DaemonMemoryGraphRepairDecisionResult {
  dryRun: boolean
  generatedAt: string
  action: DaemonMemoryGraphRepairDecisionAction
  proposalId?: string
  winnerMemoryId: string
  updated: Array<{
    id: string
    beforeTags: string[]
    afterTags: string[]
  }>
  skipped: Array<{
    id: string
    reason: string
    message: string
  }>
  maintenance: {
    checkedNodes: number
    checkedEdges: number
    prunedNodes: number
    prunedEdges: number
  }
}

export type DaemonMemorySearchType = 'semantic' | 'keyword' | 'hybrid'

export interface DaemonMemorySearchOptions {
  asOf?: string
  includeInactive?: boolean
  type?: DaemonMemorySearchType
  limit?: number
  includeAllScopes?: boolean
  sources?: MemoryEntry['source'][]
  tags?: string[]
  tagsLogic?: 'and' | 'or'
  excludeTags?: string[]
  createdAfter?: string
  createdBefore?: string
  includeSuperseded?: boolean
  includeArchived?: boolean
  sortBy?: 'score' | 'createdAt' | 'updatedAt'
}

export interface DaemonMemoryRecentOptions {
  limit?: number
  includeAllScopes?: boolean
  sources?: MemoryEntry['source'][]
  tags?: string[]
  tagsLogic?: 'and' | 'or'
  excludeTags?: string[]
  createdAfter?: string
  createdBefore?: string
  includeSuperseded?: boolean
  includeArchived?: boolean
  sortBy?: 'createdAt' | 'updatedAt'
}

export interface DaemonMemoryGraphPageOptions {
  query?: string
  id?: string
  limit?: number
  evidenceLimit?: number
  includeAllScopes?: boolean
}

export interface DaemonMemoryGraphAuditOptions {
  limit?: number
  signalLimit?: number
  lowConfidenceThreshold?: number
  staleAfterDays?: number
  includeAllScopes?: boolean
}

export interface DaemonMemoryGraphRepairOptions extends DaemonMemoryGraphAuditOptions {
  dryRun?: boolean
  reason?: string
}

export interface DaemonMemoryDocumentSearchOptions extends DaemonMemorySearchOptions {
  documentId?: string
}

export interface DaemonMemoryDocumentListOptions {
  query?: string
  limit?: number
  includeAllScopes?: boolean
}

export interface DaemonMemoryAddResult {
  id: string
}

export interface DaemonMemoryAddOptions {
  evidence?: MemoryEntry['evidence']
  source?: MemoryEntry['source']
  reason?: string
}

export interface DaemonMemoryUpdateInput {
  evidence?: MemoryEntry['evidence']
  content: string
  source?: MemoryEntry['source']
  tags?: string[]
  reason?: string
}

export interface DaemonMemoryDeleteOptions {
  reason?: string
}

export type DaemonMemoryAuditAction = 'created' | 'updated' | 'deleted' | 'pruned' | 'maintenance'

export type DaemonMemoryAuditSnapshot = Omit<MemoryEntry, 'score'>

export interface DaemonMemoryAuditEntry {
  id: string
  memoryId: string
  action: DaemonMemoryAuditAction
  actor: string
  reason?: string
  before?: DaemonMemoryAuditSnapshot
  after?: DaemonMemoryAuditSnapshot
  createdAt: string
}

export interface DaemonMemoryAuditOptions {
  memoryId?: string
  limit?: number
}

export interface DaemonMemorySecurityAuditOptions {
  limit?: number
  since?: string
  actor?: string
  authKind?: string
  route?: string
}

export interface DaemonMemorySecurityAuditEvent {
  timestamp: string
  event: string
  device?: string
  route?: string
  method?: string
  actor?: string
  authKind?: string
  scopeTags?: string[]
  requested?: string
  reason?: string
  tokenId?: string
  label?: string
  tokenScopes?: string[]
  [key: string]: unknown
}

export interface DaemonMemorySecurityAuditResult {
  data: DaemonMemorySecurityAuditEvent[]
  meta: {
    limit: number
    returned: number
  }
}

export interface DaemonMemoryScopeCount {
  scope: string
  count: number
}

export interface DaemonMemoryScopes {
  fileScopes: string[]
  semanticScopes: DaemonMemoryScopeCount[]
  untaggedSemanticEntries: number
  pendingReminders: DaemonMemoryScopeCount[]
}

export interface DaemonMemoryScopeTransferInput {
  target: string
  dryRun?: boolean
  includeFile?: boolean
  includeReminders?: boolean
  reason?: string
  confirmGlobal?: boolean
  ids?: string[]
  limit?: number
}

export interface DaemonMemoryScopeTransferResult {
  dryRun: boolean
  globalSource?: boolean
  fromScope: string
  toScope: string
  matchedSemanticMemories?: number
  wouldRetagSemanticMemories?: number
  wouldUpdateReminders?: number
  wouldMoveFileBucket?: string | null
  limited?: boolean
  requiresConfirmGlobal?: boolean
  sampleSemanticIds?: string[]
  sampleReminderIds?: string[]
  retaggedSemanticMemories?: number
  retaggedReminders?: number
  fileBucketMoved?: boolean
}

export interface DaemonMemoryLifecycleOptions {
  staleAfterDays?: number
  lowImportance?: number
}

export interface DaemonMemoryLifecycleStatus {
  totalMemories: number
  conversationMemories: number
  documentMemories: number
  skillMemories: number
  userMemories: number
  staleConversationMemories: number
  lowImportanceConversationMemories: number
  pruneCandidateMemories: number
  pendingEmbeddings: number
  failedEmbeddings: number
  lastAuditAt?: string
}

export interface DaemonMemoryMaintenanceInput {
  maxAgeDays?: number
  maxImportance?: number
  dryRun?: boolean
  reason?: string
}

export interface DaemonMemoryMaintenanceResult {
  dryRun: boolean
  importanceUpdated: number
  pruned: number
  wouldPrune: number
  status: DaemonMemoryLifecycleStatus
}

export interface DaemonFileMemorySection {
  title: string
  content: string
}

export interface DaemonFileMemorySnapshot {
  memoryPath: string
  todayNotePath: string
  yesterdayNotePath: string
  longTermMemory: string
  todayNote: string
  yesterdayNote: string
  sections: DaemonFileMemorySection[]
}

export interface DaemonFileMemorySectionUpdateResult extends DaemonFileMemorySection {
  deleted: boolean
}

export type DaemonMemoryDocumentIngestInput = DocumentIngestInput

export interface DaemonUploadedFileIndexInput extends Pick<
  DocumentIngestInput,
  'title' | 'path' | 'tags'
> {
  ocr?: boolean
  ocrLanguages?: string[]
  ocrMaxPages?: number
}

export interface DaemonMemorySemanticStatus {
  status: 'disabled' | 'ready' | 'backfilling' | 'degraded' | 'reindex_required'
  configuredProviderId?: string
  configuredModel?: string
  indexedProviderId?: string
  indexedModel?: string
  dimensions?: number
  pendingCount: number
  failedCount: number
  vecAvailable: boolean
  vectorBackend:
    | 'sqlite-vec'
    | 'sqlite-scan'
    | 'qdrant'
    | 'opensearch'
    | 'elasticsearch'
    | 'meilisearch'
    | 'custom-api'
  backendAvailable: boolean
  lastError?: string
}

export interface DaemonMemoryReindexResult {
  started: boolean
  status: DaemonMemorySemanticStatus
}

export interface DaemonUploadedFile {
  id: string
  filename: string
  mimeType: string
  size: number
  uploadedAt: string
}

export interface DaemonProject {
  id: string
  name: string
  description: string
  instructions: string
  workingDirectory?: string
  sessionIds: string[]
  fileIds: string[]
  createdAt: string
  updatedAt: string
}

export interface DaemonFileUploadResult {
  files: DaemonUploadedFile[]
}

export type DaemonFileContentPart = ContentPart

export interface DaemonUsageSummary {
  inputTokens: number
  outputTokens: number
  costUsd: number
  requestCount: number
}

export interface DaemonDailyUsageSummary {
  date: string
  provider: string
  model: string
  totalInputTokens: number
  totalOutputTokens: number
  totalCostUsd: number
  requestCount: number
}

export interface DaemonUsageSnapshot {
  totalSessions: number
  totalMessages: number
  totalToolCalls: number
}

export type DaemonObservabilityRange = '24h' | '7d' | '30d'

export type DaemonObservabilitySource =
  | 'daemon'
  | 'cli'
  | 'tui'
  | 'desktop-main'
  | 'desktop-renderer'
  | 'web'
  | 'channel'

export type DaemonObservabilitySeverity = 'debug' | 'info' | 'warning' | 'error' | 'fatal'

export type DaemonObservabilityPrivacy = 'operational' | 'diagnostic' | 'sensitive'

export interface DaemonObservabilityEventInput {
  id?: string
  schemaVersion?: number
  timestamp?: string
  source: DaemonObservabilitySource
  surface?: string
  eventType: string
  severity?: DaemonObservabilitySeverity
  privacy?: DaemonObservabilityPrivacy
  sessionId?: string
  runId?: string
  messageId?: string
  taskId?: string
  channelIdHash?: string
  userIdHash?: string
  provider?: string
  model?: string
  attributes?: Record<string, unknown>
}

export interface DaemonObservabilityEvent extends Required<
  Pick<DaemonObservabilityEventInput, 'id' | 'source' | 'eventType' | 'severity' | 'privacy'>
> {
  schemaVersion: number
  timestamp: string
  surface?: string
  sessionId?: string
  runId?: string
  messageId?: string
  taskId?: string
  channelIdHash?: string
  userIdHash?: string
  provider?: string
  model?: string
  attributes: Record<string, unknown>
}

export interface DaemonFeedbackInput {
  id?: string
  timestamp?: string
  sessionId?: string
  messageId?: string
  runId?: string
  rating: 'positive' | 'neutral' | 'negative'
  reason?: string
  note?: string
  source?: DaemonObservabilitySource
  surface?: string
}

export interface DaemonFeedbackRecord {
  id: string
  timestamp: string
  sessionId?: string
  messageId?: string
  runId?: string
  rating: 'positive' | 'neutral' | 'negative'
  reason?: string
  noteRedacted?: string
  source: DaemonObservabilitySource
  surface?: string
}

export interface DaemonObservabilityPrivacySettings {
  localCollectionEnabled: boolean
  diagnosticCollectionEnabled: boolean
  sensitiveCollectionEnabled: boolean
  feedbackCollectionEnabled: boolean
  retentionDays: number
  feedbackPromptCooldownHours: number
  updatedAt: string
}

export interface DaemonObservabilityPrivacySettingsInput {
  localCollectionEnabled?: boolean
  diagnosticCollectionEnabled?: boolean
  sensitiveCollectionEnabled?: boolean
  feedbackCollectionEnabled?: boolean
  retentionDays?: number
  feedbackPromptCooldownHours?: number
}

export interface DaemonFeedbackPromptStateInput {
  surface?: string
  sessionId?: string
  messageId?: string
  userIdHash?: string
}

export interface DaemonFeedbackPromptState {
  shouldPrompt: boolean
  reason: 'eligible' | 'disabled' | 'cooldown' | 'recent-feedback'
  cooldownHours: number
  promptCount24h: number
  feedbackCount24h: number
  lastPromptAt?: string
  nextPromptAfter?: string
}

export interface DaemonObservabilitySnapshot {
  generatedAt: string
  range: {
    key: DaemonObservabilityRange
    from: string
    to: string
  }
  reliability: {
    totalEvents: number
    errorEvents: number
    fatalEvents: number
    crashReports: number
    crashFreeSessions: number | null
    providerFailureRate: number | null
    channelDeliverySuccessRate: number | null
    routeErrorRate: number | null
  }
  quality: {
    feedbackPositive: number
    feedbackNeutral: number
    feedbackNegative: number
    explicitSatisfaction: number | null
    promptResponseRate: number | null
    implicitAcceptance: number | null
    regenerationRate: number | null
    stopRate: number | null
  }
  productivity: {
    tasksStarted: number
    tasksCompleted: number
    completionRate: number | null
    assistedTaskThroughputPerDay: number
    autonomousCompletionRate: number | null
    approvalFrictionMs: number | null
    toolSuccessRate: number | null
    costPerResolvedTask: number | null
    tokensPerResolvedTask: number | null
    recoverySuccessRate: number | null
    channelResolutionRate: number | null
  }
  comparison?: {
    previousFrom: string
    previousTo: string
    errorEventsDelta: number
    crashReportsDelta: number
    satisfactionDelta: number | null
    completionRateDelta: number | null
    throughputPerDayDelta: number
    channelResolutionRateDelta: number | null
    toolSuccessRateDelta: number | null
  }
  trend?: Array<{
    label: string
    from: string
    to: string
    totalEvents: number
    errorEvents: number
    crashReports: number
    feedbackPositive: number
    feedbackNeutral: number
    feedbackNegative: number
    explicitSatisfaction: number | null
    tasksStarted: number
    tasksCompleted: number
    completionRate: number | null
    assistedTaskThroughputPerDay: number
    channelTasksStarted: number
    channelTasksCompleted: number
    channelResolutionRate: number | null
    toolSuccessRate: number | null
  }>
  segments?: Array<{
    key: string
    label: string
    source: DaemonObservabilitySource
    surface?: string
    totalEvents: number
    errorEvents: number
    crashReports: number
    feedbackPositive: number
    feedbackNeutral: number
    feedbackNegative: number
    explicitSatisfaction: number | null
    tasksStarted: number
    tasksCompleted: number
    completionRate: number | null
    channelTasksStarted: number
    channelTasksCompleted: number
    channelResolutionRate: number | null
    toolSuccessRate: number | null
  }>
  hotspots?: {
    errorEvents: Array<{
      key: string
      label: string
      count: number
      source?: DaemonObservabilitySource
      surface?: string
      eventType?: string
      severity?: DaemonObservabilitySeverity
    }>
    feedbackReasons: Array<{
      key: string
      label: string
      count: number
      source?: DaemonObservabilitySource
      surface?: string
    }>
  }
  alerts?: Array<{
    id: string
    level: 'info' | 'warning' | 'critical'
    title: string
    detail: string
    metric: string
    value: number | null
    threshold?: number
  }>
  recent: DaemonObservabilityEvent[]
}

export interface DaemonObservabilityExportInput {
  range?: DaemonObservabilityRange
  limit?: number
  includeEvents?: boolean
  includeCrashes?: boolean
  includeFeedback?: boolean
  persist?: boolean
}

export interface DaemonObservabilitySupportBundleInput extends DaemonObservabilityExportInput {
  includeHealth?: boolean
}

export interface DaemonObservabilityExportBundle {
  generatedAt: string
  privacy: DaemonObservabilityPrivacySettings
  snapshot: DaemonObservabilitySnapshot
  events: DaemonObservabilityEvent[]
  crashes: DaemonObservabilityEvent[]
  feedback: DaemonFeedbackRecord[]
  redaction: {
    attributes: 'sanitized-at-ingest'
    notes: 'redacted-or-truncated'
  }
}

export interface DaemonObservabilityExportResult {
  path?: string
  eventCount: number
  crashCount: number
  feedbackCount: number
  bundle: DaemonObservabilityExportBundle
}

export interface DaemonObservabilitySupportBundleFile {
  path: string
  mediaType: string
  bytes: number
}

export interface DaemonObservabilitySupportBundleRedaction {
  level: 'support'
  appliedAt: string
  rules: string[]
  pathAliases: Record<string, string>
}

export interface DaemonObservabilitySupportBundleManifest {
  schemaVersion: 1
  kind: 'sepilotd-observability-support-bundle'
  generatedAt: string
  range: DaemonObservabilityRange
  counts: {
    events: number
    crashes: number
    feedback: number
  }
  healthIncluded: boolean
  redaction: DaemonObservabilitySupportBundleRedaction
  contents: DaemonObservabilitySupportBundleFile[]
}

export interface DaemonObservabilitySupportBundleResult {
  path?: string
  fileCount: number
  totalBytes: number
  eventCount: number
  crashCount: number
  feedbackCount: number
  files: DaemonObservabilitySupportBundleFile[]
  manifest: DaemonObservabilitySupportBundleManifest
  redaction: DaemonObservabilitySupportBundleRedaction
}

export interface DaemonObservabilityPruneResult {
  cutoff: string
  deletedEvents: number
  deletedFeedback: number
}

export interface DaemonAgentDescriptor {
  id: string
  name: string
  description: string
  source?: 'builtin' | 'yaml' | 'plugin' | 'capability' | 'user'
}

export type DaemonGraphNodePromptKind =
  | 'system'
  | 'routing'
  | 'agent'
  | 'tool'
  | 'guard'
  | 'subgraph'
  | 'report'
  | 'internal'

export type DaemonGraphNodeRecommendedModel = 'default' | 'fast' | 'strong' | 'none'

export interface DaemonGraphNodeModelOverride {
  model?: string
}

export type DaemonGraphNodeModelOverrides = Record<
  string,
  Record<string, DaemonGraphNodeModelOverride>
>

export interface DaemonAgentGraphNode {
  id: string
  label: string
  summary: string
  lifecycleState: string
  resumeStage: string
  promptKind: DaemonGraphNodePromptKind
  prompt: string
  modelConfigurable: boolean
  recommendedModel: DaemonGraphNodeRecommendedModel
  activeModel?: string
  notes: string[]
}

export interface DaemonAgentGraphEdge {
  from: string
  to: string
  type: 'direct' | 'conditional'
  label?: string
}

export interface DaemonAgentGraphSnapshot {
  id: string
  name: string
  description: string
  source?: 'builtin' | 'yaml' | 'plugin' | 'capability' | 'user'
  startNode: string
  nodeCount: number
  edgeCount: number
  nodes: DaemonAgentGraphNode[]
  edges: DaemonAgentGraphEdge[]
  modelOverrides: Record<string, DaemonGraphNodeModelOverride>
}

export interface DaemonUserCommandSummary {
  id: string
  name: string
  description: string
  args: 'none' | 'optional' | 'required'
  agent?: string
  model?: string
}

export interface DaemonUserCommandInput {
  id: string
  name?: string
  description: string
  args?: 'none' | 'optional' | 'required'
  agent?: string
  model?: string
  body: string
}

export interface DaemonUserCommandResolved {
  id: string
  prompt: string
  agent?: string
  model?: string
}

export interface DaemonUserAgentInput {
  id: string
  name?: string
  description: string
  base?: string
  model?: string
  temperature?: number
  topP?: number
  maxTokens?: number
  maxIterations?: number
  systemPrompt: string
}

export interface McpPromptDescriptor {
  name: string
  description: string
  arguments: Array<{ name: string; description?: string; required?: boolean }>
}

export interface McpPromptMessage {
  role: string
  content: unknown
}

export interface McpResourceDescriptor {
  uri: string
  name?: string
  title?: string
  description?: string
  mimeType?: string
  size?: number
  annotations?: {
    audience?: Array<'user' | 'assistant'>
    priority?: number
    lastModified?: string
  }
  icons?: Array<{
    src: string
    mimeType?: string
    sizes?: string[]
    theme?: 'light' | 'dark'
  }>
  _meta?: Record<string, unknown>
}

export interface McpResourceTemplateDescriptor {
  uriTemplate: string
  name?: string
  title?: string
  description?: string
  mimeType?: string
  annotations?: {
    audience?: Array<'user' | 'assistant'>
    priority?: number
    lastModified?: string
  }
  icons?: Array<{
    src: string
    mimeType?: string
    sizes?: string[]
    theme?: 'light' | 'dark'
  }>
  _meta?: Record<string, unknown>
}

export interface McpResourceReadResult {
  contents: Array<{
    uri: string
    text?: string
    blob?: string
    mimeType?: string
    _meta?: Record<string, unknown>
  }>
}

export interface McpResourceSubscriptionResult {
  uri: string
  subscribed: boolean
}

export interface McpResourceUpdate {
  uri: string
  timestamp: string
}

export type McpCompletionRef =
  | { type: 'ref/prompt'; name: string }
  | { type: 'ref/resource'; uri: string }

export interface McpCompletionInput {
  ref: McpCompletionRef
  argument: { name: string; value: string }
  context?: { arguments?: Record<string, string> }
}

export interface McpCompletionResult {
  completion: {
    values: string[]
    total?: number
    hasMore?: boolean
  }
  _meta?: Record<string, unknown>
}

export type McpLoggingLevel =
  | 'debug'
  | 'info'
  | 'notice'
  | 'warning'
  | 'error'
  | 'critical'
  | 'alert'
  | 'emergency'

export interface McpLogMessage {
  level: McpLoggingLevel
  logger?: string
  data: unknown
  timestamp: string
}

export interface McpLoggingState {
  level: McpLoggingLevel | null
  messages: McpLogMessage[]
}

export type DaemonTicket = Ticket
export type DaemonMemoryContextItem = MemoryContextItem
export type DaemonContextUsage = AgentContextUsage

/** Pending human decision that blocked a run when a timeout frame fired. */
export interface DaemonPendingDecision {
  kind: 'approval' | 'question'
  id: string
  label?: string
  since: number
}

/**
 * Daemon error payload. Timeout frames (`AGENT_INACTIVITY`,
 * `APPROVAL_TIMEOUT`, `QUESTION_TIMEOUT`) carry a machine-readable `reason`,
 * the blocking `pendingDecision`, and a structured `stopReason`.
 */
export type DaemonStreamError = ApiError & {
  reason?: string
  pendingDecision?: DaemonPendingDecision
  stopReason?: RunStopReason
  provider?: string
  model?: string
  lastEventType?: string
}

export type DaemonAgentEvent =
  | (Exclude<AgentEvent, { type: 'done' } | { type: 'error' }> & { sessionId?: string })
  | { type: 'done'; usage: DaemonUsage; stopReason?: RunStopReason; sessionId?: string }
  | { type: 'error'; error: DaemonStreamError; sessionId?: string }

export interface DaemonChatSessionEvent {
  sessionId: string
}

export interface DaemonChatArtifactsEvent {
  artifacts: DaemonArtifact[]
  sessionId?: string
}

export interface DaemonChatWarningEvent {
  type: 'warning'
  code: string
  message: string
  skipped?: Array<{
    source: 'path' | 'url' | 'unknown'
    filename?: string
    reason: string
  }>
  sessionId?: string
}

/**
 * Mid-run user steering note was queued onto the live run's graph state
 * (`POST /sessions/:id/steer`). Distinct from `DaemonSteeringConsumedEvent`
 * (below), which fires once the running agent loop has actually picked the
 * note up and surfaced it to the model.
 */
export interface DaemonSteeringAckEvent {
  type: 'steering_ack'
  noteId: string
  kind: 'instruction' | 'question'
  message: string
}

/** The agent loop consumed a previously-queued steering note. */
export interface DaemonSteeringConsumedEvent {
  type: 'steering_consumed'
  noteId: string
}

export type DaemonChatStreamPayload =
  | DaemonChatSessionEvent
  | DaemonAgentEvent
  | DaemonChatArtifactsEvent
  | DaemonChatWarningEvent
  | DaemonSteeringAckEvent
  | DaemonSteeringConsumedEvent
  | Record<string, never>

export interface DaemonSessionWatchSnapshot {
  type: 'snapshot'
  session: DaemonSessionDetail
}

export interface DaemonSessionWatchDeleted {
  type: 'deleted'
  sessionId: string
}

export interface DaemonSessionWatchHeartbeat {
  type: 'heartbeat'
  sessionId: string
  timestamp: string
}

export type DaemonSessionWatchPayload =
  | DaemonSessionWatchSnapshot
  | DaemonSessionWatchDeleted
  | DaemonSessionWatchHeartbeat

type PrefixDaemonWsAgentEvent<Event extends { type: string }> = Omit<Event, 'type'> & {
  type: `agent.${Event['type']}`
  sessionId: string
}

export type DaemonWsAgentEvent = AgentEvent extends infer Event
  ? Event extends { type: string }
    ? PrefixDaemonWsAgentEvent<Event>
    : never
  : never

export type DaemonWsEvent =
  | { type: 'chat.session'; sessionId: string }
  | { type: 'chat.artifacts'; artifacts: DaemonArtifact[]; sessionId: string }
  | DaemonChatWarningEvent
  | DaemonWsAgentEvent
  | { type: 'error'; error: ApiError }
  | { type: 'pong' }

export interface McpMetricsPerTool {
  calls: number
  errors: number
  totalDurationMs: number
}

export interface McpMetricsEntry {
  totalCalls: number
  errors: number
  totalDurationMs: number
  lastCallAt: string | null
  perTool: Record<string, McpMetricsPerTool>
}

export interface McpMetricsSnapshot {
  servers: Record<string, McpMetricsEntry>
}

export interface McpServerTemplate {
  name: string
  description: string
  transport: string
  command?: string
  args?: string[]
  url?: string
  headers?: Record<string, string>
  env?: Record<string, string>
  tags: string[]
  homepage?: string
  variables?: McpServerTemplateVariable[]
  marketplace: string
}

export interface McpServerTemplateVariable {
  name: string
  label?: string
  description?: string
  placeholder?: string
  required?: boolean
  secret?: boolean
}

export interface McpMarketplace {
  name: string
  url: string
  addedAt: string
  lastSync: string | null
}

export interface McpServerToolsResult {
  enabled: string[]
  disabled: string[]
  /** Every tool currently advertised by the server, including quarantined tools. */
  advertised?: string[]
  /** True when no advertised tool is callable until its changed manifest is trusted. */
  quarantined?: boolean
}

export interface McpToolCallResult {
  output: string
  status: 'success' | 'error'
  durationMs: number
  code?: string
}

export interface McpInstallResult {
  installed: boolean
  serverName: string
}

export type {
  PersonalKnowledge,
  PersonalKnowledgeWrite,
  PersonalKnowledgeCapture,
  PersonalKnowledgeSource,
  PersonalKnowledgeSuggestion,
  PersonalKnowledgeReview,
  PersonalKnowledgeKind,
  PersonalKnowledgeStatus,
  PersonalKnowledgeRelation,
} from './knowledge-types.js'
export type {
  KnowledgeChatUpdateMode,
  KnowledgeContentProposal,
  KnowledgeMaintenanceState,
  KnowledgeActivity,
  KnowledgeLlmCall,
  KnowledgeActivityDetail,
  KnowledgeActivityPage,
} from './knowledge-types.js'
export type {
  KnowledgeBudgetState,
  KnowledgeVerification,
  KnowledgeLifecycleState,
  KnowledgeVerificationInput,
} from './knowledge-types.js'
