import { ApiHttpClient, type ApiRequestInit, type MemoryScope, type Resolvable } from '../http.js'
import type { ApprovalDecisionStatus } from '@sepilotd/core'
import type {
  ApiEnvelope,
  DaemonAgentMode,
  DaemonAgentDescriptor,
  DaemonAgentGraphSnapshot,
  DaemonAssistantRuntimeStatus,
  DaemonArtifact,
  DaemonStateBoardResponse,
  DaemonRecentArtifact,
  DaemonApprovalResponseResult,
  DaemonRememberedApproval,
  DaemonRememberedApprovalDescribeInput,
  DaemonRememberedApprovalInput,
  DaemonRememberedApprovalMatch,
  DaemonRememberedApprovalRule,
  DaemonChatBackgroundListResult,
  DaemonChatBackgroundStartResult,
  DaemonChatBackgroundStatusResult,
  DaemonChatResult,
  DaemonFileContentPart,
  DaemonFileMemorySectionUpdateResult,
  DaemonFileMemorySnapshot,
  DaemonFileUploadResult,
  DaemonConfig,
  DaemonConfigProvider,
  DaemonConfigEnvUpdateInput,
  DaemonConfigEnvUpdateResult,
  DaemonConfigUpdateInput,
  DaemonConfigUpdateResult,
  DaemonDailyUsageSummary,
  DaemonDevice,
  DaemonDoctorReport,
  DaemonHealth,
  DaemonHealthReportSnapshot,
  DaemonNativeServiceControlInput,
  DaemonNativeServiceInstallInput,
  DaemonNativeServiceLogChunk,
  DaemonNativeServiceSnapshot,
  DaemonNativeServiceUninstallResult,
  DaemonPendingApproval,
  DaemonImageGenJob,
  DaemonImageGenJobInput,
  DaemonImageGenProvider,
  DaemonSnippet,
  DaemonSnippetInput,
  DaemonWikiNode,
  DaemonWikiNodeInput,
  DaemonWikiMoveInput,
  DaemonWikiSearchHit,
  DaemonMemoryAddResult,
  DaemonMemoryAddOptions,
  DaemonMemoryAuditEntry,
  DaemonMemoryAuditOptions,
  DaemonMemoryDeleteOptions,
  DaemonMemoryDocument,
  DaemonMemoryDocumentChunk,
  DaemonMemoryDocumentIngestInput,
  DaemonMemoryDocumentListOptions,
  DaemonMemoryDocumentSearchOptions,
  DaemonMemoryEntry,
  DaemonMemoryPinnedEntry,
  DaemonMemoryGraphAuditOptions,
  DaemonMemoryGraphPageOptions,
  DaemonMemoryGraphQualityReport,
  DaemonMemoryGraphRepairDecisionInput,
  DaemonMemoryGraphRepairDecisionResult,
  DaemonMemoryGraphRepairOptions,
  DaemonMemoryGraphRepairResult,
  DaemonMemoryGraphWikiPage,
  DaemonMemoryLifecycleOptions,
  DaemonMemoryLifecycleStatus,
  DaemonMemoryMaintenanceInput,
  DaemonMemoryMaintenanceResult,
  DaemonMemoryRecentEntry,
  DaemonMemoryRecentOptions,
  DaemonMemoryReindexResult,
  DaemonMemorySearchOptions,
  DaemonMemoryScopes,
  DaemonMemoryScopeTransferInput,
  DaemonMemoryScopeTransferResult,
  DaemonMemorySecurityAuditOptions,
  DaemonMemorySecurityAuditResult,
  DaemonMemorySemanticStatus,
  DaemonMemoryUpdateInput,
  DaemonModelPullInput,
  DaemonModelPullResponse,
  DaemonModelSnapshot,
  DaemonModelSwitchInput,
  DaemonModelSwitchResult,
  DaemonPersona,
  DaemonPolicySnapshot,
  DaemonProviderInfo,
  DaemonProviderMutationResult,
  DaemonProviderDiscoverModelsInput,
  DaemonProviderDiscoverModelsResult,
  DaemonProviderRefreshModelsResult,
  DaemonProviderValidationInput,
  DaemonProviderValidationResult,
  DaemonProject,
  DaemonMcpClientConfig,
  DaemonMcpServerConfig,
  DaemonMcpServerStatus,
  McpMetricsEntry,
  McpMetricsSnapshot,
  McpCompletionInput,
  McpCompletionResult,
  McpLoggingLevel,
  McpLoggingState,
  McpPromptDescriptor,
  McpPromptMessage,
  McpResourceDescriptor,
  McpResourceReadResult,
  McpResourceSubscriptionResult,
  McpResourceTemplateDescriptor,
  McpResourceUpdate,
  McpServerTemplate,
  McpMarketplace,
  McpServerToolsResult,
  McpToolCallResult,
  McpInstallResult,
  DaemonGitHubOAuthStartResult,
  DaemonGitHubOAuthStatus,
  DaemonGitHubChatSyncConfig,
  DaemonGitHubChatSyncConfigUpdate,
  DaemonGitHubChatSyncResult,
  DaemonGitHubSyncPolicy,
  DaemonGitHubSyncRepo,
  DaemonListPage,
  DaemonNetworkConfig,
  DaemonNetworkStatus,
  DaemonNetworkProbeResult,
  DaemonFeedbackInput,
  DaemonFeedbackRecord,
  DaemonFeedbackPromptState,
  DaemonFeedbackPromptStateInput,
  DaemonObservabilityEvent,
  DaemonObservabilityEventInput,
  DaemonObservabilityExportInput,
  DaemonObservabilityExportResult,
  DaemonObservabilityPrivacySettings,
  DaemonObservabilityPrivacySettingsInput,
  DaemonObservabilityPruneResult,
  DaemonObservabilityRange,
  DaemonObservabilitySeverity,
  DaemonObservabilitySnapshot,
  DaemonObservabilitySupportBundleInput,
  DaemonObservabilitySupportBundleResult,
  DaemonPersonalDoc,
  DaemonPersonalDocContent,
  DaemonPersonalDocInput,
  DaemonBackupItem,
  DaemonPromptTemplate,
  DaemonPromptTemplateInput,
  DaemonQuickInputPublishResult,
  DaemonQuickInputSettings,
  DaemonRagDocument,
  DaemonRagDocumentContent,
  DaemonRagDocumentInput,
  DaemonRagFolder,
  DaemonRagFolderInput,
  DaemonRagConnectionTestResult,
  DaemonRagSearchHit,
  DaemonRagSyncResult,
  DaemonRagVectorDbInfo,
  DaemonOutboundWebhookConfig,
  DaemonOutboundWebhookDeadLetter,
  DaemonOutboundWebhookDelivery,
  DaemonOutboundWebhookSummary,
  DaemonWebhookEndpointConfig,
  DaemonWebhookEndpointSummary,
  DaemonExtensionTokenSummary,
  DaemonIssueExtensionTokenInput,
  DaemonIssuedExtensionToken,
  DaemonSessionCompactResult,
  DaemonSessionBranchResult,
  DaemonSessionDetail,
  DaemonSessionExportSnapshot,
  DaemonSessionImportResult,
  DaemonSessionManagementCleanupResult,
  DaemonSessionManagementSnapshot,
  DaemonSessionRunbook,
  DaemonSessionShareMode,
  DaemonSessionShareResult,
  DaemonServiceLogChunk,
  DaemonServiceLogsOptions,
  DaemonServiceRemoveInput,
  DaemonServiceRemoveResult,
  DaemonServiceSnapshot,
  DaemonServiceStartInput,
  DaemonServiceStopInput,
  DaemonSessionList,
  DaemonSessionMeta,
  DaemonSkill,
  DaemonSkillCreateInput,
  DaemonSkillCreateResult,
  DaemonSkillDetail,
  InstallSkillPreviewResponse,
  MarketplaceSkillSearchResult,
  DaemonMessageSubscriptionConfig,
  DaemonMessageSubscriptionItem,
  DaemonMessageSubscriptionOverview,
  DaemonMessageSubscriptionProcessResult,
  DaemonMessageSubscriptionRefreshResult,
  DaemonNotificationDraft,
  DaemonNotificationItem,
  DaemonNotificationSettings,
  DaemonSchedulerJob,
  DaemonSchedulerJobInput,
  DaemonSchedulerDeliveryOutboxSummary,
  DaemonSchedulerManualRunOptions,
  DaemonSchedulerManualRunResult,
  DaemonSchedulerNotificationSubscriptions,
  DaemonSchedulerJobRun,
  DaemonScheduledTaskInput,
  DaemonScheduledTaskUpdateInput,
  DaemonSettingsJsonDocument,
  DaemonDiscordChannelConfig,
  DaemonDiscordChannelSummary,
  DaemonMattermostChannelConfig,
  DaemonMattermostChannelSummary,
  DaemonSlackChannelConfig,
  DaemonSlackChannelSummary,
  DaemonTelegramChannelConfig,
  DaemonTelegramChannelSummary,
  DaemonTelegramPairingCode,
  DaemonTeamDocsActionResult,
  DaemonTeamDocsConfig,
  DaemonTeamDocsConfigInput,
  DaemonTeamDocsDocument,
  DaemonTeamDocsDocumentContent,
  DaemonTeamDocsSyncAllResult,
  DaemonUploadedFile,
  DaemonUploadedFileIndexInput,
  DaemonUsageSnapshot,
  DaemonUsageSummary,
  EditCheckpointSummary,
} from './types.js'

export const DEFAULT_DAEMON_BASE_URL = 'http://127.0.0.1:17600'
const MAX_FILES_PER_UPLOAD_REQUEST = 10

export interface ChatSkillRef {
  name: string
}

export type ChatTextDeltaMode = 'buffered' | 'live'

export interface ChatOptions {
  /**
   * Stable client-generated id for this user turn. The daemon scopes it to
   * the session and reuses the existing result/job when a transport retry
   * submits the same id again.
   */
  messageId?: string
  model?: string
  provider?: string
  /** Optional session labels persisted on newly-created chat sessions. */
  tags?: string[]
  thinkingLevel?: string
  persona?: string
  /**
   * Persona roster for the `persona-panel` mode. Each id resolves to a
   * built-in or custom persona on the daemon; ignored by other modes.
   * Capped at 6 server-side.
   */
  personaIds?: string[]
  /**
   * Execution policy for `persona-panel` turns. `sequential` calls every
   * resolved persona exactly once in roster order and gives each caller the
   * accumulated transcript. `moderated` keeps the dynamic meeting workflow.
   */
  panelStrategy?: 'sequential' | 'moderated'
  mode?: DaemonAgentMode
  /** Desktop writing-canvas document id bound to this chat turn. */
  writingDocId?: string
  projectId?: string
  maxTokens?: number
  /**
   * Agent-loop iteration cap for this turn. One iteration is one LLM round
   * (which may emit multiple parallel tool calls). Defaults to the daemon's
   * `SEPILOTD_CHAT_MAX_ITERATIONS` env or `50` if unset. Bump explicitly
   * for multi-pass skills (e.g. `software-architect`'s reverse-engineering
   * mode on a non-trivial codebase, where 50 may not be enough to inventory
   * + write the full document chain in one turn). Hard ceiling 500.
   */
  maxIterations?: number
  /** Treat maxIterations as a hard cap: disables continuation cycles and repair slack. Automation/bench only. */
  hardMaxIterations?: boolean
  /** Sampling temperature (0~2). Daemon clamps and forwards to provider. */
  temperature?: number
  cwd?: string
  /**
   * Strict filesystem boundary for this turn. Tools may not read or mutate
   * paths outside this root, and approval never widens it.
   */
  workspaceRoot?: string
  /**
   * Optional list of skill references the daemon should resolve and
   * prepend to the system prompt before running the agent loop. Daemon
   * accepts `{ name }` objects and 404s on unknown names; callers that
   * never use skills can omit this field. See
   * `packages/daemon/src/server/routes/chat-schema.ts` for the
   * authoritative server-side shape.
   */
  skillRefs?: ChatSkillRef[]
  /**
   * Optional per-turn tool allowlist. Omit to expose the daemon's normal tool
   * set; pass an empty array for a tool-free chat turn.
   */
  toolNames?: string[]
  /** Toggle daemon-side RAG retrieval for this turn. */
  ragEnabled?: boolean
  /** Tell daemon the user opted into image generation for this turn. */
  imageGenEnabled?: boolean
  /** Surface input-trust level so daemon can tighten approvals on untrusted input. */
  inputTrustLevel?: 'trusted' | 'untrusted'
  /**
   * Explicit per-turn autonomy selection. The daemon applies independent
   * channel ACL and tool-policy ceilings after resolving this session value.
   */
  autonomy?: 'readonly' | 'accept-edits' | 'workspace-write' | 'supervised' | 'autonomous'
  /**
   * Require a fresh human approval for every side-effecting tool in this turn.
   * This does not relax policy denials, read-only mode, or workspace bounds.
   */
  requireToolApproval?: boolean
  /**
   * Per-turn control for daemon intent routing. Set `{ enabled: false }`
   * when the caller's persona/skill selection should be honored verbatim.
   * Explicit concrete modes are authoritative even when routing is enabled.
   */
  intentRouting?: {
    enabled?: boolean
  }
}

export interface ChatStreamOptions extends ChatOptions {
  fileIds?: string[]
  /** Last SSE cursor seen by the caller; forwarded as body + Last-Event-ID. */
  lastEventId?: string
  /**
   * Streaming-only opt-in. The default `buffered` preserves the daemon's
   * historical behavior: provider text is parsed first, then emitted as a
   * final text_delta. `live` asks native tool-use turns to forward safe final
   * answer chunks after the daemon sees the answer protocol stem; raw tool-call
   * text and interim-progress replies are not live-streamed.
   */
  textDeltaMode?: ChatTextDeltaMode
}

export type ChatOptionDefaults = Partial<ChatOptions>

export interface DaemonRequestControlOptions {
  signal?: AbortSignal
}

/**
 * Control plane for the compatibility synchronous chat endpoint. Unlike SSE,
 * this request cannot use heartbeat frames to distinguish a live long-running
 * agent from a dead connection, so callers that deliberately need one final
 * JSON envelope may provide a bounded wall-clock timeout explicitly.
 */
export interface DaemonChatRequestControlOptions extends DaemonRequestControlOptions {
  timeoutMs?: number
  /** Node/Bun transport whose dispatcher lifetime is at least timeoutMs. */
  fetch?: typeof fetch
}

interface DaemonToolResult {
  status: 'success' | 'error'
  output: string
  durationMs: number
  code?: string
}

export const DEFAULT_CHAT_OPTION_DEFAULTS = Object.freeze({
  mode: 'auto',
  thinkingLevel: 'medium',
  persona: 'default',
  // 'default' is the TUI's sentinel for "let the daemon resolve provider/model".
  // Strip it here so the literal string never reaches the daemon, where it would
  // be looked up as a provider/model id and fail with 503 "No provider" on any
  // deployment whose default provider isn't literally named 'default'.
  provider: 'default',
  model: 'default',
} satisfies ChatOptionDefaults)

type SanitizableChatOptions = ChatOptions & {
  fileIds?: string[]
  lastEventId?: string
  textDeltaMode?: ChatTextDeltaMode
}

export interface DaemonListSessionsOptions {
  page?: number
  perPage?: number
  /** Return only sessions bound to this exact immutable workspace root. */
  workspaceRoot?: string
  /**
   * Opt-in: when true, daemon scans each session's event log to fill
   * `approvalCounters` per item. Costs an extra getEvents per row,
   * so default off for backward compat — light list views (web,
   * desktop overlay) leave it false; cli passes true so the audit
   * column shows up.
   */
  metrics?: boolean
  /**
   * Narrow the list to a single session status. Daemon filters
   * post-fetch and adjusts totalCount so "showing X of Y" reflects
   * the filtered set. Useful for cli operators who want only active
   * runs (drops abandoned/completed noise) or only completed
   * (audit-after-the-fact).
   */
  status?: 'active' | 'completed' | 'abandoned'
}

export interface DaemonSessionExportOptions {
  /**
   * Ask the daemon to redact common secret and local-path patterns before
   * returning the export. Omit to preserve the daemon's historical raw export
   * behavior; pass false when the caller is making an explicit raw-export UI.
   */
  sanitize?: boolean
}

export type DaemonSessionUpdateInput = {
  title?: string
  status?: 'active' | 'completed' | 'abandoned'
  cwd?: string | null
  workspaceIsolation?: 'policy' | 'strict'
  /** Ordered persona roster. `[]` explicitly restores the default assistant. */
  personaIds?: string[]
  starred?: boolean
  tags?: string[]
}

export interface DaemonRecordTranscriptTurnInput {
  userContent: string
  assistantContent?: string
  title?: string
  provider?: string
  model?: string
  cwd?: string
  tags?: string[]
}

export interface DaemonRecordLocalShellTurnInput {
  command: string
  cwd?: string
  shell?: string
  args?: string[]
  stdout?: string
  stderr?: string
  exitCode: number
  signal?: string | null
  durationMs: number
  timedOut?: boolean
  maxBufferExceeded?: boolean
  title?: string
  provider?: string
  model?: string
  tags?: string[]
}

export interface DaemonOutboundWebhookDeliveryOptions {
  id?: string
  status?: 'success' | 'error'
  cursor?: string
  limit?: number
}

export interface DaemonOutboundWebhookDeadLetterOptions {
  id?: string
  state?: 'open' | 'acknowledged' | 'all'
  cursor?: string
  limit?: number
}

export interface DaemonReplayOutboundWebhookDeadLettersOptions extends DaemonOutboundWebhookDeadLetterOptions {
  force?: boolean
}

export interface DaemonAcknowledgeOutboundWebhookDeadLetterOptions {
  note?: string
}

export interface DaemonUploadFileInput {
  filename: string
  content: Blob | ArrayBuffer | ArrayBufferView | string
  mimeType?: string
}

export interface DaemonCreateProjectInput {
  name: string
  description?: string
  instructions?: string
  workingDirectory?: string
}

export interface DaemonUpdateProjectInput {
  name?: string
  description?: string
  instructions?: string
  workingDirectory?: string
  sessionIds?: string[]
  fileIds?: string[]
}

export interface SubagentDispatchInput {
  isolation?: 'worktree'
  cwd?: string
  prompt: string
  system?: string
  category?: string
  agentId?: string
  maxIterations?: number
  /** Treat maxIterations as a hard cap: disables continuation cycles and repair slack. Automation/bench only. */
  hardMaxIterations?: boolean
  tools?: string[]
  model?: string
  parentSessionId?: string
}

export interface SessionInboxItem {
  seq: number
  sessionId: string
  source: 'subagent' | 'hook' | 'monitor' | 'scheduler'
  sourceId: string
  title: string
  body: string
  createdAt: number
  acknowledgedAt: number | null
}

export interface WorkItem {
  key: string
  kind: 'job' | 'run' | 'process' | 'schedule' | 'approval' | 'service' | 'chat'
  id: string
  sessionId?: string
  status: string
  title: string
  detail: string
  action: string
}

export interface SubagentDispatchResult {
  worktree?: { path: string; branch: string; baseCommit: string; retained: boolean; reason: string }
  output: string
  sessionId: string
  category?: string
  iterations: number
  usage: { inputTokens: number; outputTokens: number }
  truncated: boolean
  status: 'completed' | 'failed' | 'truncated'
  error?: string
}

export interface SubagentBackgroundDispatchResult {
  jobId: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'canceled' | string
  total: number
  createdAt: number
}

export interface SubagentDelegationCategoryInfo {
  id: string
  label: string
  description: string
  defaultMaxIterations: number
  toolHints: readonly string[] | null
}

export interface SubagentDelegationCategoriesResult {
  categories: SubagentDelegationCategoryInfo[]
}

export type WorkPlanStatus = 'draft' | 'ready' | 'running' | 'completed' | 'archived'
export type WorkPlanStepStatus = 'pending' | 'in_progress' | 'done' | 'blocked' | 'skipped'

export interface WorkPlanStep {
  id: string
  title: string
  status: WorkPlanStepStatus
  detail?: string
  children?: WorkPlanStep[]
}

export interface WorkPlan {
  id: string
  title: string
  goal: string
  status: WorkPlanStatus
  steps: WorkPlanStep[]
  acceptanceCriteria: string[]
  risks: string[]
  decisions: string[]
  sourceSessionId?: string
  lastJobId?: string
  lastStartedAt?: string
  createdAt: string
  updatedAt: string
}

export interface WorkPlanCreateInput {
  title?: string
  goal: string
  steps?: WorkPlanStep[]
  acceptanceCriteria?: string[]
  risks?: string[]
  decisions?: string[]
  sourceSessionId?: string
  status?: WorkPlanStatus
}

export interface WorkPlanPatchInput {
  title?: string
  goal?: string
  status?: WorkPlanStatus
  steps?: WorkPlanStep[]
  acceptanceCriteria?: string[]
  risks?: string[]
  decisions?: string[]
}

export interface WorkPlanListResult {
  plans: WorkPlan[]
}

export interface WorkPlanStartInput {
  category?: string
  agentId?: string
  model?: string
  tools?: string[]
  maxIterations?: number
  /** Treat maxIterations as a hard cap: disables continuation cycles and repair slack. Automation/bench only. */
  hardMaxIterations?: boolean
}

export interface WorkPlanStartResult {
  plan: WorkPlan
  job: SubagentBackgroundDispatchResult
}

export type ExternalAcpAgentName = 'opencode' | 'codex'

export interface ExternalAcpDispatchInput {
  prompt: string
  agent?: ExternalAcpAgentName
  cwd?: string
  sessionId?: string
  timeoutMs?: number
}

export interface ExternalAcpEvent {
  type: string
  text?: string
  title?: string
  status?: string
}

export interface ExternalAcpDispatchResult {
  output: string
  sessionId: string
  externalSessionId: string
  agent: string
  stopReason: string
  status: 'completed' | 'failed'
  events: ExternalAcpEvent[]
  error?: string
}

export interface DaemonClientOptions {
  baseUrl?: string
  token?: string | null
  /** UI/client surface label sent as X-Sepilotd-Surface for routing and diagnostics. */
  surface?: Resolvable<string | null | undefined>
  /**
   * Memory scope to attach to outgoing requests as X-Memory-Scope-*
   * headers. Surfaces typically resolve this lazily by fetching
   * GET /api/v1/system/scope once on boot. Resolvable so the value can
   * arrive after the client is constructed.
   */
  memoryScope?: Resolvable<MemoryScope | null | undefined>
}

interface ResolvedDaemonClientOptions {
  baseUrl: string
  token: string | null
  surface?: Resolvable<string | null | undefined>
  memoryScope?: Resolvable<MemoryScope | null | undefined>
}

function parseOptions(options?: string | DaemonClientOptions): ResolvedDaemonClientOptions {
  if (typeof options === 'string') {
    return { baseUrl: options, token: null }
  }
  return {
    baseUrl: options?.baseUrl ?? DEFAULT_DAEMON_BASE_URL,
    token: options?.token ?? null,
    surface: options?.surface,
    memoryScope: options?.memoryScope,
  }
}

function apiPath(path: string): string {
  return `/api/v1${path.startsWith('/') ? path : `/${path}`}`
}

function capabilityPath(path: string): string {
  return path.startsWith('/') ? path : `/${path}`
}

function normalizeOptionalString(value?: string | null): string | undefined {
  const trimmed = value?.trim()
  return trimmed ? trimmed : undefined
}

function omitDefaultString(
  value: string | undefined,
  defaultValue?: string | null,
): string | undefined {
  const normalizedDefault = normalizeOptionalString(defaultValue)
  return value && value === normalizedDefault ? undefined : value
}

function appendDefinedQuery(params: URLSearchParams, key: string, value: unknown): void {
  if (value !== undefined && value !== null) {
    params.set(key, String(value))
  }
}

function appendServiceLogsQuery(params: URLSearchParams, options?: DaemonServiceLogsOptions): void {
  appendDefinedQuery(params, 'stdoutOffset', options?.stdoutOffset)
  appendDefinedQuery(params, 'stderrOffset', options?.stderrOffset)
  appendDefinedQuery(params, 'limitBytes', options?.limitBytes)
  appendDefinedQuery(params, 'tailBytes', options?.tailBytes)
  appendDefinedQuery(params, 'followMs', options?.followMs)
  appendDefinedQuery(params, 'pollIntervalMs', options?.pollIntervalMs)
}

function serviceLogsPath(basePath: string, options?: DaemonServiceLogsOptions): string {
  const params = new URLSearchParams()
  appendServiceLogsQuery(params, options)
  const qs = params.toString()
  return qs ? `${basePath}?${qs}` : basePath
}

function parseDaemonToolOutput<T>(result: DaemonToolResult): T {
  if (result.status === 'error') {
    const code = result.code ? `${result.code}: ` : ''
    throw new Error(`${code}${result.output}`)
  }

  try {
    return JSON.parse(result.output) as T
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(`Daemon tool returned invalid JSON: ${message}`)
  }
}

export function sanitizeChatOptions(
  options?: SanitizableChatOptions,
  defaults?: ChatOptionDefaults,
): SanitizableChatOptions | undefined {
  if (!options) return undefined

  const next: SanitizableChatOptions = { ...options }
  const nextRecord = next as Record<string, unknown>

  const provider = omitDefaultString(normalizeOptionalString(options.provider), defaults?.provider)
  const model = omitDefaultString(normalizeOptionalString(options.model), defaults?.model)
  const thinkingLevel = omitDefaultString(
    normalizeOptionalString(options.thinkingLevel),
    defaults?.thinkingLevel,
  )
  const persona = omitDefaultString(normalizeOptionalString(options.persona), defaults?.persona)
  const projectId = omitDefaultString(
    normalizeOptionalString(options.projectId),
    defaults?.projectId,
  )
  const writingDocId = normalizeOptionalString(options.writingDocId)
  const messageId = normalizeOptionalString(options.messageId)
  const mode = omitDefaultString(normalizeOptionalString(options.mode), defaults?.mode) as
    | DaemonAgentMode
    | undefined

  if (provider) next.provider = provider
  else delete next.provider

  if (model) next.model = model
  else delete next.model

  if (thinkingLevel) next.thinkingLevel = thinkingLevel
  else delete next.thinkingLevel

  if (persona) next.persona = persona
  else delete next.persona

  if (projectId) next.projectId = projectId
  else delete next.projectId

  if (writingDocId) next.writingDocId = writingDocId
  else delete next.writingDocId

  if (messageId) next.messageId = messageId
  else delete next.messageId

  if (mode) next.mode = mode
  else delete next.mode

  if ('tags' in nextRecord) {
    const tags = Array.isArray(nextRecord.tags)
      ? nextRecord.tags
          .map((tag) => (typeof tag === 'string' ? normalizeOptionalString(tag) : undefined))
          .filter((tag): tag is string => Boolean(tag))
          .slice(0, 16)
      : undefined
    if (tags?.length) {
      nextRecord.tags = tags
    } else {
      delete nextRecord.tags
    }
  }

  if (
    typeof options.maxTokens === 'number' &&
    Number.isFinite(options.maxTokens) &&
    options.maxTokens > 0
  ) {
    next.maxTokens = Math.floor(options.maxTokens)
  } else {
    delete next.maxTokens
  }

  // Clamp to the same [1, 500] range the daemon's chat-schema enforces, so
  // callers get a predictable rejection (drop) rather than a 400 when they
  // pass nonsense.
  if (
    typeof options.maxIterations === 'number' &&
    Number.isFinite(options.maxIterations) &&
    options.maxIterations >= 1
  ) {
    next.maxIterations = Math.min(500, Math.floor(options.maxIterations))
  } else {
    delete next.maxIterations
  }
  if (options.hardMaxIterations === true) next.hardMaxIterations = true
  else delete next.hardMaxIterations

  if (
    typeof options.temperature === 'number' &&
    Number.isFinite(options.temperature) &&
    options.temperature >= 0 &&
    options.temperature <= 2
  ) {
    next.temperature = options.temperature
  } else {
    delete next.temperature
  }

  const cwd = normalizeOptionalString(options.cwd)
  if (cwd) next.cwd = cwd
  else delete next.cwd

  const workspaceRoot = normalizeOptionalString(options.workspaceRoot)
  if (workspaceRoot) next.workspaceRoot = workspaceRoot
  else delete next.workspaceRoot

  if (nextRecord.textDeltaMode === 'live') {
    nextRecord.textDeltaMode = 'live'
  } else {
    delete nextRecord.textDeltaMode
  }

  if ('fileIds' in nextRecord) {
    const fileIds = Array.isArray(nextRecord.fileIds)
      ? nextRecord.fileIds
          .map((fileId) =>
            typeof fileId === 'string' ? normalizeOptionalString(fileId) : undefined,
          )
          .filter((fileId): fileId is string => Boolean(fileId))
      : undefined

    if (fileIds?.length) {
      nextRecord.fileIds = fileIds
    } else {
      delete nextRecord.fileIds
    }
  }

  if ('personaIds' in nextRecord) {
    const personaIds = Array.isArray(nextRecord.personaIds)
      ? nextRecord.personaIds
          .map((id) => (typeof id === 'string' ? normalizeOptionalString(id) : undefined))
          .filter((id): id is string => Boolean(id))
          .slice(0, 6)
      : undefined
    if (personaIds?.length) {
      nextRecord.personaIds = personaIds
    } else {
      delete nextRecord.personaIds
    }
  }

  if (
    nextRecord.panelStrategy !== 'sequential'
    && nextRecord.panelStrategy !== 'moderated'
  ) {
    delete nextRecord.panelStrategy
  }

  if ('toolNames' in nextRecord) {
    if (Array.isArray(nextRecord.toolNames)) {
      nextRecord.toolNames = nextRecord.toolNames
        .map((name) => (typeof name === 'string' ? normalizeOptionalString(name) : undefined))
        .filter((name): name is string => Boolean(name))
        .slice(0, 128)
    } else {
      delete nextRecord.toolNames
    }
  }

  for (const key of Object.keys(next) as (keyof SanitizableChatOptions)[]) {
    if (next[key] === undefined) {
      delete next[key]
    }
  }

  return Object.keys(next).length > 0 ? next : undefined
}

export class DaemonClient {
  readonly baseUrl: string
  readonly token: string | null
  private readonly transport: ApiHttpClient

  constructor(options?: string | DaemonClientOptions) {
    const resolved = parseOptions(options)
    this.baseUrl = resolved.baseUrl
    this.token = resolved.token
    this.transport = new ApiHttpClient({
      baseUrl: this.baseUrl,
      token: this.token,
      surface: resolved.surface,
      memoryScope: resolved.memoryScope,
    })
  }

  getHeaders(extra?: HeadersInit): Record<string, string> {
    const headers = new Headers(extra)
    if (this.token && !headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${this.token}`)
    }
    return Object.fromEntries(headers.entries())
  }

  fetch(path: string, init: ApiRequestInit = {}): Promise<Response> {
    return this.transport.fetch(path, init)
  }

  request<T = unknown>(path: string, init: ApiRequestInit = {}): Promise<T> {
    return this.transport.request(path, init)
  }

  private async get<T>(path: string, init?: Pick<ApiRequestInit, 'signal'>): Promise<T> {
    const { data } = await this.request<ApiEnvelope<T>>(apiPath(path), {
      method: 'GET',
      signal: init?.signal,
    })
    return data
  }

  private async getCapability<T>(path: string, init?: Pick<ApiRequestInit, 'signal'>): Promise<T> {
    return this.request<T>(capabilityPath(path), {
      method: 'GET',
      signal: init?.signal,
    })
  }

  private async post<T>(
    path: string,
    body?: ApiRequestInit['body'],
    init?: Pick<ApiRequestInit, 'signal' | 'timeoutMs' | 'fetch'>,
  ): Promise<T> {
    const { data } = await this.request<ApiEnvelope<T>>(apiPath(path), {
      method: 'POST',
      body,
      signal: init?.signal,
      timeoutMs: init?.timeoutMs,
      fetch: init?.fetch,
    })
    return data
  }

  private async postCapability<T>(
    path: string,
    body?: ApiRequestInit['body'],
    init?: Pick<ApiRequestInit, 'signal'>,
  ): Promise<T> {
    return this.request<T>(capabilityPath(path), {
      method: 'POST',
      body,
      signal: init?.signal,
    })
  }

  private async put<T>(path: string, body?: ApiRequestInit['body']): Promise<T> {
    const { data } = await this.request<ApiEnvelope<T>>(apiPath(path), {
      method: 'PUT',
      body,
    })
    return data
  }

  private async patch<T>(path: string, body?: ApiRequestInit['body']): Promise<T> {
    const { data } = await this.request<ApiEnvelope<T>>(apiPath(path), {
      method: 'PATCH',
      body,
    })
    return data
  }

  private async putCapability<T>(path: string, body?: ApiRequestInit['body']): Promise<T> {
    return this.request<T>(capabilityPath(path), {
      method: 'PUT',
      body,
    })
  }

  private async del(path: string): Promise<void> {
    const res = await this.fetch(apiPath(path), { method: 'DELETE' })
    if (!res.ok && res.status !== 204) {
      throw new Error(`${res.status}: ${await res.text()}`)
    }
  }

  private async delCapability(path: string): Promise<void> {
    const res = await this.fetch(capabilityPath(path), { method: 'DELETE' })
    if (!res.ok && res.status !== 204) {
      throw new Error(`${res.status}: ${await res.text()}`)
    }
  }

  async health(options?: DaemonRequestControlOptions): Promise<DaemonHealth> {
    return this.get('/health', { signal: options?.signal })
  }

  async doctor(options?: DaemonRequestControlOptions): Promise<DaemonDoctorReport> {
    return this.request<DaemonDoctorReport>(apiPath('/doctor'), {
      method: 'GET',
      signal: options?.signal,
    })
  }

  /**
   * Trigger a graceful daemon shutdown via HTTP. The daemon replies before
   * tearing the process down, so the returned promise resolves once the
   * acknowledgement is received — not when the daemon has fully exited.
   * Callers that need to wait for full exit should health-poll until it fails.
   */
  async systemShutdown(options?: DaemonRequestControlOptions): Promise<{ status: string }> {
    return this.post('/system/shutdown', undefined, { signal: options?.signal })
  }

  /**
   * Snapshot of active WS/SSE connections plus the daemon's idle window. The
   * desktop tray polls this so it can show "N connected" before the user
   * picks Quit; idle reaper consumers (operations dashboards) read it too.
   */
  async systemClients(options?: DaemonRequestControlOptions): Promise<{
    count: number
    byKind: { ws: number; sse: number }
    idleMs: number
    clients: Array<{
      id: string
      kind: 'ws' | 'sse'
      label: string
      client: string | null
      openedAt: number
    }>
  }> {
    return this.get('/system/clients', { signal: options?.signal })
  }

  /** Non-secret readiness of the runtime capabilities used by assistant mode. */
  async assistantRuntimeStatus(
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonAssistantRuntimeStatus> {
    return this.get('/system/assistant', { signal: options?.signal })
  }

  async listServices(options?: DaemonRequestControlOptions): Promise<DaemonServiceSnapshot[]> {
    return parseDaemonToolOutput(
      await this.get<DaemonToolResult>('/services', { signal: options?.signal }),
    )
  }

  async startService(
    input: DaemonServiceStartInput,
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonServiceSnapshot> {
    return parseDaemonToolOutput(
      await this.post<DaemonToolResult>('/services', input, { signal: options?.signal }),
    )
  }

  async serviceStatus(
    id: string,
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonServiceSnapshot> {
    return parseDaemonToolOutput(
      await this.get<DaemonToolResult>(`/services/${encodeURIComponent(id)}`, {
        signal: options?.signal,
      }),
    )
  }

  async serviceLogs(
    id: string,
    logOptions?: DaemonServiceLogsOptions,
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonServiceLogChunk> {
    return parseDaemonToolOutput(
      await this.get<DaemonToolResult>(
        serviceLogsPath(`/services/${encodeURIComponent(id)}/logs`, logOptions),
        { signal: options?.signal },
      ),
    )
  }

  async serviceHealthcheck(
    id: string,
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonServiceSnapshot> {
    return parseDaemonToolOutput(
      await this.post<DaemonToolResult>(
        `/services/${encodeURIComponent(id)}/healthcheck`,
        {},
        { signal: options?.signal },
      ),
    )
  }

  async stopService(
    id: string,
    input: DaemonServiceStopInput = {},
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonServiceSnapshot> {
    return parseDaemonToolOutput(
      await this.post<DaemonToolResult>(`/services/${encodeURIComponent(id)}/stop`, input, {
        signal: options?.signal,
      }),
    )
  }

  async restartService(
    id: string,
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonServiceSnapshot> {
    return parseDaemonToolOutput(
      await this.post<DaemonToolResult>(
        `/services/${encodeURIComponent(id)}/restart`,
        {},
        { signal: options?.signal },
      ),
    )
  }

  async removeService(
    id: string,
    input: DaemonServiceRemoveInput = {},
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonServiceRemoveResult> {
    const params = new URLSearchParams()
    appendDefinedQuery(params, 'force', input.force)
    appendDefinedQuery(params, 'deleteLogs', input.deleteLogs)
    const qs = params.toString()
    const { data } = await this.request<ApiEnvelope<DaemonToolResult>>(
      apiPath(`/services/${encodeURIComponent(id)}${qs ? `?${qs}` : ''}`),
      { method: 'DELETE', signal: options?.signal },
    )
    return parseDaemonToolOutput(data)
  }

  async installNativeService(
    input: DaemonNativeServiceInstallInput,
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonNativeServiceSnapshot> {
    return parseDaemonToolOutput(
      await this.post<DaemonToolResult>('/native-services', input, { signal: options?.signal }),
    )
  }

  async nativeServiceStatus(
    id: string,
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonNativeServiceSnapshot> {
    return parseDaemonToolOutput(
      await this.get<DaemonToolResult>(`/native-services/${encodeURIComponent(id)}`, {
        signal: options?.signal,
      }),
    )
  }

  async nativeServiceLogs(
    id: string,
    logOptions?: DaemonServiceLogsOptions,
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonNativeServiceLogChunk> {
    return parseDaemonToolOutput(
      await this.get<DaemonToolResult>(
        serviceLogsPath(`/native-services/${encodeURIComponent(id)}/logs`, logOptions),
        { signal: options?.signal },
      ),
    )
  }

  async enableNativeService(
    id: string,
    input: DaemonNativeServiceControlInput = {},
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonNativeServiceSnapshot> {
    return parseDaemonToolOutput(
      await this.post<DaemonToolResult>(
        `/native-services/${encodeURIComponent(id)}/enable`,
        input,
        { signal: options?.signal },
      ),
    )
  }

  async disableNativeService(
    id: string,
    input: DaemonNativeServiceControlInput = {},
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonNativeServiceSnapshot> {
    return parseDaemonToolOutput(
      await this.post<DaemonToolResult>(
        `/native-services/${encodeURIComponent(id)}/disable`,
        input,
        { signal: options?.signal },
      ),
    )
  }

  async uninstallNativeService(
    id: string,
    input: DaemonNativeServiceControlInput = {},
    options?: DaemonRequestControlOptions,
  ): Promise<DaemonNativeServiceUninstallResult> {
    return parseDaemonToolOutput(
      await this.post<DaemonToolResult>(
        `/native-services/${encodeURIComponent(id)}/uninstall`,
        input,
        { signal: options?.signal },
      ),
    )
  }

  watchHealthStream(options?: { signal?: AbortSignal }): Promise<Response> {
    return this.fetch(apiPath('/health/watch'), {
      method: 'GET',
      signal: options?.signal,
    })
  }

  async healthReport(format?: 'markdown'): Promise<string>
  async healthReport(format: 'json'): Promise<DaemonHealthReportSnapshot>
  async healthReport(
    format: 'markdown' | 'json' = 'markdown',
  ): Promise<string | DaemonHealthReportSnapshot> {
    if (format === 'json') {
      return this.get('/health/export?format=json')
    }

    const response = await this.fetch(apiPath('/health/export?format=markdown'), {
      method: 'GET',
    })
    if (!response.ok) {
      throw new Error(`${response.status}: ${await response.text()}`)
    }
    return response.text()
  }

  async chat(
    message: string,
    sessionId?: string,
    options?: ChatOptions,
    request?: DaemonChatRequestControlOptions,
  ): Promise<DaemonChatResult> {
    return this.post('/chat', {
      message,
      sessionId,
      ...sanitizeChatOptions(options),
    }, request)
  }

  async startBackgroundChat(
    message: string,
    sessionId?: string,
    options?: ChatOptions,
  ): Promise<DaemonChatBackgroundStartResult> {
    return this.post('/chat/background', {
      message,
      sessionId,
      ...sanitizeChatOptions(options),
    })
  }

  backgroundChatStatus(jobId: string): Promise<DaemonChatBackgroundStatusResult> {
    return this.get(`/chat/background/${encodeURIComponent(jobId)}`)
  }

  cancelBackgroundChat(jobId: string): Promise<DaemonChatBackgroundStatusResult> {
    return this.request<ApiEnvelope<DaemonChatBackgroundStatusResult>>(
      apiPath(`/chat/background/${encodeURIComponent(jobId)}`),
      { method: 'DELETE' },
    ).then((envelope) => envelope.data)
  }

  backgroundChatJobs(): Promise<DaemonChatBackgroundListResult> {
    return this.get('/chat/background')
  }

  chatStream(
    message: string,
    sessionId?: string,
    options?: ChatStreamOptions,
    request?: DaemonRequestControlOptions,
  ): Promise<Response> {
    const lastEventId = options?.lastEventId?.trim()
    const sanitizedOptions = sanitizeChatOptions(options)
    if (lastEventId) {
      sanitizedOptions!.lastEventId = lastEventId
    } else if (sanitizedOptions) {
      delete sanitizedOptions.lastEventId
    }
    return this.fetch(apiPath('/chat/stream'), {
      method: 'POST',
      signal: request?.signal,
      headers: lastEventId ? { 'Last-Event-ID': lastEventId } : undefined,
      body: {
        message,
        sessionId,
        ...sanitizedOptions,
      },
    })
  }

  cancelActiveRun(sessionId: string): Promise<{ cancelled: boolean }> {
    return this.post(`/runs/${encodeURIComponent(sessionId)}/cancel`, {})
  }

  async sessions(query?: string, options?: DaemonListSessionsOptions): Promise<DaemonSessionList> {
    const searchParams = new URLSearchParams()
    if (query) {
      searchParams.set('query', query)
    }
    if (options?.page) {
      searchParams.set('page', String(options.page))
    }
    if (options?.perPage) {
      searchParams.set('perPage', String(options.perPage))
    }
    if (options?.workspaceRoot) {
      searchParams.set('workspaceRoot', options.workspaceRoot)
    }
    if (options?.metrics) {
      searchParams.set('metrics', 'true')
    }
    if (options?.status) {
      searchParams.set('status', options.status)
    }
    const qs = searchParams.toString()
    return this.get(`/sessions${qs ? `?${qs}` : ''}`)
  }

  async sessionManagement(): Promise<DaemonSessionManagementSnapshot> {
    return this.get('/sessions/management')
  }

  async cleanupSessionManagement(
    options: { dryRun?: boolean } = {},
  ): Promise<DaemonSessionManagementCleanupResult> {
    return this.post('/sessions/management/cleanup', {
      dryRun: options.dryRun ?? true,
    })
  }

  watchSessionsStream(
    query?: string,
    options?: DaemonListSessionsOptions & { signal?: AbortSignal },
  ): Promise<Response> {
    const searchParams = new URLSearchParams()
    if (query) {
      searchParams.set('query', query)
    }
    if (options?.page) {
      searchParams.set('page', String(options.page))
    }
    if (options?.perPage) {
      searchParams.set('perPage', String(options.perPage))
    }
    if (options?.workspaceRoot) {
      searchParams.set('workspaceRoot', options.workspaceRoot)
    }
    const qs = searchParams.toString()
    return this.fetch(apiPath(`/sessions/watch${qs ? `?${qs}` : ''}`), {
      method: 'GET',
      signal: options?.signal,
    })
  }

  async session(id: string): Promise<DaemonSessionDetail> {
    return this.get(`/sessions/${encodeURIComponent(id)}`)
  }

  /**
   * Queue a mid-run steering note (an instruction or question) onto a live,
   * in-flight run. 202s with the queued note id; daemon 409s if the session
   * has no active run (surfaced as a request error by the shared transport).
   */
  async steerSession(
    sessionId: string,
    message: string,
    kind?: 'instruction' | 'question',
  ): Promise<{ noteId: string; pendingSteeringNoteCount?: number }> {
    const { noteId, pendingSteeringNoteCount } = await this.post<{
      noteId: string
      queued: boolean
      pendingSteeringNoteCount?: number
    }>(
      `/sessions/${encodeURIComponent(sessionId)}/steer`,
      { message, kind },
    )
    return typeof pendingSteeringNoteCount === 'number'
      ? { noteId, pendingSteeringNoteCount }
      : { noteId }
  }

  /** Retract steering that has not yet been consumed by the active run. */
  async cancelSessionSteering(
    sessionId: string,
    target: { selector: 'latest' | 'all' } | { noteId: string },
  ): Promise<{
      status: 'cancelled'
      cancelledNoteIds: string[]
      pendingSteeringNoteCount: number
    }> {
    return this.post(
      `/sessions/${encodeURIComponent(sessionId)}/steer/cancel`,
      target,
    )
  }

  /** Fetch the agent state board for a run (D4 canonical endpoint). */
  async stateBoard(sessionId: string): Promise<DaemonStateBoardResponse> {
    return this.get(`/runs/${encodeURIComponent(sessionId)}/state`)
  }

  async sessionRunbook(id: string): Promise<DaemonSessionRunbook> {
    return this.get(`/sessions/${encodeURIComponent(id)}/runbook`)
  }

  async sessionExport(
    id: string,
    format?: 'markdown',
    options?: DaemonSessionExportOptions,
  ): Promise<string>
  async sessionExport(
    id: string,
    format: 'json',
    options?: DaemonSessionExportOptions,
  ): Promise<DaemonSessionExportSnapshot>
  async sessionExport(
    id: string,
    format: 'markdown' | 'json' = 'markdown',
    options: DaemonSessionExportOptions = {},
  ): Promise<string | DaemonSessionExportSnapshot> {
    const encodedId = encodeURIComponent(id)
    const searchParams = new URLSearchParams({ format })
    if (typeof options.sanitize === 'boolean') {
      searchParams.set('sanitize', options.sanitize ? '1' : '0')
    }
    const path = `/sessions/${encodedId}/export?${searchParams.toString()}`
    if (format === 'json') {
      return this.get(path)
    }

    const response = await this.fetch(apiPath(path), {
      method: 'GET',
    })
    if (!response.ok) {
      throw new Error(`${response.status}: ${await response.text()}`)
    }
    return response.text()
  }

  async shareSession(
    id: string,
    mode: DaemonSessionShareMode = 'knowledge',
  ): Promise<DaemonSessionShareResult> {
    return this.post(`/sessions/${encodeURIComponent(id)}/share`, { mode })
  }

  async importSharedSession(
    shareUrl: string,
    options?: { title?: string },
  ): Promise<DaemonSessionImportResult> {
    return this.post('/sessions/import', {
      shareUrl,
      ...(options?.title ? { title: options.title } : {}),
    })
  }

  async sessionArtifacts(id: string): Promise<DaemonArtifact[]> {
    return this.get(`/sessions/${encodeURIComponent(id)}/artifacts`)
  }

  async recentArtifacts(limit?: number): Promise<DaemonRecentArtifact[]> {
    const query = limit ? `?limit=${encodeURIComponent(String(limit))}` : ''
    return this.get(`/artifacts${query}`)
  }

  resumeSessionStream(
    id: string,
    options?: { force?: boolean },
    request?: DaemonRequestControlOptions,
  ): Promise<Response> {
    return this.fetch(apiPath(`/sessions/${encodeURIComponent(id)}/resume`), {
      method: 'POST',
      signal: request?.signal,
      body: options ?? {},
    })
  }

  async branchSession(
    id: string,
    options?: { fromEventIndex?: number },
  ): Promise<DaemonSessionBranchResult> {
    return this.post(`/sessions/${encodeURIComponent(id)}/branch`, options ?? {})
  }

  async recordTranscriptTurn(
    id: string,
    input: DaemonRecordTranscriptTurnInput,
  ): Promise<DaemonSessionMeta> {
    const { data } = await this.request<ApiEnvelope<DaemonSessionMeta>>(
      apiPath(`/sessions/${encodeURIComponent(id)}/transcript-turn`),
      {
        method: 'POST',
        body: input,
      },
    )
    return data
  }

  async recordLocalShellTurn(
    id: string,
    input: DaemonRecordLocalShellTurnInput,
  ): Promise<DaemonSessionMeta> {
    const { data } = await this.request<ApiEnvelope<DaemonSessionMeta>>(
      apiPath(`/sessions/${encodeURIComponent(id)}/local-shell-turn`),
      {
        method: 'POST',
        body: input,
      },
    )
    return data
  }

  async setSessionAgent(
    id: string,
    agentId: string,
  ): Promise<{ sessionId: string; primaryAgentId: string }> {
    return this.post(`/sessions/${encodeURIComponent(id)}/agent`, { agentId })
  }

  async answerSessionQuestion(
    id: string,
    questionId: string,
    answer: string,
  ): Promise<{ answered: boolean }> {
    return this.post(
      `/sessions/${encodeURIComponent(id)}/questions/${encodeURIComponent(questionId)}`,
      { answer },
    )
  }

  watchSessionStream(id: string, options?: { signal?: AbortSignal }): Promise<Response> {
    return this.fetch(apiPath(`/sessions/${encodeURIComponent(id)}/watch`), {
      method: 'GET',
      signal: options?.signal,
    })
  }

  async respondApproval(
    requestId: string,
    approved: boolean | ApprovalDecisionStatus,
    options: {
      sessionId?: string
      scope?: 'once' | 'session' | 'always' | 'run' | 'session-all'
      rule?: { tool: string; pattern: string }
      note?: string
      /** With a denial: stop the run immediately instead of granting a read-only follow-up turn. */
      stop?: boolean
    } & DaemonRequestControlOptions = {},
  ): Promise<DaemonApprovalResponseResult> {
    const { sessionId, scope = 'once', rule, note, stop, ...request } = options
    const body =
      typeof approved === 'boolean'
        ? { requestId, approved, sessionId, scope, rule, note, ...(stop ? { stop } : {}) }
        : {
            requestId,
            decision: approved,
            approved: approved === 'approved',
            sessionId,
            scope,
            rule,
            note,
            ...(stop ? { stop } : {}),
          }
    return this.post('/approvals/respond', body, request)
  }

  resumeApprovalStream(
    requestId: string,
    approved: boolean | ApprovalDecisionStatus,
    sessionId?: string,
    note?: string,
    request?: DaemonRequestControlOptions,
  ): Promise<Response> {
    const body =
      typeof approved === 'boolean'
        ? { requestId, approved, sessionId, note }
        : {
            requestId,
            decision: approved,
            approved: approved === 'approved',
            sessionId,
            note,
          }
    return this.fetch(apiPath('/approvals/resume'), {
      method: 'POST',
      signal: request?.signal,
      body,
    })
  }

  async listRememberedApprovals(): Promise<{ decisions: DaemonRememberedApproval[] }> {
    return this.get('/approvals/decisions')
  }

  async describeRememberedApprovalRule(
    input: DaemonRememberedApprovalDescribeInput,
  ): Promise<{ rule: DaemonRememberedApprovalRule }> {
    return this.post('/approvals/decisions/describe', input)
  }

  async upsertRememberedApproval(
    decision: DaemonRememberedApprovalInput,
  ): Promise<{ decision: DaemonRememberedApproval }> {
    return this.post('/approvals/decisions', decision)
  }

  async updateRememberedApproval(
    match: DaemonRememberedApprovalMatch,
    decision: DaemonRememberedApprovalInput,
  ): Promise<{ decision: DaemonRememberedApproval }> {
    return this.patch('/approvals/decisions', { match, decision })
  }

  async deleteRememberedApproval(
    match: DaemonRememberedApprovalMatch,
  ): Promise<{ removed: boolean }> {
    const query = new URLSearchParams()
    query.set('tool', match.tool)
    query.set('pattern', match.pattern)
    query.set('scope', match.scope)
    if (match.sessionId) query.set('sessionId', match.sessionId)
    const { data } = await this.request<ApiEnvelope<{ removed: boolean }>>(
      apiPath(`/approvals/decisions?${query.toString()}`),
      { method: 'DELETE' },
    )
    return data
  }

  async clearRememberedApprovals(
    params: {
      scope?: 'session' | 'always'
      sessionId?: string
      /** When true, sends ?stale=true so the daemon only removes
       * rules matching isStaleRememberedDecision (hitCount=0,
       * createdAt ≥ STALE_RULE_THRESHOLD_DAYS old). */
      stale?: boolean
      /** Exact tool name; daemon removes only entries with matching
       * tool. Pairs with cli `decisions list --tool`. */
      tool?: string
      /** Optional outcome filter. */
      approved?: boolean
    } = {},
  ): Promise<void> {
    const query = new URLSearchParams()
    if (params.scope) query.set('scope', params.scope)
    if (params.sessionId) query.set('sessionId', params.sessionId)
    if (params.stale) query.set('stale', 'true')
    if (params.tool) query.set('tool', params.tool)
    if (params.approved !== undefined) query.set('approved', String(params.approved))
    const qs = query.toString()
    await this.del(`/approvals/decisions${qs ? `?${qs}` : ''}`)
  }

  async deleteSession(id: string): Promise<void> {
    await this.del(`/sessions/${encodeURIComponent(id)}`)
  }

  async updateSession(
    id: string,
    patch: DaemonSessionUpdateInput,
  ): Promise<DaemonSessionMeta> {
    return this.patch(`/sessions/${encodeURIComponent(id)}`, patch)
  }

  async sessionApprovals(id: string): Promise<DaemonPendingApproval[]> {
    return this.get(`/sessions/${encodeURIComponent(id)}/approvals`)
  }

  async compactSession(id: string): Promise<DaemonSessionCompactResult> {
    return this.post(`/sessions/${encodeURIComponent(id)}/compact`)
  }

  async skills(options?: {
    includeDisabled?: boolean
    includeBuiltins?: boolean
    cwd?: string
    workspaceRoot?: string
  }): Promise<DaemonSkill[]> {
    const params = new URLSearchParams()
    if (options?.includeDisabled) params.set('includeDisabled', 'true')
    if (options?.includeBuiltins === false) params.set('includeBuiltins', 'false')
    if (options?.cwd) params.set('cwd', options.cwd)
    if (options?.workspaceRoot) params.set('workspaceRoot', options.workspaceRoot)
    const suffix = params.toString()
    return this.get(`/skills${suffix ? `?${suffix}` : ''}`)
  }

  async skill(
    name: string,
    options?: { cwd?: string; workspaceRoot?: string },
  ): Promise<DaemonSkillDetail> {
    const params = new URLSearchParams()
    if (options?.cwd) params.set('cwd', options.cwd)
    if (options?.workspaceRoot) params.set('workspaceRoot', options.workspaceRoot)
    const suffix = params.toString()
    return this.get(`/skills/${encodeURIComponent(name)}${suffix ? `?${suffix}` : ''}`)
  }

  async setSkillEnabled(id: string, enabled: boolean): Promise<{ id: string; enabled: boolean }> {
    return this.post(`/skills/${encodeURIComponent(id)}/${enabled ? 'enable' : 'disable'}`, {})
  }

  async searchSkills(
    query: string,
    options?: { includeBuiltins?: boolean; cwd?: string; workspaceRoot?: string },
  ): Promise<DaemonSkill[]> {
    const params = new URLSearchParams({ query })
    if (options?.includeBuiltins === false) params.set('includeBuiltins', 'false')
    if (options?.cwd) params.set('cwd', options.cwd)
    if (options?.workspaceRoot) params.set('workspaceRoot', options.workspaceRoot)
    return this.get(`/skills/search?${params.toString()}`)
  }

  async searchMarketplaceSkills(
    query: string,
    options: { marketplace?: string; limit?: number } = {},
  ): Promise<MarketplaceSkillSearchResult[]> {
    const params = new URLSearchParams({ query })
    if (options.marketplace) params.set('marketplace', options.marketplace)
    if (options.limit) params.set('limit', String(options.limit))
    return this.get(`/skills/marketplace/search?${params.toString()}`)
  }

  async installSkill(req: {
    source: string
    force?: boolean
    expectedDigest?: string
  }): Promise<{ installed: DaemonSkill[] }> {
    return this.post('/skills/install', req)
  }

  async previewSkillInstall(req: {
    source: string
  }): Promise<InstallSkillPreviewResponse> {
    return this.post('/skills/install/preview', req)
  }

  async createSkill(req: DaemonSkillCreateInput): Promise<DaemonSkillCreateResult> {
    return this.post('/skills/create', req)
  }

  async uninstallSkill(id: string): Promise<{ removed: boolean }> {
    const { data } = await this.request<ApiEnvelope<{ removed: boolean }>>(
      apiPath(`/skills/${encodeURIComponent(id)}`),
      { method: 'DELETE' },
    )
    return data
  }

  async updateSkill(
    id: string,
  ): Promise<{ changed: boolean; from?: string; to?: string; reason?: string }> {
    return this.post(`/skills/${encodeURIComponent(id)}/update`, {})
  }

  async listMarketplaces(): Promise<
    Array<{ name: string; url: string; addedAt: string; lastSync: string | null }>
  > {
    return this.get('/marketplaces')
  }

  async addMarketplace(
    name: string,
    url: string,
  ): Promise<{ name: string; url: string; addedAt: string; lastSync: string | null }> {
    return this.post('/marketplaces', { name, url })
  }

  async removeMarketplace(name: string): Promise<{ removed: boolean }> {
    const { data } = await this.request<ApiEnvelope<{ removed: boolean }>>(
      apiPath(`/marketplaces/${encodeURIComponent(name)}`),
      { method: 'DELETE' },
    )
    return data
  }

  async config(): Promise<DaemonConfig> {
    return this.get('/config')
  }

  /** Read-only snapshot of the active tool policy. Includes the rule
   * table the daemon is enforcing right now (defaults, per-tool modes,
   * deny lists, allow lists). Useful for surfaces that want to render the
   * current safety posture without reading the YAML file directly. */
  async policy(): Promise<DaemonPolicySnapshot> {
    return this.get('/policy')
  }

  async updateConfig(updates: DaemonConfigUpdateInput): Promise<DaemonConfigUpdateResult> {
    return this.put('/config', updates)
  }

  async updateConfigEnv(input: DaemonConfigEnvUpdateInput): Promise<DaemonConfigEnvUpdateResult> {
    return this.put('/config/env', input)
  }

  async replaceMcpServers(servers: DaemonMcpServerConfig[]): Promise<DaemonConfigUpdateResult> {
    return this.updateConfig({ 'mcp.servers': servers })
  }

  async updateMcpClient(client: DaemonMcpClientConfig): Promise<DaemonConfigUpdateResult> {
    return this.updateConfig({ 'mcp.client': client })
  }

  async upsertMcpServer(server: DaemonMcpServerConfig): Promise<DaemonConfigUpdateResult> {
    return this.post('/config/mcp/servers', server)
  }

  async deleteMcpServer(name: string): Promise<DaemonConfigUpdateResult> {
    const { data } = await this.request<ApiEnvelope<DaemonConfigUpdateResult>>(
      apiPath(`/config/mcp/servers/${encodeURIComponent(name)}`),
      { method: 'DELETE' },
    )
    return data
  }

  async setMcpServerEnabled(name: string, enabled: boolean): Promise<DaemonConfigUpdateResult> {
    return this.post(`/config/mcp/servers/${encodeURIComponent(name)}/enable`, {
      enabled,
    })
  }

  async trustMcpServerManifest(
    name: string,
  ): Promise<{ name: string; digest: string; trusted: true }> {
    return this.post(`/config/mcp/servers/${encodeURIComponent(name)}/trust-manifest`, {})
  }

  async replaceOutboundWebhooks(
    webhooks: DaemonOutboundWebhookConfig[],
  ): Promise<DaemonConfigUpdateResult> {
    return this.updateConfig({ 'hooks.outboundWebhooks': webhooks })
  }

  async outboundWebhooks(): Promise<DaemonOutboundWebhookSummary[]> {
    return this.get('/config/hooks/outbound-webhooks')
  }

  async outboundWebhookDeliveries(
    options?: DaemonOutboundWebhookDeliveryOptions,
  ): Promise<DaemonOutboundWebhookDelivery[]> {
    return (await this.outboundWebhookDeliveriesPage(options)).data
  }

  async outboundWebhookDeliveriesPage(
    options?: DaemonOutboundWebhookDeliveryOptions,
  ): Promise<DaemonListPage<DaemonOutboundWebhookDelivery>> {
    const searchParams = new URLSearchParams()
    if (options?.id) {
      searchParams.set('id', options.id)
    }
    if (options?.status) {
      searchParams.set('status', options.status)
    }
    if (options?.cursor) {
      searchParams.set('cursor', options.cursor)
    }
    if (typeof options?.limit === 'number') {
      searchParams.set('limit', String(options.limit))
    }
    const suffix = searchParams.size > 0 ? `?${searchParams.toString()}` : ''
    return this.request<DaemonListPage<DaemonOutboundWebhookDelivery>>(
      apiPath(`/config/hooks/outbound-webhooks/deliveries${suffix}`),
      { method: 'GET' },
    )
  }

  async outboundWebhookDeadLetters(
    options?: DaemonOutboundWebhookDeadLetterOptions,
  ): Promise<DaemonOutboundWebhookDeadLetter[]> {
    return (await this.outboundWebhookDeadLettersPage(options)).data
  }

  async outboundWebhookDeadLettersPage(
    options?: DaemonOutboundWebhookDeadLetterOptions,
  ): Promise<DaemonListPage<DaemonOutboundWebhookDeadLetter>> {
    const searchParams = new URLSearchParams()
    if (options?.id) {
      searchParams.set('id', options.id)
    }
    if (options?.state) {
      searchParams.set('state', options.state)
    }
    if (options?.cursor) {
      searchParams.set('cursor', options.cursor)
    }
    if (typeof options?.limit === 'number') {
      searchParams.set('limit', String(options.limit))
    }
    const suffix = searchParams.size > 0 ? `?${searchParams.toString()}` : ''
    return this.request<DaemonListPage<DaemonOutboundWebhookDeadLetter>>(
      apiPath(`/config/hooks/outbound-webhooks/dead-letters${suffix}`),
      { method: 'GET' },
    )
  }

  async acknowledgeOutboundWebhookDeadLetter(
    rootDeliveryId: string,
    options?: DaemonAcknowledgeOutboundWebhookDeadLetterOptions,
  ): Promise<DaemonOutboundWebhookDeadLetter> {
    return this.post(
      `/config/hooks/outbound-webhooks/dead-letters/${encodeURIComponent(rootDeliveryId)}/ack`,
      options?.note ? { note: options.note } : undefined,
    )
  }

  async replayOutboundWebhookDelivery(
    deliveryId: string,
    options?: { force?: boolean },
  ): Promise<DaemonOutboundWebhookDelivery> {
    return this.post(
      `/config/hooks/outbound-webhooks/deliveries/${encodeURIComponent(deliveryId)}/replay`,
      options?.force ? { force: true } : undefined,
    )
  }

  async replayOutboundWebhookDeadLetters(
    options?: DaemonReplayOutboundWebhookDeadLettersOptions,
  ): Promise<DaemonOutboundWebhookDelivery[]> {
    const deadLetters = await this.outboundWebhookDeadLetters(options)
    const replays: DaemonOutboundWebhookDelivery[] = []
    for (const deadLetter of deadLetters) {
      replays.push(
        await this.replayOutboundWebhookDelivery(
          deadLetter.latestDeliveryId,
          options?.force ? { force: true } : undefined,
        ),
      )
    }
    return replays
  }

  async upsertOutboundWebhook(
    webhook: DaemonOutboundWebhookConfig,
  ): Promise<DaemonConfigUpdateResult> {
    return this.post('/config/hooks/outbound-webhooks', webhook)
  }

  async deleteOutboundWebhook(id: string): Promise<DaemonConfigUpdateResult> {
    const { data } = await this.request<ApiEnvelope<DaemonConfigUpdateResult>>(
      apiPath(`/config/hooks/outbound-webhooks/${encodeURIComponent(id)}`),
      { method: 'DELETE' },
    )
    return data
  }

  async setOutboundWebhookEnabled(id: string, enabled: boolean): Promise<DaemonConfigUpdateResult> {
    return this.post(`/config/hooks/outbound-webhooks/${encodeURIComponent(id)}/enable`, {
      enabled,
    })
  }

  async webhookEndpoints(): Promise<DaemonWebhookEndpointSummary[]> {
    return this.get('/config/channels/webhook/endpoints')
  }

  async upsertWebhookEndpoint(
    endpoint: DaemonWebhookEndpointConfig,
  ): Promise<DaemonConfigUpdateResult> {
    return this.post('/config/channels/webhook/endpoints', endpoint)
  }

  async deleteWebhookEndpoint(id: string): Promise<DaemonConfigUpdateResult> {
    const { data } = await this.request<ApiEnvelope<DaemonConfigUpdateResult>>(
      apiPath(`/config/channels/webhook/endpoints/${encodeURIComponent(id)}`),
      { method: 'DELETE' },
    )
    return data
  }

  async setWebhookEndpointEnabled(id: string, enabled: boolean): Promise<DaemonConfigUpdateResult> {
    return this.post(`/config/channels/webhook/endpoints/${encodeURIComponent(id)}/enable`, {
      enabled,
    })
  }

  async telegramChannel(): Promise<DaemonTelegramChannelSummary | null> {
    return this.get('/config/channels/telegram')
  }

  async upsertTelegramChannel(
    channel: DaemonTelegramChannelConfig,
  ): Promise<DaemonConfigUpdateResult> {
    return this.post('/config/channels/telegram', channel)
  }

  async deleteTelegramChannel(): Promise<DaemonConfigUpdateResult> {
    const { data } = await this.request<ApiEnvelope<DaemonConfigUpdateResult>>(
      apiPath('/config/channels/telegram'),
      { method: 'DELETE' },
    )
    return data
  }

  async setTelegramChannelEnabled(enabled: boolean): Promise<DaemonConfigUpdateResult> {
    return this.post('/config/channels/telegram/enable', { enabled })
  }

  async telegramPairingCode(): Promise<DaemonTelegramPairingCode> {
    return this.post('/config/channels/telegram/pairing-code')
  }

  async revokeTelegramAllowedUser(userId: string): Promise<DaemonConfigUpdateResult> {
    const { data } = await this.request<ApiEnvelope<DaemonConfigUpdateResult>>(
      apiPath(`/config/channels/telegram/allowed-users/${encodeURIComponent(userId)}`),
      { method: 'DELETE' },
    )
    return data
  }

  async discordChannel(): Promise<DaemonDiscordChannelSummary | null> {
    return this.get('/config/channels/discord')
  }

  async upsertDiscordChannel(
    channel: DaemonDiscordChannelConfig,
  ): Promise<DaemonConfigUpdateResult> {
    return this.post('/config/channels/discord', channel)
  }

  async deleteDiscordChannel(): Promise<DaemonConfigUpdateResult> {
    const { data } = await this.request<ApiEnvelope<DaemonConfigUpdateResult>>(
      apiPath('/config/channels/discord'),
      { method: 'DELETE' },
    )
    return data
  }

  async setDiscordChannelEnabled(enabled: boolean): Promise<DaemonConfigUpdateResult> {
    return this.post('/config/channels/discord/enable', { enabled })
  }

  async mattermostChannel(): Promise<DaemonMattermostChannelSummary | null> {
    return this.get('/config/channels/mattermost')
  }

  async upsertMattermostChannel(
    channel: DaemonMattermostChannelConfig,
  ): Promise<DaemonConfigUpdateResult> {
    return this.post('/config/channels/mattermost', channel)
  }

  async deleteMattermostChannel(): Promise<DaemonConfigUpdateResult> {
    const { data } = await this.request<ApiEnvelope<DaemonConfigUpdateResult>>(
      apiPath('/config/channels/mattermost'),
      { method: 'DELETE' },
    )
    return data
  }

  async setMattermostChannelEnabled(enabled: boolean): Promise<DaemonConfigUpdateResult> {
    return this.post('/config/channels/mattermost/enable', { enabled })
  }

  async slackChannel(): Promise<DaemonSlackChannelSummary | null> {
    return this.get('/config/channels/slack')
  }

  async upsertSlackChannel(
    channel: DaemonSlackChannelConfig,
  ): Promise<DaemonConfigUpdateResult> {
    return this.post('/config/channels/slack', channel)
  }

  async deleteSlackChannel(): Promise<DaemonConfigUpdateResult> {
    const { data } = await this.request<ApiEnvelope<DaemonConfigUpdateResult>>(
      apiPath('/config/channels/slack'),
      { method: 'DELETE' },
    )
    return data
  }

  async setSlackChannelEnabled(enabled: boolean): Promise<DaemonConfigUpdateResult> {
    return this.post('/config/channels/slack/enable', { enabled })
  }

  async teamDocs(): Promise<DaemonTeamDocsConfig[]> {
    return this.getCapability('/team-docs')
  }

  watchTeamDocsStream(options?: { signal?: AbortSignal }): Promise<Response> {
    return this.fetch(capabilityPath('/team-docs/watch'), {
      method: 'GET',
      signal: options?.signal,
    })
  }

  async upsertTeamDocsConfig(input: DaemonTeamDocsConfigInput): Promise<DaemonTeamDocsConfig> {
    return this.postCapability('/team-docs', input)
  }

  async deleteTeamDocsConfig(id: string): Promise<void> {
    await this.delCapability(`/team-docs/${encodeURIComponent(id)}`)
  }

  async teamDocsDocuments(id: string): Promise<DaemonTeamDocsDocument[]> {
    return this.getCapability(`/team-docs/${encodeURIComponent(id)}/documents`)
  }

  async teamDocsDocument(id: string, path: string): Promise<DaemonTeamDocsDocumentContent> {
    return this.getCapability(
      `/team-docs/${encodeURIComponent(id)}/document?path=${encodeURIComponent(path)}`,
    )
  }

  async testTeamDocsConnection(id: string): Promise<DaemonTeamDocsActionResult> {
    return this.postCapability(`/team-docs/${encodeURIComponent(id)}/test-connection`, {})
  }

  async syncTeamDocsConfig(id: string): Promise<DaemonTeamDocsActionResult> {
    return this.postCapability(`/team-docs/${encodeURIComponent(id)}/sync`, {})
  }

  async syncAllTeamDocs(): Promise<DaemonTeamDocsSyncAllResult> {
    return this.postCapability('/team-docs/sync-all', {})
  }

  async personalDocs(): Promise<DaemonPersonalDoc[]> {
    return this.getCapability('/personal-docs')
  }

  async personalDoc(id: string): Promise<DaemonPersonalDocContent> {
    return this.getCapability(`/personal-docs/${encodeURIComponent(id)}`)
  }

  async upsertPersonalDoc(input: DaemonPersonalDocInput): Promise<DaemonPersonalDoc> {
    return this.postCapability('/personal-docs', input)
  }

  async deletePersonalDoc(id: string): Promise<void> {
    await this.delCapability(`/personal-docs/${encodeURIComponent(id)}`)
  }

  async promptTemplates(): Promise<DaemonPromptTemplate[]> {
    return this.getCapability('/prompt-templates')
  }

  async upsertPromptTemplate(input: DaemonPromptTemplateInput): Promise<DaemonPromptTemplate> {
    return this.postCapability('/prompt-templates', input)
  }

  async deletePromptTemplate(id: string): Promise<void> {
    await this.delCapability(`/prompt-templates/${encodeURIComponent(id)}`)
  }

  async backups(): Promise<DaemonBackupItem[]> {
    return this.getCapability('/backup')
  }

  async createBackup(): Promise<DaemonBackupItem> {
    return this.postCapability('/backup', {})
  }

  async deleteBackup(id: string): Promise<void> {
    await this.delCapability(`/backup/${encodeURIComponent(id)}`)
  }

  async quickInputSettings(): Promise<DaemonQuickInputSettings> {
    return this.getCapability('/quick-input/settings')
  }

  watchQuickInputSettingsStream(options?: { signal?: AbortSignal }): Promise<Response> {
    return this.fetch(capabilityPath('/quick-input/settings/watch'), {
      method: 'GET',
      signal: options?.signal,
    })
  }

  async updateQuickInputSettings(
    input: DaemonQuickInputSettings,
  ): Promise<DaemonQuickInputSettings> {
    return this.putCapability('/quick-input/settings', input)
  }

  async publishQuickInput(text: string): Promise<DaemonQuickInputPublishResult> {
    return this.postCapability('/quick-input/publish', { text })
  }

  async settingsJson(): Promise<DaemonSettingsJsonDocument> {
    return this.getCapability('/settings/json')
  }

  watchSettingsJsonStream(options?: { signal?: AbortSignal }): Promise<Response> {
    return this.fetch(capabilityPath('/settings/json/watch'), {
      method: 'GET',
      signal: options?.signal,
    })
  }

  async updateSettingsJson(input: DaemonSettingsJsonDocument): Promise<DaemonSettingsJsonDocument> {
    return this.putCapability('/settings/json', input)
  }

  async ragFolders(): Promise<DaemonRagFolder[]> {
    return this.getCapability('/rag/folders')
  }

  async upsertRagFolder(input: DaemonRagFolderInput): Promise<DaemonRagFolder> {
    return this.postCapability('/rag/folders', input)
  }

  async deleteRagFolder(id: string): Promise<void> {
    await this.delCapability(`/rag/folders/${encodeURIComponent(id)}`)
  }

  async ragDocuments(folderId: string): Promise<DaemonRagDocument[]> {
    return this.getCapability(`/rag/documents?folder=${encodeURIComponent(folderId)}`)
  }

  async ragDocument(id: string): Promise<DaemonRagDocumentContent> {
    return this.getCapability(`/rag/documents/${encodeURIComponent(id)}`)
  }

  async upsertRagDocument(input: DaemonRagDocumentInput): Promise<DaemonRagDocument> {
    return this.postCapability('/rag/documents', input)
  }

  async deleteRagDocument(id: string): Promise<void> {
    await this.delCapability(`/rag/documents/${encodeURIComponent(id)}`)
  }

  async searchRag(query: string, limit?: number): Promise<DaemonRagSearchHit[]> {
    const params = new URLSearchParams({ q: query })
    if (typeof limit === 'number') {
      params.set('limit', String(limit))
    }
    return this.getCapability(`/rag/search?${params.toString()}`)
  }

  async syncRag(): Promise<DaemonRagSyncResult> {
    return this.postCapability('/rag/sync', {})
  }

  async ragVectorDbInfo(): Promise<DaemonRagVectorDbInfo> {
    return this.getCapability('/rag/vector-db')
  }

  async testRagVectorDbConnection(): Promise<DaemonRagConnectionTestResult> {
    return this.postCapability('/rag/vector-db/test', {})
  }

  async testRagRerankConnection(): Promise<DaemonRagConnectionTestResult> {
    return this.postCapability('/rag/rerank/test', {})
  }

  async networkConfig(): Promise<DaemonNetworkConfig> {
    return this.getCapability('/network')
  }

  async networkStatus(): Promise<DaemonNetworkStatus> {
    return this.getCapability('/network/status')
  }

  watchNetworkConfigStream(options?: { signal?: AbortSignal }): Promise<Response> {
    return this.fetch(capabilityPath('/network/watch'), {
      method: 'GET',
      signal: options?.signal,
    })
  }

  async updateNetworkConfig(input: DaemonNetworkConfig): Promise<DaemonNetworkConfig> {
    return this.putCapability('/network', input)
  }

  async probeNetwork(url: string): Promise<DaemonNetworkProbeResult> {
    return this.postCapability('/network/probe', { url })
  }

  async startGitHubOAuth(): Promise<DaemonGitHubOAuthStartResult> {
    return this.postCapability('/github/oauth/start', {})
  }

  async githubOAuthStatus(): Promise<DaemonGitHubOAuthStatus> {
    return this.getCapability('/github/oauth/status')
  }

  async githubSyncRepos(): Promise<DaemonGitHubSyncRepo[]> {
    return this.getCapability('/github/sync/repos')
  }

  async updateGitHubSyncRepos(repos: DaemonGitHubSyncRepo[]): Promise<DaemonGitHubSyncRepo[]> {
    return this.putCapability('/github/sync/repos', repos)
  }

  async githubSyncPolicy(): Promise<DaemonGitHubSyncPolicy> {
    return this.getCapability('/github/sync/policy')
  }

  async updateGitHubSyncPolicy(policy: DaemonGitHubSyncPolicy): Promise<DaemonGitHubSyncPolicy> {
    return this.putCapability('/github/sync/policy', policy)
  }

  async githubChatSyncConfig(): Promise<DaemonGitHubChatSyncConfig> {
    return this.getCapability('/github/chat-sync')
  }

  async updateGitHubChatSyncConfig(
    input: DaemonGitHubChatSyncConfigUpdate,
  ): Promise<DaemonGitHubChatSyncConfig> {
    return this.putCapability('/github/chat-sync', input)
  }

  async runGitHubChatSync(): Promise<DaemonGitHubChatSyncResult> {
    return this.postCapability('/github/chat-sync/run', {})
  }

  async messageSubscriptionOverview(): Promise<DaemonMessageSubscriptionOverview> {
    return this.getCapability('/message-subscription')
  }

  async updateMessageSubscriptionConfig(
    input: DaemonMessageSubscriptionConfig,
  ): Promise<DaemonMessageSubscriptionOverview> {
    return this.putCapability('/message-subscription', input)
  }

  async startMessageSubscription(): Promise<DaemonMessageSubscriptionOverview> {
    return this.postCapability('/message-subscription/start', {})
  }

  async stopMessageSubscription(): Promise<DaemonMessageSubscriptionOverview> {
    return this.postCapability('/message-subscription/stop', {})
  }

  async refreshMessageSubscription(): Promise<DaemonMessageSubscriptionRefreshResult> {
    return this.postCapability('/message-subscription/refresh', {})
  }

  async processPendingMessageSubscription(): Promise<DaemonMessageSubscriptionProcessResult> {
    return this.postCapability('/message-subscription/process', {})
  }

  watchMessageSubscriptionStream(options?: { signal?: AbortSignal }): Promise<Response> {
    return this.fetch(capabilityPath('/message-subscription/watch'), {
      method: 'GET',
      signal: options?.signal,
    })
  }

  async messageSubscriptionMessages(
    status?: DaemonMessageSubscriptionItem['status'],
    limit?: number,
  ): Promise<DaemonMessageSubscriptionItem[]> {
    const params = new URLSearchParams()
    if (status) params.set('status', status)
    if (typeof limit === 'number') params.set('limit', String(limit))
    const suffix = params.size > 0 ? `?${params.toString()}` : ''
    return this.getCapability(`/message-subscription/messages${suffix}`)
  }

  async reprocessMessageSubscriptionMessage(
    hash: string,
  ): Promise<DaemonMessageSubscriptionItem | null> {
    return this.postCapability(
      `/message-subscription/messages/${encodeURIComponent(hash)}/reprocess`,
      {},
    )
  }

  async deleteMessageSubscriptionMessage(hash: string): Promise<void> {
    await this.delCapability(`/message-subscription/messages/${encodeURIComponent(hash)}`)
  }

  async notifications(options?: {
    unread?: boolean
    limit?: number
  }): Promise<DaemonNotificationItem[]> {
    const params = new URLSearchParams()
    if (options?.unread !== undefined) params.set('unread', String(options.unread))
    if (options?.limit !== undefined) params.set('limit', String(options.limit))
    const suffix = params.size > 0 ? `?${params.toString()}` : ''
    return this.getCapability(`/notifications${suffix}`)
  }

  async notification(id: string): Promise<DaemonNotificationItem> {
    return this.getCapability(`/notifications/${encodeURIComponent(id)}`)
  }

  async notificationInventory(): Promise<{ total: number; unread: number }> {
    return this.getCapability('/notifications/inventory')
  }

  watchNotificationsStream(options?: { signal?: AbortSignal }): Promise<Response> {
    return this.fetch(capabilityPath('/notifications/watch'), {
      method: 'GET',
      signal: options?.signal,
    })
  }

  async publishNotification(input: DaemonNotificationDraft): Promise<DaemonNotificationItem> {
    return this.postCapability('/notifications', {
      id: input.id,
      title: input.title,
      body: input.body ?? '',
      url: input.url ?? null,
      topic: input.topic ?? null,
      audience: input.audience ?? null,
    })
  }

  async markNotificationRead(id: string): Promise<void> {
    await this.postCapability(`/notifications/${encodeURIComponent(id)}/read`, {})
  }

  async markAllNotificationsRead(): Promise<number> {
    const result = await this.postCapability<{ marked: number }>('/notifications/read-all', {})
    return result.marked
  }

  async notificationSettings(): Promise<DaemonNotificationSettings> {
    return this.getCapability('/notifications/settings')
  }

  async updateNotificationSettings(
    input: DaemonNotificationSettings,
  ): Promise<DaemonNotificationSettings> {
    return this.putCapability('/notifications/settings', input)
  }

  async schedulerJobs(): Promise<DaemonSchedulerJob[]> {
    return this.getCapability('/scheduler/jobs')
  }

  async schedulerJobRuns(id: string, limit?: number): Promise<DaemonSchedulerJobRun[]> {
    const query = limit ? `?limit=${encodeURIComponent(String(limit))}` : ''
    return this.getCapability(`/scheduler/jobs/${encodeURIComponent(id)}/runs${query}`)
  }

  async schedulerJobRun(id: string, runId: string): Promise<DaemonSchedulerJobRun> {
    return this.getCapability(
      `/scheduler/jobs/${encodeURIComponent(id)}/runs/${encodeURIComponent(runId)}`,
    )
  }

  async schedulerJobNotificationSubscriptions(
    id: string,
  ): Promise<DaemonSchedulerNotificationSubscriptions> {
    return this.getCapability(`/scheduler/jobs/${encodeURIComponent(id)}/notifications`)
  }

  // --- /scheduled-tasks REST surface (natural-language `when`, retry/timezone) ---

  async createScheduledTask(input: DaemonScheduledTaskInput): Promise<DaemonSchedulerJob> {
    return this.post('/scheduled-tasks', input)
  }

  async listScheduledTasks(opts?: {
    all?: boolean
    channel?: string
  }): Promise<DaemonSchedulerJob[]> {
    const params = new URLSearchParams()
    if (opts?.all) params.set('status', 'all')
    if (opts?.channel) params.set('channel', opts.channel)
    const query = params.toString()
    return this.get(`/scheduled-tasks${query ? `?${query}` : ''}`)
  }

  async scheduledTaskDeliveryStatus(): Promise<DaemonSchedulerDeliveryOutboxSummary> {
    return this.get('/scheduled-tasks/delivery-status')
  }

  async getScheduledTask(id: string): Promise<DaemonSchedulerJob> {
    return this.get(`/scheduled-tasks/${encodeURIComponent(id)}`)
  }

  async updateScheduledTask(
    id: string,
    input: DaemonScheduledTaskUpdateInput,
  ): Promise<DaemonSchedulerJob> {
    return this.patch(`/scheduled-tasks/${encodeURIComponent(id)}`, input)
  }

  async listScheduledTaskRuns(id: string, limit?: number): Promise<DaemonSchedulerJobRun[]> {
    const query = limit ? `?limit=${encodeURIComponent(String(limit))}` : ''
    return this.get(`/scheduled-tasks/${encodeURIComponent(id)}/runs${query}`)
  }

  async getScheduledTaskRun(id: string, runId: string): Promise<DaemonSchedulerJobRun> {
    return this.get(
      `/scheduled-tasks/${encodeURIComponent(id)}/runs/${encodeURIComponent(runId)}`,
    )
  }

  async scheduledTaskNotificationSubscriptions(
    id: string,
  ): Promise<DaemonSchedulerNotificationSubscriptions> {
    return this.get(`/scheduled-tasks/${encodeURIComponent(id)}/notifications`)
  }

  async updateScheduledTaskNotificationSubscriptions(
    id: string,
    subscribers: string[] | null,
  ): Promise<DaemonSchedulerNotificationSubscriptions> {
    return this.put(`/scheduled-tasks/${encodeURIComponent(id)}/notifications`, { subscribers })
  }

  async setScheduledTaskNotificationSubscription(
    id: string,
    input: { surface?: string; subscribed: boolean },
  ): Promise<DaemonSchedulerNotificationSubscriptions> {
    return this.post(`/scheduled-tasks/${encodeURIComponent(id)}/notifications/subscription`, input)
  }

  async deleteScheduledTask(id: string): Promise<void> {
    await this.request<void>(apiPath(`/scheduled-tasks/${encodeURIComponent(id)}`), {
      method: 'DELETE',
    })
  }

  async runScheduledTask(
    id: string,
    options: DaemonSchedulerManualRunOptions = {},
  ): Promise<DaemonSchedulerManualRunResult> {
    return this.post(`/scheduled-tasks/${encodeURIComponent(id)}/run`, {
      ...(options.suppressDelivery ? { suppressDelivery: true } : {}),
      ...(options.waitForCompletion === false ? { waitForCompletion: false } : {}),
    })
  }

  async pauseScheduledTask(id: string): Promise<DaemonSchedulerJob> {
    return this.post(`/scheduled-tasks/${encodeURIComponent(id)}/pause`, {})
  }

  async resumeScheduledTask(id: string): Promise<DaemonSchedulerJob> {
    return this.post(`/scheduled-tasks/${encodeURIComponent(id)}/resume`, {})
  }

  watchSchedulerJobsStream(options?: { signal?: AbortSignal }): Promise<Response> {
    return this.fetch(capabilityPath('/scheduler/jobs/watch'), {
      method: 'GET',
      signal: options?.signal,
    })
  }

  async upsertSchedulerJob(input: DaemonSchedulerJobInput): Promise<DaemonSchedulerJob> {
    return this.postCapability('/scheduler/jobs', input)
  }

  async updateSchedulerJobNotificationSubscriptions(
    id: string,
    subscribers: string[] | null,
  ): Promise<DaemonSchedulerNotificationSubscriptions> {
    return this.putCapability(`/scheduler/jobs/${encodeURIComponent(id)}/notifications`, {
      subscribers,
    })
  }

  async setSchedulerJobNotificationSubscription(
    id: string,
    input: { surface?: string; subscribed: boolean },
  ): Promise<DaemonSchedulerNotificationSubscriptions> {
    return this.postCapability(
      `/scheduler/jobs/${encodeURIComponent(id)}/notifications/subscription`,
      input,
    )
  }

  async runSchedulerJob(
    id: string,
    options: DaemonSchedulerManualRunOptions = {},
  ): Promise<DaemonSchedulerManualRunResult> {
    return this.postCapability(`/scheduler/jobs/${encodeURIComponent(id)}/run`, {
      ...(options.suppressDelivery ? { suppressDelivery: true } : {}),
      ...(options.waitForCompletion === false ? { waitForCompletion: false } : {}),
    })
  }

  async pauseSchedulerJob(id: string): Promise<DaemonSchedulerJob> {
    return this.postCapability(`/scheduler/jobs/${encodeURIComponent(id)}/pause`, {})
  }

  async resumeSchedulerJob(id: string): Promise<DaemonSchedulerJob> {
    return this.postCapability(`/scheduler/jobs/${encodeURIComponent(id)}/resume`, {})
  }

  /**
   * Let a job run without a human present: its tool calls stop waiting on
   * approval prompts that nobody would answer while it fires.
   */
  async setSchedulerJobUnattended(id: string, unattended: boolean): Promise<DaemonSchedulerJob> {
    return this.postCapability(`/scheduler/jobs/${encodeURIComponent(id)}/unattended`, { unattended })
  }

  async deleteSchedulerJob(id: string): Promise<void> {
    await this.delCapability(`/scheduler/jobs/${encodeURIComponent(id)}`)
  }

  async imageGenProviders(): Promise<DaemonImageGenProvider[]> {
    return this.getCapability('/image-gen/providers')
  }

  async imageGenJobs(limit = 100): Promise<DaemonImageGenJob[]> {
    const safeLimit = Math.max(1, Math.min(500, Math.floor(limit)))
    return this.getCapability(`/image-gen/jobs?limit=${safeLimit}`)
  }

  async imageGenJob(id: string): Promise<DaemonImageGenJob> {
    return this.getCapability(`/image-gen/jobs/${encodeURIComponent(id)}`)
  }

  async createImageGenJob(input: DaemonImageGenJobInput): Promise<DaemonImageGenJob> {
    return this.postCapability('/image-gen/jobs', input)
  }

  async cancelImageGenJob(id: string): Promise<{ ok: boolean }> {
    return this.postCapability(`/image-gen/jobs/${encodeURIComponent(id)}/cancel`, {})
  }

  async imageGenFile(id: string): Promise<Blob> {
    const response = await this.fetch(
      capabilityPath(`/image-gen/files/${encodeURIComponent(id)}`),
      { method: 'GET' },
    )
    if (!response.ok) {
      throw new Error(`${response.status}: ${await response.text()}`)
    }
    return response.blob()
  }

  async snippets(query?: string): Promise<DaemonSnippet[]> {
    const suffix = query && query.trim() ? `?q=${encodeURIComponent(query.trim())}` : ''
    return this.getCapability(`/snippets${suffix}`)
  }

  async createSnippet(input: DaemonSnippetInput): Promise<DaemonSnippet> {
    return this.postCapability('/snippets', input)
  }

  async updateSnippet(id: string, input: DaemonSnippetInput): Promise<DaemonSnippet> {
    return this.putCapability(`/snippets/${encodeURIComponent(id)}`, input)
  }

  async deleteSnippet(id: string): Promise<void> {
    return this.delCapability(`/snippets/${encodeURIComponent(id)}`)
  }

  async wikiTree(): Promise<DaemonWikiNode[]> {
    return this.getCapability('/wiki/tree')
  }

  async searchWiki(query: string, limit = 50): Promise<DaemonWikiSearchHit[]> {
    const safeLimit = Math.max(1, Math.min(200, Math.trunc(limit)))
    return this.getCapability(`/wiki/search?q=${encodeURIComponent(query)}&limit=${safeLimit}`)
  }

  async upsertWikiNode(input: DaemonWikiNodeInput): Promise<DaemonWikiNode> {
    return this.postCapability('/wiki/nodes', input)
  }

  async deleteWikiNode(id: string): Promise<void> {
    return this.delCapability(`/wiki/nodes/${encodeURIComponent(id)}`)
  }

  async moveWikiNode(id: string, input: DaemonWikiMoveInput): Promise<DaemonWikiNode> {
    return this.postCapability(`/wiki/nodes/${encodeURIComponent(id)}/move`, input)
  }

  async extensionTokens(): Promise<DaemonExtensionTokenSummary[]> {
    return this.get('/auth/tokens')
  }

  async issueExtensionToken(
    input: DaemonIssueExtensionTokenInput,
  ): Promise<DaemonIssuedExtensionToken> {
    return this.post('/auth/tokens', input)
  }

  async revokeExtensionToken(id: string): Promise<DaemonExtensionTokenSummary> {
    const { data } = await this.request<ApiEnvelope<DaemonExtensionTokenSummary>>(
      apiPath(`/auth/tokens/${encodeURIComponent(id)}`),
      { method: 'DELETE' },
    )
    return data
  }

  async mcpServers(): Promise<DaemonMcpServerStatus[]> {
    return this.get('/mcp/servers')
  }

  async mcpPrompts(server: string): Promise<McpPromptDescriptor[]> {
    return this.get(`/mcp/servers/${encodeURIComponent(server)}/prompts`)
  }

  async getMcpPrompt(
    server: string,
    prompt: string,
    args: Record<string, string> = {},
  ): Promise<{ messages: McpPromptMessage[] }> {
    return this.post(
      `/mcp/servers/${encodeURIComponent(server)}/prompts/${encodeURIComponent(prompt)}`,
      { arguments: args },
    )
  }

  async mcpResources(server: string): Promise<McpResourceDescriptor[]> {
    return this.get(`/mcp/servers/${encodeURIComponent(server)}/resources`)
  }

  async mcpResourceTemplates(server: string): Promise<McpResourceTemplateDescriptor[]> {
    return this.get(`/mcp/servers/${encodeURIComponent(server)}/resources/templates`)
  }

  async mcpReadResource(server: string, uri: string): Promise<McpResourceReadResult> {
    return this.post(`/mcp/servers/${encodeURIComponent(server)}/resources/read`, { uri })
  }

  async mcpSubscribeResource(server: string, uri: string): Promise<McpResourceSubscriptionResult> {
    return this.post(`/mcp/servers/${encodeURIComponent(server)}/resources/subscribe`, { uri })
  }

  async mcpUnsubscribeResource(
    server: string,
    uri: string,
  ): Promise<McpResourceSubscriptionResult> {
    return this.post(`/mcp/servers/${encodeURIComponent(server)}/resources/unsubscribe`, { uri })
  }

  async mcpResourceSubscriptions(server: string): Promise<string[]> {
    return this.get(`/mcp/servers/${encodeURIComponent(server)}/resources/subscriptions`)
  }

  async mcpResourceUpdates(server: string, limit?: number): Promise<McpResourceUpdate[]> {
    const qs = limit ? `?limit=${encodeURIComponent(String(limit))}` : ''
    return this.get(`/mcp/servers/${encodeURIComponent(server)}/resources/updates${qs}`)
  }

  async mcpComplete(server: string, input: McpCompletionInput): Promise<McpCompletionResult> {
    return this.post(`/mcp/servers/${encodeURIComponent(server)}/completion`, input)
  }

  async mcpSetLoggingLevel(
    server: string,
    level: McpLoggingLevel,
  ): Promise<{ level: McpLoggingLevel }> {
    return this.post(`/mcp/servers/${encodeURIComponent(server)}/logging/level`, { level })
  }

  async mcpLogs(server: string): Promise<McpLoggingState> {
    return this.get(`/mcp/servers/${encodeURIComponent(server)}/logs`)
  }

  async mcpMetrics(): Promise<McpMetricsSnapshot> {
    return this.get('/mcp/metrics')
  }

  async mcpMetricsForServer(server: string): Promise<McpMetricsEntry> {
    return this.get(`/mcp/metrics/${encodeURIComponent(server)}`)
  }

  async mcpMarketplaceSearch(query: string): Promise<McpServerTemplate[]> {
    return this.get(`/mcp/marketplace/search?q=${encodeURIComponent(query)}`)
  }

  async mcpInstall(
    name: string,
    marketplace?: string,
    variables?: Record<string, string>,
  ): Promise<McpInstallResult> {
    return this.post('/mcp/install', { name, marketplace, variables })
  }

  async mcpServerTools(serverName: string): Promise<McpServerToolsResult> {
    return this.get(`/mcp/servers/${encodeURIComponent(serverName)}/tools`)
  }

  async mcpDisableTool(serverName: string, tool: string): Promise<{ ok: boolean }> {
    return this.post(
      `/mcp/servers/${encodeURIComponent(serverName)}/tools/${encodeURIComponent(tool)}/disable`,
      {},
    )
  }

  async mcpEnableTool(serverName: string, tool: string): Promise<{ ok: boolean }> {
    return this.post(
      `/mcp/servers/${encodeURIComponent(serverName)}/tools/${encodeURIComponent(tool)}/enable`,
      {},
    )
  }

  async mcpCallTool(
    serverName: string,
    tool: string,
    args: Record<string, unknown> = {},
  ): Promise<McpToolCallResult> {
    return this.post(
      `/mcp/servers/${encodeURIComponent(serverName)}/tools/${encodeURIComponent(tool)}/call`,
      { arguments: args },
    )
  }

  async mcpMarketplaceList(): Promise<McpMarketplace[]> {
    return this.get('/mcp/marketplaces')
  }

  async mcpMarketplaceAdd(name: string, url: string): Promise<McpMarketplace> {
    return this.post('/mcp/marketplaces', { name, url })
  }

  async mcpMarketplaceRemove(name: string): Promise<{ removed: boolean }> {
    const { data } = await this.request<ApiEnvelope<{ removed: boolean }>>(
      apiPath(`/mcp/marketplaces/${encodeURIComponent(name)}`),
      { method: 'DELETE' },
    )
    return data
  }

  async providers(): Promise<DaemonProviderInfo[]> {
    return this.get('/config/providers')
  }

  async model(): Promise<DaemonModelSnapshot> {
    return this.get('/config/model')
  }

  async switchDefaultModel(input: DaemonModelSwitchInput): Promise<DaemonModelSwitchResult> {
    return this.post('/config/model/switch', input)
  }

  async pullModel(input: DaemonModelPullInput): Promise<DaemonModelPullResponse> {
    return this.post('/config/model/pull', input)
  }

  async validateProvider(
    input: DaemonProviderValidationInput,
  ): Promise<DaemonProviderValidationResult> {
    return this.post('/config/providers/validate', input)
  }

  async discoverProviderModels(
    input: DaemonProviderDiscoverModelsInput,
  ): Promise<DaemonProviderDiscoverModelsResult> {
    return this.post('/config/providers/discover-models', input)
  }

  /**
   * Server-side model rediscovery for a stored provider. The daemon uses its
   * own (unredacted) stored credentials, so this works for key-protected
   * providers where the client only ever sees a redacted apiKey.
   */
  async refreshProviderModels(providerId: string): Promise<DaemonProviderRefreshModelsResult> {
    return this.post(`/config/providers/${encodeURIComponent(providerId)}/refresh-models`)
  }

  async upsertProvider(
    provider: DaemonConfigProvider,
    mode: 'create' | 'update',
  ): Promise<DaemonProviderMutationResult> {
    return this.put(
      `/config/providers/${encodeURIComponent(provider.id)}?mode=${encodeURIComponent(mode)}`,
      provider,
    )
  }

  async deleteProvider(providerId: string): Promise<void> {
    await this.del(`/config/providers/${encodeURIComponent(providerId)}`)
  }

  async personas(): Promise<DaemonPersona[]> {
    return this.get('/personas')
  }

  async devices(): Promise<DaemonDevice[]> {
    return this.get('/devices')
  }

  async agents(): Promise<DaemonAgentDescriptor[]> {
    return this.get('/agents')
  }

  async agentGraphs(): Promise<DaemonAgentGraphSnapshot[]> {
    return this.get('/agents/graphs')
  }

  async createUserAgent(
    input: import('./types.js').DaemonUserAgentInput,
  ): Promise<DaemonAgentDescriptor> {
    return this.post('/agents', input)
  }

  async deleteUserAgent(id: string): Promise<void> {
    await this.del(`/agents/${encodeURIComponent(id)}`)
  }

  async userCommands(): Promise<import('./types.js').DaemonUserCommandSummary[]> {
    return this.get('/commands')
  }

  async createUserCommand(
    input: import('./types.js').DaemonUserCommandInput,
  ): Promise<import('./types.js').DaemonUserCommandSummary> {
    return this.post('/commands', input)
  }

  async deleteUserCommand(id: string): Promise<void> {
    await this.del(`/commands/${encodeURIComponent(id)}`)
  }

  async resolveUserCommand(
    id: string,
    args?: string,
  ): Promise<import('./types.js').DaemonUserCommandResolved> {
    return this.post(`/commands/${encodeURIComponent(id)}/resolve`, { args: args ?? '' })
  }

  async undoSession(
    sessionId: string,
  ): Promise<{ sessionId: string; eventCount: number; undoDepth: number; redoDepth: number }> {
    return this.post(`/sessions/${encodeURIComponent(sessionId)}/undo`, {})
  }

  async redoSession(
    sessionId: string,
  ): Promise<{ sessionId: string; eventCount: number; undoDepth: number; redoDepth: number }> {
    return this.post(`/sessions/${encodeURIComponent(sessionId)}/redo`, {})
  }

  /**
   * Durable committed edit checkpoints for a session, oldest first. File
   * time travel — distinct from conversation undo/redo above, which
   * rewrites the event log without touching the working tree.
   */
  async listSessionCheckpoints(sessionId: string): Promise<EditCheckpointSummary[]> {
    const { checkpoints } = await this.get<{ checkpoints: EditCheckpointSummary[] }>(
      `/sessions/${encodeURIComponent(sessionId)}/checkpoints`,
    )
    return checkpoints
  }

  async listSessionInbox(sessionId: string, options: { after?: number; limit?: number; unreadOnly?: boolean } = {}): Promise<{ items: SessionInboxItem[]; nextCursor: number | null }> {
    const query = new URLSearchParams()
    for (const [key, value] of Object.entries(options)) if (value !== undefined) query.set(key, String(value))
    return this.get(`/sessions/${encodeURIComponent(sessionId)}/inbox?${query}`)
  }

  async listWork(): Promise<{ items: WorkItem[]; unavailable: string[]; truncated: string[] }> {
    const [work, chats] = await Promise.allSettled([
      this.get<{ items: WorkItem[]; unavailable: string[]; truncated: string[] }>('/work'), this.backgroundChatJobs(),
    ])
    const result = work.status === 'fulfilled' ? work.value : { items: [] as WorkItem[], unavailable: ['work'], truncated: [] as string[] }
    if (chats.status === 'fulfilled') {
      if (chats.value.jobs.length > 100) result.truncated.push('chat')
      result.items.push(...chats.value.jobs.slice(0, 100).map(j => ({ key: `chat:${j.jobId}`, kind: 'chat' as const, id: j.jobId, sessionId: j.sessionId, status: j.status, title: 'Background chat', detail: j.progress?.label ?? '', action: `sepilot tasks inspect chat:${j.jobId}` })))
    } else result.unavailable.push('chat')
    if (work.status === 'rejected' && chats.status === 'rejected') throw work.reason
    return result
  }

  async readManagedProcess(id: string, offsets: { stdoutOffset?: number; stderrOffset?: number } = {}): Promise<{ status: string; stdout: string; stderr: string; screen?: string; nextStdoutOffset: number; nextStderrOffset: number }> {
    const query = new URLSearchParams()
    for (const [key, value] of Object.entries(offsets)) if (value !== undefined) query.set(key, String(value))
    return this.get(`/work/process/${encodeURIComponent(id)}?${query}`)
  }

  async stopManagedProcess(id: string): Promise<{ status: string }> {
    return this.request<ApiEnvelope<{ status: string }>>(apiPath(`/work/process/${encodeURIComponent(id)}`), { method: 'DELETE' }).then(r => r.data)
  }

  async writeManagedProcess(id: string, text: string, end = false): Promise<{ accepted: boolean }> {
    return this.post(`/work/process/${encodeURIComponent(id)}/input`, { text, end })
  }

  async acknowledgeSessionInbox(sessionId: string, seq: number): Promise<{ acknowledged: boolean }> {
    return this.post(`/sessions/${encodeURIComponent(sessionId)}/inbox/ack`, { seq })
  }

  /**
   * Restore every file touched at or after the checkpoint to its state
   * before it, then drop the rewound checkpoints — git-reset semantics:
   * history past the rewind point no longer describes the working tree.
   */
  async rewindSessionFiles(
    sessionId: string,
    checkpointId: string,
    expectedCheckpointIds?: string[],
  ): Promise<{
    checkpoint: EditCheckpointSummary
    restoredFiles: string[]
    incompleteHistory: boolean
  }> {
    return this.post(`/sessions/${encodeURIComponent(sessionId)}/rewind`, { checkpointId, ...(expectedCheckpointIds ? { expectedCheckpointIds } : {}) })
  }

  async previewSessionRewind(sessionId: string, checkpointId: string): Promise<{ checkpoint: EditCheckpointSummary; checkpointIds: string[]; files: Array<{ path: string; operation: 'restore' | 'remove'; diff?: string; conflict: boolean }>; incompleteHistory: boolean }> {
    return this.get(`/sessions/${encodeURIComponent(sessionId)}/checkpoints/${encodeURIComponent(checkpointId)}/preview`)
  }

  async projects(): Promise<DaemonProject[]> {
    return this.get('/projects')
  }

  async createProject(input: DaemonCreateProjectInput): Promise<DaemonProject> {
    return this.post('/projects', input)
  }

  async updateProject(id: string, input: DaemonUpdateProjectInput): Promise<DaemonProject> {
    return this.put(`/projects/${encodeURIComponent(id)}`, input)
  }

  async deleteProject(id: string): Promise<void> {
    await this.del(`/projects/${encodeURIComponent(id)}`)
  }

  async attachSessionToProject(projectId: string, sessionId: string): Promise<DaemonProject> {
    return this.post(`/projects/${encodeURIComponent(projectId)}/sessions`, {
      sessionId,
    })
  }

  coworkStream(message: string, sessionId?: string, options?: ChatOptions): Promise<Response> {
    return this.fetch(apiPath('/cowork/run'), {
      method: 'POST',
      body: { message, sessionId, ...options },
    })
  }

  async listRecentMemory(
    options?: DaemonMemoryRecentOptions,
  ): Promise<DaemonMemoryRecentEntry[]> {
    const params = new URLSearchParams()
    if (options?.limit != null) params.set('limit', String(options.limit))
    if (options?.includeAllScopes != null)
      params.set('includeAllScopes', String(options.includeAllScopes))
    if (options?.sources && options.sources.length > 0) params.set('sources', options.sources.join(','))
    if (options?.tags && options.tags.length > 0) params.set('tags', options.tags.join(','))
    if (options?.tagsLogic) params.set('tagsLogic', options.tagsLogic)
    if (options?.excludeTags && options.excludeTags.length > 0)
      params.set('excludeTags', options.excludeTags.join(','))
    if (options?.createdAfter) params.set('createdAfter', options.createdAfter)
    if (options?.createdBefore) params.set('createdBefore', options.createdBefore)
    if (options?.includeSuperseded != null)
      params.set('includeSuperseded', String(options.includeSuperseded))
    if (options?.includeArchived != null)
      params.set('includeArchived', String(options.includeArchived))
    if (options?.sortBy) params.set('sortBy', options.sortBy)
    const query = params.toString()
    return this.get(query ? `/memory/recent?${query}` : '/memory/recent')
  }

  async searchMemory(
    query: string,
    options?: DaemonMemorySearchOptions,
  ): Promise<DaemonMemoryEntry[]> {
    const params = new URLSearchParams({ query })
    if (options?.asOf) params.set('asOf', options.asOf)
    if (options?.includeInactive != null) params.set('includeInactive', String(options.includeInactive))
    if (options?.type) params.set('type', options.type)
    if (options?.limit != null) params.set('limit', String(options.limit))
    if (options?.includeAllScopes != null)
      params.set('includeAllScopes', String(options.includeAllScopes))
    if (options?.sources && options.sources.length > 0) params.set('sources', options.sources.join(','))
    if (options?.tags && options.tags.length > 0) params.set('tags', options.tags.join(','))
    if (options?.tagsLogic) params.set('tagsLogic', options.tagsLogic)
    if (options?.excludeTags && options.excludeTags.length > 0)
      params.set('excludeTags', options.excludeTags.join(','))
    if (options?.createdAfter) params.set('createdAfter', options.createdAfter)
    if (options?.createdBefore) params.set('createdBefore', options.createdBefore)
    if (options?.includeSuperseded != null)
      params.set('includeSuperseded', String(options.includeSuperseded))
    if (options?.includeArchived != null)
      params.set('includeArchived', String(options.includeArchived))
    if (options?.sortBy) params.set('sortBy', options.sortBy)
    return this.get(`/memory/search?${params.toString()}`)
  }

  async listPinnedMemories(limit?: number): Promise<DaemonMemoryPinnedEntry[]> {
    const query = limit ? `?limit=${encodeURIComponent(String(limit))}` : ''
    return this.get(`/memory/pinned${query}`)
  }

  async pinMemory(id: string): Promise<{ pinned: true }> {
    return this.post(`/memory/${encodeURIComponent(id)}/pin`)
  }

  async unpinMemory(id: string): Promise<{ pinned: false }> {
    const { data } = await this.request<ApiEnvelope<{ pinned: false }>>(
      apiPath(`/memory/${encodeURIComponent(id)}/pin`),
      { method: 'DELETE' },
    )
    return data
  }

  async memoryGraphPage(
    options: DaemonMemoryGraphPageOptions,
  ): Promise<DaemonMemoryGraphWikiPage> {
    const params = new URLSearchParams()
    if (options.id) params.set('id', options.id)
    if (options.query) params.set('query', options.query)
    if (options.limit != null) params.set('limit', String(options.limit))
    if (options.evidenceLimit != null) params.set('evidenceLimit', String(options.evidenceLimit))
    if (options.includeAllScopes != null)
      params.set('includeAllScopes', String(options.includeAllScopes))
    return this.get(`/memory/graph/page?${params.toString()}`)
  }

  async memoryGraphAudit(
    options?: DaemonMemoryGraphAuditOptions,
  ): Promise<DaemonMemoryGraphQualityReport> {
    const params = new URLSearchParams()
    if (options?.limit != null) params.set('limit', String(options.limit))
    if (options?.signalLimit != null) params.set('signalLimit', String(options.signalLimit))
    if (options?.lowConfidenceThreshold != null)
      params.set('lowConfidenceThreshold', String(options.lowConfidenceThreshold))
    if (options?.staleAfterDays != null) params.set('staleAfterDays', String(options.staleAfterDays))
    if (options?.includeAllScopes != null)
      params.set('includeAllScopes', String(options.includeAllScopes))
    const query = params.toString()
    return this.get(query ? `/memory/graph/audit?${query}` : '/memory/graph/audit')
  }

  async memoryGraphRepair(
    options: DaemonMemoryGraphRepairOptions = {},
  ): Promise<DaemonMemoryGraphRepairResult> {
    return this.post('/memory/graph/repair', options)
  }

  async applyMemoryGraphRepairDecision(
    input: DaemonMemoryGraphRepairDecisionInput,
  ): Promise<DaemonMemoryGraphRepairDecisionResult> {
    return this.post('/memory/graph/repair/decision', input)
  }

  async memoryStatus(): Promise<DaemonMemorySemanticStatus> {
    return this.get('/memory/status')
  }

  async memoryLifecycle(
    options?: DaemonMemoryLifecycleOptions,
  ): Promise<DaemonMemoryLifecycleStatus> {
    const params = new URLSearchParams()
    if (options?.staleAfterDays != null)
      params.set('staleAfterDays', String(options.staleAfterDays))
    if (options?.lowImportance != null) params.set('lowImportance', String(options.lowImportance))
    const query = params.toString()
    return this.get(query ? `/memory/lifecycle?${query}` : '/memory/lifecycle')
  }

  async memoryAudit(options?: DaemonMemoryAuditOptions): Promise<DaemonMemoryAuditEntry[]> {
    const params = new URLSearchParams()
    if (options?.memoryId) params.set('memoryId', options.memoryId)
    if (options?.limit != null) params.set('limit', String(options.limit))
    const query = params.toString()
    return this.get(query ? `/memory/audit?${query}` : '/memory/audit')
  }

  async memorySecurityAudit(
    options?: DaemonMemorySecurityAuditOptions,
  ): Promise<DaemonMemorySecurityAuditResult> {
    const params = new URLSearchParams()
    if (options?.limit != null) params.set('limit', String(options.limit))
    if (options?.since) params.set('since', options.since)
    if (options?.actor) params.set('actor', options.actor)
    if (options?.authKind) params.set('authKind', options.authKind)
    if (options?.route) params.set('route', options.route)
    const query = params.toString()
    return this.request<DaemonMemorySecurityAuditResult>(
      apiPath(query ? `/memory/security-audit?${query}` : '/memory/security-audit'),
      { method: 'GET' },
    )
  }

  async memoryScopes(): Promise<DaemonMemoryScopes> {
    return this.get('/memory/scopes')
  }

  async transferMemoryScope(
    source: string,
    input: DaemonMemoryScopeTransferInput,
  ): Promise<DaemonMemoryScopeTransferResult> {
    return this.post(`/memory/scopes/${encodeURIComponent(source)}/transfer`, input)
  }

  async runMemoryMaintenance(
    input: DaemonMemoryMaintenanceInput = {},
  ): Promise<DaemonMemoryMaintenanceResult> {
    return this.post('/memory/maintenance', input)
  }

  async fileMemory(): Promise<DaemonFileMemorySnapshot> {
    return this.get('/memory/file')
  }

  async updateFileMemorySection(
    sectionTitle: string,
    content: string,
  ): Promise<DaemonFileMemorySectionUpdateResult> {
    return this.put(`/memory/file/sections/${encodeURIComponent(sectionTitle)}`, {
      content,
    })
  }

  async deleteFileMemorySection(sectionTitle: string): Promise<{ deleted: boolean }> {
    const { data } = await this.request<ApiEnvelope<{ deleted: boolean }>>(
      apiPath(`/memory/file/sections/${encodeURIComponent(sectionTitle)}`),
      { method: 'DELETE' },
    )
    return data
  }

  async searchMemoryDocuments(
    query: string,
    options?: DaemonMemoryDocumentSearchOptions,
  ): Promise<DaemonMemoryDocumentChunk[]> {
    const params = new URLSearchParams({ query })
    if (options?.type) params.set('type', options.type)
    if (options?.limit != null) params.set('limit', String(options.limit))
    if (options?.documentId) params.set('documentId', options.documentId)
    if (options?.includeAllScopes != null)
      params.set('includeAllScopes', String(options.includeAllScopes))
    if (options?.sources && options.sources.length > 0) params.set('sources', options.sources.join(','))
    if (options?.tags && options.tags.length > 0) params.set('tags', options.tags.join(','))
    if (options?.tagsLogic) params.set('tagsLogic', options.tagsLogic)
    if (options?.excludeTags && options.excludeTags.length > 0)
      params.set('excludeTags', options.excludeTags.join(','))
    if (options?.createdAfter) params.set('createdAfter', options.createdAfter)
    if (options?.createdBefore) params.set('createdBefore', options.createdBefore)
    if (options?.sortBy) params.set('sortBy', options.sortBy)
    return this.get(`/memory/documents/search?${params.toString()}`)
  }

  async listMemoryDocuments(
    options?: DaemonMemoryDocumentListOptions,
  ): Promise<DaemonMemoryDocument[]> {
    const params = new URLSearchParams()
    if (options?.query?.trim()) params.set('query', options.query.trim())
    if (options?.limit != null) params.set('limit', String(options.limit))
    if (options?.includeAllScopes != null)
      params.set('includeAllScopes', String(options.includeAllScopes))
    const query = params.toString()
    return this.get(query ? `/memory/documents?${query}` : '/memory/documents')
  }

  async ingestMemoryDocument(
    input: DaemonMemoryDocumentIngestInput,
  ): Promise<DaemonMemoryDocument> {
    return this.post('/memory/documents', input)
  }

  async memoryDocument(id: string): Promise<DaemonMemoryDocument> {
    return this.get(`/memory/documents/${encodeURIComponent(id)}`)
  }

  async deleteMemoryDocument(id: string): Promise<void> {
    await this.del(`/memory/documents/${encodeURIComponent(id)}`)
  }

  async addMemory(
    content: string,
    tags?: string[],
    options?: DaemonMemoryAddOptions,
  ): Promise<DaemonMemoryAddResult> {
    return this.post('/memory', {
      content,
      tags,
      source: options?.source,
      evidence: options?.evidence,
      reason: options?.reason,
    })
  }

  async updateMemory(id: string, input: DaemonMemoryUpdateInput): Promise<DaemonMemoryEntry> {
    return this.put(`/memory/${encodeURIComponent(id)}`, input)
  }

  async deleteMemory(id: string, options?: DaemonMemoryDeleteOptions): Promise<void> {
    const params = new URLSearchParams()
    if (options?.reason) params.set('reason', options.reason)
    const query = params.toString()
    await this.del(`/memory/${encodeURIComponent(id)}${query ? `?${query}` : ''}`)
  }

  async reindexMemory(): Promise<DaemonMemoryReindexResult> {
    return this.post('/memory/reindex')
  }

  async uploadFiles(files: DaemonUploadFileInput[]): Promise<DaemonFileUploadResult> {
    const uploaded: DaemonUploadedFile[] = []

    for (let index = 0; index < files.length; index += MAX_FILES_PER_UPLOAD_REQUEST) {
      const batch = files.slice(index, index + MAX_FILES_PER_UPLOAD_REQUEST)
      const form = new FormData()

      for (const file of batch) {
        form.append('files', toUploadBlob(file.content, file.mimeType), file.filename)
      }

      const result = await this.post<DaemonFileUploadResult>('/files/upload', form)
      uploaded.push(...result.files)
    }

    return { files: uploaded }
  }

  async file(id: string): Promise<DaemonUploadedFile> {
    return this.get(`/files/${encodeURIComponent(id)}`)
  }

  async deleteFile(id: string): Promise<void> {
    await this.del(`/files/${encodeURIComponent(id)}`)
  }

  async fileContentPart(id: string): Promise<DaemonFileContentPart> {
    return this.get(`/files/${encodeURIComponent(id)}/content-part`)
  }

  async indexUploadedFile(
    id: string,
    input?: DaemonUploadedFileIndexInput,
  ): Promise<DaemonMemoryDocument> {
    return this.post(`/files/${encodeURIComponent(id)}/index`, input ?? {})
  }

  async usage(): Promise<DaemonUsageSummary> {
    return this.get('/usage')
  }

  async usageDaily(days?: number): Promise<DaemonDailyUsageSummary[]> {
    return this.get(`/usage/daily${days ? `?days=${days}` : ''}`)
  }

  async usageSnapshot(): Promise<DaemonUsageSnapshot> {
    return this.get('/usage/snapshot')
  }

  async observabilitySnapshot(
    range: DaemonObservabilityRange = '7d',
  ): Promise<DaemonObservabilitySnapshot> {
    return this.get(`/observability/snapshot?range=${encodeURIComponent(range)}`)
  }

  async recordObservabilityEvents(
    events: DaemonObservabilityEventInput[],
  ): Promise<{ accepted: number }> {
    return this.post('/observability/events', { events })
  }

  async observabilityEvents(
    options: {
      limit?: number
      severity?: DaemonObservabilitySeverity
      eventType?: string
    } = {},
  ): Promise<DaemonObservabilityEvent[]> {
    const params = new URLSearchParams()
    if (typeof options.limit === 'number') {
      params.set('limit', String(options.limit))
    }
    if (options.severity) {
      params.set('severity', options.severity)
    }
    if (options.eventType) {
      params.set('eventType', options.eventType)
    }
    const query = params.toString()
    return this.get(`/observability/events${query ? `?${query}` : ''}`)
  }

  async observabilityCrashes(limit?: number): Promise<DaemonObservabilityEvent[]> {
    return this.get(
      `/observability/crashes${limit ? `?limit=${encodeURIComponent(String(limit))}` : ''}`,
    )
  }

  async submitFeedback(input: DaemonFeedbackInput): Promise<DaemonFeedbackRecord> {
    return this.post('/feedback', input)
  }

  async feedbackPromptState(
    input: DaemonFeedbackPromptStateInput = {},
  ): Promise<DaemonFeedbackPromptState> {
    const params = new URLSearchParams()
    if (input.surface) params.set('surface', input.surface)
    if (input.sessionId) params.set('sessionId', input.sessionId)
    if (input.messageId) params.set('messageId', input.messageId)
    if (input.userIdHash) params.set('userIdHash', input.userIdHash)
    const query = params.toString()
    return this.get(`/feedback/prompt-state${query ? `?${query}` : ''}`)
  }

  async observabilityPrivacy(): Promise<DaemonObservabilityPrivacySettings> {
    return this.get('/observability/privacy')
  }

  async updateObservabilityPrivacy(
    input: DaemonObservabilityPrivacySettingsInput,
  ): Promise<DaemonObservabilityPrivacySettings> {
    return this.patch('/observability/privacy', input)
  }

  async exportObservability(
    input: DaemonObservabilityExportInput = {},
  ): Promise<DaemonObservabilityExportResult> {
    return this.post('/observability/export', input)
  }

  async exportObservabilitySupportBundle(
    input: DaemonObservabilitySupportBundleInput = {},
  ): Promise<DaemonObservabilitySupportBundleResult> {
    return this.post('/observability/support-bundle', input)
  }

  async pruneObservability(): Promise<DaemonObservabilityPruneResult> {
    return this.post('/observability/prune', {})
  }

  /**
   * Dispatch an isolated subagent on the daemon. The HTTP route returns the
   * raw `SubagentDispatchResult` (no `ApiEnvelope` wrap) — we hit it via
   * `fetch` so the caller does not lose the `error` field on the typed
   * result. Privilege-escalation rejections surface as a 400 with
   * `code: 'SUBAGENT_TOOL_ESCALATION'`; the helper preserves that signal
   * by throwing an `Error` whose message contains the error code.
   */
  async dispatchSubagent(input: SubagentDispatchInput): Promise<SubagentDispatchResult> {
    const response = await this.fetch(apiPath('/subagents/dispatch'), {
      method: 'POST',
      body: { ...input } as Record<string, unknown>,
    })
    if (!response.ok) {
      let code = 'HTTP_ERROR'
      let message = `subagent dispatch failed: ${response.status}`
      try {
        const body = (await response.json()) as {
          error?: { code?: string; message?: string }
        }
        if (body?.error?.code) code = body.error.code
        if (body?.error?.message) message = body.error.message
      } catch {
        /* non-JSON error body — fall through to status-only message */
      }
      const err = new Error(`${code}: ${message}`) as Error & {
        code?: string
        statusCode?: number
      }
      err.code = code
      err.statusCode = response.status
      throw err
    }
    return response.json() as Promise<SubagentDispatchResult>
  }

  async dispatchSubagentBackground(
    input: SubagentDispatchInput,
  ): Promise<SubagentBackgroundDispatchResult> {
    const response = await this.fetch(apiPath('/jobs/subagent'), {
      method: 'POST',
      body: { ...input } as Record<string, unknown>,
    })
    if (!response.ok) {
      let message = `subagent background dispatch failed: ${response.status}`
      try {
        const body = (await response.json()) as {
          error?: { message?: string }
        }
        if (body?.error?.message) message = body.error.message
      } catch {
        /* non-JSON error body — fall through to status-only message */
      }
      throw new Error(message)
    }
    return response.json() as Promise<SubagentBackgroundDispatchResult>
  }

  async listSubagentCategories(): Promise<SubagentDelegationCategoriesResult> {
    const response = await this.fetch(apiPath('/subagents/categories'), {
      method: 'GET',
    })
    if (!response.ok) {
      throw new Error(`subagent categories failed: ${response.status}`)
    }
    return response.json() as Promise<SubagentDelegationCategoriesResult>
  }

  async createPlan(input: WorkPlanCreateInput): Promise<WorkPlan> {
    const response = await this.fetch(apiPath('/plans'), {
      method: 'POST',
      body: { ...input } as Record<string, unknown>,
    })
    if (!response.ok) throw new Error(`create plan failed: ${response.status}`)
    return response.json() as Promise<WorkPlan>
  }

  async listPlans(): Promise<WorkPlanListResult> {
    const response = await this.fetch(apiPath('/plans'), { method: 'GET' })
    if (!response.ok) throw new Error(`list plans failed: ${response.status}`)
    return response.json() as Promise<WorkPlanListResult>
  }

  async getPlan(id: string): Promise<WorkPlan> {
    const response = await this.fetch(apiPath(`/plans/${encodeURIComponent(id)}`), {
      method: 'GET',
    })
    if (!response.ok) throw new Error(`get plan failed: ${response.status}`)
    return response.json() as Promise<WorkPlan>
  }

  async updatePlan(id: string, input: WorkPlanPatchInput): Promise<WorkPlan> {
    const response = await this.fetch(apiPath(`/plans/${encodeURIComponent(id)}`), {
      method: 'PATCH',
      body: { ...input } as Record<string, unknown>,
    })
    if (!response.ok) throw new Error(`update plan failed: ${response.status}`)
    return response.json() as Promise<WorkPlan>
  }

  async startPlan(id: string, input: WorkPlanStartInput = {}): Promise<WorkPlanStartResult> {
    const response = await this.fetch(apiPath(`/plans/${encodeURIComponent(id)}/start`), {
      method: 'POST',
      body: { ...input } as Record<string, unknown>,
    })
    if (!response.ok) throw new Error(`start plan failed: ${response.status}`)
    return response.json() as Promise<WorkPlanStartResult>
  }

  async runExternalAcpAgent(input: ExternalAcpDispatchInput): Promise<ExternalAcpDispatchResult> {
    const response = await this.fetch(apiPath('/acp/agents/run'), {
      method: 'POST',
      body: { ...input } as Record<string, unknown>,
    })
    if (!response.ok) {
      let code = 'HTTP_ERROR'
      let message = `external ACP agent run failed: ${response.status}`
      try {
        const body = (await response.json()) as {
          error?: { code?: string; message?: string }
        }
        if (body?.error?.code) code = body.error.code
        if (body?.error?.message) message = body.error.message
      } catch {
        /* non-JSON error body — fall through to status-only message */
      }
      const err = new Error(`${code}: ${message}`) as Error & {
        code?: string
        statusCode?: number
      }
      err.code = code
      err.statusCode = response.status
      throw err
    }
    return response.json() as Promise<ExternalAcpDispatchResult>
  }

  async setSecret(key: string, value: string): Promise<{ ok: boolean }> {
    return this.post(`/secrets/${encodeURIComponent(key)}`, { value })
  }

  async listSecrets(): Promise<{ keys: string[] }> {
    return this.get('/secrets')
  }

  async removeSecret(key: string): Promise<{ removed: boolean }> {
    const { data } = await this.request<ApiEnvelope<{ removed: boolean }>>(
      apiPath(`/secrets/${encodeURIComponent(key)}`),
      { method: 'DELETE' },
    )
    return data
  }
}

function toUploadBlob(content: DaemonUploadFileInput['content'], mimeType?: string): Blob {
  if (content instanceof Blob) {
    if (!mimeType || content.type === mimeType) return content
    return new Blob([content], { type: mimeType })
  }

  if (content instanceof ArrayBuffer) {
    return new Blob([content], {
      type: mimeType ?? 'application/octet-stream',
    })
  }

  if (ArrayBuffer.isView(content)) {
    return new Blob([content as BlobPart], {
      type: mimeType ?? 'application/octet-stream',
    })
  }

  return new Blob([content], {
    type: mimeType ?? 'text/plain',
  })
}
