/**
 * Narrow capability types extracted from RuntimeServices.
 *
 * Each route module should depend only on the slice it actually uses,
 * not the full service locator. These Pick aliases make the
 * dependency surface explicit and auditable.
 *
 * Migrate route modules incrementally: update the function signature
 * to accept the narrow type, then pass `runtime` at the call site
 * (TypeScript structurally subtypes RuntimeServices to any Pick).
 */
import type { RuntimeServices } from './types.js'

// ── Session detail ───────────────────────────────────────────────
export type SessionDetailCapabilities = Pick<
  RuntimeServices,
  | 'sessions'
  | 'questions'
  | 'approvalRegistry'
  | 'approvalCheckpoints'
  | 'runCheckpoints'
  | 'toolRegistry'
  | 'toolExecutions'
  | 'delegationWorker'
  | 'primaryAgents'
  | 'activeRuns'
  | 'semanticIndex'
  | 'dreaming'
> &
  Partial<Pick<RuntimeServices, 'sessionRuntimeSnapshots'>>

// ── Config mutation ──────────────────────────────────────────────
export type ConfigMutationCapabilities = Pick<
  RuntimeServices,
  'config' | 'configMutationService' | 'auditLogger'
>

// ── Channel pipeline (router + pipeline + sub-steps) ────────────
export type ChannelPipelineCapabilities = Pick<
  RuntimeServices,
  | 'channelPipelineMonitor'
  | 'autonomy'
  | 'sessions'
  | 'questions'
  | 'customDefs'
  | 'providerRegistry'
  | 'approvalRegistry'
  | 'channelAcl'
  | 'channelReplayStore'
  | 'channelSessionStore'
  | 'channelOriginStore'
  | 'config'
  | 'auditLogger'
  | 'gatewayClient'
  | 'dreaming'
  | 'toolRegistry'
  | 'skillRegistry'
  | 'policyEngine'
  | 'semanticIndex'
  | 'fileMemory'
  | 'fileMemoryRegistry'
  | 'remindersStore'
  | 'usageTracker'
  | 'hookRegistry'
  | 'observability'
  | 'providerCircuitBreaker'
  | 'graphRegistry'
  | 'channels'
  | 'jobStore'
  // Needed by the /model channel command: switch the default provider/model
  // through the single config-mutation service and persist + rebind runtime.
  | 'configMutationService'
  | 'dataDir'
  | 'providerFactoryRegistry'
  | 'modelRouter'
  | 'mcpManager'
  | 'mcpPromptsRegistry'
  | 'skillSourceUrlPolicy'
  | 'marketplaceCatalog'
  | 'installPipeline'
  | 'sessionWatchBroker'
>

// ── Delegation worker ──────────────────────────────────────────
export type DelegationWorkerCapabilities = Pick<
  RuntimeServices,
  | 'config'
  | 'delegator'
  | 'sessions'
  | 'questions'
  | 'providerRegistry'
  | 'toolRegistry'
  | 'skillRegistry'
  | 'fileMemory'
  | 'policyEngine'
  | 'autonomy'
  | 'semanticIndex'
  | 'auditLogger'
  | 'usageTracker'
  | 'hookRegistry'
  | 'llmCache'
  | 'providerCircuitBreaker'
  | 'graphRegistry'
  | 'runCheckpoints'
  | 'toolExecutions'
  | 'dreaming'
  | 'sessionWatchBroker'
>

// ── Lifecycle (startup/shutdown orchestration) ──────────────────
export type LifecycleCapabilities = Pick<
  RuntimeServices,
  | 'channels'
  | 'managedProcesses'
  | 'lsp'
  | 'pluginLoader'
  | 'mcpManager'
  | 'graphAgentLoader'
  | 'configWatcher'
  | 'delegationWorker'
  | 'notificationRelayWorker'
  | 'mdns'
  | 'updater'
  | 'schedulerEngine'
  | 'serviceSupervisor'
  | 'channelPipelineMonitor'
  | 'semanticIndex'
  | 'usageTracker'
  | 'telemetry'
  | 'swarmRunRegistry'
>

// ── Startup (side-effect service boot) ──────────────────────────
export type StartupCapabilities = Pick<
  RuntimeServices,
  | 'channels'
  | 'mdns'
  | 'updater'
  | 'providerRegistry'
  | 'toolRegistry'
  | 'policyEngine'
  | 'autonomy'
  | 'auditLogger'
  | 'config'
  | 'dreaming'
  | 'semanticIndex'
  | 'delegationWorker'
  | 'notificationRelayWorker'
  | 'graphAgentLoader'
  // SchedulerStack fields populated by lifecycle.ts after buildSchedulerStack()
  | 'jobStore'
  | 'schedulerEngine'
  | 'parseWhen'
  | 'schedulerDefaultTimezone'
  | 'triggerSchedulerJob'
  // ChannelPipelineCapabilities fields needed for buildSchedulerStack()
  | 'questions'
  | 'customDefs'
  | 'approvalRegistry'
  | 'channelAcl'
  | 'channelReplayStore'
  | 'channelSessionStore'
  | 'channelOriginStore'
  | 'gatewayClient'
  | 'skillRegistry'
  | 'fileMemory'
  | 'usageTracker'
  | 'hookRegistry'
  | 'observability'
  | 'providerCircuitBreaker'
  | 'graphRegistry'
  | 'sessions'
  | 'channelPipelineMonitor'
  | 'runLimiter'
  | 'runCheckpoints'
  | 'toolExecutions'
  | 'approvalCheckpoints'
  // Required so the channel pipeline (via ChannelPipelineCapabilities) can run
  // the /model command's config-mutation + runtime-rebind path.
  | 'configMutationService'
  | 'dataDir'
  | 'providerFactoryRegistry'
  | 'modelRouter'
  | 'mcpManager'
  | 'mcpPromptsRegistry'
  | 'skillSourceUrlPolicy'
  | 'marketplaceCatalog'
  | 'installPipeline'
  | 'sessionWatchBroker'
>

// ── Config persistence ─────────────────────────────────────────
export type ConfigPersistenceCapabilities = Pick<RuntimeServices, 'config' | 'dataDir'>

// ── Config rebind (providers + hooks + MCP) ────────────────────
export type ConfigRebindCapabilities = Pick<
  RuntimeServices,
  | 'config'
  | 'dataDir'
  | 'providerFactoryRegistry'
  | 'providerRegistry'
  | 'modelRouter'
  | 'dreaming'
  | 'semanticIndex'
  | 'hookRegistry'
  | 'auditLogger'
  | 'mcpManager'
  | 'mcpPromptsRegistry'
  | 'toolRegistry'
  | 'skillSourceUrlPolicy'
>

// ── Config autonomy rebind ─────────────────────────────────────
export type ConfigAutonomyCapabilities = Pick<RuntimeServices, 'config' | 'autonomy' | 'channelAcl'>

// ── Channel rebind ────────────────────────────────────────────
export type ChannelRebindCapabilities = Pick<
  RuntimeServices,
  | 'config'
  | 'configMutationService'
  | 'channels'
  | 'gatewayClient'
  | 'channelFactoryRegistry'
  | 'channelAcl'
  | 'channelRouter'
  | 'dataDir'
>

// ── Audit query ──────────────────────────────────────────────
export type AuditQueryCapabilities = Pick<RuntimeServices, 'auditLogger'>

// ── Audit logging with device name ────────────────────────────
export type AuditLogCapabilities = Pick<RuntimeServices, 'auditLogger' | 'config'>

// ── Channel summary / webhook security ─────────────────────────
export type ChannelSummaryCapabilities = Pick<RuntimeServices, 'config' | 'channels'>

// ── Provider factory ─────────────────────────────────────────
export type ProviderFactoryCapabilities = Pick<RuntimeServices, 'providerFactoryRegistry'>

// ── Auth (extension token validation) ────────────────────────
export type AuthCapabilities = Pick<RuntimeServices, 'extensionTokenStore'>
