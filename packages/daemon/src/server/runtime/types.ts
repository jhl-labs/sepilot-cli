import type {
  AutonomyLevel,
  IChannel,
  ISessionStore,
  ManualJobRunOptions,
  ManualJobRunResult,
} from '@sepilotd/core'
import type { SepilotdConfig } from '../../config/schema.js'
import type { ConfigWatcher } from './config-watcher.js'
import type { GraphAgentRegistry } from '../../agent/graph/registry.js'
import type { YamlGraphAgentLoader } from '../../agent/graph/yaml-loader.js'
import type { TaskDelegator } from '../../agent/delegator.js'
import type { DelegationWorker } from '../../agent/delegation-worker.js'
import type { ChannelRouter } from '../../channels/router.js'
import type { MdnsDiscovery } from '../../discovery/mdns.js'
import type { GatewayClient } from '../../gateway/client.js'
import type { HookRegistry } from '../../hook/registry.js'
import type { McpManager } from '../../mcp/manager.js'
import type { McpMarketplaceCatalog } from '../../mcp/marketplace-catalog.js'
import type { McpMarketplaceSource } from '../../mcp/marketplace-source.js'
import type { ConfigWriter } from '../../mcp/config-writer.js'
import type { McpPromptsRegistry } from '../../mcp/prompts-registry.js'
import type { DreamingEngine } from '../../memory/dreaming.js'
import type { FileMemory } from '../../memory/file-memory.js'
import type { ScopedFileMemoryRegistry } from '../../memory/scoped-file-memory.js'
import type { RemindersStore } from '../../memory/reminders.js'
import type { SemanticMemoryStore } from '../../memory/types.js'
import type { UsageTracker } from '../../memory/usage-tracker.js'
import type { TelemetryManager } from '../../observability/telemetry.js'
import type { ObservabilityRepo } from '../../observability/events.js'
import type { PluginLoader } from '../../plugins/loader.js'
import type { IChannelFactoryRegistry, IProviderFactoryRegistry } from '../../plugins/contracts.js'
import type { LLMCache } from '../../providers/cache.js'
import type { ProviderCircuitBreaker } from '../../providers/circuit-breaker.js'
import type { ModelFallbackState } from '../../providers/model-fallback.js'
import type { ModelRouter } from '../../providers/model-router.js'
import type { ProviderRegistry } from '../../providers/registry.js'
import type { JobStore } from '../../scheduler/job-store.js'
import type { SchedulerEngine } from '../../scheduler/engine.js'
import type { parseWhen } from '../../scheduler/time-parser.js'
import type { PolicyEngine } from '../../security/policy-engine.js'
import type { JsonlAuditLogger } from '../../security/audit-logger.js'
import type { ChannelAcl } from '../../security/channel-acl.js'
import type { EncryptionManager } from '../../security/encryption.js'
import type { SecretVault } from '../../security/secret-vault.js'
import type { FileSkillRegistry } from '../../skills/registry.js'
import type { SkillStore } from '../../skills/store.js'
import type { MarketplaceCatalog } from '../../skills/marketplace-catalog.js'
import type { InstallPipeline } from '../../skills/install-pipeline.js'
import type { SkillSourceUrlPolicy } from '../../skills/source-url-policy.js'
import type { ToolRegistry } from '../../tools/registry.js'
import type { ManagedProcessRegistry } from '../../tools/process.js'
import type { ServiceSupervisor } from '../../service-supervisor/supervisor.js'
import type { PendingQuestionStore } from '../../tools/question.js'
import type { SubagentDispatcher } from '../../agent/subagent-dispatcher.js'
import type { BackgroundSubagents } from '../../jobs/subagent.js'
import type { CustomDefsService } from '../../agent/custom/service.js'
import type { PrimaryAgentStore } from '../../agent/primary-agent-store.js'
import type { LspLayer } from '../../lsp/layer.js'
import type { UpdateChecker } from '../../updater/checker.js'
import type { ApprovalRegistry } from './approvals.js'
import type { ApprovalDecisionStore } from './approval-decisions.js'
import type { ChannelPipelineMonitor } from './channel-pipeline-monitor.js'
import type { ChannelOriginStore } from '../../channels/channel-origin-store.js'
import type { DevicePairingRegistry } from './device-pairing.js'
import type { ChannelReplayStore } from './channel-replays.js'
import type { ChannelSessionStore } from './channel-sessions.js'
import type { ApprovalCheckpointStore } from './checkpoints.js'
import type { RunLimiter } from './run-limiter.js'
import type { RunCheckpointStore } from './runs.js'
import type { ActiveRunRegistry } from './active-runs.js'
import type { SessionBusyRegistry } from './session-busy.js'
import type { ToolExecutionStore } from './tool-executions.js'
import type { SessionWatchBroker } from './session-watch.js'
import type { SessionRuntimeSnapshotStore } from './session-runtime-snapshots.js'
import type { ExtensionAccessTokenStore } from './extension-tokens.js'
import type { ConfigMutationService } from './config-mutation-service.js'
import type { ExternalAcpAgentDispatcher } from '../../acp/external-agent.js'
import type { A2ATaskStore } from '../../a2a/server.js'

export interface RuntimeServices {
  dataDir: string
  config: SepilotdConfig
  /**
   * True when startup used SEPILOTD_CONFIG_DEGRADED_OK after a parse or
   * validation failure. Mutating config in this state can overwrite the
   * damaged-but-recoverable file with a blank/default document.
   */
  configLoadFailed?: boolean
  configLoadError?: string
  storageDegraded?: Array<{ store: string; error: string }>
  providerFactoryRegistry: IProviderFactoryRegistry
  providerRegistry: ProviderRegistry
  channelFactoryRegistry: IChannelFactoryRegistry
  toolRegistry: ToolRegistry
  managedProcesses: ManagedProcessRegistry
  serviceSupervisor: ServiceSupervisor
  questions: PendingQuestionStore
  customDefs: CustomDefsService
  primaryAgents: PrimaryAgentStore
  lsp: LspLayer
  autoApprove: boolean
  policyEngine: PolicyEngine
  sessions: ISessionStore
  auditLogger: JsonlAuditLogger
  autonomy: AutonomyLevel
  skillRegistry: FileSkillRegistry
  fileMemory?: FileMemory
  fileMemoryRegistry?: ScopedFileMemoryRegistry
  remindersStore?: RemindersStore
  semanticIndex: SemanticMemoryStore
  usageTracker: UsageTracker
  hookRegistry: HookRegistry
  gatewayClient: GatewayClient
  channelAcl: ChannelAcl
  encryption: EncryptionManager
  secretVault: SecretVault | null
  mcpManager: McpManager
  mcpMarketplaceCatalog: McpMarketplaceCatalog
  mcpMarketplaceSource: McpMarketplaceSource
  mcpConfigWriter: ConfigWriter | null
  mcpPromptsRegistry: McpPromptsRegistry
  telemetry: TelemetryManager
  observability?: ObservabilityRepo
  mdns: MdnsDiscovery
  updater: UpdateChecker
  extensionTokenStore: ExtensionAccessTokenStore
  channels: IChannel[]
  channelRouter: ChannelRouter
  delegationWorker: DelegationWorker
  notificationRelayWorker: import('../../notifications/publish.js').NotificationRelayDeliveryWorker
  jobStore: JobStore
  schedulerEngine: SchedulerEngine
  parseWhen: typeof parseWhen
  schedulerDefaultTimezone: string
  triggerSchedulerJob: (id: string, options?: ManualJobRunOptions) => Promise<ManualJobRunResult>
  delegator: TaskDelegator
  llmCache: LLMCache
  providerCircuitBreaker: ProviderCircuitBreaker
  modelFallbackState: ModelFallbackState
  editSnapshotStore: import('../../agent/edit-rollback/store.js').EditSnapshotStore
  toolStatsStore: import('../../agent/tool-learning/store.js').ToolStatsStore
  workspaceMutationTracker: import('../../agent/workspace-mutation/tracker.js').WorkspaceMutationTracker
  sessionUndoStack: import('./undo-stack.js').SessionUndoStack
  pluginEvents: import('../../plugins/event-bus.js').PluginEventBus
  modelRouter: ModelRouter
  pluginLoader: PluginLoader
  graphRegistry: GraphAgentRegistry
  graphAgentLoader?: YamlGraphAgentLoader
  userAgentLoader?: import('../../agent/user-agents/loader.js').UserAgentLoader
  userCommandStore?: import('../../agent/user-commands/loader.js').UserCommandStore
  dreaming: DreamingEngine
  skillStore: SkillStore
  channelPipelineMonitor: ChannelPipelineMonitor
  channelOriginStore?: ChannelOriginStore
  devicePairingRegistry: DevicePairingRegistry
  channelReplayStore?: ChannelReplayStore
  channelSessionStore?: ChannelSessionStore
  approvalRegistry: ApprovalRegistry
  approvalDecisions: ApprovalDecisionStore
  runLimiter: RunLimiter
  activeRuns: ActiveRunRegistry
  /** At-most-one-active-turn-per-session lock shared by WS + HTTP/SSE paths. */
  sessionBusy: SessionBusyRegistry
  approvalCheckpoints: ApprovalCheckpointStore
  runCheckpoints: RunCheckpointStore
  toolExecutions: ToolExecutionStore
  sessionWatchBroker: SessionWatchBroker
  sessionRuntimeSnapshots: SessionRuntimeSnapshotStore
  configWatcher: ConfigWatcher | null
  marketplaceCatalog: MarketplaceCatalog
  skillSourceUrlPolicy: SkillSourceUrlPolicy
  installPipeline: InstallPipeline
  configMutationService: ConfigMutationService
  subagentDispatcher: SubagentDispatcher
  backgroundSubagents?: BackgroundSubagents
  /** Null when the `acp` feature is disabled in this build. */
  externalAcpAgentDispatcher: ExternalAcpAgentDispatcher | null
  /** Null when the `a2a` feature is disabled in this build. */
  a2aTaskStore: A2ATaskStore | null
  swarmRunRegistry: import('../../agent/swarm/index.js').SwarmRunRegistry
  swarmRunStore: import('../../agent/swarm/index.js').SwarmRunStore
  swarmTmuxPool: import('../../agent/swarm/index.js').TmuxSessionPool
  swarmLauncher: import('../../agent/swarm/index.js').AgentLauncher
  swarmAgentRuntime: import('../../agent/swarm/index.js').SwarmAgentRuntimeAdapter
  /**
   * Start a swarm run. Task 18 wires a stub that just calls run.start();
   * Task 20 replaces it with the real engine kickoff (warm pool launch +
   * supervisor agent loop).
   */
  startSwarmRun: (
    run: import('../../agent/swarm/run/swarm-run.js').SwarmRun,
    opts: { warmPool: readonly string[]; autoApproveAgents: boolean; noSupervisor?: boolean },
  ) => Promise<void>
  /**
   * Forward keystrokes / resize to the tmux session backing one agent
   * handle. Task 18 stubs this with `throw 'not implemented'`; Task 20
   * wires it through SwarmTmuxPool.
   */
  forwardSwarmKeys: (
    runId: string,
    handle: string,
    payload: {
      keys?: string
      keyName?: string
      enter?: boolean
      resize?: { cols: number; rows: number }
      principal?: string
    },
  ) => Promise<void>
  /** Capture the current cleaned tmux pane contents for a swarm agent. */
  captureSwarmAgent: (
    runId: string,
    handle: string,
    options?: { lines?: number; raw?: boolean },
  ) => Promise<string>
  /**
   * Stop all warm-pool tmux sessions for a swarm run, immediately. The
   * DELETE route uses this so cancel terminates child agents promptly
   * instead of waiting for the supervisor's next idle step.
   */
  killSwarmRunAgents: (runId: string) => Promise<void>
  /** Drive a swarm agent through send/wait/observe turns without a separate supervisor LLM. */
  driveSwarmAgent: (
    runId: string,
    input: import('../../agent/swarm/index.js').SwarmDriveInput,
  ) => Promise<import('../../agent/swarm/index.js').SwarmDriveResult>
}
