import type { RuntimeServices } from './types.js'

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

export type ConfigMutationCapabilities = Pick<
  RuntimeServices,
  'config' | 'configMutationService' | 'auditLogger'
>

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
  | 'eventRouting'
  | 'eventRoutingProducers'
>

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
  | 'jobStore'
  | 'schedulerEngine'
  | 'parseWhen'
  | 'schedulerDefaultTimezone'
  | 'triggerSchedulerJob'
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
  | 'eventRouting'
  | 'eventRoutingLlmRunner'
  | 'eventRoutingProducers'
>

export type ConfigPersistenceCapabilities = Pick<RuntimeServices, 'config' | 'dataDir'>

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

export type ConfigAutonomyCapabilities = Pick<RuntimeServices, 'config' | 'autonomy' | 'channelAcl'>

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

export type AuditQueryCapabilities = Pick<RuntimeServices, 'auditLogger'>

export type AuditLogCapabilities = Pick<RuntimeServices, 'auditLogger' | 'config'>

export type ChannelSummaryCapabilities = Pick<RuntimeServices, 'config' | 'channels'>

export type ProviderFactoryCapabilities = Pick<RuntimeServices, 'providerFactoryRegistry'>

export type AuthCapabilities = Pick<RuntimeServices, 'extensionTokenStore'>
