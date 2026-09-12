export * from './http.js'
export * from './desktop.js'
export * from './apps.js'
export * from './app-references.js'
export * from './provider-discovery.js'
export { createJobsClient } from './jobs.js'
export type {
  JobsClient,
  JobsClientDeps,
  JobsBatchSubmitRequest,
  JobsBatchSubmitResult,
  JobSnapshot,
  JobItem,
  JobItemsPage,
} from './jobs.js'
export { createMigrationClient } from './migration.js'
export type {
  MigrationClient,
  MigrationClientDeps,
  MigrationRunRequest,
  MigrationRunResult,
  MigrationSnapshot,
  MigrationReport,
  MigrationStepProgress,
  MigrationStepError,
} from './migration.js'
export * from './daemon/http.js'
export * from './daemon/system-scope.js'
export * from './daemon/chat-surface.js'
export * from './daemon/chat-metrics.js'
export * from './daemon/session-evidence-surface.js'
export * from './daemon/chat-skill-selection.js'
export * from './daemon/scheduled-agent-profile.js'
export * from './daemon/scheduled-app-sources.js'
export * from './daemon/scheduler-misfire.js'
export * from './daemon/chat-transport.js'
export * from './daemon/token-speed.js'
export * from './daemon/health-surface.js'
export * from './daemon/notification-surface.js'
export * from './daemon/visible-web-surface.js'
export * from './daemon/scheduler-delivery-surface.js'
export * from './daemon/scheduler-run-evidence.js'
export * from './daemon/image-canvas.js'
export * from './daemon/types.js'
export * from './daemon/stream.js'
export * from './daemon/doc.js'
export * from './acp/protocol.js'
export * from './gateway/http.js'
export * from './gateway/watch-surface.js'
export type {
  ToolCall,
  SessionEvent,
  SessionStartEvent,
  SessionAttachmentRef,
  UserMessageEvent,
  AssistantMessageEvent,
  MemoryContextEvent,
  ToolCallEvent,
  ToolResultEvent,
  ApprovalRequestEvent,
  ApprovalResponseEvent,
  ContextCompactEvent,
  DelegationStateEvent,
  DelegationResultEvent,
  SessionEndEvent,
  SessionMeta,
} from '@sepilotd/core'
export type {
  EditCheckpointSummary,
  DebateDecision,
  DebateRoundSummary,
  PlannerWorkingMemory,
  PlannerHierarchicalStep,
  PlannerStepStatus,
} from '@sepilotd/core'
// Surfaces can't import @sepilotd/core directly (CLAUDE.md
// dependency rule 7), so re-export the staleness predicate the cli
// formatter and chat shell both rely on. Keeps cleanup logic in
// lockstep with the daemon's clear({stale}) implementation.
export { STALE_RULE_THRESHOLD_DAYS, isStaleRememberedDecision } from '@sepilotd/core'
export * from './daemon/swarm.js'
export type {
  DesktopAgentId,
  DesktopAgentState,
  DesktopAgentEvent,
  DesktopAgentSession,
  DesktopAgentSshTarget,
} from './daemon/desktop-agent.js'
