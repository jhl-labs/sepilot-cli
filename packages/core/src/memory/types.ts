import type { Timestamp, SessionId } from '../types/common.js'
import type { MemoryContextItem } from './semantic-index.js'
import type { LiveViewRecord } from '../views/types.js'
import type {
  DebateRoundSummary,
  EditCheckpointSummary,
  AgentRunContract,
  AgentAcceptanceCriterion,
  AgentFailedAttempt,
  AgentOpenQuestion,
  AgentStateBoardPlanStep,
  Message,
  PlannerWorkingMemory,
  TokenUsage,
  AgentRecoveryAction,
  AgentRecoveryKind,
  AgentRecoveryScope,
  LlmRequestDigest,
  ToolExecutionPosture,
  ToolResultMetadata,
  RunStopReason,
} from '../agent/types.js'
import type {
  ApprovalDecisionStatus,
  ApprovalScope,
  AutoApprovalScope,
} from '../security/approval-decisions.js'

interface BaseEvent {
  id: string
  timestamp: Timestamp
}

export interface SessionStartEvent extends BaseEvent {
  type: 'session_start'
  metadata: { provider: string; model: string; device: string }
}

export interface SessionAttachmentRef {
  fileId: string
  mimeType: string
  filename: string
  size: number
}

export interface UserMessageEvent extends BaseEvent {
  type: 'user_message'
  content: string
  attachments?: SessionAttachmentRef[]
}

export interface AssistantMessageEvent extends BaseEvent {
  type: 'assistant_message'
  content: string
  thinkingLevel?: string
  thinking?: string
  origin?: 'scheduler'
  views?: LiveViewRecord[]
}

export interface RouterDecisionEvent extends BaseEvent {
  type: 'router_decision'
  decision: {
    mode: string
    persona: string
    skillIds: string[]
    memoryPolicy?: { read: boolean; write: boolean; automaticRead?: boolean; automaticWrite?: boolean }
    toolGroups?: string[]
    reason: string
    confidence: 'high' | 'medium' | 'low'
    fallback: boolean
    skipped?: boolean
  }
}

export interface MemoryContextEvent extends BaseEvent {
  type: 'memory_context'
  items: MemoryContextItem[]
}

export interface LlmRequestEvent extends BaseEvent {
  type: 'llm_request'
  turnId: string
  iteration: number
  requestDigest: LlmRequestDigest
}

export interface ModeRouteDecisionEvent extends BaseEvent {
  type: 'mode_route_decision'
  chosen: string
  persona?: string
  candidates?: string[]
  reason?: string
  confidence?: number
  fallback: boolean
}

export interface QualityGateVerdictEvent extends BaseEvent {
  type: 'quality_gate_verdict'
  phase: string
  decision: 'pass' | 'retry' | 'incomplete'
  blockingReason?: string
  backtrackCount: number
}

export interface BacktrackEvent extends BaseEvent {
  type: 'backtrack'
  phase: string
  reason: string
  attempt: number
}

export interface NodeTraceEvent extends BaseEvent {
  type: 'node_trace'
  node: string
  durationMs: number
  nextEdge?: string
}

export interface ToolCallEvent extends BaseEvent {
  type: 'tool_call'
  tool: string
  input: Record<string, unknown>
  status: 'pending' | 'approved' | 'denied' | 'executing'
}

export interface ToolResultEvent extends BaseEvent {
  type: 'tool_result'
  toolCallId: string
  output: string
  status: 'success' | 'error' | 'timeout' | 'cancelled'
  duration_ms: number
  recovery?: 'journal' | 'probe'
  executionPosture?: ToolExecutionPosture
  metadata?: ToolResultMetadata
}

export interface ApprovalRequestEvent extends BaseEvent {
  type: 'approval_request'
  toolCallId: string
  tool: string
  input: Record<string, unknown>
  status: 'pending' | 'resolved'
  context?: string
}

export interface ApprovalResponseEvent extends BaseEvent {
  type: 'approval_response'
  requestId: string
  decision: ApprovalDecisionStatus
  approved: boolean
  note?: string
  approvedBy: string
  scope?: ApprovalScope
}

export interface AutoApprovalEvent extends BaseEvent {
  type: 'auto_approval'
  requestId: string
  toolCallId: string
  tool: string
  input: Record<string, unknown>
  decision: 'approved' | 'denied'
  scope: AutoApprovalScope
  rule: { tool: string; pattern: string }
}

export interface CoworkPlanEvent extends BaseEvent {
  type: 'cowork_plan'
  plan: Array<{ role: string; instruction: string }>
}

export interface CoworkTaskStartEvent extends BaseEvent {
  type: 'cowork_task_start'
  role: string
  instruction: string
}

export interface CoworkTaskCompleteEvent extends BaseEvent {
  type: 'cowork_task_complete'
  role: string
  instruction: string
  result: string
}

export interface CoworkTaskFailedEvent extends BaseEvent {
  type: 'cowork_task_failed'
  role: string
  instruction: string
  error: string
}

export interface CoworkSynthesizingEvent extends BaseEvent {
  type: 'cowork_synthesizing'
  summary: string
}

export interface CoworkDiscussRequestEvent extends BaseEvent {
  type: 'cowork_discuss_request'
  prompt: string
  choices?: string[]
}

export interface CoworkDiscussResponseEvent extends BaseEvent {
  type: 'cowork_discuss_response'
  prompt: string
  response: string
}

export interface ContextCompactEvent extends BaseEvent {
  type: 'context_compact'
  beforeTokens: number
  afterTokens: number
  summary: string
  strategy?: 'preserve_tail' | 'summary_only'
  removedMessageCount?: number
  preservedMessageCount?: number
  preservedMessages?: Message[]
  throughEventId?: string
}

export interface MemorySummaryEvent extends BaseEvent {
  type: 'memory_summary'
  source: string
  stage: 'turn' | 'session_end'
  turnId?: string
  lightCaptured: boolean
  semanticMemoriesExtracted: number
  ragContextPromotions?: number
  ragPromotedContextIds?: string[]
}

export interface ProviderAttemptEvent extends BaseEvent {
  type: 'provider_attempt'
  provider: string
  model: string
  source: 'request_provider' | 'session' | 'model_router' | 'default_provider'
  selectionReason?: string
  rank: number
  attempt: number
  status: 'started' | 'failed' | 'succeeded'
  retryable?: boolean
  errorCode?: string
  errorMessage?: string
  nextProvider?: string
  nextModel?: string
}

export interface RecoveryEvent extends BaseEvent {
  type: 'recovery'
  scope: AgentRecoveryScope
  kind: AgentRecoveryKind
  action: AgentRecoveryAction
  message: string
  recoverable: boolean
  details?: Record<string, unknown>
}

export interface RunContractEvent extends BaseEvent {
  type: 'run_contract'
  contract: AgentRunContract
}

export interface DelegationStateEvent extends BaseEvent {
  type: 'delegation_state'
  delegationId: string
  targetDevice: string
  claimHealth: 'healthy' | 'degraded' | 'lost'
  startedAt: string
  detail: string
  degradedSince?: string
  lastHeartbeatAt?: string
  lastError?: string
  source?: 'gateway' | 'comments' | 'transport'
}

export interface DelegationResultEvent extends BaseEvent {
  type: 'delegation_result'
  delegationId: string
  targetDevice: string
  executionId?: string
  status: 'completed' | 'failed' | 'timeout' | 'cancelled'
  result?: string
  artifactHandles?: string[]
  source: 'recovery'
}

export interface SessionEndEvent extends BaseEvent {
  type: 'session_end'
  totalTokens: { input: number; output: number }
  totalCost: number
  duration_ms: number
  stopReason?: RunStopReason
}

export type TodoStatus = 'pending' | 'in_progress' | 'completed' | 'blocked' | 'cancelled'

export interface TodoItem {
  id: string
  content: string
  status: TodoStatus
}

export interface TodoListEvent extends BaseEvent {
  type: 'todo_list'
  items: TodoItem[]
}

export interface EditCheckpointOpenedEvent extends BaseEvent {
  type: 'edit_checkpoint_opened'
  checkpoint: EditCheckpointSummary
}

export interface EditCheckpointResolvedEvent extends BaseEvent {
  type: 'edit_checkpoint_resolved'
  checkpoint: EditCheckpointSummary
}

export interface DebateRoundEvent extends BaseEvent {
  type: 'debate_round'
  round: DebateRoundSummary
}

export interface PlannerWorkingMemoryUpdatedEvent extends BaseEvent {
  type: 'planner_working_memory_updated'
  workingMemory: PlannerWorkingMemory
}

export interface PhaseChangeEvent extends BaseEvent {
  type: 'phase_change'
  enteredPhase: string | null
  closedPhase?: { phase: string; usage: TokenUsage }
  phaseUsages?: Record<string, TokenUsage>
}

export interface PostEditFindingsEvent extends BaseEvent {
  type: 'post_edit_findings'
  editedFiles: string[]
  impactedExternalModules: string[]
  impactedLocalModules: string[]
  reverseCallers: string[]
  diagnostics: Array<{ file: string; summary: string }>
  analyzedAt: string
}

export interface PanelOpenEvent extends BaseEvent {
  type: 'panel_open'
  personas: Array<{ id: string; name: string; description?: string }>
}

export interface PanelTurnCompleteEvent extends BaseEvent {
  type: 'panel_turn_complete'
  personaId: string
  personaName: string
  text: string
}

export interface PanelTurnFailedEvent extends BaseEvent {
  type: 'panel_turn_failed'
  personaId: string
  personaName: string
  error: string
}

export interface PanelSynthesizingEvent extends BaseEvent {
  type: 'panel_synthesizing'
  panelists: number
}

export interface AgentStateBoardSnapshot {
  goal: string
  completionCriteria: AgentAcceptanceCriterion[]
  contract?: AgentRunContract
  plan: AgentStateBoardPlanStep[]
  todos: TodoItem[]
  decisions: string[]
  failedAttempts: AgentFailedAttempt[]
  openQuestions: AgentOpenQuestion[]
  evidenceSection: string | null
  completion?: AgentStateBoardCompletion
}

export interface AgentStateBoardCriterionVerdict {
  id: string
  verdict: 'met' | 'unmet' | 'not_applicable'
  evidenceToolCallIds?: string[]
}

export interface AgentStateBoardCompletion {
  criterionVerdicts: AgentStateBoardCriterionVerdict[]
  gate?: {
    decision: 'pass' | 'block'
    unmet: string[]
    reason?: string
    budgetExhausted?: boolean
  }
}

export interface StateBoardEvent extends BaseEvent {
  type: 'state_board'
  at: number
  board: AgentStateBoardSnapshot
}

export interface SteeringAckEvent extends BaseEvent {
  type: 'steering_ack'
  noteId: string
  kind: 'instruction' | 'question'
  message: string
}

export interface SteeringConsumedEvent extends BaseEvent {
  type: 'steering_consumed'
  noteId: string
}

export interface SteeringCancelledEvent extends BaseEvent {
  type: 'steering_cancelled'
  noteId: string
}

export type SessionEvent =
  | SessionStartEvent
  | UserMessageEvent
  | AssistantMessageEvent
  | RouterDecisionEvent
  | MemoryContextEvent
  | LlmRequestEvent
  | ModeRouteDecisionEvent
  | QualityGateVerdictEvent
  | BacktrackEvent
  | NodeTraceEvent
  | ToolCallEvent
  | ToolResultEvent
  | ApprovalRequestEvent
  | ApprovalResponseEvent
  | AutoApprovalEvent
  | CoworkPlanEvent
  | CoworkTaskStartEvent
  | CoworkTaskCompleteEvent
  | CoworkTaskFailedEvent
  | CoworkSynthesizingEvent
  | CoworkDiscussRequestEvent
  | CoworkDiscussResponseEvent
  | ContextCompactEvent
  | MemorySummaryEvent
  | ProviderAttemptEvent
  | RecoveryEvent
  | RunContractEvent
  | DelegationStateEvent
  | DelegationResultEvent
  | SessionEndEvent
  | TodoListEvent
  | EditCheckpointOpenedEvent
  | EditCheckpointResolvedEvent
  | DebateRoundEvent
  | PlannerWorkingMemoryUpdatedEvent
  | PhaseChangeEvent
  | PostEditFindingsEvent
  | PanelOpenEvent
  | PanelTurnCompleteEvent
  | PanelTurnFailedEvent
  | PanelSynthesizingEvent
  | StateBoardEvent
  | SteeringAckEvent
  | SteeringConsumedEvent
  | SteeringCancelledEvent

export interface SessionMeta {
  id: SessionId
  title: string
  createdAt: Timestamp
  updatedAt: Timestamp
  provider: string
  model: string
  device: string
  status: 'active' | 'completed' | 'abandoned'
  cwd?: string
  workspaceIsolation?: 'policy' | 'strict'
  starred?: boolean
  messageCount: number
  totalTokens: { input: number; output: number }
  totalCost: number
  tags: string[]
  personaIds?: string[]
  memoryNamespace?: string
  preferPromptReact?: boolean
}
