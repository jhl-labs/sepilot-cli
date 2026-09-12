import type { Timestamp, SessionId } from '../types/common.js'
import type { MemoryContextItem } from './semantic-index.js'
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

/**
 * Durable, non-secret reference to a daemon-managed chat upload.
 *
 * Session journals intentionally keep only the opaque file id and display
 * metadata. The backing path and bytes remain inside the authenticated daemon
 * file service, so session history never exposes an absolute local path or
 * inflates JSONL records with base64 payloads.
 */
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
  /**
   * Set when something other than a live chat turn wrote this message — today,
   * a scheduled job reporting into the conversation it was created from. Such a
   * message is a complete answer of its own, not an interim step of the last
   * user turn, so surfaces must not fold it into that turn's execution log.
   */
  origin?: 'scheduler'
}

export interface RouterDecisionEvent extends BaseEvent {
  type: 'router_decision'
  decision: {
    mode: string
    persona: string
    skillIds: string[]
    toolGroups?: string[]
    reason: string
    confidence: 'high' | 'medium' | 'low'
    fallback: boolean
    /** The classifier was not consulted (explicit caller mode or command). */
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
  /** Assistant-text tail captured with the request — audit trail for *why* the tool was proposed. */
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

/**
 * Persisted analogue of the transient `auto_approval` AgentEvent. The
 * agent emits the AgentEvent for the cli to render a visible
 * "auto-approved by <scope> rule" banner; this journal entry captures
 * the same decision durably so a later `sessions show` audit can answer
 * "what tools ran without an explicit prompt, and which remembered rule
 * authorised them?". Without it, an operator reviewing a long session
 * could see tool_call → tool_result pairs with no consent trail and have
 * no way to tell whether each tool was approved interactively, by an
 * always-rule, or by a session-rule that had been registered earlier.
 */
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
  /** Structured stop cause of the run that ended this session turn. */
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

/**
 * Serializable snapshot of the agent state board. Mirrors the daemon's
 * in-memory board (`agent/graph/state-board.ts`) using core-owned types so the
 * board can travel through the append-only session journal, the daemon read
 * endpoint, api-client, and surfaces without any of them importing daemon
 * internals. Assembly stays purely structural — no content parsing.
 */
export interface AgentStateBoardSnapshot {
  /** Verbatim contract goal — never compressed or summarized. */
  goal: string
  /** Verbatim acceptance criteria. */
  completionCriteria: AgentAcceptanceCriterion[]
  /** Full contract for verbatim rendering, when a contract exists. */
  contract?: AgentRunContract
  /** Planner working-memory plan, flattened with hierarchy depth. */
  plan: AgentStateBoardPlanStep[]
  /** Session todo list. */
  todos: TodoItem[]
  /** Planner decisions already taken (1 line each). */
  decisions: string[]
  /** Failed attempts re-injected every turn as "do NOT repeat" input. */
  failedAttempts: AgentFailedAttempt[]
  /** Open questions to verify or escalate rather than guess. */
  openQuestions: AgentOpenQuestion[]
  /** Pre-rendered evidence-ledger section (counts, gaps, validation runs). */
  evidenceSection: string | null
  /**
   * Terminal completion diagnostics accepted by the presentation boundary.
   * These are auditable criterion verdicts, not proof by themselves: a `met`
   * verdict still needs criterion-scoped tool evidence before an evaluation
   * surface may call the criterion supported.
   */
  completion?: AgentStateBoardCompletion
}

export interface AgentStateBoardCriterionVerdict {
  id: string
  verdict: 'met' | 'unmet' | 'not_applicable'
  /** Exact tool-call ids explicitly linked by the accepted criterion protocol. */
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

/**
 * Board change journaled to the append-only session log. Persisting it via
 * `appendEvent` also broadcasts it through `WatchedSessionStore` — persistence
 * and notification are one event (opencode-style), no second store.
 */
export interface StateBoardEvent extends BaseEvent {
  type: 'state_board'
  /** Board snapshot time in epoch ms (distinct from the ISO `timestamp`). */
  at: number
  board: AgentStateBoardSnapshot
}

/**
 * Acknowledgement journaled when a user steering note (mid-run instruction or
 * question) is accepted for an active run. The message is persisted here as
 * durable recovery evidence; the live run's `AgentState.steeringNotes` carries
 * the mutable in-memory copy consumed by the agent loop.
 */
export interface SteeringAckEvent extends BaseEvent {
  type: 'steering_ack'
  noteId: string
  kind: 'instruction' | 'question'
  message: string
}

/**
 * Journaled when the agent loop consumes a queued steering note at an
 * iteration boundary (see daemon `agent/graph/state-board.ts`
 * `takeUnconsumedSteeringNotes`) — the durable record that the note was
 * surfaced to the model, distinct from `SteeringAckEvent` (queued) above.
 */
export interface SteeringConsumedEvent extends BaseEvent {
  type: 'steering_consumed'
  noteId: string
}

/**
 * Journaled when an operator retracts a queued steering note before the agent
 * consumes it. A note id may have exactly one of consumed or cancelled as its
 * terminal state.
 */
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
  /** CLI working directory uses normal tool policy; omitted legacy bindings stay strict. */
  workspaceIsolation?: 'policy' | 'strict'
  starred?: boolean
  messageCount: number
  totalTokens: { input: number; output: number }
  totalCost: number
  tags: string[]
  /**
   * Persona roster bound to this conversation. Order is significant for
   * collaborative turns. An empty array explicitly restores the default
   * assistant, while an omitted field means no roster has been configured.
   */
  personaIds?: string[]
  /** Server-bound isolated persona memory identity. */
  memoryNamespace?: string
  /**
   * Observed transport-flip capability for this session's model: once a run
   * detects the model cannot surface reasoning in native tool-calling mode, the
   * flip to the prompt-react text protocol is persisted here so later turns skip
   * the re-detection cost. Structural capability fact, not a model-name branch.
   */
  preferPromptReact?: boolean
}
