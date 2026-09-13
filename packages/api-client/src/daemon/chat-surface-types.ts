import type {
  DaemonContextUsage,
  DaemonMemoryContextItem,
  DaemonPendingApproval,
  DaemonUsage,
} from './types.js'
import type {
  AgentExecutionPolicy,
  AgentRunContract,
  DebateRoundSummary,
  EditCheckpointSummary,
  PlannerRisk,
  RunStopReason,
  TodoItem,
} from '@sepilotd/core'

export type { RunStopReason }

export type ToolStatus = 'running' | 'success' | 'error' | 'pending'
export type AgentState = 'idle' | 'thinking' | 'acting' | 'observing' | 'done' | 'error'
export type ActivityStatus = 'neutral' | ToolStatus
export type ApprovalState = 'live' | 'stale'
export type ToolRecovery = 'journal' | 'probe'
export type ApprovalDecisionOutcome = 'resumed' | 'recovered' | 'recorded' | 'resolved'

export interface ProviderWaitState {
  providerId?: string
  model: string
  source?: string
  auxiliary: boolean
  startedAt: number
  timeoutMs?: number
}

export interface RecoverApprovalDecisionInput {
  sessionId: string
  prompt: string
  startLabel: string
  startDetail: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'tool' | 'context' | 'todo' | 'system'
  content: string
  /** Session that produced this message, when the stream payload carries it. */
  sessionId?: string
  /** ISO timestamp from the originating session event, when available. */
  timestamp?: string
  branchFromEventIndex?: number
  contextItems?: DaemonMemoryContextItem[]
  citations?: DaemonMemoryContextItem[]
  todoItems?: TodoItem[]
  toolName?: string
  toolInput?: Record<string, unknown>
  toolStatus?: ToolStatus
  toolResult?: string
  /**
   * Unified diff of the file change this tool made (tool_result
   * metadata.editDiff) or would make (approval_request previewDiff), so
   * surfaces can render the actual change instead of a byte-count summary.
   */
  toolDiff?: string
  toolMeta?: string
  toolNeedsApproval?: boolean
  approvalRequestId?: string
  approvalState?: ApprovalState
  resumeAvailable?: boolean
  questionRequestId?: string
  questionPrompt?: string
  questionChoices?: string[]
  questionState?: 'pending' | 'answered'
  personaId?: string
  personaName?: string
  usage?: { inputTokens: number; outputTokens: number; estimatedCost?: number }
}

export interface ActivityItem {
  id: string
  kind:
    | 'state'
    | 'thinking'
    | 'tool'
    | 'approval'
    | 'result'
    | 'usage'
    | 'compact'
    | 'context'
    | 'error'
    | 'subagent'
  label: string
  detail: string
  status: ActivityStatus
  meta?: string
}

export interface StoredSessionState {
  messages: Message[]
  activities: ActivityItem[]
  usage: Message['usage']
  pendingApproval: DaemonPendingApproval | null
  provider: string
  model: string
}

export type TerminalChatStreamFrame =
  | { kind: 'session'; sessionId: string }
  | { kind: 'artifacts'; count: number }
  | { kind: 'context'; items: DaemonMemoryContextItem[] }
  | { kind: 'run_contract'; contract: AgentRunContract }
  | {
      /**
       * Throttled progress snapshot derived from the agent state board.
       * No "met" count for criteria — `AgentAcceptanceCriterion` has no
       * structured met/status field today, so consumers should render a
       * bare total instead of inferring completion from text.
       */
      kind: 'state_board'
      criteriaTotal: number
      planTotal: number
      planDone: number
      todosTotal: number
      todosDone: number
      currentNode?: string
      nodeState?: AgentState
      iteration?: number
      /** Bounded live todo items (daemon caps count and text length). */
      todos?: Array<{ content: string; status: string }>
      /** Bounded live plan steps. */
      plan?: Array<{ title: string; status: string; depth: number }>
    }
  | { kind: 'inline_text_start' }
  | { kind: 'inline_text_end' }
  | { kind: 'text'; text: string }
  | { kind: 'message'; content: string; streamed: boolean }
  | {
      /**
       * Safe LLM-call metadata emitted before a provider request starts.
       * Carries no prompt text; surfaces use it to show that a long run is
       * waiting on a model call instead of appearing frozen.
       */
      kind: 'llm_request'
      turnId: string
      iteration: number
      model: string
      providerId?: string
      source?: string
      startedAt?: number
      timeoutMs?: number
      auxiliary?: boolean
      toolNames: string[]
      traceRef?: string
    }
  | {
      /**
       * Graph node completion trace. The daemon already persists this for
       * diagnostics; terminal surfaces render a compact line so operators can
       * see graph progress without opening the session trace log.
       */
      kind: 'node_trace'
      node: string
      durationMs: number
      nextEdge?: string
    }
  | {
      /**
       * Compact planner working-memory update for terminal surfaces. The TUI
       * renders the full planner panel; terminals get a bounded summary so
       * planning progress is visible without dumping the internal JSON block.
       */
      kind: 'planner_working_memory'
      taskSummary: string
      currentStepTitle?: string
      currentStepRationale?: string
      planTotal: number
      planDone: number
      decisions: number
      risks: PlannerRisk[]
      openAssumptions: number
      abandonedAlternatives: number
    }
  | { kind: 'thinking'; content: string }
  | { kind: 'reasoning_step'; label: string; detail?: string }
  | {
      kind: 'action_progress'
      summary: string
      nextStep: string
      toolNames: string[]
    }
  | {
      kind: 'mode_route_decision'
      chosen: string
      persona?: string
      candidates?: string[]
      reason?: string
      confidence?: number
      fallback: boolean
    }
  | {
      kind: 'router_decision'
      id: string
      mode: string
      persona: string
      skillIds: string[]
      reason: string
      confidence: 'high' | 'medium' | 'low'
      fallback: boolean
    }
  | {
      kind: 'quality_gate_verdict'
      phase: string
      decision: 'pass' | 'retry' | 'incomplete'
      blockingReason?: string
      backtrackCount: number
    }
  | { kind: 'backtrack'; phase: string; reason: string; attempt: number }
  | {
      kind: 'recovery'
      scope: string
      recoveryKind: string
      action: string
      message: string
      recoverable: boolean
    }
  | {
      kind: 'panel_open'
      personas: Array<{ id: string; name: string; description?: string }>
    }
  | { kind: 'panel_turn_start'; personaName: string }
  | { kind: 'panel_turn_complete'; personaName: string; text: string }
  | { kind: 'panel_turn_failed'; personaName: string; error: string }
  | { kind: 'panel_synthesizing'; panelists: number }
  | { kind: 'debate_round'; round: DebateRoundSummary }
  | { kind: 'edit_checkpoint_opened'; checkpoint: EditCheckpointSummary }
  | { kind: 'edit_checkpoint_resolved'; checkpoint: EditCheckpointSummary }
  | {
      kind: 'cowork_plan'
      plan: Array<{ role: string; instruction: string }>
    }
  | { kind: 'cowork_task_start'; role: string; instruction: string }
  | { kind: 'cowork_task_complete'; role: string; instruction: string; result: string }
  | { kind: 'cowork_task_failed'; role: string; instruction: string; error: string }
  | { kind: 'cowork_synthesizing'; summary: string }
  | { kind: 'cowork_discuss_request'; prompt: string; choices?: string[] }
  | { kind: 'cowork_discuss_response'; prompt: string; response: string }
  | {
      /**
       * Bounded nested progress from an isolated subagent dispatched by the
       * parent run. Keeps terminal users aware that delegated work is moving
       * without leaking the full child transcript into the parent stream.
       */
      kind: 'subagent_progress'
      subagentId: string
      label?: string
      detail: string
      failed: boolean
    }
  | {
      kind: 'question_request'
      questionId: string
      prompt: string
      choices?: string[]
      sessionId?: string
    }
  | { kind: 'tool_call'; toolName: string; preview: string }
  | {
      kind: 'approval_request'
      /** Child consent must not switch the active parent conversation. */
      subagentId?: string
      toolName: string
      requestId: string
      /** Pre-formatted `name(args-json)` so the cli can show what is actually
       * being approved without having to re-render arguments. */
      preview: string
      /** Raw arguments — surfaces with richer ui can render them as a tree. */
      input?: Record<string, unknown>
      sessionId?: string
    }
  | {
      /**
       * Emitted when a remembered session/always rule short-circuited the
       * operator prompt. Surfaces render this as a positive
       * "auto-approved by <scope> rule '<pattern>'" line so the operator
       * can tell *why* the tool ran without a prompt — without it,
       * accumulated session rules made the chat shell feel unresponsive
       * to consent ("did I really approve this last time?").
       */
      kind: 'auto_approval'
      toolName: string
      requestId: string
      preview: string
      decision: 'approved' | 'denied'
      pattern: string
      scope: 'session' | 'always' | 'run' | 'session-all'
    }
  | {
      kind: 'tool_result'
      status: 'success' | 'error'
      output: string
      recoveryLabel?: string
      postureLabel?: string
    }
  | {
      kind: 'done'
      usage: DaemonUsage
      /** Structured stop cause; absent from older daemons. */
      stopReason?: RunStopReason
    }
  | {
      kind: 'error'
      message: string
      /** Optional structured error code (e.g. AGENT_INACTIVITY,
       * INTERNAL_ERROR). Surfaces use it to render a friendlier message
       * than the raw daemon error text. */
      code?: string
      /** Machine-readable cause for timeout frames (`stalled`, `awaiting_approval`, ...). */
      reason?: string
      /** The human decision the run was blocked on when a timeout frame fired. */
      pendingDecision?: {
        kind: 'approval' | 'question'
        id: string
        label?: string
        since: number
      }
      /** Structured stop cause so surfaces can offer the next action. */
      stopReason?: RunStopReason
    }
  | { kind: 'state_change'; state: AgentState }
  | {
      /**
       * Daemon collapsed N earlier messages into a summary to free up
       * context window. Surfaces use this to tell the user *why* the
       * running token totals dropped and the assistant suddenly stopped
       * referencing earlier turns directly — without it a long-running
       * cli session looks like the agent silently lost its memory.
       */
      kind: 'context_compact'
      summary: string
      beforeTokens: number
      afterTokens: number
    }
  | {
      /**
       * Graph entered a new named phase (implementation, validation,
       * review, finalize, …). `closedPhase` carries the previous phase's
       * incremental token spend; on the first transition it's absent.
       * Surfaces render this as a status line so operators can see live
       * progress through long runs without polling state.
       */
      kind: 'phase_change'
      enteredPhase: string | null
      closedPhase?: { phase: string; closedTokens: number }
    }
  | {
      /**
       * Post-edit analysis ran and produced structural findings: the
       * files just edited, their forward + reverse callers, and any
       * outstanding LSP diagnostics (with `[caller]`-prefixed entries
       * marking files broken downstream by the edit). Surfaces render
       * this as an activity entry so operators can see the blast
       * radius live, instead of having it disappear into reflectionMemo.
       */
      kind: 'post_edit_findings'
      editedFiles: string[]
      reverseCallers: string[]
      brokenCallers: Array<{ file: string; summary: string }>
      ownDiagnostics: Array<{ file: string; summary: string }>
    }
  | {
      /**
       * A mid-run steering note was queued onto the live run's graph state.
       * Distinct from `steering_consumed`, which fires once the agent loop
       * has actually picked the note up.
       */
      kind: 'steering_ack'
      noteId: string
      steeringKind: 'instruction' | 'question'
      steeringMessage?: string
    }
  | {
      /** The agent loop consumed a previously-queued steering note. */
      kind: 'steering_consumed'
      noteId: string
    }

export interface TerminalChatStreamPresenter {
  handleEvent: (event: import('./types.js').DaemonChatStreamPayload) => TerminalChatStreamFrame[]
  getContent: () => string
}

export interface TerminalChatStreamFrameConsumerOptions {
  onFrame: (frame: TerminalChatStreamFrame) => void
}

export interface TerminalChatStreamFrameConsumer {
  handleEvent: (event: import('./types.js').DaemonChatStreamPayload) => void
  getContent: () => string
}

export interface TerminalChatStreamRendererOptions {
  write: (text: string) => void
}

export interface TerminalChatStreamRenderer {
  handleEvent: (event: import('./types.js').DaemonChatStreamPayload) => void
  getContent: () => string
}

export type MessageListSetter = (next: Message[] | ((prev: Message[]) => Message[])) => void

export type ActivityListSetter = (
  next: ActivityItem[] | ((prev: ActivityItem[]) => ActivityItem[]),
) => void

/**
 * Structured progress snapshot forwarded from `state_board` agent events so
 * surfaces can render live plan/todo/node progress without parsing prose.
 * Field-for-field mirror of the daemon event (core `agent/types.ts`).
 */
export interface StateBoardSnapshot {
  criteriaTotal: number
  planTotal: number
  planDone: number
  todosTotal: number
  todosDone: number
  currentNode?: string
  nodeState?: AgentState
  iteration?: number
  todos?: Array<{ content: string; status: string }>
  plan?: Array<{ title: string; status: string; depth: number }>
}

/** Graph node completion trace forwarded from `node_trace` agent events. */
export interface NodeTraceEntry {
  node: string
  durationMs: number
  nextEdge?: string
}

/**
 * Steering-note lifecycle updates from the live stream. `cancelled` never
 * appears here — cancellation is a REST response, not a stream event — so
 * surfaces reconcile cancels from the `cancelSessionSteering` result.
 */
export type SteeringStreamEvent =
  | { kind: 'ack'; noteId: string; message: string; noteKind: 'instruction' | 'question' }
  | { kind: 'consumed'; noteId: string }

export interface StreamEventControllerBindings {
  setMessages: MessageListSetter
  setActivities: ActivityListSetter
  setAgentState: (state: AgentState) => void
  setStatus: (status: string | null) => void
  /**
   * Live model reasoning belongs in the conversation surface, not in the
   * one-line operational status. Keeping it structured also prevents status
   * truncation from discarding most of a reasoning segment.
   */
  setThinking?: (content: string) => void
  setError: (error: string | null) => void
  /**
   * Structured stop cause of the last terminal event (`done` or a timeout
   * error frame). Surfaces render next-action affordances from it.
   */
  setStopReason?: (reason: RunStopReason | null) => void
  /** Effective daemon-selected permission posture for the current run. */
  setExecutionPolicy?: (policy: AgentExecutionPolicy) => void
  setContextUsage?: (context: DaemonContextUsage) => void
  setProviderWait?: (wait: ProviderWaitState | null) => void
  /**
   * High-frequency progress frames. Deliberately binding-only (no activity
   * push): a long run emits one per graph node and would drown the feed.
   */
  onStateBoard?: (snapshot: StateBoardSnapshot) => void
  onNodeTrace?: (trace: NodeTraceEntry) => void
  onSteeringEvent?: (event: SteeringStreamEvent) => void
}

export interface StreamEventControllerOptions {
  assistantId: string
  createId: () => string
  doneFallbackText: string
  errorFallbackText: string
  errorLabel: string
  maxActivityItems?: number
  approvalResumeAvailable?: boolean
  formatErrorMessage?: (message: string) => string
  onSupersededProgress?: (content: string) => void
  /**
   * Fired when the run reaches `done` with nothing to show — no assistant
   * text, no tool outcome, no progress summary (e.g. every provider attempt
   * failed before the first token). Surfaces use this to flip the turn into
   * an error/retry state instead of leaving a dead-end placeholder bubble.
   */
  onEmptyCompletion?: () => void
  /** Clock injection for deterministic stream-coalescing tests. */
  now?: () => number
  /** Minimum interval between UI updates for token-sized thinking deltas. */
  thinkingUiIntervalMs?: number
}

export type ArtifactListSetter<T extends { id: string }> = (
  next: T[] | ((prev: T[]) => T[]),
) => void

export interface ArtifactEventHandlerParams<T extends { id: string }> {
  createId: () => string
  setArtifacts: ArtifactListSetter<T>
  pushActivity: (item: ActivityItem) => void
}
