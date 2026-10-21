import type { LiveViewRecord } from '@sepilotd/core'
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
export type ApprovalState = 'live' | 'stale' | 'parked'
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
  sessionId?: string
  timestamp?: string
  branchFromEventIndex?: number
  contextItems?: DaemonMemoryContextItem[]
  citations?: DaemonMemoryContextItem[]
  views?: LiveViewRecord[]
  liveViewIds?: string[]
  todoItems?: TodoItem[]
  toolName?: string
  toolInput?: Record<string, unknown>
  toolStatus?: ToolStatus
  toolResult?: string
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
  | {
      kind: 'view_still'
      viewId: string
      title: string
      subtitle?: string
      image: string
      width: number
      height: number
    }
  | { kind: 'session'; sessionId: string }
  | { kind: 'artifacts'; count: number }
  | { kind: 'context'; items: DaemonMemoryContextItem[] }
  | { kind: 'run_contract'; contract: AgentRunContract }
  | {
      kind: 'state_board'
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
  | { kind: 'inline_text_start' }
  | { kind: 'inline_text_end' }
  | { kind: 'text'; text: string }
  | { kind: 'message'; content: string; streamed: boolean }
  | {
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
      kind: 'node_trace'
      node: string
      durationMs: number
      nextEdge?: string
    }
  | {
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
      subagentId?: string
      toolName: string
      requestId: string
      preview: string
      input?: Record<string, unknown>
      sessionId?: string
    }
  | {
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
      stopReason?: RunStopReason
    }
  | {
      kind: 'error'
      message: string
      code?: string
      reason?: string
      pendingDecision?: {
        kind: 'approval' | 'question'
        id: string
        label?: string
        since: number
      }
      stopReason?: RunStopReason
    }
  | { kind: 'state_change'; state: AgentState }
  | {
      kind: 'context_compact'
      summary: string
      beforeTokens: number
      afterTokens: number
    }
  | {
      kind: 'phase_change'
      enteredPhase: string | null
      closedPhase?: { phase: string; closedTokens: number }
    }
  | {
      kind: 'post_edit_findings'
      editedFiles: string[]
      reverseCallers: string[]
      brokenCallers: Array<{ file: string; summary: string }>
      ownDiagnostics: Array<{ file: string; summary: string }>
    }
  | {
      kind: 'steering_ack'
      noteId: string
      steeringKind: 'instruction' | 'question'
      steeringMessage?: string
    }
  | {
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

export interface NodeTraceEntry {
  node: string
  durationMs: number
  nextEdge?: string
}

export type SteeringStreamEvent =
  | { kind: 'ack'; noteId: string; message: string; noteKind: 'instruction' | 'question' }
  | { kind: 'consumed'; noteId: string }

export interface StreamEventControllerBindings {
  setMessages: MessageListSetter
  setActivities: ActivityListSetter
  setAgentState: (state: AgentState) => void
  setStatus: (status: string | null) => void
  setThinking?: (content: string) => void
  setError: (error: string | null) => void
  setStopReason?: (reason: RunStopReason | null) => void
  setExecutionPolicy?: (policy: AgentExecutionPolicy) => void
  setContextUsage?: (context: DaemonContextUsage) => void
  setProviderWait?: (wait: ProviderWaitState | null) => void
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
  onEmptyCompletion?: () => void
  now?: () => number
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
