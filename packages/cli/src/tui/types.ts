import type {
  ActivityItem,
  DaemonAgentMode,
  DaemonArtifact,
  DaemonMemoryContextItem,
  DebateRoundSummary,
  EditCheckpointSummary,
  PlannerWorkingMemory,
  DaemonPendingQuestion,
  DaemonContextUsage,
  ProviderWaitState,
} from '@sepilotd/api-client'

export interface MessageAttachment {
  id?: string
  path: string
  filename: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system' | 'tool'
  /** Presentation-only variant for transient conversation content. */
  variant?: 'thinking'
  content: string
  citations?: DaemonMemoryContextItem[]
  attachments?: MessageAttachment[]
  toolCall?: ToolCallState
  toolCalls?: ToolCallState[]
  timestamp: number
}

export interface ToolCallState {
  id: string
  name: string
  arguments: string
  input: Record<string, unknown>
  status: 'running' | 'success' | 'error' | 'pending'
  superseded?: boolean
  output?: string
  meta?: string
  collapsed: boolean
  approvalRequestId?: string
  approvalState?: 'live' | 'stale'
  resumeAvailable?: boolean
  previousContent?: string | null
  editDiff?: string
}

export interface GraphTraceEntry {
  node: string
  phase?: string | null
  lifecycleState?: string
  iteration?: number
  status: 'running' | 'completed' | 'interrupted' | 'error'
  durationMs?: number
  nextEdge?: string
}

export interface RunWorkProgress {
  criteriaTotal: number
  planTotal: number
  planDone: number
  todosTotal: number
  todosDone: number
  todos?: Array<{ content: string; status: string }>
  plan?: Array<{ title: string; status: string; depth: number }>
}

export interface ApprovalRequest {
  requestId: string
  sessionId?: string
  toolCallId: string
  toolName: string
  input: Record<string, unknown>
  requestedAt?: string
  expiresAt?: string
  phase?: string
  repeatCount?: number
  state: 'live' | 'stale'
  resumeAvailable?: boolean
  suggestedRule?: { tool: string; pattern: string }
  /** Unified diff generated before a file-editing tool is approved. */
  previewDiff?: string
  /** Assistant-text tail explaining why the agent proposed this tool call. */
  context?: string
}

export interface ChatState {
  messages: Message[]
  artifacts: DaemonArtifact[]
  activities: ActivityItem[]
  currentMessage: string
  toolCalls: ToolCallState[]
  isStreaming: boolean
  streamStatus: string | null
  streamStartedAt: number | null
  providerWait?: ProviderWaitState | null
  isThinking: boolean
  thinkingText: string
  isHydratingSession: boolean
  sessionId: string | null
  projectId: string | null
  projectName: string | null
  model: string
  provider: string
  mode: DaemonAgentMode
  thinkingLevel: 'auto' | 'off' | 'low' | 'medium' | 'high' | 'max'
  maxTokens: number | null
  autonomy: 'readonly' | 'accept-edits' | 'workspace-write' | 'supervised' | 'autonomous'
  usage: { input: number; output: number; cost: number }
  /** Latest single provider-request context, distinct from cumulative usage. */
  contextUsage?: DaemonContextUsage | null
  pendingApproval: ApprovalRequest | null
  pendingQuestions: DaemonPendingQuestion[]
  denialFollowup: { toolName: string; at: number } | null
  plannerWorkingMemory?: PlannerWorkingMemory | null
  editRollbacks?: EditCheckpointSummary[]
  debateRounds?: DebateRoundSummary[]
  error: string | null
  /** Current named run phase (implementation/validation/review/finalize), from `phase_change` frames. */
  currentPhase?: string | null
  /** Actual graph traversal for the latest run, including the currently active node. */
  graphTrace?: GraphTraceEntry[]
  /** Last structured plan/todo snapshot; retained after completion for F11 inspection. */
  runWorkProgress?: RunWorkProgress | null
  /** Structured snapshot from the `state_board` stream frame. No text parsing. */
  stateBoardCounts?: RunWorkProgress | null
}

export interface AppConfig {
  url: string
  model?: string
  provider?: string
  sessionId?: string
  resume?: boolean
}
