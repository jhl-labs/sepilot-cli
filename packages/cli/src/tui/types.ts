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
import type { TurnOutcome } from './utils/turn-outcome.js'

export interface MessageAttachment {
  id?: string
  path: string
  filename: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system' | 'tool'
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
  approvalState?: 'live' | 'stale' | 'parked'
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
  state: 'live' | 'stale' | 'parked'
  resumeAvailable?: boolean
  suggestedRule?: { tool: string; pattern: string }
  previewDiff?: string
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
  bot?: string | null
  routedBot?: string | null
  thinkingLevel: 'auto' | 'off' | 'low' | 'medium' | 'high' | 'max'
  thinkingLevelChosen?: boolean
  maxTokens: number | null
  autonomy: 'readonly' | 'accept-edits' | 'workspace-write' | 'supervised' | 'autonomous'
  usage: { input: number; output: number; cost: number }
  contextUsage?: DaemonContextUsage | null
  tokensPerSecond?: number | null
  pendingApproval: ApprovalRequest | null
  pendingQuestions: DaemonPendingQuestion[]
  denialFollowup: { toolName: string; at: number } | null
  plannerWorkingMemory?: PlannerWorkingMemory | null
  editRollbacks?: EditCheckpointSummary[]
  debateRounds?: DebateRoundSummary[]
  error: string | null
  turnOutcome?: TurnOutcome | null
  currentPhase?: string | null
  graphTrace?: GraphTraceEntry[]
  runWorkProgress?: RunWorkProgress | null
  stateBoardCounts?: RunWorkProgress | null
}

export interface AppConfig {
  url: string
  model?: string
  provider?: string
  sessionId?: string
  resume?: boolean
}
