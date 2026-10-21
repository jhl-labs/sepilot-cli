import type { AgentEvent, AgentExecutionPolicy, AgentState } from './types.js'
import type { MemoryContextItem } from '../memory/semantic-index.js'
import type { ActiveSkillExecutionPolicy } from '../skill/types.js'

export interface AgentContext {
  sessionId: string
  provider: string
  model: string
  cwd?: string
  workspaceRoot?: string
  workspaceIsolation?: 'policy' | 'strict'
  executionPolicy?: AgentExecutionPolicy
  backgroundJobId?: string
  surface?: string
  roleBotId?: string
  providerAttempt?: {
    attempt: number
    isFallbackPending?: () => boolean
  }
  requestedToolNames?: string[]
  requestAuthorityComplete?: boolean
  writingDocId?: string
  systemPrompt?: string
  executionHandoff?: {
    messages: import('./types.js').Message[]
    usage: import('./types.js').TokenUsage
    iterations: number
  }
  modeControlState?: {
    mode: string
    transferCount: number
    turnMaxIterations: number
    visibleToolNames: string[]
  }
  previousMessages?: import('./types.js').Message[]
  currentUserContent?: import('./types.js').Message['content']
  selectedSkillIds?: string[]
  executionSkillIds?: string[]
  skillToolNames?: string[]
  toolAllowlist?: string[]
  delegationToolAllowlist?: string[]
  skillExecutionPolicies?: ActiveSkillExecutionPolicy[]
  runContract?: import('./types.js').AgentRunContract
  runContractScope?: 'current-turn' | 'previous-turn'
  deadlineAt?: number
  completionRequest?: string
  preferPromptReact?: boolean
  memoryPolicy?: { read: boolean; write: boolean; automaticRead?: boolean; automaticWrite?: boolean }
  memoryQuery?: string
  relevantMemories?: string[]
  relevantContextItems?: MemoryContextItem[]
  primaryAgentId?: string
  autoApprove?: boolean
  requireToolApproval?: boolean
  scopeTags?: string[]
  channelContext?: {
    channel: string
    chatKey: string
    triggerMessageId?: string
  }
}

export interface IAgentEngine {
  run(input: string, context: AgentContext): AsyncIterable<AgentEvent>
  stop(): Promise<void>
  getState(): AgentState
}
