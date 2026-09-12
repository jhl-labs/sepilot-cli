import type { AgentEvent, AgentExecutionPolicy, AgentState } from './types.js'
import type { MemoryContextItem } from '../memory/semantic-index.js'
import type { ActiveSkillExecutionPolicy } from '../skill/types.js'

export interface AgentContext {
  sessionId: string
  provider: string
  model: string
  cwd?: string
  /**
   * Canonical filesystem capability root for this turn. When present, every
   * path-aware tool must remain inside this directory regardless of autonomy
   * or human approval. `cwd` is only the working directory and may never
   * widen this boundary.
   */
  workspaceRoot?: string
  /** Server-bound workspace profile, preserved across approval and crash resumes. */
  workspaceIsolation?: 'policy' | 'strict'
  /**
   * Server-resolved permission context for this turn. Surfaces must display
   * the effective value rather than assuming their requested autonomy won the
   * daemon-side downgrade clamp.
   */
  executionPolicy?: AgentExecutionPolicy
  /** Detached chat job that owns this run, when the turn was queued through
   * the background-chat transport. Propagated to completion hooks so external
   * observers can correlate the final outcome without scraping session logs. */
  backgroundJobId?: string
  /** Originating user surface (cli, desktop, mobile, web, ...), when known. */
  surface?: string
  /**
   * Desktop writing-canvas document bound to this turn. When present, doc.*
   * tools must target this document instead of any process-wide active doc.
   */
  writingDocId?: string
  systemPrompt?: string
  /** Runtime-only continuation within the same user turn; preserves evidence and spent budget. */
  executionHandoff?: {
    messages: import('./types.js').Message[]
    usage: import('./types.js').TokenUsage
    iterations: number
  }
  /** Runtime checkpoint of execution strategy; never grants tool authority. */
  modeControlState?: {
    mode: string
    transferCount: number
    turnMaxIterations: number
    visibleToolNames: string[]
  }
  previousMessages?: import('./types.js').Message[]
  /**
   * Structured content for the current user turn when the transport supplied
   * attachments. `run()` still receives the user's primary instruction as a
   * string so routing, contracts, and completion policy cannot be polluted by
   * attachment text; engines use this field only when constructing the model
   * message that carries the attachment context.
   */
  currentUserContent?: import('./types.js').Message['content']
  /**
   * Stable ids of every canonical built-in skill explicitly loaded for this
   * turn. Routing uses the complete set so one policy-bearing skill cannot
   * hide a second selected workflow that has no deterministic stage policy.
   */
  selectedSkillIds?: string[]
  /**
   * Canonical selected skill ids that carry trusted deterministic execution
   * policies. This is intentionally a subset of selectedSkillIds.
   */
  executionSkillIds?: string[]
  /** Tool declarations resolved only from skills actually loaded this turn. */
  skillToolNames?: string[]
  /** Authorized tool ceiling for this turn, independent of temporary mode visibility. */
  toolAllowlist?: string[]
  /** Immutable runtime constraints resolved from trusted loaded skills. */
  skillExecutionPolicies?: ActiveSkillExecutionPolicy[]
  /**
   * Runtime-generated task contract for durable agent execution. Unlike a
   * graph-specific plan, this can be attached to any agent mode so the model
   * has explicit acceptance criteria and continuation rules before it starts
   * using tools.
   */
  runContract?: import('./types.js').AgentRunContract
  /**
   * Whether runContract was restored from an earlier user turn or was created
   * for the request currently being dispatched. Previous-turn contracts are
   * inherited only by explicit action continuations.
   */
  runContractScope?: 'current-turn' | 'previous-turn'
  /**
   * Persisted transport-flip capability for the session model (see
   * SessionMeta.preferPromptReact). Seeds the initial graph state so a
   * known-bad-native model does not re-detect the flip every turn.
   */
  preferPromptReact?: boolean
  memoryQuery?: string
  relevantMemories?: string[]
  relevantContextItems?: MemoryContextItem[]
  primaryAgentId?: string
  autoApprove?: boolean
  /**
   * Per-turn human-in-the-loop guard. When true, every policy-allowed
   * side-effecting tool must receive a fresh explicit approval even if the
   * configured autonomy or CI auto-approve setting would normally allow it.
   * Read-only tools and policy denials keep their existing behavior.
   */
  requireToolApproval?: boolean
  /**
   * Canonical scope tags for the active session/sender (e.g. "scope:user:42",
   * "scope:channel:telegram:99"). Memory-aware tools attach these to new
   * memories and filter retrieval/audit results by them so multi-user setups
   * don't leak. Empty/undefined means global behavior.
   */
  scopeTags?: string[]
  /**
   * When the agent is invoked from a channel pipeline (telegram/cli/web/...),
   * carries the originating channel info so tools can schedule follow-up
   * replies back to the same chat. Undefined in REST/internal contexts.
   */
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
