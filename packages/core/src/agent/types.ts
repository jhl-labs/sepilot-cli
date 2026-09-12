import type { ApiError } from '../types/errors.js'
import type { MemoryContextItem } from '../memory/semantic-index.js'
import type { ApprovalDecisionStatus, ApprovalRule } from '../security/approval-decisions.js'

export enum AutonomyLevel {
  ReadOnly = 'readonly',
  AcceptEdits = 'accept-edits',
  WorkspaceWrite = 'workspace-write',
  Supervised = 'supervised',
  Autonomous = 'autonomous',
}

export enum ThinkingLevel {
  Off = 'off',
  Low = 'low',
  Medium = 'medium',
  High = 'high',
  Max = 'max',
}

export type MessageRole = 'system' | 'user' | 'assistant' | 'tool'

export type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'image'; source: ImageSource }
  | { type: 'document'; source: DocumentSource }

export interface ImageSource {
  type: 'base64' | 'url'
  mediaType: string
  data: string
}

export interface DocumentSource {
  type: 'base64' | 'url'
  mediaType: string
  data: string
}

export interface Message {
  role: MessageRole
  content: string | ContentPart[]
  toolCallId?: string
  toolCalls?: ToolCall[]
  /**
   * The original tool/function name this message relates to. Set on `tool`
   * result messages so providers that correlate results by function name
   * (Gemini's `functionResponse.name`) can restore the real name instead of
   * the synthetic UUID kept in `toolCallId`. Optional and only meaningful for
   * the `tool` role; providers that correlate by id ignore it.
   */
  name?: string
  /**
   * Free-form structural metadata. Used e.g. to tag recurring system
   * reminders with `reminderKind` so stale duplicates can be superseded
   * by structure (tag + order) without content matching.
   */
  metadata?: Record<string, unknown>
}

export interface ToolDefinition {
  name: string
  description: string
  inputSchema: Record<string, unknown>
}

/**
 * Security-relevant effect of a tool invocation.
 *
 * This is deliberately about the world the invocation can change, not the
 * English verb in the tool name. `internal-state` covers bounded agent-owned
 * bookkeeping such as todos/questions; it must not be treated like a user
 * workspace or remote-system mutation. `dynamic` tools need an input-aware
 * classifier (for example terminal.run is observational when a fail-closed
 * read-only workspace sandbox is part of the invocation context, but a host
 * process action without that capability is not).
 */
export type ToolSecurityEffect =
  | 'observe'
  | 'internal-state'
  | 'workspace-write'
  | 'external-write'
  | 'process-lifecycle'
  | 'dynamic'
  | 'unknown'

export interface ToolSecurityDescriptor {
  effect: ToolSecurityEffect
  /** Why this classification is safe to rely on during policy review. */
  rationale?: string
  /**
   * Whether the tool's result must be observed before a later workspace edit
   * from the same model response may run. Omitted means `required`, which is
   * the conservative default for built-ins and plugins.
   */
  mutationResultBoundary?: 'required' | 'commutative'
}

/** Effective permission context selected by the daemon for one run. */
export interface AgentExecutionPolicy {
  requestedAutonomy?: AutonomyLevel
  configuredAutonomy: AutonomyLevel
  effectiveAutonomy: AutonomyLevel
  clamped: boolean
  clampReason?: string
  agentMode: string
  primaryAgentId?: string
  workspaceBoundary: 'strict' | 'unrestricted'
  freshApprovalRequired: boolean
}

export interface ToolCall {
  id: string
  name: string
  arguments: Record<string, unknown>
}

export interface AgentAcceptanceCriterion {
  id: string
  text: string
}

export interface AgentRequiredArtifact {
  path: string
  kind: 'file' | 'directory' | 'document' | 'other' | (string & {})
  description?: string
}

export interface AgentEvidenceRequirement {
  kind: 'source' | 'repository' | 'artifact' | 'validation' | 'other' | (string & {})
  description: string
  /** Minimum successful observed source calls, independent of filesystem paths. */
  minSourceObservations?: number
  minSourceFiles?: number
  minSourceScopes?: number
  /** Exact observation tool names that may support this requirement. */
  sourceToolNames?: string[]
  requiresArtifactEvidenceMap?: boolean
  requiresArtifactSelfReview?: boolean
  requiresSearch?: boolean
}

export interface AgentArtifactSection {
  id: string
  title: string
  description?: string
  artifactPath?: string
  required?: boolean
}

export type AgentExecutionIntentKind =
  | 'operational-action'
  | 'workspace-change'
  | 'inspection'
  | 'artifact-production'
  | 'conversation'

export type AgentWorkspaceMutationIntent = 'forbidden' | 'allowed' | 'required'

export type AgentExecutionCapability =
  | 'process'
  | 'service'
  | 'terminal'
  | 'browser'
  | 'filesystem-read'
  | 'filesystem-write'
  | 'network'
  /** Durable application state (preferences, tasks, remote records), not workspace files. */
  | 'application-state'

/**
 * A single argv-safe terminal invocation copied from the user's request.
 * Keeping this structured prevents a model from silently adding flags,
 * commands, pipes, or unrelated discovery steps to an explicitly bounded
 * "run this once" turn.
 */
export interface AgentRequestedTerminalCommand {
  executable: string
  args: string[]
}

/**
 * A managed process start copied from an explicitly bounded user request.
 * This is separate from a terminal command because background lifecycle,
 * working directory, and managed-loopback reachability are semantic parts of
 * the requested action, not argv that may be silently discarded.
 */
export interface AgentRequestedProcessStart extends AgentRequestedTerminalCommand {
  cwd?: string
  /** Exact bounded lifetime requested for process.start, in milliseconds. */
  ttlMs?: number
  lifetime?: 'bounded' | 'session'
  network?: 'none' | {
    mode: 'none' | 'loopback'
    ports?: number[]
  }
}

/**
 * Semantic execution posture chosen by the contract planner. This is kept
 * separate from acceptance criteria so graph routing and tool scoping can use
 * structured data rather than re-interpreting prompt wording.
 */
export interface AgentExecutionIntent {
  kind: AgentExecutionIntentKind
  workspaceMutation: AgentWorkspaceMutationIntent
  capabilities: AgentExecutionCapability[]
  /** A semantic router declared this capability list as the turn's upper bound. */
  capabilityPolicy?: 'closed'
  /**
   * Exact tool surface explicitly authorized by the user. The contract
   * planner and grounding auditor own this semantic decision; runtime nodes
   * only enforce the structured boundary uniformly.
   */
  allowedTools?: string[]
  /** Ordered tool boundary chosen semantically when the user requires order. */
  toolSequence?: string[]
  /** Explicit user policy for whether a failed or blocked action may be retried. */
  retryPolicy?: 'forbidden'
  /** User-named workspace targets that the semantic router explicitly authorizes for mutation. */
  authorizedWriteTargets?: string[]
  /** User-named workspace targets that the semantic router explicitly protects from mutation. */
  protectedWriteTargets?: string[]
  requestedTerminalCommand?: AgentRequestedTerminalCommand
  requestedProcessStart?: AgentRequestedProcessStart
  /**
   * Exact process.start options embedded in a broader multi-tool workflow.
   * Unlike requestedProcessStart this does not collapse the turn to one
   * deterministic launch; it only prevents a recovery turn from silently
   * changing the executable, argv, cwd, lifecycle, or network capability.
   */
  constrainedProcessStart?: AgentRequestedProcessStart
}

export interface AgentRunContract {
  summary: string
  acceptanceCriteria: AgentAcceptanceCriterion[]
  constraints: string[]
  outOfScope: string[]
  requiredArtifacts?: AgentRequiredArtifact[]
  evidenceRequirements?: AgentEvidenceRequirement[]
  artifactSections?: AgentArtifactSection[]
  executionIntent?: AgentExecutionIntent
  source: 'planner' | 'fallback'
}

/**
 * A structurally-identified action that already failed a run. `signature` is
 * the stuck-tool-repeat structural signature (tool name + stable-serialized
 * arguments) — re-running "the same action" is decided deterministically, not
 * by content similarity. Serializable mirror of the daemon state-board type so
 * durable session events and surfaces can carry it without importing daemon.
 */
export interface AgentFailedAttempt {
  signature: string
  tool: string
  reason: string
  ts: number
}

/**
 * A question the agent surfaced as unresolved. Blocking questions are escalated
 * to the human when progress stalls instead of letting the agent guess.
 */
export interface AgentOpenQuestion {
  id: string
  text: string
  blocking: boolean
  askedAt?: number
}

/** One planner working-memory plan step, flattened with hierarchy depth. */
export interface AgentStateBoardPlanStep {
  id: string
  title: string
  status: string
  depth: number
}

export interface ToolExecutionPosture {
  sandbox: {
    requested: boolean
    active: boolean
    mode: 'host' | 'docker' | 'ssh' | 'unknown' | (string & {})
    fallbackReason?: string
  }
  filesystem: {
    boundary:
      | 'working-directory'
      | 'session_cwd'
      | 'process_cwd'
      | 'workspace_policy'
      | 'tool_specific'
      | 'unknown'
      | (string & {})
    cwd?: string
    isolated: boolean
    /** True when the active filesystem boundary prevents workspace writes. */
    readOnly?: boolean
    note?: string
  }
  network: {
    isolated: boolean
    mode: 'host' | 'none' | 'bridge' | 'disabled' | 'unknown' | (string & {})
  }
}

export interface ToolResultMetadata {
  /** Executor-attested prerequisite that only the user can satisfy. */
  userActionRequired?: string
  executionPosture?: ToolExecutionPosture
  [key: string]: unknown
}

export interface ChatRequest {
  model: string
  messages: Message[]
  systemPrompt?: string
  tools?: ToolDefinition[]
  /**
   * Provider-neutral tool selection posture. `required` is reserved for
   * bounded control turns whose exposed tools include every valid outcome
   * (including an explicit stop/block transition), so the model cannot spend
   * the turn on unexecutable prose instead of selecting a state transition.
   */
  toolChoice?: 'auto' | 'required' | 'none'
  temperature?: number
  maxTokens?: number
  thinkingLevel?: ThinkingLevel
  stopSequences?: string[]
  timeoutMs?: number
}

export interface TokenUsage {
  inputTokens: number
  outputTokens: number
  thinkingTokens?: number
  cacheReadTokens?: number
  cacheCreationTokens?: number
  estimatedCost?: number
}

/**
 * Input-context pressure for one concrete provider request. This is distinct
 * from TokenUsage, which is accumulated across every LLM call in an agent run
 * for billing and throughput reporting.
 */
export interface AgentContextUsage {
  inputTokens: number
  contextWindowTokens?: number
  reservedOutputTokens?: number
  iteration: number
  source: 'estimated' | 'provider'
}

export interface ChatResponse {
  message: Message
  thinking?: string
  usage: TokenUsage
  finishReason: 'stop' | 'length' | 'tool_use' | 'content_filter'
  /**
   * Why a non-`stop` turn ended, when the coarse `finishReason` cannot say.
   * `truncated_output` is a real output-cap hit: a prefix exists and can be
   * continued. `incomplete_stream` is a transport failure — the stream ended
   * with no terminal event, so there is no trustworthy resumption point and
   * the request should be re-issued rather than "continued". Both surface as
   * `finishReason: 'length'` so existing consumers keep their behaviour.
   */
  finishDetail?: 'truncated_output' | 'incomplete_stream'
  raw?: Record<string, unknown>
}

export interface LlmRequestDigest {
  model: string
  providerId?: string
  source?: string
  startedAt?: number
  timeoutMs?: number
  auxiliary?: boolean
  messageCount: number
  systemPromptChars: number
  /**
   * Serialized size of the tool schemas carried beside the messages. Native
   * tool transport keeps schemas out of the system prompt, so
   * `systemPromptChars` alone understates the real fixed prompt overhead there;
   * compaction thresholds need both halves.
   */
  toolSchemaChars?: number
  toolNames: string[]
  traceRef?: string
}

export type StreamChunk =
  | { type: 'text'; text: string }
  | { type: 'thinking'; text: string }
  | { type: 'tool_call_start'; toolCall: Partial<ToolCall> }
  | { type: 'tool_call_delta'; toolCallId: string; delta: string }
  | { type: 'tool_call_end'; toolCallId: string }
  | { type: 'usage'; usage: TokenUsage }
  | { type: 'done'; finishReason: ChatResponse['finishReason'] }
  | { type: 'error'; error: ApiError }

export type ModelToolTransport = 'auto' | 'native' | 'prompt-react' | 'adaptive'
export type ModelAnswerProtocol = 'auto' | 'repair-left-truncated-answer-stem'

/**
 * Operator-owned compatibility policy for one provider/model pair.
 *
 * Provider endpoints can serve the same model with different protocol
 * behaviour, so this profile is deliberately attached to the resolved
 * ModelInfo rather than inferred from a model family name. Notes preserve the
 * evidence/rationale behind an override; they do not change runtime behaviour.
 */
export interface ModelCompatibilityProfile {
  toolTransport?: ModelToolTransport
  /**
   * Evidence-scoped repair for endpoints that reproducibly remove an
   * arbitrary left prefix from the required `ANSWER:` final-response stem.
   * This is presentation protocol compatibility, not a model capability.
   */
  answerProtocol?: ModelAnswerProtocol
  notes?: string[]
}

export interface ModelInfo {
  id: string
  name: string
  contextWindow: number
  maxOutputTokens: number
  capabilities: {
    vision: boolean
    toolUse: boolean
    streaming: boolean
    embedding: boolean
    thinking: boolean
    /**
     * Operator-declared request dialect for controlling a reasoning model on
     * an OpenAI-compatible endpoint. Compatible servers do not expose this
     * reliably, so the runtime must not guess from provider or model names.
     */
    thinkingControl?: 'reasoning-effort' | 'chat-template-kwargs'
    /**
     * Legacy compatibility knob for unreliable native function calling.
     * New configuration should use
     * `ModelInfo.compatibility.toolTransport = 'adaptive'`, which separates
     * transport policy from factual capabilities. Retained so older configs
     * keep their behaviour without a migration.
     */
    adaptivePromptReact?: boolean
    /**
     * Operator-declared tool transport preference. The model can use tools,
     * but its configured OpenAI-compatible endpoint is more reliable when
     * tool definitions/results are encoded in the portable prompt protocol
     * instead of native function-calling fields.
     *
     * This is deliberately independent from `toolUse`: prompt-react still
     * executes the same structured tools. `adaptivePromptReact` remains the
     * native-first recovery mode; this flag starts prompt-react immediately.
     */
    promptReactPreferred?: boolean
    /**
     * Run the coder graph's pre-implement deep analysis (codebase exploration +
     * LLM planner) for this model even on plain code fixes. Off by default: code
     * fixes go lean, which strong models do best (the scaffolding makes them
     * over-deliberate). Weaker models instead benefit from the scaffolding to
     * structure the work (validated on SWE-bench: nemotron-3-ultra lean 20 ->
     * deep 23, while strong qwen models are higher lean). Set it for models the
     * operator judges weak; a per-model capability fact, not a model-name branch.
     */
    deepCoderAnalysis?: boolean
  }
  compatibility?: ModelCompatibilityProfile
  inputCostPer1k?: number
  outputCostPer1k?: number
}

export type AgentState = 'idle' | 'thinking' | 'acting' | 'observing' | 'done' | 'error'

export type AgentRecoveryScope =
  | 'provider_protocol'
  | 'tool_execution'
  | 'output_synthesis'
  | 'session_resume'
  | (string & {})

export type AgentRecoveryKind =
  | 'missing_tool_result'
  | 'thinking_block_order'
  | 'thinking_disabled_violation'
  | 'empty_message_content'
  | 'journal_replay'
  | 'probe_recovery'
  | 'synthetic_final_message'
  | (string & {})

export type AgentRecoveryAction =
  | 'retry_sanitized_messages'
  | 'retry_sanitized_messages_without_thinking'
  | 'reuse_journaled_tool_result'
  | 'probe_side_effect'
  | 'append_synthetic_message'
  | (string & {})

/** Coarse outcome class of a finished agent run. */
export type RunStopKind = 'completed' | 'incomplete' | 'blocked' | 'cancelled' | 'error'

/** Machine-readable cause of a run stop; surfaces localize by code. */
export type RunStopCode =
  | 'completed'
  | 'iteration_budget'
  | 'node_budget'
  | 'observation_budget'
  | 'no_progress'
  | 'stuck_repeat'
  | 'completion_gate'
  | 'cost_gate'
  | 'spend_budget'
  | 'approval_denied'
  | 'approval_timeout'
  | 'user_action_required'
  | 'policy_blocked'
  | 'wall_clock'
  | 'inactivity'
  | 'user_abort'
  | 'provider_error'

/** Action a surface can offer the user after a run stopped. */
export type RunStopNextAction =
  | 'resume'
  | 'switch_autonomy'
  | 'approve_pending'
  | 'raise_budget'
  | 'retry'

export interface RunStopReason {
  kind: RunStopKind
  code: RunStopCode
  /** Short operator-facing summary in English; surfaces localize by code. */
  summary?: string
  detail?: {
    budget?: number
    used?: number
    tool?: string
    criteria?: string[]
    requestId?: string
    layer?: string
  }
  resumable: boolean
  nextActions: RunStopNextAction[]
}

export type AgentEvent =
  | { type: 'state_change'; state: AgentState }
  | { type: 'execution_policy'; policy: AgentExecutionPolicy }
  | { type: 'memory_context'; id: string; items: MemoryContextItem[] }
  | {
      type: 'llm_request'
      turnId: string
      iteration: number
      requestDigest: LlmRequestDigest
    }
  | {
      /** Latest input-context pressure for one provider request. */
      type: 'context_usage'
      context: AgentContextUsage
    }
  | {
      type: 'mode_route_decision'
      chosen: string
      persona?: string
      candidates?: string[]
      reason?: string
      confidence?: number
      fallback: boolean
    }
  | {
      type: 'quality_gate_verdict'
      phase: string
      decision: 'pass' | 'retry' | 'incomplete'
      blockingReason?: string
      backtrackCount: number
    }
  | {
      type: 'backtrack'
      phase: string
      reason: string
      attempt: number
    }
  | {
      type: 'node_trace'
      node: string
      durationMs: number
      nextEdge?: string
    }
  | { type: 'thinking'; content: string }
  | {
      /**
       * Discrete intermediate reasoning step from a planning graph node
       * (tree-of-thought branch selection, auto-decompose plan, …). Unlike
       * `thinking` — which streams the provider's raw chain-of-thought
       * tokens — this event is emitted *by the graph itself* so surfaces
       * can show a structured, ChatGPT-style "Thought for X seconds" card
       * with discrete bullet steps. `detail` is optional long-form text
       * shown when the user expands the step (e.g. the judge's rationale,
       * the list of sub-tasks).
       */
      type: 'reasoning_step'
      label: string
      detail?: string
    }
  | {
      /**
       * Concise, model-authored narration attached to the same turn as one or
       * more tool calls. It explains the immediate purpose and expected
       * follow-up without exposing raw chain-of-thought. Surfaces may render
       * this as a stable one-line progress item immediately before the tools.
       */
      type: 'action_progress'
      summary: string
      nextStep: string
      toolNames: string[]
    }
  | { type: 'text_delta'; text: string }
  | { type: 'message'; content: string }
  | {
      /**
       * The run is blocked waiting for a human answer to a concrete question.
       * This is a live operational signal; the pending question store remains
       * the authoritative answer/resume surface.
       */
      type: 'question_request'
      questionId: string
      prompt: string
      choices?: string[]
    }
  | {
      /**
       * Append-only recovery marker emitted when the runtime repairs an
       * internal harness failure without mutating earlier conversation graph
       * events. The human-facing answer remains separate; this event gives
       * replay, export, and diagnostics a durable provenance trail.
       */
      type: 'recovery'
      scope: AgentRecoveryScope
      kind: AgentRecoveryKind
      action: AgentRecoveryAction
      message: string
      recoverable: boolean
      details?: Record<string, unknown>
    }
  | { type: 'tool_call'; toolCall: ToolCall }
  | {
      type: 'tool_result'
      toolCallId: string
      output: string
      status: 'success' | 'error'
      /**
       * Ephemeral visual output intended for the interactive chat surface.
       * Session persistence intentionally omits these bytes; tools opt in per
       * image so ordinary screenshots are not broadcast accidentally.
       */
      contentParts?: ContentPart[]
      recovery?: 'journal' | 'probe'
      executionPosture?: ToolExecutionPosture
      metadata?: ToolResultMetadata
    }
  | {
      type: 'approval_request'
      toolCall: ToolCall
      requestId: string
      suggestedRule?: ApprovalRule
      /**
       * Unified diff of the pending file edit so surfaces can show what
       * would change before the user approves. Only present for
       * file-editing tools whose change can be previewed read-only.
       */
      previewDiff?: string
      /**
       * Tail of the assistant text that immediately preceded this tool
       * call, so approval surfaces can show *why* the agent wants to run
       * the tool instead of a bare command prompt. Display-only
       * truncation of text the model already produced — no extra LLM
       * call, no parsing.
       */
      context?: string
    }
  | {
      /**
       * Emitted instead of `approval_request` when a remembered
       * session/always rule short-circuits the operator prompt. Carries
       * the matched rule + scope so surfaces can show a positive
       * "auto-approved by <scope> rule '<pattern>'" signal — without it
       * operators only see the tool result and have to grep `decisions
       * list` to figure out why no prompt appeared.
       */
      type: 'auto_approval'
      toolCall: ToolCall
      requestId: string
      decision: 'approved' | 'denied'
      rule: ApprovalRule
      scope: 'session' | 'always' | 'run' | 'session-all'
    }
  | {
      type: 'approval_response'
      requestId: string
      decision: ApprovalDecisionStatus
      approved: boolean
      note?: string
    }
  | {
      type: 'cowork_plan'
      plan: Array<{ role: string; instruction: string }>
    }
  | {
      /**
       * Coding/planning graphs have converted the user request into a
       * durable run contract. Surfaces can show this as the criteria the
       * agent will validate against, and session replay can audit drift
       * without scraping planner prose.
       */
      type: 'run_contract'
      contract: AgentRunContract
    }
  | {
      type: 'cowork_task_start'
      role: string
      instruction: string
    }
  | {
      type: 'cowork_task_complete'
      role: string
      instruction: string
      result: string
    }
  | {
      type: 'cowork_task_failed'
      role: string
      instruction: string
      error: string
    }
  | {
      type: 'cowork_synthesizing'
      summary: string
    }
  | {
      type: 'cowork_discuss_request'
      prompt: string
      choices?: string[]
    }
  | {
      type: 'cowork_discuss_response'
      prompt: string
      response: string
    }
  | {
      type: 'panel_open'
      personas: Array<{ id: string; name: string; description?: string }>
    }
  | {
      type: 'panel_turn_start'
      personaId: string
      personaName: string
    }
  | {
      type: 'panel_turn_complete'
      personaId: string
      personaName: string
      text: string
    }
  | {
      type: 'panel_turn_failed'
      personaId: string
      personaName: string
      error: string
    }
  | {
      type: 'panel_synthesizing'
      panelists: number
    }
  | {
      type: 'edit_checkpoint_opened'
      checkpoint: EditCheckpointSummary
    }
  | {
      type: 'edit_checkpoint_resolved'
      checkpoint: EditCheckpointSummary
    }
  | {
      type: 'debate_round'
      round: DebateRoundSummary
    }
  | {
      type: 'planner_working_memory_updated'
      workingMemory: PlannerWorkingMemory
    }
  | {
      /**
       * Emitted by the `markPhase` graph node when control transitions
       * between named phases (implementation / validation / review /
       * finalize, …). `enteredPhase` is the new active phase or `null`
       * when the run is closing the last phase out. `closedPhase` is the
       * phase that just ended and its incremental token usage; absent on
       * the very first transition. UIs can subscribe to this to render a
       * live phase-by-phase budget breakdown.
       */
      type: 'phase_change'
      enteredPhase: string | null
      closedPhase?: { phase: string; usage: TokenUsage }
      phaseUsages?: Record<string, TokenUsage>
    }
  | {
      /**
       * Emitted by the `post_edit_analysis` node after it has run
       * `code.dependencies` / `code.diagnostics` / LSP references on
       * the just-edited files. UIs can use this to surface blast
       * radius (forward + reverse callers) and any outstanding
       * diagnostics — including `[caller]`-prefixed entries that
       * indicate the edit broke a downstream file. Without this event
       * the data only existed inside the agent's reflectionMemo and
       * never reached the operator.
       */
      type: 'post_edit_findings'
      editedFiles: string[]
      impactedExternalModules: string[]
      impactedLocalModules: string[]
      reverseCallers: string[]
      diagnostics: Array<{ file: string; summary: string }>
      analyzedAt: string
    }
  | {
      /**
       * Daemon collapsed N earlier messages into a single summary to
       * free up context window head-room. Emitted on the chat stream so
       * cli/TUI surfaces can tell the user *why* token totals just
       * dropped and the assistant suddenly stopped referencing earlier
       * turns directly. (Also persisted as a session event for replay.)
       */
      type: 'context_compact'
      summary: string
      beforeTokens: number
      afterTokens: number
    }
  | {
      type: 'done'
      usage: TokenUsage
      /**
       * Structured termination cause. Optional for backward compatibility:
       * older daemons omit it and surfaces must treat a missing value as an
       * unknown (not necessarily successful) stop.
       */
      stopReason?: RunStopReason
    }
  | { type: 'error'; error: ApiError }
  | {
      /**
       * A progress event from a dispatched subagent, forwarded to the
       * parent run so surfaces can show nested activity (the tool calls,
       * reasoning steps, and thinking happening inside subagent.dispatch)
       * instead of the subagent appearing frozen until it returns its
       * single final result. `subagentId` is the subagent's session id so
       * a surface can group a fan-out's concurrent subagents. `inner` is
       * the subagent's own AgentEvent (terminal `done`/`error` are not
       * forwarded — the dispatch tool result reports those).
       */
      type: 'subagent_progress'
      subagentId: string
      label?: string
      inner: AgentEvent
    }
  | {
      /**
       * Emitted once per turn when the IntentRouter has chosen the
       * combination of mode + persona + skills for the upcoming agent
       * loop. Surfaces use this for the "Using mode=… persona=… skills=
       * [..] — <reason>" line. Always emitted, including the disabled
       * and fallback paths, so callers can rely on its presence to
       * decide whether to display the routing decision.
       */
      type: 'router_decision'
      id: string
      decision: {
        mode: string
        persona: string
        skillIds: string[]
        toolGroups?: string[]
        reason: string
        confidence: 'high' | 'medium' | 'low'
        fallback: boolean
      }
    }
  | {
      /**
       * Throttled progress snapshot derived from the agent state board
       * (criteria/plan/todos), emitted at the same cadence as the internal
       * board journal so cli/gui surfaces can render a lightweight
       * heartbeat during long runs without parsing prose. Only carries
       * structured counts — no free-text — and no "met" count for
       * criteria because `AgentAcceptanceCriterion` has no structured
       * met/status field today; surfaces should render a bare total in
       * that case rather than infer completion from text.
       */
      type: 'state_board'
      criteriaTotal: number
      planTotal: number
      planDone: number
      todosTotal: number
      todosDone: number
      /** Exact graph node active when this snapshot was emitted. */
      currentNode?: string
      /** Public lifecycle state assigned to the active graph node. */
      nodeState?: AgentState
      /** Agent-loop iteration at node entry. */
      iteration?: number
      /**
       * Bounded item snapshots so surfaces can render a live todo/plan list
       * during a run (not just counts). Text is truncated and the lists are
       * capped by the emitter; absent when the board has no items.
       */
      todos?: Array<{ content: string; status: string }>
      plan?: Array<{ title: string; status: string; depth: number }>
    }
  | {
      /**
       * Emitted at the iteration boundary where a mid-run user steering note
       * (`POST /sessions/:id/steer`) is consumed and surfaced to the model
       * (see daemon `agent/graph/state-board.ts` `takeUnconsumedSteeringNotes`
       * and the `buildAgentMessages` consume site in `agent/graph/nodes.ts`).
       * Distinct from the durable `steering_consumed` `SessionEvent`
       * (journaled separately, already-existing acknowledgement record) —
       * this lightweight AgentEvent only needs to reach the live chat stream
       * so cli/gui surfaces can render a "steering note applied" line.
       */
      type: 'steering_consumed'
      id: string
      noteId: string
      message: string
      kind: 'instruction' | 'question'
      iteration: number
    }

export interface EditCheckpointFile {
  path: string
  hadFileBefore: boolean
  bytesBefore: number
}

export interface EditCheckpointSummary {
  checkpointId: string
  label: string
  status: 'open' | 'committed' | 'reverted'
  files: EditCheckpointFile[]
  createdAt: string
  closedAt?: string
  revertedAt?: string
  revertReason?: string
}

export type DebateRole = 'proposer' | 'critic' | 'resolver'

export interface DebateRoundEntry {
  role: DebateRole
  content: string
  tokensUsed: number
  startedAt: string
  endedAt: string
}

export type DebateDecision = 'accept' | 'revise' | 'reject'

export interface DebateRoundSummary {
  roundId: string
  topic: string
  entries: DebateRoundEntry[]
  finalDecision: DebateDecision
  rationale: string
  createdAt: string
}

export type PlannerStepStatus = 'pending' | 'in_progress' | 'done' | 'blocked' | 'skipped'

export interface PlannerHierarchicalStep {
  id: string
  title: string
  status: PlannerStepStatus
  detail?: string
  children?: PlannerHierarchicalStep[]
}

export interface PlannerDecision {
  at: string
  text: string
}

export interface PlannerRisk {
  severity: 'low' | 'medium' | 'high'
  text: string
}

export interface PlannerAbandonedAlternative {
  /** Short description of the approach the planner tried and dropped. */
  description: string
  /** Why it was abandoned (failed test, dead-end, unsafe, …). */
  reason: string
  /**
   * True when this entry is a rollup digest that folds older abandoned
   * alternatives so the list stays bounded without fully deleting history.
   */
  digest?: boolean
}

export interface PlannerOpenAssumption {
  /** The unverified premise being held while planning ahead. */
  text: string
  /** True once the assumption has been positively confirmed. */
  verified?: boolean
  /**
   * True when this entry is a rollup digest that folds older assumptions so
   * the list stays bounded without fully deleting history.
   */
  digest?: boolean
}

export interface PlannerWorkingMemory {
  taskSummary: string
  currentSubtaskId?: string
  plan: PlannerHierarchicalStep[]
  decisions: PlannerDecision[]
  risks: PlannerRisk[]
  /**
   * Approaches the planner tried and abandoned. Surfaced back to the
   * planner on the next turn so the agent doesn't fall into the same
   * dead-end twice.
   */
  abandonedAlternatives?: PlannerAbandonedAlternative[]
  /**
   * Premises the plan currently depends on but hasn't yet verified.
   * Lets the planner explicitly track "I'm assuming X is true for now"
   * instead of forgetting; the validator can then turn an unverified
   * assumption into a concrete check before the run finalises.
   */
  openAssumptions?: PlannerOpenAssumption[]
  /**
   * One-sentence rationale for *why* the current step is the right next
   * action. Anchors longer multi-step runs against drift.
   */
  currentStepRationale?: string
  updatedAt: string
}
