import type { ModeControl } from '../mode-control.js'
import type {
  AgentArtifactSection,
  AgentEvidenceRequirement,
  AgentRequiredArtifact,
  AgentStateBoardSnapshot,
  AutonomyLevel,
  RunStopReason,
} from '@sepilotd/core'
import type {
  AgentContext,
  AgentEvent,
  ChatRequest,
  ChatResponse,
  IAuditLogger,
  ISemanticIndex,
  ILLMProvider,
  Message,
  ToolCall,
  ToolExecutionPosture,
  ToolSecurityEffect,
  TokenUsage,
} from '@sepilotd/core'
import type {
  ApprovalCallback,
  AutoApprovalEvaluator,
  PendingToolExecution,
} from '../tool-execution.js'
import type { HookRegistry } from '../../hook/registry.js'
import type { UsageTracker } from '../../memory/usage-tracker.js'
import type { LLMCache } from '../../providers/cache.js'
import type { ProviderCircuitBreaker } from '../../providers/circuit-breaker.js'
import type { PolicyEngine } from '../../security/policy-engine.js'
import type { ApprovalRunCheckpoint } from '../../server/runtime/checkpoints.js'
import type { RunResumeStage, SessionRunCheckpoint } from '../../server/runtime/runs.js'
import type { ActiveRunRegistry } from '../../server/runtime/active-runs.js'
import type { ToolExecutionRecord } from '../../server/runtime/tool-executions.js'
import type { ToolRegistry } from '../../tools/registry.js'
import type {
  FinalAnswerPresentationDiagnostics,
} from '../final-answer-presentation.js'

export interface GraphRuntimeState {
  modeControlContinue?: boolean
  currentStep: string
  totalUsage: TokenUsage
  iteration: number
  maxIterations: number
  maxToolCallsPerTurn?: number
  shouldStop: boolean
}

export interface AgentToolResultSummary {
  toolCallId: string
  toolName?: string
  output: string
  status: 'success' | 'error'
}

/**
 * Criterion protocol accepted against one exact tool-result episode. The
 * snapshot is durable loop state rather than presentation text: later writer
 * passes may omit the internal protocol, while a new tool result reopens the
 * criterion evidence boundary.
 */
export interface AgentCriterionVerdictSnapshot {
  toolResultCount: number
  /** SHA-256 of run-scoped tool result identity/status/timestamp/output. */
  toolResultFingerprint: string
  criterionVerdicts: FinalAnswerPresentationDiagnostics['removedCriterionVerdicts']
}

export interface AgentCriterionEvidenceReviewEpisode {
  toolResultCount: number
  toolResultFingerprint: string
  status: 'accepted' | 'invalid' | 'error'
  verdictCount: number
}

export interface AgentAcceptanceCriterion {
  id: string
  text: string
}

export interface AgentSeedContract {
  summary: string
  acceptanceCriteria: AgentAcceptanceCriterion[]
  constraints: string[]
  outOfScope: string[]
  requiredArtifacts?: AgentRequiredArtifact[]
  evidenceRequirements?: AgentEvidenceRequirement[]
  artifactSections?: AgentArtifactSection[]
  executionIntent?: import('@sepilotd/core').AgentExecutionIntent
  source: 'planner' | 'fallback'
}

export interface AgentCodebaseMap {
  generatedAt: string
  scoutPrompts: string[]
  summaries: Array<{
    toolCallId: string
    status: 'success' | 'error'
    /** False only when policy/approval rejected the call before tool execution. */
    executionObserved?: boolean
    summary: string
  }>
}

export interface AgentEvidenceLedgerEntry {
  /** Stable protocol identity shared with the durable session tool_call event. */
  toolCallId?: string
  tool: string
  status: 'success' | 'error'
  ts: number
  /**
   * Monotonic in-run evidence order. `ts` is millisecond-resolution wall time
   * and several tool results can share it, so ordering-sensitive gates such as
   * "validation after the latest write" must use this when present.
   */
  order?: number
  path?: string
  query?: string
  command?: string
  bytes?: number
  summary?: string
  /**
   * Controlled tool outputs can surface validation defects as structured
   * warnings, for example browser.screenshot DOM/canvas layout-audit warnings.
   * These are tool-observed issues, not model claims, and completion guards can
   * require a later clean validation pass before accepting rendered UI work.
   */
  defects?: string[]
  /**
   * Stable identity for browser-rendered UI audits. Lets completion gating
   * treat a later clean desktop/mobile screenshot as superseding an earlier
   * warning for the same URL+viewport without hiding warnings from a different
   * viewport or active state.
   */
  uiAuditKey?: string
  /**
   * True when the browser tool output actually included the structured
   * DOM/canvas layout-audit section. A successful screenshot file alone is not
   * enough for rendered-UI viewport coverage because interrupted recovery may
   * only prove that a PNG exists, not that layout/blank-band/canvas checks ran.
   */
  uiAuditHasLayout?: boolean
  /**
   * True when the browser tool output actually included the structured
   * browser console/page audit section. Rendered-UI completion summaries must
   * not claim console/page-error validation from screenshots whose audit text
   * was not recovered or was never emitted.
   */
  uiAuditHasConsole?: boolean
  /**
   * True when the browser tool output attached the saved screenshot image to
   * the next model turn. A saved PNG path and layout-audit text alone do not
   * prove the model could visually inspect the rendered result.
   */
  uiAuditHasImageAttachment?: boolean
  /**
   * True when the browser action exercised an actual rendered UI surface
   * instead of merely proving that a browser.evaluate call returned a literal
   * value or passive DOM reading. Used by rendered-UI completion gating for
   * games/apps/tools.
   */
  uiInteractionSmoke?: boolean
  /**
   * Claimed-vs-verified axis. True only when the entry is a *verification*
   * signal — a successful validation/check run or a successful read-back of a
   * required artifact after it was written. Plain reads/searches/writes stay
   * claimed-only (false/absent). Decided purely from the tool bucket and the
   * result status (structural signals), never from content parsing.
   */
  verified?: boolean
  /** Structured purpose declared on the originating tool call. */
  actionPurpose?: string
  /** Registry-owned effect classification captured at execution time. */
  securityEffect?: ToolSecurityEffect
  /** True only when the daemon observed the tool executor run. */
  executionObserved?: boolean
  /** Executor-confirmed posture for criterion-scoped operational evidence. */
  executionPosture?: ToolExecutionPosture
  /**
   * Provenance for evidence rolled up from an isolated subagent (subagent.dispatch,
   * large-codebase scout, cross-device delegation). Absent on evidence the parent
   * run gathered directly. Lets the parent board show "who found this" and lets the
   * prompt renderer tag the entry as `(via subagent <category>)`. Structural
   * provenance only — never used to re-derive findings from text.
   */
  origin?: { sessionId: string; category: string }
}

/** Structural progress counters compared across continuation cycles. */
export interface ContinuationProgressSnapshot {
  semanticRevision?: number
  verifiedEvidence: number
  executedToolCalls: number
  evidenceOrder: number
  artifactMutations: number
}

export interface AgentEvidenceLedger {
  sourceReads: AgentEvidenceLedgerEntry[]
  sourceSearches: AgentEvidenceLedgerEntry[]
  artifactWrites: AgentEvidenceLedgerEntry[]
  artifactReadBacks: AgentEvidenceLedgerEntry[]
  validationRuns: AgentEvidenceLedgerEntry[]
  errors: AgentEvidenceLedgerEntry[]
}

/**
 * A structurally-identified action that already failed this run. `signature`
 * is the stuck-tool-repeat structural signature (tool name + stable-serialized
 * arguments), so re-running "the same action" is decided deterministically,
 * never by content similarity. Canonical type for the state-board stack.
 */
export interface AgentFailedAttempt {
  signature: string
  tool: string
  reason: string
  ts: number
}

/**
 * A question the agent surfaced as unresolved. Blocking questions are
 * escalated to the human (HITL requestQuestion) when progress stalls instead
 * of letting the agent guess. Canonical type for the state-board stack.
 */
export interface AgentOpenQuestion {
  id: string
  text: string
  blocking: boolean
  askedAt?: number
}

export type AgentCoworkTaskStatus = 'complete' | 'incomplete' | 'unverified' | 'retry_scheduled'

export interface AgentCoworkTaskResult {
  sequence: number
  planIndex: number
  role: string
  instruction: string
  status: AgentCoworkTaskStatus
  result: string
  recordedAt: string
  /**
   * True when this entry is a rollup digest that folds older cowork task
   * results so the list stays bounded without fully deleting history.
   */
  digest?: boolean
}

export type AgentSpecialistRoute =
  | 'simple'
  | 'generalist'
  | 'creative'
  | 'reviewer'
  | 'coder'
  | 'researcher'

export interface AgentState extends GraphRuntimeState {
  input: string
  /** Full structured user message; `input` remains the primary instruction. */
  currentUserContent?: Message['content']
  messages: Message[]
  /** Provider-recovery window used after the declared model context was optimistic. */
  effectiveContextWindowTokens?: number
  plan?: string[]
  coworkPlan?: Array<{ role: string; instruction: string; choices?: string[] }>
  coworkDiscussPrompt?: string
  coworkDiscussChoices?: string[]
  coworkDiscussCount?: number
  coworkArtifactRetryCount?: number
  coworkDispatchCount?: number
  coworkBudgetExhausted?: boolean
  coworkBudgetReason?: string
  coworkTaskResults?: AgentCoworkTaskResult[]
  validationPlan?: string[]
  /**
   * User steering notes queued mid-run via `POST /sessions/:id/steer`
   * (see `state-board.ts` `appendSteeringNote` / `takeUnconsumedSteeringNotes`).
   */
  steeringNotes?: import('./state-board.js').AgentSteeringNote[]
  seedContract?: AgentSeedContract
  specialistRoute?: AgentSpecialistRoute
  specialistReason?: string
  specialistBrief?: string
  planIndex: number
  toolCalls: ToolCall[]
  /** Tool-call ids whose model-authored action progress was already streamed. */
  announcedActionProgressToolCallIds?: string[]
  /**
   * How many tool batches in this run lacked a model-authored purpose/next-step
   * annotation. Only the first emits the `action_progress_missing` recovery
   * event; later ones execute silently.
   */
  actionProgressMissingCount?: number
  /** Progress metadata consumed from reserved tool arguments for the pending batch. */
  pendingActionProgress?: {
    summary: string
    nextStep: string
    toolCallIds: string[]
  }
  /**
   * Tool-call id created by the focused loop's trusted deterministic initial
   * action. Completion shortcuts must key off this exact id; unrelated
   * observation results in the same turn must never terminate pending work.
   */
  deterministicInitialToolCallId?: string
  toolResults: AgentToolResultSummary[]
  recentToolResults?: AgentToolResultSummary[]
  memories: string[]
  output: string
  /**
   * A trusted, explicit operator denial ends the current turn. This is kept
   * separate from ordinary tool errors so reflection and completion guards do
   * not reinterpret a human stop decision as a recoverable execution failure.
   */
  /** Executor-attested prerequisite; ends this turn without speculative recovery. */
  userActionRequired?: string
  approvalDenied?: {
    toolCallId: string
    toolName?: string
  }
  /**
   * A plain (non-stop) denial granted the model one side-effect-free turn:
   * the tool catalog is filtered to policy read-only tools and any
   * side-effecting call ends the run as `approval_denied`. `toolTurnsRemaining`
   * bounds how many read-only tool turns the grace may spend before the model
   * must answer.
   */
  approvalDenialGrace?: {
    toolCallId: string
    toolName?: string
    toolTurnsRemaining: number
  }
  /**
   * Number of tool calls refused by policy or approval this run. Friction,
   * not failure: it never feeds stuck-repeat/failed-attempt accounting. Past
   * the threshold the model is told once to stop probing for the permission.
   */
  frictionCount?: number
  frictionWarningIssued?: boolean
  /** Bounded retry count for turns that explicitly require fresh tool evidence. */
  currentTurnEvidenceRetryCount?: number
  /** Shared across graph-node re-entry; fresh executor results renew the budget. */
  outcomeReviewRecovery?: {
    evidenceCount: number
    toolRepairs: number
    synthesisRepairs: number
    noProgressRepairs: number
  }
  /**
   * Set when screenshot/image tool output cannot be attached to the current
   * graph model, either because the model does not advertise vision support or
   * because the provider rejected image input despite being treated as
   * vision-capable. The graph keeps screenshot files on disk but stops
   * attaching image parts to later model turns.
   */
  visualAttachmentsDisabled?: boolean
  imageInputRecoveryCount?: number
  /**
   * Set when a graph stops because its execution budget was exhausted while
   * work may remain. The graph runner keeps the run checkpoint available so
   * `/resume` can continue instead of clearing state as a successful finish.
   */
  budgetExhausted?: boolean
  /**
   * Structured stop cause recorded by the node that ended the run. The graph
   * runner attaches it to the terminal `done` event; when absent it derives
   * one from `budgetExhausted`, `forcedFinalSynthesisReason`, and the
   * completion-gate diagnostics.
   */
  stopReason?: RunStopReason
  /**
   * Progress snapshot taken when the current continuation cycle started (or
   * at run start). The graph runner only grants another continuation cycle
   * when verified evidence, executed (non-blocked) tool calls or artifact
   * mutations advanced past this snapshot; otherwise the run stops as
   * `no_progress` even with cycles left.
   */
  continuationProgressSnapshot?: ContinuationProgressSnapshot
  /**
   * Loop-control questions already asked this run (stuck-repeat and
   * no-progress doom loops). Bounded by MAX_LOOP_CONTROL_QUESTIONS; past the
   * bound the run falls back to the forced final synthesis.
   */
  loopControlQuestionCount?: number
  /**
   * One-shot grace granted by a loop-control answer (`continue` /
   * `different_approach`): the stuck-repeat exhaustion check skips this many
   * agent turns before it may force a final synthesis again.
   */
  loopControlGraceTurns?: number
  codebaseExploration?: string
  codebaseMap?: AgentCodebaseMap
  evidenceLedger?: AgentEvidenceLedger
  toolRecommendationSummary?: string
  analysisSummary?: string
  findingsSummary?: string
  implementationSummary?: string
  /**
   * Bounded, checkpoint-scoped before/after mutation evidence retained after
   * implementation checkpoints close so later validation and review models
   * can judge the actual patch rather than tool names or byte counts.
   */
  implementationMutationEvidence?: Array<{
    checkpointId: string
    summary: string
  }>
  implementationRetryRequested?: boolean
  implementationNoEditRetryCount?: number
  /** Tool-history boundary captured when the current implementation phase starts. */
  implementationToolHistoryStartIndex?: number
  /**
   * Set by the graph engine when it restarts the same run after an internal
   * execution-budget continuation. The next phase marker consumes this flag
   * so per-turn convergence state is preserved across that internal restart.
   */
  internalGraphContinuation?: boolean
  /** Tool names rejected by the most recent allowlist-constrained model turn. */
  lastRejectedToolCallNames?: string[]
  /**
   * How many times the implementation phase has nudged the agent to stop
   * gathering read-only context and commit to an edit. Used to escalate the
   * convergence directive (and only fire it at increasing read thresholds)
   * when the agent over-explores without ever editing.
   */
  implementationConvergenceNudgeCount?: number
  /** Bounded convergence turns after the pre-edit discovery budget. */
  implementationPreActionStallCount?: number
  /**
   * Bounded recovery turns spent after a concrete implementation action when
   * the agent resumes repository observation without another edit, executable
   * check, managed runtime/browser action, or checklist transition. Reset by
   * the next concrete progress checkpoint. This closes the gap between the
   * pre-edit convergence guard and the run-wide iteration budget.
   */
  implementationPostActionStallCount?: number
  /**
   * Consecutive model turns whose requested read-only calls were all covered
   * by successful observations from the current user turn. A positive value
   * closes discovery on the next turn so duplicate observations cannot consume
   * the generic no-progress or graph-iteration budget.
   */
  implementationObservationReuseOnlyCount?: number
  implementationPendingTodoRetryCount?: number
  /**
   * Adaptive analysis depth. When a model keeps flailing in the implement loop
   * (ignores repeated convergence nudges, still no edit), the completion guard
   * requests the planner scaffolding mid-run so weaker models that need a
   * structured plan get one — without slowing strong models, which converge and
   * edit before this ever triggers. `Requested` is set by the guard and routes
   * to the late-planner node; `Applied` is set once it has run so it escalates
   * at most once per run.
   */
  implementationScaffoldingRequested?: boolean
  implementationScaffoldingApplied?: boolean
  /**
   * Tool-history boundary recorded when the adaptive implementation checkpoint
   * is applied. The next implementation turns may use bounded source reads
   * and one targeted sibling-source discovery before the
   * tool surface converges back to concrete edits.
   */
  implementationScaffoldToolHistoryBaseline?: number
  implementationCompleteRequested?: boolean
  /**
   * Adaptive tool-calling mode. Some models (e.g. gpt-oss on the Ollama
   * relay) advertise native function calling support, but in practice they
   * emit `tool_calls` with an empty `content` field — their reasoning goes
   * to the hidden `reasoning_content` channel and never surfaces in
   * `content`, so system-prompt nudges about plan-adherence / checkpointing
   * have nothing to land in.
   *
   * The agent node observes repeated unusable native-mode responses and flips
   * this to `true` when prompt transport should be tried. A prompt transport
   * that exhausts bounded format repair while a model-selected mutation is
   * pending may re-probe native once; it is not a permanent capability label.
   * This is a structural capability detection (single-shot per session, no
   * per-model branches, no behavior pattern matching) similar in spirit to
   * `capabilities.toolUse` but observed from the live response rather than
   * advertised.
   *
   * `undefined` = not yet observed. `false` = native is fine for this model.
   * `true` = use prompt-react for the current bounded transport episode.
   */
  preferPromptReact?: boolean
  /**
   * Counter of consecutive native-mode turns that emitted `tool_calls`
   * without any `content`. Used as the confidence threshold for the
   * `preferPromptReact` flip — single/double empty-content turns are too
   * noisy. Some strong native models (e.g. qwen3.6:27b) emit no content on
   * the first 2 calls (responding to user message + first tool result),
   * then emit content from turn 3 onward once accumulated tool outputs are
   * in context. Models that decline content entirely (e.g. gpt-oss:120b on
   * the Ollama relay) stay empty regardless. We require 3 consecutive
   * empties before flipping. Any native turn with content locks
   * `preferPromptReact=false`.
   */
  emptyNativeTurnsCount?: number
  /** Provider-observed per-run output ceiling when the configured catalog is stale. */
  effectiveMaxOutputTokens?: number
  /**
   * Set when the stuck-tool-repeat guard exhausted its repair budget and the
   * loop persisted anyway. The next LLM turn is the run's final evidence turn:
   * the runtime switches to a bounded, tool-free synthesis request and repairs
   * one malformed response without reopening the tool catalog. This closes the
   * run with a synthesis (or an honest incomplete result) instead of burning
   * the remaining iteration budget on a loop the repair messages could not break.
   */
  stuckRepeatForcedFinal?: boolean
  /** Observe-only stuck-loop repair messages already issued in the current evidence episode. */
  observeOnlyStuckRepeatRepairCount?: number
  /** Why tool access was closed for the next bounded final synthesis turn. */
  forcedFinalSynthesisReason?:
    | 'stuck-repeat'
    | 'inspection-observation-budget'
    | 'iteration-budget'
    | 'ordered-workflow'
    | 'no-retry-action-failed'
    | 'exact-tool-budget'
    | 'recovery-exhausted'
    | 'provider-no-progress'
    | 'completion-gate-evidence-closure'
  /** One-time runway granted when fresh tool evidence arrives at the loop cap. */
  iterationBudgetFinalSynthesisGranted?: boolean
  /**
   * Counter of consecutive native-mode turns that emitted ordinary assistant
   * text, no tool calls, and no final-answer protocol while a tool-backed
   * contract still has unfinished work. This is a different failure mode from
   * empty-content native tool calls: the model says it will act, but does not
   * attach a tool call in the same response. After a small threshold the graph
   * falls back to prompt-react so the transport contract becomes textual and
   * explicit.
   */
  contentOnlyNativeTurnsCount?: number
  /**
   * Number of times an adaptively selected prompt-tool transport exhausted
   * its own format-repair budget and control was returned to native tools.
   * This per-run circuit breaker makes fallback reversible without allowing
   * unbounded transport oscillation.
   */
  promptReactNativeReprobeCount?: number
  verificationSummary?: string
  /**
   * Number of completed search rounds in the researcher graph. This is
   * diagnostic unless an operator explicitly configures a round ceiling.
   */
  researchRounds?: number
  /** Unique successful observation packet before the active search pass. */
  researchSearchBaselineFingerprint?: string
  /** Whether the latest completed search pass added any new successful evidence. */
  researchSearchAddedEvidence?: boolean
  /** Executor-observed calls accumulated across all search/verification phases. */
  researchRunToolCallCount?: number
  /** Run-derived execution ceiling fixed before entering the first research phase. */
  researchRunToolCallBudget?: number
  /** Executor-observed calls accumulated inside the active research phase. */
  researchPhaseToolCallCount?: number
  /** Tool-history boundary at entry to the active bounded research phase. */
  researchPhaseToolHistoryBaseline?: number
  /**
   * Deep-thinking verifier loopback bookkeeping. When the verifier returns a
   * `revise` verdict but supplies no corrected answer, it routes back to the
   * agent for one bounded revision pass instead of shipping the known-bad
   * answer. `deepAnswerReviseRequested` drives the loopback edge for the next
   * hop; `deepAnswerReviseCount` bounds it.
   */
  deepAnswerReviseRequested?: boolean
  deepAnswerReviseCount?: number
  validationSummary?: string
  /**
   * Index into toolCallHistory immediately after the latest successful
   * implementation mutation. Successful checks after this boundary remain
   * valid when control moves from implementation to validation.
   */
  validationToolHistoryStartIndex?: number
  /**
   * Index into toolCallHistory when the current validation phase started.
   * Kept separate from the evidence boundary so phase-local discovery-loop
   * control does not discard post-mutation checks run during implementation.
   */
  validationPhaseToolHistoryStartIndex?: number
  /**
   * Bounded count of validation convergence nudges. Used when the validator
   * has already gathered concrete validation evidence but keeps calling tools
   * instead of producing the required VERIFIED:/UNVERIFIED: stem.
   */
  validationConvergenceNudgeCount?: number
  reviewSummary?: string
  qualityGateDecision?: 'pass' | 'retry' | 'incomplete'
  qualityGateSummary?: string
  /**
   * Internal completion/presentation audit. The user-facing answer deliberately
   * omits completion-protocol lines and phase token accounting, while the raw
   * reporter output and everything removed from presentation remain available
   * to run diagnostics.
   */
  completionDiagnostics?: {
    rawReporterOutput?: string
    presentation?: FinalAnswerPresentationDiagnostics
    criterionVerdictSnapshot?: AgentCriterionVerdictSnapshot
    /** At most one semantic criterion/evidence review per exact result episode. */
    criterionEvidenceReview?: AgentCriterionEvidenceReviewEpisode
    /** Independent adequacy judgment bound to the exact user premises and candidate. */
    providedContextReview?: { fingerprint: string; status: 'accepted' | 'rejected' | 'invalid' | 'error' }
    gate?: {
      decision: 'pass' | 'block'
      unmet: string[]
      reason?: string
      budgetExhausted?: boolean
    }
    phaseUsages?: Record<string, TokenUsage>
  }
  backtrackCount?: number
  backtrackReason?: string
  /**
   * Reasons from prior quality-gate backtracks this run (most recent last,
   * capped at 5). Injected into retry prompts to discourage repeating an
   * approach that already failed validation.
   */
  backtrackReasons?: string[]
  /**
   * How many times the completion gate has blocked a final answer this run
   * because acceptance criteria were not closed as MET with verified
   * evidence. Bounded by MAX_COMPLETION_GATE_BLOCKS so the gate cannot loop
   * forever; past the budget the run passes and reports honestly.
   */
  completionGateBlocks?: number
  /** Run-scoped observation novelty; preserved across graph phases/children. */
  workProgress?: import('../work-progress.js').WorkProgress
  /**
   * Latest final-answer draft the completion gate rejected (protocol stems
   * stripped). Surfaced as an explicitly-unverified draft when the gate's
   * block budget exhausts, so completed tool work is not discarded into a
   * bare "no final answer" message.
   */
  completionGateRejectedDraft?: string
  /**
   * Consecutive loop iterations that executed no tool call and produced no
   * accepted final answer (the model kept emitting plan/progress prose). Used
   * by the iteration guard's no-progress stop: reset whenever the executed
   * tool-call count grows, warned once, then the run stops honestly instead
   * of burning the whole iteration budget repeating the same plan.
   */
  noProgressIterations?: number
  /**
   * Number of accepted independent LLM recovery judgments used in the current
   * implementation recovery episode. Fresh observation evidence does not
   * reset this action budget; after it is exhausted, one model-owned semantic
   * transition may interpret the latest packet without scheduling another
   * intervening action. A durable implementation action or a real phase
   * re-entry resets the episode.
   */
  noProgressRecoveryJudgmentCount?: number
  /**
   * Number of bounded recovery-controller calls that returned no valid LLM
   * judgment (timeout, provider failure, or invalid structured output).
   * Kept separate from accepted judgments so telemetry and terminal messages
   * never claim that the model made a decision when it did not.
   */
  noProgressRecoveryControllerFailureCount?: number
  /**
   * Structural signature of the successful source-evidence packet against
   * which recovery-controller transport failures were recorded. A genuinely
   * new successful observation opens a fresh bounded transport budget, but it
   * does not replenish the episode-wide accepted-action budget above.
   */
  noProgressRecoveryControllerEvidenceSignature?: string
  /**
   * Total auxiliary provider calls spent by convergence control against the
   * current immutable source-evidence checkpoint. Causal diagnosis, response
   * transport repair, and completion audit share this one budget so nested
   * recovery layers cannot multiply latency at the same checkpoint.
   */
  noProgressRecoveryProviderCallCount?: number
  /**
   * Run-level totals that no evidence checkpoint, phase re-entry, or durable
   * action replenishes. The per-checkpoint counters above legitimately reset
   * when new evidence arrives, but a run whose controller keeps failing can
   * produce a "new" observation (an empty glob, a failed read) at every
   * checkpoint and spin for the whole iteration budget. These totals bound the
   * whole convergence machinery per run; only explicit user steering resets
   * them because that is a new intent, not new evidence.
   */
  recoveryControllerFailureTotal?: number
  recoveryProviderCallTotal?: number
  /**
   * How many times the recovery-exhausted final synthesis was scheduled in
   * this run. It is a terminal contract: once it has run, an unsupported or
   * rejected final must end the run honestly instead of reopening recovery.
   */
  recoveryExhaustedFinalCount?: number
  /**
   * The controller-unavailable fallback has already granted its single
   * bounded main-model turn in the current no-progress episode. This limits
   * recovery retries without changing which policy-allowed tools the main
   * model may choose from.
   */
  implementationControllerFallbackTurnGranted?: boolean
  /**
   * Legacy checkpoint marker from the former edit-only recovery flow. New
   * transitions keep this false: controller guidance must never rewrite the
   * normal policy-filtered tool surface.
   */
  implementationActionOnlyRecovery?: boolean
  /**
   * Legacy companion marker retained only for checkpoint compatibility.
   * Bounded controller fallback is tracked independently by
   * implementationControllerFallbackTurnGranted.
   */
  implementationActionOnlyRecoveryAttempted?: boolean
  /**
   * Number of bounded corrections granted inside the current action-only
   * recovery episode after an edit invocation failed or changed an artifact
   * that did not satisfy the active implementation contract. This budget is
   * deliberately independent from the ordinary no-edit retry counter.
   */
  implementationActionOnlyCorrectionCount?: number
  /**
   * Structured semantic decision produced by the independent convergence
   * controller for the single focused mutation turn. The controller reaches
   * this state only after it judges the retained evidence sufficient for a
   * mutation, so the next turn exposes the normal policy-filtered workspace
   * mutation capabilities rather than reopening observation/validation.
   * Keeping this separate from
   * conversation prose prevents the main model from restarting generic
   * discovery after the controller has already identified the causal path.
   */
  implementationMutationHandoff?: {
    reason: string
    guidance: string
  }
  /**
   * Marks a mutation handoff produced by a controller turn whose tool surface
   * allowed only a final semantic phase transition. Unlike an intervening
   * advisory handoff, this is an authoritative capability boundary: the next
   * main-model turn may mutate the workspace or reconcile agent-owned state,
   * but it may not reopen observation, validation, or process management.
   */
  implementationMutationCapabilityBoundary?: 'authoritative'
  /**
   * A concrete action selected by the independent recovery controller has
   * been queued and its result still needs a model-owned convergence
   * decision. The structural signature ties the transition to exactly that
   * tool invocation without inferring meaning from prompt text or tool names.
   */
  implementationRecoveryActionPending?: {
    signature: string
    actionPurpose: 'observe' | 'mutate' | 'validate' | 'unblock'
  }
  /**
   * Bounded, model-produced analysis retained from a recovery-controller
   * response that could not satisfy the structured decision transport. The
   * text is advisory evidence for the next main-model turn, not an executable
   * action and never bypasses normal tool policy.
   */
  implementationRecoveryHandoff?: string
  /** The handoff came from the dedicated unstructured diagnosis transport. */
  implementationRecoveryHandoffReady?: boolean
  /**
   * One bounded LLM causal analysis of a reported behavior that remains
   * unresolved at a convergence checkpoint. This is advisory source reasoning
   * for the action controller and never executes a tool itself.
   */
  implementationCausalDiagnosis?: string
  /**
   * Exact read-only observation selected by the causal-analysis LLM when its
   * source judgment is incomplete. The graph validates policy and current-turn
   * coverage, but never derives this action from prompt text or repository
   * conventions.
   */
  implementationCausalObservation?: {
    tool: string
    input: Record<string, unknown>
    /**
     * Bounded projection of an authoritative successful observation that was
     * outside the prior controller packet. This is evidence replay, not a new
     * tool execution, and is consumed by the next causal judgment only.
     */
    replayEvidence?: string
  }
  /**
   * Semantic phase transition selected by the same causal-analysis LLM that
   * compared the observed branches. Keeping diagnosis and phase ownership in
   * one model decision avoids a second controller reinterpreting (or failing
   * to serialize) an already-complete causal judgment.
   */
  implementationCausalTransition?: {
    decision: 'mutate' | 'complete_phase' | 'blocked'
    goalStatus: 'unresolved' | 'satisfied' | 'blocked'
    phaseStatus: 'unresolved' | 'satisfied' | 'blocked'
    reason: string
    guidance: string
  }
  /**
   * Structural rejection from the last causal observation proposal (for
   * example, already-covered evidence). It is fed back to the same bounded LLM
   * lane so the next proposal is a model decision rather than a graph guess.
   */
  implementationCausalObservationRejection?: string
  /** The causal-diagnosis lane has run at least once in the current mutation episode. */
  implementationCausalDiagnosisAttempted?: boolean
  /**
   * Fingerprint of the retained source-evidence packet used by the latest
   * successfully structured causal diagnosis. Empty, failed, or malformed
   * attempts do not populate this cache key and may be retried within the
   * bounded attempt budget.
   */
  implementationCausalDiagnosisEvidenceSignature?: string
  /** Evidence fingerprint whose malformed/failed diagnosis attempts are being bounded. */
  implementationCausalDiagnosisAttemptEvidenceSignature?: string
  /** Number of bounded causal-diagnosis attempts for the current evidence fingerprint. */
  implementationCausalDiagnosisAttemptCount?: number
  /**
   * Durable file delta currently present in the open implementation edit
   * checkpoint. A successful edit tool invocation is only an attempted
   * mutation; later corrective edits may restore every touched file to its
   * checkpoint baseline. When the snapshot store can inspect that baseline,
   * these fields keep convergence and completion decisions tied to the net
   * workspace state rather than historical tool success.
   */
  implementationNetMutationPresent?: boolean
  implementationNetMutationPaths?: string[]
  /**
   * The implementation model exhausted its local response-format repair, but
   * the enclosing coder graph still owns the semantic continue/stop decision.
   * Routes the result to the graph convergence controller instead of turning
   * a transport failure into a terminal task judgment.
   */
  implementationModelRecoveryRequested?: boolean
  /**
   * A closed structured implementation checklist has fresh execution evidence
   * and needs one model-owned phase decision. This is separate from generic
   * no-progress recovery so an already-complete workspace can hand off to
   * validation without manufacturing a no-op edit.
   */
  implementationCompletionAuditRequested?: boolean
  /** Tool-history boundary already offered to the completion auditor. */
  implementationCompletionAuditToolCallCount?: number
  /**
   * The active quality-phase model exhausted response/tool serialization, but
   * the enclosing graph still owns the semantic VERIFIED/UNVERIFIED decision.
   * This explicit handoff keeps a local protocol failure from terminating the
   * whole run before the bounded quality-conclusion controller can execute.
   */
  qualityConclusionRecoveryRequested?: 'validation' | 'review'
  /**
   * Marks the next capture as the outcome of the required-tool quality
   * controller: a structured decision or its fail-closed unavailable marker,
   * never another free-form validator response. This prevents recursive
   * normalization and persists across cancellation between graph nodes.
   * An unavailable outcome remains UNVERIFIED with retry target blocked.
   */
  qualityConclusionResolvedPhase?: 'validation' | 'review'
  /** LLM-selected recovery phase, or blocked when the required controller is unavailable. */
  qualityConclusionRetryTarget?: 'implementation' | 'validation' | 'blocked'
  /** Executed tool-call count observed at the last iteration boundary. */
  lastIterationToolCallCount?: number
  /**
   * How many supervisor-generated artifact revision drafts have run this
   * run. Each draft is a full auxiliary LLM call plus a file write; without a
   * cap a model that never closes with ANSWER: keeps the revision treadmill
   * spinning indefinitely (observed: 58 drafts in one run). Past the cap the
   * supervisor stops drafting and the run finalizes honestly.
   */
  artifactRevisionDrafts?: number
  /**
   * The finishReason of the most recent agent LLM turn. Propagated verbatim
   * from the provider (`length`/`content_filter`/`tool_use`/`stop`) so the
   * supervisor can act on a length truncation (bounded continuation) instead
   * of finalizing a cut-off turn as a complete answer.
   */
  lastTurnFinishReason?: ChatResponse['finishReason']
  /**
   * How many times a length-truncated turn has been auto-continued this run.
   * Bounded by SEPILOTD_LENGTH_CONTINUE_MAX (default 1).
   */
  lengthContinuationCount?: number
  /** Buffered, non-public text from length-truncated turns awaiting a clean continuation. */
  lengthContinuationPrefix?: string
  /** Current truncated fragment, transferred to lengthContinuationPrefix by the supervisor. */
  pendingLengthContinuationText?: string
  /** The pending fragment is an incomplete prompt-protocol tool call and must be discarded. */
  pendingLengthContinuationIsToolCall?: boolean
  /**
   * Actions recorded as failed this run, keyed by structural signature.
   * Consulted by the failed-attempt pre-execution guard before re-running a
   * structurally-identical tool call.
   */
  failedAttempts?: AgentFailedAttempt[]
  /**
   * How many times the failed-attempt guard has blocked a tool call this
   * run. Bounded by MAX_FAILED_ATTEMPT_BLOCKS: past the budget the call is
   * executed anyway (with a warning) so the guard can never deadlock a run.
   */
  failedAttemptBlocks?: number
  /**
   * Unresolved questions the agent surfaced during this run (promoted from
   * structured planner working memory). Blocking entries are escalated to the
   * human via requestQuestion when progress stalls instead of guessing.
   */
  openQuestions?: AgentOpenQuestion[]
  /**
   * Session todo list as first-class loop state. Updated structurally from
   * successful `todowrite` tool calls (never scraped from output text),
   * re-injected every turn via the state board, and carried through
   * checkpoints so `/resume` keeps it. Incomplete items are a blocking
   * signal for the completion gate (P022 T3 / D6).
   */
  todoList?: import('@sepilotd/core').TodoItem[]
  /**
   * How many open questions have been escalated to the human this run.
   * Bounded by MAX_ESCALATIONS so a stalled run cannot spam the user.
   */
  escalationCount?: number
  /**
   * Verified-evidence count observed the last time the loop checked for a
   * progress stall, with the iteration it last increased at. Together these
   * define "stalled" structurally: no new verified evidence for
   * PROGRESS_STALL_ITERATIONS iterations.
   */
  lastVerifiedEvidenceCount?: number
  lastVerifiedEvidenceIteration?: number
  /**
   * Observation ids from computer.observe whose screenshot files have
   * already been injected into the model context as multimodal image
   * parts during this run.
   */
  computerUseObservationIds?: string[]
  currentEditCheckpointId?: string
  editRollbacks?: Array<{
    checkpointId: string
    reason: string
    files: string[]
    revertedAt: string
  }>
  debateRounds?: import('@sepilotd/core').DebateRoundSummary[]
  plannerWorkingMemory?: import('@sepilotd/core').PlannerWorkingMemory
  subgraphState?: {
    node: string
    state: AgentState
  }
  taskType: 'simple' | 'complex' | 'code' | 'creative'
  /**
   * Reflexion-style self-critique notes generated after tool failures or
   * stalled progress. The agent node injects the most recent entries as a
   * system reminder on the next LLM turn so the model revises strategy
   * based on what just went wrong instead of repeating the same mistakes.
   */
  reflectionMemo?: string[]
  /**
   * Cumulative log of every tool call made during this run, in order. Used
   * by the reporter node to extract reusable skill candidates (Voyager-
   * style) from successful workflows. Bounded; older entries fall off.
   */
  toolCallHistory?: Array<{
    toolCallId?: string
    tool: string
    input: Record<string, unknown>
    status: 'success' | 'error'
    /** False when policy or validation rejected the call before tool execution. */
    executionObserved?: boolean
    /** Registry-owned effect classification captured before result handling. */
    securityEffect?: ToolSecurityEffect
    executionPosture?: ToolExecutionPosture
    /** Stable tool-protocol failure class retained separately from human-readable output. */
    failureCode?: string
    /** Refused before execution by policy or approval (friction, not failure). */
    blocked?: boolean
    ts: number
    /** SHA-256 of the complete bounded tool result before history text clipping. */
    outputFingerprint?: string
    /** Bounded model-visible output retained so compaction-safe observation
     * reuse can return evidence instead of silently dropping a repeated call. */
    output?: string
  }>
  /**
   * Structured outcome of an enforced validation command (test/build run).
   * Populated by `captureValidationOutcome` after `enforceValidationCommand`
   * dispatches the command via `terminal.run`. The qualityGate prefers this
   * structure over LLM self-report when present, so a coding run cannot
   * silently claim "tests passed" without an actual exit-zero proof.
   */
  validationOutcome?: {
    command?: string
    exitCode?: number
    passed: boolean
    failedSignals: string[]
    rawOutput?: string
  }
  /**
   * Findings from a post-edit analysis pass: which files the run just
   * touched, what callers/imports those files have (via code.dependencies),
   * and any LSP diagnostics still present (via code.diagnostics). Folded
   * into `reflectionMemo` so the agent's next turn sees the impact radius
   * and outstanding errors without having to ask for them.
   */
  postEditFindings?: {
    editedFiles: string[]
    impactedExternalModules: string[]
    impactedLocalModules: string[]
    /**
     * Reverse callers: best-effort list of files that reference each
     * edited file's basename (likely importers). Computed via
     * `code.symbols` so it survives even when the language has no LSP
     * server. False positives possible — humans/agents should treat
     * this as candidates, not authoritative.
     */
    reverseCallers: string[]
    diagnostics: Array<{
      file: string
      summary: string
    }>
    analyzedAt: string
  }
  /**
   * Semantic compression bookkeeping. When the contextManager runs an LLM
   * compaction pass, it stashes the resulting summary here so subsequent
   * passes don't re-summarize already-compressed history. `upToIndex`
   * tracks how many non-system conversation messages have been folded in
   * so far (rolling window).
   */
  compressedHistorySummary?: string
  compressedHistoryUpToIndex?: number
  /**
   * Per-phase token accounting. Updated by `markPhase` whenever the
   * graph hands control to a new phase (implementation, validation,
   * review, finalize). The value at a phase key is the cumulative
   * delta between the phase's start and end. Used by qualityGate's
   * `phaseTokenBudget` and emitted through structured phase-change/run
   * diagnostics. It is not appended to the conversational answer.
   */
  phaseUsages?: Record<string, TokenUsage>
  /**
   * Internal: snapshot of `totalUsage` taken when the current phase
   * started, so the next `markPhase` call can compute a delta and
   * rotate the active phase. Not user-visible.
   */
  phaseUsageStart?: { phase: string; usage: TokenUsage }
}

export interface GraphNodeDeps {
  provider: ILLMProvider
  tools: ToolRegistry
  policy: PolicyEngine
  autonomy: AutonomyLevel
  semanticIndex?: ISemanticIndex
  systemPrompt?: string
  usageTracker?: UsageTracker
  providerCircuitBreaker?: ProviderCircuitBreaker
  auxiliaryLlmBudget?: import('../auxiliary-llm.js').AuxiliaryLlmTurnBudget
  /**
   * Lazily-created budget for bounded control-plane decisions that must remain
   * available after optional planning/review work has consumed or outlived the
   * shared auxiliary budget. This lane is for graph convergence and safety
   * decisions, not task execution or general-purpose extra model calls.
   */
  controlLlmBudget?: import('../auxiliary-llm.js').AuxiliaryLlmTurnBudget
  /** Ephemeral request identities already surfaced live by the graph engine. */
  emittedLlmRequestObjects?: WeakSet<ChatRequest>
  /**
   * Directory where skills live (`~/.sepilotd/skills` by default). When set,
   * the reporter can drop auto-extracted skill candidates into
   * `<skillsDir>/auto/<id>/SKILL.md` after a successful run.
   */
  skillsDir?: string
}

export interface GraphExecutionContext extends GraphNodeDeps {
  modeControl?: ModeControl
  graphId: string
  agentContext: AgentContext
  /** Hard per-turn contract: no graph node may expose or recover a tool. */
  toolsForbiddenByUser?: boolean
  /**
   * Structural tool-effect boundary for evidence-only phases. Unlike a tool
   * name allowlist, this applies the registry's audited security contract to
   * built-ins and plugins uniformly.
   */
  toolSecurityEffectBoundary?: 'observe-only'
  /** Mirrors agentContext.requireToolApproval for graph tool executors. */
  requireToolApproval?: boolean
  activeGraphNodeId?: string
  activeGraphIteration?: number
  /**
   * Set while an AgentGraph is executing as a child of another AgentGraph.
   * Child subgraphs normally emit intermediate specialist outputs. A
   * generalist child that owns terminal synthesis may run the internal
   * completion gate before handing its protocol-bearing result to the outer
   * reporter for validation and conversational presentation.
   */
  agentSubgraphNodeId?: string
  /**
   * Node-execution budget the engine has left for THIS run, refreshed before
   * each node dispatch. A nested subgraph reads it to cap its own budget so
   * nesting shares one descending budget instead of restarting a full budget
   * at every level.
   */
  remainingNodeBudget?: number
  pendingAgentEvents?: AgentEvent[]
  graphNodeModelOverrides?: import('../../config/schema.js').SepilotdConfig['agent']['graphNodeModelOverrides']
  signal?: AbortSignal
  auditLogger?: IAuditLogger
  hookRegistry?: HookRegistry
  deviceName?: string
  thinkingLevel?: string
  maxTokens?: number
  temperature?: number
  auxModel?: string
  textDeltaMode?: 'buffered' | 'live'
  /**
   * Set by the top-level agent supervisor when a final answer candidate still
   * has to pass a structural gate before it is safe to stream to the user.
   */
  suppressSpeculativeFinalDeltas?: boolean
  /**
   * Persona roster for the persona-panel graph. Other graphs ignore it.
   * Resolved in chat-stream from the request's personaIds and threaded
   * through AgentModeRouter so the panel graph can answer with the same
   * persona objects (built-in + custom) the desktop UI shows.
   */
  panelPersonas?: import('../personas.js').Persona[]
  /**
   * Sequential panels call the roster exactly once in order; moderated panels
   * use the facilitator to select speakers dynamically.
   */
  panelStrategy?: 'sequential' | 'moderated'
  llmCache?: LLMCache
  approvalCallback?: ApprovalCallback
  evaluateAutoApproval?: AutoApprovalEvaluator
  saveApprovalCheckpoint?: (checkpoint: ApprovalRunCheckpoint) => Promise<void>
  clearApprovalCheckpoint?: (requestId: string) => Promise<void>
  saveRunCheckpoint?: (checkpoint: SessionRunCheckpoint) => Promise<void>
  clearRunCheckpoint?: (sessionId: string) => Promise<void>
  /**
   * Journal a board snapshot to the append-only session log at the same cadence
   * as `saveRunCheckpoint`. Persists and broadcasts the board in one event.
   */
  journalStateBoard?: (board: AgentStateBoardSnapshot) => Promise<void>
  /**
   * Journal a `steering_consumed` session event when a mid-run user steering
   * note (see `state-board.ts` `takeUnconsumedSteeringNotes`) is surfaced to
   * the model. Distinct from `SteeringAckEvent`, which is journaled when the
   * note is first queued via `POST /sessions/:id/steer`.
   */
  journalSteeringConsumed?: (noteId: string) => Promise<void>
  activeRuns?: ActiveRunRegistry
  loadToolExecution?: (sessionId: string) => Promise<ToolExecutionRecord | null>
  saveToolExecution?: (record: ToolExecutionRecord) => Promise<void>
  clearToolExecution?: (sessionId: string) => Promise<void>
  pendingToolExecution?: PendingToolExecution
  editSnapshotStore?: import('../edit-rollback/store.js').EditSnapshotStore
  toolStatsStore?: import('../tool-learning/store.js').ToolStatsStore
  workspaceMutationTracker?: import('../workspace-mutation/tracker.js').WorkspaceMutationTracker
  pluginEvents?: import('../../plugins/event-bus.js').PluginEventBus
  strictFinalAnswerProtocol?: boolean
  maxContinuationCycles?: number
  maxToolCallsPerTurn?: number
  requestQuestion?: (input: {
    sessionId: string
    prompt: string
    choices?: string[]
  }) => Promise<string>
}

export type GraphEdgeTarget = string | '__end__'

export interface GraphNodeMeta {
  lifecycleState?: import('@sepilotd/core').AgentState
  resumeStage?: RunResumeStage
  pendingToolExecutionNode?: boolean
}

export interface GraphNodeDefinition<State extends GraphRuntimeState, Context> {
  run: NodeFn<State, Context>
  meta?: GraphNodeMeta
}

export interface DirectGraphEdge {
  type: 'direct'
  to: GraphEdgeTarget
}

export interface ConditionalGraphEdge<State extends GraphRuntimeState, Context> {
  type: 'conditional'
  router: RouterFn<State, Context>
  targets: GraphEdgeTarget[]
}

export type GraphEdgeDefinition<State extends GraphRuntimeState, Context> =
  | DirectGraphEdge
  | ConditionalGraphEdge<State, Context>

export type StreamingNodeFn<State extends GraphRuntimeState, Context> = (
  state: State,
  context?: Context,
) => AsyncGenerator<AgentEvent, State, void>

export type NodeFn<State extends GraphRuntimeState, Context> = (
  state: State,
  context?: Context,
) => Promise<State> | AsyncGenerator<AgentEvent, State, void>

export type RouterFn<State extends GraphRuntimeState, Context> = (
  state: State,
  context?: Context,
) => GraphEdgeTarget
