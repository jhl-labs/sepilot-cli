import { repeatedBrowserClickWithoutProgress } from './browser-progress.js'
import { resolveThinkingLevel } from '../providers/thinking-policy.js'
import { userActionRequiredOutput } from './user-action-required-output.js'
import { MODE_TRANSFER_TOOL, type ModeControl } from './mode-control.js'
import { randomUUID } from 'node:crypto'
import type { ActiveRunRegistry, LiveRunState } from '../server/runtime/active-runs.js'
import { activeUserInstructions, formatActiveUserInstructions } from './user-steering.js'
import { recoveredReactSteering, withoutSteeringCheckpoint, withSteeringCheckpoint } from './react-steering.js'
import type {
  ILLMProvider,
  IAgentEngine,
  ChatRequest,
  ChatResponse,
  ContentPart,
  Message,
  AgentEvent,
  AgentState,
  AgentContext,
  AgentStateBoardSnapshot,
  TokenUsage,
  IAuditLogger,
  ToolCall,
} from '@sepilotd/core'
import { AutonomyLevel, ThinkingLevel } from '@sepilotd/core'
import type { ToolRegistry } from '../tools/registry.js'
import {
  isPolicyReadOnlyRequest,
  isPolicyReadOnlyTool,
  type PolicyEngine,
} from '../security/policy-engine.js'
import type { UsageTracker } from '../memory/usage-tracker.js'
import { checkSpendBudget, type SpendBudgetConfig } from './spend-guard.js'
import { delegatedUsageFromMetadata } from './tool-execution-helpers.js'
import { describeStaleSignal } from './workspace-mutation/tracker.js'
import type { HookRegistry } from '../hook/registry.js'
import type { LLMCache } from '../providers/cache.js'
import {
  guardedProviderChat,
  guardedProviderStream,
  isContextLengthProviderError,
  resolveProviderStreamFirstTokenMs,
  toProviderApiError,
  type ProviderCircuitBreaker,
} from '../providers/circuit-breaker.js'
import {
  isProviderModelImageInputRejected,
  markProviderModelImageInputRejected,
} from '../providers/vision-capability-state.js'
import { tokenCalibration } from '../providers/token-calibration.js'
import { decodeMultimodalInput } from './multimodal-input.js'
import { parseToolCallArguments, readLengthContinuationMax } from './tool-call-args.js'
import { lengthRecoveryInstruction } from './length-recovery.js'
import {
  cloneCheckpointExecutionSkillIds,
  cloneCheckpointScopeTags,
  cloneCheckpointSkillExecutionPolicies,
  cloneCheckpointSkillToolNames,
  cloneCheckpointToolAllowlist,
  type ApprovalRunCheckpoint,
} from '../server/runtime/checkpoints.js'
import type { RunResumeStage, SessionRunCheckpoint } from '../server/runtime/runs.js'
import type { ToolExecutionRecord } from '../server/runtime/tool-executions.js'
import { createAbortError, isAbortError } from '../abort.js'

function isApprovalParkedSignal(error: unknown): boolean {
  return Boolean(
    error
    && typeof error === 'object'
    && (error as { code?: unknown }).code === 'APPROVAL_PARKED',
  )
}
import {
  compactOversizedToolProtocolUnits,
  DEFAULT_UNKNOWN_MODEL_CONTEXT_WINDOW,
  emergencyContextRecovery,
  evaluateCurrentTurnToolObservationBudget,
  fitProviderContext,
  IrreducibleContextOverflowError,
  semanticCompress,
  supersedeStaleReadResults,
  supersedeStaleSystemReminders,
  type ProviderContextFitResult,
} from './context-manager.js'
import { summarizeProblemTools, type ToolStatsRecord } from './tool-learning/store.js'
import {
  buildStuckToolRepeatMessage,
  DEFAULT_TARGET_REPEAT_THRESHOLD,
  detectStuckToolRepeat,
  signatureOf,
  type StuckToolRepeatEntry,
} from './stuck-tool-repeat.js'
import {
  buildDegenerateResponseFallback,
  buildDegenerateResponseRecoveryMessage,
  isDegenerateRepeatedResponse,
} from './response-degeneration.js'
import {
  buildCanonicalExactOnceToolRepairMessage,
  buildCanonicalReadTargetRepairMessage,
  buildToolCallBatchCapMessage,
  capToolCallBatch,
  closedExactOnceToolBudgetComplete,
  currentTurnHasNoRetrySemanticActionFailure,
  exactOnceToolBudgetComplete,
  hasToolCallBatchCap,
  inputClosesExactOnceToolSet,
  partitionToolCallsByExactOnceBudget,
  partitionToolCallsByCanonicalReadTarget,
  remainingExactOnceToolNames,
  resolveToolCardinalityIdentities,
  resolveToolCanonicalReadIdentities,
  resolveToolCallBatchLimit,
  toolCallBudgetHistoryFromCurrentTurnMessages,
} from './tool-call-budget.js'
import {
  explicitlyForbidsToolUse,
  focusedRepositoryLookupNeedsDiff,
  isFocusedBoundedCodingTask,
  isFocusedRepositoryChangeLookup,
  isSubstantiveRepositoryChangeReview,
  isFocusedTerminalCommandExecution,
  extractFocusedProcessStart,
  requestsRawTerminalCommandOutput,
} from './request-shape.js'
import {
  buildRequiredArtifactPathEvidenceRepairMessage,
  isRequiredArtifactPathEvidenceBlock,
} from './artifact-evidence-recovery.js'
import {
  buildPromptFinalMessages,
  buildPromptReActMessages,
  buildPromptReActRepairMessages,
  buildTruncatedPromptToolCallRecoveryMessage,
  containsPromptToolCallEnvelope,
  containsPromptToolCallMarkup,
  extractPromptFinalCandidate,
  extractPromptFinalOutput,
  extractTaggedPayload,
  hasPromptFinalEnvelopeIntent,
  parsePromptToolCalls,
  repairNativeToolCalls,
  repairPromptToolCalls,
  resolveExplicitFinalTransportText,
  shouldAttemptPromptReActRepair,
  stripPromptReActThinkingArtifacts,
  UNUSABLE_PROMPT_TOOL_CALL_OUTPUT,
} from './prompt-react.js'
import {
  consumeToolCallActionProgress,
  extractAgentActionProgress,
  withAgentActionProgressSchemas,
} from './action-progress.js'
import {
  normalizeModelAnswerProtocol,
  resolveModelToolTransport,
} from './model-compatibility.js'
import {
  buildMaxOutputTokenRecoveryEvent,
  detectProviderMaxOutputTokenLimit,
} from './max-output-token-recovery.js'
import {
  appendAnswerProtocolSystemPrompt,
  buildEmptyFinalFallbackMessage,
  buildEmptyFinalRepairMessage,
  buildBoundedEmptyFinalRepairMessage,
  buildInvalidToolResponseRepairMessage,
  buildInvalidFinalResponseMessage,
  buildInterimProgressRepairMessageWithContext,
  buildInterimProgressFallbackMessage,
  buildMissingAnswerProtocolRepairMessage,
  buildUnsupportedCitationRepairMessage,
  hasAnyAnswerProtocolStem,
  hasFinalAnswerStem,
  hasIncompleteAnswerStem,
  isLikelyInterimProgressUpdate,
  shouldRepairEmptyFinalReply,
  shouldRepairInterimProgressReply,
  shouldRepairMissingAnswerProtocolReply,
  shouldRepairUnsupportedCitationReply,
  stripFinalAnswerStem,
  stripInternalPlannerBlocks,
  stripUnsupportedCitationReferences,
} from './interim-progress.js'
import {
  buildDuplicateToolCallRepairMessage,
  findDuplicateToolCallGroups,
  shouldRepairDuplicateToolCalls,
} from './duplicate-tool-calls.js'
import {
  buildSkillExecutionCompletionFailureOutput,
  buildSkillExecutionCompletionRecoveryMessage,
  buildSkillExecutionToolRepairMessage,
  countSkillExecutionCompletionRecoveryMessages,
  countSkillExecutionToolRepairMessages,
  evaluateSkillExecutionCompletion,
  hasDeterministicSkillCompletionPolicy,
  partitionSkillExecutionToolCalls,
} from './skill-execution-policy.js'
import { resolveActiveSkillExecutionPolicies } from '../skills/execution-policy.js'
import {
  availableScheduleEvidenceTools,
  buildScheduleCompletionFailureOutput,
  buildScheduleCompletionRecoveryMessage,
  countScheduleCompletionRecoveryPrompts,
  isExplicitScheduleCreateRequest,
  scheduleCompletionOutcomeFromMessages,
} from './schedule-completion.js'
import {
  deterministicScheduleResultFromMessages,
  normalizeExplicitScheduleListToolCalls,
} from './schedule-result-completion.js'
import {
  WriteLoopTracker,
  buildWriteLoopRepairMessage,
  DEFAULT_WRITE_LOOP_LIMIT,
} from './write-loop-guard.js'
import {
  buildRunOutcomeRecoveryMessage,
  evaluateRunOutcome,
  shouldRepairRunOutcome,
} from './outcome-recovery.js'
import {
  buildProviderMessageRecoveryMessage,
  buildProviderMessageRecoveryEvent,
  countProviderMessageRecoveryPrompts,
  detectProviderMessageRecovery,
  repairProviderMessageProtocol,
  shouldRecoverProviderMessageError,
} from './provider-message-recovery.js'
import {
  buildToolTransportRecoveryEvent,
  buildToolTransportRecoveryMessage,
  detectEmptyNativeToolResponse,
  detectNativeToolTransportRejection,
  hasToolTransportRecoveryMessage,
} from './tool-transport-recovery.js'
import {
  buildEmptyRunOutcomeReviewRecovery,
  buildOutcomeReviewExhaustedMessage,
  buildRunOutcomeReviewDuplicateSuggestedToolCallsMessage,
  buildRunOutcomeReviewSuggestedToolCallPlan,
  buildRunOutcomeReviewNoProgressMessage,
  buildRunOutcomeReviewRecoveryMessage,
  dropStaleRunOutcomeReviewRecovery,
  buildRunOutcomeReviewRequest,
  enforceRunOutcomeReviewEvidenceFloor,
  unavailableOutcomeReviewReason,
  hasPendingRunOutcomeReviewRecovery,
  MAX_OUTCOME_REVIEW_REPAIRS,
  MAX_OUTCOME_REVIEW_SYNTHESIS_REPAIRS,
  OUTCOME_REVIEW_MAX_TOKENS,
  parseRunOutcomeReviewTransport,
  shouldRepairRunOutcomeReview,
  shouldReviewOutcomeWithLLM,
} from './outcome-review.js'
import { cloneMessage, cloneToolCall, runToolExecution } from './tool-execution.js'
import { createToolCallProgressTracker } from './tool-call-progress.js'
import {
  canExecuteCompletedToolCalls,
  canPublishUserFacingText,
  interruptedUserFacingResponse,
  INTERRUPTED_USER_FACING_RESPONSE,
} from './response-safety.js'
import type {
  ApprovalCallback,
  AutoApprovalEvaluator,
  PendingToolExecution,
} from './tool-execution.js'
import {
  blockSourceFromMetadata,
  buildApprovalDeniedTurnOutput,
  createApprovalGraceState,
  observeApprovalOutcomeAfterTools,
  type TrustedApprovalDenial,
} from './approval-failure.js'
import {
  buildReadOnlyInvocationRepairMessage,
  buildReadOnlyPolicyBlockSynthesisMessage,
  buildReadOnlyPolicyBlockedTurnOutput,
  countReadOnlyInvocationRepairMessages,
  hasSuccessfulToolEvidenceInCurrentTurn,
  isRepairableReadOnlyInvocationFailure,
  latestTrustedPolicyFailure,
  readOnlyInvocationRepairToolCallIds,
  type TrustedPolicyFailure,
} from './policy-failure.js'
import {
  isIncompleteOutput,
  stopReasonApprovalDenied,
  stopReasonBudget,
  stopReasonCompleted,
  stopReasonCompletionGate,
  stopReasonNoProgress,
  stopReasonObservationBudget,
  stopReasonPolicyBlocked,
  stopReasonSpendBudget,
  stopReasonUserAbort,
  stopReasonUserActionRequired,
  stopReasonWallClock,
} from './stop-reason.js'
import {
  describeToolApprovalPosture,
  toolApprovalDescriptionSuffix,
  type ToolApprovalPostureMap,
} from './tool-approval-catalog.js'
import {
  buildToolApprovalPostureParagraph,
  TOOL_APPROVAL_POSTURE_PROMPT_PREFIX,
} from './system-prompt.js'
import {
  CONTINUATION_MARKER_METADATA_KEY,
  evaluateProgressSince,
  findProgressMarkerIndex,
  parseRunWallClockBudgetMs,
} from './progress-signal.js'
import {
  logAgentDebugTrace,
  logAgentRunTrace,
  logLlmCallTrace,
} from '../observability/agent-trace.js'
import {
  buildLlmRequestEvent,
  buildLlmTurnId,
} from '../observability/llm-request-event.js'
import {
  buildContextUsageEvent,
  buildProviderContextUsageEvent,
} from './context-usage.js'
import { createLiveTextDeltaEmitter } from './live-text-delta.js'
import { runAuxiliaryLlmChat, AuxiliaryLlmTurnBudget, DEFAULT_AUXILIARY_LLM_TURN_BUDGET_MS } from './auxiliary-llm.js'
import {
  buildMemoryWriteFailureOutput,
  buildMemoryWriteRecoveryMessage,
  countMemoryWriteRecoveryPrompts,
  isExplicitMemoryWriteRequest,
  latestUserText,
  MEMORY_REMEMBER_TOOL_NAME,
  memoryWriteOutcomeFromMessages,
  recordToolResultStatus,
  restoreSuppressedMemoryWriteDraft,
} from './memory-write-completion.js'
import {
  buildActionCompletionRecoveryMessage,
  buildActionCompletionFailureOutput,
  countActionCompletionRecoveryPrompts,
  evaluateActionCompletion,
  extractFocusedProcessObservationCall,
  inputRequestsFocusedSingleProcessObservation,
  requiredActionEvidenceKindsForTurn,
} from './action-completion.js'
import {
  beginCurrentAgentTurnUserMessage,
  CURRENT_AGENT_TURN_USER_METADATA_KEY,
} from './turn-context.js'
import {
  buildObservationReuseMessage,
  buildObservationReuseToolResultMessages,
  observationHistoryFromCurrentTurnMessages,
  observationHistoryOutput,
  partitionCallsCoveredByCurrentTurnObservations,
  type ObservationHistoryEntry,
} from './observation-coverage.js'
import { buildBoundedFinalSynthesisContext } from './bounded-final-context.js'
import {
  buildBudgetExhaustedMessage,
  buildContinuationPrompt,
  formatRunContractForPrompt,
  inputPositiveActionScope,
  inputRequestsFocusedSingleBrowserObservation,
  resolveMaxContinuationCycles,
} from './task-contract.js'
import { deterministicFocusedActionResult } from './focused-action-result.js'
import { isVisibleUrlOpenRequest } from './desktop-control-intent.js'
import {
  buildCriterionEvidenceBoardSnapshot,
  buildCriterionEvidenceReviewRepairRequest,
  buildCriterionEvidenceReviewRequest,
  criterionEvidenceStateFromMessages,
  parseCriterionEvidenceReviewTransport,
} from './criterion-evidence-review.js'

export type { ApprovalCallback, AutoApprovalEvaluator } from './tool-execution.js'

const FOCUSED_BOUNDED_CODING_TOOLS = new Set([
  'apply_patch',
  'code.diagnostics',
  'fs.append',
  'fs.edit',
  'fs.list',
  'fs.read',
  'fs.write',
  'terminal.run',
])
const FOCUSED_PROCESS_OBSERVATION_TOOLS = new Set([
  'process.follow',
  'process.list',
  'process.read',
  'process.sessions',
  'process.wait',
])

// On the final allowed iteration, append a system message that tells the model
// to stop chasing tool calls and emit its best answer with the information it
// already has. This stops slow models like glm-4.7 served via Ollama from
// looping <tool_call> indefinitely on knowledge questions.
const PLAN_MODE_HINT = [
  'The user has put this session in PLAN mode.',
  'You can read files, search, and explain — but every write tool',
  '(fs.write, fs.append, terminal.run, browser.*, etc.) will be blocked by policy.',
  'Do not attempt to execute them. Instead, produce a written plan,',
  'propose changes as text/diffs, and ask the user to exit plan mode',
  'before any step that would modify state.',
].join(' ')

const READ_ONLY_AUTONOMY_HINT = [
  'The user has enabled READ ONLY autonomy. This is not PLAN mode.',
  'Use dedicated read/search tools and structurally classified read-only terminal queries when they are sufficient.',
  'For Kubernetes inspection, terminal.run may execute safe kubectl queries such as get, describe, logs, top, events, and version.',
  'Call terminal.run with kubectl as the direct executable and an argv array; use timeoutMs for bounded observation instead of shell timeout commands or command chaining.',
  'For an intentional watch/follow window where silence validly means no changes, set timeoutOutcome to observation_complete; use success_if_output only when captured stdout is required for a usable result.',
  'Do not use either successful-timeout mode for commands that must exit normally to prove completion.',
  'When the request is specifically about changes during that window, prefer a native changes-only or watch-only option so initial state is not misreported as a new change; once the window completes, interpret silence as no observed changes instead of repeating the watch.',
  'If a before/after comparison genuinely requires a delay, call sleep directly with a finite duration and a slightly longer timeoutMs; never use a shell wrapper.',
  'Secret reads, raw API access, impersonation, shell wrappers, and every state-changing command remain blocked.',
  'A shell-wrapper invocation-shape block may receive one supervisor-directed repair as a direct policy-checked call; never use alternate execution or delegation tools to bypass a block.',
  'For every other required policy block, report the exact ReadOnly policy blocker and do not call it PLAN mode.',
].join(' ')

export function buildAutonomyHint(
  autonomy: AutonomyLevel,
  primaryAgentId?: string,
): string | null {
  // Plan mode is selected via primaryAgentId ('plan'), independently of the
  // autonomy level (CLI Shift+Tab flips the agent without touching autonomy).
  // Policy blocks writes for the plan agent, but the model must also be told
  // it is planning — otherwise it tries writes that policy then rejects
  // mid-turn and reacts to the denial instead of producing a plan.
  if (primaryAgentId?.trim().toLowerCase() === 'plan') {
    return PLAN_MODE_HINT
  }
  switch (autonomy) {
    case AutonomyLevel.ReadOnly:
      return READ_ONLY_AUTONOMY_HINT
    case AutonomyLevel.AcceptEdits:
      return [
        'The user has enabled ACCEPT EDITS mode.',
        'File edits (fs.write/fs.append) will be applied automatically without asking.',
        'Shell commands, browser actions, and other side-effectful tools',
        'still require explicit user approval, so keep those minimal and',
        'describe what you would run before requesting them.',
      ].join(' ')
    case AutonomyLevel.WorkspaceWrite:
      return [
        'The user has enabled WORKSPACE WRITE mode.',
        'File edits inside the active workspace may be applied automatically.',
        'Edits outside the workspace will be blocked by policy, and shell',
        'commands, browser actions, and other side-effectful tools still',
        'require explicit user approval.',
      ].join(' ')
    case AutonomyLevel.Supervised:
      return [
        'The user has enabled SUPERVISED mode.',
        'If the task requires changing files, running commands, browsing, or any',
        'other side effect, call the relevant tool directly instead of stopping',
        'at a written proposal.',
        'The runtime will pause and ask the user for approval when required.',
        'Do not ask a redundant chat-level permission question unless the user',
        'must choose between materially different actions.',
      ].join(' ')
    case AutonomyLevel.Autonomous:
      return [
        'The user has enabled AUTONOMOUS mode.',
        'Policy-allowed tools (rule `autonomous` or `supervised`) run directly once',
        'the deny rules pass, without interactive approval prompts.',
        'Hard denies still apply regardless of autonomy: deny_patterns, deny_paths,',
        'deny_executables, workspace boundaries, blocked tools, and unmatched tools',
        'under a deny-by-default policy. A denied tool will not become available by',
        'retrying or rephrasing the call; choose a policy-allowed alternative instead.',
        'A tool with an explicit `ask` policy (and raw shell wrappers such as',
        '`bash -c`) may still pause on an interactive surface for human approval;',
        'unattended runs block it instead.',
        'Do not assume every available tool can run unattended. Prefer allowed',
        'read/search tools or allowlisted commands when they are sufficient.',
        'If the task genuinely needs an approval-required tool and no interactive',
        'approval surface is available, state the concrete blocker and the exact',
        'command, and continue with whatever parts of the task do not need it.',
      ].join(' ')
    default:
      return null
  }
}

/**
 * Append a one-shot system message naming tools that have repeatedly
 * failed in the current session so the model can change tactics instead
 * of retrying the same broken call. The hint is injected per-iteration
 * into the outbound message list — it must not be persisted into
 * `messages`, otherwise old hints accumulate across turns.
 */
export function withProblemToolFeedback(
  messages: Message[],
  stats: ToolStatsRecord[],
  iteration: number,
): Message[] {
  // No useful signal until at least one tool result has been seen.
  if (iteration < 1 || stats.length === 0) {
    return messages
  }
  const problems = summarizeProblemTools(stats)
  if (problems.length === 0) {
    return messages
  }
  const lines = problems.map((record) => {
    const recent = record.recentErrorOutputs[record.recentErrorOutputs.length - 1] ?? ''
    const recentSnippet = recent
      ? ` (recent error: "${recent.slice(0, 120).replace(/\s+/g, ' ').trim()}")`
      : ''
    return `- ${record.tool}: ${record.errorCount} of ${record.totalCount} calls failed${recentSnippet}`
  })
  const hasArtifactEvidenceBlock = problems.some((record) =>
    (record.tool === 'fs.write' || record.tool === 'fs.append')
    && record.recentErrorOutputs.some((output) =>
      output.includes('unsupported_required_artifact_path_claims')
      || output.includes('cites repository paths without prior read/search/glob/write evidence')
    )
  )
  const content = [
    '[Tool feedback] These tools have been failing in this session — switch tactics rather than retrying with the same arguments:',
    ...lines,
    hasArtifactEvidenceBlock
      ? 'Required artifact writes are blocked by missing path evidence. Do not call fs.write/fs.append again until you have called fs.glob/fs.search/fs.read for the missing paths, or removed those concrete path claims.'
      : '',
  ].filter(Boolean).join('\n')
  return [...messages, { role: 'system', content }]
}

function appendForceFinalSystemMessage(messages: Message[]): Message[] {
  return [
    ...messages,
    {
      role: 'system',
      content: [
        'You have used all available reasoning steps for this turn.',
        'Stop calling tools. If every acceptance criterion is genuinely met,',
        'respond with the final answer. If anything remains incomplete, respond',
        'with INCOMPLETE: and state the concrete next step so the preserved',
        'checkpoint can be resumed. Do not present partial work as complete.',
      ].join(' '),
    },
  ]
}

function appendFocusedRepositorySynthesisMessage(messages: Message[]): Message[] {
  return [
    ...messages,
    {
      role: 'system',
      content: [
        'The focused repository evidence requested by the user is already available in the structured Git tool results.',
        'Do not call more tools, repeat Git queries with different path spellings, inspect unrelated source files, or use terminal.run as a second Git interface.',
        'Your next response is the final response: start it exactly with `ANSWER:` and do not use future-tense progress language such as "I will check" or "확인하겠습니다".',
        'Answer in the user\'s language. Summarize the observed commits/status/diff, distinguish committed from uncommitted changes, and state any real evidence limitation briefly.',
      ].join(' '),
    },
  ]
}

function appendSubstantiveRepositoryReviewMessage(messages: Message[]): Message[] {
  return [
    ...messages,
    {
      role: 'system',
      content: [
        '[Repository change review]',
        'This is a read-only assessment, not a commit-history summary.',
        'Use structured Git tools for commit and diff evidence, then file/code read tools only when source context is needed.',
        'Do not invent paths, switch to a shell interface, or repeat evidence already present in tool results.',
        'When the evidence is sufficient, stop calling tools and return the complete user-facing review; prioritize concrete correctness, regression, security, and missing-validation findings.',
        'A finding must identify an introduced defect or risk mechanism and point to the observed changed code or missing validation that causes it.',
        'Do not present uncertainty, unclear intent, or a request to confirm something as a defect when the retained diff or source evidence already answers it. Re-check the observed evidence first; if no concrete issue remains, explicitly report that no significant issue was found.',
        'Do not delay a usable review merely to satisfy a cosmetic response prefix.',
        'If evidence cannot support a review, state that it is incomplete and name the exact missing evidence.',
      ].join(' '),
    },
  ]
}

/**
 * Warn the model that it has only one more turn after this so it can
 * choose to wrap up rather than starting a fresh investigation. Without
 * this nudge the agent often spends the penultimate turn launching new
 * tool calls and then gets the harder force-final message on the last
 * turn with nothing useful to say.
 */
function appendBudgetWarningSystemMessage(messages: Message[]): Message[] {
  return [
    ...messages,
    {
      role: 'system',
      content:
        'You have one tool-using turn left after this one. Only call a tool now if its result directly produces information you need for the final answer — otherwise start synthesising what you already know.',
    },
  ]
}

function countOutcomeRecoveryPrompts(messages: Message[]): number {
  return messages.filter(
    (message) =>
      message.role === 'system' &&
      typeof message.content === 'string' &&
      message.content.includes('[Outcome supervisor]'),
  ).length
}

/**
 * How many outcome-supervisor blocks stay in the prompt.
 *
 * A run retries up to MAX_OUTCOME_RECOVERY_REPAIRS times and every attempt left
 * its critique behind: one captured request carried six of them, 8,768
 * characters, re-sent on every model call after that. Unlike the retry guards
 * these are not copies of each other — each names what that attempt got wrong —
 * so dropping all but the newest would erase the "this keeps failing" signal
 * the model needs to change approach. Keep the two most recent: the latest
 * critique to act on, and one predecessor to show the pattern.
 */
export const MAX_LIVE_OUTCOME_RECOVERY_PROMPTS = 2

export function dropStaleOutcomeRecoveryPrompts(messages: Message[]): void {
  const indexes = messages.flatMap((message, index) =>
    message.role === 'system'
      && typeof message.content === 'string'
      && message.content.includes('[Outcome supervisor]')
      ? [index]
      : [],
  )
  // One is about to be appended, so keep room for it.
  const excess = indexes.length - (MAX_LIVE_OUTCOME_RECOVERY_PROMPTS - 1)
  if (excess <= 0) return
  const drop = new Set(indexes.slice(0, excess))
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (drop.has(index)) messages.splice(index, 1)
  }
}

const ANALYSIS_CADENCE_READ_TOOL_NAMES = new Set([
  'fs.read',
  'fs.list',
  'fs.glob',
  'fs.search',
  'terminal.run',
  'git.log',
  'git.diff',
  'git.status',
  'code.dependencies',
  'code.symbols',
])

const REACT_CONSECUTIVE_READ_GUARD_TOOLS = new Set([
  'fs.read',
  'fs.list',
  'fs.glob',
  'fs.search',
  'git.log',
  'git.diff',
  'git.status',
  'code.dependencies',
  'code.symbols',
  'terminal.run',
  'memory.search',
  'memory.graph.search',
])

const STRUCTURED_TOOL_ERROR_CODE_PATTERN = /^\[error:\s*([A-Z0-9_]+)\]/

function structuredToolFailureCode(message: Message | undefined): string | undefined {
  if (!message || typeof message.content !== 'string') return undefined
  return message.content.match(STRUCTURED_TOOL_ERROR_CODE_PATTERN)?.[1]
}

const FOCUSED_REPOSITORY_LOOKUP_TOOLS = new Set([
  'git.log',
  'git.diff',
  'git.status',
])

const SUBSTANTIVE_REPOSITORY_REVIEW_TOOLS = new Set([
  'git.log',
  'git.diff',
  'git.status',
  'fs.read',
  'fs.search',
  'code.symbols',
  'code.dependencies',
])
const SUBSTANTIVE_REPOSITORY_REVIEW_OBSERVATION_TOOLS = new Set([
  ...SUBSTANTIVE_REPOSITORY_REVIEW_TOOLS,
  // Some providers and tool profiles expose repository inspection only
  // through the structured terminal adapter. Count those observations too,
  // otherwise the review budget can never converge in a terminal-only route.
  'terminal.run',
])
const SUBSTANTIVE_REPOSITORY_REVIEW_TOOL_TURN_LIMIT = 3
// Interactive read-only runs should converge after two distinct evidence
// batches once they have enough successful observations to answer.  Counting
// both batches and observations avoids cutting off a pair of narrow lookups,
// while preventing diagnostic agents from repeatedly widening a completed
// investigation.  Mutation-capable runs and skill-governed workflows keep
// their existing, larger budgets.
const READ_ONLY_EVIDENCE_TURN_LIMIT = 2
const READ_ONLY_EVIDENCE_MIN_SUCCESSFUL_OBSERVATIONS = 4
// Iterative tool turns need only enough room to emit a compact structured call.
// Preserve the default 512-token floor for tool-free/final synthesis, but let
// active tool schemas take precedence when a small-context model is tight.
const TOOL_ENABLED_MIN_OUTPUT_TOKENS = 256
const BOUNDED_FINAL_MAX_OUTPUT_TOKENS = 3_072
// A model that emits hidden reasoning spends the same output budget on its
// reasoning and on the answer. A cap sized for the answer alone therefore
// guarantees a truncated final turn — the one turn that has no tool access
// left to recover with. Give reasoning models explicit headroom.
const BOUNDED_FINAL_THINKING_MAX_OUTPUT_TOKENS = 12_288
const BOUNDED_FINAL_TIMEOUT_MS = 120_000

const ANALYSIS_CADENCE_WRITE_TOOL_NAMES = new Set(['fs.write', 'fs.append'])
const ANALYSIS_CADENCE_MAX_TOOL_RESULTS_SINCE_WRITE = 6
const ANALYSIS_CADENCE_MAX_CHARS_SINCE_WRITE = 32_000

export interface AnalysisWriteCadence {
  toolResultsSinceWrite: number
  charsSinceWrite: number
  hasSeenWrite: boolean
  shouldForceWrite: boolean
}

function messageTextLength(message: Message): number {
  if (typeof message.content === 'string') return message.content.length
  return message.content.reduce((sum, part) => {
    if (part.type === 'text') return sum + part.text.length
    return sum + 100
  }, 0)
}

function hasToolResultForName(messages: Message[], targetName: string): boolean {
  let currentTurnStart = -1
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true) {
      currentTurnStart = index
      break
    }
  }
  const toolNamesById = new Map<string, string>()
  for (const message of messages.slice(currentTurnStart + 1)) {
    for (const toolCall of message.toolCalls ?? []) {
      toolNamesById.set(toolCall.id, toolCall.name)
    }
    if (
      message.role === 'tool'
      && message.toolCallId
      && toolNamesById.get(message.toolCallId) === targetName
    ) {
      return true
    }
  }
  return false
}

type ToolResultEvent = Extract<AgentEvent, { type: 'tool_result' }>

export function evaluateAnalysisWriteCadence(
  messages: Message[],
  availableToolNames: ReadonlySet<string>,
): AnalysisWriteCadence {
  const toolNamesById = new Map<string, string>()
  let toolResultsSinceWrite = 0
  let charsSinceWrite = 0
  let hasSeenWrite = false

  for (const message of messages) {
    for (const toolCall of message.toolCalls ?? []) {
      toolNamesById.set(toolCall.id, toolCall.name)
    }

    if (message.role !== 'tool' || !message.toolCallId) {
      continue
    }

    const toolName = toolNamesById.get(message.toolCallId)
    if (toolName && ANALYSIS_CADENCE_WRITE_TOOL_NAMES.has(toolName)) {
      hasSeenWrite = true
      toolResultsSinceWrite = 0
      charsSinceWrite = 0
      continue
    }

    if (toolName && ANALYSIS_CADENCE_READ_TOOL_NAMES.has(toolName)) {
      toolResultsSinceWrite += 1
      charsSinceWrite += messageTextLength(message)
    }
  }

  const shouldForceWrite =
    [...ANALYSIS_CADENCE_WRITE_TOOL_NAMES].some((toolName) => availableToolNames.has(toolName))
    && hasSeenWrite
    && (
      toolResultsSinceWrite >= ANALYSIS_CADENCE_MAX_TOOL_RESULTS_SINCE_WRITE
      || charsSinceWrite >= ANALYSIS_CADENCE_MAX_CHARS_SINCE_WRITE
    )

  return {
    toolResultsSinceWrite,
    charsSinceWrite,
    hasSeenWrite,
    shouldForceWrite,
  }
}

export function buildAnalysisWriteCadenceMessage(cadence: AnalysisWriteCadence): Message {
  return {
    role: 'system',
    content: [
      '[Analysis cadence supervisor]',
      `You have collected ${cadence.toolResultsSinceWrite} read/search/command tool result(s)`,
      `(~${cadence.charsSinceWrite} chars retained in context) since the last fs.write/fs.append.`,
      'Before any more repository discovery, write or update the evidence ledger and current draft artifacts with fs.write or fs.append.',
      'This is a structural context-budget guard, not a natural-language keyword classifier.',
      'If you cannot write a useful artifact from the evidence already gathered, respond with INCOMPLETE: and name the specific missing evidence.',
    ].join(' '),
  }
}

export interface AgentEngineOptions {
  provider: ILLMProvider
  tools: ToolRegistry
  policy: PolicyEngine
  autonomy: AutonomyLevel
  modeControl?: ModeControl
  semanticRouting?: boolean
  maxIterations?: number
  /** Treat maxIterations as a hard provider-turn cap; disables repair inflation and continuation cycles. */
  hardMaxIterations?: boolean
  /** Maximum model-generated tool calls executed from one response. */
  maxToolCallsPerTurn?: number
  auditLogger?: IAuditLogger
  usageTracker?: UsageTracker
  hookRegistry?: HookRegistry
  deviceName?: string
  thinkingLevel?: string
  maxTokens?: number
  temperature?: number
  textDeltaMode?: 'buffered' | 'live'
  llmCache?: LLMCache
  providerCircuitBreaker?: ProviderCircuitBreaker
  auxiliaryLlmBudget?: AuxiliaryLlmTurnBudget
  approvalCallback?: ApprovalCallback
  evaluateAutoApproval?: AutoApprovalEvaluator
  saveApprovalCheckpoint?: (checkpoint: ApprovalRunCheckpoint) => Promise<void>
  clearApprovalCheckpoint?: (requestId: string) => Promise<void>
  saveRunCheckpoint?: (checkpoint: SessionRunCheckpoint) => Promise<void>
  clearRunCheckpoint?: (sessionId: string) => Promise<void>
  loadToolExecution?: (sessionId: string) => Promise<ToolExecutionRecord | null>
  saveToolExecution?: (record: ToolExecutionRecord) => Promise<void>
  clearToolExecution?: (sessionId: string) => Promise<void>
  /** Optional in-session stats store. When provided, the engine surfaces
   * a "tool feedback" system message on iterations after the first if any
   * tool has been repeatedly failing — gives the model the context it
   * needs to switch tactics instead of retrying the same call. */
  toolStats?: import('./tool-learning/store.js').ToolStatsStore
  workspaceMutationTracker?: import('./workspace-mutation/tracker.js').WorkspaceMutationTracker
  editSnapshotStore?: import('./edit-rollback/store.js').EditSnapshotStore
  /** Persist exact criterion-to-observation links for lean ReAct runs. */
  journalStateBoard?: (sessionId: string, board: AgentStateBoardSnapshot) => Promise<void>
  activeRuns?: ActiveRunRegistry
  journalSteeringConsumed?: (sessionId: string, noteId: string) => Promise<void>
  /**
   * When true, assistant text with no tool call must explicitly start with
   * ANSWER: or INCOMPLETE:. This is a structural completion contract used
   * by skill/high-stakes runs; it does not infer intent from user keywords.
   */
  strictFinalAnswerProtocol?: boolean
  /**
   * When true, run the LLM outcome judge even if the model tried to finish
   * without using any tools. User-facing surfaces use this to catch cases
   * where a file/report/build request is answered as prose instead of action.
   */
  reviewOutcomes?: boolean
  reviewToollessFinals?: boolean
  /**
   * Number of times the runtime may automatically continue the same turn after
   * the normal iteration budget is exhausted. This is a graph-independent
   * long-running task guard; set 0 to require manual /resume only.
   */
  maxContinuationCycles?: number
  /**
   * When false, the engine does not fire its own `pre:/post:agent:run` hooks.
   * The mode-router sets this so it can own a single run-boundary gate that
   * covers react *and* graph turns uniformly (see turn-hooks.ts); direct
   * engine.run() callers (subagents, supervisor, coordinator) leave it at the
   * default `true` so nested runs stay gated. Defaults to true.
   */
  emitAgentRunHooks?: boolean
  /**
   * Daily/session USD spend caps checked before every LLM dispatch. undefined
   * (default) means unlimited. Only priced (costKnown) spend counts.
   */
  spendBudget?: SpendBudgetConfig
}

/**
 * Max writes to a single target before the write-loop guard trips
 * (CLI_BACKLOG.md B6). Operator-tunable via SEPILOTD_WRITE_LOOP_LIMIT; floored
 * at 2 so a normal write-then-fix never trips, and defaulted generously.
 */
function resolveWriteLoopLimit(): number {
  const raw = Number(process.env.SEPILOTD_WRITE_LOOP_LIMIT)
  if (Number.isFinite(raw) && raw >= 2) return Math.floor(raw)
  return DEFAULT_WRITE_LOOP_LIMIT
}

/**
 * Operator override for how many distinct tool observations one turn may hold
 * in working memory before the request is served from a compacted projection.
 * The observation-budget stop reason advertises `raise_budget` as a next
 * action, so that action needs a real knob. Unset means "scale with the model
 * context window".
 */
function resolveMaxTurnToolObservations(): number | undefined {
  const raw = Number(process.env.SEPILOTD_MAX_TURN_TOOL_OBSERVATIONS)
  return Number.isFinite(raw) && raw >= 1 ? Math.floor(raw) : undefined
}

function errorMessageText(error: unknown): string {
  if (error instanceof Error) return error.message
  if (error && typeof error === 'object') {
    const message = (error as { message?: unknown }).message
    if (typeof message === 'string') return message
    const apiMessage = (error as { apiError?: { message?: unknown } }).apiError?.message
    if (typeof apiMessage === 'string') return apiMessage
  }
  return String(error)
}

function isImageInputUnsupportedProviderError(error: unknown): boolean {
  const message = errorMessageText(error).toLowerCase()
  return (
    message.includes('does not support image input') ||
    message.includes('image input is not supported') ||
    message.includes('image input not supported') ||
    (message.includes('unsupported') && message.includes('image')) ||
    (message.includes('vision') && message.includes('not supported'))
  )
}

function stripImageParts(content: Message['content']): {
  content: Message['content']
  removed: number
} {
  if (!Array.isArray(content)) return { content, removed: 0 }

  let removed = 0
  const retained: ContentPart[] = []
  for (const part of content) {
    if (part.type === 'image') {
      removed += 1
    } else {
      retained.push(part)
    }
  }
  if (removed === 0) return { content, removed: 0 }
  if (retained.length === 0) {
    return {
      content: '[image omitted because the provider rejected image input]',
      removed,
    }
  }
  return { content: retained, removed }
}

function stripImagePartsFromMessages(messages: Message[]): number {
  let removed = 0
  for (const message of messages) {
    const stripped = stripImageParts(message.content)
    if (stripped.removed > 0) {
      message.content = stripped.content
      removed += stripped.removed
    }
  }
  return removed
}

function messagesContainImageParts(messages: Message[]): boolean {
  return messages.some((message) =>
    Array.isArray(message.content) && message.content.some((part) => part.type === 'image')
  )
}

function buildImageInputRecoveryMessage(
  provider: string,
  model: string,
  removedImageParts: number,
): Message {
  return {
    role: 'system',
    content: [
      '[provider recovery] The provider rejected image input for this model.',
      `Provider/model: ${provider}/${model}.`,
      `Removed ${removedImageParts} image content part(s) from the retained conversation and disabled new visual attachments for this run.`,
      'Do not claim any omitted screenshot or image was visually inspected.',
      'Do not spend turns searching for a visual workaround or dispatching speculative visual subagents unless the runtime has already identified a concrete vision-capable model/surface for this run.',
      'If visual inspection is required and no such verified vision-capable route is already available, stop the visual-validation loop and end with UNVERIFIED naming this blocker after any non-visual checks you can complete.',
    ].join(' '),
  }
}

export class AgentEngine implements IAgentEngine {
  private provider: ILLMProvider
  private tools: ToolRegistry
  private policy: PolicyEngine
  private autonomy: AutonomyLevel
  /** Absolute wall-clock ceiling for the active run; read once per run entry. */
  private runWallClock: { startedAt: number; budgetMs?: number } | null = null
  private modeControl?: ModeControl
  private semanticRouting: boolean
  private maxIterations: number
  private hardMaxIterations: boolean
  private maxToolCallsPerTurn?: number
  private state: AgentState = 'idle'
  private aborted = false
  private auditLogger?: IAuditLogger
  private usageTracker?: UsageTracker
  private hookRegistry?: HookRegistry
  private deviceName?: string
  private thinkingLevel?: string
  private maxTokens?: number
  private temperature?: number
  private textDeltaMode: 'buffered' | 'live'
  private llmCache?: LLMCache
  private providerCircuitBreaker?: ProviderCircuitBreaker
  private auxiliaryLlmBudget?: AuxiliaryLlmTurnBudget
  private approvalCallback?: ApprovalCallback
  private evaluateAutoApproval?: AutoApprovalEvaluator
  private saveApprovalCheckpoint?: (checkpoint: ApprovalRunCheckpoint) => Promise<void>
  private clearApprovalCheckpoint?: (requestId: string) => Promise<void>
  private saveRunCheckpoint?: (checkpoint: SessionRunCheckpoint) => Promise<void>
  private clearRunCheckpointFn?: (sessionId: string) => Promise<void>
  private loadToolExecution?: (sessionId: string) => Promise<ToolExecutionRecord | null>
  private saveToolExecution?: (record: ToolExecutionRecord) => Promise<void>
  private clearToolExecutionFn?: (sessionId: string) => Promise<void>
  private toolStats?: import('./tool-learning/store.js').ToolStatsStore
  private workspaceMutationTracker?: import('./workspace-mutation/tracker.js').WorkspaceMutationTracker
  private editSnapshotStore?: import('./edit-rollback/store.js').EditSnapshotStore
  private journalStateBoard?: (sessionId: string, board: AgentStateBoardSnapshot) => Promise<void>
  private strictFinalAnswerProtocol: boolean
  private reviewOutcomes: boolean
  private reviewToollessFinals: boolean
  private maxContinuationCycles: number
  private emitAgentRunHooks: boolean
  private spendBudget?: SpendBudgetConfig
  private runAbortController: AbortController | null = null
  private activeRuns?: ActiveRunRegistry
  private journalSteeringConsumed?: (sessionId: string, noteId: string) => Promise<void>
  private liveRunState?: LiveRunState

  constructor(options: AgentEngineOptions) {
    this.modeControl = options.modeControl
    this.semanticRouting = options.semanticRouting ?? false
    this.provider = options.provider
    this.tools = options.tools
    this.policy = options.policy
    this.autonomy = options.autonomy
    this.maxIterations = options.maxIterations ?? 10
    this.hardMaxIterations = options.hardMaxIterations ?? false
    this.maxToolCallsPerTurn = options.maxToolCallsPerTurn
    this.auditLogger = options.auditLogger
    this.usageTracker = options.usageTracker
    this.hookRegistry = options.hookRegistry
    this.deviceName = options.deviceName
    this.thinkingLevel = options.thinkingLevel
    this.maxTokens = options.maxTokens
    this.temperature = options.temperature
    this.textDeltaMode = options.textDeltaMode ?? 'buffered'
    this.llmCache = options.llmCache
    this.providerCircuitBreaker = options.providerCircuitBreaker
    this.auxiliaryLlmBudget = options.auxiliaryLlmBudget
    this.approvalCallback = options.approvalCallback
    this.evaluateAutoApproval = options.evaluateAutoApproval
    this.saveApprovalCheckpoint = options.saveApprovalCheckpoint
    this.clearApprovalCheckpoint = options.clearApprovalCheckpoint
    this.saveRunCheckpoint = options.saveRunCheckpoint
    this.clearRunCheckpointFn = options.clearRunCheckpoint
    this.loadToolExecution = options.loadToolExecution
    this.saveToolExecution = options.saveToolExecution
    this.clearToolExecutionFn = options.clearToolExecution
    this.toolStats = options.toolStats
    this.workspaceMutationTracker = options.workspaceMutationTracker
    this.editSnapshotStore = options.editSnapshotStore
    this.journalStateBoard = options.journalStateBoard
    this.activeRuns = options.activeRuns
    this.journalSteeringConsumed = options.journalSteeringConsumed
    this.strictFinalAnswerProtocol =
      options.strictFinalAnswerProtocol ?? (process.env.SEPILOTD_STRICT_ANSWER_PROTOCOL === '1')
    this.reviewOutcomes = options.reviewOutcomes ?? true
    this.reviewToollessFinals =
      options.reviewToollessFinals ?? (process.env.SEPILOTD_REVIEW_TOOLLESS_FINALS === '1')
    this.maxContinuationCycles = this.hardMaxIterations
      ? 0
      : resolveMaxContinuationCycles(options.maxContinuationCycles)
    this.emitAgentRunHooks = options.emitAgentRunHooks ?? true
    this.spendBudget = options.spendBudget
  }

  private async safeTriggerPostHook(
    event: 'post:agent:run' | 'post:llm:call',
    data: Record<string, unknown>,
  ): Promise<void> {
    // When the run boundary is owned by the mode-router, suppress the engine's
    // own post:agent:run so a single mode-router turn does not double-fire it.
    if (event === 'post:agent:run' && !this.emitAgentRunHooks) return
    try {
      await this.hookRegistry?.trigger({ event, data })
    } catch {
      // Observation hooks must not fail the main agent flow.
    }
  }

  /**
   * ReAct has no graph state object to carry criterion snapshots. At the same
   * terminal boundary, reconstruct only trusted current-turn observations,
   * ask the shared semantic judge for exact criterion links, and journal the
   * same mode-neutral board shape consumed by session evaluation.
   */
  private async *journalReactCriterionEvidence(
    messages: readonly Message[],
    finalContent: string,
    context: AgentContext,
    totalUsage: TokenUsage,
    iteration: number,
    signal?: AbortSignal,
  ): AsyncGenerator<AgentEvent> {
    const contract = context.runContract
    if (!this.journalStateBoard || !contract) return

    const evidenceState = criterionEvidenceStateFromMessages({
      messages,
      contract,
      securityEffectForTool: (tool) => this.tools.securityDescriptor(tool).effect,
    })
    const request = buildCriterionEvidenceReviewRequest({
      model: context.model,
      state: { ...evidenceState, steeringNotes: this.liveRunState?.steeringNotes },
      assistantAnswer: finalContent,
    })
    if (!request) return

    const turnId = buildLlmTurnId(context.sessionId, iteration, 'criterion-evidence-review')
    let activeRequest = request
    let activeTurnId = turnId
    let activeNode = 'criterion-evidence-review'
    try {
      yield buildLlmRequestEvent({
        sessionId: context.sessionId,
        iteration,
        source: 'criterion-evidence-review',
        request,
        turnId,
        providerId: this.provider.id,
        auxiliary: true,
      })
      // Criterion adjudication is a required completion-evidence boundary,
      // not an optional planner/reviewer. Match Graph's guarded call instead
      // of sharing the auxiliary wall-clock budget: a preceding outcome review
      // may legitimately consume that budget, but must not erase exact
      // criterion evidence from an otherwise completed read-only run.
      let response = await guardedProviderChat({
        provider: this.provider,
        request,
        breaker: this.providerCircuitBreaker,
        signal,
      })
      this.recordLlmUsage(context, response.usage)
      totalUsage.inputTokens += response.usage.inputTokens
      totalUsage.outputTokens += response.usage.outputTokens
      await logLlmCallTrace({
        source: 'react',
        mode: 'react',
        sessionId: context.sessionId,
        provider: context.provider,
        model: context.model,
        iteration,
        request,
        response,
        meta: { node: 'criterion-evidence-review', turnId },
      })
      await this.safeTriggerPostHook('post:llm:call', {
        sessionId: context.sessionId,
        provider: context.provider,
        model: context.model,
        iteration,
        request,
        response,
        node: 'criterion-evidence-review',
      })
      let verdicts = parseCriterionEvidenceReviewTransport(
        this.responseTextContent(response),
        response.thinking,
        evidenceState,
      )
      if (verdicts.length < contract.acceptanceCriteria.length) {
        const repairRequest = buildCriterionEvidenceReviewRepairRequest({
          model: context.model,
          state: { ...evidenceState, steeringNotes: this.liveRunState?.steeringNotes },
          assistantAnswer: finalContent,
          acceptedVerdicts: verdicts,
        })
        if (repairRequest) {
          activeRequest = repairRequest
          activeNode = 'criterion-evidence-review-repair'
          activeTurnId = buildLlmTurnId(context.sessionId, iteration, activeNode)
          yield buildLlmRequestEvent({
            sessionId: context.sessionId,
            iteration,
            source: activeNode,
            request: repairRequest,
            turnId: activeTurnId,
            providerId: this.provider.id,
            auxiliary: true,
          })
          response = await guardedProviderChat({
            provider: this.provider,
            request: repairRequest,
            breaker: this.providerCircuitBreaker,
            signal,
          })
          this.recordLlmUsage(context, response.usage)
          totalUsage.inputTokens += response.usage.inputTokens
          totalUsage.outputTokens += response.usage.outputTokens
          await logLlmCallTrace({
            source: 'react',
            mode: 'react',
            sessionId: context.sessionId,
            provider: context.provider,
            model: context.model,
            iteration,
            request: repairRequest,
            response,
            meta: { node: activeNode, turnId: activeTurnId },
          })
          await this.safeTriggerPostHook('post:llm:call', {
            sessionId: context.sessionId,
            provider: context.provider,
            model: context.model,
            iteration,
            request: repairRequest,
            response,
            node: activeNode,
          })
          verdicts = parseCriterionEvidenceReviewTransport(
            this.responseTextContent(response),
            response.thinking,
            evidenceState,
          )
        }
      }
      await logAgentDebugTrace({
        event: 'supervisor.criterion-evidence-review',
        source: 'react',
        sessionId: context.sessionId,
        runId: context.sessionId,
        mode: 'react',
        iteration,
        status: verdicts.length > 0 ? 'accepted' : 'invalid',
        data: {
          verdictCount: verdicts.length,
          criterionCount: contract.acceptanceCriteria.length,
          toolResultCount: evidenceState.toolCallHistory?.length ?? 0,
        },
      })
      if (verdicts.length === 0) return
      await this.journalStateBoard(
        context.sessionId,
        buildCriterionEvidenceBoardSnapshot({
          contract,
          state: evidenceState,
          verdicts,
        }),
      )
    } catch (error) {
      await logLlmCallTrace({
        source: 'react',
        mode: 'react',
        sessionId: context.sessionId,
        provider: context.provider,
        model: context.model,
        iteration,
        request: activeRequest,
        error: error instanceof Error ? error.message : String(error),
        meta: { node: activeNode, turnId: activeTurnId, failed: true },
      })
      if (isAbortError(error) || (signal?.aborted ?? false)) throw error
      // Session evidence remains honestly unverified when the criterion judge
      // is unavailable; never fail or rewrite an otherwise valid user answer.
    }
  }

  private async finalizeReactContextLengthFailure(
    context: AgentContext,
    totalUsage: TokenUsage,
    iteration: number,
    message: string,
  ): Promise<void> {
    await this.safeTriggerPostHook('post:agent:run', {
      status: 'error',
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      usage: { ...totalUsage },
      error: message,
      iteration,
    })
    await logAgentRunTrace({
      source: 'react',
      status: 'error',
      mode: 'react',
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      iteration,
      usage: { ...totalUsage },
      error: message,
    })
    await this.clearRunCheckpoint(context.sessionId)
  }

  private async chatWithRetry(
    request: ChatRequest,
    signal?: AbortSignal,
    maxRetries = 3,
  ): Promise<ChatResponse> {
    return guardedProviderChat({
      provider: this.provider,
      request,
      signal,
      breaker: this.providerCircuitBreaker,
      maxRetries: Math.max(0, maxRetries - 1),
      cache: this.llmCache,
    })
  }

  private responseTextContent(response: ChatResponse): string {
    return typeof response.message.content === 'string'
      ? response.message.content
      : response.message.content
          .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
          .map((part) => part.text)
          .join('')
  }

  private shouldFallbackFromEmptyStream(response: ChatResponse): boolean {
    return (
      this.responseTextContent(response).trim().length === 0 &&
      !response.thinking?.trim() &&
      (response.message.toolCalls?.length ?? 0) === 0 &&
      response.usage.inputTokens === 0 &&
      response.usage.outputTokens === 0 &&
      response.finishReason === 'stop'
    )
  }

  private withParsedPromptTextToolCalls(
    response: ChatResponse,
    allowedToolNames?: Set<string>,
    modelId?: string,
  ): ChatResponse {
    const availableToolNames = new Set([...this.tools.toToolDefinitions(), ...(this.modeControl?.tools ?? [])].map((tool) => tool.name))
    const rawContent = this.responseTextContent(response)
    const parsedTransport = stripPromptReActThinkingArtifacts(rawContent)
    const content = normalizeModelAnswerProtocol(
      resolveExplicitFinalTransportText(rawContent, response.thinking),
      modelId ? this.resolveModelInfo(modelId) : undefined,
    )
    const normalizedThinking = [response.thinking, parsedTransport.thinking]
      .filter((value): value is string => Boolean(value?.trim()))
      .join('\n')
    const normalizedResponse = {
      ...response,
      message: content === rawContent ? response.message : { ...response.message, content },
      thinking: normalizedThinking || undefined,
    }
    if (!canExecuteCompletedToolCalls(normalizedResponse.finishReason)) {
      return {
        ...normalizedResponse,
        message: {
          ...normalizedResponse.message,
          content: normalizedResponse.finishReason === 'content_filter'
            ? INTERRUPTED_USER_FACING_RESPONSE
            : content,
          toolCalls: undefined,
        },
      }
    }
    const isExplicitFinal = hasPromptFinalEnvelopeIntent(content)
    if (isExplicitFinal) {
      const hasConflictingNativeToolCall = (normalizedResponse.message.toolCalls?.length ?? 0) > 0
      const finalCandidate = extractPromptFinalCandidate(content)
      return {
        ...normalizedResponse,
        message: {
          ...normalizedResponse.message,
          content:
            hasConflictingNativeToolCall || containsPromptToolCallEnvelope(content)
              ? UNUSABLE_PROMPT_TOOL_CALL_OUTPUT
              : finalCandidate,
          toolCalls: undefined,
        },
        finishReason:
          normalizedResponse.finishReason === 'length' || normalizedResponse.finishReason === 'content_filter'
            ? normalizedResponse.finishReason
            : 'stop',
      }
    }

    if ((normalizedResponse.message.toolCalls?.length ?? 0) > 0) {
      const originalToolCalls = normalizedResponse.message.toolCalls ?? []
      const repairedToolCalls = repairNativeToolCalls(
        originalToolCalls,
        availableToolNames,
        allowedToolNames,
      )
      if (repairedToolCalls.length === originalToolCalls.length) {
        return {
          ...normalizedResponse,
          message: {
            ...normalizedResponse.message,
            toolCalls: repairedToolCalls,
          },
          finishReason: 'tool_use',
        }
      }

      return {
        ...normalizedResponse,
        message: {
          ...normalizedResponse.message,
          content: UNUSABLE_PROMPT_TOOL_CALL_OUTPUT,
          toolCalls: undefined,
        },
        finishReason: 'stop',
      }
    }

    const isProtectedLiveProgress = this.textDeltaMode === 'live'
      && isLikelyInterimProgressUpdate(content)
    if (isProtectedLiveProgress) {
      if (!containsPromptToolCallEnvelope(content)) {
        return normalizedResponse
      }
      return {
        ...normalizedResponse,
        message: {
          ...normalizedResponse.message,
          content: UNUSABLE_PROMPT_TOOL_CALL_OUTPUT,
          toolCalls: undefined,
        },
        finishReason:
          normalizedResponse.finishReason === 'length' || normalizedResponse.finishReason === 'content_filter'
            ? normalizedResponse.finishReason
            : 'stop',
      }
    }

    const rawParsedToolCalls = parsePromptToolCalls(content)
    const parsedToolCalls = repairPromptToolCalls(
      rawParsedToolCalls,
      availableToolNames,
      allowedToolNames,
    )
    if (rawParsedToolCalls.length > 0 && parsedToolCalls.length !== rawParsedToolCalls.length) {
      return {
        ...normalizedResponse,
        message: {
          ...normalizedResponse.message,
          content: UNUSABLE_PROMPT_TOOL_CALL_OUTPUT,
          toolCalls: undefined,
        },
        finishReason: 'stop',
      }
    }
    if (parsedToolCalls.length > 0) {
      return {
        ...normalizedResponse,
        message: {
          ...normalizedResponse.message,
          content: content.trim() || '<tool_call/>',
          toolCalls: parsedToolCalls,
        },
        finishReason: 'tool_use',
      }
    }

    if (containsPromptToolCallEnvelope(content)) {
      return {
        ...normalizedResponse,
        message: {
          ...normalizedResponse.message,
          content: UNUSABLE_PROMPT_TOOL_CALL_OUTPUT,
          toolCalls: undefined,
        },
        finishReason:
          normalizedResponse.finishReason === 'length' || normalizedResponse.finishReason === 'content_filter'
            ? normalizedResponse.finishReason
            : 'stop',
      }
    }

    return normalizedResponse
  }

  private async fallbackFromEmptyStream(
    request: ChatRequest,
    signal?: AbortSignal,
  ): Promise<ChatResponse | null> {
    try {
      return await this.chatWithRetry(request, signal, 1)
    } catch {
      return null
    }
  }

  /**
   * Stream LLM response, yielding text_delta events for each text chunk.
   * Returns the assembled ChatResponse for tool processing.
   */
  private async *streamWithDeltas(
    request: ChatRequest,
    signal?: AbortSignal,
    allowedToolNames?: Set<string>,
    suppressTextDeltas = false,
    deferredTextDeltas?: string[],
  ): AsyncGenerator<AgentEvent, ChatResponse> {
    let text = ''
    let thinking = ''
    const toolCalls: ToolCall[] = []
    const toolCallArgs: Map<string, string> = new Map()
    const toolProgress = createToolCallProgressTracker()
    const totalUsage: TokenUsage = { inputTokens: 0, outputTokens: 0 }
    let finishReason: ChatResponse['finishReason'] = 'stop'
    let sawDone = false
    let sawNonTerminalChunk = false
    const liveTextDeltas = this.textDeltaMode === 'live'
    const bufferedLiveTextDeltas: string[] = []
    const liveTextDeltaEmitter =
      liveTextDeltas
        ? createLiveTextDeltaEmitter({
            // Plain text is only known to be a final answer once the provider
            // turn ends with no tool calls. ANSWER: chunks can preserve their
            // boundaries, while unmarked text falls back to one safe buffered
            // delta after the response is assembled.
            allowUnmarkedFinal: false,
            isAllowedUnmarkedFinalStart: (candidate) =>
              !containsPromptToolCallMarkup(candidate),
          })
        : null

    for await (const chunk of guardedProviderStream({
      provider: this.provider,
      request,
      signal,
      breaker: this.providerCircuitBreaker,
    })) {
      if (chunk.type !== 'done') {
        sawNonTerminalChunk = true
      }
      switch (chunk.type) {
        case 'text':
          text += chunk.text
          if (liveTextDeltaEmitter && chunk.text) {
            const textDelta = liveTextDeltaEmitter.push(chunk.text)
            if (textDelta) {
              if (suppressTextDeltas) {
                bufferedLiveTextDeltas.push(textDelta)
              } else {
                // `live` only enters this path after an explicit ANSWER:
                // stem. Forward safe answer chunks as the provider produces
                // them; strict/reviewed turns keep using the deferred buffer.
                yield { type: 'text_delta', text: textDelta }
              }
            }
          }
          break
        case 'thinking':
          thinking += chunk.text
          yield { type: 'thinking', content: chunk.text }
          break
        case 'tool_call_start':
          if (chunk.toolCall.id && chunk.toolCall.name) {
            toolProgress.start(chunk.toolCall)
            toolCalls.push({
              id: chunk.toolCall.id,
              name: chunk.toolCall.name,
              arguments: {},
            })
            toolCallArgs.set(chunk.toolCall.id, '')
          }
          break
        case 'tool_call_delta':
          if (chunk.toolCallId) {
            const prev = toolCallArgs.get(chunk.toolCallId) ?? ''
            toolCallArgs.set(chunk.toolCallId, prev + chunk.delta)
            const progressEvent = toolProgress.delta(chunk.toolCallId, chunk.delta)
            if (progressEvent) {
              yield progressEvent
            }
          }
          break
        case 'usage':
          totalUsage.inputTokens += chunk.usage.inputTokens
          totalUsage.outputTokens += chunk.usage.outputTokens
          if (chunk.usage.thinkingTokens) {
            totalUsage.thinkingTokens = (totalUsage.thinkingTokens ?? 0) + chunk.usage.thinkingTokens
          }
          if (chunk.usage.cacheReadTokens) {
            totalUsage.cacheReadTokens = (totalUsage.cacheReadTokens ?? 0) + chunk.usage.cacheReadTokens
          }
          if (chunk.usage.cacheCreationTokens) {
            totalUsage.cacheCreationTokens =
              (totalUsage.cacheCreationTokens ?? 0) + chunk.usage.cacheCreationTokens
          }
          break
        case 'done':
          sawDone = true
          finishReason = chunk.finishReason
          break
        case 'error':
          throw new Error(chunk.error.message)
      }
    }
    // The adapter only emits `done` when the provider reported a finish
    // reason. Reaching EOF without one is a transport failure, not a known
    // output-cap hit; keep the conservative `length` classification but record
    // the distinction so recovery can retry instead of inventing a
    // continuation point.
    let finishDetail: ChatResponse['finishDetail']
    if (!sawDone && sawNonTerminalChunk) {
      finishReason = 'length'
      finishDetail = 'incomplete_stream'
    }

    const trailingLiveText = liveTextDeltaEmitter?.flush()
    if (trailingLiveText) {
      if (suppressTextDeltas) {
        bufferedLiveTextDeltas.push(trailingLiveText)
      } else {
        yield { type: 'text_delta', text: trailingLiveText }
      }
    }

    // Never execute a tool whose non-empty args failed to parse: that is a
    // truncated tool call (typically finishReason==='length'). Drop it and
    // surface the truncation so the run can retry with more budget instead of
    // silently running the tool with `{}` and losing the intended arguments.
    const truncatedToolCallIds = new Set<string>()
    for (const tc of toolCalls) {
      const parsed = parseToolCallArguments(toolCallArgs.get(tc.id))
      if (parsed.truncated) {
        truncatedToolCallIds.add(tc.id)
      } else {
        tc.arguments = parsed.arguments
      }
    }
    const usableToolCalls =
      truncatedToolCallIds.size > 0
        ? toolCalls.filter((tc) => !truncatedToolCallIds.has(tc.id))
        : toolCalls
    if (truncatedToolCallIds.size > 0 && finishReason !== 'content_filter') {
      finishReason = 'length'
      // A cut-off tool-call argument body is positive evidence that generation
      // stopped mid-token, whatever the stream did afterwards.
      finishDetail = 'truncated_output'
    }

    const parsedTransport = stripPromptReActThinkingArtifacts(text)
    const normalizedTransportText = normalizeModelAnswerProtocol(
      resolveExplicitFinalTransportText(text, thinking),
      this.resolveModelInfo(request.model),
    )
    const normalizedThinking = [thinking, parsedTransport.thinking]
      .filter((value) => value.trim())
      .join('\n')
    const streamedResponse = this.withParsedPromptTextToolCalls(
      {
        message: {
          role: 'assistant' as const,
          content: normalizedTransportText,
          toolCalls: usableToolCalls.length > 0 ? usableToolCalls : undefined,
        },
        thinking: normalizedThinking || undefined,
        usage: totalUsage,
        finishReason,
        ...(finishDetail ? { finishDetail } : {}),
      },
      allowedToolNames,
    )

    if (this.shouldFallbackFromEmptyStream(streamedResponse)) {
      const fallback = await this.fallbackFromEmptyStream(request, signal)
      if (fallback) {
        const parsedFallback = this.withParsedPromptTextToolCalls(
          fallback,
          allowedToolNames,
          request.model,
        )
        if (fallback.thinking) {
          yield { type: 'thinking', content: fallback.thinking }
        }
        const fallbackContent = this.responseTextContent(parsedFallback)
        const finalFallbackContent = stripFinalAnswerStem(fallbackContent)
        if (
          canPublishUserFacingText(parsedFallback.finishReason)
          && suppressTextDeltas
          && deferredTextDeltas
          && (parsedFallback.message.toolCalls?.length ?? 0) === 0
          && !containsPromptToolCallMarkup(fallbackContent)
          && !isLikelyInterimProgressUpdate(fallbackContent)
          && finalFallbackContent
        ) {
          deferredTextDeltas.push(finalFallbackContent)
        }
        if (
          canPublishUserFacingText(parsedFallback.finishReason)
          && !suppressTextDeltas &&
          (parsedFallback.message.toolCalls?.length ?? 0) === 0 &&
          (
            !liveTextDeltas
            || (
              !containsPromptToolCallMarkup(fallbackContent)
              && !isLikelyInterimProgressUpdate(fallbackContent)
            )
          ) &&
          finalFallbackContent
        ) {
          yield { type: 'text_delta', text: finalFallbackContent }
        }
        return parsedFallback
      }
    }

    const streamedContent = this.responseTextContent(streamedResponse)
    const normalizedPublishText = hasPromptFinalEnvelopeIntent(normalizedTransportText)
      ? extractPromptFinalCandidate(normalizedTransportText)
      : normalizedTransportText
    const streamedContentMatchesRawText = streamedContent === normalizedPublishText
    if (
      canPublishUserFacingText(streamedResponse.finishReason)
      && streamedContentMatchesRawText
      && (streamedResponse.message.toolCalls?.length ?? 0) === 0
      && !containsPromptToolCallMarkup(streamedContent)
    ) {
      if (
        liveTextDeltas
        && bufferedLiveTextDeltas.length === 0
        && !liveTextDeltaEmitter?.hasEmitted()
        && streamedContent
        && !isLikelyInterimProgressUpdate(streamedContent)
      ) {
        const bufferedFinalText = stripFinalAnswerStem(streamedContent)
        if (bufferedFinalText) {
          bufferedLiveTextDeltas.push(bufferedFinalText)
        }
      }
      if (suppressTextDeltas) {
        deferredTextDeltas?.push(...bufferedLiveTextDeltas)
      } else {
        for (const textDelta of bufferedLiveTextDeltas) {
          yield { type: 'text_delta', text: textDelta }
        }
      }
    }

    if (
      canPublishUserFacingText(streamedResponse.finishReason)
      && streamedContentMatchesRawText
      && !suppressTextDeltas
      && !liveTextDeltas
      && (streamedResponse.message.toolCalls?.length ?? 0) === 0
      && streamedContent
      && !liveTextDeltaEmitter?.hasEmitted()
    ) {
      const finalText = stripFinalAnswerStem(streamedContent)
      if (finalText && (!liveTextDeltaEmitter || !isLikelyInterimProgressUpdate(streamedContent))) {
        yield { type: 'text_delta', text: finalText }
      }
    }

    return streamedResponse
  }

  private async *streamWithPromptReActFallback(
    request: ChatRequest,
    signal?: AbortSignal,
    allowedToolNames?: Set<string>,
    suppressTextDeltas = false,
    deferredTextDeltas?: string[],
  ): AsyncGenerator<AgentEvent, ChatResponse> {
    const provider = this.provider
    const breaker = this.providerCircuitBreaker
    const responseModelInfo = this.resolveModelInfo(request.model)
    const availableToolNames = new Set([...this.tools.toToolDefinitions(), ...(this.modeControl?.tools ?? [])].map((tool) => tool.name))

    const runPromptTurn = async function* (
      this: void,
      turnRequest: ChatRequest,
    ): AsyncGenerator<
      AgentEvent,
      {
        rawText: string
        toolCalls: ToolCall[]
        usage: TokenUsage
        finishReason: ChatResponse['finishReason']
        finishDetail?: ChatResponse['finishDetail']
        thinking?: string
      },
      void
    > {
      const textChunks: string[] = []
      let thinking = ''
      const toolCalls: ToolCall[] = []
      const toolCallArgs: Map<string, string> = new Map()
      const toolProgress = createToolCallProgressTracker()
      const usage: TokenUsage = { inputTokens: 0, outputTokens: 0 }
      let finishReason: ChatResponse['finishReason'] = 'stop'
      let sawDone = false
      let sawNonTerminalChunk = false

      for await (const chunk of guardedProviderStream({
        provider,
        request: turnRequest,
        signal,
        breaker,
      })) {
        if (chunk.type !== 'done') {
          sawNonTerminalChunk = true
        }
        switch (chunk.type) {
          case 'text':
            textChunks.push(chunk.text)
            break
          case 'thinking':
            thinking += chunk.text
            yield { type: 'thinking', content: chunk.text }
            break
          case 'tool_call_start':
            if (chunk.toolCall.id && chunk.toolCall.name) {
              toolProgress.start(chunk.toolCall)
              toolCalls.push({
                id: chunk.toolCall.id,
                name: chunk.toolCall.name,
                arguments: {},
              })
              toolCallArgs.set(chunk.toolCall.id, '')
            }
            break
          case 'tool_call_delta':
            if (chunk.toolCallId) {
              const previous = toolCallArgs.get(chunk.toolCallId) ?? ''
              toolCallArgs.set(chunk.toolCallId, previous + chunk.delta)
              const progressEvent = toolProgress.delta(chunk.toolCallId, chunk.delta)
              if (progressEvent) {
                yield progressEvent
              }
            }
            break
          case 'usage':
            usage.inputTokens += chunk.usage.inputTokens
            usage.outputTokens += chunk.usage.outputTokens
            if (chunk.usage.thinkingTokens) {
              usage.thinkingTokens = (usage.thinkingTokens ?? 0) + chunk.usage.thinkingTokens
            }
            if (chunk.usage.cacheReadTokens) {
              usage.cacheReadTokens = (usage.cacheReadTokens ?? 0) + chunk.usage.cacheReadTokens
            }
            if (chunk.usage.cacheCreationTokens) {
              usage.cacheCreationTokens = (usage.cacheCreationTokens ?? 0) + chunk.usage.cacheCreationTokens
            }
            break
          case 'done':
            sawDone = true
            finishReason = chunk.finishReason
            break
          case 'error':
            throw new Error(chunk.error.message)
        }
      }
      // See `streamWithDeltas`: EOF without a provider finish reason is a
      // transport failure, not a known output-cap hit.
      let finishDetail: ChatResponse['finishDetail']
      if (!sawDone && sawNonTerminalChunk) {
        finishReason = 'length'
        finishDetail = 'incomplete_stream'
      }

      // Drop truncated tool calls (non-empty unparseable args) rather than
      // executing them with `{}`; flag the turn as length-truncated.
      const truncatedToolCallIds = new Set<string>()
      for (const toolCall of toolCalls) {
        const parsed = parseToolCallArguments(toolCallArgs.get(toolCall.id))
        if (parsed.truncated) {
          truncatedToolCallIds.add(toolCall.id)
        } else {
          toolCall.arguments = parsed.arguments
        }
      }
      if (truncatedToolCallIds.size > 0) {
        for (let i = toolCalls.length - 1; i >= 0; i--) {
          if (truncatedToolCallIds.has(toolCalls[i]!.id)) toolCalls.splice(i, 1)
        }
        if (finishReason !== 'content_filter') {
          finishReason = 'length'
          finishDetail = 'truncated_output'
        }
      }

      const strippedText = stripPromptReActThinkingArtifacts(textChunks.join(''))
      const rawText = strippedText.text
      thinking = [thinking, strippedText.thinking].filter(Boolean).join('\n')
      const promptToolParseText = rawText.trim() ? rawText : thinking ?? ''
      const promptTextIsExplicitFinal = hasPromptFinalEnvelopeIntent(promptToolParseText)
      const normalizedResponseText = normalizeModelAnswerProtocol(
        resolveExplicitFinalTransportText(rawText, thinking),
        responseModelInfo,
      )

      const rawParsedPromptToolCalls =
        toolCalls.length === 0 && !promptTextIsExplicitFinal
          ? parsePromptToolCalls(promptToolParseText)
          : []
      const candidateToolCalls = toolCalls.length > 0
        ? toolCalls
        : rawParsedPromptToolCalls
      const repairedToolCalls = toolCalls.length > 0
        ? repairNativeToolCalls(candidateToolCalls, availableToolNames, allowedToolNames)
        : repairPromptToolCalls(candidateToolCalls, availableToolNames, allowedToolNames)
      const rejectedToolCallBatch = (
        promptTextIsExplicitFinal && toolCalls.length > 0
      ) || repairedToolCalls.length !== candidateToolCalls.length
      const acceptedToolCalls = canExecuteCompletedToolCalls(finishReason) && !rejectedToolCallBatch
        ? repairedToolCalls
        : []
      const safeRawText = finishReason === 'content_filter'
        ? INTERRUPTED_USER_FACING_RESPONSE
        : canExecuteCompletedToolCalls(finishReason) && rejectedToolCallBatch
          ? UNUSABLE_PROMPT_TOOL_CALL_OUTPUT
          : normalizedResponseText

      return {
        rawText: safeRawText,
        toolCalls: acceptedToolCalls,
        usage,
        finishReason,
        finishDetail,
        thinking: thinking || undefined,
      }
    }

    const firstTurn = runPromptTurn(request)
    let firstTurnResult = await firstTurn.next()
    while (!firstTurnResult.done) {
      yield firstTurnResult.value
      firstTurnResult = await firstTurn.next()
    }

    let {
      rawText,
      toolCalls: resolvedToolCalls,
      usage,
      finishReason,
      finishDetail,
      thinking,
    } = firstTurnResult.value

    if (
      canPublishUserFacingText(finishReason)
      && resolvedToolCalls.length === 0 &&
      shouldAttemptPromptReActRepair(rawText)
      // A native -> prompt transport recovery already injected an exact,
      // provider-neutral response contract. Retrying a fully empty prompt
      // reply with a second format-only prompt cannot recover any malformed
      // content because there is no content to repair. Let the graph's shared
      // no-progress guard terminate this model cleanly instead of paying for
      // another identical provider turn.
      && !(
        rawText.trim().length === 0
        && hasToolTransportRecoveryMessage(request.messages)
      )
    ) {
      const repairTurn = runPromptTurn({
        ...request,
        messages: buildPromptReActRepairMessages(request.messages, rawText),
      })
      let repairTurnResult = await repairTurn.next()
      while (!repairTurnResult.done) {
        yield repairTurnResult.value
        repairTurnResult = await repairTurn.next()
      }

      rawText = repairTurnResult.value.rawText
      resolvedToolCalls = repairTurnResult.value.toolCalls
      usage = {
        inputTokens: usage.inputTokens + repairTurnResult.value.usage.inputTokens,
        outputTokens: usage.outputTokens + repairTurnResult.value.usage.outputTokens,
      }
      finishReason = repairTurnResult.value.finishReason
      finishDetail = repairTurnResult.value.finishDetail
      thinking = [thinking, repairTurnResult.value.thinking].filter(Boolean).join('\n') || undefined
    }

    let promptFinalContent =
      extractTaggedPayload(rawText, ['final', 'answer']) ?? rawText.trim()
    const streamedResponse: ChatResponse = {
      message: {
        role: 'assistant',
        content:
          resolvedToolCalls.length > 0
            ? rawText.trim() || '<tool_call/>'
            : promptFinalContent,
        toolCalls: resolvedToolCalls.length > 0 ? resolvedToolCalls : undefined,
      },
      thinking,
      usage,
      finishReason: resolvedToolCalls.length > 0 ? 'tool_use' : finishReason,
      ...(finishDetail ? { finishDetail } : {}),
    }

    if (this.shouldFallbackFromEmptyStream(streamedResponse)) {
      const fallback = await this.fallbackFromEmptyStream(request, signal)
      if (fallback) {
        const parsedFallback = this.withParsedPromptTextToolCalls(
          fallback,
          allowedToolNames,
          request.model,
        )
        rawText = this.responseTextContent(parsedFallback)
        resolvedToolCalls = parsedFallback.message.toolCalls ?? []
        usage = parsedFallback.usage
        finishReason = parsedFallback.finishReason
        finishDetail = parsedFallback.finishDetail
        thinking = parsedFallback.thinking
        promptFinalContent =
          extractTaggedPayload(rawText, ['final', 'answer']) ?? rawText.trim()
      }
    }

    if (
      canPublishUserFacingText(finishReason)
      && resolvedToolCalls.length === 0
      && containsPromptToolCallEnvelope(rawText)
    ) {
      rawText = UNUSABLE_PROMPT_TOOL_CALL_OUTPUT
      promptFinalContent = UNUSABLE_PROMPT_TOOL_CALL_OUTPUT
      if (finishReason === 'tool_use') {
        finishReason = 'stop'
      }
    }

    const taggedFinal = extractTaggedPayload(rawText, ['final', 'answer'])
    const explicitFinal = taggedFinal !== null
      ? stripFinalAnswerStem(taggedFinal)
      : hasFinalAnswerStem(rawText)
        ? stripFinalAnswerStem(rawText)
        : null
    const finalTextDelta = this.textDeltaMode === 'live'
      ? explicitFinal
      : taggedFinal !== null
        ? stripFinalAnswerStem(taggedFinal)
        : extractPromptFinalOutput(rawText)
    if (
      canPublishUserFacingText(finishReason)
      && resolvedToolCalls.length === 0
      && !containsPromptToolCallMarkup(rawText)
      && finalTextDelta
      && (explicitFinal !== null || !isLikelyInterimProgressUpdate(finalTextDelta))
    ) {
      if (suppressTextDeltas) {
        deferredTextDeltas?.push(finalTextDelta)
      } else {
        yield { type: 'text_delta', text: finalTextDelta }
      }
    }

    return {
      message: {
        role: 'assistant',
        content:
          resolvedToolCalls.length > 0
            ? rawText.trim() || '<tool_call/>'
            : promptFinalContent,
        toolCalls: resolvedToolCalls.length > 0 ? resolvedToolCalls : undefined,
      },
      thinking,
      usage,
      finishReason: resolvedToolCalls.length > 0 ? 'tool_use' : finishReason,
      ...(finishDetail ? { finishDetail } : {}),
    }
  }

  /**
   * Exact match only. The previous `?? models[0]` fallback silently dressed an
   * unknown model in another model's specs, and `models[0]` is whatever order
   * the provider returned — alphabetical for Ollama, so an embedding model can
   * sort first. A model the operator pulled but never added to config then ran
   * with `toolUse: false` and `maxOutputTokens: 0`, which reads downstream as
   * "no native tool calling, 2 output tokens" rather than as a lookup miss.
   *
   * Every caller already uses optional chaining and has its own default, so an
   * unknown model now falls back per-field instead of wholesale.
   */
  private resolveModelInfo(modelId: string) {
    return this.provider.models.find((model) => model.id === modelId)
  }

  private canAttachToolVisualContent(
    context: Pick<AgentContext, 'provider' | 'model'>,
    disabledForRun = false,
  ): boolean {
    return (
      this.resolveModelInfo(context.model)?.capabilities.vision === true
      && !disabledForRun
      && !isProviderModelImageInputRejected(context.provider, context.model)
    )
  }

  private supportsNativeToolUse(modelId: string): boolean {
    // `adaptivePromptReact` enables an observed-failure fallback; it does not
    // disable a transport that the model explicitly declares it supports.
    // Starting prompt-react eagerly is especially unsafe for OpenAI-compatible
    // servers that consume tool-envelope sentinel tokens in their chat
    // template: the visible response can then be empty even though native
    // function calling works correctly. Graph runners already use native-first
    // semantics; continueRun applies the same bounded, evidence-driven fallback
    // when an adaptive model's native response is structurally unusable.
    const model = this.resolveModelInfo(modelId)
    // An unknown model means the catalog is stale, not that the model lacks
    // tool calling — a freshly pulled Ollama model is the common case. Assume
    // native: guessing wrong surfaces a provider error naming the model, while
    // guessing `false` silently drops every tool from the request and renders
    // the whole tool surface into the prompt instead, which looks to the user
    // like the agent simply refusing to act.
    return resolveModelToolTransport(model, 'react').initial === 'native'
  }

  /**
   * The system prompt opens with a plain `Available tools: a, b, c…` list —
   * 2,932 characters on a full 171-tool registry, re-sent on every call. It
   * carries nothing the request does not already say: native runs get the same
   * names in the tool schemas, and prompt-react runs get them in the rendered
   * catalog (`- name: description / schema=…`), which is appended to the same
   * request. The graph path already drops it for native runs; runs that
   * execute without a graph are where the whole registry is sent.
   */
  private stripDuplicateToolNameList(systemPrompt: string | undefined): string | undefined {
    if (!systemPrompt) return systemPrompt
    return systemPrompt
      .replace(/^Available tools:[^\n]*\n?/m, '')
      .replace(new RegExp(`^${TOOL_APPROVAL_POSTURE_PROMPT_PREFIX}[^\\n]*\\n?`, 'm'), '')
      .replace(
        /^Your autonomy level is: [^.\n]+\.$/m,
        `Your autonomy level is: ${this.autonomy}.`,
      )
  }

  /**
   * Static approval posture of every registered tool for this run. Evaluated
   * from rule modes, autonomy and the primary agent only — no input, no
   * execution — so the model learns which tools will prompt or are
   * unavailable before it plans around them.
   */
  private describeRunToolApprovalPosture(context: AgentContext): ToolApprovalPostureMap {
    return describeToolApprovalPosture(
      this.tools.list(),
      this.policy,
      this.autonomy,
      context.primaryAgentId,
      { autoApprove: context.autoApprove },
    )
  }

  private buildInitialMessages(input: string, context: AgentContext): Message[] {
    const messages: Message[] = []
    const autonomyHint = buildAutonomyHint(this.autonomy, context.primaryAgentId)
    const postureParagraph = buildToolApprovalPostureParagraph(
      this.describeRunToolApprovalPosture(context),
    )
    const systemPrompt = this.stripDuplicateToolNameList(context.systemPrompt)
    if (systemPrompt || autonomyHint || postureParagraph) {
      messages.push({
        role: 'system',
        content: appendAnswerProtocolSystemPrompt(
          [systemPrompt, autonomyHint, postureParagraph].filter(Boolean).join('\n\n'),
        ),
      })
    } else {
      messages.push({
        role: 'system',
        content: appendAnswerProtocolSystemPrompt(undefined),
      })
    }
    if (context.relevantMemories?.length) {
      messages.push({
        role: 'system',
        content: [
          '[Relevant memories]',
          'These are historical memory or document snippets and may be stale; verify volatile facts before relying on them.',
          'When a snippet includes an id such as mem:<id>, cite it in that form when you reference it so the user can correct the right entry.',
          context.relevantMemories.join('\n'),
        ].join('\n'),
      })
    }
    const runContractMessage = formatRunContractForPrompt(context.runContract)
    if (runContractMessage) {
      messages.push({ role: 'system', content: runContractMessage })
    }
    // Generated context is rebuilt on mode transfer. Mark only these messages,
    // retaining history, compaction summaries and tool receipts verbatim.
    for (const message of messages) {
      message.metadata = { ...message.metadata, runtimeContext: true }
    }
    if (context.previousMessages?.length) {
      messages.push(...context.previousMessages.map(cloneMessage))
    }
    // chat-stream wraps multimodal turns with a sentinel so we can
    // restore them as ContentPart[] here. Plain text inputs flow through
    // unchanged. If the route already persisted this exact input as the last
    // user message, mark that message as the current turn boundary; otherwise
    // append a marked user message. The marker survives checkpoints and keeps
    // synthetic user-role continuation prompts from becoming a false boundary.
    const decodedParts = decodeMultimodalInput(input)
    const userContent: Message['content'] = context.currentUserContent ?? decodedParts ?? input
    return beginCurrentAgentTurnUserMessage(
      messages,
      userContent,
      context.currentUserContent ? input : undefined,
    )
  }

  private async persistApprovalCheckpoint(
    requestId: string,
    context: AgentContext,
    messages: Message[],
    toolCalls: ToolCall[],
    currentToolIndex: number,
    totalUsage: TokenUsage,
    iteration: number,
  ): Promise<void> {
    if (!this.saveApprovalCheckpoint) {
      return
    }

    await this.saveApprovalCheckpoint({
      autonomy: this.autonomy,
      requestId,
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      mode: this.modeControl?.state.mode ?? 'react',
      modeControlState: this.modeControl ? structuredClone(this.modeControl.state) : undefined,
      messages: withSteeringCheckpoint(messages, this.liveRunState?.steeringNotes).map(cloneMessage),
      cwd: context.cwd,
      workspaceRoot: context.workspaceRoot,
      workspaceIsolation: context.workspaceIsolation,
      scopeTags: cloneCheckpointScopeTags(context.scopeTags),
      executionSkillIds: cloneCheckpointExecutionSkillIds(context.executionSkillIds),
      skillToolNames: cloneCheckpointSkillToolNames(context.skillToolNames),
      toolAllowlist: cloneCheckpointToolAllowlist(context.toolAllowlist),
      skillExecutionPolicies: cloneCheckpointSkillExecutionPolicies(
        context.skillExecutionPolicies,
      ),
      requireToolApproval: context.requireToolApproval,
      runContract: context.runContract,
      toolCalls: toolCalls.map(cloneToolCall),
      currentToolIndex,
      totalUsage: { ...totalUsage },
      iteration,
      maxIterations: this.maxIterations,
      thinkingLevel: this.thinkingLevel,
      ...(this.textDeltaMode === 'live' ? { textDeltaMode: 'live' as const } : {}),
      createdAt: new Date().toISOString(),
    })
  }

  private async clearCheckpoint(requestId: string): Promise<void> {
    await this.clearApprovalCheckpoint?.(requestId)
  }

  private async persistRunCheckpoint(
    context: AgentContext,
    messages: Message[],
    totalUsage: TokenUsage,
    iteration: number,
    stage: RunResumeStage,
    pendingToolExecution?: PendingToolExecution,
  ): Promise<void> {
    if (!this.saveRunCheckpoint) {
      return
    }

    await this.saveRunCheckpoint({
      autonomy: this.autonomy,
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      mode: this.modeControl?.state.mode ?? 'react',
      modeControlState: this.modeControl ? structuredClone(this.modeControl.state) : undefined,
      messages: withSteeringCheckpoint(messages, this.liveRunState?.steeringNotes).map(cloneMessage),
      cwd: context.cwd,
      workspaceRoot: context.workspaceRoot,
      workspaceIsolation: context.workspaceIsolation,
      scopeTags: cloneCheckpointScopeTags(context.scopeTags),
      executionSkillIds: cloneCheckpointExecutionSkillIds(context.executionSkillIds),
      skillToolNames: cloneCheckpointSkillToolNames(context.skillToolNames),
      toolAllowlist: cloneCheckpointToolAllowlist(context.toolAllowlist),
      skillExecutionPolicies: cloneCheckpointSkillExecutionPolicies(
        context.skillExecutionPolicies,
      ),
      requireToolApproval: context.requireToolApproval,
      runContract: context.runContract,
      totalUsage: { ...totalUsage },
      iteration,
      maxIterations: this.maxIterations,
      thinkingLevel: this.thinkingLevel,
      ...(this.textDeltaMode === 'live' ? { textDeltaMode: 'live' as const } : {}),
      stage,
      checkpointedAt: new Date().toISOString(),
      pendingToolExecution: pendingToolExecution
        ? {
            toolCalls: pendingToolExecution.toolCalls.map(cloneToolCall),
            startIndex: pendingToolExecution.startIndex,
            currentExecutionId: pendingToolExecution.currentExecutionId,
          }
        : undefined,
    })

    if (!pendingToolExecution?.currentExecutionId) {
      await this.clearToolExecution(context.sessionId)
    }
  }

  private async clearRunCheckpoint(sessionId: string): Promise<void> {
    await this.clearRunCheckpointFn?.(sessionId)
  }

  private async *finishAfterUserActionRequired(content: string, context: AgentContext, usage: TokenUsage): AsyncIterable<AgentEvent> {
    yield { type: 'message', content }
    this.state = 'done'
    yield { type: 'state_change', state: 'done' }
    yield { type: 'done', usage, stopReason: stopReasonUserActionRequired(content) }
    await this.safeTriggerPostHook('post:agent:run', { status: 'incomplete', sessionId: context.sessionId, provider: context.provider, model: context.model, usage: { ...usage }, output: content })
    await this.clearRunCheckpoint(context.sessionId)
  }

  private async *finishAfterApprovalDenial(
    context: AgentContext,
    totalUsage: TokenUsage,
    iteration: number,
    denial: TrustedApprovalDenial,
    userInput: string,
  ): AsyncIterable<AgentEvent> {
    const finalContent = buildApprovalDeniedTurnOutput(denial.toolName, userInput)
    yield { type: 'message', content: finalContent }
    this.state = 'done'
    yield { type: 'state_change', state: 'done' }
    yield {
      type: 'done',
      usage: totalUsage,
      stopReason: stopReasonApprovalDenied(denial.toolName ?? 'tool'),
    }
    // A denied approval ends the turn without completing the user's request;
    // recording it as 'success' would skew run statistics (plan F6).
    await this.safeTriggerPostHook('post:agent:run', {
      status: 'incomplete',
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      usage: { ...totalUsage },
      output: finalContent,
      iteration,
    })
    await logAgentRunTrace({
      source: 'react',
      status: 'incomplete',
      mode: 'react',
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      iteration,
      usage: { ...totalUsage },
      output: finalContent,
    })
    await this.clearRunCheckpoint(context.sessionId)
  }

  private async *finishWithObservedCommandOutput(
    context: AgentContext,
    totalUsage: TokenUsage,
    iteration: number,
    result: ToolResultEvent,
  ): AsyncIterable<AgentEvent> {
    const observedOutput = result.output.trimEnd()
    const finalContent = observedOutput || (result.status === 'success'
      ? '[Command completed successfully with no output.]'
      : '[Command failed without output.]')
    yield { type: 'message', content: finalContent }
    this.state = 'done'
    yield { type: 'state_change', state: 'done' }
    yield { type: 'done', usage: totalUsage, stopReason: stopReasonCompleted() }
    await this.safeTriggerPostHook('post:agent:run', {
      status: 'success',
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      usage: { ...totalUsage },
      output: finalContent,
      iteration,
    })
    await logAgentRunTrace({
      source: 'react',
      status: 'success',
      mode: 'react',
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      iteration,
      usage: { ...totalUsage },
      output: finalContent,
    })
    await this.clearRunCheckpoint(context.sessionId)
  }

  private async *finishAfterReadOnlyPolicyFailure(
    context: AgentContext,
    totalUsage: TokenUsage,
    iteration: number,
    failure: TrustedPolicyFailure,
    userInput: string,
  ): AsyncIterable<AgentEvent> {
    const finalContent = buildReadOnlyPolicyBlockedTurnOutput(failure, userInput)
    yield { type: 'message', content: finalContent }
    this.state = 'done'
    yield { type: 'state_change', state: 'done' }
    yield {
      type: 'done',
      usage: totalUsage,
      stopReason: stopReasonPolicyBlocked({ tool: failure.toolName, layer: 'readonly' }),
    }
    await this.safeTriggerPostHook('post:agent:run', {
      status: 'success',
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      usage: { ...totalUsage },
      output: finalContent,
      iteration,
    })
    await logAgentRunTrace({
      source: 'react',
      status: 'success',
      mode: 'react',
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      iteration,
      usage: { ...totalUsage },
      output: finalContent,
    })
    await this.clearRunCheckpoint(context.sessionId)
  }

  private recordLlmUsage(context: AgentContext, usage: TokenUsage): void {
    if (!(usage.inputTokens || usage.outputTokens || usage.thinkingTokens || usage.cacheReadTokens || usage.cacheCreationTokens)) return
    this.usageTracker?.record({
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      inputTokens: usage.inputTokens,
      outputTokens: usage.outputTokens,
      thinkingTokens: usage.thinkingTokens,
      cacheReadTokens: usage.cacheReadTokens,
      cacheWriteTokens: usage.cacheCreationTokens,
    })
  }

  private async clearToolExecution(sessionId: string): Promise<void> {
    await this.clearToolExecutionFn?.(sessionId)
  }

  private async *runToolCalls(
    messages: Message[],
    toolCalls: ToolCall[],
    context: AgentContext,
    iteration: number,
    totalUsage: TokenUsage,
    pending?: PendingToolExecution,
    signal?: AbortSignal,
    canAttachVisualContent = this.canAttachToolVisualContent(context),
    onToolResult?: (event: ToolResultEvent) => void,
  ): AsyncGenerator<AgentEvent, string | undefined> {
    let userActionRequired: string | undefined
    // Map each tool call id to its tool name so the tool_result event (which
    // only carries the id) can be recorded against the right tool. Without
    // this the usage tracker never populated tool_name and Agent Stats
    // "Tool calls" was permanently 0.
    const toolNameByCallId = new Map<string, string>(
      toolCalls.map((toolCall) => [toolCall.id, toolCall.name]),
    )
    const toolStatusByCallId = new Map<string, 'success' | 'error'>()
    const recordObservedToolStatuses = () => {
      for (const [toolCallId, status] of toolStatusByCallId) {
        recordToolResultStatus(messages, toolCallId, status)
      }
    }
    const workspaceMutation = this.workspaceMutationTracker
      ? {
          recordRead: (path: string) =>
            this.workspaceMutationTracker!.recordRead(context.sessionId, path),
          recordWrite: (path: string) =>
            this.workspaceMutationTracker!.recordWrite(context.sessionId, path),
          detectStale: async (path: string) => {
            const stale = await this.workspaceMutationTracker!.detectStale(context.sessionId, path)
            return stale
              ? { path: stale.path, description: describeStaleSignal(stale) }
              : null
          },
          lookupReadObservation: (path: string, viewKey: string) =>
            this.workspaceMutationTracker!.lookupReadObservation(context.sessionId, path, viewKey),
          recordReadObservation: (path: string, viewKey: string, output: string) =>
            this.workspaceMutationTracker!.recordReadObservation(
              context.sessionId,
              path,
              viewKey,
              output,
            ),
        }
      : undefined

    let editCheckpointId: string | undefined
    let editSummary: import('@sepilotd/core').EditCheckpointSummary | undefined
    const snapshots = this.editSnapshotStore
    const editCheckpoint = snapshots ? {
      recordPreEdit: async (path: string) => {
        editCheckpointId ??= snapshots.openCheckpoint(context.sessionId, `Tool batch ${iteration}`)
        await snapshots.recordPreEdit(context.sessionId, editCheckpointId, path)
      },
    } : undefined
    try {
    for await (const event of runToolExecution({
      delegationToolNames: this.modeControl?.delegationToolNames,
      messages,
      toolCalls,
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      tools: this.tools,
      policy: this.policy,
      autonomy: this.autonomy,
      primaryAgentId: context.primaryAgentId,
      autoApprove: context.autoApprove,
      requireToolApproval: context.requireToolApproval,
      auditLogger: this.auditLogger,
      hookRegistry: this.hookRegistry,
      deviceName: this.deviceName,
      approvalCallback: this.approvalCallback,
      evaluateAutoApproval: this.evaluateAutoApproval,
      pending,
      signal,
      cwd: context.cwd,
      workspaceRoot: context.workspaceRoot,
      runContract: context.runContract,
      writingDocId: context.writingDocId,
      scopeTags: context.scopeTags,
      channelContext: context.channelContext,
      canAttachVisualContent,
      persistApprovalCheckpoint: (requestId, nextToolCalls, currentToolIndex) => {
        // The next tool can pause for approval before this iterator reaches
        // its final annotation pass. Persist results already observed in this
        // batch now so a resumed explicit memory request still sees the prior
        // successful memory.remember result as structured evidence.
        recordObservedToolStatuses()
        return this.persistApprovalCheckpoint(
          requestId,
          context,
          messages,
          nextToolCalls,
          currentToolIndex,
          totalUsage,
          iteration,
        )
      },
      clearApprovalCheckpoint: (requestId) => this.clearCheckpoint(requestId),
      clearRunCheckpoint: () => this.clearRunCheckpoint(context.sessionId),
      persistRunCheckpoint: (pendingToolExecution) => {
        recordObservedToolStatuses()
        return this.persistRunCheckpoint(
          context,
          messages,
          totalUsage,
          iteration,
          'acting',
          pendingToolExecution,
        )
      },
      loadToolExecution: () => this.loadToolExecution?.(context.sessionId) ?? Promise.resolve(null),
      saveToolExecution: (record) => this.saveToolExecution?.(record) ?? Promise.resolve(),
      toolStats: this.toolStats,
      workspaceMutation,
      editCheckpoint,
    })) {
      if (event.type === 'tool_call') {
        toolNameByCallId.set(event.toolCall.id, event.toolCall.name)
      }

      // One tool_name-tagged record per executed tool so Agent Stats can count
      // tool calls. Zero token/cost — the LLM-call cost is recorded when received with
      // a NULL tool_name (the turn), keeping the two accounting streams
      // separate exactly as getSnapshot() expects.
      if (event.type === 'tool_result' && this.usageTracker) {
        const toolName = toolNameByCallId.get(event.toolCallId)
        if (toolName) {
          this.usageTracker.record({
            sessionId: context.sessionId,
            provider: context.provider,
            model: context.model,
            inputTokens: 0,
            outputTokens: 0,
            toolName,
          })
        }
      }

      if (event.type === 'tool_result') {
        if (event.status === 'error' && typeof event.metadata?.userActionRequired === 'string' && event.metadata.userActionRequired.trim()) {
          userActionRequired ??= userActionRequiredOutput(event.metadata.userActionRequiredCode, event.metadata.userActionRequired.trim().slice(0, 2000), latestUserText(messages), event.metadata.userActionRequiredDetail)
        }
        onToolResult?.(event)
        toolStatusByCallId.set(event.toolCallId, event.status)
        const delegated = delegatedUsageFromMetadata(event.metadata)
        if (delegated) {
          totalUsage.inputTokens += delegated.inputTokens
          totalUsage.outputTokens += delegated.outputTokens
        }
      }

      yield event
      if (event.type === 'tool_result' && event.status === 'error' && isRequiredArtifactPathEvidenceBlock(event.output)) {
        messages.push({
          role: 'system',
          content: buildRequiredArtifactPathEvidenceRepairMessage(event.output),
        })
      }
    }
    } finally {
      // Preserve recoverable edits even when cancellation interrupts the tool
      // iterator. A committed checkpoint records bytes, not task completion.
      if (snapshots && editCheckpointId) {
        editSummary = snapshots.commitCheckpoint(context.sessionId, editCheckpointId)
        await snapshots.flush()
      }
    }
    if (editSummary) yield { type: 'edit_checkpoint_resolved', checkpoint: editSummary }

    // runToolExecution appends the matching tool message after yielding each
    // result. Checkpoint callbacks above annotate intermediate results before
    // cloning; this final pass covers the last result when no later checkpoint
    // boundary occurs.
    recordObservedToolStatuses()
    return userActionRequired
  }

  private async *finalizeDeterministicScheduleResult(
    content: string,
    context: AgentContext,
    totalUsage: TokenUsage,
    iteration: number,
  ): AsyncIterable<AgentEvent> {
    if (content) {
      yield { type: 'text_delta', text: content }
    }
    yield { type: 'message', content }
    this.state = 'done'
    yield { type: 'state_change', state: 'done' }
    yield { type: 'done', usage: totalUsage, stopReason: stopReasonCompleted() }
    await this.safeTriggerPostHook('post:agent:run', {
      status: 'success',
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      usage: { ...totalUsage },
      output: content,
      iteration,
    })
    await logAgentRunTrace({
      source: 'react',
      status: 'success',
      mode: 'react',
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      iteration,
      usage: { ...totalUsage },
      output: content,
    })
    await this.clearRunCheckpoint(context.sessionId)
  }

  private hasPendingSteering(): boolean {
    return (this.liveRunState?.steeringNotes ?? []).some(
      (note) => note.consumedAt === undefined && note.cancelledAt === undefined,
    )
  }

  private async *runWithLiveState(
    checkpointMessages: Message[], context: AgentContext, totalUsage: TokenUsage,
    startIteration = 0, pendingToolExecution?: PendingToolExecution, signal?: AbortSignal,
  ): AsyncIterable<AgentEvent> {
    const liveState: LiveRunState = { steeringNotes: recoveredReactSteering(checkpointMessages) }
    this.liveRunState = liveState
    this.activeRuns?.upsert({ sessionId: context.sessionId, graphId: 'react',
      currentNode: 'agent', iteration: startIteration, maxIterations: this.maxIterations })
    this.activeRuns?.registerLiveState(context.sessionId, liveState)
    try {
      yield* this.continueRun(withoutSteeringCheckpoint(checkpointMessages), context, totalUsage,
        startIteration, pendingToolExecution, signal)
    } finally {
      if (this.activeRuns?.getLiveState(context.sessionId) === liveState) {
        this.activeRuns.finish(context.sessionId)
      }
      if (this.liveRunState === liveState) this.liveRunState = undefined
    }
  }

  private async *continueRun(
    messages: Message[],
    context: AgentContext,
    totalUsage: TokenUsage,
    startIteration = 0,
    pendingToolExecution?: PendingToolExecution,
    signal?: AbortSignal,
    continuationCycle = 0,
    currentTurnObservationHistory?: ObservationHistoryEntry[],
  ): AsyncIterable<AgentEvent> {
    const activeSkillExecutionPolicies = context.skillExecutionPolicies
      ?? resolveActiveSkillExecutionPolicies(context.executionSkillIds)
    let repairedInterimProgressCount = 0
    let repairedEmptyFinalReplyCount = 0
    let repairedLengthContinuationCount = 0
    let lengthContinuationPrefix = ''
    let hadLengthContinuation = false
    const lengthContinuationMax = readLengthContinuationMax()
    let repairedDegenerateResponseCount = 0
    let repairedInvalidToolResponseCount = 0
    let forceFinalAfterDegenerateOutput = false
    let lastUsableRecoveryCandidate: string | null = null
    let repairedMissingAnswerProtocolCount = 0
    let repairedUnsupportedCitationCount = 0
    let repairedDuplicateToolCallCount = 0
    let repairedSkillExecutionToolCount = countSkillExecutionToolRepairMessages(messages)
    let repairedSkillExecutionCompletionCount =
      countSkillExecutionCompletionRecoveryMessages(messages)
    let repairedMemoryWriteCount = countMemoryWriteRecoveryPrompts(messages)
    let repairedScheduleCompletionCount = countScheduleCompletionRecoveryPrompts(messages)
    let repairedActionCompletionCount = countActionCompletionRecoveryPrompts(messages)
    const writeLoopLimit = resolveWriteLoopLimit()
    const writeLoopTracker = new WriteLoopTracker(writeLoopLimit)
    const readOnlyToolHistory: StuckToolRepeatEntry[] = []
    // Structured current-run evidence survives message compaction. It is also
    // mutation-aware, unlike the narrower stuck-read tracker below.
    const observationHistory = currentTurnObservationHistory
      ?? observationHistoryFromCurrentTurnMessages(messages)
    let readOnlyEvidenceTurns = 0
    let lastReadOnlyToolSignature: string | null = null
    let forceFinalAfterRepeatedRead = false
    let forceFinalAfterObservationBudget = hasToolCallBatchCap(messages)
    let announcedObservationCompaction = false
    // Rolling in-run history summary, carried across iterations so a later
    // compaction extends the previous summary instead of re-summarizing the
    // whole run from scratch.
    let compressedHistorySummary: string | undefined
    let compressedHistoryUpToIndex: number | undefined
    let forceFinalAfterNoRetryActionFailure = messages.some((message) => (
      message.role === 'system'
      && message.metadata?.reminderKind === 'no_retry_action_failure'
    ))
    let closedExactToolWorkflowComplete = messages.some((message) => (
      message.role === 'system'
      && message.metadata?.reminderKind === 'closed_exact_tool_call_budget'
    ))
    let forceFinalAfterReadOnlyPolicyFailure = false
    let forceFinalAfterApprovalGrace = false
    const approvalGraceState = createApprovalGraceState()
    let repairedReadOnlyInvocationCount = countReadOnlyInvocationRepairMessages(messages)
    const ignoredReadOnlyPolicyFailureToolCallIds = new Set(
      readOnlyInvocationRepairToolCallIds(messages),
    )
    const focusedRepositoryRequest = latestUserText(messages)
    const toolApprovalPosture = this.describeRunToolApprovalPosture(context)
    const exactToolCardinalityIdentities = resolveToolCardinalityIdentities(
      this.tools.list(),
    )
    const canonicalReadTargetIdentities = resolveToolCanonicalReadIdentities(
      this.tools.list(),
    )
    const toolsForbiddenByUser = this.semanticRouting ? this.tools.list().length === 0 && !this.modeControl?.tools.length : explicitlyForbidsToolUse(focusedRepositoryRequest)
    const focusedTerminalExecution = !this.semanticRouting && isFocusedTerminalCommandExecution(focusedRepositoryRequest)
    const focusedProcessStart = !this.semanticRouting && extractFocusedProcessStart(focusedRepositoryRequest)
    const focusedProcessObservation = !this.semanticRouting && inputRequestsFocusedSingleProcessObservation(
      focusedRepositoryRequest,
    )
    const focusedProcessObservationTool = focusedProcessObservation
      ? [...FOCUSED_PROCESS_OBSERVATION_TOOLS]
          .find((toolName) => focusedRepositoryRequest.includes(toolName))
      : undefined
    const focusedBrowserObservation = !this.semanticRouting && inputRequestsFocusedSingleBrowserObservation(
      focusedRepositoryRequest,
    )
    const focusedVisibleUrlOpen = !this.semanticRouting && isVisibleUrlOpenRequest(focusedRepositoryRequest)
    const focusedBoundedCodingTask = !this.semanticRouting && isFocusedBoundedCodingTask(focusedRepositoryRequest)
    const substantiveRepositoryReview =
      !this.semanticRouting && isSubstantiveRepositoryChangeReview(focusedRepositoryRequest)
    // A repository review needs evidence and a complete answer, but it does
    // not need a provider-neutral control prefix. Some otherwise capable
    // models repeatedly omit literal ANSWER:/INCOMPLETE: stems; treating that
    // cosmetic miss as a failed turn can multiply expensive final-synthesis
    // calls after all tools have already finished. Explicit strict-protocol
    // surfaces still retain their existing enforcement.
    const requireFinalAnswerProtocol = this.strictFinalAnswerProtocol
    const rawTerminalOutputRequested = requestsRawTerminalCommandOutput(focusedRepositoryRequest)
    const focusedRepositoryLookup = !this.semanticRouting && isFocusedRepositoryChangeLookup(focusedRepositoryRequest)
    const focusedRepositoryNeedsDiff = focusedRepositoryLookupNeedsDiff(focusedRepositoryRequest)
    const focusedRepositoryObservedTools = new Set(
      messages
        .filter((message) =>
          message.role === 'tool'
          && typeof message.name === 'string'
          && FOCUSED_REPOSITORY_LOOKUP_TOOLS.has(message.name)
        )
        .map((message) => message.name as string),
    )
    let focusedRepositoryToolTurns = messages.filter((message) =>
      message.role === 'assistant'
      && message.toolCalls?.some((call) => FOCUSED_REPOSITORY_LOOKUP_TOOLS.has(call.name)),
    ).length
    let substantiveRepositoryReviewToolTurns = messages.filter((message) =>
      message.role === 'assistant'
      && message.toolCalls?.some((call) =>
        SUBSTANTIVE_REPOSITORY_REVIEW_OBSERVATION_TOOLS.has(call.name)),
    ).length
    const focusedRepositoryEvidenceReady = (): boolean =>
      focusedRepositoryToolTurns >= 2
      || (
        focusedRepositoryObservedTools.has('git.log')
        && (!focusedRepositoryNeedsDiff || focusedRepositoryObservedTools.has('git.diff'))
      )
    let repairedRunOutcomeCount = countOutcomeRecoveryPrompts(messages)
    let repairedOutcomeReviewCount = 0
    let repairedOutcomeReviewSynthesisCount = 0
    // Mutations require final adjudication. Optional router/planner failure must
    // not prevent that safety boundary from running. Share one bounded reserve
    // across all mutation-review repairs, never reset it for every attempt.
    let mutationReviewBudget: AuxiliaryLlmTurnBudget | undefined
    let outcomeReviewSynthesisEvidenceCount = messages.filter(
      (message) => message.role === 'tool',
    ).length
    let repairedOutcomeReviewNoProgressCount = 0
    let repairedProviderMessageCount = countProviderMessageRecoveryPrompts(messages)
    let disableThinkingForProviderRecovery = false
    let contextRecoveries = 0
    let maxOutputTokenRecoveries = 0
    let effectiveMaxOutputTokens: number | undefined
    let effectiveContextWindow: number | null = null
    let imageInputRecoveries = 0
    let disableVisualAttachmentsForProviderRecovery = false
    let preferPromptReact =
      this.resolveModelInfo(context.model)?.capabilities.promptReactPreferred === true
    let toolTransportRecoveryCount = 0
    const runOutcomeRecoveryReasons: string[] = []
    let lastUnsupportedCitationCandidate: string | null = null
    let lastMissingAnswerProtocolCandidate: string | null = null
    const supervisorRepairCount = (): number =>
      repairedInterimProgressCount
      + repairedEmptyFinalReplyCount
      + repairedMissingAnswerProtocolCount
      + repairedUnsupportedCitationCount
      + repairedDuplicateToolCallCount
      + repairedSkillExecutionToolCount
      + repairedSkillExecutionCompletionCount
      + repairedMemoryWriteCount
      + repairedScheduleCompletionCount
      + repairedActionCompletionCount
      + repairedDegenerateResponseCount
      + repairedInvalidToolResponseCount
      + repairedReadOnlyInvocationCount
    const effectiveMaxIterations = (): number => this.hardMaxIterations
      ? this.maxIterations
      : this.maxIterations
        + repairedRunOutcomeCount
        + repairedOutcomeReviewCount
        + repairedOutcomeReviewNoProgressCount
        + repairedProviderMessageCount
        + supervisorRepairCount() * 2
    const outcomeReviewRecoveryReasons: string[] = []
    const resolveReadOnlyPolicyFailure = (
      failure: TrustedPolicyFailure,
    ): 'repair' | 'synthesize' | 'terminate' => {
      if (hasSuccessfulToolEvidenceInCurrentTurn(messages)) {
        // A model gets exactly one tool-free chance to synthesize from the
        // successful evidence. If it ignores that boundary and emits another
        // blocked tool call, terminate deterministically instead of feeding
        // the same blocker through an unbounded synthesis loop.
        if (forceFinalAfterReadOnlyPolicyFailure) return 'terminate'
        messages.push(buildReadOnlyPolicyBlockSynthesisMessage(failure))
        forceFinalAfterReadOnlyPolicyFailure = true
        return 'synthesize'
      }

      if (
        repairedReadOnlyInvocationCount < 1
        && isRepairableReadOnlyInvocationFailure(messages, failure)
      ) {
        messages.push(buildReadOnlyInvocationRepairMessage(failure))
        repairedReadOnlyInvocationCount += 1
        if (failure.toolCallId) {
          ignoredReadOnlyPolicyFailureToolCallIds.add(failure.toolCallId)
        }
        return 'repair'
      }

      return 'terminate'
    }
    const resolveApprovalOutcomeAfterTools = (): TrustedApprovalDenial | null => {
      const outcome = observeApprovalOutcomeAfterTools(messages, approvalGraceState)
      if (outcome === 'terminate') return approvalGraceState.denial
      if (
        outcome === null
        && approvalGraceState.denial
        && approvalGraceState.toolTurnsRemaining > 0
      ) {
        approvalGraceState.toolTurnsRemaining -= 1
        forceFinalAfterApprovalGrace = true
      }
      return null
    }

    for (
      let i = startIteration;
      i < effectiveMaxIterations() &&
      !this.aborted &&
      !(signal?.aborted ?? false);
      i++
    ) {
      const wallClock = this.runWallClock
      if (
        wallClock?.budgetMs !== undefined
        && Date.now() - wallClock.startedAt >= wallClock.budgetMs
      ) {
        this.state = 'done'
        await this.persistRunCheckpoint(context, messages, totalUsage, i, 'observing')
        const wallClockContent = buildBudgetExhaustedMessage({
          mode: 'react',
          layer: 'wall_clock',
          iterationBudget: wallClock.budgetMs,
          contract: context.runContract,
        })
        if (this.textDeltaMode !== 'live') {
          yield { type: 'text_delta', text: wallClockContent }
        }
        yield { type: 'message', content: wallClockContent }
        yield {
          type: 'done',
          usage: totalUsage,
          stopReason: stopReasonWallClock(wallClock.budgetMs),
        }
        await this.safeTriggerPostHook('post:agent:run', {
          status: 'incomplete',
          sessionId: context.sessionId,
          provider: context.provider,
          model: context.model,
          usage: { ...totalUsage },
          iteration: i,
          output: wallClockContent,
        })
        await logAgentRunTrace({
          source: 'react',
          status: 'incomplete',
          mode: 'react',
          sessionId: context.sessionId,
          provider: context.provider,
          model: context.model,
          iteration: i,
          usage: { ...totalUsage },
        })
        return
      }
      this.activeRuns?.upsert({ sessionId: context.sessionId, graphId: 'react',
        currentNode: pendingToolExecution ? 'tools' : 'agent', iteration: i,
        maxIterations: effectiveMaxIterations(), tokensInput: totalUsage.inputTokens,
        tokensOutput: totalUsage.outputTokens })
      const modelInfo = this.resolveModelInfo(context.model)
      const canAttachVisualContent = this.canAttachToolVisualContent(
        context,
        disableVisualAttachmentsForProviderRecovery,
      )
      if (pendingToolExecution) {
        const pendingCalls = pendingToolExecution.toolCalls
        let focusedPendingActionResult: ToolResultEvent | undefined
        await this.persistRunCheckpoint(
          context,
          messages,
          totalUsage,
          i,
          'acting',
          pendingToolExecution,
        )
        this.state = 'acting'
        yield { type: 'state_change', state: 'acting' }
        const requiredUserAction = yield* this.runToolCalls(
          messages,
          pendingToolExecution.toolCalls,
          context,
          i,
          totalUsage,
          pendingToolExecution,
          signal,
          canAttachVisualContent,
          focusedProcessObservation
            ? (event) => { focusedPendingActionResult = event }
            : undefined,
        )
        if (requiredUserAction) {
          yield* this.finishAfterUserActionRequired(requiredUserAction, context, totalUsage)
          return
        }
        const approvalDenial = resolveApprovalOutcomeAfterTools()
        if (approvalDenial) {
          yield* this.finishAfterApprovalDenial(
            context,
            totalUsage,
            i + 1,
            approvalDenial,
            latestUserText(messages),
          )
          return
        }
        const resumedPolicyFailure = this.autonomy === AutonomyLevel.ReadOnly
          ? latestTrustedPolicyFailure(messages, ignoredReadOnlyPolicyFailureToolCallIds)
          : null
        if (resumedPolicyFailure) {
          const resolution = resolveReadOnlyPolicyFailure(resumedPolicyFailure)
          if (resolution !== 'terminate') {
            pendingToolExecution = undefined
            this.state = 'observing'
            yield {
              type: 'thinking',
              content: resolution === 'repair'
                ? '[supervisor] Read-only policy blocked a shell wrapper; allowing one policy-checked direct invocation repair.'
                : '[supervisor] Read-only policy blocked a follow-up; synthesizing once from successful current-turn evidence.',
            }
            yield { type: 'state_change', state: 'observing' }
            continue
          }
          yield* this.finishAfterReadOnlyPolicyFailure(
            context,
            totalUsage,
            i + 1,
            resumedPolicyFailure,
            latestUserText(messages),
          )
          return
        }
        pendingToolExecution = undefined
        if (focusedPendingActionResult) {
          const focusedToolName = pendingCalls.find(
            (call) => call.id === focusedPendingActionResult?.toolCallId,
          )?.name
          const deterministicResult = focusedToolName
            ? deterministicFocusedActionResult(focusedRepositoryRequest, {
                toolName: focusedToolName,
                status: focusedPendingActionResult.status,
                output: focusedPendingActionResult.output,
              })
            : undefined
          if (deterministicResult) {
            yield* this.finalizeDeterministicScheduleResult(
              deterministicResult,
              context,
              totalUsage,
              i + 1,
            )
            return
          }
        }
        this.state = 'observing'
        yield { type: 'state_change', state: 'observing' }
        continue
      }

      const freshSteering = (this.liveRunState?.steeringNotes ?? []).filter(
        (note) => note.consumedAt === undefined && note.cancelledAt === undefined,
      )
      for (const note of freshSteering) {
        note.consumedAt = Date.now()
        await this.journalSteeringConsumed?.(context.sessionId, note.id)
        yield { type: 'steering_consumed', id: randomUUID(), noteId: note.id,
          message: note.message, kind: note.kind, iteration: i }
      }

      await this.persistRunCheckpoint(
        context,
        messages,
        totalUsage,
        i,
        i === 0 ? 'thinking' : 'observing',
      )

      if (
        !forceFinalAfterNoRetryActionFailure
        && context.runContract?.executionIntent?.retryPolicy === 'forbidden'
        && currentTurnHasNoRetrySemanticActionFailure(messages)
      ) {
        forceFinalAfterNoRetryActionFailure = true
        messages.push({
          role: 'system',
          metadata: { reminderKind: 'no_retry_action_failure' },
          content: [
            '[No-retry semantic action failed]',
            'A required non-observation action executed and failed while the active contract forbids retries.',
            'The tool surface is closed. Report the workflow as incomplete from the retained result; do not retry, substitute, promise another action, or claim that later required actions ran.',
          ].join(' '),
        })
        yield {
          type: 'thinking',
          content: '[supervisor] A required action failed under the no-retry contract; closing tool access for an honest incomplete result.',
        }
      }

      const declaredContextWindow = modelInfo?.contextWindow
        ?? DEFAULT_UNKNOWN_MODEL_CONTEXT_WINDOW
      const contextWindow = effectiveContextWindow === null
        ? declaredContextWindow
        : Math.min(effectiveContextWindow, declaredContextWindow)
      const charsPerToken = tokenCalibration.charsPerToken(this.provider.id, context.model)
      const observationBudget = evaluateCurrentTurnToolObservationBudget(messages, {
        contextWindowTokens: contextWindow,
        charsPerToken,
        maxObservationCount: resolveMaxTurnToolObservations(),
      })
      // Working-memory pressure is handled by the compacted projection built
      // below; only a turn that keeps gathering past several full loads has
      // stopped converging and earns a closed tool surface.
      if (observationBudget.exceeded) forceFinalAfterObservationBudget = true
      if (observationBudget.compactionRequired && !observationBudget.exceeded
        && !announcedObservationCompaction) {
        announcedObservationCompaction = true
        yield {
          type: 'thinking',
          content: `[supervisor] Current-turn evidence (${observationBudget.observationCount} observations) exceeds bounded working memory; older observations continue as a digest while evidence gathering stays open.`,
        }
      }
      if (repeatedBrowserClickWithoutProgress(messages)) forceFinalAfterRepeatedRead = true
      // Proactive in-run compaction, matching the graph agent node. Without it
      // a long react run only reacts *after* the provider rejects the prompt,
      // and until then `fitProviderContext` silently drops the oldest turns —
      // losing the user's goal and earlier decisions with no summary left
      // behind. `semanticCompress` self-guards: it returns null while the run
      // still fits, so turns that are not under pressure cost nothing.
      if (process.env.SEPILOTD_SEMANTIC_COMPRESSION !== '0') {
        try {
          const compressed = await semanticCompress(
            messages,
            contextWindow,
            this.provider,
            {
              model: context.model,
              signal,
              breaker: this.providerCircuitBreaker,
              auxiliaryLlmBudget: this.auxiliaryLlmBudget,
              previousSummary: compressedHistorySummary,
              previousUpToIndex: compressedHistoryUpToIndex,
              charsPerToken,
            },
          )
          if (compressed) {
            const foldedCount = messages.length - compressed.messages.length
            messages.splice(0, messages.length, ...compressed.messages)
            compressedHistorySummary = compressed.summary
            compressedHistoryUpToIndex = compressed.upToIndex
            // Fold the compression round-trip into the run total so cost and
            // spend gates account for it.
            if (compressed.usage) {
              totalUsage.inputTokens += compressed.usage.inputTokens
              totalUsage.outputTokens += compressed.usage.outputTokens
            }
            yield {
              type: 'thinking',
              content: `[context] Folded ${Math.max(0, foldedCount)} earlier message(s) into a running summary to stay inside the context window.`,
            }
          }
        } catch {
          // Compression is an optimization, never a fatal step: the request
          // fitter still bounds whatever is left.
        }
      }
      const compactedReadMessages = compactOversizedToolProtocolUnits(
        supersedeStaleSystemReminders(supersedeStaleReadResults(messages)),
        { contextWindowTokens: contextWindow, charsPerToken },
      )
      // Inject a per-iteration system message naming tools that have been
      // failing in this session, so the model can change tactics instead
      // of looping on the same broken call. The hint lives only in the
      // outbound request, not in `messages`, so it doesn't accumulate.
      let messagesForRequest = withProblemToolFeedback(
        compactedReadMessages,
        this.toolStats?.list(context.sessionId) ?? [],
        i,
      )
      const nativeToolUse = this.supportsNativeToolUse(context.model)
      // An explicit maxTokens option remains preferred over the model default,
      // but the final request fitter may lower it when the concrete prompt
      // would otherwise exceed the model's physical context window.
      const resolvedMaxTokens = this.maxTokens ?? modelInfo?.maxOutputTokens
      const iterationBudget = effectiveMaxIterations()
      const isLastIteration = i >= iterationBudget - 1
      const isPenultimateIteration =
        !isLastIteration && i === iterationBudget - 2 && iterationBudget >= 3
      const skipOutcomeReview = !this.reviewOutcomes || focusedRepositoryLookup
        || substantiveRepositoryReview
        || focusedTerminalExecution
        || toolsForbiddenByUser
        || forceFinalAfterObservationBudget
        || forceFinalAfterDegenerateOutput
        || forceFinalAfterReadOnlyPolicyFailure
        || forceFinalAfterApprovalGrace
        || forceFinalAfterNoRetryActionFailure
      const allToolsForRequest = this.tools.toToolDefinitions(context).filter((tool) => (
        this.autonomy !== AutonomyLevel.ReadOnly
        || tool.name === 'terminal.run'
        || isPolicyReadOnlyTool(tool.name, this.tools.securityDescriptor(tool.name))
      ) && (
        // A denial grants one side-effect-free turn: offer only policy
        // read-only tools so the catalog and the grace message agree.
        !approvalGraceState.denial
        || isPolicyReadOnlyTool(tool.name)
      )).map((tool) => {
        // Announce the static approval posture in the catalog itself so the
        // model does not discover prompts and blocks one call at a time.
        const suffix = toolApprovalDescriptionSuffix(toolApprovalPosture.get(tool.name))
        return suffix ? { ...tool, description: `${tool.description}${suffix}` } : tool
      })
      const allToolNames = new Set(allToolsForRequest.map((tool) => tool.name))
      const skillCompletionBeforeRequest = evaluateSkillExecutionCompletion(
        messages,
        activeSkillExecutionPolicies,
      )
      const deterministicSkillCompletionPolicyActive =
        hasDeterministicSkillCompletionPolicy(activeSkillExecutionPolicies)
      const deterministicSkillCompletionSatisfiedBeforeRequest =
        deterministicSkillCompletionPolicyActive
        && skillCompletionBeforeRequest.missing.length === 0
      const requiresVerifiedSkillExecution =
        deterministicSkillCompletionPolicyActive
        && skillCompletionBeforeRequest.missing.length > 0
      const missingSkillStageToolNames = new Set(
        skillCompletionBeforeRequest.missing.flatMap((missing) => missing.tools),
      )
      const missingSkillStageToolsAvailable = skillCompletionBeforeRequest.missing.every(
        (missing) => missing.tools.some((toolName) => allToolNames.has(toolName)),
      )
      // A generic observation budget closes unrelated discovery, but it must
      // not make a selected skill's still-missing required stage unreachable.
      // Re-open only the declared stage tools, under the skill's own bounded
      // completion retry budget. Strong terminal boundaries remain final.
      const prioritizeMissingSkillStage =
        (forceFinalAfterRepeatedRead || forceFinalAfterObservationBudget)
        && requiresVerifiedSkillExecution
        && missingSkillStageToolsAvailable
        && skillCompletionBeforeRequest.maxRetries > 0
        // The retry counter is incremented when the recovery prompt is
        // issued, before the provider gets the tool-enabled turn it grants.
        && repairedSkillExecutionCompletionCount <= skillCompletionBeforeRequest.maxRetries
        && !closedExactToolWorkflowComplete
        && !forceFinalAfterDegenerateOutput
        && !forceFinalAfterReadOnlyPolicyFailure
        && !forceFinalAfterNoRetryActionFailure
      const contractCommandToolNames = context.runContract?.executionIntent?.kind === 'operational-action'
        ? (context.runContract.executionIntent.allowedTools ?? [])
            .filter((toolName) => allToolNames.has(toolName))
        : []
      // A closed capability boundary is not an exact tool-name allowlist.
      // Only an explicitly present allowedTools list (including []) narrows
      // this surface; absence must retain the request-scoped registry.
      const closedContractToolNames =
        context.runContract?.executionIntent?.allowedTools !== undefined
          ? context.runContract.executionIntent.allowedTools
              .filter((toolName) => allToolNames.has(toolName))
          : undefined
      const commandEvidenceToolNames = [...new Set([
        ...contractCommandToolNames,
        ...allToolsForRequest
          .filter((tool) => {
            const effect = this.tools.securityDescriptor(tool.name)?.effect
            return effect === 'external-write' || effect === 'process-lifecycle'
          })
          .map((tool) => tool.name),
      ])]
      const explicitMemoryWriteIntent = !this.semanticRouting && isExplicitMemoryWriteRequest(latestUserText(messages))
      const requiresVerifiedMemoryWrite = explicitMemoryWriteIntent
      const explicitScheduleCreateIntent = !this.semanticRouting && isExplicitScheduleCreateRequest(latestUserText(messages))
      const requiresVerifiedScheduleCreate = explicitScheduleCreateIntent
      const requiredActionKinds = this.semanticRouting ? [] : requiredActionEvidenceKindsForTurn({
        messages,
        userInput: latestUserText(messages),
        availableToolNames: [...allToolNames],
      })
      const requiresVerifiedAction = requiredActionKinds.length > 0
      const canAutoContinueAfterBudget = continuationCycle < this.maxContinuationCycles
      const cadence = this.strictFinalAnswerProtocol
        ? evaluateAnalysisWriteCadence(messages, allToolNames)
        : undefined
      const forceWriteBeforeMoreDiscovery = Boolean(cadence?.shouldForceWrite) && !isLastIteration
      const cadenceWriteToolNames = [...ANALYSIS_CADENCE_WRITE_TOOL_NAMES]
        .filter((toolName) => allToolNames.has(toolName))
      if (forceWriteBeforeMoreDiscovery && cadence) {
        messagesForRequest = [
          ...messagesForRequest,
          buildAnalysisWriteCadenceMessage(cadence),
        ]
      }
      const focusedEvidenceReady = focusedRepositoryLookup && focusedRepositoryEvidenceReady()
      const focusedTerminalCommandCompleted = focusedTerminalExecution
        && hasToolResultForName(messages, 'terminal.run')
      const forceSynthesisFromObservationBudget =
        (
          (forceFinalAfterRepeatedRead || forceFinalAfterObservationBudget)
          && !prioritizeMissingSkillStage
        )
        || forceFinalAfterDegenerateOutput
        || forceFinalAfterReadOnlyPolicyFailure
        || forceFinalAfterApprovalGrace
        || forceFinalAfterNoRetryActionFailure
        || focusedTerminalCommandCompleted
      if (substantiveRepositoryReview) {
        messagesForRequest = appendSubstantiveRepositoryReviewMessage(messagesForRequest)
      }
      if (focusedEvidenceReady) {
        messagesForRequest = appendFocusedRepositorySynthesisMessage(messagesForRequest)
      } else if (forceSynthesisFromObservationBudget) {
        messagesForRequest = appendForceFinalSystemMessage(messagesForRequest)
      }
      const useBoundedFinalContext = activeSkillExecutionPolicies.length === 0
        && (context.executionSkillIds?.length ?? 0) === 0
        && (
          forceFinalAfterRepeatedRead
          || forceFinalAfterObservationBudget
          || forceFinalAfterReadOnlyPolicyFailure
          || forceFinalAfterApprovalGrace
          || forceFinalAfterNoRetryActionFailure
        )
      if (useBoundedFinalContext) {
        messagesForRequest = buildBoundedFinalSynthesisContext(messagesForRequest, {
          preserveLatestAssistantDraft: hadLengthContinuation,
          preferredToolNames: substantiveRepositoryReview
            ? ['git.log', 'git.diff', 'git.status']
            : undefined,
        })
      }
      const allowedToolNames: Set<string> | undefined = toolsForbiddenByUser
        ? new Set<string>()
        : prioritizeMissingSkillStage
          ? new Set(
              [...missingSkillStageToolNames].filter((toolName) => allToolNames.has(toolName)),
            )
        : forceSynthesisFromObservationBudget
          ? new Set<string>()
        : closedContractToolNames
          ? new Set(closedContractToolNames)
        : focusedTerminalExecution
          ? new Set(allToolNames.has('terminal.run') ? ['terminal.run'] : [])
        : focusedProcessObservationTool
          ? new Set(
              allToolNames.has(focusedProcessObservationTool)
                ? [focusedProcessObservationTool]
                : [],
            )
        : focusedVisibleUrlOpen
          ? new Set(
              ['web.search', 'computer.open_url']
                .filter((toolName) => allToolNames.has(toolName)),
            )
        : focusedBoundedCodingTask
          ? new Set(
              [...FOCUSED_BOUNDED_CODING_TOOLS]
                .filter((toolName) => allToolNames.has(toolName)),
            )
        : substantiveRepositoryReview
          ? new Set(
              (() => {
                const structuredTools = [...SUBSTANTIVE_REPOSITORY_REVIEW_TOOLS]
                  .filter((toolName) => allToolNames.has(toolName))
                return structuredTools.length > 0
                  ? structuredTools
                  : allToolNames.has('terminal.run')
                    ? ['terminal.run']
                    : []
              })(),
            )
        : focusedRepositoryLookup
          ? focusedEvidenceReady
          ? new Set<string>()
          : new Set(
              [...FOCUSED_REPOSITORY_LOOKUP_TOOLS]
                .filter((toolName) => {
                  if (!allToolNames.has(toolName)) return false
                  if (toolName === 'git.status') return focusedRepositoryToolTurns === 0
                  if (toolName === 'git.diff') return focusedRepositoryNeedsDiff
                    && !focusedRepositoryObservedTools.has(toolName)
                  return !focusedRepositoryObservedTools.has(toolName)
                }),
            )
          : forceWriteBeforeMoreDiscovery
            ? new Set(cadenceWriteToolNames)
            : undefined
      const toolsForRequest = allowedToolNames
        ? allToolsForRequest.filter((tool) => allowedToolNames.has(tool.name))
        : allToolsForRequest
      // A mode-local final turn may still hand off within the original turn
      // budget. Do not reopen actions/discovery or evade a terminal boundary.
      const canTransferOnLocalFinal = isLastIteration
        && (this.modeControl?.state.turnMaxIterations ?? 0) > i + 1
        && !forceSynthesisFromObservationBudget
        && !approvalGraceState.denial
        && !toolsForbiddenByUser
      const controlToolsForRequest = !forceSynthesisFromObservationBudget && !approvalGraceState.denial
        ? (this.modeControl?.tools ?? []).filter(tool => !isLastIteration
          || (canTransferOnLocalFinal && tool.name === MODE_TRANSFER_TOOL)) : []
      const progressAwareToolsForRequest = [
        ...withAgentActionProgressSchemas(toolsForRequest), ...controlToolsForRequest,
      ]
      if (this.modeControl?.prompt) messagesForRequest = [...messagesForRequest, { role: 'system', content: this.modeControl.prompt }]
      if (allowedToolNames) for (const tool of controlToolsForRequest) allowedToolNames.add(tool.name)
      // Adaptive OpenAI-compatible models can emit an empty native completion
      // after tool history when the final request itself has no tools. A
      // bounded final has no opportunity for another transport-recovery turn,
      // so serialize that one request through the provider-safe prompt path
      // immediately instead of paying for a known empty native attempt first.
      const useNativeToolUse = nativeToolUse
        && !preferPromptReact
        && !(useBoundedFinalContext && modelInfo?.capabilities.adaptivePromptReact === true)

      // Keep amendments outside compactable history and after older contract hints.
      const steeringContent = [
        formatActiveUserInstructions(activeUserInstructions(this.liveRunState?.steeringNotes)),
        ...freshSteering.filter((note) => note.kind === 'question').map((note) =>
          `[User question during execution] ${note.message}`),
      ].filter(Boolean).join('\n\n')
      if (steeringContent) messagesForRequest = [...messagesForRequest, { role: 'system', content: steeringContent }]
      const backgroundEvidence = this.activeRuns?.backgroundEvidence(context.sessionId)
      if (backgroundEvidence) messagesForRequest = [...messagesForRequest, { role: 'user', content: backgroundEvidence }]
      const messagesWithBudgetHint = isLastIteration && !canAutoContinueAfterBudget
        ? canTransferOnLocalFinal && controlToolsForRequest.length > 0
          ? [...messagesForRequest, { role: 'system' as const, content: 'This is the final step in the current mode. Answer from retained evidence, or call agent.transfer alone if the remaining goal needs another execution mode. No further actions or discovery can run in this mode. The original turn budget and all authorization boundaries remain in force.' }]
          : appendForceFinalSystemMessage(messagesForRequest)
        : isPenultimateIteration && !canAutoContinueAfterBudget
          ? appendBudgetWarningSystemMessage(messagesForRequest)
          : messagesForRequest
      const providerProtocolRepair = repairProviderMessageProtocol(messagesWithBudgetHint)
      let requestMessages = providerProtocolRepair.messages

      // A satisfied deterministic skill policy is already grounded in exact
      // successful tool evidence. Running a second LLM reviewer in that state
      // can only ask for duplicate work, so the shared policy gate is final.
      const willReviewCandidateFinal = !deterministicSkillCompletionSatisfiedBeforeRequest
        && !skipOutcomeReview
        && shouldReviewOutcomeWithLLM(messages, {
          includeToollessFinal: this.reviewToollessFinals,
        })
      const configuredRequestedOutputTokens = resolvedMaxTokens
        ?? Math.min(4096, Math.max(256, Math.floor(contextWindow / 2)))
      const normalRequestedOutputTokens = effectiveMaxOutputTokens === undefined
        ? configuredRequestedOutputTokens
        : Math.min(configuredRequestedOutputTokens, effectiveMaxOutputTokens)
      const requestedOutputTokens = Math.min(
        normalRequestedOutputTokens,
        useBoundedFinalContext
          ? modelInfo?.capabilities.thinking === true
            ? BOUNDED_FINAL_THINKING_MAX_OUTPUT_TOKENS
            : BOUNDED_FINAL_MAX_OUTPUT_TOKENS
          : Number.POSITIVE_INFINITY,
        forceFinalAfterDegenerateOutput ? 2_048 : Number.POSITIVE_INFINITY,
      )
      const finalStepTools = isLastIteration && !canAutoContinueAfterBudget
        ? controlToolsForRequest : progressAwareToolsForRequest
      let requestTools = useNativeToolUse && finalStepTools.length > 0
        ? finalStepTools
        : undefined
      let effectiveAllowedToolNames = new Set((useNativeToolUse
        ? requestTools ?? []
        : useBoundedFinalContext ? [] : finalStepTools).map((tool) => tool.name))
      let renderedToolDefinitions = finalStepTools
      let contextFit: ProviderContextFitResult | undefined
      try {
        const toolDefinitionsActive = useNativeToolUse
          ? (requestTools?.length ?? 0) > 0
          : !useBoundedFinalContext && renderedToolDefinitions.length > 0
        contextFit = fitProviderContext(requestMessages, {
          contextWindowTokens: contextWindow,
          requestedOutputTokens,
          modelMaxOutputTokens: this.maxTokens === undefined
            ? effectiveMaxOutputTokens === undefined
              ? modelInfo?.maxOutputTokens
              : Math.min(effectiveMaxOutputTokens, modelInfo?.maxOutputTokens ?? Number.POSITIVE_INFINITY)
            : undefined,
          minimumOutputTokens: toolDefinitionsActive
            ? TOOL_ENABLED_MIN_OUTPUT_TOKENS
            : undefined,
          charsPerToken,
          tools: requestTools,
          renderMessages: useNativeToolUse
            ? undefined
            : useBoundedFinalContext
              ? buildPromptFinalMessages
              : (selectedMessages) => buildPromptReActMessages(
                  selectedMessages,
                  renderedToolDefinitions,
                ),
        })
      } catch (error) {
        let fitError: unknown = error
        if (error instanceof IrreducibleContextOverflowError && toolsForRequest.length > 0) {
          try {
            requestMessages = repairProviderMessageProtocol(
              appendForceFinalSystemMessage(messagesForRequest),
            ).messages
            requestTools = undefined
            effectiveAllowedToolNames = new Set<string>()
            renderedToolDefinitions = []
            contextFit = fitProviderContext(requestMessages, {
              contextWindowTokens: contextWindow,
              requestedOutputTokens,
              modelMaxOutputTokens: this.maxTokens === undefined
                ? modelInfo?.maxOutputTokens
                : undefined,
              charsPerToken,
              renderMessages: useNativeToolUse
                ? undefined
                : useBoundedFinalContext
                  ? buildPromptFinalMessages
                  : (selectedMessages) => buildPromptReActMessages(selectedMessages, []),
            })
            yield {
              type: 'thinking',
              content: '[context budget] Tool definitions no longer fit beside the retained evidence; disabled further tools and reserved this request for a final synthesis.',
            }
          } catch (fallbackError) {
            fitError = fallbackError
          }
        }
        if (!contextFit) {
          if (!(fitError instanceof IrreducibleContextOverflowError)) throw fitError
          this.state = 'error'
          // A model missing from the catalog has no declared context window, so
          // the conservative default applies and a normal turn can overflow it.
          // Say so: the window in the message is a guess, not the model's real
          // limit, and refreshing the provider's model list is the actual fix.
          const message = modelInfo
            ? fitError.message
            : `${fitError.message} Model '${context.model}' is not in provider `
              + `'${this.provider.id}' model list, so a conservative default `
              + 'context window was assumed. Refresh the provider\'s models '
              + '(Settings → AI Models → 모델 가져오기) so its real limits are used.'
          yield {
            type: 'error',
            error: {
              code: 'CONTEXT_LENGTH',
              message,
            },
          }
          await this.finalizeReactContextLengthFailure(context, totalUsage, i + 1, message)
          return
        }
      }
      if (!contextFit) throw new Error('Context fitting completed without a result.')
      const request: ChatRequest = {
        model: context.model,
        messages: contextFit.requestMessages,
        tools: requestTools,
        thinkingLevel: useBoundedFinalContext
          ? ThinkingLevel.Off
          : disableThinkingForProviderRecovery
          ? undefined
          : resolveThinkingLevel(this.thinkingLevel, {
              toolNames: requestTools?.map((tool) => tool.name),
            }),
        maxTokens:
          !useBoundedFinalContext
          && !forceFinalAfterDegenerateOutput
          && resolvedMaxTokens === undefined
          && !contextFit.outputTokensReduced
          && contextFit.droppedMessageCount === 0
          ? undefined
          : contextFit.maxOutputTokens,
        temperature: this.temperature,
        timeoutMs: useBoundedFinalContext ? BOUNDED_FINAL_TIMEOUT_MS : undefined,
      }
      const suppressTextDeltas =
        requireFinalAnswerProtocol
        || useBoundedFinalContext
        || willReviewCandidateFinal
        || requiresVerifiedMemoryWrite
        || requiresVerifiedScheduleCreate
        || requiresVerifiedAction
        || requiresVerifiedSkillExecution
        || lengthContinuationPrefix.length > 0
      const llmTurnId = buildLlmTurnId(context.sessionId, i + 1, 'react')

      // Spend-cap gate — refuse the dispatch BEFORE the provider call so a
      // runaway loop cannot burn unbounded provider dollars. Only priced spend
      // counts, so local-model runs are never blocked.
      if (this.usageTracker && this.spendBudget) {
        const spend = checkSpendBudget(this.usageTracker, this.spendBudget, {
          sessionId: context.sessionId,
        })
        if (!spend.allowed) {
          yield {
            type: 'error',
            error: { code: 'SPEND_BUDGET_EXCEEDED', message: spend.reason ?? 'spend budget exceeded' },
          }
          this.state = 'done'
          yield { type: 'done', usage: totalUsage, stopReason: stopReasonSpendBudget() }
          return
        }
      }

      // pre:llm:call gate — fires before every provider call so a policy hook
      // can block (or inspect) the outbound request. Previously advertised in the
      // command-hook schema but never emitted, so registering it was a silent
      // no-op; emit it here at the react LLM-call boundary.
      if (this.hookRegistry) {
        const preLlm = await this.hookRegistry.trigger({
          event: 'pre:llm:call',
          data: {
            sessionId: context.sessionId,
            provider: context.provider,
            model: context.model,
            iteration: i + 1,
            request,
          },
        }, signal)
        if (signal?.aborted) {
          this.state = 'done'
          yield { type: 'done', usage: totalUsage, stopReason: stopReasonUserAbort() }
          return
        }
        if (preLlm.action === 'abort') {
          yield {
            type: 'error',
            error: { code: 'FORBIDDEN', message: preLlm.reason ?? 'LLM call blocked by hook' },
          }
          return
        }
      }

      try {
        const deferredTextDeltas: string[] = []
        let deferredCandidateAccepted = !willReviewCandidateFinal
        yield buildContextUsageEvent({
          inputTokens: contextFit.estimatedInputTokens,
          contextWindowTokens: contextWindow,
          reservedOutputTokens: contextFit.maxOutputTokens,
          iteration: i + 1,
          source: 'estimated',
        })
        yield buildLlmRequestEvent({
          sessionId: context.sessionId,
          iteration: i + 1,
          source: 'react',
          request,
          turnId: llmTurnId,
          providerId: this.provider.id,
          timeoutMs: resolveProviderStreamFirstTokenMs(),
        })
        const streamGen = useNativeToolUse
          ? this.streamWithDeltas(
              request,
              signal,
              effectiveAllowedToolNames,
              suppressTextDeltas,
              deferredTextDeltas,
            )
          : this.streamWithPromptReActFallback(
              request,
              signal,
              effectiveAllowedToolNames,
              suppressTextDeltas,
              deferredTextDeltas,
            )
        let streamResult = await streamGen.next()
        while (!streamResult.done) {
          yield streamResult.value
          streamResult = await streamGen.next()
        }
        const response = streamResult.value
        const providerContextUsage = buildProviderContextUsageEvent({
          usage: response.usage,
          contextWindowTokens: contextWindow,
          reservedOutputTokens: contextFit.maxOutputTokens,
          iteration: i + 1,
        })
        if (providerContextUsage) yield providerContextUsage
        await logLlmCallTrace({
          source: 'react',
          mode: 'react',
          sessionId: context.sessionId,
          provider: context.provider,
          model: context.model,
          iteration: i + 1,
          request,
          response,
          meta: {
            turnId: llmTurnId,
            finishReason: response.finishReason,
            toolCallCount: response.message.toolCalls?.length ?? 0,
          },
        })
        await this.safeTriggerPostHook('post:llm:call', {
          sessionId: context.sessionId,
          provider: context.provider,
          model: context.model,
          iteration: i + 1,
          request,
          response,
          finishReason: response.finishReason,
          toolCallCount: response.message.toolCalls?.length ?? 0,
        })

        this.recordLlmUsage(context, response.usage)
        totalUsage.inputTokens += response.usage.inputTokens
        totalUsage.outputTokens += response.usage.outputTokens

        // A response planned before a new instruction must not execute stale
        // actions or finish the turn. Account for its usage, then re-plan.
        if (this.aborted || signal?.aborted) break
        if (this.hasPendingSteering()) continue

        const requestHasNativeTools = (request.tools?.length ?? 0) > 0
        const hasCompletedNativeToolHistory =
          messages.some((message) =>
            message.role === 'assistant' && (message.toolCalls?.length ?? 0) > 0)
          && messages.some((message) => message.role === 'tool')
        const detectedEmptyNativeToolResponse =
          useNativeToolUse
          && (requestHasNativeTools || hasCompletedNativeToolHistory)
          && modelInfo?.capabilities.adaptivePromptReact === true
          && toolTransportRecoveryCount < 1
            ? detectEmptyNativeToolResponse(response)
            : null
        const emptyNativeToolResponse =
          detectedEmptyNativeToolResponse && !requestHasNativeTools
            ? { ...detectedEmptyNativeToolResponse, source: 'empty_native_completion' as const }
            : detectedEmptyNativeToolResponse
        if (emptyNativeToolResponse) {
          toolTransportRecoveryCount += 1
          preferPromptReact = true
          messages.push(buildToolTransportRecoveryMessage(emptyNativeToolResponse))
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield buildToolTransportRecoveryEvent(emptyNativeToolResponse)
          yield {
            type: 'thinking',
            content: '[provider recovery] Native tool transport completed without a usable response; retrying once with prompt tool transport.',
          }
          i -= 1
          continue
        }

        const controlResult = this.modeControl?.handle(response.message, {
          messages, usage: { ...totalUsage }, iterations: i + 1,
        })
        yield* this.modeControl?.drainEvents() ?? []
        if (controlResult === 'transfer') return
        if (controlResult === 'continue') continue

        const suppressToolCallsForFinalIteration =
          isLastIteration
          && !canAutoContinueAfterBudget
          && (response.message.toolCalls?.length ?? 0) > 0
        const toolCalls = suppressToolCallsForFinalIteration
          ? undefined
          : response.message.toolCalls
        const contentActionProgress = toolCalls?.length
          ? extractAgentActionProgress([
              this.responseTextContent(response),
              response.thinking ?? '',
            ].filter(Boolean).join('\n'))
          : null
        const argumentActionProgress = toolCalls?.length
          ? consumeToolCallActionProgress(toolCalls, request.tools)
          : null
        const actionProgress = argumentActionProgress ?? contentActionProgress
        if (toolCalls && toolCalls.length > 0) {
          if (approvalGraceState.denial) {
            // The side-effect-free turn after a denial may inspect, never act.
            // A non-read-only call (including a retry of the denied one) is
            // not executed; the run ends as approval_denied instead.
            const sideEffectingCall = toolCalls.find((call) => (
              !isPolicyReadOnlyTool(call.name)
            ))
            if (sideEffectingCall || approvalGraceState.toolTurnsRemaining <= 0) {
              yield {
                type: 'thinking',
                content: sideEffectingCall
                  ? `[supervisor] The model requested ${sideEffectingCall.name} after the user declined ${approvalGraceState.denial.toolName ?? 'the tool'}; stopping without executing it.`
                  : '[supervisor] The post-denial turn budget is spent; stopping without further tool calls.',
              }
              yield* this.finishAfterApprovalDenial(
                context,
                totalUsage,
                i + 1,
                approvalGraceState.denial,
                latestUserText(messages),
              )
              return
            }
          }
          const duplicateToolCallGroups = findDuplicateToolCallGroups(toolCalls)
          if (
            shouldRepairDuplicateToolCalls({
              duplicateGroups: duplicateToolCallGroups,
              repairedCount: repairedDuplicateToolCallCount,
              isLastIteration,
            })
          ) {
            repairedDuplicateToolCallCount += 1
            messages.push(response.message)
            messages.push(buildDuplicateToolCallRepairMessage(duplicateToolCallGroups))
            this.state = 'observing'
            yield { type: 'state_change', state: 'observing' }
            continue
          }

          let executableToolCalls = toolCalls
          if (activeSkillExecutionPolicies.length > 0) {
            const partition = partitionSkillExecutionToolCalls(
              messages,
              toolCalls,
              activeSkillExecutionPolicies,
            )
            if (partition.rejectedCalls.length > 0) {
              if (repairedSkillExecutionToolCount < 1) {
                repairedSkillExecutionToolCount += 1
              }
              executableToolCalls = partition.executableCalls
              messages.push(buildSkillExecutionToolRepairMessage(
                [...new Set(partition.rejectedCalls.map((call) => call.name))],
                [...new Set(executableToolCalls.map((call) => call.name))],
              ))
              yield {
                type: 'thinking',
                content: '[supervisor] Skipped tool calls that violated the active skill execution policy while preserving valid calls.',
              }
              if (executableToolCalls.length === 0) {
                this.state = 'observing'
                yield { type: 'state_change', state: 'observing' }
                continue
              }
            }
          }

          executableToolCalls = normalizeExplicitScheduleListToolCalls(
            executableToolCalls,
            latestUserText(messages),
          )

          const deferredExactToolBudgetRepairs: Array<() => Message> = []
          const deferredCanonicalReadTargetRepairs: Array<() => Message> = []
          const canonicalReadTarget = partitionToolCallsByCanonicalReadTarget(
            focusedRepositoryRequest,
            executableToolCalls,
            canonicalReadTargetIdentities,
          )
          executableToolCalls = canonicalReadTarget.executable
          if (canonicalReadTarget.nonCanonical.length > 0) {
            deferredCanonicalReadTargetRepairs.push(
              () => buildCanonicalReadTargetRepairMessage(
                canonicalReadTarget.nonCanonical,
              ),
            )
            await logAgentDebugTrace({
              event: 'supervisor.canonical-integration-read-target',
              source: 'react',
              sessionId: context.sessionId,
              runId: context.sessionId,
              iteration: i + 1,
              status: 'blocked-generic-transport',
              data: {
                rejected: canonicalReadTarget.nonCanonical.map(({ call, canonicalTool }) => ({
                  requestedTool: call.name,
                  canonicalTool,
                })),
                executable: executableToolCalls.map((call) => call.name),
              },
            })
            yield {
              type: 'thinking',
              content: '[supervisor] Rejected a generic URL transport for an integration-owned read route; use its registered canonical tool.',
            }
          }
          const exactToolCallHistory = toolCallBudgetHistoryFromCurrentTurnMessages(messages)
          const exactToolBudget = partitionToolCallsByExactOnceBudget(
            focusedRepositoryRequest,
            exactToolCallHistory,
            executableToolCalls,
            exactToolCardinalityIdentities,
          )
          executableToolCalls = exactToolBudget.executable
          if (exactToolBudget.nonCanonical.length > 0) {
            deferredExactToolBudgetRepairs.push(
              () => buildCanonicalExactOnceToolRepairMessage(exactToolBudget.nonCanonical),
            )
            await logAgentDebugTrace({
              event: 'supervisor.canonical-exact-tool-capability',
              source: 'react',
              sessionId: context.sessionId,
              runId: context.sessionId,
              iteration: i + 1,
              status: 'blocked-substitute',
              data: {
                rejected: exactToolBudget.nonCanonical.map(({ call, canonicalTool }) => ({
                  requestedTool: call.name,
                  canonicalTool,
                })),
                executable: executableToolCalls.map((call) => call.name),
              },
            })
            yield {
              type: 'thinking',
              content: '[supervisor] Rejected an endpoint-targeted substitute so the registered canonical capability keeps its exactly-once allowance.',
            }
          }
          if (exactToolBudget.exhausted.length > 0) {
            const exhaustedNames = [
              ...new Set(exactToolBudget.exhausted.map((call) => call.name)),
            ]
            const buildRepairMessage = (): Message => {
              const remaining = remainingExactOnceToolNames(
                focusedRepositoryRequest,
                toolCallBudgetHistoryFromCurrentTurnMessages(messages),
                exactToolCardinalityIdentities,
              )
              const closesToolSurface = inputClosesExactOnceToolSet(
                focusedRepositoryRequest,
                exactToolCardinalityIdentities,
              )
              const pendingSkillStages = evaluateSkillExecutionCompletion(
                messages,
                activeSkillExecutionPolicies,
              ).missing
              return {
                role: 'system',
                metadata: { reminderKind: 'exact_tool_call_budget' },
                content: [
                  '[Exact tool-call budget enforced]',
                  `Skipped ${exactToolBudget.exhausted.length} duplicate call(s) after their explicit exactly-once budget was already consumed: ${exhaustedNames.join(', ')}.`,
                  remaining.length > 0
                    ? `Do not retry or substitute the exhausted tool. Continue only with the still-required exactly-once tool(s): ${remaining.join(', ')}; then produce the final answer.`
                    : closesToolSurface
                      ? 'Do not retry or substitute another tool. Use the existing current-turn result and produce the requested final answer; if that result failed, report the failure honestly.'
                      : pendingSkillStages.length > 0
                        ? `Do not retry the exhausted tool. Continue only with the active skill's missing stage(s): ${pendingSkillStages.flatMap((stage) => stage.tools).join(', ')}; then produce the final answer.`
                        : 'Do not retry or substitute the exhausted tool. Use the existing current-turn result and produce the requested final answer; if that result failed, report the failure honestly.',
                ].join(' '),
              }
            }
            await logAgentDebugTrace({
              event: 'supervisor.exact-tool-call-budget',
              source: 'react',
              sessionId: context.sessionId,
              runId: context.sessionId,
              iteration: i + 1,
              status: 'blocked-duplicate',
              data: {
                exhausted: exactToolBudget.exhausted.map((call) => call.name),
                executable: executableToolCalls.map((call) => call.name),
              },
            })
            yield {
              type: 'thinking',
              content: `[supervisor] Enforced the explicit exactly-once tool budget for ${exhaustedNames.join(', ')}; duplicate execution was skipped.`,
            }
            deferredExactToolBudgetRepairs.push(buildRepairMessage)
          }
          if (
            (
              deferredExactToolBudgetRepairs.length > 0
              || deferredCanonicalReadTargetRepairs.length > 0
            )
            && executableToolCalls.length === 0
          ) {
            messages.push({ ...response.message, toolCalls: undefined })
            messages.push(...deferredCanonicalReadTargetRepairs.map((build) => build()))
            messages.push(...deferredExactToolBudgetRepairs.map((build) => build()))
            if (
              deferredExactToolBudgetRepairs.length > 0
              && exactOnceToolBudgetComplete(
                focusedRepositoryRequest,
                exactToolCallHistory,
                exactToolCardinalityIdentities,
              )
              && (
                inputClosesExactOnceToolSet(
                  focusedRepositoryRequest,
                  exactToolCardinalityIdentities,
                )
                || evaluateSkillExecutionCompletion(
                  messages,
                  activeSkillExecutionPolicies,
                ).missing.length === 0
              )
            ) {
              forceFinalAfterObservationBudget = true
            }
            this.state = 'observing'
            yield { type: 'state_change', state: 'observing' }
            continue
          }

          const toolCallBatch = capToolCallBatch(
            executableToolCalls,
            resolveToolCallBatchLimit(contextWindow, this.maxToolCallsPerTurn),
          )
          if (toolCallBatch.capped) {
            executableToolCalls = toolCallBatch.accepted
            forceFinalAfterObservationBudget = true
            yield {
              type: 'thinking',
              content: `[supervisor] Bounded a ${toolCallBatch.requestedCount}-call tool batch to ${toolCallBatch.accepted.length}; the next step will synthesize the retained evidence.`,
            }
          }

          const observationReuse = partitionCallsCoveredByCurrentTurnObservations(
            messages,
            executableToolCalls,
            this.tools,
            { cwd: context.cwd, workspaceRoot: context.workspaceRoot },
            isPolicyReadOnlyTool,
            observationHistory,
          )
          let observationReuseMessage: Message | undefined
          if (
            observationReuse.coveredCalls.length > 0
            || observationReuse.narrowedCalls.length > 0
          ) {
            executableToolCalls = observationReuse.executableCalls
            observationReuseMessage = buildObservationReuseMessage(
              observationReuse.coveredCalls,
              observationReuse.narrowedCalls,
            )
            await logAgentDebugTrace({
              event: 'supervisor.observation-reuse',
              source: 'react',
              sessionId: context.sessionId,
              runId: context.sessionId,
              iteration: i + 1,
              status: executableToolCalls.length === 0 ? 'force-final' : 'partial-reuse',
              data: {
                skippedCalls: observationReuse.coveredCalls.map(({ requested, observed }) => ({
                  requestedTool: requested.name,
                  requestedCallId: requested.id,
                  observedCallId: observed?.id,
                })),
                narrowedCalls: observationReuse.narrowedCalls.map(({ requested, replacements }) => ({
                  requestedTool: requested.name,
                  requestedCallId: requested.id,
                  replacementCount: replacements.length,
                })),
                executableCalls: executableToolCalls.map((call) => call.name),
              },
            })
            yield {
              type: 'thinking',
              content: `[supervisor] Reused ${observationReuse.coveredCalls.length} successful current-turn read observation(s), narrowed ${observationReuse.narrowedCalls.length} overlapping request(s), and retained ${executableToolCalls.length} still-unobserved call(s).`,
            }
            if (executableToolCalls.length === 0) {
              forceFinalAfterRepeatedRead = true
              messages.push(response.message)
              messages.push(...buildObservationReuseToolResultMessages(
                observationReuse.coveredCalls,
              ))
              messages.push(observationReuseMessage)
              this.state = 'observing'
              yield { type: 'state_change', state: 'observing' }
              continue
            }
          }

          // Write-loop guard (CLI_BACKLOG.md B6): if the model keeps rewriting
          // the same target, skip this write turn and tell it to stop/finalize
          // instead of letting it burn the iteration budget rewriting one file
          // dozens of times. Mirrors the duplicate-tool-call repair shape, so no
          // control-flow change; the limit is deliberately generous.
          const writeLoopTarget = writeLoopTracker.offendingTarget(executableToolCalls)
          if (writeLoopTarget) {
            messages.push(response.message)
            messages.push(buildWriteLoopRepairMessage(writeLoopTarget, writeLoopLimit))
            this.state = 'observing'
            yield { type: 'state_change', state: 'observing' }
            continue
          }

          const guardedObservationCalls = executableToolCalls.filter((call) => (
            REACT_CONSECUTIVE_READ_GUARD_TOOLS.has(call.name)
            && isPolicyReadOnlyRequest({
              tool: call.name,
              input: call.arguments,
              cwd: context.cwd,
              workspaceRoot: context.workspaceRoot,
              registrationSource: this.tools.registrationSource(call.name),
              security: this.tools.securityDescriptor(call.name),
            })
          ))
          const proposedReadEntries: StuckToolRepeatEntry[] = guardedObservationCalls
            .map((call) => ({
              tool: call.name,
              input: call.arguments,
              status: 'success',
              ts: Date.now(),
            }))
          const readLoop = detectStuckToolRepeat(
            [...readOnlyToolHistory, ...proposedReadEntries],
            {
              trackedTools: REACT_CONSECUTIVE_READ_GUARD_TOOLS,
              targetRepeatThreshold: DEFAULT_TARGET_REPEAT_THRESHOLD,
            },
          )
          if (readLoop.stuck && readLoop.tool && readLoop.count) {
            forceFinalAfterRepeatedRead = true
            messages.push(response.message)
            messages.push(buildStuckToolRepeatMessage(
              readLoop.tool,
              readLoop.count,
              readLoop.kind,
            ))
            await logAgentDebugTrace({
              event: 'supervisor.read-loop',
              source: 'react',
              sessionId: context.sessionId,
              runId: context.sessionId,
              iteration: i + 1,
              status: 'force-final',
              data: {
                kind: readLoop.kind,
                tool: readLoop.tool,
                count: readLoop.count,
                uniqueSignatures: readLoop.uniqueSignatures,
                observedReadCalls: readOnlyToolHistory.length,
                skippedCalls: proposedReadEntries.length,
              },
            })
            this.state = 'observing'
            yield { type: 'state_change', state: 'observing' }
            yield {
              type: 'thinking',
              content: `[supervisor] Stopped a ${readLoop.kind ?? 'repeated'} read loop after ${readLoop.count} matching calls; synthesizing from existing evidence.`,
            }
            continue
          }

          if (executableToolCalls.length === 1) {
            const [onlyCall] = executableToolCalls
            const signature = onlyCall && guardedObservationCalls.includes(onlyCall)
              ? signatureOf({ tool: onlyCall.name, input: onlyCall.arguments })
              : null
            if (signature && signature === lastReadOnlyToolSignature) {
              forceFinalAfterRepeatedRead = true
              messages.push(response.message)
              messages.push(buildStuckToolRepeatMessage(onlyCall!.name, 2, 'exact'))
              this.state = 'observing'
              yield { type: 'state_change', state: 'observing' }
              yield {
                type: 'thinking',
                content: `[supervisor] Skipped repeated ${onlyCall!.name}; the identical result is already available, so synthesize the answer now.`,
              }
              continue
            }
            lastReadOnlyToolSignature = signature
          } else {
            lastReadOnlyToolSignature = null
          }

          this.state = 'acting'
          if (actionProgress) {
            yield {
              type: 'action_progress',
              summary: actionProgress.summary,
              nextStep: actionProgress.nextStep,
              toolNames: [...new Set(executableToolCalls.map((call) => call.name))],
            }
          }
          yield { type: 'state_change', state: 'acting' }
          messages.push(
            executableToolCalls === toolCalls
              ? response.message
              : { ...response.message, toolCalls: executableToolCalls },
          )
          let focusedActionResult: ToolResultEvent | undefined
          const requiredUserAction = yield* this.runToolCalls(
            messages,
            executableToolCalls,
            context,
            i,
            totalUsage,
            undefined,
            signal,
            canAttachVisualContent,
            rawTerminalOutputRequested
              || focusedProcessStart
              || focusedProcessObservation
              || focusedBrowserObservation
              || focusedVisibleUrlOpen
              ? (event) => { focusedActionResult = event }
              : undefined,
          )
          if (requiredUserAction) {
            yield* this.finishAfterUserActionRequired(requiredUserAction, context, totalUsage)
            return
          }
          if (closedExactOnceToolBudgetComplete(
            focusedRepositoryRequest,
            toolCallBudgetHistoryFromCurrentTurnMessages(messages),
            exactToolCardinalityIdentities,
          )) {
            forceFinalAfterObservationBudget = true
            closedExactToolWorkflowComplete = true
            messages.push({
              role: 'system',
              metadata: { reminderKind: 'closed_exact_tool_call_budget' },
              content: [
                '[Closed exact tool-call workflow complete]',
                'Every explicitly permitted exactly-once tool has a trusted current-turn outcome.',
                'The user excluded all other tools, so the evidence phase is closed even when an outcome failed.',
                'Produce one concise, honest user-facing final answer from the stored result; do not retry, substitute, or ask for more tool evidence.',
              ].join(' '),
            })
            yield {
              type: 'thinking',
              content: '[supervisor] The closed exactly-once tool workflow has an outcome; switching directly to one tool-free final synthesis.',
            }
          }
          if (deferredExactToolBudgetRepairs.length > 0) {
            messages.push(...deferredExactToolBudgetRepairs.map((build) => build()))
            if (exactOnceToolBudgetComplete(
              focusedRepositoryRequest,
              toolCallBudgetHistoryFromCurrentTurnMessages(messages),
              exactToolCardinalityIdentities,
            ) && (
              inputClosesExactOnceToolSet(
                focusedRepositoryRequest,
                exactToolCardinalityIdentities,
              )
              || evaluateSkillExecutionCompletion(
                messages,
                activeSkillExecutionPolicies,
              ).missing.length === 0
            )) {
              forceFinalAfterObservationBudget = true
            }
          }
          if (deferredCanonicalReadTargetRepairs.length > 0) {
            messages.push(...deferredCanonicalReadTargetRepairs.map((build) => build()))
          }
          if (observationReuseMessage) messages.push(observationReuseMessage)
          const completedReadEntries = guardedObservationCalls.map((call) => {
            const resultMessage = [...messages].reverse().find((message) => (
              message.role === 'tool' && message.toolCallId === call.id
            ))
            const status = resultMessage?.metadata?.toolResultStatus === 'success'
              ? 'success' as const
              : 'error' as const
            const blocked = blockSourceFromMetadata(resultMessage?.metadata) !== null
            return {
              tool: call.name,
              input: call.arguments,
              status,
              ...(status === 'error' && !blocked
                ? { failureCode: structuredToolFailureCode(resultMessage) }
                : {}),
              ...(blocked ? { blocked: true } : {}),
              ts: Date.now(),
            }
          })
          readOnlyToolHistory.push(...completedReadEntries)
          for (const call of executableToolCalls) {
            const resultMessage = [...messages].reverse().find((message) => (
              message.role === 'tool' && message.toolCallId === call.id
            ))
            observationHistory.push({
              tool: call.name,
              input: { ...call.arguments },
              status: resultMessage?.metadata?.toolResultStatus === 'success'
                ? 'success'
                : 'error',
              ts: Date.now(),
              ...(typeof resultMessage?.content === 'string'
                ? { output: observationHistoryOutput(call.name, resultMessage.content) }
                : {}),
            })
          }
          const batchWasEntirelyReadOnly = guardedObservationCalls.length > 0
            && guardedObservationCalls.length === executableToolCalls.length
          if (batchWasEntirelyReadOnly) {
            readOnlyEvidenceTurns += 1
          }
          const successfulReadObservationCount = readOnlyToolHistory.filter(
            (entry) => entry.status === 'success',
          ).length
          if (
            this.autonomy === AutonomyLevel.ReadOnly
            && activeSkillExecutionPolicies.length === 0
            && (context.executionSkillIds?.length ?? 0) === 0
            && readOnlyEvidenceTurns >= READ_ONLY_EVIDENCE_TURN_LIMIT
            && successfulReadObservationCount >= READ_ONLY_EVIDENCE_MIN_SUCCESSFUL_OBSERVATIONS
          ) {
            forceFinalAfterObservationBudget = true
            yield {
              type: 'thinking',
              content: `[supervisor] Read-only evidence budget reached after ${readOnlyEvidenceTurns} tool turns and ${successfulReadObservationCount} successful observations; switching to bounded final synthesis.`,
            }
          }
          if (toolCallBatch.capped) {
            messages.push(buildToolCallBatchCapMessage(toolCallBatch))
          }
          const approvalDenial = resolveApprovalOutcomeAfterTools()
          if (approvalDenial) {
            yield* this.finishAfterApprovalDenial(
              context,
              totalUsage,
              i + 1,
              approvalDenial,
              latestUserText(messages),
            )
            return
          }
          const currentPolicyFailure = this.autonomy === AutonomyLevel.ReadOnly
            ? latestTrustedPolicyFailure(messages, ignoredReadOnlyPolicyFailureToolCallIds)
            : null
          if (currentPolicyFailure) {
            const resolution = resolveReadOnlyPolicyFailure(currentPolicyFailure)
            await logAgentDebugTrace({
              event: 'supervisor.policy-block',
              source: 'react',
              sessionId: context.sessionId,
              runId: context.sessionId,
              iteration: i + 1,
              status: resolution === 'repair'
                ? 'bounded-invocation-repair'
                : resolution === 'synthesize'
                  ? 'bounded-evidence-synthesis'
                  : 'deterministic-final',
              data: {
                autonomy: this.autonomy,
                toolName: currentPolicyFailure.toolName,
                reason: currentPolicyFailure.reason,
              },
            })
            if (resolution !== 'terminate') {
              this.state = 'observing'
              yield {
                type: 'thinking',
                content: resolution === 'repair'
                  ? '[supervisor] Read-only policy blocked a shell wrapper; allowing one policy-checked direct invocation repair.'
                  : '[supervisor] Read-only policy blocked a follow-up; synthesizing once from successful current-turn evidence.',
              }
              yield { type: 'state_change', state: 'observing' }
              continue
            }
            yield* this.finishAfterReadOnlyPolicyFailure(
              context,
              totalUsage,
              i + 1,
              currentPolicyFailure,
              latestUserText(messages),
            )
            return
          }
          if (rawTerminalOutputRequested && focusedActionResult) {
            yield* this.finishWithObservedCommandOutput(
              context,
              totalUsage,
              i + 1,
              focusedActionResult,
            )
            return
          }
          if (focusedActionResult) {
            const focusedToolName = executableToolCalls.find(
              (call) => call.id === focusedActionResult?.toolCallId,
            )?.name
            const deterministicResult = focusedToolName
              ? deterministicFocusedActionResult(focusedRepositoryRequest, {
                  toolName: focusedToolName,
                  status: focusedActionResult.status,
                  output: focusedActionResult.output,
                })
              : undefined
            if (deterministicResult) {
              yield* this.finalizeDeterministicScheduleResult(
                deterministicResult,
                context,
                totalUsage,
                i + 1,
              )
              return
            }
          }
          if (focusedRepositoryLookup) {
            const focusedCalls = executableToolCalls.filter((call) =>
              FOCUSED_REPOSITORY_LOOKUP_TOOLS.has(call.name)
            )
            if (focusedCalls.length > 0) {
              focusedRepositoryToolTurns += 1
              for (const call of focusedCalls) {
                focusedRepositoryObservedTools.add(call.name)
              }
            }
          }
          if (
            substantiveRepositoryReview
            && executableToolCalls.some((call) =>
              SUBSTANTIVE_REPOSITORY_REVIEW_OBSERVATION_TOOLS.has(call.name))
          ) {
            substantiveRepositoryReviewToolTurns += 1
            if (
              substantiveRepositoryReviewToolTurns
              >= SUBSTANTIVE_REPOSITORY_REVIEW_TOOL_TURN_LIMIT
            ) {
              forceFinalAfterObservationBudget = true
              yield {
                type: 'thinking',
                content: `[supervisor] Repository review evidence budget reached after ${substantiveRepositoryReviewToolTurns} tool turns; switching to bounded final synthesis.`,
              }
            }
          }
          for (const executedCall of executableToolCalls) {
            writeLoopTracker.record(executedCall.name, executedCall.arguments)
          }
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          const deterministicScheduleResult = !this.semanticRouting && deterministicScheduleResultFromMessages(
            messages,
            latestUserText(messages),
          )
          if (deterministicScheduleResult) {
            yield* this.finalizeDeterministicScheduleResult(
              deterministicScheduleResult,
              context,
              totalUsage,
              i + 1,
            )
            return
          }
          if (
            i >= iterationBudget - 1
            && canAutoContinueAfterBudget
            && evaluateProgressSince(messages, findProgressMarkerIndex(messages)).progressed
          ) {
            const nextCycle = continuationCycle + 1
            await this.persistRunCheckpoint(
              context,
              messages,
              totalUsage,
              0,
              'observing',
            )
            messages.push({
              role: 'system',
              metadata: { [CONTINUATION_MARKER_METADATA_KEY]: nextCycle },
              content: buildContinuationPrompt({
                mode: 'react',
                cycle: nextCycle,
                maxCycles: this.maxContinuationCycles,
                contract: context.runContract,
              }),
            })
            yield {
              type: 'thinking',
              content: `[continuation] Iteration budget reached; continuing automatically (${nextCycle}/${this.maxContinuationCycles}).`,
            }
            yield* this.continueRun(
              messages,
              context,
              totalUsage,
              0,
              undefined,
            signal,
            nextCycle,
            observationHistory,
            )
            return
          }
          continue
        }

        const responseContent =
          typeof response.message.content === 'string'
            ? response.message.content
            : response.message.content
                .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
                .map((p) => p.text)
                .join('')
        const responseWithoutThinking = stripPromptReActThinkingArtifacts(responseContent).text
        const visibleResponseContent = stripInternalPlannerBlocks(responseWithoutThinking)
        const degenerateResponse = isDegenerateRepeatedResponse(
          [responseContent, response.thinking ?? ''].join('\n'),
        )
        if (
          degenerateResponse
          && repairedDegenerateResponseCount < 1
          && !forceFinalAfterReadOnlyPolicyFailure
        ) {
          repairedDegenerateResponseCount += 1
          forceFinalAfterDegenerateOutput = true
          lengthContinuationPrefix = ''
          repairedLengthContinuationCount = 0
          messages.push(buildDegenerateResponseRecoveryMessage())
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield {
            type: 'thinking',
            content: '[supervisor] Discarded repeated provider output and reserved one concise, tool-free final synthesis attempt.',
          }
          continue
        }
        let content = suppressToolCallsForFinalIteration
          ? ''
          : visibleResponseContent || (responseWithoutThinking.trim()
            ? 'INCOMPLETE: Internal planner update only.'
            : '')
        const responseMessageForHistory: Message = {
          ...response.message,
          content,
        }

        if (
          content.trim() === UNUSABLE_PROMPT_TOOL_CALL_OUTPUT
          && approvalGraceState.denial
          && !suppressToolCallsForFinalIteration
        ) {
          // During the side-effect-free grace turn the outbound catalog holds
          // only read-only tools, so a call that was rejected as unavailable is
          // a retry or rewrite of the declined side effect. Repairing it would
          // only invite another attempt; end the run as approval_denied.
          yield {
            type: 'thinking',
            content: `[supervisor] The model requested an unavailable tool after the user declined ${approvalGraceState.denial.toolName ?? 'the tool'}; stopping without executing it.`,
          }
          yield* this.finishAfterApprovalDenial(
            context,
            totalUsage,
            i + 1,
            approvalGraceState.denial,
            latestUserText(messages),
          )
          return
        }
        if (
          content.trim() === UNUSABLE_PROMPT_TOOL_CALL_OUTPUT
          && repairedInvalidToolResponseCount < 1
          && !forceFinalAfterReadOnlyPolicyFailure
          && !suppressToolCallsForFinalIteration
        ) {
          const evidenceAvailable = hasSuccessfulToolEvidenceInCurrentTurn(messages)
          // An observation is evidence, not proof that the whole goal is
          // complete. Preserve the single repair opportunity when a permitted
          // handoff can activate a missing capability; never expand authority.
          const synthesisOnly = evidenceAvailable
            && !controlToolsForRequest.some(tool => tool.name === MODE_TRANSFER_TOOL)
          repairedInvalidToolResponseCount += 1
          if (synthesisOnly) forceFinalAfterDegenerateOutput = true
          messages.push(buildInvalidToolResponseRepairMessage({
            evidenceAvailable: synthesisOnly,
            // Repair against the actual outbound surface, including mode
            // control. A hidden action may require discovery/transfer first.
            availableToolNames: progressAwareToolsForRequest.map((tool) => tool.name),
          }))
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield {
            type: 'thinking',
            content: synthesisOnly
              ? '[supervisor] Rejected an invalid follow-up tool call; preserving successful evidence and switching to bounded final synthesis.'
              : '[supervisor] Rejected an invalid tool call; allowing one repair using only currently available tools.',
          }
          continue
        }

        const truncatedPromptToolCall =
          !useNativeToolUse
          && response.finishReason === 'length'
          && containsPromptToolCallEnvelope(content)

        // A tool call is an atomic protocol message. Never concatenate a
        // length-truncated JSON/envelope with a later model turn. The fragment
        // was not executed, so discard it and ask for one fresh bounded call.
        if (
          truncatedPromptToolCall
          && !degenerateResponse
          && !useBoundedFinalContext
          && !forceFinalAfterReadOnlyPolicyFailure
          && !suppressToolCallsForFinalIteration
          && repairedLengthContinuationCount < lengthContinuationMax
        ) {
          repairedLengthContinuationCount += 1
          hadLengthContinuation = true
          lengthContinuationPrefix = ''
          messages.push(buildTruncatedPromptToolCallRecoveryMessage())
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield {
            type: 'thinking',
            content: `[continuation] Discarded a truncated tool call and requested a fresh bounded call (${repairedLengthContinuationCount}/${lengthContinuationMax}).`,
          }
          continue
        }

        // A provider stream that ended without any terminal event published
        // nothing and executed no tool. That is a transport failure, not a
        // truncated answer: asking the model to "continue" from an empty
        // prefix invents a resumption point that never existed. Re-issue the
        // identical request instead, under the same bounded budget.
        if (
          !degenerateResponse
          && response.finishReason === 'length'
          && response.finishDetail === 'incomplete_stream'
          && !content.trim()
          && (response.message.toolCalls?.length ?? 0) === 0
          && !truncatedPromptToolCall
          && repairedLengthContinuationCount < lengthContinuationMax
        ) {
          repairedLengthContinuationCount += 1
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield {
            type: 'thinking',
            content: `[continuation] Provider stream ended without a terminal event and published nothing; retrying the same request (${repairedLengthContinuationCount}/${lengthContinuationMax}).`,
          }
          continue
        }

        // Length-truncation continuation: the model stopped because it hit the
        // output cap (finishReason==='length') with no usable tool call. Do not
        // finalize this partial turn as a complete answer — keep the partial
        // assistant text and ask it to continue, bounded by
        // SEPILOTD_LENGTH_CONTINUE_MAX (default 1).
        //
        // This applies to a forced tool-free synthesis turn too. That turn has
        // no tool access left to recover with, so refusing it the continuation
        // repair makes the single most important turn of the run the only one
        // that dies on truncation and reports an internal fallback to the user.
        // The iteration budget is a different constraint and still applies:
        // there is no next turn to continue into on the final iteration.
        if (
          !degenerateResponse
          && response.finishReason === 'length'
          && !truncatedPromptToolCall
          && !suppressToolCallsForFinalIteration
          && repairedLengthContinuationCount < lengthContinuationMax
        ) {
          repairedLengthContinuationCount += 1
          hadLengthContinuation = true
          lengthContinuationPrefix += content
          if (content.trim()) messages.push(responseMessageForHistory)
          messages.push({
            role: 'user',
            content: lengthRecoveryInstruction(content),
          })
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield {
            type: 'thinking',
            content: `[continuation] Previous turn was length-truncated; continuing (${repairedLengthContinuationCount}/${lengthContinuationMax}).`,
          }
          continue
        }

        const interruptedResponse = !canPublishUserFacingText(response.finishReason)
        const repeatedRecoveryFailed =
          degenerateResponse && repairedDegenerateResponseCount >= 1
        if (repeatedRecoveryFailed) {
          content = buildDegenerateResponseFallback(lastUsableRecoveryCandidate)
          responseMessageForHistory.content = content
          lengthContinuationPrefix = ''
        } else if (interruptedResponse) {
          content = interruptedUserFacingResponse(observationHistory)
          responseMessageForHistory.content = content
          lengthContinuationPrefix = ''
        } else if (lengthContinuationPrefix) {
          const continuedContent = `${lengthContinuationPrefix}${content}`
          content = hasAnyAnswerProtocolStem(content)
            ? extractPromptFinalOutput(content)
            : containsPromptToolCallEnvelope(continuedContent)
              ? UNUSABLE_PROMPT_TOOL_CALL_OUTPUT
              : extractPromptFinalOutput(continuedContent)
          responseMessageForHistory.content = content
          lengthContinuationPrefix = ''
        }
        const emptyDegenerationRecovery =
          forceFinalAfterDegenerateOutput
          && !interruptedResponse
          && content.trim().length === 0
        if (emptyDegenerationRecovery) {
          content = buildDegenerateResponseFallback(lastUsableRecoveryCandidate)
          responseMessageForHistory.content = content
        }
        if (response.finishReason !== 'length') {
          repairedLengthContinuationCount = 0
        }

        const memoryWriteOutcome = memoryWriteOutcomeFromMessages(messages)
        // An interrupted provider turn is already a terminal safety result.
        // Do not reinterpret its synthetic INCOMPLETE message as ordinary
        // progress and spend more iterations retrying an unpublishable turn.
        let terminalResponseFailure =
          interruptedResponse
          || repeatedRecoveryFailed
          || emptyDegenerationRecovery
          || closedExactToolWorkflowComplete
          || forceFinalAfterNoRetryActionFailure
          || content.trim() === UNUSABLE_PROMPT_TOOL_CALL_OUTPUT
          || (forceFinalAfterDegenerateOutput && hasIncompleteAnswerStem(content))
        const scheduleCompletionOutcome = scheduleCompletionOutcomeFromMessages(messages)
        if (
          !interruptedResponse
          && explicitScheduleCreateIntent
          && scheduleCompletionOutcome !== 'success'
        ) {
          const availableScheduleTools = availableScheduleEvidenceTools(allToolNames)
          if (
            repairedScheduleCompletionCount < 1
            && availableScheduleTools.length > 0
          ) {
            repairedScheduleCompletionCount += 1
            messages.push(responseMessageForHistory)
            messages.push(buildScheduleCompletionRecoveryMessage(availableScheduleTools))
            this.state = 'observing'
            yield { type: 'state_change', state: 'observing' }
            yield {
              type: 'thinking',
              content: '[supervisor] Future reminder request had no successful scheduling result; retrying once with an exact scheduling tool.',
            }
            continue
          }
          content = buildScheduleCompletionFailureOutput(
            scheduleCompletionOutcome,
            latestUserText(messages),
          )
          responseMessageForHistory.content = content
          terminalResponseFailure = true
        }

        if (
          !terminalResponseFailure
          && explicitMemoryWriteIntent
          && memoryWriteOutcome === 'success'
        ) {
          content = restoreSuppressedMemoryWriteDraft(messages, content)
          responseMessageForHistory.content = content
        } else if (
          !terminalResponseFailure
          && explicitMemoryWriteIntent
          && memoryWriteOutcome !== 'success'
        ) {
          const memoryRememberAvailable = allToolNames.has(MEMORY_REMEMBER_TOOL_NAME)
          if (
            memoryWriteOutcome === 'missing'
            && repairedMemoryWriteCount < 1
            && memoryRememberAvailable
            && !forceFinalAfterReadOnlyPolicyFailure
          ) {
            repairedMemoryWriteCount += 1
            messages.push(responseMessageForHistory)
            messages.push(buildMemoryWriteRecoveryMessage())
            this.state = 'observing'
            yield { type: 'state_change', state: 'observing' }
            yield {
              type: 'thinking',
              content: '[supervisor] Explicit memory request had no successful memory.remember result; retrying once with a required tool call.',
            }
            continue
          }
          content = buildMemoryWriteFailureOutput(memoryWriteOutcome, {
            candidate: content,
            userInput: latestUserText(messages),
          })
          responseMessageForHistory.content = content
          terminalResponseFailure = true
        }

        const skillExecutionCompletion = evaluateSkillExecutionCompletion(
          messages,
          activeSkillExecutionPolicies,
        )
        const deterministicSkillCompletionSatisfied =
          hasDeterministicSkillCompletionPolicy(activeSkillExecutionPolicies)
          && skillExecutionCompletion.missing.length === 0
        if (!terminalResponseFailure && skillExecutionCompletion.missing.length > 0) {
          const missingStageToolsAvailable = skillExecutionCompletion.missing.every((missing) =>
            missing.tools.some((toolName) => allToolNames.has(toolName))
          )
          if (
            missingStageToolsAvailable
            && !forceFinalAfterReadOnlyPolicyFailure
            && repairedSkillExecutionCompletionCount < skillExecutionCompletion.maxRetries
            // A completion repair contributes its own bounded inflation to
            // effectiveMaxIterations(). Allow it to start at the ordinary
            // budget boundary; otherwise the guard prevents the counter from
            // increasing precisely when that extra budget is needed. An
            // explicit hard cap must remain absolute.
            && (!isLastIteration || canAutoContinueAfterBudget || !this.hardMaxIterations)
          ) {
            repairedSkillExecutionCompletionCount += 1
            messages.push(responseMessageForHistory)
            messages.push(buildSkillExecutionCompletionRecoveryMessage(skillExecutionCompletion))
            this.state = 'observing'
            yield { type: 'state_change', state: 'observing' }
            yield {
              type: 'thinking',
              content: '[supervisor] Active skill completion lacked exact successful tool evidence; retrying the missing policy stage.',
            }
            continue
          }
          content = buildSkillExecutionCompletionFailureOutput(skillExecutionCompletion)
          responseMessageForHistory.content = content
          terminalResponseFailure = true
        }

        const actionCompletion = evaluateActionCompletion({
          messages,
          content,
          userInput: latestUserText(messages),
          availableToolNames: [...allToolNames],
          commandEvidenceToolNames,
          inferActionsFromWording: !this.semanticRouting,
        })
        if (
          actionCompletion.outcome !== 'not_required'
          && actionCompletion.outcome !== 'success'
          && actionCompletion.outcome !== 'blocked'
        ) {
          const currentTurnStart = messages.findLastIndex((message) =>
            message.role === 'user'
            && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true,
          )
          await logAgentDebugTrace({
            event: 'supervisor.action-completion-missing',
            source: 'react',
            sessionId: context.sessionId,
            runId: context.sessionId,
            iteration: i + 1,
            status: actionCompletion.outcome,
            data: {
              userInput: latestUserText(messages),
              positiveActionScope: inputPositiveActionScope(latestUserText(messages)),
              requiredKinds: actionCompletion.requiredKinds,
              missingKinds: actionCompletion.missingKinds,
              hasAvailableEvidenceTool: actionCompletion.hasAvailableEvidenceTool,
              currentTurnStart,
              currentTurnEvidence: messages
                .slice(currentTurnStart + 1)
                .filter((message) => message.role === 'tool')
                .map((message) => ({
                  name: message.name,
                  toolCallId: message.toolCallId,
                  status: message.metadata?.toolResultStatus,
                })),
              availableEvidenceTools: [...allToolNames].filter((name) =>
                name === 'terminal.run'
                || name.startsWith('process.')
                || name.startsWith('browser.')
                || name.startsWith('fs.'),
              ),
            },
          })
        }
        const missingDirectoryInventory = actionCompletion.missingKinds.includes('file-inventory')

        if (
          !terminalResponseFailure &&
          !forceFinalAfterReadOnlyPolicyFailure &&
          missingDirectoryInventory &&
          actionCompletion.outcome === 'missing' &&
          actionCompletion.hasAvailableEvidenceTool &&
          !isLastIteration &&
          repairedActionCompletionCount < 1
        ) {
          repairedActionCompletionCount += 1
          messages.push(responseMessageForHistory)
          messages.push(buildActionCompletionRecoveryMessage(actionCompletion))
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield {
            type: 'thinking',
            content: `[supervisor] Requested action lacked actual tool evidence (${actionCompletion.missingKinds.join(', ')}); retrying once with the exact evidence tool.`,
          }
          continue
        }

        if (
          !terminalResponseFailure &&
          actionCompletion.outcome !== 'not_required' &&
          actionCompletion.outcome !== 'success' &&
          actionCompletion.outcome !== 'blocked' &&
          (
            !actionCompletion.hasAvailableEvidenceTool ||
            isLastIteration ||
            (
              missingDirectoryInventory &&
              (
                actionCompletion.outcome === 'failed' ||
                repairedActionCompletionCount >= 1
              )
            )
          )
        ) {
          content = buildActionCompletionFailureOutput(
            actionCompletion,
            latestUserText(messages),
          )
          responseMessageForHistory.content = content
          terminalResponseFailure = true
        }

        if (content.trim().length > 0) {
          repairedEmptyFinalReplyCount = 0
        }

        if (
          !forceFinalAfterReadOnlyPolicyFailure
          && (!forceFinalAfterDegenerateOutput || useBoundedFinalContext)
          && (!useBoundedFinalContext || repairedEmptyFinalReplyCount < 1)
          && shouldRepairEmptyFinalReply({
            content,
            repairedCount: repairedEmptyFinalReplyCount,
            isLastIteration: suppressToolCallsForFinalIteration && !useBoundedFinalContext,
          })
        ) {
          repairedEmptyFinalReplyCount += 1
          messages.push(
            useBoundedFinalContext
              ? buildBoundedEmptyFinalRepairMessage()
              : buildEmptyFinalRepairMessage(),
          )
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          continue
        }

        if (
          !terminalResponseFailure
          && !deterministicSkillCompletionSatisfied
          && !useBoundedFinalContext
          && !forceFinalAfterDegenerateOutput
          && !forceFinalAfterReadOnlyPolicyFailure
          && shouldRepairInterimProgressReply({
            content,
            messages,
            repairedCount: repairedInterimProgressCount,
          })
        ) {
          repairedInterimProgressCount += 1
          messages.push(responseMessageForHistory)
          messages.push(buildInterimProgressRepairMessageWithContext(messages))
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          continue
        }

        if (
          !useBoundedFinalContext
          && !forceFinalAfterDegenerateOutput
          && !forceFinalAfterReadOnlyPolicyFailure
          && shouldRepairMissingAnswerProtocolReply({
            content,
            repairedCount: repairedMissingAnswerProtocolCount,
            strict: requireFinalAnswerProtocol,
          })
        ) {
          const fallbackCandidate = stripUnsupportedCitationReferences(stripFinalAnswerStem(content)).trim()
          if (fallbackCandidate) {
            lastMissingAnswerProtocolCandidate = fallbackCandidate
            lastUsableRecoveryCandidate = fallbackCandidate
          }
          repairedMissingAnswerProtocolCount += 1
          messages.push(responseMessageForHistory)
          messages.push(buildMissingAnswerProtocolRepairMessage())
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield {
            type: 'thinking',
            content: '[supervisor] Final answer missed the required ANSWER:/INCOMPLETE: protocol. Retrying.',
          }
          continue
        }

        if (
          !useBoundedFinalContext
          && !forceFinalAfterDegenerateOutput
          && !forceFinalAfterReadOnlyPolicyFailure
          && shouldRepairUnsupportedCitationReply({
            content,
            repairedCount: repairedUnsupportedCitationCount,
            strict: this.strictFinalAnswerProtocol,
          })
        ) {
          const sanitizedCandidate = stripUnsupportedCitationReferences(stripFinalAnswerStem(content)).trim()
          if (sanitizedCandidate) {
            lastUnsupportedCitationCandidate = sanitizedCandidate
            lastUsableRecoveryCandidate = sanitizedCandidate
          }
          repairedUnsupportedCitationCount += 1
          messages.push(responseMessageForHistory)
          messages.push(buildUnsupportedCitationRepairMessage())
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield {
            type: 'thinking',
            content: '[supervisor] Final answer used unsupported approximate line references. Retrying.',
          }
          continue
        }

        let exhaustedOutcomeReviewReason: string | null = null
        if (
          !terminalResponseFailure
          &&
          !deterministicSkillCompletionSatisfied
          && !skipOutcomeReview
          && shouldReviewOutcomeWithLLM(messages, {
            includeToollessFinal: this.reviewToollessFinals,
          })
          && allToolsForRequest.length > 0
          && hasPendingRunOutcomeReviewRecovery(messages)
          && !hasIncompleteAnswerStem(content)
          && repairedOutcomeReviewNoProgressCount < MAX_OUTCOME_REVIEW_REPAIRS
        ) {
          repairedOutcomeReviewNoProgressCount += 1
          messages.push(responseMessageForHistory)
          messages.push(buildRunOutcomeReviewNoProgressMessage())
          this.state = 'observing'
          yield {
            type: 'thinking',
            content: '[supervisor] Outcome review requested more evidence, but the next reply had no new tool work. Retrying with an explicit tool-evidence requirement.',
          }
          yield { type: 'state_change', state: 'observing' }
          continue
        }

        if (
          !terminalResponseFailure
          &&
          !deterministicSkillCompletionSatisfied
          && !skipOutcomeReview
          && shouldReviewOutcomeWithLLM(messages, {
            includeToollessFinal: this.reviewToollessFinals,
          })
          && hasPendingRunOutcomeReviewRecovery(messages)
          && !hasIncompleteAnswerStem(content)
          && repairedOutcomeReviewNoProgressCount >= MAX_OUTCOME_REVIEW_REPAIRS
        ) {
          exhaustedOutcomeReviewReason =
            'Outcome review requested more evidence, but subsequent replies did not perform new tool work before the recovery limit.'
        }

        const outcomeEvaluation = evaluateRunOutcome({
          messages,
          content,
          availableToolNames: allToolsForRequest.map((tool) => tool.name),
          commandEvidenceToolNames,
          inferActionsFromWording: !this.semanticRouting,
        })
        if (
          !terminalResponseFailure
          && !forceFinalAfterReadOnlyPolicyFailure
          && !forceSynthesisFromObservationBudget
          &&
          shouldRepairRunOutcome({
            evaluation: outcomeEvaluation,
            repairedCount: repairedRunOutcomeCount,
            isLastIteration,
          })
        ) {
          const fallbackCandidate = stripUnsupportedCitationReferences(
            stripFinalAnswerStem(content),
          ).trim()
          if (fallbackCandidate) lastUsableRecoveryCandidate = fallbackCandidate
          repairedRunOutcomeCount += 1
          runOutcomeRecoveryReasons.push(...outcomeEvaluation.reasons)
          messages.push(responseMessageForHistory)
          dropStaleOutcomeRecoveryPrompts(messages)
          messages.push(buildRunOutcomeRecoveryMessage(outcomeEvaluation))
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield {
            type: 'thinking',
            content: `[supervisor] Final answer looked incomplete (${outcomeEvaluation.reasons.join('; ')}). Retrying with a different approach.`,
          }
          continue
        }

        if (
          !terminalResponseFailure
          && !deterministicSkillCompletionSatisfied
          && !skipOutcomeReview
          && shouldReviewOutcomeWithLLM(messages, {
            includeToollessFinal: this.reviewToollessFinals,
          })
        ) {
          const reviewRequest = buildRunOutcomeReviewRequest({
            model: context.model,
            messages,
            assistantAnswer: content,
            userInstructions: activeUserInstructions(this.liveRunState?.steeringNotes),
            availableToolNames: allToolsForRequest.map((tool) => tool.name),
            runContract: context.runContract,
            maxTokens: OUTCOME_REVIEW_MAX_TOKENS,
          })
          const reviewTurnId = buildLlmTurnId(context.sessionId, i + 1, 'outcome-review')
          const requiresMutationReview = unavailableOutcomeReviewReason(messages) !== null
          if (requiresMutationReview && !mutationReviewBudget) {
            mutationReviewBudget = new AuxiliaryLlmTurnBudget(DEFAULT_AUXILIARY_LLM_TURN_BUDGET_MS)
          }
          const reviewBudget = requiresMutationReview ? mutationReviewBudget : this.auxiliaryLlmBudget
          try {
            yield buildLlmRequestEvent({
              sessionId: context.sessionId,
              iteration: i + 1,
              source: 'outcome-review',
              request: reviewRequest,
              turnId: reviewTurnId,
              providerId: this.provider.id,
              timeoutMs: reviewBudget?.remainingMs(),
              auxiliary: true,
            })
            const reviewResponse = await runAuxiliaryLlmChat({
              provider: this.provider,
              request: reviewRequest,
              label: 'outcome review',
              breaker: this.providerCircuitBreaker,
              signal,
              budget: reviewBudget,
            })
            this.recordLlmUsage(context, reviewResponse.usage)
            totalUsage.inputTokens += reviewResponse.usage.inputTokens
            totalUsage.outputTokens += reviewResponse.usage.outputTokens
            await logLlmCallTrace({
              source: 'react',
              mode: 'react',
              sessionId: context.sessionId,
              provider: context.provider,
              model: context.model,
              iteration: i + 1,
              request: reviewRequest,
              response: reviewResponse,
              meta: { node: 'outcome-review', turnId: reviewTurnId },
            })
            await this.safeTriggerPostHook('post:llm:call', {
              sessionId: context.sessionId,
              provider: context.provider,
              model: context.model,
              iteration: i + 1,
              request: reviewRequest,
              response: reviewResponse,
              node: 'outcome-review',
            })
            const reviewText = this.responseTextContent(reviewResponse)
            const parsedReview = parseRunOutcomeReviewTransport(
              reviewText,
              reviewResponse.thinking,
            )
              ?? buildEmptyRunOutcomeReviewRecovery(reviewText)
            const review = enforceRunOutcomeReviewEvidenceFloor({
              review: parsedReview,
              messages,
              runContract: context.runContract,
              assistantAnswer: content,
            })
            if (review?.status === 'needs_recovery') {
              outcomeReviewRecoveryReasons.push(review.reason)
            }
            const currentToolResultCount = messages.filter(
              (message) => message.role === 'tool',
            ).length
            if (currentToolResultCount > outcomeReviewSynthesisEvidenceCount) {
              repairedOutcomeReviewSynthesisCount = 0
              outcomeReviewSynthesisEvidenceCount = currentToolResultCount
            }
            const synthesisRecovery = review?.recoveryMode === 'synthesis'
            const shouldRepairReview = shouldRepairRunOutcomeReview({
              review,
              repairedCount: synthesisRecovery
                ? repairedOutcomeReviewSynthesisCount
                : repairedOutcomeReviewCount,
              ...(synthesisRecovery
                ? { maxRepairs: MAX_OUTCOME_REVIEW_SYNTHESIS_REPAIRS }
                : {}),
            })
            if (shouldRepairReview) {
              const fallbackCandidate = stripUnsupportedCitationReferences(
                stripFinalAnswerStem(content),
              ).trim()
              if (fallbackCandidate) lastUsableRecoveryCandidate = fallbackCandidate
              repairedOutcomeReviewCount += 1
              if (synthesisRecovery) repairedOutcomeReviewSynthesisCount += 1
              messages.push(responseMessageForHistory)
              dropStaleRunOutcomeReviewRecovery(messages)
              if (forceSynthesisFromObservationBudget) {
                // A read/observation guard already decided that more evidence
                // gathering is counterproductive and removed all tools. Do not
                // install the normal recovery prompt, which requires new tool
                // evidence and creates an impossible supervisor loop. Give the
                // reviewer one bounded, tool-free synthesis retry instead.
                messages.push({
                  role: 'system',
                  content: [
                    '[Bounded final synthesis after outcome review]',
                    `The reviewer found this gap: ${review!.reason}`,
                    review!.instruction ? `Address it this way: ${review!.instruction}` : '',
                    'The evidence-gathering budget is closed and no tools are available in this final retry.',
                    'Use the tool evidence already present in the conversation to provide the substantive final answer now.',
                    'Do not promise another read, describe future work, or return a progress-only response.',
                  ].filter(Boolean).join('\n'),
                })
                forceFinalAfterDegenerateOutput = true
                await logAgentDebugTrace({
                  event: 'supervisor.outcome-review',
                  source: 'react',
                  sessionId: context.sessionId,
                  runId: context.sessionId,
                  iteration: i + 1,
                  status: 'bounded-final-synthesis',
                  data: {
                    reason: review!.reason,
                    instruction: review!.instruction,
                    repairedOutcomeReviewCount,
                    forceFinalAfterRepeatedRead,
                    forceFinalAfterObservationBudget,
                  },
                })
                this.state = 'observing'
                yield {
                  type: 'thinking',
                  content: '[supervisor] Outcome review found a gap after the evidence budget closed; reserving one tool-free final synthesis retry.',
                }
                yield { type: 'state_change', state: 'observing' }
                continue
              }
              messages.push(buildRunOutcomeReviewRecoveryMessage(review!))
              const suggestedToolCallPlan = buildRunOutcomeReviewSuggestedToolCallPlan(
                review!,
                new Set(allToolsForRequest.map((tool) => tool.name)),
                { idPrefix: `outcome-review-${i + 1}`, messages },
              )
              const suggestedToolCalls = suggestedToolCallPlan.toolCalls
              if (
                suggestedToolCalls.length === 0
                && suggestedToolCallPlan.skippedDuplicateCount > 0
              ) {
                messages.push(buildRunOutcomeReviewDuplicateSuggestedToolCallsMessage(review!))
              }
              if (suggestedToolCalls.length > 0) {
                messages.push({
                  role: 'assistant',
                  content: '',
                  toolCalls: suggestedToolCalls,
                })
                this.state = 'acting'
                yield {
                  type: 'thinking',
                  content: `[supervisor] LLM outcome review requested evidence; running ${suggestedToolCalls.length} suggested read-only tool call(s).`,
                }
                yield { type: 'state_change', state: 'acting' }
                const requiredUserAction = yield* this.runToolCalls(
                  messages,
                  suggestedToolCalls,
                  context,
                  i,
                  totalUsage,
                  undefined,
                  signal,
                  canAttachVisualContent,
                )
                if (requiredUserAction) {
                  yield* this.finishAfterUserActionRequired(requiredUserAction, context, totalUsage)
                  return
                }
                const approvalDenial = resolveApprovalOutcomeAfterTools()
                if (approvalDenial) {
                  yield* this.finishAfterApprovalDenial(
                    context,
                    totalUsage,
                    i + 1,
                    approvalDenial,
                    latestUserText(messages),
                  )
                  return
                }
                const reviewPolicyFailure = this.autonomy === AutonomyLevel.ReadOnly
                  ? latestTrustedPolicyFailure(messages, ignoredReadOnlyPolicyFailureToolCallIds)
                  : null
                if (reviewPolicyFailure) {
                  const resolution = resolveReadOnlyPolicyFailure(reviewPolicyFailure)
                  if (resolution !== 'terminate') {
                    this.state = 'observing'
                    yield {
                      type: 'thinking',
                      content: resolution === 'repair'
                        ? '[supervisor] Read-only policy blocked a review shell wrapper; allowing one policy-checked direct invocation repair.'
                        : '[supervisor] Read-only policy blocked a review follow-up; synthesizing once from successful current-turn evidence.',
                    }
                    yield { type: 'state_change', state: 'observing' }
                    continue
                  }
                  yield* this.finishAfterReadOnlyPolicyFailure(
                    context,
                    totalUsage,
                    i + 1,
                    reviewPolicyFailure,
                    latestUserText(messages),
                  )
                  return
                }
                this.state = 'observing'
                yield { type: 'state_change', state: 'observing' }
                continue
              }
              this.state = 'observing'
              yield {
                type: 'thinking',
                content: `[supervisor] LLM outcome review requested another step: ${review!.reason}`,
              }
              yield { type: 'state_change', state: 'observing' }
              continue
            }
            if (review?.status === 'needs_recovery') {
              exhaustedOutcomeReviewReason = review.reason
              if (
                synthesisRecovery
                && repairedOutcomeReviewSynthesisCount >= MAX_OUTCOME_REVIEW_SYNTHESIS_REPAIRS
              ) {
                yield {
                  type: 'recovery',
                  scope: 'output_synthesis',
                  kind: 'unchanged_evidence_synthesis_exhausted',
                  action: 'stop_repeated_synthesis',
                  message: 'Outcome review still rejected the answer after the bounded unchanged-evidence synthesis passes; ending honestly instead of repeating the same evidence loop.',
                  recoverable: false,
                  details: {
                    synthesisRepairs: repairedOutcomeReviewSynthesisCount,
                    toolResults: messages.filter((message) => message.role === 'tool').length,
                  },
                }
              }
            } else if (review?.status === 'complete') {
              deferredCandidateAccepted = true
            }
          } catch (error) {
            await logLlmCallTrace({
              source: 'react',
              mode: 'react',
              sessionId: context.sessionId,
              provider: context.provider,
              model: context.model,
              iteration: i + 1,
              request: reviewRequest,
              error: error instanceof Error ? error.message : String(error),
              meta: { node: 'outcome-review', turnId: reviewTurnId, failed: true },
            })
            if (isAbortError(error) || signal?.aborted) throw error
            exhaustedOutcomeReviewReason = unavailableOutcomeReviewReason(messages)
          }
        }

        // The deterministic evidence floor also applies when the auxiliary
        // judge is unavailable or its budget expired. Never report a missing
        // required artifact as completed merely because review could not run.
        const terminalEvidenceFloor = context.runContract?.requiredArtifacts?.length
          ? enforceRunOutcomeReviewEvidenceFloor({
              review: { status: 'complete', reason: 'Candidate reached the terminal acceptance boundary.' },
              messages,
              runContract: context.runContract,
              assistantAnswer: content,
            })
          : null
        if (terminalEvidenceFloor?.status === 'needs_recovery') {
          exhaustedOutcomeReviewReason = terminalEvidenceFloor.reason
        }
        let finalContent = forceFinalAfterNoRetryActionFailure
          ? 'A required action failed under the no-retry contract, so the workflow is incomplete and later required actions were not executed.'
          : stripFinalAnswerStem(content)
        if (exhaustedOutcomeReviewReason) {
          finalContent = buildOutcomeReviewExhaustedMessage(exhaustedOutcomeReviewReason)
        } else if (!finalContent) {
          const exhaustedOutcomeReason = outcomeReviewRecoveryReasons.at(-1)
          finalContent = exhaustedOutcomeReason
            ? buildOutcomeReviewExhaustedMessage(exhaustedOutcomeReason)
            : lastUnsupportedCitationCandidate
            ?? lastMissingAnswerProtocolCandidate
            ?? buildEmptyFinalFallbackMessage(messages)
        } else if (content.trim() === UNUSABLE_PROMPT_TOOL_CALL_OUTPUT) {
          finalContent = buildInvalidFinalResponseMessage(messages)
        } else if (
          requireFinalAnswerProtocol
          && content.trim().length > 0
          && !hasAnyAnswerProtocolStem(content)
        ) {
          finalContent = buildInterimProgressFallbackMessage(`INCOMPLETE: ${content}`, messages)
        } else if (!terminalResponseFailure && isLikelyInterimProgressUpdate(finalContent)) {
          finalContent = buildInterimProgressFallbackMessage(finalContent, messages)
        }
        finalContent = stripInternalPlannerBlocks(stripUnsupportedCitationReferences(finalContent))

        yield* this.journalReactCriterionEvidence(
          messages,
          finalContent,
          context,
          totalUsage,
          i + 1,
          signal,
        )

        // A follow-up can also arrive while the completion reviewer is running.
        if (this.aborted || signal?.aborted) break
        if (this.hasPendingSteering()) continue

        let releasedDeferredText = false
        if (
          deferredTextDeltas.length > 0
          && deferredCandidateAccepted
          && deferredTextDeltas.join('') === finalContent
        ) {
          for (const textDelta of deferredTextDeltas) {
            yield { type: 'text_delta', text: textDelta }
          }
          releasedDeferredText = true
        }
        if (
          !releasedDeferredText
          && suppressTextDeltas
          && (this.textDeltaMode !== 'live' || hadLengthContinuation)
          && finalContent
        ) {
          yield { type: 'text_delta', text: finalContent }
        }
        yield { type: 'message', content: finalContent }
        this.state = 'done'
        yield { type: 'state_change', state: 'done' }
        const finalStopReason = approvalGraceState.denial && isIncompleteOutput(finalContent)
          ? stopReasonApprovalDenied(approvalGraceState.denial.toolName ?? 'tool')
          : forceFinalAfterObservationBudget
          ? stopReasonObservationBudget({
              incomplete: isIncompleteOutput(finalContent),
              contract: context.runContract,
            })
          : exhaustedOutcomeReviewReason || isIncompleteOutput(finalContent)
            || hasIncompleteAnswerStem(content) || forceFinalAfterNoRetryActionFailure
            ? stopReasonCompletionGate({})
            : stopReasonCompleted()
        yield {
          type: 'done',
          usage: totalUsage,
          stopReason: finalStopReason,
        }
        await this.safeTriggerPostHook('post:agent:run', {
          status: finalStopReason.kind === 'completed' ? 'success' : 'incomplete',
          sessionId: context.sessionId,
          provider: context.provider,
          model: context.model,
          usage: { ...totalUsage },
          output: finalContent,
          iteration: i + 1,
        })
        await logAgentRunTrace({
          source: 'react',
          status: finalStopReason.kind === 'completed' ? 'success' : 'incomplete',
          mode: 'react',
          sessionId: context.sessionId,
          provider: context.provider,
          model: context.model,
          iteration: i + 1,
          usage: { ...totalUsage },
          output: finalContent,
        })
        await this.clearRunCheckpoint(context.sessionId)
        return
      } catch (err: unknown) {
        // A parked approval (timeout without an answer) must unwind to the
        // caller so the run lease is released and the checkpoint is kept for
        // /approvals/resume; it is neither a provider failure nor a user abort.
        if (isApprovalParkedSignal(err)) throw err
        if ((signal?.aborted ?? false) || (this.aborted && isAbortError(err))) {
          this.state = 'done'
          yield { type: 'done', usage: totalUsage, stopReason: stopReasonUserAbort() }
          await logAgentRunTrace({
            source: 'react',
            status: 'aborted',
            mode: 'react',
            sessionId: context.sessionId,
            provider: context.provider,
            model: context.model,
            iteration: i + 1,
            usage: { ...totalUsage },
          })
          return
        }
        const observedMaxOutputTokens = maxOutputTokenRecoveries < 1
          ? detectProviderMaxOutputTokenLimit(err, request.maxTokens ?? requestedOutputTokens)
          : null
        if (observedMaxOutputTokens) {
          maxOutputTokenRecoveries += 1
          effectiveMaxOutputTokens = observedMaxOutputTokens
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield buildMaxOutputTokenRecoveryEvent(
            request.maxTokens ?? requestedOutputTokens,
            observedMaxOutputTokens,
          )
          yield {
            type: 'thinking',
            content: `[provider recovery] Provider catalog overstated the model output limit; retrying once with max_tokens=${observedMaxOutputTokens}.`,
          }
          i -= 1
          continue
        }
        if (
          contextRecoveries < 2
          && isContextLengthProviderError(err)
          && !(signal?.aborted ?? false)
          && !this.aborted
        ) {
          contextRecoveries += 1
          let recovered: Awaited<ReturnType<typeof emergencyContextRecovery>>
          try {
            recovered = await emergencyContextRecovery(
              messages,
              contextWindow,
              this.provider,
              {
                model: context.model,
                signal,
                // Carry the rolling summary in and back out. Without it a
                // second recovery re-summarizes the whole run from scratch,
                // paying for a longer call and losing the first pass's
                // wording (the graph path already threads it through).
                previousSummary: compressedHistorySummary,
                previousUpToIndex: compressedHistoryUpToIndex,
                charsPerToken: tokenCalibration.charsPerToken(this.provider.id, context.model),
                auxiliaryLlmBudget: this.auxiliaryLlmBudget,
              },
            )
          } catch (recoveryError) {
            if (!(recoveryError instanceof IrreducibleContextOverflowError)) {
              throw recoveryError
            }
            this.state = 'error'
            const message = recoveryError.message
            yield {
              type: 'error',
              error: {
                code: 'CONTEXT_LENGTH',
                message,
              },
            }
            await this.finalizeReactContextLengthFailure(
              context,
              totalUsage,
              i + 1,
              message,
            )
            return
          }
          effectiveContextWindow = recovered.effectiveContextWindowTokens
          messages.splice(0, messages.length, ...recovered.messages)
          if (recovered.summary) {
            compressedHistorySummary = recovered.summary
            compressedHistoryUpToIndex = recovered.upToIndex
          }
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield {
            type: 'thinking',
            content: '[provider recovery] Provider rejected the prompt as too large; compacted context and retrying.',
          }
          i -= 1
          continue
        }
        if (
          imageInputRecoveries < 1
          && isImageInputUnsupportedProviderError(err)
          && messagesContainImageParts(messages)
          && !(signal?.aborted ?? false)
          && !this.aborted
        ) {
          imageInputRecoveries += 1
          disableVisualAttachmentsForProviderRecovery = true
          markProviderModelImageInputRejected(
            context.provider,
            context.model,
            errorMessageText(err),
          )
          const removedImageParts = stripImagePartsFromMessages(messages)
          messages.push(buildImageInputRecoveryMessage(
            context.provider,
            context.model,
            removedImageParts,
          ))
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield {
            type: 'thinking',
            content:
              '[provider recovery] Provider rejected image input; retrying without retained image parts and disabling new visual attachments for this run.',
          }
          await this.safeTriggerPostHook('post:llm:call', {
            sessionId: context.sessionId,
            provider: context.provider,
            model: context.model,
            iteration: i + 1,
            request,
            error: errorMessageText(err),
            failed: true,
            recovered: true,
            recoveryKind: 'image_input_unsupported',
            removedImageParts,
          })
          i -= 1
          continue
        }
        const nativeToolTransportRejection =
          useNativeToolUse
          && (request.tools?.length ?? 0) > 0
          && modelInfo?.capabilities.adaptivePromptReact === true
          && toolTransportRecoveryCount < 1
            ? detectNativeToolTransportRejection(err)
            : null
        if (nativeToolTransportRejection) {
          toolTransportRecoveryCount += 1
          preferPromptReact = true
          messages.push(buildToolTransportRecoveryMessage(nativeToolTransportRejection))
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield buildToolTransportRecoveryEvent(nativeToolTransportRejection)
          yield {
            type: 'thinking',
            content: '[provider recovery] Provider rejected native function calling; retrying once with prompt tool transport.',
          }
          await this.safeTriggerPostHook('post:llm:call', {
            sessionId: context.sessionId,
            provider: context.provider,
            model: context.model,
            iteration: i + 1,
            request,
            error: nativeToolTransportRejection.reason,
            failed: true,
            recovered: true,
            recoveryKind: 'native_tool_transport',
          })
          i -= 1
          continue
        }
        const providerMessageRecovery = detectProviderMessageRecovery(err)
        if (
          providerMessageRecovery &&
          shouldRecoverProviderMessageError({
            recovery: providerMessageRecovery,
            repairedCount: repairedProviderMessageCount,
            signal,
            aborted: this.aborted,
          })
        ) {
          repairedProviderMessageCount += 1
          if (providerMessageRecovery.disableThinking) {
            disableThinkingForProviderRecovery = true
          }
          messages.push(buildProviderMessageRecoveryMessage(providerMessageRecovery))
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield buildProviderMessageRecoveryEvent(providerMessageRecovery)
          yield {
            type: 'thinking',
            content: `[provider recovery] Provider rejected the message protocol (${providerMessageRecovery.kind}). Retrying with sanitized provider-facing messages.`,
          }
          await this.safeTriggerPostHook('post:llm:call', {
            sessionId: context.sessionId,
            provider: context.provider,
            model: context.model,
            iteration: i + 1,
            request,
            error: providerMessageRecovery.reason,
            failed: true,
            recovered: true,
            recoveryKind: providerMessageRecovery.kind,
          })
          continue
        }
        this.state = 'error'
        const mappedError = toProviderApiError(err)
        const message = mappedError.message
        await logLlmCallTrace({
          source: 'react',
          mode: 'react',
          sessionId: context.sessionId,
          provider: context.provider,
          model: context.model,
          iteration: i + 1,
          request,
          error: message,
          meta: { turnId: llmTurnId, failed: true },
        })
        await this.safeTriggerPostHook('post:llm:call', {
          sessionId: context.sessionId,
          provider: context.provider,
          model: context.model,
          iteration: i + 1,
          request,
          error: message,
          failed: true,
        })
        yield { type: 'error', error: mappedError }
        await this.safeTriggerPostHook('post:agent:run', {
          status: 'error',
          sessionId: context.sessionId,
          provider: context.provider,
          model: context.model,
          usage: { ...totalUsage },
          error: message,
          iteration: i + 1,
        })
        await logAgentRunTrace({
          source: 'react',
          status: 'error',
          mode: 'react',
          sessionId: context.sessionId,
          provider: context.provider,
          model: context.model,
          iteration: i + 1,
          usage: { ...totalUsage },
          error: message,
        })
        await this.clearRunCheckpoint(context.sessionId)
        return
      }
    }

    if (this.aborted || (signal?.aborted ?? false)) {
      this.state = 'done'
      yield { type: 'done', usage: totalUsage, stopReason: stopReasonUserAbort() }
      await logAgentRunTrace({
        source: 'react',
        status: 'aborted',
        mode: 'react',
        sessionId: context.sessionId,
        provider: context.provider,
        model: context.model,
        usage: { ...totalUsage },
      })
      return
    }

    const finalIterationBudget = effectiveMaxIterations()
    // Continuation is progress-driven, not count-driven: a cycle that produced
    // new tool evidence, artifact mutations or a verified partial may continue
    // while cycles remain; a cycle that produced nothing stops now even if
    // cycles remain. A zero-iteration budget never ran a cycle to judge.
    const progressSinceCycleStart = finalIterationBudget > 0
      ? evaluateProgressSince(messages, findProgressMarkerIndex(messages))
      : undefined
    if (
      progressSinceCycleStart
      && !progressSinceCycleStart.progressed
      && this.maxContinuationCycles > 0
    ) {
      this.state = 'done'
      await this.persistRunCheckpoint(context, messages, totalUsage, 0, 'observing')
      const noProgressContent = buildBudgetExhaustedMessage({
        mode: 'react',
        layer: 'continuation',
        iterationBudget: finalIterationBudget,
        contract: context.runContract,
      })
      if (this.textDeltaMode !== 'live') {
        yield { type: 'text_delta', text: noProgressContent }
      }
      yield { type: 'message', content: noProgressContent }
      yield {
        type: 'done',
        usage: totalUsage,
        stopReason: stopReasonNoProgress({
          layer: 'continuation',
          budget: this.maxContinuationCycles,
          used: continuationCycle,
          contract: context.runContract,
        }),
      }
      await this.safeTriggerPostHook('post:agent:run', {
        status: 'incomplete',
        sessionId: context.sessionId,
        provider: context.provider,
        model: context.model,
        usage: { ...totalUsage },
        iteration: finalIterationBudget,
        output: noProgressContent,
      })
      await logAgentRunTrace({
        source: 'react',
        status: 'incomplete',
        mode: 'react',
        sessionId: context.sessionId,
        provider: context.provider,
        model: context.model,
        iteration: finalIterationBudget,
        usage: { ...totalUsage },
      })
      return
    }
    if (continuationCycle < this.maxContinuationCycles && finalIterationBudget > 0) {
      const nextCycle = continuationCycle + 1
      await this.persistRunCheckpoint(
        context,
        messages,
        totalUsage,
        0,
        'observing',
      )
      messages.push({
        role: 'system',
        metadata: { [CONTINUATION_MARKER_METADATA_KEY]: nextCycle },
        content: buildContinuationPrompt({
          mode: 'react',
          cycle: nextCycle,
          maxCycles: this.maxContinuationCycles,
          contract: context.runContract,
        }),
      })
      this.state = 'observing'
      yield { type: 'state_change', state: 'observing' }
      yield {
        type: 'thinking',
        content: `[continuation] Iteration budget reached; continuing automatically (${nextCycle}/${this.maxContinuationCycles}).`,
      }
      yield* this.continueRun(
        messages,
        context,
        totalUsage,
        0,
        undefined,
        signal,
        nextCycle,
        observationHistory,
      )
      return
    }

    this.state = 'done'
    await this.persistRunCheckpoint(
      context,
      messages,
      totalUsage,
      0,
      'observing',
    )
    const incompleteContent = buildBudgetExhaustedMessage({
      mode: 'react',
      layer: 'iteration',
      iterationBudget: finalIterationBudget,
      contract: context.runContract,
    })
    if (this.textDeltaMode !== 'live') {
      yield { type: 'text_delta', text: incompleteContent }
    }
    yield { type: 'message', content: incompleteContent }
    yield {
      type: 'done',
      usage: totalUsage,
      stopReason: stopReasonBudget(
        'iteration',
        finalIterationBudget,
        finalIterationBudget,
        context.runContract,
      ),
    }
    await this.safeTriggerPostHook('post:agent:run', {
      status: 'incomplete',
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      usage: { ...totalUsage },
      iteration: finalIterationBudget,
      output: incompleteContent,
    })
    await logAgentRunTrace({
      source: 'react',
      status: 'incomplete',
      mode: 'react',
      sessionId: context.sessionId,
      provider: context.provider,
      model: context.model,
      iteration: finalIterationBudget,
      usage: { ...totalUsage },
      output: incompleteContent,
    })
  }

  async *run(input: string, context: AgentContext): AsyncIterable<AgentEvent> {
    this.aborted = false
    const abortController = new AbortController()
    this.runAbortController = abortController
    this.runWallClock = {
      startedAt: Date.now(),
      budgetMs: parseRunWallClockBudgetMs(process.env.SEPILOTD_RUN_MAX_WALL_MS),
    }
    try {
      if (this.hookRegistry && this.emitAgentRunHooks) {
        const hookResult = await this.hookRegistry.trigger({
          event: 'pre:agent:run',
          data: { input, context },
        }, abortController.signal)
        if (hookResult.action === 'abort' && !abortController.signal.aborted) {
          yield { type: 'error', error: { code: 'FORBIDDEN', message: 'Agent run blocked by hook' } }
          return
        }
      }

      if (abortController.signal.aborted) {
        this.state = 'done'
        yield { type: 'done', usage: { inputTokens: 0, outputTokens: 0 }, stopReason: stopReasonUserAbort() }
        return
      }
      this.state = 'thinking'
      yield { type: 'state_change', state: 'thinking' }

      const messages = context.executionHandoff
        ? [
            ...this.buildInitialMessages(input, { ...context, previousMessages: [] }).filter((message) => message.role === 'system' && !context.executionHandoff!.messages.some((retained) => !retained.metadata?.runtimeContext && retained.role === 'system' && JSON.stringify(retained.content) === JSON.stringify(message.content))),
            ...context.executionHandoff.messages.filter((message) => !message.metadata?.runtimeContext).map(cloneMessage),
          ]
        : this.buildInitialMessages(input, context)
      const totalUsage: TokenUsage = { ...(context.executionHandoff?.usage ?? { inputTokens: 0, outputTokens: 0 }) }
      const focusedProcessObservation = !this.semanticRouting && !context.executionHandoff && extractFocusedProcessObservationCall(input)
      const focusedObservationToolAvailable = focusedProcessObservation
        ? this.tools.toToolDefinitions().some((tool) => tool.name === focusedProcessObservation.toolName)
        : false
      const initialPendingToolExecution: PendingToolExecution | undefined =
        focusedProcessObservation && focusedObservationToolAvailable
          ? {
              toolCalls: [{
                id: `focused-process-observation-${randomUUID()}`,
                name: focusedProcessObservation.toolName,
                arguments: focusedProcessObservation.arguments,
              }],
              startIndex: 0,
            }
          : undefined
      if (initialPendingToolExecution) {
        messages.push({
          role: 'assistant',
          content: '',
          toolCalls: initialPendingToolExecution.toolCalls.map(cloneToolCall),
        })
      }
      if (context.runContract) {
        yield { type: 'run_contract', contract: context.runContract }
      }
      yield* this.runWithLiveState(
        messages,
        context,
        totalUsage,
        context.executionHandoff?.iterations ?? 0,
        initialPendingToolExecution,
        abortController.signal,
      )
    } finally {
      if (this.runAbortController === abortController) {
        this.runAbortController = null
      }
    }
  }

  async *resumeFromCheckpoint(
    checkpoint: ApprovalRunCheckpoint,
    approved: boolean | import('@sepilotd/core').ApprovalDecision,
  ): AsyncIterable<AgentEvent> {
    this.aborted = false
    const abortController = new AbortController()
    this.runAbortController = abortController
    this.runWallClock = {
      startedAt: Date.now(),
      budgetMs: parseRunWallClockBudgetMs(process.env.SEPILOTD_RUN_MAX_WALL_MS),
    }
    try {
      yield* this.runWithLiveState(
        checkpoint.messages.map(cloneMessage),
        {
          sessionId: checkpoint.sessionId,
          provider: checkpoint.provider,
          model: checkpoint.model,
          cwd: checkpoint.cwd,
          workspaceRoot: checkpoint.workspaceRoot,
          workspaceIsolation: checkpoint.workspaceIsolation,
          scopeTags: cloneCheckpointScopeTags(checkpoint.scopeTags),
          executionSkillIds: cloneCheckpointExecutionSkillIds(checkpoint.executionSkillIds),
          skillToolNames: cloneCheckpointSkillToolNames(checkpoint.skillToolNames),
          toolAllowlist: cloneCheckpointToolAllowlist(checkpoint.toolAllowlist),
          skillExecutionPolicies: cloneCheckpointSkillExecutionPolicies(
            checkpoint.skillExecutionPolicies,
          ),
          requireToolApproval: checkpoint.requireToolApproval,
          runContract: checkpoint.runContract,
        },
        { ...checkpoint.totalUsage },
        checkpoint.iteration,
        {
          toolCalls: checkpoint.toolCalls.map(cloneToolCall),
          startIndex: checkpoint.currentToolIndex,
          batchSize: 1,
          currentExecutionId: undefined,
          initialApprovalDecision: approved,
          skipToolCallEventForStart: true,
        },
        abortController.signal,
      )
    } finally {
      if (this.runAbortController === abortController) {
        this.runAbortController = null
      }
    }
  }

  async *resumeFromRunCheckpoint(checkpoint: SessionRunCheckpoint): AsyncIterable<AgentEvent> {
    this.aborted = false
    const abortController = new AbortController()
    this.runAbortController = abortController
    this.runWallClock = {
      startedAt: Date.now(),
      budgetMs: parseRunWallClockBudgetMs(process.env.SEPILOTD_RUN_MAX_WALL_MS),
    }
    const resumedState: AgentState = checkpoint.pendingToolExecution?.toolCalls.length
      ? 'acting'
      : 'thinking'
    this.state = resumedState
    yield { type: 'state_change', state: resumedState }
    try {
      yield* this.runWithLiveState(
        checkpoint.messages.map(cloneMessage),
        {
          sessionId: checkpoint.sessionId,
          provider: checkpoint.provider,
          model: checkpoint.model,
          cwd: checkpoint.cwd,
          workspaceRoot: checkpoint.workspaceRoot,
          workspaceIsolation: checkpoint.workspaceIsolation,
          scopeTags: cloneCheckpointScopeTags(checkpoint.scopeTags),
          executionSkillIds: cloneCheckpointExecutionSkillIds(checkpoint.executionSkillIds),
          skillToolNames: cloneCheckpointSkillToolNames(checkpoint.skillToolNames),
          toolAllowlist: cloneCheckpointToolAllowlist(checkpoint.toolAllowlist),
          skillExecutionPolicies: cloneCheckpointSkillExecutionPolicies(
            checkpoint.skillExecutionPolicies,
          ),
          requireToolApproval: checkpoint.requireToolApproval,
          runContract: checkpoint.runContract,
        },
        { ...checkpoint.totalUsage },
        checkpoint.iteration,
        checkpoint.pendingToolExecution
          ? {
              toolCalls: checkpoint.pendingToolExecution.toolCalls.map(cloneToolCall),
              startIndex: checkpoint.pendingToolExecution.startIndex,
              batchSize: checkpoint.pendingToolExecution.batchSize,
              currentExecutionId: checkpoint.pendingToolExecution.currentExecutionId,
            }
          : undefined,
        abortController.signal,
      )
    } finally {
      if (this.runAbortController === abortController) {
        this.runAbortController = null
      }
    }
  }

  async stop(): Promise<void> {
    this.aborted = true
    this.runAbortController?.abort(createAbortError('Agent execution stopped'))
  }

  getState(): AgentState {
    return this.state
  }
}
