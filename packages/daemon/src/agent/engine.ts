import { repeatedBrowserClickWithoutProgress } from './browser-progress.js'
import { terminalRunHasReliableSuccessStatus } from './graph/evidence-ledger.js'
import { RUN_STABLE_SYSTEM_METADATA_KEY } from './prompt-sections.js'
import { advanceAutomaticObservationBudget, canAutomaticallyDisableThinking, consecutiveExecutionFailures, resolveThinkingLevel, shouldReduceAutomaticThinking, structuredToolFailureCode as structuredToolFailureCodeFromOutput, type ThinkingLevelSource } from '../providers/thinking-policy.js'
import { userActionRequiredOutput } from './user-action-required-output.js'
import { workspaceExecutionBlocker } from './workspace-execution.js'
import { MODE_TRANSFER_TOOL, type ModeControl } from './mode-control.js'
import { createHash, randomUUID } from 'node:crypto'
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
import { AutonomyLevel } from '@sepilotd/core'
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
import {
  buildTruncatedToolCallRecoveryEvent,
  lengthRecoveryInstruction,
  recordTruncatedToolCallAttempt,
} from './length-recovery.js'
import type { AgentFailedAttempt } from './graph/types.js'
import {
  cloneCheckpointExecutionSkillIds,
  cloneCheckpointScopeTags,
  cloneCheckpointSkillExecutionPolicies,
  cloneCheckpointSkillToolNames,
  cloneCheckpointMemoryPolicy,
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
  fitControlRequest,
  IrreducibleContextOverflowError,
  semanticCompress,
  supersedeStaleReadResults,
  supersedeStaleSystemReminders,
  type ProviderContextFitResult,
} from './context-manager.js'
import { summarizeProblemTools, type ToolStatsRecord } from './tool-learning/store.js'
import {
  buildStuckToolRepeatMessage,
  DEFAULT_EXECUTION_FAILURE_CHURN_THRESHOLD,
  DEFAULT_PERMANENT_FAILURE_CHURN_THRESHOLD,
  DEFAULT_STUCK_REPEAT_THRESHOLD,
  DEFAULT_STUCK_REPEAT_WINDOW,
  DEFAULT_TARGET_REPEAT_THRESHOLD,
  detectStuckToolRepeat,
  findRepeatedInvalidToolCall,
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
  advanceValidationConvergence,
  initialValidationConvergenceState,
  type ValidationConvergenceEntry,
} from './validation-convergence.js'
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
  parseTruncatedPromptToolCallNames,
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
  resolveRepairThinkingLevel,
} from './model-compatibility.js'
import {
  buildMaxOutputTokenRecoveryEvent,
  detectProviderMaxOutputTokenLimit,
  recallObservedMaxOutputTokens,
  rememberObservedMaxOutputTokens,
  resolveModelOutputAllowance,
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
import { cloneMessage, cloneToolCall, runToolExecution, toolRejectedByExecutionIntentForEveryInput } from './tool-execution.js'
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
  TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY,
  type TrustedPolicyFailure,
} from './policy-failure.js'
import {
  isIncompleteOutput,
  stopReasonApprovalDenied,
  stopReasonBudget,
  stopReasonCompleted,
  stopReasonCompletionGate,
  incompleteOutputReason,
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
  withSystemToolGuidance,
  TOOL_APPROVAL_POSTURE_PROMPT_PREFIX,
} from './system-prompt.js'
import {
  CONTINUATION_MARKER_METADATA_KEY,
  evaluateProgressSince,
  findProgressMarkerIndex,
  resolveRunWallClock,
  runRequestTimeoutMs,
  resolveWallClockSynthesisReserveMs,
} from './progress-signal.js'
import { observeWorkProgress, type WorkProgress } from './work-progress.js'
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
import { runAuxiliaryLlmChat, AuxiliaryLlmTurnBudget } from './auxiliary-llm.js'
import { resolveControlCallTimeoutMs } from './control-call-policy.js'
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

export function withProblemToolFeedback(
  messages: Message[],
  stats: ToolStatsRecord[],
  iteration: number,
): Message[] {
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

function appendWallClockSynthesisMessage(messages: Message[]): Message[] {
  return [
    ...messages,
    {
      role: 'system',
      content: [
        'The run is about to reach its wall-clock limit, so tools are closed for this final turn.',
        'Write the final answer now from the evidence already gathered: report every finding you observed,',
        'mark items you could not verify as unverified, and name the concrete checks that remain.',
        'Do not reply only with a resume notice; the gathered evidence must reach the user.',
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

export const MAX_LIVE_OUTCOME_RECOVERY_PROMPTS = 2

export function dropStaleOutcomeRecoveryPrompts(messages: Message[]): void {
  const indexes = messages.flatMap((message, index) =>
    message.role === 'system'
      && typeof message.content === 'string'
      && message.content.includes('[Outcome supervisor]')
      ? [index]
      : [],
  )
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
  'terminal.run',
])
const SUBSTANTIVE_REPOSITORY_REVIEW_TOOL_TURN_LIMIT = 3
const READ_ONLY_EVIDENCE_TURN_LIMIT = 2
const READ_ONLY_EVIDENCE_MIN_SUCCESSFUL_OBSERVATIONS = 4
const TOOL_ENABLED_MIN_OUTPUT_TOKENS = 256
const BOUNDED_FINAL_MAX_OUTPUT_TOKENS = 3_072
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
  systemToolGuidance?: import('./system-prompt.js').SystemToolGuidanceOptions
  provider: ILLMProvider
  tools: ToolRegistry
  policy: PolicyEngine
  autonomy: AutonomyLevel
  modeControl?: ModeControl
  semanticRouting?: boolean
  maxIterations?: number
  hardMaxIterations?: boolean
  maxToolCallsPerTurn?: number
  auditLogger?: IAuditLogger
  usageTracker?: UsageTracker
  hookRegistry?: HookRegistry
  deviceName?: string
  thinkingLevel?: string
  thinkingLevelSource?: ThinkingLevelSource
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
  toolStats?: import('./tool-learning/store.js').ToolStatsStore
  workspaceMutationTracker?: import('./workspace-mutation/tracker.js').WorkspaceMutationTracker
  editSnapshotStore?: import('./edit-rollback/store.js').EditSnapshotStore
  journalStateBoard?: (sessionId: string, board: AgentStateBoardSnapshot) => Promise<void>
  activeRuns?: ActiveRunRegistry
  journalSteeringConsumed?: (sessionId: string, noteId: string) => Promise<void>
  strictFinalAnswerProtocol?: boolean
  reviewOutcomes?: boolean
  reviewToollessFinals?: boolean
  maxContinuationCycles?: number
  emitAgentRunHooks?: boolean
  spendBudget?: SpendBudgetConfig
}

function resolveWriteLoopLimit(): number {
  const raw = Number(process.env.SEPILOTD_WRITE_LOOP_LIMIT)
  if (Number.isFinite(raw) && raw >= 2) return Math.floor(raw)
  return DEFAULT_WRITE_LOOP_LIMIT
}

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
  private thinkingLevelSource?: ThinkingLevelSource
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
  private readonly systemToolGuidance: AgentEngineOptions['systemToolGuidance']
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
    this.systemToolGuidance = options.systemToolGuidance
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
    this.thinkingLevelSource = options.thinkingLevelSource
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
    if (event === 'post:agent:run' && !this.emitAgentRunHooks) return
    try {
      await this.hookRegistry?.trigger({ event, data })
    } catch {
    }
    if (event === 'post:agent:run' && data.status !== 'incomplete') {
      const sessionId = typeof data.sessionId === 'string' ? data.sessionId : undefined
      if (sessionId) {
        await this.workspaceMutationTracker?.cleanupScratch(sessionId).catch(() => {})
      }
    }
  }

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
    const hasNativeToolCalls = (normalizedResponse.message.toolCalls?.length ?? 0) > 0
    const isExplicitFinal = !hasNativeToolCalls && hasPromptFinalEnvelopeIntent(content)
    if (isExplicitFinal) {
      const finalCandidate = extractPromptFinalCandidate(content)
      return {
        ...normalizedResponse,
        message: {
          ...normalizedResponse.message,
          content:
            containsPromptToolCallEnvelope(content)
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
          if (chunk.usage.generationMs) {
            totalUsage.generationMs = (totalUsage.generationMs ?? 0) + chunk.usage.generationMs
          }
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

    const truncatedToolCallIds = new Set<string>()
    const truncatedToolCalls: NonNullable<ChatResponse['truncatedToolCalls']> = []
    for (const tc of toolCalls) {
      const parsed = parseToolCallArguments(toolCallArgs.get(tc.id))
      if (parsed.truncated) {
        truncatedToolCallIds.add(tc.id)
        truncatedToolCalls.push({
          name: tc.name,
          streamedChars: toolCallArgs.get(tc.id)?.length ?? 0,
        })
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
        ...(truncatedToolCalls.length > 0 && finishReason === 'length'
          ? { truncatedToolCalls }
          : {}),
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
        truncatedToolCalls?: ChatResponse['truncatedToolCalls']
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
            if (chunk.usage.generationMs) {
              usage.generationMs = (usage.generationMs ?? 0) + chunk.usage.generationMs
            }
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
      let finishDetail: ChatResponse['finishDetail']
      if (!sawDone && sawNonTerminalChunk) {
        finishReason = 'length'
        finishDetail = 'incomplete_stream'
      }

      const truncatedToolCallIds = new Set<string>()
      const truncatedToolCalls: NonNullable<ChatResponse['truncatedToolCalls']> = []
      for (const toolCall of toolCalls) {
        const parsed = parseToolCallArguments(toolCallArgs.get(toolCall.id))
        if (parsed.truncated) {
          truncatedToolCallIds.add(toolCall.id)
          truncatedToolCalls.push({
            name: toolCall.name,
            streamedChars: toolCallArgs.get(toolCall.id)?.length ?? 0,
          })
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
      const promptTextIsExplicitFinal = toolCalls.length === 0
        && hasPromptFinalEnvelopeIntent(promptToolParseText)
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
        ...(truncatedToolCalls.length > 0 && finishReason === 'length'
          ? { truncatedToolCalls }
          : {}),
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
      truncatedToolCalls,
      thinking,
    } = firstTurnResult.value

    if (
      canPublishUserFacingText(finishReason)
      && resolvedToolCalls.length === 0 &&
      shouldAttemptPromptReActRepair(rawText)
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
      truncatedToolCalls = repairTurnResult.value.truncatedToolCalls
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
      ...(truncatedToolCalls?.length && resolvedToolCalls.length === 0
        ? { truncatedToolCalls }
        : {}),
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
      ...(truncatedToolCalls?.length && resolvedToolCalls.length === 0
        ? { truncatedToolCalls }
        : {}),
    }
  }

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
    const model = this.resolveModelInfo(modelId)
    return resolveModelToolTransport(model, 'react').initial === 'native'
  }

  private stripDuplicateToolNameList(systemPrompt: string | undefined): string | undefined {
    if (!systemPrompt) return systemPrompt
    const withoutCatalog = this.systemToolGuidance
      ? systemPrompt
      : systemPrompt.replace(/^Available tools:[^\n]*\n?/m, '')
    return withoutCatalog
      .replace(new RegExp(`^${TOOL_APPROVAL_POSTURE_PROMPT_PREFIX}[^\\n]*\\n?`, 'm'), '')
      .replace(
        /^Your autonomy level is: [^.\n]+\.$/m,
        `Your autonomy level is: ${this.autonomy}.`,
      )
  }

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
    if (context.memoryPolicy?.read !== false && context.memoryPolicy?.automaticRead !== false && context.relevantMemories?.length) {
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
    for (const message of messages) {
      message.metadata = { ...message.metadata, runtimeContext: true }
    }
    if (context.previousMessages?.length) {
      messages.push(...context.previousMessages.map(cloneMessage))
    }
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
      deadlineAt: this.runWallClock?.budgetMs === undefined ? context.deadlineAt : this.runWallClock.startedAt + this.runWallClock.budgetMs,
      memoryPolicy: cloneCheckpointMemoryPolicy(context.memoryPolicy),
      toolAllowlist: cloneCheckpointToolAllowlist(context.toolAllowlist),
      delegationToolAllowlist: cloneCheckpointToolAllowlist(context.delegationToolAllowlist),
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
      ...(this.thinkingLevelSource ? { thinkingLevelSource: this.thinkingLevelSource } : {}),
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
      deadlineAt: this.runWallClock?.budgetMs === undefined ? context.deadlineAt : this.runWallClock.startedAt + this.runWallClock.budgetMs,
      memoryPolicy: cloneCheckpointMemoryPolicy(context.memoryPolicy),
      toolAllowlist: cloneCheckpointToolAllowlist(context.toolAllowlist),
      delegationToolAllowlist: cloneCheckpointToolAllowlist(context.delegationToolAllowlist),
      skillExecutionPolicies: cloneCheckpointSkillExecutionPolicies(
        context.skillExecutionPolicies,
      ),
      requireToolApproval: context.requireToolApproval,
      runContract: context.runContract,
      totalUsage: { ...totalUsage },
      iteration,
      maxIterations: this.maxIterations,
      thinkingLevel: this.thinkingLevel,
      ...(this.thinkingLevelSource ? { thinkingLevelSource: this.thinkingLevelSource } : {}),
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
          recordScratch: (path: string) =>
            this.workspaceMutationTracker!.recordScratch(context.sessionId, path),
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
      if (snapshots && editCheckpointId) {
        editSummary = snapshots.commitCheckpoint(context.sessionId, editCheckpointId)
        await snapshots.flush()
      }
    }
    if (editSummary) yield { type: 'edit_checkpoint_resolved', checkpoint: editSummary }

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
    reasoningBudgetExhausted = false,
    observationBudgetPressure = false,
    observationBudgetFraction = 0,
    continuationState?: {
      validationConvergence: ReturnType<typeof initialValidationConvergenceState>
      forceFinalAfterValidationConvergence: boolean
      writeLoopTracker: WriteLoopTracker
      invocationHistory: StuckToolRepeatEntry[]
      invalidInvocationRepairUsed: boolean
      pendingOutcomeReviewReason: string | null
      repairedOutcomeReviewCount: number
      mutationReviewBudget?: AuxiliaryLlmTurnBudget
    },
  ): AsyncIterable<AgentEvent> {
    const observationScope = () => JSON.stringify([this.provider.id, context.model, context.runContract])
    let observationBudgetScope = observationScope()
    const activeSkillExecutionPolicies = context.skillExecutionPolicies
      ?? resolveActiveSkillExecutionPolicies(context.executionSkillIds)
    let repairedInterimProgressCount = 0
    let repairedEmptyFinalReplyCount = 0
    let repairedLengthContinuationCount = 0
    let lengthContinuationPrefix = ''
    let hadLengthContinuation = false
    const lengthContinuationMax = readLengthContinuationMax()
    const truncatedToolCallAttempts: { failedAttempts?: AgentFailedAttempt[] } = {}
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
    const writeLoopTracker = continuationState?.writeLoopTracker ?? new WriteLoopTracker(writeLoopLimit)
    const readOnlyToolHistory: StuckToolRepeatEntry[] = []
    const invocationHistory: StuckToolRepeatEntry[] = continuationState?.invocationHistory ?? []
    let invalidInvocationRepairUsed = continuationState?.invalidInvocationRepairUsed ?? false
    let pendingOutcomeReviewReason = continuationState?.pendingOutcomeReviewReason ?? null
    const observationHistory = currentTurnObservationHistory
      ?? observationHistoryFromCurrentTurnMessages(messages)
    let observationReuseOnlyCount = 0
    let observationProgress: WorkProgress = { revision: 0, seen: [] }
    let observationProgressCursor = 0
    let readOnlyEvidenceTurns = 0
    let lastReadOnlyToolSignature: string | null = null
    let forceFinalAfterRepeatedRead = false
    let forceFinalAfterObservationBudget = hasToolCallBatchCap(messages)
    let forceFinalAfterValidationConvergence = continuationState?.forceFinalAfterValidationConvergence ?? false
    let validationConvergence = continuationState?.validationConvergence ?? initialValidationConvergenceState()
    let announcedObservationCompaction = false
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
    let forceFinalAfterWallClockReserve = false
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
    const requireFinalAnswerProtocol = this.strictFinalAnswerProtocol
    const rawTerminalOutputRequested = requestsRawTerminalCommandOutput(focusedRepositoryRequest)
    const focusedRepositoryLookup = !this.semanticRouting && isFocusedRepositoryChangeLookup(focusedRepositoryRequest)
    const focusedRepositoryNeedsDiff = focusedRepositoryLookupNeedsDiff(focusedRepositoryRequest)
    const validationConvergenceEnabled =
      !substantiveRepositoryReview
      && !focusedRepositoryLookup
      && activeSkillExecutionPolicies.length === 0
      && (context.executionSkillIds?.length ?? 0) === 0
      && (context.runContract?.requiredArtifacts?.length ?? 0) === 0
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
    let repairedOutcomeReviewCount = continuationState?.repairedOutcomeReviewCount ?? 0
    let repairedOutcomeReviewSynthesisCount = 0
    let mutationReviewBudget = continuationState?.mutationReviewBudget
    const retainedContinuationState = () => ({
      validationConvergence, forceFinalAfterValidationConvergence, writeLoopTracker, invocationHistory, invalidInvocationRepairUsed,
      pendingOutcomeReviewReason, repairedOutcomeReviewCount, mutationReviewBudget,
    })
    let outcomeReviewSynthesisEvidenceCount = messages.filter(
      (message) => message.role === 'tool',
    ).length
    let repairedOutcomeReviewNoProgressCount = 0
    let repairedProviderMessageCount = countProviderMessageRecoveryPrompts(messages)
    let disableThinkingForProviderRecovery = false
    let contextRecoveries = 0
    let maxOutputTokenRecoveries = 0
    let effectiveMaxOutputTokens: number | undefined = recallObservedMaxOutputTokens(this.provider.id, context.model)
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
      const executionBlocker = workspaceExecutionBlocker(this.tools, context, context.runContract)
      if (executionBlocker) {
        yield* this.finishAfterUserActionRequired(
          userActionRequiredOutput('workspace_execution_unavailable', executionBlocker, latestUserText(messages)),
          context,
          totalUsage,
        )
        return
      }
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
      if (
        wallClock?.budgetMs !== undefined
        && !forceFinalAfterWallClockReserve
        && Date.now() - wallClock.startedAt
          >= wallClock.budgetMs - resolveWallClockSynthesisReserveMs(wallClock.budgetMs)
      ) {
        forceFinalAfterWallClockReserve = true
        yield {
          type: 'thinking',
          content: '[supervisor] Wall-clock limit is near; tools are closed and the final answer is written from the evidence gathered so far.',
        }
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

      const priorObservationRevision = observationProgress.revision
      for (; observationProgressCursor < observationHistory.length; observationProgressCursor++) {
        observationProgress = observeWorkProgress(
          observationProgress,
          observationHistory[observationProgressCursor]!,
        )
      }
      if (observationProgress.revision > priorObservationRevision) observationReuseOnlyCount = 0
      const freshSteering = (this.liveRunState?.steeringNotes ?? []).filter(
        (note) => note.consumedAt === undefined && note.cancelledAt === undefined,
      )
      if (freshSteering.length > 0) {
        observationReuseOnlyCount = 0
        reasoningBudgetExhausted = false
        observationBudgetPressure = false
        observationBudgetFraction = 0
      }
      if (observationBudgetScope !== observationScope()) {
        observationBudgetScope = observationScope()
        observationBudgetPressure = false
        observationBudgetFraction = 0
      }
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
        }
      }
      const compactedReadMessages = compactOversizedToolProtocolUnits(
        supersedeStaleSystemReminders(supersedeStaleReadResults(messages)),
        { contextWindowTokens: contextWindow, charsPerToken },
      )
      let messagesForRequest = withProblemToolFeedback(
        compactedReadMessages,
        this.toolStats?.list(context.sessionId) ?? [],
        i,
      )
      const nativeToolUse = this.supportsNativeToolUse(context.model)
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
        || forceFinalAfterWallClockReserve
      const allToolsForRequest = this.tools.toToolDefinitions(context).filter((tool) => (
        this.autonomy !== AutonomyLevel.ReadOnly
        || tool.name === 'terminal.run'
        || isPolicyReadOnlyTool(tool.name, this.tools.securityDescriptor(tool.name))
      ) && (
        !approvalGraceState.denial
        || isPolicyReadOnlyTool(tool.name)
      ) && (
        !toolRejectedByExecutionIntentForEveryInput(context.runContract, tool.name, this.tools)
      )).map((tool) => {
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
      const observationReuseExhausted = observationReuseOnlyCount >= DEFAULT_STUCK_REPEAT_THRESHOLD
      const prioritizeMissingSkillStage =
        (forceFinalAfterRepeatedRead || forceFinalAfterObservationBudget
          || forceFinalAfterValidationConvergence || observationReuseExhausted)
        && requiresVerifiedSkillExecution
        && missingSkillStageToolsAvailable
        && skillCompletionBeforeRequest.maxRetries > 0
        && repairedSkillExecutionCompletionCount <= skillCompletionBeforeRequest.maxRetries
        && !closedExactToolWorkflowComplete
        && !forceFinalAfterDegenerateOutput
        && !forceFinalAfterReadOnlyPolicyFailure
        && !forceFinalAfterNoRetryActionFailure
        && !forceFinalAfterWallClockReserve
      const contractCommandToolNames = context.runContract?.executionIntent?.kind === 'operational-action'
        ? (context.runContract.executionIntent.allowedTools ?? [])
            .filter((toolName) => allToolNames.has(toolName))
        : []
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
          (forceFinalAfterRepeatedRead || forceFinalAfterObservationBudget
            || forceFinalAfterValidationConvergence || observationReuseExhausted)
          && !prioritizeMissingSkillStage
        )
        || forceFinalAfterDegenerateOutput
        || forceFinalAfterReadOnlyPolicyFailure
        || forceFinalAfterApprovalGrace
        || forceFinalAfterNoRetryActionFailure
        || forceFinalAfterWallClockReserve
        || focusedTerminalCommandCompleted
      if (substantiveRepositoryReview) {
        messagesForRequest = appendSubstantiveRepositoryReviewMessage(messagesForRequest)
      }
      if (focusedEvidenceReady) {
        messagesForRequest = appendFocusedRepositorySynthesisMessage(messagesForRequest)
      } else if (forceFinalAfterWallClockReserve) {
        messagesForRequest = appendWallClockSynthesisMessage(messagesForRequest)
      } else if (forceSynthesisFromObservationBudget) {
        messagesForRequest = appendForceFinalSystemMessage(messagesForRequest)
      }
      const useBoundedFinalContext = activeSkillExecutionPolicies.length === 0
        && (context.executionSkillIds?.length ?? 0) === 0
        && (
          forceFinalAfterRepeatedRead
          || observationReuseExhausted
          || forceFinalAfterObservationBudget
          || forceFinalAfterValidationConvergence
          || forceFinalAfterReadOnlyPolicyFailure
          || forceFinalAfterApprovalGrace
          || forceFinalAfterNoRetryActionFailure
          || forceFinalAfterWallClockReserve
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
      if (this.modeControl?.prompt) messagesForRequest = [...messagesForRequest, { role: 'system', content: this.modeControl.prompt, metadata: { [RUN_STABLE_SYSTEM_METADATA_KEY]: true } }]
      if (allowedToolNames) for (const tool of controlToolsForRequest) allowedToolNames.add(tool.name)
      const useNativeToolUse = nativeToolUse
        && !preferPromptReact
        && !(useBoundedFinalContext && modelInfo?.capabilities.adaptivePromptReact === true)

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

      const willReviewCandidateFinal = !deterministicSkillCompletionSatisfiedBeforeRequest
        && !skipOutcomeReview
        && shouldReviewOutcomeWithLLM(messages, {
          includeToollessFinal: this.reviewToollessFinals || context.runContract?.executionIntent?.workspaceMutation === 'required',
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
      requestMessages = withSystemToolGuidance(requestMessages, [...effectiveAllowedToolNames], this.systemToolGuidance)
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
            requestMessages = withSystemToolGuidance(requestMessages, [], this.systemToolGuidance)
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
          ? resolveRepairThinkingLevel(modelInfo, this.thinkingLevel)
          : disableThinkingForProviderRecovery
          ? undefined
          : resolveThinkingLevel(this.thinkingLevel, {
              taskComplexity: context.runContract?.reasoningComplexity,
              toolNames: [...effectiveAllowedToolNames],
              reasoningBudgetExhausted,
              observationBudgetPressure,
              observationBudgetExhausted: observationBudgetFraction >= 2,
              canDisableThinking: canAutomaticallyDisableThinking(modelInfo),
              executionFailure: observationHistory.at(-1)?.executionObserved === true
                && observationHistory.at(-1)?.status === 'error',
              executionFailureEffect: observationHistory.at(-1)?.securityEffect,
              executionFailureCode: observationHistory.at(-1)?.status === 'error'
                ? structuredToolFailureCodeFromOutput(observationHistory.at(-1)?.output)
                : undefined,
              consecutiveExecutionFailures: consecutiveExecutionFailures(observationHistory),
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
        timeoutMs: this.runWallClock?.budgetMs !== undefined
          ? runRequestTimeoutMs(this.runWallClock, useBoundedFinalContext ? BOUNDED_FINAL_TIMEOUT_MS : resolveProviderStreamFirstTokenMs() ?? undefined)
          : useBoundedFinalContext ? BOUNDED_FINAL_TIMEOUT_MS : undefined,
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
        if (this.runWallClock?.budgetMs !== undefined) request.timeoutMs = runRequestTimeoutMs(this.runWallClock, request.timeoutMs)
        yield buildLlmRequestEvent({
          sessionId: context.sessionId,
          iteration: i + 1,
          source: 'react',
          request,
          turnId: llmTurnId,
          providerId: this.provider.id,
          timeoutMs: request.timeoutMs ?? resolveProviderStreamFirstTokenMs(),
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

        if (this.aborted || signal?.aborted) break
        if (this.hasPendingSteering()) continue

        if (shouldReduceAutomaticThinking(this.thinkingLevel, request.thinkingLevel, {
          toolEnabled: effectiveAllowedToolNames.size > 0,
          finishReason: response.finishReason,
          finishDetail: response.finishDetail,
          content: this.responseTextContent(response),
          thinking: response.thinking,
          hasToolOutput: (response.message.toolCalls?.length ?? 0) > 0,
          outputTokens: response.usage.outputTokens,
          maxTokens: request.maxTokens,
        })) reasoningBudgetExhausted = true

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
              status: executableToolCalls.length === 0 ? 'force-next-turn' : 'partial-reuse',
              data: {
                reuseOnlySelections: observationReuseOnlyCount + (executableToolCalls.length === 0 ? 1 : 0),
                novelEvidenceRevision: observationProgress.revision,
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
              observationReuseOnlyCount += 1
              messages.push(response.message)
              messages.push(...buildObservationReuseToolResultMessages(
                observationReuse.coveredCalls,
              ))
              messages.push({
                ...observationReuseMessage,
                content: `${observationReuseMessage.content}\nReuse-only selections since last new evidence: ${observationReuseOnlyCount}/${DEFAULT_STUCK_REPEAT_THRESHOLD}. Continue the active task with a genuinely new observation, authorized action, verification, or grounded final answer. Reuse is not evidence that the task is finished.`,
              })
              if (observationReuseOnlyCount >= DEFAULT_STUCK_REPEAT_THRESHOLD) {
                yield { type: 'thinking', content: `[supervisor] ${observationReuseOnlyCount} cache-only decisions produced no new executor evidence; switching to bounded synthesis with the completion checks still active.` }
              }
              this.state = 'observing'
              yield { type: 'state_change', state: 'observing' }
              continue
            }
          }

          const writeLoopTarget = writeLoopTracker.offendingTarget(executableToolCalls)
          if (writeLoopTarget) {
            messages.push(response.message)
            messages.push(buildWriteLoopRepairMessage(writeLoopTarget, writeLoopLimit))
            this.state = 'observing'
            yield { type: 'state_change', state: 'observing' }
            continue
          }

          const invalidRepeat = findRepeatedInvalidToolCall(invocationHistory,
            executableToolCalls.map(call => ({ tool: call.name, input: call.arguments })))
          if (invalidRepeat) {
            const exhausted = invalidInvocationRepairUsed
            invalidInvocationRepairUsed = true
            if (exhausted) forceFinalAfterNoRetryActionFailure = true
            messages.push({
              role: 'system',
              metadata: { reminderKind: exhausted ? 'no_retry_action_failure' : 'invalid_invocation_repair' },
              content: exhausted
                ? `[Invalid invocation boundary] ${invalidRepeat.tool} was proposed again with unchanged rejected input after the input-contract repair turn. This batch was not executed. Tool recovery is exhausted; summarize retained evidence and report the remaining blocker as INCOMPLETE. Do not claim the rejected action succeeded.`
                : `[Tool input correction] ${invalidRepeat.tool} has twice rejected the same input before execution. This repeated batch was not executed. One input-contract repair turn is available: use the reported validation error and the schema below to supply corrected arguments, or choose an appropriate alternative tool. Do not repeat the unchanged input. If you cannot correct it, report the concrete blocker. This does not authorize any additional action. Input schema: ${JSON.stringify(this.tools.get(invalidRepeat.tool)?.inputSchema ?? {})}`,
            })
            this.state = 'observing'
            yield { type: 'state_change', state: 'observing' }
            yield {
              type: 'thinking',
              content: exhausted
                ? `[supervisor] Input-contract repair did not resolve repeated invalid ${invalidRepeat.tool} calls; reporting the remaining blocker.`
                : `[supervisor] Skipped unchanged invalid ${invalidRepeat.tool}; allowing one explicit input-contract repair.`,
            }
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
              executionObserved: false,
              ts: Date.now(),
            }))
          const readLoop = detectStuckToolRepeat(
            [...readOnlyToolHistory, ...proposedReadEntries],
            {
              trackedTools: REACT_CONSECUTIVE_READ_GUARD_TOOLS,
              targetRepeatThreshold: DEFAULT_TARGET_REPEAT_THRESHOLD,
              permanentFailureChurnThreshold: DEFAULT_PERMANENT_FAILURE_CHURN_THRESHOLD,
              executionFailureChurnThreshold: DEFAULT_EXECUTION_FAILURE_CHURN_THRESHOLD,
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
          const readResultFingerprints = new Map<string, string>()
          const guardedReadIds = new Set(guardedObservationCalls.map(call => call.id))
          const proposedInputs = new Map(executableToolCalls.map((call) => [call.id, call.arguments]))
          const requiredUserAction = yield* this.runToolCalls(
            messages,
            executableToolCalls,
            context,
            i,
            totalUsage,
            undefined,
            signal,
            canAttachVisualContent,
            (event) => {
              if (guardedReadIds.has(event.toolCallId)) {
                readResultFingerprints.set(event.toolCallId, createHash('sha256').update(event.output).digest('hex'))
              }
              if (rawTerminalOutputRequested || focusedProcessStart || focusedProcessObservation
                || focusedBrowserObservation || focusedVisibleUrlOpen) focusedActionResult = event
            },
          )
          if (requiredUserAction) {
            yield* this.finishAfterUserActionRequired(requiredUserAction, context, totalUsage)
            return
          }
          if (validationConvergenceEnabled) {
            const validationEvidenceEntries: ValidationConvergenceEntry[] = executableToolCalls.map((call) => {
              const resultMessage = [...messages].reverse().find((message) => (
                message.role === 'tool' && message.toolCallId === call.id
              ))
              return {
                status: resultMessage?.metadata?.toolResultStatus === 'success'
                  ? 'success'
                  : 'error',
                executionObserved: resultMessage?.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === true,
                reliableSuccess: call.name !== 'terminal.run'
                  || terminalRunHasReliableSuccessStatus(call.arguments),
                securityEffect: this.tools.securityDescriptor(call.name)?.effect ?? 'unknown',
                ...(Object.hasOwn(call.arguments, 'actionPurpose')
                  ? { actionPurpose: call.arguments.actionPurpose }
                  : {}),
              }
            })
            const validationConvergenceTransition = advanceValidationConvergence(
              validationConvergence,
              validationEvidenceEntries,
            )
            validationConvergence = validationConvergenceTransition.state
            if (validationConvergenceTransition.forceFinal) {
              forceFinalAfterValidationConvergence = true
              messages.push({
                role: 'system',
                metadata: { reminderKind: 'post_validation_convergence' },
                content: [
                  '[Post-validation convergence boundary]',
                  'A workspace mutation has already been followed by successful validation evidence.',
                  'The bounded follow-up observation allowance is exhausted. Do not widen discovery or create a new reproduction; synthesize the current result now.',
                  'If an acceptance criterion is still genuinely unmet, respond with INCOMPLETE and name that concrete gap instead of calling more read-only tools.',
                ].join(' '),
              })
              yield {
                type: 'thinking',
                content: '[supervisor] Successful post-mutation validation reached the bounded convergence boundary; switching to final synthesis.',
              }
            }
          }
          const successfulNonObservation = executableToolCalls.some((call) => {
            if (this.tools.securityDescriptor(call.name).effect === 'observe') return false
            const resultMessage = [...messages].reverse().find((message) => (
              message.role === 'tool' && message.toolCallId === call.id
            ))
            return resultMessage?.metadata?.toolResultStatus === 'success'
          })
          const staleObservationScope = observationBudgetScope !== observationScope()
          if (successfulNonObservation || staleObservationScope) observationBudgetFraction = 0
          if (!staleObservationScope) observationBudgetFraction = advanceAutomaticObservationBudget(this.thinkingLevel, request.thinkingLevel, observationBudgetFraction, {
            finishReason: response.finishReason,
            finishDetail: response.finishDetail,
            outputTokens: response.usage.outputTokens,
            maxTokens: request.maxTokens,
            thinkingChars: response.thinking?.length ?? 0,
            visibleChars: this.responseTextContent(response).length,
            observationOnlyTools: executableToolCalls.length > 0
              && executableToolCalls.every((call) => this.tools.securityDescriptor(call.name).effect === 'observe'),
            taskComplexity: context.runContract?.reasoningComplexity,
          })
          observationBudgetPressure = observationBudgetFraction >= 1
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
          const completedInvocationEntries = executableToolCalls.map((call) => {
            const resultMessage = [...messages].reverse().find((message) => (
              message.role === 'tool' && message.toolCallId === call.id
            ))
            const status = resultMessage?.metadata?.toolResultStatus === 'success'
              ? 'success' as const
              : 'error' as const
            const blocked = blockSourceFromMetadata(resultMessage?.metadata) !== null
            return {
              tool: call.name,
              input: proposedInputs.get(call.id) ?? call.arguments,
              status,
              executionObserved: resultMessage?.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === true,
              securityEffect: this.tools.securityDescriptor(call.name).effect,
              outputFingerprint: readResultFingerprints.get(call.id),
              ...(typeof resultMessage?.content === 'string'
                ? { output: resultMessage.content.slice(0, 4_000) }
                : {}),
              ...(status === 'error' && !blocked
                ? { failureCode: structuredToolFailureCode(resultMessage) }
                : {}),
              ...(blocked ? { blocked: true } : {}),
              ts: Date.now(),
            }
          })
          readOnlyToolHistory.push(...completedInvocationEntries.filter((_, index) => (
            guardedReadIds.has(executableToolCalls[index]!.id)
          )))
          invocationHistory.push(...completedInvocationEntries)
          invocationHistory.splice(0, Math.max(0, invocationHistory.length - DEFAULT_STUCK_REPEAT_WINDOW))
          for (const call of executableToolCalls) {
            const resultMessage = [...messages].reverse().find((message) => (
              message.role === 'tool' && message.toolCallId === call.id
            ))
            observationHistory.push({
              tool: call.name,
              input: { ...call.arguments },
              securityEffect: this.tools.securityDescriptor(call.name)?.effect ?? 'unknown',
              executionObserved: resultMessage?.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === true,
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
              reasoningBudgetExhausted,
              observationBudgetPressure,
              observationBudgetFraction,
              retainedContinuationState(),
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
          const synthesisOnly = evidenceAvailable
            && !controlToolsForRequest.some(tool => tool.name === MODE_TRANSFER_TOOL)
          repairedInvalidToolResponseCount += 1
          if (synthesisOnly) forceFinalAfterDegenerateOutput = true
          messages.push(buildInvalidToolResponseRepairMessage({
            evidenceAvailable: synthesisOnly,
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

        const truncatedNativeToolCalls = response.finishReason === 'length'
          ? response.truncatedToolCalls ?? []
          : []
        const truncatedPromptToolCall =
          truncatedNativeToolCalls.length > 0
          || (
            !useNativeToolUse
            && response.finishReason === 'length'
            && containsPromptToolCallEnvelope(content)
          )

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
          const truncatedToolCalls = truncatedNativeToolCalls.length > 0
            ? truncatedNativeToolCalls
            : parseTruncatedPromptToolCallNames(content)
          const attempt = recordTruncatedToolCallAttempt(
            truncatedToolCallAttempts,
            truncatedToolCalls,
            request.maxTokens,
          )
          const recovery = buildTruncatedPromptToolCallRecoveryMessage({
            toolCalls: truncatedToolCalls,
            maxOutputTokens: request.maxTokens,
            attempt,
          })
          const recoveryPrefix = String(recovery.content).split('\n')[0] ?? ''
          for (let i = messages.length - 1; i >= 0; i--) {
            const message = messages[i]!
            if (message.role === 'system' && typeof message.content === 'string'
              && recoveryPrefix && message.content.startsWith(recoveryPrefix)) {
              messages.splice(i, 1)
            }
          }
          messages.push(recovery)
          yield buildTruncatedToolCallRecoveryEvent(
            truncatedToolCalls,
            request.maxTokens,
            attempt,
            { budget: lengthContinuationMax, used: repairedLengthContinuationCount },
          )
          this.state = 'observing'
          yield { type: 'state_change', state: 'observing' }
          yield {
            type: 'thinking',
            content: `[continuation] Discarded a truncated tool call and requested a fresh bounded call (${repairedLengthContinuationCount}/${lengthContinuationMax}).`,
          }
          continue
        }

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
        let terminalResponseFailure =
          interruptedResponse
          || repeatedRecoveryFailed
          || emptyDegenerationRecovery
          || closedExactToolWorkflowComplete
          || forceFinalAfterNoRetryActionFailure
          || forceFinalAfterWallClockReserve
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
            includeToollessFinal: this.reviewToollessFinals || context.runContract?.executionIntent?.workspaceMutation === 'required',
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
            includeToollessFinal: this.reviewToollessFinals || context.runContract?.executionIntent?.workspaceMutation === 'required',
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
            includeToollessFinal: this.reviewToollessFinals || context.runContract?.executionIntent?.workspaceMutation === 'required',
          })
        ) {
          const reviewRequest = buildRunOutcomeReviewRequest({
            model: context.model,
            messages,
            assistantAnswer: content,
            userInstructions: activeUserInstructions(this.liveRunState?.steeringNotes),
            availableToolNames: allToolsForRequest.map((tool) => tool.name),
            runContract: context.runContract,
            maxTokens: resolveModelOutputAllowance({
              requested: this.maxTokens,
              modelMaximum: modelInfo?.maxOutputTokens,
              observedMaximum: effectiveMaxOutputTokens,
              fallback: OUTCOME_REVIEW_MAX_TOKENS,
            }),
          })
          const reviewTurnId = buildLlmTurnId(context.sessionId, i + 1, 'outcome-review')
          const requiresMutationReview = unavailableOutcomeReviewReason(messages, context.runContract) !== null
          if (requiresMutationReview && !mutationReviewBudget) {
            mutationReviewBudget = new AuxiliaryLlmTurnBudget(resolveControlCallTimeoutMs())
          }
          const reviewBudget = requiresMutationReview ? mutationReviewBudget : this.auxiliaryLlmBudget
          try {
            Object.assign(reviewRequest, fitControlRequest(reviewRequest, modelInfo))
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
              controlResponseAccepted: (response) => parseRunOutcomeReviewTransport(
                this.responseTextContent(response),
                response.thinking,
              ) !== null,
              label: 'outcome review',
              breaker: this.providerCircuitBreaker,
              signal,
              budget: reviewBudget,
              timeoutMs: requiresMutationReview ? null : undefined,
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
              ?? (reviewResponse.finishReason === 'length'
                ? null : buildEmptyRunOutcomeReviewRecovery(reviewText))
            if (!parsedReview && reviewResponse.finishReason === 'length') {
              exhaustedOutcomeReviewReason = 'Completion review was unavailable after its bounded output recovery; no complete judgment was returned.'
            }
            const review = enforceRunOutcomeReviewEvidenceFloor({
              review: parsedReview,
              messages,
              runContract: context.runContract,
              assistantAnswer: content,
            })
            if (review?.status === 'needs_recovery') {
              outcomeReviewRecoveryReasons.push(review.reason)
              if (review.recoveryMode !== 'synthesis' && (requiresMutationReview
                || review.recoveryMode === 'tool' || review.recoveryMode === 'user-input')) {
                pendingOutcomeReviewReason = review.reason
              }
            } else if (review?.status === 'complete') {
              pendingOutcomeReviewReason = null
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
            exhaustedOutcomeReviewReason = unavailableOutcomeReviewReason(messages, context.runContract)
          }
        }

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
        if (pendingOutcomeReviewReason && !exhaustedOutcomeReviewReason) {
          exhaustedOutcomeReviewReason = pendingOutcomeReviewReason
        }
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
          : forceFinalAfterWallClockReserve && isIncompleteOutput(finalContent)
          ? stopReasonWallClock(this.runWallClock?.budgetMs)
          : (forceFinalAfterObservationBudget || forceFinalAfterValidationConvergence)
          ? stopReasonObservationBudget({
              incomplete: isIncompleteOutput(finalContent),
              contract: context.runContract,
            })
          : isIncompleteOutput(finalContent)
          ? stopReasonCompletionGate({
              source: exhaustedOutcomeReviewReason ? 'gate' : 'self_reported',
              unmet: [exhaustedOutcomeReviewReason ?? incompleteOutputReason(content) ?? incompleteOutputReason(finalContent)]
                .filter((x): x is string => Boolean(x)),
            })
          : exhaustedOutcomeReviewReason || hasIncompleteAnswerStem(content) || forceFinalAfterNoRetryActionFailure
            ? stopReasonCompletionGate({ unmet: exhaustedOutcomeReviewReason ? [exhaustedOutcomeReviewReason] : undefined })
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
        if (this.runWallClock?.budgetMs !== undefined && Date.now() - this.runWallClock.startedAt >= this.runWallClock.budgetMs) continue
        const observedMaxOutputTokens = maxOutputTokenRecoveries < 1
          ? detectProviderMaxOutputTokenLimit(err, request.maxTokens ?? requestedOutputTokens)
          : null
        if (observedMaxOutputTokens) {
          maxOutputTokenRecoveries += 1
          effectiveMaxOutputTokens = observedMaxOutputTokens
          rememberObservedMaxOutputTokens(this.provider.id, context.model, observedMaxOutputTokens)
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
        reasoningBudgetExhausted,
        observationBudgetPressure,
        observationBudgetFraction,
        retainedContinuationState(),
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

  setAutonomy(level: AutonomyLevel): void {
    this.autonomy = level
  }

  async *run(input: string, context: AgentContext): AsyncIterable<AgentEvent> {
    this.aborted = false
    const abortController = new AbortController()
    this.runAbortController = abortController
    this.runWallClock = resolveRunWallClock(Date.now(), context.deadlineAt)
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
    this.runWallClock = resolveRunWallClock(Date.now(), checkpoint.deadlineAt)
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
          deadlineAt: checkpoint.deadlineAt,
          memoryPolicy: cloneCheckpointMemoryPolicy(checkpoint.memoryPolicy),
          toolAllowlist: cloneCheckpointToolAllowlist(checkpoint.toolAllowlist),
          delegationToolAllowlist: cloneCheckpointToolAllowlist(checkpoint.delegationToolAllowlist),
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
    this.runWallClock = resolveRunWallClock(Date.now(), checkpoint.deadlineAt)
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
          deadlineAt: checkpoint.deadlineAt,
          memoryPolicy: cloneCheckpointMemoryPolicy(checkpoint.memoryPolicy),
          toolAllowlist: cloneCheckpointToolAllowlist(checkpoint.toolAllowlist),
          delegationToolAllowlist: cloneCheckpointToolAllowlist(checkpoint.delegationToolAllowlist),
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
