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

export interface AgentCriterionVerdictSnapshot {
  toolResultCount: number
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
  reasoningComplexity?: import('@sepilotd/core').AgentRunContract['reasoningComplexity']
  source: 'planner' | 'fallback'
}

export interface AgentCodebaseMap {
  generatedAt: string
  scoutPrompts: string[]
  summaries: Array<{
    toolCallId: string
    status: 'success' | 'error'
    executionObserved?: boolean
    summary: string
  }>
}

export interface AgentEvidenceLedgerEntry {
  toolCallId?: string
  tool: string
  status: 'success' | 'error'
  ts: number
  order?: number
  path?: string
  query?: string
  command?: string
  managedProcessId?: string
  managedProcessStartedAt?: string
  managedProcessReliableStatus?: boolean
  bytes?: number
  summary?: string
  defects?: string[]
  uiAuditKey?: string
  uiAuditHasLayout?: boolean
  uiAuditHasConsole?: boolean
  uiAuditHasImageAttachment?: boolean
  uiInteractionSmoke?: boolean
  verified?: boolean
  actionPurpose?: string
  securityEffect?: ToolSecurityEffect
  executionObserved?: boolean
  executionPosture?: ToolExecutionPosture
  origin?: { sessionId: string; category: string }
}

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

export interface AgentFailedAttempt {
  signature: string
  tool: string
  reason: string
  ts: number
}

export interface AgentTruncatedToolCall {
  name: string
  streamedChars: number
}

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
  currentUserContent?: Message['content']
  messages: Message[]
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
  steeringNotes?: import('./state-board.js').AgentSteeringNote[]
  seedContract?: AgentSeedContract
  specialistRoute?: AgentSpecialistRoute
  specialistReason?: string
  specialistBrief?: string
  planIndex: number
  toolCalls: ToolCall[]
  announcedActionProgressToolCallIds?: string[]
  actionProgressMissingCount?: number
  pendingActionProgress?: {
    summary: string
    nextStep: string
    toolCallIds: string[]
  }
  deterministicInitialToolCallId?: string
  toolResults: AgentToolResultSummary[]
  recentToolResults?: AgentToolResultSummary[]
  memories: string[]
  output: string
  userActionRequired?: string
  approvalDenied?: {
    toolCallId: string
    toolName?: string
  }
  approvalDenialGrace?: {
    toolCallId: string
    toolName?: string
    toolTurnsRemaining: number
  }
  frictionCount?: number
  frictionWarningIssued?: boolean
  currentTurnEvidenceRetryCount?: number
  outcomeReviewRecovery?: {
    evidenceCount: number
    toolRepairs: number
    synthesisRepairs: number
    noProgressRepairs: number
  }
  visualAttachmentsDisabled?: boolean
  imageInputRecoveryCount?: number
  budgetExhausted?: boolean
  stopReason?: RunStopReason
  continuationProgressSnapshot?: ContinuationProgressSnapshot
  loopControlQuestionCount?: number
  loopControlGraceTurns?: number
  codebaseExploration?: string
  codebaseMap?: AgentCodebaseMap
  evidenceLedger?: AgentEvidenceLedger
  toolRecommendationSummary?: string
  analysisSummary?: string
  findingsSummary?: string
  implementationSummary?: string
  implementationMutationEvidence?: Array<{
    checkpointId: string
    summary: string
  }>
  implementationRetryRequested?: boolean
  implementationNoEditRetryCount?: number
  implementationStemlessProseCount?: number
  deferredFileEdits?: Array<{ toolCallId: string; tool: string; paths: string[] }>
  deferredFileEditReminderCount?: number
  implementationToolHistoryStartIndex?: number
  internalGraphContinuation?: boolean
  lastRejectedToolCallNames?: string[]
  unexecutedToolSelections?: Array<{
    toolCallId: string
    toolName: string
    phase: string
    argumentsExcerpt: string
    assistantExcerpt: string
  }>
  implementationConvergenceNudgeCount?: number
  implementationPreActionStallCount?: number
  implementationPostActionStallCount?: number
  implementationObservationReuseOnlyCount?: number
  reviewObservationTurnCount?: number
  implementationPendingTodoRetryCount?: number
  implementationScaffoldingRequested?: boolean
  implementationScaffoldingApplied?: boolean
  implementationScaffoldToolHistoryBaseline?: number
  implementationCompleteRequested?: boolean
  preferPromptReact?: boolean
  emptyNativeTurnsCount?: number
  effectiveMaxOutputTokens?: number
  adaptiveMainToolTurnMaxTokens?: number
  autoThinkingBudgetExhaustedScope?: string
  autoThinkingObservationBudgetScope?: string
  autoThinkingObservationMutationCount?: number
  autoThinkingObservationBudgetFraction?: number
  lastTurnAutoThinkingBudgetExhausted?: boolean
  lastMainToolTurnMaxTokens?: number
  stuckRepeatForcedFinal?: boolean
  observeOnlyStuckRepeatRepairCount?: number
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
  iterationBudgetFinalSynthesisGranted?: boolean
  contentOnlyNativeTurnsCount?: number
  promptReactNativeReprobeCount?: number
  verificationSummary?: string
  researchRounds?: number
  researchSearchBaselineFingerprint?: string
  researchSearchAddedEvidence?: boolean
  researchRunToolCallCount?: number
  researchRunToolCallBudget?: number
  researchPhaseToolCallCount?: number
  researchPhaseToolHistoryBaseline?: number
  deepAnswerReviseRequested?: boolean
  deepAnswerReviseCount?: number
  validationSummary?: string
  validationToolHistoryStartIndex?: number
  validationPhaseToolHistoryStartIndex?: number
  validationConvergenceNudgeCount?: number
  reviewSummary?: string
  qualityGateDecision?: 'pass' | 'retry' | 'incomplete'
  qualityGateSummary?: string
  completionDiagnostics?: {
    rawReporterOutput?: string
    presentation?: FinalAnswerPresentationDiagnostics
    criterionVerdictSnapshot?: AgentCriterionVerdictSnapshot
    criterionEvidenceReview?: AgentCriterionEvidenceReviewEpisode
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
  backtrackReasons?: string[]
  completionGateBlocks?: number
  workProgress?: import('../work-progress.js').WorkProgress
  completionGateRejectedDraft?: string
  noProgressIterations?: number
  noProgressRecoveryJudgmentCount?: number
  noProgressRecoveryControllerFailureCount?: number
  noProgressRecoveryControllerEvidenceSignature?: string
  noProgressRecoveryProviderCallCount?: number
  recoveryControllerFailureTotal?: number
  recoveryProviderCallTotal?: number
  recoveryExhaustedFinalCount?: number
  implementationControllerFallbackTurnGranted?: boolean
  focusedRecoveryAnchorToolCallId?: string
  implementationActionOnlyRecovery?: boolean
  implementationActionOnlyRecoveryAttempted?: boolean
  implementationActionOnlyCorrectionCount?: number
  implementationMutationHandoff?: {
    reason: string
    guidance: string
    startedAtIteration?: number
  }
  implementationMutationCapabilityBoundary?: 'authoritative'
  implementationRecoveryActionPending?: {
    signature: string
    actionPurpose: 'observe' | 'mutate' | 'validate' | 'unblock'
  }
  implementationRecoveryHandoff?: string
  implementationRecoveryHandoffReady?: boolean
  implementationCausalDiagnosis?: string
  implementationCausalObservation?: {
    tool: string
    input: Record<string, unknown>
    replayEvidence?: string
  }
  implementationCausalTransition?: {
    decision: 'mutate' | 'complete_phase' | 'blocked'
    goalStatus: 'unresolved' | 'satisfied' | 'blocked'
    phaseStatus: 'unresolved' | 'satisfied' | 'blocked'
    reason: string
    guidance: string
  }
  implementationCausalObservationRejection?: string
  implementationCausalDiagnosisAttempted?: boolean
  implementationCausalDiagnosisEvidenceSignature?: string
  implementationCausalDiagnosisAttemptEvidenceSignature?: string
  implementationCausalDiagnosisAttemptCount?: number
  implementationNetMutationPresent?: boolean
  implementationNetMutationPaths?: string[]
  implementationModelRecoveryRequested?: boolean
  implementationCompletionAuditRequested?: boolean
  implementationCompletionAuditToolCallCount?: number
  qualityConclusionRecoveryRequested?: 'validation' | 'review'
  qualityConclusionResolvedPhase?: 'validation' | 'review'
  reviewConclusionReceipt?: {
    fingerprint: string
    decision: 'VERIFIED' | 'UNVERIFIED'
    summary: string
    nextPhase: 'none' | 'implementation' | 'validation' | 'blocked'
  }
  qualityConclusionRetryTarget?: 'implementation' | 'validation' | 'blocked'
  qualityConclusionTransportFailure?: 'validation' | 'review'
  validationRetryEvidenceBaseline?: string[]
  lastIterationToolCallCount?: number
  artifactRevisionDrafts?: number
  lastTurnFinishReason?: ChatResponse['finishReason']
  lastTurnNeedsProtocolRepair?: boolean
  modelResponseRecoveryCount?: number
  modelResponseRecoveryPending?: boolean
  lengthContinuationCount?: number
  lengthContinuationPrefix?: string
  pendingLengthContinuationText?: string
  pendingLengthContinuationIsToolCall?: boolean
  pendingTruncatedToolCalls?: AgentTruncatedToolCall[]
  failedAttempts?: AgentFailedAttempt[]
  failedAttemptBlocks?: number
  openQuestions?: AgentOpenQuestion[]
  todoList?: import('@sepilotd/core').TodoItem[]
  escalationCount?: number
  lastVerifiedEvidenceCount?: number
  lastVerifiedEvidenceIteration?: number
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
  reflectionMemo?: string[]
  toolCallHistory?: Array<{
    toolCallId?: string
    tool: string
    input: Record<string, unknown>
    status: 'success' | 'error'
    executionObserved?: boolean
    securityEffect?: ToolSecurityEffect
    executionPosture?: ToolExecutionPosture
    failureCode?: string
    blocked?: boolean
    ts: number
    outputFingerprint?: string
    observationContentHash?: string
    output?: string
  }>
  validationOutcome?: {
    command?: string
    exitCode?: number
    passed: boolean
    failedSignals: string[]
    rawOutput?: string
  }
  postEditFindings?: {
    editedFiles: string[]
    impactedExternalModules: string[]
    impactedLocalModules: string[]
    reverseCallers: string[]
    diagnostics: Array<{
      file: string
      summary: string
    }>
    analyzedAt: string
  }
  compressedHistorySummary?: string
  compressedHistoryUpToIndex?: number
  phaseUsages?: Record<string, TokenUsage>
  phaseUsageStart?: { phase: string; usage: TokenUsage }
}

export interface GraphNodeDeps {
  provider: ILLMProvider
  tools: ToolRegistry
  policy: PolicyEngine
  autonomy: AutonomyLevel
  semanticIndex?: ISemanticIndex
  systemPrompt?: string
  systemToolGuidance?: import('../system-prompt.js').SystemToolGuidanceOptions
  usageTracker?: UsageTracker
  providerCircuitBreaker?: ProviderCircuitBreaker
  auxiliaryLlmBudget?: import('../auxiliary-llm.js').AuxiliaryLlmTurnBudget
  controlLlmBudget?: import('../auxiliary-llm.js').AuxiliaryLlmTurnBudget
  emittedLlmRequestObjects?: WeakSet<ChatRequest>
  skillsDir?: string
}

export interface GraphExecutionContext extends GraphNodeDeps {
  modeControl?: ModeControl
  graphId: string
  stagePolicy?: import('../bots/types.js').RoleBotStagePolicy
  executionControl?: 'model-directed'
  agentContext: AgentContext
  toolsForbiddenByUser?: boolean
  toolSecurityEffectBoundary?: 'observe-only'
  requireToolApproval?: boolean
  activeGraphNodeId?: string
  activeGraphIteration?: number
  agentSubgraphNodeId?: string
  remainingNodeBudget?: number
  pendingAgentEvents?: AgentEvent[]
  graphNodeModelOverrides?: import('../../config/schema.js').SepilotdConfig['agent']['graphNodeModelOverrides']
  signal?: AbortSignal
  auditLogger?: IAuditLogger
  hookRegistry?: HookRegistry
  deviceName?: string
  thinkingLevel?: string
  thinkingLevelSource?: import('../../providers/thinking-policy.js').ThinkingLevelSource
  maxTokens?: number
  temperature?: number
  auxModel?: string
  textDeltaMode?: 'buffered' | 'live'
  suppressSpeculativeFinalDeltas?: boolean
  panelPersonas?: import('../personas.js').Persona[]
  panelStrategy?: 'sequential' | 'moderated'
  llmCache?: LLMCache
  approvalCallback?: ApprovalCallback
  evaluateAutoApproval?: AutoApprovalEvaluator
  saveApprovalCheckpoint?: (checkpoint: ApprovalRunCheckpoint) => Promise<void>
  clearApprovalCheckpoint?: (requestId: string) => Promise<void>
  saveRunCheckpoint?: (checkpoint: SessionRunCheckpoint) => Promise<void>
  clearRunCheckpoint?: (sessionId: string) => Promise<void>
  journalStateBoard?: (board: AgentStateBoardSnapshot) => Promise<void>
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
