import type { Message } from '@sepilotd/core'
import type { AgentState, GraphExecutionContext } from './types.js'
import {
  cloneCheckpointExecutionSkillIds,
  cloneCheckpointScopeTags,
  cloneCheckpointSkillExecutionPolicies,
  cloneCheckpointSkillToolNames,
  cloneCheckpointToolAllowlist,
  type ApprovalRunCheckpoint,
} from '../../server/runtime/checkpoints.js'
import type { SessionRunCheckpoint, RunResumeStage } from '../../server/runtime/runs.js'
import type { PendingToolExecution } from '../tool-execution.js'
import { cloneMessage, cloneToolCall } from '../tool-execution.js'
import { cloneEvidenceLedger } from './evidence-ledger.js'

function cloneToolHistoryInput(input: Record<string, unknown>): Record<string, unknown> {
  return JSON.parse(JSON.stringify(input)) as Record<string, unknown>
}

/**
 * Default subgraph recursion depth kept verbatim when serializing a checkpoint.
 * Beyond this depth the nested subgraph's message bodies are elided to count
 * stubs so a deeply-nested orchestration doesn't embed many full message
 * histories. Overridable via `SEPILOTD_CHECKPOINT_SUBGRAPH_DEPTH`.
 */
export const MAX_SUBGRAPH_DEPTH = 2

/**
 * Resolve the effective checkpoint subgraph depth. Defaults to
 * `MAX_SUBGRAPH_DEPTH`; `SEPILOTD_CHECKPOINT_SUBGRAPH_DEPTH` overrides it
 * (clamped to a safe range). General knob — no model/dataset/graphId branching.
 */
export function resolveCheckpointSubgraphDepth(): number {
  const raw = process.env.SEPILOTD_CHECKPOINT_SUBGRAPH_DEPTH
  if (!raw) return MAX_SUBGRAPH_DEPTH
  const parsed = Number.parseInt(raw, 10)
  if (!Number.isFinite(parsed)) return MAX_SUBGRAPH_DEPTH
  return Math.max(1, Math.min(5, parsed))
}

export interface CloneGraphStateOptions {
  /**
   * Slim the clone for durable checkpoint serialization: bound subgraph
   * recursion depth (elide deeper nested message bodies to count stubs).
   * Omit (live clone) to keep the result byte-identical to the input.
   */
  forCheckpoint?: boolean
  /** Internal recursion depth. Callers should not set this. */
  depth?: number
}

/**
 * Elide a message body for checkpoint serialization: keep the structural
 * envelope (role, toolCallId) but drop the (potentially large) content,
 * recording the original character length so a reader knows the body was
 * compressed here — this is a bounded stub, not a silent deletion.
 */
function elideMessageBody(message: Message): Message {
  const chars = typeof message.content === 'string'
    ? message.content.length
    : JSON.stringify(message.content).length
  return {
    role: message.role,
    content: '',
    ...(message.toolCallId ? { toolCallId: message.toolCallId } : {}),
    metadata: { ...(message.metadata ?? {}), elided: true, chars },
  }
}

export function cloneGraphState(
  state: AgentState,
  options?: CloneGraphStateOptions,
): AgentState {
  const forCheckpoint = options?.forCheckpoint ?? false
  const depth = options?.depth ?? 0
  const elideMessages = forCheckpoint && depth > resolveCheckpointSubgraphDepth()
  return {
    ...state,
    input: state.input,
    effectiveContextWindowTokens: state.effectiveContextWindowTokens,
    messages: elideMessages
      ? state.messages.map(elideMessageBody)
      : state.messages.map(cloneMessage),
    currentStep: state.currentStep,
    plan: state.plan ? [...state.plan] : undefined,
    coworkPlan: state.coworkPlan
      ? state.coworkPlan.map((item) => ({
          role: item.role,
          instruction: item.instruction,
          choices: item.choices ? [...item.choices] : undefined,
        }))
      : undefined,
    coworkDiscussPrompt: state.coworkDiscussPrompt,
    coworkDiscussChoices: state.coworkDiscussChoices ? [...state.coworkDiscussChoices] : undefined,
    coworkDiscussCount: state.coworkDiscussCount,
    coworkArtifactRetryCount: state.coworkArtifactRetryCount,
    coworkTaskResults: state.coworkTaskResults
      ? state.coworkTaskResults.map((result) => ({ ...result }))
      : undefined,
    validationPlan: state.validationPlan ? [...state.validationPlan] : undefined,
    seedContract: state.seedContract
      ? {
          ...state.seedContract,
          acceptanceCriteria: state.seedContract.acceptanceCriteria.map((criterion) => ({
            ...criterion,
          })),
          constraints: [...state.seedContract.constraints],
          outOfScope: [...state.seedContract.outOfScope],
          requiredArtifacts: state.seedContract.requiredArtifacts
            ? state.seedContract.requiredArtifacts.map((artifact) => ({ ...artifact }))
            : undefined,
          evidenceRequirements: state.seedContract.evidenceRequirements
            ? state.seedContract.evidenceRequirements.map((requirement) => ({ ...requirement }))
            : undefined,
          artifactSections: state.seedContract.artifactSections
            ? state.seedContract.artifactSections.map((section) => ({ ...section }))
            : undefined,
          executionIntent: state.seedContract.executionIntent
            ? {
                ...state.seedContract.executionIntent,
                capabilities: [...state.seedContract.executionIntent.capabilities],
                allowedTools: state.seedContract.executionIntent.allowedTools
                  ? [...state.seedContract.executionIntent.allowedTools]
                  : undefined,
                toolSequence: state.seedContract.executionIntent.toolSequence
                  ? [...state.seedContract.executionIntent.toolSequence]
                  : undefined,
                authorizedWriteTargets: state.seedContract.executionIntent.authorizedWriteTargets
                  ? [...state.seedContract.executionIntent.authorizedWriteTargets]
                  : undefined,
                protectedWriteTargets: state.seedContract.executionIntent.protectedWriteTargets
                  ? [...state.seedContract.executionIntent.protectedWriteTargets]
                  : undefined,
                requestedTerminalCommand: state.seedContract.executionIntent.requestedTerminalCommand
                  ? {
                      executable: state.seedContract.executionIntent.requestedTerminalCommand.executable,
                      args: [...state.seedContract.executionIntent.requestedTerminalCommand.args],
                    }
                  : undefined,
                requestedProcessStart: state.seedContract.executionIntent.requestedProcessStart
                  ? structuredClone(state.seedContract.executionIntent.requestedProcessStart)
                  : undefined,
                constrainedProcessStart: state.seedContract.executionIntent.constrainedProcessStart
                  ? structuredClone(state.seedContract.executionIntent.constrainedProcessStart)
                  : undefined,
              }
            : undefined,
        }
      : undefined,
    specialistRoute: state.specialistRoute,
    specialistReason: state.specialistReason,
    specialistBrief: state.specialistBrief,
    planIndex: state.planIndex,
    toolCalls: state.toolCalls.map(cloneToolCall),
    toolResults: state.toolResults.map((result) => ({ ...result })),
    recentToolResults: state.recentToolResults?.map((result) => ({ ...result })),
    memories: [...state.memories],
    output: state.output,
    codebaseExploration: state.codebaseExploration,
    codebaseMap: state.codebaseMap
      ? {
          generatedAt: state.codebaseMap.generatedAt,
          scoutPrompts: [...state.codebaseMap.scoutPrompts],
          summaries: state.codebaseMap.summaries.map((summary) => ({ ...summary })),
        }
      : undefined,
    evidenceLedger: cloneEvidenceLedger(state.evidenceLedger),
    toolRecommendationSummary: state.toolRecommendationSummary,
    analysisSummary: state.analysisSummary,
    findingsSummary: state.findingsSummary,
    implementationSummary: state.implementationSummary,
    implementationMutationEvidence: state.implementationMutationEvidence
      ? state.implementationMutationEvidence.map((evidence) => ({ ...evidence }))
      : undefined,
    implementationRetryRequested: state.implementationRetryRequested,
    implementationNoEditRetryCount: state.implementationNoEditRetryCount,
    noProgressRecoveryControllerEvidenceSignature:
      state.noProgressRecoveryControllerEvidenceSignature,
    noProgressRecoveryProviderCallCount:
      state.noProgressRecoveryProviderCallCount,
    implementationControllerFallbackTurnGranted:
      state.implementationControllerFallbackTurnGranted,
    implementationActionOnlyRecovery: state.implementationActionOnlyRecovery,
    implementationActionOnlyRecoveryAttempted: state.implementationActionOnlyRecoveryAttempted,
    implementationActionOnlyCorrectionCount: state.implementationActionOnlyCorrectionCount,
    implementationRecoveryActionPending: state.implementationRecoveryActionPending
      ? { ...state.implementationRecoveryActionPending }
      : undefined,
    implementationMutationHandoff: state.implementationMutationHandoff
      ? { ...state.implementationMutationHandoff }
      : undefined,
    implementationCausalObservation: state.implementationCausalObservation
      ? {
          tool: state.implementationCausalObservation.tool,
          input: cloneToolHistoryInput(state.implementationCausalObservation.input),
          ...(state.implementationCausalObservation.replayEvidence
            ? { replayEvidence: state.implementationCausalObservation.replayEvidence }
            : {}),
        }
      : undefined,
    implementationCausalTransition: state.implementationCausalTransition
      ? { ...state.implementationCausalTransition }
      : undefined,
    qualityConclusionRecoveryRequested: state.qualityConclusionRecoveryRequested,
    qualityConclusionResolvedPhase: state.qualityConclusionResolvedPhase,
    qualityConclusionRetryTarget: state.qualityConclusionRetryTarget,
    implementationToolHistoryStartIndex: state.implementationToolHistoryStartIndex,
    internalGraphContinuation: state.internalGraphContinuation,
    lastRejectedToolCallNames: state.lastRejectedToolCallNames
      ? [...state.lastRejectedToolCallNames]
      : undefined,
    implementationConvergenceNudgeCount: state.implementationConvergenceNudgeCount,
    implementationPreActionStallCount: state.implementationPreActionStallCount,
    implementationPostActionStallCount: state.implementationPostActionStallCount,
    implementationObservationReuseOnlyCount: state.implementationObservationReuseOnlyCount,
    implementationPendingTodoRetryCount: state.implementationPendingTodoRetryCount,
    implementationCompleteRequested: state.implementationCompleteRequested,
    verificationSummary: state.verificationSummary,
    researchRounds: state.researchRounds,
    researchSearchBaselineFingerprint: state.researchSearchBaselineFingerprint,
    researchSearchAddedEvidence: state.researchSearchAddedEvidence,
    researchRunToolCallCount: state.researchRunToolCallCount,
    researchRunToolCallBudget: state.researchRunToolCallBudget,
    researchPhaseToolCallCount: state.researchPhaseToolCallCount,
    researchPhaseToolHistoryBaseline: state.researchPhaseToolHistoryBaseline,
    deepAnswerReviseRequested: state.deepAnswerReviseRequested,
    deepAnswerReviseCount: state.deepAnswerReviseCount,
    validationSummary: state.validationSummary,
    validationToolHistoryStartIndex: state.validationToolHistoryStartIndex,
    validationPhaseToolHistoryStartIndex: state.validationPhaseToolHistoryStartIndex,
    validationConvergenceNudgeCount: state.validationConvergenceNudgeCount,
    reviewSummary: state.reviewSummary,
    qualityGateDecision: state.qualityGateDecision,
    qualityGateSummary: state.qualityGateSummary,
    backtrackCount: state.backtrackCount,
    backtrackReason: state.backtrackReason,
    backtrackReasons: state.backtrackReasons ? [...state.backtrackReasons] : undefined,
    todoList: state.todoList ? state.todoList.map((item) => ({ ...item })) : undefined,
    computerUseObservationIds: state.computerUseObservationIds
      ? [...state.computerUseObservationIds]
      : undefined,
    currentEditCheckpointId: state.currentEditCheckpointId,
    editRollbacks: state.editRollbacks
      ? state.editRollbacks.map((rollback) => ({
          checkpointId: rollback.checkpointId,
          reason: rollback.reason,
          files: [...rollback.files],
          revertedAt: rollback.revertedAt,
        }))
      : undefined,
    debateRounds: state.debateRounds
      ? state.debateRounds.map((round) => ({ ...round }))
      : undefined,
    plannerWorkingMemory: state.plannerWorkingMemory
      ? { ...state.plannerWorkingMemory }
      : undefined,
    subgraphState: state.subgraphState
      ? {
          node: state.subgraphState.node,
          state: cloneGraphState(
            state.subgraphState.state,
            forCheckpoint ? { forCheckpoint, depth: depth + 1 } : undefined,
          ),
        }
      : undefined,
    totalUsage: { ...state.totalUsage },
    validationOutcome: state.validationOutcome
      ? {
          command: state.validationOutcome.command,
          exitCode: state.validationOutcome.exitCode,
          passed: state.validationOutcome.passed,
          failedSignals: [...state.validationOutcome.failedSignals],
          rawOutput: state.validationOutcome.rawOutput,
        }
      : undefined,
    postEditFindings: state.postEditFindings
      ? {
          editedFiles: [...state.postEditFindings.editedFiles],
          impactedExternalModules: [...state.postEditFindings.impactedExternalModules],
          impactedLocalModules: [...state.postEditFindings.impactedLocalModules],
          reverseCallers: [...state.postEditFindings.reverseCallers],
          diagnostics: state.postEditFindings.diagnostics.map((diagnostic) => ({
            file: diagnostic.file,
            summary: diagnostic.summary,
          })),
          analyzedAt: state.postEditFindings.analyzedAt,
        }
      : undefined,
    compressedHistorySummary: state.compressedHistorySummary,
    compressedHistoryUpToIndex: state.compressedHistoryUpToIndex,
    phaseUsages: state.phaseUsages
      ? Object.fromEntries(
          Object.entries(state.phaseUsages).map(([phase, usage]) => [phase, { ...usage }]),
        )
      : undefined,
    phaseUsageStart: state.phaseUsageStart
      ? {
          phase: state.phaseUsageStart.phase,
          usage: { ...state.phaseUsageStart.usage },
        }
      : undefined,
    reflectionMemo: state.reflectionMemo ? [...state.reflectionMemo] : undefined,
    toolCallHistory: state.toolCallHistory
      ? state.toolCallHistory.map((entry) => ({
          ...(entry.toolCallId ? { toolCallId: entry.toolCallId } : {}),
          tool: entry.tool,
          input: cloneToolHistoryInput(entry.input),
          status: entry.status,
          ...(typeof entry.executionObserved === 'boolean'
            ? { executionObserved: entry.executionObserved }
            : {}),
          ...(entry.securityEffect ? { securityEffect: entry.securityEffect } : {}),
          ...(entry.executionPosture
            ? { executionPosture: structuredClone(entry.executionPosture) }
            : {}),
          ...(entry.failureCode ? { failureCode: entry.failureCode } : {}),
          ts: entry.ts,
          ...(entry.outputFingerprint
            ? { outputFingerprint: entry.outputFingerprint }
            : {}),
          ...(typeof entry.output === 'string' ? { output: entry.output } : {}),
        }))
      : undefined,
    iteration: state.iteration,
    maxIterations: state.maxIterations,
    shouldStop: state.shouldStop,
    taskType: state.taskType,
  }
}

export function createGraphApprovalCheckpoint(
  requestId: string,
  state: AgentState,
  context: GraphExecutionContext,
  toolCalls: readonly import('@sepilotd/core').ToolCall[],
  currentToolIndex: number,
): ApprovalRunCheckpoint {
  return {
    requestId,
    sessionId: context.agentContext.sessionId,
    provider: context.agentContext.provider,
    model: context.agentContext.model,
    mode: context.graphId,
    modeControlState: context.modeControl ? structuredClone(context.modeControl.state) : undefined,
    systemPrompt: context.systemPrompt ?? context.agentContext.systemPrompt,
    cwd: context.agentContext.cwd,
    workspaceRoot: context.agentContext.workspaceRoot,
    workspaceIsolation: context.agentContext.workspaceIsolation,
    scopeTags: cloneCheckpointScopeTags(context.agentContext.scopeTags),
    executionSkillIds: cloneCheckpointExecutionSkillIds(
      context.agentContext.executionSkillIds,
    ),
    skillToolNames: cloneCheckpointSkillToolNames(
      context.agentContext.skillToolNames,
    ),
    toolAllowlist: cloneCheckpointToolAllowlist(context.agentContext.toolAllowlist),
    skillExecutionPolicies: cloneCheckpointSkillExecutionPolicies(
      context.agentContext.skillExecutionPolicies,
    ),
    requireToolApproval: context.agentContext.requireToolApproval,
    runContract: state.seedContract,
    messages: state.messages.map(cloneMessage),
    toolCalls: toolCalls.map(cloneToolCall),
    currentToolIndex,
    totalUsage: { ...state.totalUsage },
    iteration: state.iteration,
    maxIterations: state.maxIterations,
    thinkingLevel: context.thinkingLevel,
    ...(context.textDeltaMode === 'live' ? { textDeltaMode: 'live' as const } : {}),
    graphState: cloneGraphState(state, { forCheckpoint: true }),
    createdAt: new Date().toISOString(),
  }
}

export function createGraphRunCheckpoint(
  state: AgentState,
  context: GraphExecutionContext,
  stage: RunResumeStage,
  pendingToolExecution?: PendingToolExecution,
): SessionRunCheckpoint {
  return {
    sessionId: context.agentContext.sessionId,
    provider: context.agentContext.provider,
    model: context.agentContext.model,
    mode: context.graphId,
    modeControlState: context.modeControl ? structuredClone(context.modeControl.state) : undefined,
    systemPrompt: context.systemPrompt ?? context.agentContext.systemPrompt,
    cwd: context.agentContext.cwd,
    workspaceRoot: context.agentContext.workspaceRoot,
    workspaceIsolation: context.agentContext.workspaceIsolation,
    scopeTags: cloneCheckpointScopeTags(context.agentContext.scopeTags),
    executionSkillIds: cloneCheckpointExecutionSkillIds(
      context.agentContext.executionSkillIds,
    ),
    skillToolNames: cloneCheckpointSkillToolNames(
      context.agentContext.skillToolNames,
    ),
    toolAllowlist: cloneCheckpointToolAllowlist(context.agentContext.toolAllowlist),
    skillExecutionPolicies: cloneCheckpointSkillExecutionPolicies(
      context.agentContext.skillExecutionPolicies,
    ),
    requireToolApproval: context.agentContext.requireToolApproval,
    runContract: state.seedContract,
    messages: state.messages.map(cloneMessage),
    totalUsage: { ...state.totalUsage },
    iteration: state.iteration,
    maxIterations: state.maxIterations,
    thinkingLevel: context.thinkingLevel,
    ...(context.textDeltaMode === 'live' ? { textDeltaMode: 'live' as const } : {}),
    stage,
    checkpointedAt: new Date().toISOString(),
    pendingToolExecution: pendingToolExecution
      ? {
          toolCalls: pendingToolExecution.toolCalls.map(cloneToolCall),
          startIndex: pendingToolExecution.startIndex,
          batchSize: pendingToolExecution.batchSize,
          currentExecutionId: pendingToolExecution.currentExecutionId,
        }
      : undefined,
    graphState: cloneGraphState(state, { forCheckpoint: true }),
  }
}
