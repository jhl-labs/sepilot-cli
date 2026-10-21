import type { AgentState } from './types.js'
import { cloneGraphState } from './checkpoints.js'

export function childStateFrom(
  parent: AgentState,
  overrides: Partial<AgentState>,
): AgentState {
  return {
    ...cloneGraphState(parent),
    currentStep: '',
    planIndex: 0,
    toolCalls: [],
    toolResults: [],
    output: '',
    iteration: 0,
    shouldStop: false,
    budgetExhausted: undefined,
    qualityGateDecision: undefined,
    ...overrides,
  }
}

export function mergeChildInto(
  parent: AgentState,
  child: AgentState,
  opts: { keepParentContract?: boolean } = {},
): AgentState {
  const parentSeedContract = parent.seedContract
  const childClone = cloneGraphState(child)

  parent.messages = childClone.messages
  parent.effectiveContextWindowTokens = childClone.effectiveContextWindowTokens
    ?? parent.effectiveContextWindowTokens
  parent.memories = childClone.memories
  parent.totalUsage = childClone.totalUsage
  parent.recentToolResults = childClone.recentToolResults
  parent.toolCallHistory = (childClone.toolCallHistory ?? []).slice(-200)
  parent.unexecutedToolSelections = childClone.unexecutedToolSelections
  parent.evidenceLedger = childClone.evidenceLedger
  parent.workProgress = childClone.workProgress ?? parent.workProgress
  parent.recoveryProviderCallTotal = Math.max(parent.recoveryProviderCallTotal ?? 0, childClone.recoveryProviderCallTotal ?? 0)
  parent.recoveryControllerFailureTotal = Math.max(parent.recoveryControllerFailureTotal ?? 0, childClone.recoveryControllerFailureTotal ?? 0)
  parent.recoveryExhaustedFinalCount = Math.max(parent.recoveryExhaustedFinalCount ?? 0, childClone.recoveryExhaustedFinalCount ?? 0)
  parent.backtrackCount = childClone.backtrackCount
  parent.backtrackReason = childClone.backtrackReason
  parent.backtrackReasons = childClone.backtrackReasons
  parent.qualityGateSummary = childClone.qualityGateSummary
  parent.toolCalls = []
  parent.toolResults = []
  parent.output = ''
  if (childClone.userActionRequired || childClone.approvalDenied) {
    parent.userActionRequired = childClone.userActionRequired
    parent.approvalDenied = childClone.approvalDenied
    parent.stopReason = childClone.stopReason
    parent.shouldStop = true
    parent.output = childClone.output
  }
  if (
    childClone.stuckRepeatForcedFinal === true
    && childClone.forcedFinalSynthesisReason === 'exact-tool-budget'
  ) {
    parent.stuckRepeatForcedFinal = true
    parent.forcedFinalSynthesisReason = 'exact-tool-budget'
  }
  if (opts.keepParentContract !== false) {
    parent.seedContract = parentSeedContract
  } else {
    parent.seedContract = childClone.seedContract
  }
  return parent
}
