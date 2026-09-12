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
  parent.evidenceLedger = childClone.evidenceLedger
  parent.backtrackCount = childClone.backtrackCount
  parent.backtrackReason = childClone.backtrackReason
  parent.backtrackReasons = childClone.backtrackReasons
  parent.qualityGateSummary = childClone.qualityGateSummary
  parent.toolCalls = []
  parent.toolResults = []
  parent.output = ''
  // User-owned prerequisites and denied approvals close the whole run, unlike
  // a child's local iteration limit. Do not let a sibling phase continue or
  // discard the explanation when merging its evidence.
  if (childClone.userActionRequired || childClone.approvalDenied) {
    parent.userActionRequired = childClone.userActionRequired
    parent.approvalDenied = childClone.approvalDenied
    parent.stopReason = childClone.stopReason
    parent.shouldStop = true
    parent.output = childClone.output
  }
  // An explicitly closed exact-call workflow is a run-wide user boundary,
  // not phase-local subgraph state. Once a child executor has recorded every
  // permitted outcome, the parent must not reopen a verification or discovery
  // tool surface. Other forced-final reasons remain local to their child
  // phase and are intentionally not promoted here.
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
