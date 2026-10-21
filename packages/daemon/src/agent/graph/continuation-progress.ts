import type { AgentState, ContinuationProgressSnapshot } from './types.js'
import { observeWorkProgress } from '../work-progress.js'
import { parseRunWallClockBudgetMs } from '../progress-signal.js'

export function captureContinuationProgress(
  state: Pick<AgentState, 'evidenceLedger' | 'toolCallHistory' | 'workProgress'>,
): ContinuationProgressSnapshot {
  state.workProgress = (state.toolCallHistory ?? []).reduce(
    (progress, entry) => observeWorkProgress(progress, entry),
    state.workProgress ?? { revision: 0, seen: [] },
  )
  return {
    semanticRevision: state.workProgress.revision,
    verifiedEvidence: 0,
    executedToolCalls: 0,
    evidenceOrder: 0,
    artifactMutations: 0,
  }
}

export function hasContinuationProgress(
  previous: ContinuationProgressSnapshot | undefined,
  current: ContinuationProgressSnapshot,
): boolean {
  if (!previous) return true
  return (current.semanticRevision ?? 0) > (previous.semanticRevision ?? 0)
}

export function resolveRunMaxWallMs(
  raw: string | undefined = process.env.SEPILOTD_RUN_MAX_WALL_MS,
): number | undefined {
  return parseRunWallClockBudgetMs(raw)
}
