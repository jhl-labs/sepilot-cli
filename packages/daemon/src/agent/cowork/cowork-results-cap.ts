import type { AgentCoworkTaskResult } from '../graph/types.js'

/**
 * Default cap for `coworkTaskResults`. Generous — only large orchestrations
 * with many subtasks ever hit it. Overridable via `SEPILOTD_COWORK_RESULTS_CAP`.
 */
export const MAX_COWORK_RESULTS = 40

/**
 * Resolve the effective cowork-results cap. Defaults to `MAX_COWORK_RESULTS`;
 * `SEPILOTD_COWORK_RESULTS_CAP` overrides it (clamped to a safe range). General
 * knob — no model/dataset/graphId branching.
 */
export function resolveCoworkResultsCap(): number {
  const raw = process.env.SEPILOTD_COWORK_RESULTS_CAP
  if (!raw) return MAX_COWORK_RESULTS
  const parsed = Number.parseInt(raw, 10)
  if (!Number.isFinite(parsed)) return MAX_COWORK_RESULTS
  return Math.max(5, Math.min(500, parsed))
}

/** Read the folded count we previously stamped into a rollup digest entry. */
function digestFoldedCount(result: string): number {
  const match = result.match(/(\d+)/)
  return match ? Number(match[1]) : 1
}

/**
 * Rolling FIFO cap for cowork task results. Structure-only: the decision uses
 * array length and the explicit `digest` flag — never content meaning. Keeps
 * the most recent `max` full entries; older ones fold into a single
 * `{ digest: true }` summary kept at the front. A pre-existing digest
 * contributes its recorded count so the running total survives repeated caps.
 */
export function capCoworkResults(
  results: AgentCoworkTaskResult[],
  max: number = resolveCoworkResultsCap(),
): AgentCoworkTaskResult[] {
  if (results.length <= max) return results
  const overflow = results.length - max
  const folded = results.slice(0, overflow)
  const kept = results.slice(overflow)
  let foldedCount = 0
  for (const entry of folded) {
    foldedCount += entry.digest ? digestFoldedCount(entry.result) : 1
  }
  const digest: AgentCoworkTaskResult = {
    sequence: 0,
    planIndex: kept[0]?.planIndex ?? 0,
    role: 'digest',
    instruction: 'folded cowork task results',
    // 'complete' so the digest is never treated as an unresolved blocker.
    status: 'complete',
    result: `${foldedCount} earlier tasks folded`,
    recordedAt: new Date().toISOString(),
    digest: true,
  }
  return [digest, ...kept]
}
