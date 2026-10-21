import type { AgentCoworkTaskResult } from '../graph/types.js'

export const MAX_COWORK_RESULTS = 40

export function resolveCoworkResultsCap(): number {
  const raw = process.env.SEPILOTD_COWORK_RESULTS_CAP
  if (!raw) return MAX_COWORK_RESULTS
  const parsed = Number.parseInt(raw, 10)
  if (!Number.isFinite(parsed)) return MAX_COWORK_RESULTS
  return Math.max(5, Math.min(500, parsed))
}

function digestFoldedCount(result: string): number {
  const match = result.match(/(\d+)/)
  return match ? Number(match[1]) : 1
}

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
    status: 'complete',
    result: `${foldedCount} earlier tasks folded`,
    recordedAt: new Date().toISOString(),
    digest: true,
  }
  return [digest, ...kept]
}
