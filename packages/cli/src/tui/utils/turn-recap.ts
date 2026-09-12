import type { ToolCallState } from '../types.js'

/** Snapshot taken when a run starts, so the recap covers only this turn. */
export interface TurnRecapSnapshot {
  startedAt: number
  toolCallCount: number
}

const FILE_WRITE_TOOLS = new Set([
  'fs.write',
  'fs.edit',
  'fs.append',
  'apply_patch',
  'doc.replace_section',
  'doc.replace_range',
  'doc.insert_after_section',
  'doc.append',
  'doc.rewrite',
])

const MAX_RECAP_FILES = 6
const MAX_RECAP_TOOLS = 6

export function formatRecapDuration(ms: number): string {
  if (ms < 1000) return '<1s'
  const totalSeconds = Math.round(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes === 0) return `${seconds}s`
  const hours = Math.floor(minutes / 60)
  if (hours === 0) return `${minutes}m ${seconds}s`
  return `${hours}h ${minutes % 60}m`
}

function extractFilePaths(call: ToolCallState): string[] {
  if (!FILE_WRITE_TOOLS.has(call.name)) return []
  const input = call.input ?? {}
  const candidate = input.path ?? input.file_path ?? input.filePath ?? input.file
  if (typeof candidate === 'string' && candidate.trim()) return [candidate.trim()]

  if (call.name === 'apply_patch' && typeof input.patch === 'string') {
    const paths: string[] = []
    for (const match of input.patch.matchAll(/^\*\*\* (?:Add|Update|Delete) File:\s*(.+)$/gm)) {
      const path = match[1]?.trim()
      if (path) paths.push(path)
    }
    return paths
  }

  return []
}

function formatTokens(count: number): string {
  return count >= 10_000 ? `${(count / 1000).toFixed(1)}k` : String(count)
}

/**
 * Builds a compact end-of-turn recap. Duration is always reported so users
 * get a clear terminal marker even when a provider omits usage and finalized
 * tool-call metadata.
 */
export function buildTurnRecap(
  snapshot: TurnRecapSnapshot,
  toolCalls: ToolCallState[],
  usage: { input: number; output: number; cost: number } | null,
  endedAt: number,
): string {
  const turnCalls = toolCalls.slice(snapshot.toolCallCount)
  // Daemon `done.usage` is scoped to one run, and chatReducer replaces usage
  // with that latest assistant value. Subtracting the prior turn therefore
  // makes a smaller second turn appear as zero tokens.
  const inputTokens = Math.max(0, usage?.input ?? 0)
  const outputTokens = Math.max(0, usage?.output ?? 0)
  const turnCost = Math.max(0, usage?.cost ?? 0)
  const parts: string[] = [`Recap · ${formatRecapDuration(endedAt - snapshot.startedAt)}`]

  if (turnCalls.length > 0) {
    const counts = new Map<string, number>()
    for (const call of turnCalls) {
      counts.set(call.name, (counts.get(call.name) ?? 0) + 1)
    }
    const errorCount = turnCalls.filter((call) => call.status === 'error').length
    const toolSummary = [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, MAX_RECAP_TOOLS)
      .map(([name, count]) => (count > 1 ? `${name}×${count}` : name))
      .join(', ')
    const overflow = counts.size > MAX_RECAP_TOOLS ? ', …' : ''
    parts.push(
      `${turnCalls.length} tool call${turnCalls.length === 1 ? '' : 's'}${
        errorCount > 0 ? ` (${errorCount} failed)` : ''
      }: ${toolSummary}${overflow}`,
    )

    const files = [...new Set(turnCalls.flatMap(extractFilePaths))]
    if (files.length > 0) {
      const shown = files.slice(0, MAX_RECAP_FILES).join(', ')
      const more = files.length > MAX_RECAP_FILES ? ` +${files.length - MAX_RECAP_FILES} more` : ''
      parts.push(`files: ${shown}${more}`)
    }
  }

  if (inputTokens > 0 || outputTokens > 0) {
    const cost = turnCost > 0 ? ` · $${turnCost.toFixed(4)}` : ''
    parts.push(`tokens ${formatTokens(inputTokens)} in / ${formatTokens(outputTokens)} out${cost}`)
  }

  return parts.join('\n  ')
}
