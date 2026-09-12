import type { AgentState } from './types.js'

const MAX_SLICE_LIST_ITEMS = 8

/**
 * Minimal structural view of a shared-board entry, declared locally so this
 * pure graph module does not import the server/runtime SharedBoard type.
 */
export interface ScopedBoardSiblingEntry {
  origin: { sessionId: string; category?: string }
  kind: 'failed_attempt' | 'open_question' | 'evidence'
  payload: Record<string, unknown>
}

/**
 * Render the scoped board slice a subagent should start from: the run goal, its
 * acceptance criteria, open questions, and — for categories where repeating a
 * dead-end is the risk — the parent's known failed attempts (agent-1hour's
 * "block failures across agents"). This is a deterministic text block built from
 * structured board fields only; it is a *slice*, never the parent's full state
 * (no messages, no raw tool output), so depth-1 isolation and context hygiene
 * are preserved. PLAN_065 T3.
 */
export function buildScopedBoardSlice(
  parent: Pick<AgentState, 'seedContract' | 'failedAttempts' | 'openQuestions'>,
  opts: { category: string; siblingEntries?: ScopedBoardSiblingEntry[] },
): string {
  const sections: string[] = []

  const goal = parent.seedContract?.summary?.trim()
  if (goal) {
    sections.push(`Goal: ${goal}`)
  }

  const criteria = parent.seedContract?.acceptanceCriteria ?? []
  if (criteria.length > 0) {
    sections.push(
      [
        'Acceptance criteria:',
        ...criteria
          .slice(0, MAX_SLICE_LIST_ITEMS)
          .map((criterion) => `- ${criterion.id}: ${criterion.text}`),
      ].join('\n'),
    )
  }

  const openQuestions = parent.openQuestions ?? []
  if (openQuestions.length > 0) {
    sections.push(
      [
        'Open questions:',
        ...openQuestions
          .slice(0, MAX_SLICE_LIST_ITEMS)
          .map((question) => `- ${question.blocking ? '[blocking] ' : ''}${question.text}`),
      ].join('\n'),
    )
  }

  // Explore-style read-only scouts do not act, so parent dead-ends are not
  // relevant to them; acting categories (coder/general/…) inherit the failed
  // attempts so they do not re-run a known-bad approach.
  const includeFailedAttempts = opts.category !== 'explore'
  const failedAttempts = parent.failedAttempts ?? []
  if (includeFailedAttempts && failedAttempts.length > 0) {
    sections.push(
      [
        'Known failed attempts (do not repeat these approaches):',
        ...failedAttempts
          .slice(-MAX_SLICE_LIST_ITEMS)
          .map((attempt) => `- ${attempt.tool}: ${attempt.reason}`),
      ].join('\n'),
    )
  }

  // Findings from concurrent sibling subagents under the same parent run, so a
  // subagent inherits a sibling's dead-end/open-question instead of
  // rediscovering it (opencode shared-block). Best-effort: may be empty when
  // this subagent starts before siblings have reported.
  const siblingLines = renderSiblingEntries(opts.siblingEntries ?? [], opts.category)
  if (siblingLines.length > 0) {
    sections.push(['Findings from sibling subagents:', ...siblingLines].join('\n'))
  }

  if (sections.length === 0) {
    return ''
  }

  return ['[Shared board — scoped slice from parent run]', ...sections].join('\n\n')
}

function renderSiblingEntries(
  entries: ScopedBoardSiblingEntry[],
  category: string,
): string[] {
  const includeFailedAttempts = category !== 'explore'
  const lines: string[] = []
  for (const entry of entries.slice(-MAX_SLICE_LIST_ITEMS)) {
    if (entry.kind === 'failed_attempt') {
      if (!includeFailedAttempts) continue
      const tool = typeof entry.payload.tool === 'string' ? entry.payload.tool : 'tool'
      const reason = typeof entry.payload.reason === 'string' ? entry.payload.reason : 'failed'
      lines.push(`- [failed] ${tool}: ${reason}`)
    } else if (entry.kind === 'open_question') {
      const text = typeof entry.payload.text === 'string' ? entry.payload.text : ''
      if (text) lines.push(`- [question] ${text}`)
    }
  }
  return lines
}
