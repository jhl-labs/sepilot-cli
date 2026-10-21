import type { AgentState } from './types.js'

const MAX_SLICE_LIST_ITEMS = 8

export interface ScopedBoardSiblingEntry {
  origin: { sessionId: string; category?: string }
  kind: 'failed_attempt' | 'open_question' | 'evidence'
  payload: Record<string, unknown>
}

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
