import type { Message } from '../../types.js'

const REPEATABLE_OBSERVATIONS = new Set(['fs.read', 'fs.list', 'fs.glob', 'fs.search', 'git.status', 'git.diff'])

function canonicalValue(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalValue).join(',')}]`
  if (value !== null && typeof value === 'object') {
    return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
      .map(([key, item]) => `${JSON.stringify(key)}:${canonicalValue(item)}`).join(',')}}`
  }
  return JSON.stringify(value) ?? 'undefined'
}

export function sameObservationRequest(left: Message, right: Message): boolean {
  const a = left.toolCall
  const b = right.toolCall
  return left.role === 'tool' && right.role === 'tool'
    && a !== undefined && b !== undefined
    && REPEATABLE_OBSERVATIONS.has(a.name)
    && a.name === b.name
    && !a.approvalRequestId && !b.approvalRequestId
    && !a.approvalState && !b.approvalState
    && canonicalValue(a.input) === canonicalValue(b.input)
}

export function conversationPresentation(messages: Message[], isStreaming: boolean, sealedRepeatCounts?: ReadonlyMap<string, number>): Message[] {
  const latestUser = messages.findLastIndex(({ role }) => role === 'user')
  const latestProgress = isStreaming
    ? messages.findLastIndex((message, index) => index > latestUser && message.variant === 'progress')
    : -1
  const rows: Message[] = []
  for (const message of messages) {
    if (message.variant === 'progress') continue
    const previous = rows.at(-1)
    if (previous && sameObservationRequest(previous, message)
      && (previous.toolCall?.repeatCount ?? 1) < (sealedRepeatCounts?.get(previous.id) ?? Infinity)
      && previous.toolCall?.status === 'success' && message.toolCall?.status === 'success'
      && typeof previous.toolCall.output === 'string'
      && previous.toolCall.output === message.toolCall.output
      && previous.toolCall.meta === message.toolCall.meta
      && previous.content === message.content) {
      rows[rows.length - 1] = {
        ...previous,
        toolCall: { ...previous.toolCall, repeatCount: (previous.toolCall.repeatCount ?? 1) + 1 },
      }
    } else {
      rows.push(message)
    }
  }
  if (latestProgress >= 0) rows.push(messages[latestProgress])
  return rows
}
