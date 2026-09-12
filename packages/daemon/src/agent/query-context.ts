import type { Message } from '@sepilotd/core'

export interface RetrievalQueryOptions {
  minQueryChars?: number
  recentMessages?: number
  previousMessageCharLimit?: number
}

const DEFAULT_MIN_QUERY_CHARS = 8
const DEFAULT_RECENT_MESSAGES = 6
const DEFAULT_PREVIOUS_MESSAGE_CHARS = 240
const FOLLOWUP_RE =
  /(?:\b(?:it|that|this|those|these|them|again|same|above|previous|earlier)\b|그거|그것|이거|이것|저거|저것|다시|위에|이전|앞서|계속)/iu

function messageText(message: Message): string {
  if (typeof message.content === 'string') {
    return message.content
  }
  if (Array.isArray(message.content)) {
    return message.content
      .map((part) => {
        if (part && typeof part === 'object' && 'text' in part && typeof part.text === 'string') {
          return part.text
        }
        return ''
      })
      .filter(Boolean)
      .join(' ')
  }
  return ''
}

function normalizeQueryText(value: string): string {
  return value.trim().replace(/\s+/g, ' ')
}

function truncateQueryText(value: string, maxChars: number): string {
  const normalized = normalizeQueryText(value)
  if (normalized.length <= maxChars) return normalized
  return normalized.slice(0, maxChars).trimEnd()
}

function isFollowupQuery(query: string, minQueryChars: number): boolean {
  const normalized = normalizeQueryText(query)
  return normalized.length < minQueryChars || FOLLOWUP_RE.test(normalized)
}

function latestPreviousUserMessage(
  previousMessages: Message[],
  limit: number,
): string | undefined {
  for (const message of previousMessages.slice(-limit).reverse()) {
    if (message.role !== 'user') continue
    const text = normalizeQueryText(messageText(message))
    if (text.length >= 3) return text
  }
  return undefined
}

export function buildRetrievalQuery(
  latest: string,
  previousMessages: Message[] = [],
  options: RetrievalQueryOptions = {},
): string {
  const query = normalizeQueryText(latest)
  if (!query) return ''

  const minQueryChars = options.minQueryChars ?? DEFAULT_MIN_QUERY_CHARS
  if (!isFollowupQuery(query, minQueryChars)) {
    return query
  }

  const previous = latestPreviousUserMessage(
    previousMessages,
    options.recentMessages ?? DEFAULT_RECENT_MESSAGES,
  )
  if (!previous || normalizeQueryText(previous) === query) {
    return query
  }

  return [
    truncateQueryText(previous, options.previousMessageCharLimit ?? DEFAULT_PREVIOUS_MESSAGE_CHARS),
    query,
  ].join('\n')
}
