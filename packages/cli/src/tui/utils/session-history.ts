import type { SessionEvent } from '@sepilotd/api-client'

export interface RewindTarget {
  turns: number
  fromEventIndex: number
  copiedEvents: number
  omittedEvents: number
  userPreview: string
  timestamp: string | null
}

function collectUserMessageEventIndexes(events: SessionEvent[]): number[] {
  return events.flatMap((event, index) => (
    event.type === 'user_message' ? [index] : []
  ))
}

function normalizeTurns(turns = 1): number {
  return Number.isFinite(turns)
    ? Math.max(1, Math.floor(turns))
    : 1
}

function previewUserMessage(content: string, maxLength = 88): string {
  const normalized = content.replace(/\s+/g, ' ').trim()
  if (!normalized) {
    return '(empty user message)'
  }
  return normalized.length > maxLength
    ? `${normalized.slice(0, Math.max(0, maxLength - 1))}…`
    : normalized
}

export function findRewindEventIndex(
  events: SessionEvent[],
  turns = 1,
): number | null {
  const normalizedTurns = normalizeTurns(turns)
  const userMessageIndexes = collectUserMessageEventIndexes(events)

  if (userMessageIndexes.length === 0) {
    return null
  }

  if (normalizedTurns >= userMessageIndexes.length) {
    return 0
  }

  return userMessageIndexes[userMessageIndexes.length - normalizedTurns] ?? 0
}

export function findRewindTarget(
  events: SessionEvent[],
  turns = 1,
): RewindTarget | null {
  const normalizedTurns = normalizeTurns(turns)
  const userMessageIndexes = collectUserMessageEventIndexes(events)

  if (userMessageIndexes.length === 0) {
    return null
  }

  const targetUserIndex = normalizedTurns >= userMessageIndexes.length
    ? userMessageIndexes[0]!
    : userMessageIndexes[userMessageIndexes.length - normalizedTurns]!
  const fromEventIndex = normalizedTurns >= userMessageIndexes.length
    ? 0
    : targetUserIndex
  const targetEvent = events[targetUserIndex]

  return {
    turns: normalizedTurns,
    fromEventIndex,
    copiedEvents: Math.min(fromEventIndex, events.length),
    omittedEvents: Math.max(0, events.length - fromEventIndex),
    userPreview: targetEvent?.type === 'user_message'
      ? previewUserMessage(targetEvent.content)
      : 'selected user turn',
    timestamp: targetEvent?.timestamp ?? null,
  }
}

export function listRewindTargets(
  events: SessionEvent[],
  maxTargets = 5,
): RewindTarget[] {
  const targetCount = Number.isFinite(maxTargets)
    ? Math.max(0, Math.floor(maxTargets))
    : 5
  const availableTurns = collectUserMessageEventIndexes(events).length

  return Array.from({ length: Math.min(targetCount, availableTurns) }, (_, index) => (
    findRewindTarget(events, index + 1)
  )).filter((target): target is RewindTarget => target !== null)
}

export function describeRewindTurns(turns: number): string {
  return `${turns} turn${turns === 1 ? '' : 's'}`
}

export function findAdjacentRecentSessionId(
  recentSessionIds: string[],
  currentSessionId: string | null,
  direction: 'older' | 'newer',
): string | null {
  const ordered = Array.from(new Set(recentSessionIds.filter(Boolean)))
  if (ordered.length === 0) {
    return null
  }

  if (!currentSessionId) {
    return direction === 'older' ? ordered[0] ?? null : null
  }

  const currentIndex = ordered.indexOf(currentSessionId)
  if (currentIndex === -1) {
    return direction === 'older' ? ordered[0] ?? null : null
  }

  return direction === 'older'
    ? ordered[currentIndex + 1] ?? null
    : ordered[currentIndex - 1] ?? null
}
