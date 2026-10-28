import type { Message } from '../../types.js'
import { conversationPresentation, sameObservationRequest } from './conversation-presentation.js'

export interface CommitLog {
  committed: Set<string>
  committedRepeatCounts: Map<string, number>
}

export interface CommitSelection {
  commits: Message[]
  live: Message[]
}

export function createCommitLog(): CommitLog {
  return { committed: new Set<string>(), committedRepeatCounts: new Map<string, number>() }
}

export function selectCommits(
  log: CommitLog,
  messages: Message[],
  isStreaming: boolean,
): CommitSelection {
  messages = conversationPresentation(messages, isStreaming, log.committedRepeatCounts)
  const presentIds = new Set(messages.map(({ id }) => id))
  for (const id of log.committed) {
    if (!presentIds.has(id)) {
      log.committed.delete(id)
      log.committedRepeatCounts.delete(id)
    }
  }

  const latestUserIndex = messages.findLastIndex(({ role }) => role === 'user')
  const latestContentIndex = messages.findLastIndex(({ variant }) => variant !== 'progress' && variant !== 'thinking')
  const latestTurnToolIndex = messages.findLastIndex((message, index) => (
    message.role === 'tool' && index > latestUserIndex
  ))
  let liveTailStart = latestTurnToolIndex >= 0 ? latestTurnToolIndex : -1
  if (liveTailStart > 0) {
    const latest = messages[liveTailStart]
    const previous = messages[liveTailStart - 1]
    if (latest.toolCall?.status === 'running'
      && previous.toolCall?.status === 'success'
      && sameObservationRequest(previous, latest)) liveTailStart -= 1
  }

  const commits: Message[] = []
  const live: Message[] = []
  for (let index = 0; index < messages.length; index += 1) {
    const current = messages[index]
    if (current.variant === 'thinking' || current.variant === 'progress') {
      live.push(current)
      continue
    }
    if (log.committed.has(current.id)) {
      commits.push(current)
      continue
    }
    const mutableTool = current.role === 'tool'
      && (current.toolCall?.status === 'running' || current.toolCall?.status === 'pending')
    const trailingLiveMessage = isStreaming
      && index === latestContentIndex
      && current.role !== 'user'
    const latestToolTail = isStreaming && liveTailStart >= 0 && index >= liveTailStart
    if (mutableTool || trailingLiveMessage || latestToolTail) {
      live.push(current)
      continue
    }
    log.committed.add(current.id)
    if (current.toolCall) log.committedRepeatCounts.set(current.id, current.toolCall.repeatCount ?? 1)
    commits.push(current)
  }

  return { commits, live }
}
