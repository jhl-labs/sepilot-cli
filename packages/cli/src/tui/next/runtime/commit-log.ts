import type { Message } from '../../types.js'

export interface CommitLog {
  committed: Set<string>
}

export interface CommitSelection {
  commits: Message[]
  live: Message[]
}

export function createCommitLog(): CommitLog {
  return { committed: new Set<string>() }
}

export function selectCommits(
  log: CommitLog,
  messages: Message[],
  isStreaming: boolean,
): CommitSelection {
  const presentIds = new Set(messages.map(({ id }) => id))
  for (const id of log.committed) {
    if (!presentIds.has(id)) log.committed.delete(id)
  }

  const latestUserIndex = messages.findLastIndex(({ role }) => role === 'user')
  const latestTurnToolIndex = messages.findLastIndex((message, index) => (
    message.role === 'tool' && index > latestUserIndex
  ))
  const liveTailStart = latestTurnToolIndex >= 0 ? latestTurnToolIndex : -1

  const commits: Message[] = []
  const live: Message[] = []
  for (let index = 0; index < messages.length; index += 1) {
    const current = messages[index]
    if (current.variant === 'thinking') {
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
      && index === messages.length - 1
      && current.role !== 'user'
    const latestToolTail = liveTailStart >= 0 && index >= liveTailStart
    if (mutableTool || trailingLiveMessage || latestToolTail) {
      live.push(current)
      continue
    }
    log.committed.add(current.id)
    commits.push(current)
  }

  return { commits, live }
}
