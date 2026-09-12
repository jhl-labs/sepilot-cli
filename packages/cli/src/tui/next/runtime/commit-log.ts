import type { Message } from '../../types.js'

export interface CommitLog {
  committed: Set<string>
}

export interface CommitSelection {
  /** Cumulative finalized messages passed to Ink's length-indexed Static list. */
  commits: Message[]
  /** Messages that remain in the re-rendering area. */
  live: Message[]
}

export function createCommitLog(): CommitLog {
  return { committed: new Set<string>() }
}

/**
 * Select finalized messages while keeping the active trailing message live.
 * Ink's Static component expects the complete append-only list on every
 * render; passing only the newly finalized slice makes its internal length
 * cursor reset whenever that slice becomes empty, which can print an older
 * live block repeatedly into terminal scrollback.
 */
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
  // Keep the newest tool and everything after it in Ink's mutable region.
  // The newest action can then remain expanded like Claude/OpenCode while the
  // preceding tool is committed compactly to native terminal scrollback as
  // soon as a newer action starts. A new user turn commits the previous tail.
  const liveTailStart = latestTurnToolIndex >= 0 ? latestTurnToolIndex : -1

  const commits: Message[] = []
  const live: Message[] = []
  for (let index = 0; index < messages.length; index += 1) {
    const current = messages[index]
    // Reasoning is a mutable preview, not a durable transcript entry. It may
    // precede an answer while still changing and disappears on the next turn.
    if (current.variant === 'thinking') {
      live.push(current)
      continue
    }
    // Once a message reaches terminal scrollback it must never return to the
    // mutable region, even if it temporarily becomes the transcript tail.
    if (log.committed.has(current.id)) {
      commits.push(current)
      continue
    }
    const mutableTool = current.role === 'tool'
      && (current.toolCall?.status === 'running' || current.toolCall?.status === 'pending')
    // Submitted user turns are immutable. Commit them immediately instead of
    // first rendering them in Ink's mutable area and then promoting them to
    // Static on the first tool/assistant update. That promotion writes the
    // same user block to terminal scrollback twice in inline mode.
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
