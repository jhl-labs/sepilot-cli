import { randomUUID } from 'node:crypto'
import type { AgentStateBoardSnapshot, ISessionStore } from '@sepilotd/core'

/**
 * Append a board snapshot to the append-only session journal. Because the
 * journal is written through `WatchedSessionStore`, this single `appendEvent`
 * both persists the board and broadcasts it over `SessionWatchBroker` — one
 * event, no second store (opencode-style persist+broadcast). The board is a
 * lightweight, human-readable summary journaled at the same cadence as the
 * heavier run checkpoint, so restart/resume can replay the latest board.
 *
 * `now` is injectable so tests can assert deterministic snapshot times.
 */
export async function appendStateBoardEvent(
  sessions: ISessionStore,
  sessionId: string,
  board: AgentStateBoardSnapshot,
  now: () => number = () => Date.now(),
): Promise<void> {
  const at = now()
  await sessions.appendEvent(sessionId, {
    type: 'state_board',
    id: randomUUID(),
    timestamp: new Date(at).toISOString(),
    at,
    board,
  })
}
