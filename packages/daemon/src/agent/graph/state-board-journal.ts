import { randomUUID } from 'node:crypto'
import type { AgentStateBoardSnapshot, ISessionStore } from '@sepilotd/core'

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
