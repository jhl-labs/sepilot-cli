import { useCallback, useState } from 'react'
import type { AgentStateBoardSnapshot, DaemonStateBoardResponse } from '@sepilotd/api-client'
import { formatStateBoardText } from '@sepilotd/presentation'
import type { SlashCommand } from '../commands/slash-registry.js'

/**
 * Structural shape of a `state_board` session event as it arrives over the
 * session-watch stream. Typed locally (not imported from daemon) so this stays
 * within surface dependency boundaries.
 */
export interface StateBoardEventLike {
  type: string
  board?: AgentStateBoardSnapshot
}

/**
 * Reducer core: return the board from a `state_board` event, or the previous
 * board for any other event. Pure and structural — no content parsing — so it
 * can route live session-watch frames into the panel deterministically.
 */
export function applyStateBoardEvent(
  previous: AgentStateBoardSnapshot | null,
  event: StateBoardEventLike,
): AgentStateBoardSnapshot | null {
  if (event.type === 'state_board' && event.board) return event.board
  return previous
}

export interface StateBoardSlashDeps {
  getSessionId: () => string | undefined
  fetchBoard: (sessionId: string) => Promise<DaemonStateBoardResponse>
  render: (content: string) => void
  onBoard?: (response: DaemonStateBoardResponse) => void
}

/**
 * Build the `/state` slash command. Fetches the board for the current (or an
 * explicitly named) session and renders it into the transcript, mirroring how
 * `/usage current` and `/mcp` surface daemon state. Extracted as a factory so
 * it is unit-testable without rendering the full App.
 */
export function createStateBoardSlashCommand(deps: StateBoardSlashDeps): SlashCommand {
  return {
    name: '/state',
    description: 'Show the agent state board (goal, criteria, evidence, failed attempts, open questions)',
    handler: async ({ args }) => {
      const sessionId = args[0] ?? deps.getSessionId()
      if (!sessionId) {
        deps.render('No active session — start a turn first or pass /state <sessionId>.')
        return
      }
      let response: DaemonStateBoardResponse
      try {
        response = await deps.fetchBoard(sessionId)
      } catch (error) {
        deps.render(`Failed to load state board: ${error instanceof Error ? error.message : String(error)}`)
        return
      }
      deps.onBoard?.(response)
      if (!response.board || response.source === 'none') {
        deps.render(`No state board for session ${sessionId}.`)
        return
      }
      deps.render(formatStateBoardText(response.board) ?? 'State board is empty.')
    },
  }
}

export interface UseStateBoardPanelResult {
  board: AgentStateBoardSnapshot | null
  source: DaemonStateBoardResponse['source'] | null
  updatedAt: number | null
  setFromResponse: (response: DaemonStateBoardResponse) => void
  applyEvent: (event: StateBoardEventLike) => void
  reset: () => void
}

/**
 * Live board state container for a future exclusive panel. Holds the latest
 * board plus a routing method for incoming session-watch `state_board` frames.
 */
export function useStateBoardPanel(): UseStateBoardPanelResult {
  const [board, setBoard] = useState<AgentStateBoardSnapshot | null>(null)
  const [source, setSource] = useState<DaemonStateBoardResponse['source'] | null>(null)
  const [updatedAt, setUpdatedAt] = useState<number | null>(null)

  const setFromResponse = useCallback((response: DaemonStateBoardResponse) => {
    setBoard(response.board)
    setSource(response.source)
    setUpdatedAt(response.updatedAt)
  }, [])

  const applyEvent = useCallback((event: StateBoardEventLike) => {
    setBoard((prev) => applyStateBoardEvent(prev, event))
    if (event.type === 'state_board' && event.board) setSource('live')
  }, [])

  const reset = useCallback(() => {
    setBoard(null)
    setSource(null)
    setUpdatedAt(null)
  }, [])

  return { board, source, updatedAt, setFromResponse, applyEvent, reset }
}
