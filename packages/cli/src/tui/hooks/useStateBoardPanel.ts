import { useCallback, useState } from 'react'
import type { AgentStateBoardSnapshot, DaemonStateBoardResponse } from '@sepilotd/api-client'
import { formatStateBoardText } from '@sepilotd/presentation'
import type { SlashCommand } from '../commands/slash-registry.js'

export interface StateBoardEventLike {
  type: string
  board?: AgentStateBoardSnapshot
}

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
