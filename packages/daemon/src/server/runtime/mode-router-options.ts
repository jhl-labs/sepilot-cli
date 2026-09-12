import { randomUUID } from 'node:crypto'
import type { ModeRouterOptions } from '../../agent/mode-router.js'
import { appendStateBoardEvent } from '../../agent/graph/state-board-journal.js'
import { isStateBoardEnabled } from '../../agent/graph/state-board.js'
import type { ISessionStore } from '@sepilotd/core'
import type { RuntimeServices } from './types.js'
import type { SessionWatchBroker } from './session-watch.js'

/**
 * Build the `journalStateBoard` callback shared by every AgentModeRouter
 * construction site. Kept as a standalone helper (rather than only inline in
 * {@link createRuntimeBackedModeRouterOptions}) so sites that can't pull in
 * the full `RuntimeServices`-backed option set (e.g. narrow capability types)
 * can still wire up identical journaling instead of duplicating the logic.
 */
export function createJournalStateBoard(
  sessions: ISessionStore,
): NonNullable<ModeRouterOptions['journalStateBoard']> {
  return (sessionId, board) =>
    isStateBoardEnabled()
      ? appendStateBoardEvent(sessions, sessionId, board)
      : Promise.resolve()
}

/**
 * Build the `journalSteeringConsumed` callback shared by every AgentModeRouter
 * construction site. See {@link createJournalStateBoard} for why this is a
 * standalone export.
 */
export function createJournalSteeringConsumed(
  sessions: ISessionStore,
  sessionWatchBroker?: SessionWatchBroker,
): NonNullable<ModeRouterOptions['journalSteeringConsumed']> {
  return async (sessionId, noteId) => {
    await sessions.appendEvent(sessionId, {
      type: 'steering_consumed',
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      noteId,
    })
    sessionWatchBroker?.emit({
      type: 'steering_consumed',
      sessionId,
      noteId,
    })
  }
}

type RuntimeBackedModeRouterOptions = Pick<
  ModeRouterOptions,
  | 'saveApprovalCheckpoint'
  | 'clearApprovalCheckpoint'
  | 'saveRunCheckpoint'
  | 'clearRunCheckpoint'
  | 'journalStateBoard'
  | 'journalSteeringConsumed'
  | 'activeRuns'
  | 'loadToolExecution'
  | 'saveToolExecution'
  | 'clearToolExecution'
  | 'editSnapshotStore'
  | 'toolStatsStore'
  | 'workspaceMutationTracker'
  | 'pluginEvents'
  | 'graphNodeModelOverrides'
  | 'auxModel'
>

export function createRuntimeBackedModeRouterOptions(
  runtime: RuntimeServices,
): RuntimeBackedModeRouterOptions {
  return {
    saveApprovalCheckpoint: (checkpoint) =>
      runtime.approvalCheckpoints?.save(checkpoint) ?? Promise.resolve(),
    clearApprovalCheckpoint: (requestId) =>
      runtime.approvalCheckpoints?.delete(requestId) ?? Promise.resolve(),
    saveRunCheckpoint: (checkpoint) =>
      runtime.runCheckpoints?.save(checkpoint) ?? Promise.resolve(),
    clearRunCheckpoint: (sessionId) =>
      runtime.runCheckpoints?.delete(sessionId) ?? Promise.resolve(),
    journalStateBoard: createJournalStateBoard(runtime.sessions),
    journalSteeringConsumed: createJournalSteeringConsumed(runtime.sessions, runtime.sessionWatchBroker),
    activeRuns: runtime.activeRuns,
    loadToolExecution: (sessionId) =>
      runtime.toolExecutions?.get(sessionId) ?? Promise.resolve(null),
    saveToolExecution: (record) =>
      runtime.toolExecutions?.save(record) ?? Promise.resolve(),
    clearToolExecution: (sessionId) =>
      runtime.toolExecutions?.clearActive(sessionId) ?? Promise.resolve(),
    editSnapshotStore: runtime.editSnapshotStore,
    toolStatsStore: runtime.toolStatsStore,
    workspaceMutationTracker: runtime.workspaceMutationTracker,
    pluginEvents: runtime.pluginEvents,
    graphNodeModelOverrides: runtime.config.agent.graphNodeModelOverrides,
    auxModel: runtime.config.agent.auxModel,
  }
}
