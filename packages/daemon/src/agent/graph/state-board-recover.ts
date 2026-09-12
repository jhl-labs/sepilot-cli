import type {
  AgentFailedAttempt,
  AgentOpenQuestion,
  AgentStateBoardSnapshot,
  SessionEvent,
} from '@sepilotd/core'
import type { AgentSteeringNote } from './state-board.js'

/**
 * Recover the latest board from a session journal by scanning back-to-front for
 * the most recent `state_board` event. Used on resume: after the run checkpoint
 * rehydrates `graphState`, any board journaled *after* that checkpoint is
 * replayed so evidence/failed-attempts accumulated since the last checkpoint
 * are not lost. Selection is purely structural (event type + order) — no
 * content parsing.
 */
export function recoverStateBoard(
  events: readonly SessionEvent[],
): AgentStateBoardSnapshot | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i]
    if (event?.type === 'state_board') return event.board
  }
  return undefined
}

/** Epoch-ms `at` of the most recent `state_board` event, or undefined. */
export function latestStateBoardAt(events: readonly SessionEvent[]): number | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i]
    if (event?.type === 'state_board') return event.at
  }
  return undefined
}

/**
 * Minimal state slice a recovered board can restore. The board is a *projection*
 * of run state, so the reverse merge is deliberately narrow and non-destructive:
 * only the two fields whose board types are identical to their canonical
 * `AgentState` fields, and only when the rehydrated checkpoint left them empty
 * (i.e. a crash between the board journal append and the checkpoint write lost
 * them). Existing non-empty state is never overwritten — a projection must not
 * clobber the source of truth.
 */
export interface RecoverableBoardState {
  failedAttempts?: AgentFailedAttempt[]
  openQuestions?: AgentOpenQuestion[]
}

/**
 * Fill empty `failedAttempts`/`openQuestions` on the rehydrated graph state from
 * a strictly-newer journal board. Returns true if anything was restored. Pure
 * structured field replacement — no heuristics, no content parsing.
 */
export function applyRecoveredBoard(
  state: RecoverableBoardState,
  board: AgentStateBoardSnapshot | undefined,
): boolean {
  if (!board) return false
  let restored = false
  if ((state.failedAttempts?.length ?? 0) === 0 && board.failedAttempts.length > 0) {
    state.failedAttempts = board.failedAttempts.map((attempt) => ({ ...attempt }))
    restored = true
  }
  if ((state.openQuestions?.length ?? 0) === 0 && board.openQuestions.length > 0) {
    state.openQuestions = board.openQuestions.map((question) => ({ ...question }))
    restored = true
  }
  return restored
}

export interface RecoverableSteeringState {
  steeringNotes?: AgentSteeringNote[]
}

/**
 * Reconcile durable steering journal events with a checkpointed graph state.
 *
 * A steering acknowledgement can be appended after the last graph checkpoint,
 * and a cancellation can likewise happen before the next checkpoint. Replaying
 * the explicit note ids and terminal events closes both crash windows without
 * inferring intent from message text. Cancelled notes and delivered questions
 * are never reintroduced. Delivered instructions remain active without another
 * delivery acknowledgement.
 */
export function reconcileRecoveredSteeringNotes(
  state: RecoverableSteeringState,
  events: readonly SessionEvent[],
): boolean {
  const journal = new Map<string, AgentSteeringNote>()
  const turnStart = events.findLastIndex((event) => event.type === 'user_message')
  for (const event of events.slice(Math.max(0, turnStart))) {
    if (event.type === 'steering_ack') {
      const createdAt = Date.parse(event.timestamp)
      journal.set(event.noteId, {
        id: event.noteId,
        message: event.message,
        kind: event.kind,
        createdAt: Number.isFinite(createdAt) ? createdAt : 0,
      })
      continue
    }
    if (event.type !== 'steering_consumed' && event.type !== 'steering_cancelled') continue
    const note = journal.get(event.noteId)
    if (!note) continue
    const terminalAt = Date.parse(event.timestamp)
    const at = Number.isFinite(terminalAt) ? terminalAt : 0
    if (event.type === 'steering_consumed') note.consumedAt = at
    else note.cancelledAt = at
  }

  let changed = false
  const existing = state.steeringNotes ?? []
  const existingIds = new Set(existing.map((note) => note.id))
  for (const note of existing) {
    const durable = journal.get(note.id)
    if (!durable) continue
    if (durable.consumedAt !== undefined && note.consumedAt === undefined) {
      note.consumedAt = durable.consumedAt
      changed = true
    }
    if (durable.cancelledAt !== undefined && note.cancelledAt === undefined) {
      note.cancelledAt = durable.cancelledAt
      changed = true
    }
  }

  const recoveredPending = [...journal.values()].filter(
    (note) =>
      !existingIds.has(note.id)
      && (note.kind === 'instruction' || note.consumedAt === undefined)
      && note.cancelledAt === undefined,
  )
  if (recoveredPending.length > 0) {
    state.steeringNotes = [...existing, ...recoveredPending].slice(-20)
    changed = true
  }
  return changed
}
