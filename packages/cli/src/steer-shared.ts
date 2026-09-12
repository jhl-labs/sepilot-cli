import { ApiHttpError } from '@sepilotd/api-client'
import { friendlyErrorMessage } from './utils/error-message.js'

/**
 * Shared guidance shown whenever a steering note is rejected because the
 * daemon reports no active run for the session (`409 NO_ACTIVE_RUN`). Used
 * by all three `/steer` entrypoints (TUI Ctrl+S, TUI/REPL slash command,
 * and the top-level `sepilot steer` command) so the message stays
 * consistent no matter which surface the user hit.
 */
export const STEER_NO_ACTIVE_RUN_GUIDANCE = '실행 중인 턴이 없습니다 — 그냥 메시지로 보내세요'

export type SteerOutcome =
  | { ok: true; noteId: string; pendingSteeringNoteCount?: number }
  // Failure outcomes carry `originalMessage` so callers with an ephemeral
  // input surface (the TUI composer) can restore the user's typed text
  // instead of losing it. This is a tested contract — keep it populated.
  | { ok: false; noActiveRun: true; guidance: string; originalMessage: string }
  | { ok: false; noActiveRun: false; message: string; originalMessage: string }

export type SteerCancellationTarget =
  | { selector: 'latest' | 'all' }
  | { noteId: string }

export type SteerCancellationOutcome =
  | {
      ok: true
      cancelledNoteIds: string[]
      pendingSteeringNoteCount: number
    }
  | {
      ok: false
      reason:
        | 'already_consumed'
        | 'already_cancelled'
        | 'no_pending'
        | 'no_active_run'
        | 'not_found'
        | 'error'
      message: string
    }

/** Minimal shape of the api-client method this helper depends on. */
export interface SteerCapableClient {
  steerSession(
    sessionId: string,
    message: string,
    kind?: 'instruction' | 'question',
  ): Promise<{ noteId: string; pendingSteeringNoteCount?: number }>
}

export interface SteerCancellationCapableClient {
  cancelSessionSteering(
    sessionId: string,
    target: SteerCancellationTarget,
  ): Promise<{
    status: 'cancelled'
    cancelledNoteIds: string[]
    pendingSteeringNoteCount: number
  }>
}

export function formatQueuedSteerNote(noteId: string, pendingSteeringNoteCount?: number): string {
  const pending =
    typeof pendingSteeringNoteCount === 'number' ? `, ${pendingSteeringNoteCount} pending` : ''
  return `queued (${noteId}${pending})`
}

function isNoActiveRunError(err: unknown): boolean {
  return (
    err instanceof ApiHttpError &&
    (err.code === 'NO_ACTIVE_RUN' || (err.status === 409 && !err.code))
  )
}

/**
 * Submit a mid-run steering note and normalize the result/error into a
 * discriminated union the caller can render without re-deriving the
 * 409/`NO_ACTIVE_RUN` guidance text itself. Never throws.
 */
export async function submitSteer(
  client: SteerCapableClient,
  sessionId: string,
  message: string,
  kind?: 'instruction' | 'question',
): Promise<SteerOutcome> {
  try {
    const { noteId, pendingSteeringNoteCount } = await client.steerSession(sessionId, message, kind)
    return typeof pendingSteeringNoteCount === 'number'
      ? { ok: true, noteId, pendingSteeringNoteCount }
      : { ok: true, noteId }
  } catch (err) {
    if (isNoActiveRunError(err)) {
      return {
        ok: false,
        noActiveRun: true,
        guidance: STEER_NO_ACTIVE_RUN_GUIDANCE,
        originalMessage: message,
      }
    }
    return {
      ok: false,
      noActiveRun: false,
      message: friendlyErrorMessage(err),
      originalMessage: message,
    }
  }
}

const STEER_CANCEL_ERROR = {
  STEERING_ALREADY_CONSUMED: {
    reason: 'already_consumed',
    message: 'Follow-up was already applied to the active run. Send a correction instead.',
  },
  STEERING_ALREADY_CANCELLED: {
    reason: 'already_cancelled',
    message: 'That follow-up was already cancelled.',
  },
  NO_PENDING_STEERING: {
    reason: 'no_pending',
    message: 'There are no pending follow-ups to cancel.',
  },
  NO_ACTIVE_RUN: {
    reason: 'no_active_run',
    message: 'There is no active run with pending follow-ups.',
  },
  STEERING_NOTE_NOT_FOUND: {
    reason: 'not_found',
    message: 'That follow-up is no longer present on the active run.',
  },
} as const

/** Cancel queued steering by explicit note id or daemon-authoritative selector. */
export async function cancelSteer(
  client: SteerCancellationCapableClient,
  sessionId: string,
  target: SteerCancellationTarget,
): Promise<SteerCancellationOutcome> {
  try {
    const result = await client.cancelSessionSteering(sessionId, target)
    return {
      ok: true,
      cancelledNoteIds: result.cancelledNoteIds,
      pendingSteeringNoteCount: result.pendingSteeringNoteCount,
    }
  } catch (error) {
    if (error instanceof ApiHttpError && error.code) {
      const known = STEER_CANCEL_ERROR[error.code as keyof typeof STEER_CANCEL_ERROR]
      if (known) return { ok: false, ...known }
    }
    return { ok: false, reason: 'error', message: friendlyErrorMessage(error) }
  }
}

export function formatCancelledSteerNote(
  cancelledNoteIds: readonly string[],
  pendingSteeringNoteCount: number,
): string {
  const noun = cancelledNoteIds.length === 1 ? 'follow-up' : 'follow-ups'
  return `Cancelled ${cancelledNoteIds.length} queued ${noun} (${pendingSteeringNoteCount} pending).`
}
