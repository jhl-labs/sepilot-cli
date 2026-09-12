/**
 * Shared live board for concurrent sibling subagents (opencode shared-block
 * style). Isolated `subagent.dispatch` runs are parallel-safe, so siblings under
 * the same parent run may execute at once; this board lets them append their
 * findings and read each other's during the run, so a sibling that already hit
 * a dead-end warns the others instead of every sibling rediscovering it.
 *
 * Best-effort by design: a single-process in-memory Map keyed by parent runId,
 * append-only, bounded FIFO. No hard synchronisation or locking — a missed read
 * only costs a duplicated exploration, never correctness. PLAN_065 T4.
 */

export const MAX_SHARED_BOARD_ENTRIES = 200

/**
 * Shared-board toggle (default on). Set SEPILOTD_SHARED_BOARD=off to disable the
 * concurrent sibling live board (append + read become no-ops at the call sites).
 * General option, not a model/dataset branch. PLAN_065 T6.
 */
export function isSharedBoardEnabled(): boolean {
  const raw = process.env.SEPILOTD_SHARED_BOARD?.trim().toLowerCase()
  return raw !== 'off' && raw !== '0' && raw !== 'false'
}

export type SharedBoardEntryKind = 'failed_attempt' | 'open_question' | 'evidence'

export interface SharedBoardEntry {
  origin: { sessionId: string; category?: string }
  kind: SharedBoardEntryKind
  payload: Record<string, unknown>
  ts: number
}

export interface SharedBoardAppend {
  origin: { sessionId: string; category?: string }
  kind: SharedBoardEntryKind
  payload: Record<string, unknown>
}

export class SharedBoard {
  private readonly boards = new Map<string, SharedBoardEntry[]>()

  append(parentRunId: string, entry: SharedBoardAppend, now = Date.now()): void {
    const list = this.boards.get(parentRunId) ?? []
    list.push({ ...entry, ts: now })
    if (list.length > MAX_SHARED_BOARD_ENTRIES) {
      list.splice(0, list.length - MAX_SHARED_BOARD_ENTRIES)
    }
    this.boards.set(parentRunId, list)
  }

  read(parentRunId: string): SharedBoardEntry[] {
    return this.boards.get(parentRunId) ?? []
  }

  /** Drop every entry for a parent run — call when the parent run ends. */
  clear(parentRunId: string): void {
    this.boards.delete(parentRunId)
  }
}

/**
 * Append a rolled-up subagent's structured findings to the shared board so
 * sibling subagents under the same parent run can see them. Structured payloads
 * only — never the free-text summary.
 */
export function appendSubagentFindingsToSharedBoard(
  board: SharedBoard,
  parentRunId: string,
  findings: {
    sessionId: string
    category?: string
    failedAttempts: Array<{ signature: string; tool: string; reason: string }>
    openQuestions: Array<{ id: string; text: string; blocking: boolean }>
  },
): void {
  const origin = { sessionId: findings.sessionId, category: findings.category }
  for (const attempt of findings.failedAttempts) {
    board.append(parentRunId, {
      origin,
      kind: 'failed_attempt',
      payload: { signature: attempt.signature, tool: attempt.tool, reason: attempt.reason },
    })
  }
  for (const question of findings.openQuestions) {
    board.append(parentRunId, {
      origin,
      kind: 'open_question',
      payload: { id: question.id, text: question.text, blocking: question.blocking },
    })
  }
}
