export const MAX_SHARED_BOARD_ENTRIES = 200

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

  clear(parentRunId: string): void {
    this.boards.delete(parentRunId)
  }
}

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
