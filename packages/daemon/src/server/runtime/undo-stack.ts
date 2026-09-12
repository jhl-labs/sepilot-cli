import type { SessionEvent } from '@sepilotd/core'

export interface UndoSnapshot {
  events: SessionEvent[]
  capturedAt: string
}

interface SessionFrame {
  undo: UndoSnapshot[]
  redo: UndoSnapshot[]
}

const DEFAULT_DEPTH = 10

export class SessionUndoStack {
  private readonly frames = new Map<string, SessionFrame>()
  constructor(private readonly depth: number = DEFAULT_DEPTH) {}

  recordSnapshot(sessionId: string, events: SessionEvent[]): void {
    const frame = this.ensure(sessionId)
    frame.undo.push({ events: cloneEvents(events), capturedAt: new Date().toISOString() })
    if (frame.undo.length > this.depth) frame.undo.shift()
    frame.redo.length = 0
  }

  popUndo(sessionId: string): UndoSnapshot | null {
    const frame = this.frames.get(sessionId)
    if (!frame || frame.undo.length === 0) return null
    return frame.undo.pop() ?? null
  }

  pushRedo(sessionId: string, snapshot: UndoSnapshot): void {
    const frame = this.ensure(sessionId)
    frame.redo.push(snapshot)
    if (frame.redo.length > this.depth) frame.redo.shift()
  }

  popRedo(sessionId: string): UndoSnapshot | null {
    const frame = this.frames.get(sessionId)
    if (!frame || frame.redo.length === 0) return null
    return frame.redo.pop() ?? null
  }

  pushUndo(sessionId: string, snapshot: UndoSnapshot): void {
    const frame = this.ensure(sessionId)
    frame.undo.push(snapshot)
    if (frame.undo.length > this.depth) frame.undo.shift()
  }

  inspect(sessionId: string): { undoDepth: number; redoDepth: number } {
    const frame = this.frames.get(sessionId)
    return {
      undoDepth: frame?.undo.length ?? 0,
      redoDepth: frame?.redo.length ?? 0,
    }
  }

  dispose(sessionId: string): void {
    this.frames.delete(sessionId)
  }

  private ensure(sessionId: string): SessionFrame {
    let frame = this.frames.get(sessionId)
    if (!frame) {
      frame = { undo: [], redo: [] }
      this.frames.set(sessionId, frame)
    }
    return frame
  }
}

function cloneEvents(events: SessionEvent[]): SessionEvent[] {
  return events.map((event) => JSON.parse(JSON.stringify(event)) as SessionEvent)
}

/**
 * Trim events back to before the most recent user_message turn boundary.
 * Returns the trimmed events that should be persisted.
 * If no user_message exists, returns null.
 */
export function trimToPreviousTurn(events: SessionEvent[]): SessionEvent[] | null {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    if (events[i].type === 'user_message') {
      return events.slice(0, i)
    }
  }
  return null
}
