import stringWidth from 'string-width'

// Ink renders its own cursor glyph while hiding the terminal cursor. Native
// IMEs, however, place their preedit/candidate UI at the terminal cursor. The
// private-use marker reserves one cell and lets the stdout wrapper recover the
// exact caret position from Ink's final (already wrapped) frame. On a TTY that
// cell is blank because the hardware cursor itself is the visible caret.
export const IME_CURSOR_MARKER = '\uE000'

const CURSOR_GLYPH = '│'
const HARDWARE_CURSOR_CELL = ' '
const HIDE_CURSOR = '\u001B[?25l'
const SHOW_CURSOR = '\u001B[?25h'
const SAVE_CURSOR = '\u001B7'
const RESTORE_CURSOR = '\u001B8'
const ANSI_ESCAPE = /\u001B\[[0-?]*[ -/]*[@-~]/g

interface CursorPosition {
  column: number
  rowsUp: number
}

export interface ImeCursorOutput {
  stdout: NodeJS.WriteStream
  enabled: boolean
  dispose: () => void
}

function locateCursor(frame: string, markerIndex: number): CursorPosition {
  const lineStart = frame.lastIndexOf('\n', markerIndex - 1) + 1
  const lineBeforeCursor = frame
    .slice(lineStart, markerIndex)
    .replace(ANSI_ESCAPE, '')
    .replace(/\r/g, '')
  const rowsUp = frame.slice(markerIndex + IME_CURSOR_MARKER.length)
    .split('\n')
    .length - 1

  return {
    column: stringWidth(lineBeforeCursor),
    rowsUp,
  }
}

function positionSequence(position: CursorPosition): string {
  const moveUp = position.rowsUp > 0 ? `\u001B[${position.rowsUp}A` : ''
  return `${SAVE_CURSOR}${moveUp}\u001B[${position.column + 1}G${SHOW_CURSOR}`
}

/**
 * Wrap Ink's stdout so the hardware cursor follows the inline composer caret.
 *
 * Ink's incremental renderer expects every write to begin at the cursor it
 * left after the previous frame. Before forwarding another write, restore that
 * saved location. A frame containing the composer marker is then rendered,
 * saved at its normal trailing position, and the visible cursor is moved back
 * over the marker for native IME placement.
 */
export function createImeCursorOutput(stdout: NodeJS.WriteStream): ImeCursorOutput {
  const enabled = Boolean(stdout.isTTY)
  let positioned = false
  let disposed = false

  const rawWrite = (value: string): void => {
    stdout.write(value)
  }

  const restoreRenderCursor = (): void => {
    if (!positioned) return
    positioned = false
    rawWrite(`${HIDE_CURSOR}${RESTORE_CURSOR}`)
  }

  const write = (chunk: unknown, ...args: unknown[]): boolean => {
    restoreRenderCursor()

    if (typeof chunk !== 'string') {
      return Reflect.apply(stdout.write, stdout, [chunk, ...args]) as boolean
    }

    const markerIndex = chunk.lastIndexOf(IME_CURSOR_MARKER)
    const rendered = markerIndex >= 0
      ? chunk.replaceAll(
        IME_CURSOR_MARKER,
        enabled && !disposed ? HARDWARE_CURSOR_CELL : CURSOR_GLYPH,
      )
      : chunk
    const result = Reflect.apply(stdout.write, stdout, [rendered, ...args]) as boolean

    if (enabled && !disposed && markerIndex >= 0) {
      rawWrite(positionSequence(locateCursor(chunk, markerIndex)))
      positioned = true
    }

    return result
  }

  const wrapped = new Proxy(stdout, {
    get(target, property) {
      if (property === 'write') return write
      const value = Reflect.get(target, property, target)
      return typeof value === 'function' ? value.bind(target) : value
    },
    set(target, property, value) {
      return Reflect.set(target, property, value, target)
    },
  })

  return {
    stdout: wrapped,
    enabled,
    dispose: () => {
      if (disposed) return
      disposed = true
      restoreRenderCursor()
      if (enabled) rawWrite(SHOW_CURSOR)
    },
  }
}
