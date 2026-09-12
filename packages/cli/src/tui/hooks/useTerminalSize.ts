import { useEffect, useState } from 'react'

export interface TerminalStreamLike {
  rows?: number
  columns?: number
  on?: (event: 'resize', listener: () => void) => void
  off?: (event: 'resize', listener: () => void) => void
  addListener?: (event: 'resize', listener: () => void) => void
  removeListener?: (event: 'resize', listener: () => void) => void
}

export interface TerminalSize {
  rows: number
  columns: number
}

const DEFAULT_TERMINAL_SIZE: TerminalSize = {
  rows: 24,
  columns: 80,
}

export function readTerminalSize(
  stdout?: TerminalStreamLike | null,
): TerminalSize {
  const rows = Number.isFinite(stdout?.rows) && (stdout?.rows ?? 0) > 0
    ? Math.floor(stdout?.rows ?? DEFAULT_TERMINAL_SIZE.rows)
    : DEFAULT_TERMINAL_SIZE.rows
  const columns = Number.isFinite(stdout?.columns) && (stdout?.columns ?? 0) > 0
    ? Math.floor(stdout?.columns ?? DEFAULT_TERMINAL_SIZE.columns)
    : DEFAULT_TERMINAL_SIZE.columns

  return { rows, columns }
}

function attachResizeListener(
  stdout: TerminalStreamLike,
  listener: () => void,
): () => void {
  if (typeof stdout.on === 'function' && typeof stdout.off === 'function') {
    stdout.on('resize', listener)
    return () => { stdout.off?.('resize', listener) }
  }

  if (
    typeof stdout.addListener === 'function'
    && typeof stdout.removeListener === 'function'
  ) {
    stdout.addListener('resize', listener)
    return () => { stdout.removeListener?.('resize', listener) }
  }

  return () => {}
}

export function useTerminalSize(
  stdout?: TerminalStreamLike | null,
): TerminalSize {
  const [size, setSize] = useState<TerminalSize>(() => readTerminalSize(stdout))

  useEffect(() => {
    const target = stdout ?? process.stdout
    if (!target) return

    const syncSize = () => {
      setSize((current) => {
        const next = readTerminalSize(target)
        return current.rows === next.rows && current.columns === next.columns
          ? current
          : next
      })
    }

    syncSize()
    return attachResizeListener(target, syncSize)
  }, [stdout])

  return size
}
