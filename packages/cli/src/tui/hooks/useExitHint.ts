// Owns the staged-Ctrl+C "press again to exit" hint. The hint string
// shows up in the input box footer; calling show() arms it (cancelling
// any prior arm timer first) and the hint auto-clears after
// timeoutMs unless show() is called again or clear() runs.
//
// Pulled out of App.tsx so the timer ref + cleanup effect + manual
// clear-on-exit branch all live behind one armed boolean for callers.

import { useCallback, useEffect, useRef, useState } from 'react'

export interface UseExitHintResult {
  /** Current hint string, or null when no hint is armed. */
  exitHint: string | null
  /** Arm a new hint, replacing any previous one and resetting the auto-clear timer. */
  show(message: string): void
  /** Clear the hint immediately and cancel the auto-clear timer. */
  clear(): void
}

export function useExitHint(timeoutMs: number): UseExitHintResult {
  const [exitHint, setExitHint] = useState<string | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const cancelTimer = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const show = useCallback((message: string) => {
    cancelTimer()
    setExitHint(message)
    timerRef.current = setTimeout(() => {
      setExitHint(null)
      timerRef.current = null
    }, timeoutMs)
  }, [cancelTimer, timeoutMs])

  const clear = useCallback(() => {
    cancelTimer()
    setExitHint(null)
  }, [cancelTimer])

  // Cancel any in-flight timer on unmount so a delayed setExitHint(null)
  // never lands on an unmounted component.
  useEffect(() => cancelTimer, [cancelTimer])

  return { exitHint, show, clear }
}
