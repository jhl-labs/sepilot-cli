import { useCallback, useEffect, useRef, useState } from 'react'

export interface UseExitHintResult {
  exitHint: string | null
  show(message: string): void
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

  useEffect(() => cancelTimer, [cancelTimer])

  return { exitHint, show, clear }
}
