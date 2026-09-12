import { useCallback, useEffect, useState } from 'react'
import { appendHistory, loadHistory } from '../utils/input-history.js'

export interface UseInputHistory {
  history: string[]
  append: (text: string) => void
}

export function useInputHistory(cwd: string = process.cwd()): UseInputHistory {
  const [history, setHistory] = useState<string[]>([])

  useEffect(() => {
    let cancelled = false
    void loadHistory(cwd).then((entries) => {
      if (!cancelled) setHistory(entries)
    })
    return () => {
      cancelled = true
    }
  }, [cwd])

  const append = useCallback(
    (text: string) => {
      const trimmed = text.trim()
      if (!trimmed) return

      let lastEntry: string | undefined
      setHistory((prev) => {
        lastEntry = prev[prev.length - 1]
        if (lastEntry === text) return prev
        return [...prev, text]
      })

      void appendHistory(cwd, text, lastEntry).catch(() => {
        // Swallow disk failures; in-memory copy already updated.
      })
    },
    [cwd],
  )

  return { history, append }
}
