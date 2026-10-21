import { useEffect, useState } from 'react'

const TICK_MS = 1000

export function useStreamProgressTimer(active: boolean): number {
  const [now, setNow] = useState<number>(() => Date.now())

  useEffect(() => {
    if (!active) return
    setNow(Date.now())
    const interval = setInterval(() => {
      setNow(Date.now())
    }, TICK_MS)
    return () => clearInterval(interval)
  }, [active])

  return now
}
