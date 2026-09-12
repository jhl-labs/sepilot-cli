// Drives the 1-Hz "now" timestamp the streaming/local-shell views read
// to decide how long the current activity has been running. The hook
// only ticks while `active` is true (typically isStreaming || a local
// shell command is in flight) so a totally idle TUI doesn't fire a
// useless setState every second.
//
// Returns Date.now() captured at the most recent tick — or, when
// inactive, the timestamp from when `active` last went false (so
// callers reading "duration" don't suddenly jump backwards).

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
