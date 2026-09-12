import { useEffect, useState } from 'react'
import { colors } from '../../theme.js'

const ACTIVITY_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'] as const
// This is the single animation clock for an active run. Ink redraws the
// complete live region for every state update, so independent status/tool/
// elapsed timers multiply both CPU use and retained PTY output. One frame per
// second still communicates liveness and also drives the elapsed-time label.
const ACTIVITY_TICK_MS = 1_000

export interface ActivityAnimation {
  frame: string | null
  color: string
}

export function activityFrameAt(tick: number): string {
  const index = Math.max(0, Math.floor(tick)) % ACTIVITY_FRAMES.length
  return ACTIVITY_FRAMES[index] ?? ACTIVITY_FRAMES[0]
}

/**
 * A redraw-only activity signal for the live region. It never enters chat
 * state or scrollback, so fast spinner frames cannot pollute session history.
 */
export function useActivityAnimation(active: boolean): ActivityAnimation {
  const [tick, setTick] = useState(0)

  useEffect(() => {
    setTick(0)
    if (!active) return
    const interval = setInterval(() => setTick((value) => value + 1), ACTIVITY_TICK_MS)
    return () => clearInterval(interval)
  }, [active])

  if (!active) {
    return { frame: null, color: colors.dimText }
  }

  const frame = activityFrameAt(tick)
  const pulse = Math.floor(tick / 3) % 3
  const color = pulse === 0 ? colors.primary : pulse === 1 ? colors.info : colors.warning
  return { frame, color }
}

export const activityAnimationTiming = {
  tickMs: ACTIVITY_TICK_MS,
} as const
