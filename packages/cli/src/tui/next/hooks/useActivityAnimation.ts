import { useEffect, useState } from 'react'
import { colors } from '../../theme.js'

const ACTIVITY_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'] as const
const ACTIVITY_TICK_MS = 1_000

export interface ActivityAnimation {
  frame: string | null
  color: string
}

export function activityFrameAt(tick: number): string {
  const index = Math.max(0, Math.floor(tick)) % ACTIVITY_FRAMES.length
  return ACTIVITY_FRAMES[index] ?? ACTIVITY_FRAMES[0]
}

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
