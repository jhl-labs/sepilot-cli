import { useState, useCallback, useEffect, useRef } from 'react'

interface ScrollState {
  offset: number
  totalLines: number
  viewportLines: number
  isAtBottom: boolean
}

export interface ScrollableApi {
  state: ScrollState
  scrollUp: (lines?: number) => void
  scrollDown: (lines?: number) => void
  pageUp: () => void
  pageDown: () => void
  scrollToBottom: () => void
}

const DEFAULT_STEP = 3

export function useScrollable(
  totalLines: number,
  viewportLines: number,
): ScrollableApi {
  const [rawOffset, setRawOffset] = useState(0)
  // Track whether the user has intentionally scrolled away from bottom.
  // A ref avoids creating spurious effect dependencies.
  const userScrolledRef = useRef(false)

  const maxOffset = Math.max(0, totalLines - viewportLines)

  // Derive the effective offset synchronously during render so clamps are
  // visible immediately (no effect batching delay in the probe-component test).
  const offset = Math.min(rawOffset, maxOffset)

  // Auto-follow bottom when new content arrives and the user hasn't scrolled.
  useEffect(() => {
    if (!userScrolledRef.current) {
      setRawOffset(0)
    }
  }, [totalLines])

  const scrollUp = useCallback(
    (lines: number = DEFAULT_STEP) => {
      setRawOffset((prev) => {
        const next = Math.min(maxOffset, prev + lines)
        if (next > 0) userScrolledRef.current = true
        return next
      })
    },
    [maxOffset],
  )

  const scrollDown = useCallback((lines: number = DEFAULT_STEP) => {
    setRawOffset((prev) => {
      const next = Math.max(0, prev - lines)
      if (next === 0) userScrolledRef.current = false
      return next
    })
  }, [])

  const scrollToBottom = useCallback(() => {
    userScrolledRef.current = false
    setRawOffset(0)
  }, [])

  const pageStep = Math.max(1, viewportLines - 2)
  const pageUp = useCallback(
    () => scrollUp(pageStep),
    [pageStep, scrollUp],
  )
  const pageDown = useCallback(
    () => scrollDown(pageStep),
    [pageStep, scrollDown],
  )

  const state: ScrollState = {
    offset,
    totalLines,
    viewportLines,
    isAtBottom: offset === 0,
  }

  return { state, scrollUp, scrollDown, pageUp, pageDown, scrollToBottom }
}
