// Coordinates "only one panel open at a time" between overlay hooks.
//
// Each panel hook (useUsageDashboard, useMemorySearch, ...) owns its own
// state and a close handler. Today the cross-panel dismissal is duplicated
// inside loadUsageDashboard / loadMemorySearch — they each manually call
// the other panel's setters and bump the other panel's request ref to
// invalidate any in-flight load. This hook absorbs that orchestration:
//
//   const { register, begin } = useExclusivePanel()
//   useEffect(() => register('usage', closeUsageDashboard), [...])
//   useEffect(() => register('memory', closeMemorySearchPanel), [...])
//
//   async function loadUsageDashboard(days) {
//     const ticket = begin('usage')          // closes 'memory', invalidates it
//     setUsageDashboardOpen(true)
//     ...
//     const summary = await httpClient.usage()
//     if (!ticket.isCurrent()) return        // stale - user opened another panel
//     setUsageSummary(summary)
//   }
//
// The hook owns no panel state itself — it only tracks who is the current
// owner and lets dismissed panels detect that they were superseded.

import { useCallback, useRef } from 'react'

interface ExclusivePanelTicket {
  /** The request id bumped for this panel when begin() was called. */
  readonly requestId: number
  /**
   * True only as long as no other panel has called begin() afterwards
   * and no one called begin() on this panel again. Use this in async
   * callbacks to drop stale results.
   */
  isCurrent(): boolean
}

export interface UseExclusivePanelResult {
  /**
   * Register a panel and the dismiss callback to invoke when another
   * panel acquires exclusivity. Calling register a second time with the
   * same id replaces the previous handler — handy when the close
   * callback identity changes between renders (e.g. is recreated by a
   * useCallback dependency change).
   */
  register(panelId: string, dismiss: () => void): void
  /**
   * Mark `panelId` as the current panel. Returns a ticket the caller
   * can use to detect if it's been superseded by a later begin() call.
   * As a side-effect, every other registered panel's dismiss callback
   * is invoked and its request counter is bumped so any of its
   * in-flight tickets immediately become stale.
   */
  begin(panelId: string): ExclusivePanelTicket
}

export function useExclusivePanel(): UseExclusivePanelResult {
  const requestRefs = useRef<Map<string, number>>(new Map())
  const dismissRefs = useRef<Map<string, () => void>>(new Map())

  const register = useCallback((panelId: string, dismiss: () => void) => {
    dismissRefs.current.set(panelId, dismiss)
    if (!requestRefs.current.has(panelId)) {
      requestRefs.current.set(panelId, 0)
    }
  }, [])

  const begin = useCallback((panelId: string): ExclusivePanelTicket => {
    const nextId = (requestRefs.current.get(panelId) ?? 0) + 1
    for (const [otherId, dismiss] of dismissRefs.current) {
      if (otherId === panelId) continue
      // Bump first so the dismiss callback running synchronously can't
      // see a stale request id, then dismiss.
      requestRefs.current.set(
        otherId,
        (requestRefs.current.get(otherId) ?? 0) + 1,
      )
      try {
        dismiss()
      } catch {
        // A faulty dismiss callback must not break the panel that's
        // trying to take exclusivity.
      }
    }
    requestRefs.current.set(panelId, nextId)
    return {
      requestId: nextId,
      isCurrent: () => requestRefs.current.get(panelId) === nextId,
    }
  }, [])

  return { register, begin }
}
