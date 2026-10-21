import { useCallback, useRef } from 'react'

interface ExclusivePanelTicket {
  readonly requestId: number
  isCurrent(): boolean
}

export interface UseExclusivePanelResult {
  register(panelId: string, dismiss: () => void): void
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
      requestRefs.current.set(
        otherId,
        (requestRefs.current.get(otherId) ?? 0) + 1,
      )
      try {
        dismiss()
      } catch {
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
