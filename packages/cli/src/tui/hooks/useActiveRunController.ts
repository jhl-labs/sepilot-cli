// Manages the single "active run controller" slot useChat keeps so that
// (a) starting a new agent run cancels and replaces any previous one and
// (b) cancelStream / handleAbortOutcome can both read whichever
// controller is currently in flight without sharing the ref directly.
//
// Detached controllers (used for short-lived approval responses, etc.)
// don't take the active slot — they're created and disposed by the
// caller and never compete with the agent loop.

import { useCallback, useRef } from 'react'
import {
  createDaemonChatRunController,
  type DaemonChatRunController,
} from '@sepilotd/api-client'

export interface UseActiveRunControllerResult {
  /**
   * Start a new active controller, cancelling and disposing the previous
   * active one if any. The returned controller is also stored in the
   * active slot.
   */
  begin(): DaemonChatRunController
  /**
   * Allocate a controller that is *not* tracked as the active one. Used
   * for short-lived async work (resolveApproval's response post, etc.)
   * that should not be cancelled by a new chat send.
   */
  beginDetached(): DaemonChatRunController
  /**
   * Dispose `controller` and clear the active slot if it still points at
   * it. Safe to call with a detached controller — the active slot only
   * clears when the slot was actually pointing at the same controller.
   */
  clear(controller: DaemonChatRunController): void
  /**
   * Cancel the active controller (if any) and clear the active slot.
   * Returns true when there was something to cancel; the caller can use
   * the boolean to skip side-effects like emitting a "cancelling" status
   * message when there's no active run.
   */
  cancel(): boolean
}

export function useActiveRunController(): UseActiveRunControllerResult {
  const activeRef = useRef<DaemonChatRunController | null>(null)

  const begin = useCallback((): DaemonChatRunController => {
    activeRef.current?.cancel()
    activeRef.current?.dispose()
    const controller = createDaemonChatRunController()
    activeRef.current = controller
    return controller
  }, [])

  const beginDetached = useCallback(
    (): DaemonChatRunController => createDaemonChatRunController(),
    [],
  )

  const clear = useCallback((controller: DaemonChatRunController): void => {
    controller.dispose()
    if (activeRef.current === controller) {
      activeRef.current = null
    }
  }, [])

  const cancel = useCallback((): boolean => {
    const controller = activeRef.current
    if (!controller) return false
    controller.cancel()
    // Drop the slot eagerly so a follow-up cancel() is a no-op rather
    // than calling cancel/dispose on an already-disposed controller.
    activeRef.current = null
    return true
  }, [])

  return { begin, beginDetached, clear, cancel }
}
