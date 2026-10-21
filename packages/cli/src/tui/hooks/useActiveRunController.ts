import { useCallback, useRef } from 'react'
import {
  createDaemonChatRunController,
  type DaemonChatRunController,
} from '@sepilotd/api-client'

export interface UseActiveRunControllerResult {
  begin(): DaemonChatRunController
  beginDetached(): DaemonChatRunController
  clear(controller: DaemonChatRunController): void
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
    activeRef.current = null
    return true
  }, [])

  return { begin, beginDetached, clear, cancel }
}
