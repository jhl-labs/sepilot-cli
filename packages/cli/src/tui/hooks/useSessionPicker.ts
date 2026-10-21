import { useCallback, useState } from 'react'
import type { DaemonSessionMeta } from '@sepilotd/api-client'

export interface UseSessionPickerResult {
  sessionPickerOpen: boolean
  setSessionPickerOpen: React.Dispatch<React.SetStateAction<boolean>>
  sessionQuery: string
  setSessionQuery: React.Dispatch<React.SetStateAction<string>>
  sessionPickerIndex: number
  setSessionPickerIndex: React.Dispatch<React.SetStateAction<number>>
  sessionItems: DaemonSessionMeta[]
  setSessionItems: React.Dispatch<React.SetStateAction<DaemonSessionMeta[]>>
  sessionsLoading: boolean
  setSessionsLoading: React.Dispatch<React.SetStateAction<boolean>>
  sessionsError: string | null
  setSessionsError: React.Dispatch<React.SetStateAction<string | null>>
  closeSessionPicker: () => void
}

export function useSessionPicker(): UseSessionPickerResult {
  const [sessionPickerOpen, setSessionPickerOpen] = useState(false)
  const [sessionQuery, setSessionQuery] = useState('')
  const [sessionPickerIndex, setSessionPickerIndex] = useState(0)
  const [sessionItems, setSessionItems] = useState<DaemonSessionMeta[]>([])
  const [sessionsLoading, setSessionsLoading] = useState(false)
  const [sessionsError, setSessionsError] = useState<string | null>(null)

  const closeSessionPicker = useCallback(() => {
    setSessionPickerOpen(false)
    setSessionQuery('')
    setSessionPickerIndex(0)
    setSessionsError(null)
  }, [])

  return {
    sessionPickerOpen,
    setSessionPickerOpen,
    sessionQuery,
    setSessionQuery,
    sessionPickerIndex,
    setSessionPickerIndex,
    sessionItems,
    setSessionItems,
    sessionsLoading,
    setSessionsLoading,
    sessionsError,
    setSessionsError,
    closeSessionPicker,
  }
}
