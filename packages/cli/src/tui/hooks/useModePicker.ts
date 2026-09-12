// Bundles the five pieces of state and the close handler that the agent
// mode picker overlay needs. Same shape as the other overlay hooks; the
// hook owns only state, while open/load orchestration stays in App.tsx
// because it has to consult the overlay-state coordinator.

import { useCallback, useState } from 'react'

export interface UseModePickerResult {
  modePickerOpen: boolean
  setModePickerOpen: React.Dispatch<React.SetStateAction<boolean>>
  modeQuery: string
  setModeQuery: React.Dispatch<React.SetStateAction<string>>
  modePickerIndex: number
  setModePickerIndex: React.Dispatch<React.SetStateAction<number>>
  agentModesLoading: boolean
  setAgentModesLoading: React.Dispatch<React.SetStateAction<boolean>>
  agentModesError: string | null
  setAgentModesError: React.Dispatch<React.SetStateAction<string | null>>
  closeModePicker: () => void
}

export function useModePicker(): UseModePickerResult {
  const [modePickerOpen, setModePickerOpen] = useState(false)
  const [modeQuery, setModeQuery] = useState('')
  const [modePickerIndex, setModePickerIndex] = useState(0)
  const [agentModesLoading, setAgentModesLoading] = useState(false)
  const [agentModesError, setAgentModesError] = useState<string | null>(null)

  const closeModePicker = useCallback(() => {
    setModePickerOpen(false)
    setModeQuery('')
    setModePickerIndex(0)
    setAgentModesError(null)
  }, [])

  return {
    modePickerOpen,
    setModePickerOpen,
    modeQuery,
    setModeQuery,
    modePickerIndex,
    setModePickerIndex,
    agentModesLoading,
    setAgentModesLoading,
    agentModesError,
    setAgentModesError,
    closeModePicker,
  }
}
