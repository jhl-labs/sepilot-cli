// Bundles the state and the close handler that the model picker overlay
// needs. Same shape as the other overlay hooks so a future coordinator can
// speak to all of them uniformly.
//
// There used to be a persistent "target" toggle (session vs daemon default)
// here. That toggle is gone: Enter always switches the session model and
// Ctrl+D always saves the daemon default (and switches the session), so the
// picker no longer needs a mode to track between those two actions.

import { useCallback, useState } from 'react'

export interface UseModelPickerResult {
  modelPickerOpen: boolean
  setModelPickerOpen: React.Dispatch<React.SetStateAction<boolean>>
  modelQuery: string
  setModelQuery: React.Dispatch<React.SetStateAction<string>>
  modelPickerIndex: number
  setModelPickerIndex: React.Dispatch<React.SetStateAction<number>>
  providersLoading: boolean
  setProvidersLoading: React.Dispatch<React.SetStateAction<boolean>>
  providersError: string | null
  setProvidersError: React.Dispatch<React.SetStateAction<string | null>>
  closeModelPicker: () => void
}

export function useModelPicker(): UseModelPickerResult {
  const [modelPickerOpen, setModelPickerOpen] = useState(false)
  const [modelQuery, setModelQuery] = useState('')
  const [modelPickerIndex, setModelPickerIndex] = useState(0)
  const [providersLoading, setProvidersLoading] = useState(false)
  const [providersError, setProvidersError] = useState<string | null>(null)

  const closeModelPicker = useCallback(() => {
    setModelPickerOpen(false)
    setModelQuery('')
    setModelPickerIndex(0)
    setProvidersError(null)
  }, [])

  return {
    modelPickerOpen,
    setModelPickerOpen,
    modelQuery,
    setModelQuery,
    modelPickerIndex,
    setModelPickerIndex,
    providersLoading,
    setProvidersLoading,
    providersError,
    setProvidersError,
    closeModelPicker,
  }
}
