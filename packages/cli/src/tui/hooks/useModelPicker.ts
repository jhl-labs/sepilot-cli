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
