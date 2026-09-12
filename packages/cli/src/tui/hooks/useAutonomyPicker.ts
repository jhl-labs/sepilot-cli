// Bundles the two pieces of state and the close handler that the autonomy
// picker overlay needs. Smallest of the picker hooks — autonomy options
// come from a static enum, so there's no items / loading / error to track.

import { useCallback, useState } from 'react'

export interface UseAutonomyPickerResult {
  autonomyPickerOpen: boolean
  setAutonomyPickerOpen: React.Dispatch<React.SetStateAction<boolean>>
  autonomyPickerIndex: number
  setAutonomyPickerIndex: React.Dispatch<React.SetStateAction<number>>
  closeAutonomyPicker: () => void
}

export function useAutonomyPicker(): UseAutonomyPickerResult {
  const [autonomyPickerOpen, setAutonomyPickerOpen] = useState(false)
  const [autonomyPickerIndex, setAutonomyPickerIndex] = useState(0)

  const closeAutonomyPicker = useCallback(() => {
    setAutonomyPickerOpen(false)
    setAutonomyPickerIndex(0)
  }, [])

  return {
    autonomyPickerOpen,
    setAutonomyPickerOpen,
    autonomyPickerIndex,
    setAutonomyPickerIndex,
    closeAutonomyPicker,
  }
}
