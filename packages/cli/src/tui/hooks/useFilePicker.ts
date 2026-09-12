// Bundles the file picker overlay's six pieces of state and its trivial
// close handler. Mirrors useUsageDashboard / useMemorySearch so all three
// overlays share a shape.
//
// openFilePicker stays in App.tsx for now because it has to consult the
// overlay-state coordinator (other pickers open) plus state.isStreaming /
// state.pendingApproval before flipping the open flag — that decision
// crosses overlay boundaries and belongs at the app level until we add a
// shared overlay coordinator.

import { useCallback, useState } from 'react'
import type { FilePickerItem } from '../components/FilePicker.js'

export interface UseFilePickerResult {
  filePickerOpen: boolean
  setFilePickerOpen: React.Dispatch<React.SetStateAction<boolean>>
  filePickerDir: string
  setFilePickerDir: React.Dispatch<React.SetStateAction<string>>
  filePickerItems: FilePickerItem[]
  setFilePickerItems: React.Dispatch<React.SetStateAction<FilePickerItem[]>>
  filePickerIndex: number
  setFilePickerIndex: React.Dispatch<React.SetStateAction<number>>
  filePickerLoading: boolean
  setFilePickerLoading: React.Dispatch<React.SetStateAction<boolean>>
  filePickerError: string | null
  setFilePickerError: React.Dispatch<React.SetStateAction<string | null>>
  closeFilePicker: () => void
}

export function useFilePicker(rootDir: string): UseFilePickerResult {
  const [filePickerOpen, setFilePickerOpen] = useState(false)
  const [filePickerDir, setFilePickerDir] = useState(rootDir)
  const [filePickerItems, setFilePickerItems] = useState<FilePickerItem[]>([])
  const [filePickerIndex, setFilePickerIndex] = useState(0)
  const [filePickerLoading, setFilePickerLoading] = useState(false)
  const [filePickerError, setFilePickerError] = useState<string | null>(null)

  const closeFilePicker = useCallback(() => {
    setFilePickerOpen(false)
    setFilePickerError(null)
  }, [])

  return {
    filePickerOpen,
    setFilePickerOpen,
    filePickerDir,
    setFilePickerDir,
    filePickerItems,
    setFilePickerItems,
    filePickerIndex,
    setFilePickerIndex,
    filePickerLoading,
    setFilePickerLoading,
    filePickerError,
    setFilePickerError,
    closeFilePicker,
  }
}
