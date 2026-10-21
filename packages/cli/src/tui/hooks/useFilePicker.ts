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
