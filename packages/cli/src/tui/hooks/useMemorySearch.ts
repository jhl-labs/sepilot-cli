import { useCallback, useRef, useState } from 'react'
import type {
  DaemonMemoryEntry,
  DaemonMemorySemanticStatus,
} from '@sepilotd/api-client'

export const MEMORY_SEARCH_LIMIT = 6

export interface UseMemorySearchResult {
  memorySearchOpen: boolean
  setMemorySearchOpen: React.Dispatch<React.SetStateAction<boolean>>
  memorySearchQuery: string
  setMemorySearchQuery: React.Dispatch<React.SetStateAction<string>>
  memorySearchResults: DaemonMemoryEntry[]
  setMemorySearchResults: React.Dispatch<
    React.SetStateAction<DaemonMemoryEntry[]>
  >
  memorySearchStatus: DaemonMemorySemanticStatus | null
  setMemorySearchStatus: React.Dispatch<
    React.SetStateAction<DaemonMemorySemanticStatus | null>
  >
  memorySearchLoading: boolean
  setMemorySearchLoading: React.Dispatch<React.SetStateAction<boolean>>
  memorySearchError: string | null
  setMemorySearchError: React.Dispatch<React.SetStateAction<string | null>>
  memoryRequestRef: React.MutableRefObject<number>
  closeMemorySearchPanel: () => void
}

export function useMemorySearch(): UseMemorySearchResult {
  const [memorySearchOpen, setMemorySearchOpen] = useState(false)
  const [memorySearchQuery, setMemorySearchQuery] = useState('')
  const [memorySearchResults, setMemorySearchResults] = useState<
    DaemonMemoryEntry[]
  >([])
  const [memorySearchStatus, setMemorySearchStatus] =
    useState<DaemonMemorySemanticStatus | null>(null)
  const [memorySearchLoading, setMemorySearchLoading] = useState(false)
  const [memorySearchError, setMemorySearchError] = useState<string | null>(
    null,
  )
  const memoryRequestRef = useRef(0)

  const closeMemorySearchPanel = useCallback(() => {
    memoryRequestRef.current += 1
    setMemorySearchOpen(false)
    setMemorySearchLoading(false)
    setMemorySearchError(null)
  }, [])

  return {
    memorySearchOpen,
    setMemorySearchOpen,
    memorySearchQuery,
    setMemorySearchQuery,
    memorySearchResults,
    setMemorySearchResults,
    memorySearchStatus,
    setMemorySearchStatus,
    memorySearchLoading,
    setMemorySearchLoading,
    memorySearchError,
    setMemorySearchError,
    memoryRequestRef,
    closeMemorySearchPanel,
  }
}
