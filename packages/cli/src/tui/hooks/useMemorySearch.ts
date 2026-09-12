// Bundles the six pieces of state, the request-ref, and the close handler
// for the memory-search overlay. Mirrors useUsageDashboard so the two
// overlays share a shape — that lets a future useExclusivePanel
// coordinator absorb their mutual exclusion without per-overlay forks.
//
// loadMemorySearch() in App.tsx still has cross-overlay coupling (it has
// to dismiss the usage dashboard before opening, and vice versa); pulling
// that orchestration up is the next step. For now this hook owns the
// per-overlay plumbing only.

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

  // Bumping memoryRequestRef invalidates any in-flight loadMemorySearch()
  // call so a late-arriving response can no longer reopen / re-render the
  // panel after the user has dismissed it.
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
