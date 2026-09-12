import { useCallback, useState } from 'react'
import type {
  DaemonRagFolder,
  DaemonRagSearchHit,
  DaemonRagSyncResult,
  DaemonRagVectorDbInfo,
} from '@sepilotd/api-client'

export const RAG_SEARCH_LIMIT = 6

export interface UseRagPanelResult {
  ragPanelOpen: boolean
  setRagPanelOpen: React.Dispatch<React.SetStateAction<boolean>>
  ragQuery: string
  setRagQuery: React.Dispatch<React.SetStateAction<string>>
  ragSources: DaemonRagFolder[]
  setRagSources: React.Dispatch<React.SetStateAction<DaemonRagFolder[]>>
  ragHits: DaemonRagSearchHit[]
  setRagHits: React.Dispatch<React.SetStateAction<DaemonRagSearchHit[]>>
  ragVectorInfo: DaemonRagVectorDbInfo | null
  setRagVectorInfo: React.Dispatch<
    React.SetStateAction<DaemonRagVectorDbInfo | null>
  >
  ragSyncResult: DaemonRagSyncResult | null
  setRagSyncResult: React.Dispatch<
    React.SetStateAction<DaemonRagSyncResult | null>
  >
  ragSelectedIndex: number
  setRagSelectedIndex: React.Dispatch<React.SetStateAction<number>>
  ragLoading: boolean
  setRagLoading: React.Dispatch<React.SetStateAction<boolean>>
  ragError: string | null
  setRagError: React.Dispatch<React.SetStateAction<string | null>>
  closeRagPanel: () => void
}

export function useRagPanel(): UseRagPanelResult {
  const [ragPanelOpen, setRagPanelOpen] = useState(false)
  const [ragQuery, setRagQuery] = useState('')
  const [ragSources, setRagSources] = useState<DaemonRagFolder[]>([])
  const [ragHits, setRagHits] = useState<DaemonRagSearchHit[]>([])
  const [ragVectorInfo, setRagVectorInfo] =
    useState<DaemonRagVectorDbInfo | null>(null)
  const [ragSyncResult, setRagSyncResult] =
    useState<DaemonRagSyncResult | null>(null)
  const [ragSelectedIndex, setRagSelectedIndex] = useState(0)
  const [ragLoading, setRagLoading] = useState(false)
  const [ragError, setRagError] = useState<string | null>(null)

  const closeRagPanel = useCallback(() => {
    setRagPanelOpen(false)
    setRagLoading(false)
    setRagError(null)
  }, [])

  return {
    ragPanelOpen,
    setRagPanelOpen,
    ragQuery,
    setRagQuery,
    ragSources,
    setRagSources,
    ragHits,
    setRagHits,
    ragVectorInfo,
    setRagVectorInfo,
    ragSyncResult,
    setRagSyncResult,
    ragSelectedIndex,
    setRagSelectedIndex,
    ragLoading,
    setRagLoading,
    ragError,
    setRagError,
    closeRagPanel,
  }
}
