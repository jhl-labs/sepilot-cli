import { useCallback, useRef, useState } from 'react'
import type { MarketplaceSkillSearchResult } from '@sepilotd/api-client'

export const SKILL_STORE_SEARCH_LIMIT = 20

export interface UseSkillStoreResult {
  skillStoreOpen: boolean
  setSkillStoreOpen: React.Dispatch<React.SetStateAction<boolean>>
  skillStoreQuery: string
  setSkillStoreQuery: React.Dispatch<React.SetStateAction<string>>
  skillStoreSearchedQuery: string
  setSkillStoreSearchedQuery: React.Dispatch<React.SetStateAction<string>>
  skillStoreResults: MarketplaceSkillSearchResult[]
  setSkillStoreResults: React.Dispatch<
    React.SetStateAction<MarketplaceSkillSearchResult[]>
  >
  skillStoreIndex: number
  setSkillStoreIndex: React.Dispatch<React.SetStateAction<number>>
  skillStoreLoading: boolean
  setSkillStoreLoading: React.Dispatch<React.SetStateAction<boolean>>
  skillStoreInstallingSource: string | null
  setSkillStoreInstallingSource: React.Dispatch<
    React.SetStateAction<string | null>
  >
  skillStoreError: string | null
  setSkillStoreError: React.Dispatch<React.SetStateAction<string | null>>
  skillStoreMessage: string | null
  setSkillStoreMessage: React.Dispatch<React.SetStateAction<string | null>>
  skillStoreRequestRef: React.MutableRefObject<number>
  skillStoreInstallRequestRef: React.MutableRefObject<number>
  closeSkillStorePicker: () => void
}

export function useSkillStore(): UseSkillStoreResult {
  const [skillStoreOpen, setSkillStoreOpen] = useState(false)
  const [skillStoreQuery, setSkillStoreQuery] = useState('')
  const [skillStoreSearchedQuery, setSkillStoreSearchedQuery] = useState('')
  const [skillStoreResults, setSkillStoreResults] = useState<
    MarketplaceSkillSearchResult[]
  >([])
  const [skillStoreIndex, setSkillStoreIndex] = useState(0)
  const [skillStoreLoading, setSkillStoreLoading] = useState(false)
  const [skillStoreInstallingSource, setSkillStoreInstallingSource] =
    useState<string | null>(null)
  const [skillStoreError, setSkillStoreError] = useState<string | null>(null)
  const [skillStoreMessage, setSkillStoreMessage] = useState<string | null>(
    null,
  )
  const skillStoreRequestRef = useRef(0)
  const skillStoreInstallRequestRef = useRef(0)

  const closeSkillStorePicker = useCallback(() => {
    skillStoreRequestRef.current += 1
    skillStoreInstallRequestRef.current += 1
    setSkillStoreOpen(false)
    setSkillStoreLoading(false)
    setSkillStoreInstallingSource(null)
    setSkillStoreError(null)
    setSkillStoreMessage(null)
  }, [])

  return {
    skillStoreOpen,
    setSkillStoreOpen,
    skillStoreQuery,
    setSkillStoreQuery,
    skillStoreSearchedQuery,
    setSkillStoreSearchedQuery,
    skillStoreResults,
    setSkillStoreResults,
    skillStoreIndex,
    setSkillStoreIndex,
    skillStoreLoading,
    setSkillStoreLoading,
    skillStoreInstallingSource,
    setSkillStoreInstallingSource,
    skillStoreError,
    setSkillStoreError,
    skillStoreMessage,
    setSkillStoreMessage,
    skillStoreRequestRef,
    skillStoreInstallRequestRef,
    closeSkillStorePicker,
  }
}
