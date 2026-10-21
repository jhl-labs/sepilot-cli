import { useState } from 'react'

export interface UseSessionRosterResult {
  projectSessionIds: string[]
  setProjectSessionIds: React.Dispatch<React.SetStateAction<string[]>>
  recentSessionIds: string[]
  setRecentSessionIds: React.Dispatch<React.SetStateAction<string[]>>
}

export function useSessionRoster(): UseSessionRosterResult {
  const [projectSessionIds, setProjectSessionIds] = useState<string[]>([])
  const [recentSessionIds, setRecentSessionIds] = useState<string[]>([])
  return {
    projectSessionIds,
    setProjectSessionIds,
    recentSessionIds,
    setRecentSessionIds,
  }
}
