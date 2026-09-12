// Two related id lists App.tsx hydrates at boot and keeps in sync as
// the user moves between sessions:
//
// - projectSessionIds — sessions belonging to the currently selected
//   project, used by the session picker to highlight grouping
// - recentSessionIds — global "most recent N sessions" sorted by last
//   access, used by Ctrl+Shift+←/→ navigation and the session picker
//   header
//
// Neither has any computed/derived behaviour; the hook just owns the
// two useStates so App.tsx adopts them as one destructured caller.

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
