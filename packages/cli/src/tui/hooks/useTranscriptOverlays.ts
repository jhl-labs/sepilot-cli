// Three small overlay/transcript flags App.tsx kept inline:
//
// - helpOpen — the F1 / /help modal visibility
// - showMcp — the MCP status panel toggle
// - transcriptClearedAt — timestamp of the last /clear; render passes
//   it to ChatView so messages older than this time are hidden until
//   the next session reset
//
// All three are simple boolean / nullable-number flags with no
// derived logic. Bundling them keeps the App.tsx hook block tidy and
// avoids three more "useState" lines drifting apart over time.

import { useState } from 'react'

export interface UseTranscriptOverlaysResult {
  helpOpen: boolean
  setHelpOpen: React.Dispatch<React.SetStateAction<boolean>>
  showMcp: boolean
  setShowMcp: React.Dispatch<React.SetStateAction<boolean>>
  transcriptClearedAt: number | null
  setTranscriptClearedAt: React.Dispatch<React.SetStateAction<number | null>>
}

export function useTranscriptOverlays(): UseTranscriptOverlaysResult {
  const [helpOpen, setHelpOpen] = useState(false)
  const [showMcp, setShowMcp] = useState(false)
  const [transcriptClearedAt, setTranscriptClearedAt] = useState<number | null>(
    null,
  )
  return {
    helpOpen,
    setHelpOpen,
    showMcp,
    setShowMcp,
    transcriptClearedAt,
    setTranscriptClearedAt,
  }
}
