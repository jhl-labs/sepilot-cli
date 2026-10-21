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
