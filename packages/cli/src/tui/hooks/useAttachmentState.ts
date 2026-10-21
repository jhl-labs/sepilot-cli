import { useState } from 'react'
import type { AttachmentCandidate } from '../utils/attachments.js'

export interface UseAttachmentStateResult {
  attachmentSuggestions: AttachmentCandidate[]
  setAttachmentSuggestions: React.Dispatch<
    React.SetStateAction<AttachmentCandidate[]>
  >
  attachmentSuggestionIndex: number
  setAttachmentSuggestionIndex: React.Dispatch<React.SetStateAction<number>>
  attachmentReindexNotice: string | null
  setAttachmentReindexNotice: React.Dispatch<
    React.SetStateAction<string | null>
  >
  fileIndexPaths: ReadonlySet<string>
  setFileIndexPaths: React.Dispatch<
    React.SetStateAction<ReadonlySet<string>>
  >
  queuedAttachmentPaths: string[]
  setQueuedAttachmentPaths: React.Dispatch<React.SetStateAction<string[]>>
}

export function useAttachmentState(): UseAttachmentStateResult {
  const [attachmentSuggestions, setAttachmentSuggestions] = useState<
    AttachmentCandidate[]
  >([])
  const [attachmentSuggestionIndex, setAttachmentSuggestionIndex] = useState(0)
  const [attachmentReindexNotice, setAttachmentReindexNotice] = useState<
    string | null
  >(null)
  const [fileIndexPaths, setFileIndexPaths] = useState<ReadonlySet<string>>(
    () => new Set<string>(),
  )
  const [queuedAttachmentPaths, setQueuedAttachmentPaths] = useState<string[]>(
    [],
  )

  return {
    attachmentSuggestions,
    setAttachmentSuggestions,
    attachmentSuggestionIndex,
    setAttachmentSuggestionIndex,
    attachmentReindexNotice,
    setAttachmentReindexNotice,
    fileIndexPaths,
    setFileIndexPaths,
    queuedAttachmentPaths,
    setQueuedAttachmentPaths,
  }
}
