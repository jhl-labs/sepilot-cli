// Bundles the five pieces of attachment-related state App.tsx kept
// inline:
//
// - attachmentSuggestions / attachmentSuggestionIndex — completion menu
//   for an in-flight @path reference
// - attachmentReindexNotice — short-lived "reindexed N entries" toast
//   shown after a Ctrl+R refresh
// - fileIndexPaths — the cached file index used to validate / suggest
//   attachment paths
// - queuedAttachmentPaths — paths the user has queued via the file
//   picker but not yet typed into the prompt
//
// The hook owns nothing but state: every consumer in App.tsx still
// reads via the same destructured names, so adopting it is a
// shape-preserving cleanup with no cascade.

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
