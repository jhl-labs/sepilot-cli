export interface DocSession {
  id: string
  path: string | null
  content: string
  version: number
  dirty: boolean
  diskMtimeMs: number | null
  createdAt: number
  updatedAt: number
}

export interface DocOutlineEntry {
  level: number
  title: string
  start: number
  end: number
  index: number
}

export interface DocChange {
  start: number
  end: number
  newText: string
}

export type DocChangeAuthor = 'user' | 'llm' | 'external'

export interface DocUpdateEvent {
  type: 'doc.updated'
  sessionId: string
  changes: DocChange[]
  version: number
  author: DocChangeAuthor
  toolName?: string
  label?: string
}

export interface DocDiffPreview {
  id: string
  sessionId: string
  baseVersion: number
  summary: string
  changes: DocChange[]
  createdAt: number
}

export interface DocDiffPendingEvent {
  type: 'doc.diff_pending'
  sessionId: string
  preview: DocDiffPreview
}

export interface DocDiffResolvedEvent {
  type: 'doc.diff_resolved'
  sessionId: string
  previewId: string
  resolution: 'accepted' | 'cancelled' | 'expired'
}

export interface DocClosedEvent {
  type: 'doc.closed'
  sessionId: string
  reason?: string
}

export type DocEvent =
  | DocUpdateEvent
  | DocDiffPendingEvent
  | DocDiffResolvedEvent
  | DocClosedEvent

export interface DocHistoryEntry {
  version: number
  reverseChanges: DocChange[]
  author: DocChangeAuthor
  toolName?: string
  at: number
}
