import type { SessionId } from '../types/common.js'
import type { SessionEvent, SessionMeta } from './types.js'
import type { PaginatedResult, PaginationParams } from '../types/pagination.js'

export interface ISessionStore {
  init?(): Promise<void>
  create(meta: Omit<SessionMeta, 'messageCount' | 'totalTokens' | 'totalCost'>): Promise<SessionMeta>
  get(id: SessionId): Promise<SessionMeta | null>
  list(params?: PaginationParams & {
    query?: string
    /** Exact immutable workspace binding used to scope resumable history. */
    workspaceRoot?: string
  }): Promise<PaginatedResult<SessionMeta>>
  delete(id: SessionId): Promise<void>
  appendEvent(sessionId: SessionId, event: SessionEvent): Promise<void>
  getEvents(sessionId: SessionId): Promise<SessionEvent[]>
  /**
   * Replace the persisted event log for a session.
   * Used by /undo and /redo when the daemon needs to rewind or
   * restore a turn boundary. Optional so non-jsonl stores can opt in
   * later.
   */
  replaceEvents?(sessionId: SessionId, events: SessionEvent[]): Promise<void>
  /**
   * Patch persisted session metadata (`title` for renames, `status`
   * for archive/reactivate flows, `cwd` for the active workspace).
   * The implementation should reject
   * unknown fields and avoid touching derived counters such as
   * messageCount/totalTokens. Optional so external stores can opt in
   * incrementally.
   */
  updateMeta?(
    sessionId: SessionId,
    patch: {
      title?: string
      status?: SessionMeta['status']
      /** `null` explicitly clears the workspace bound to this session. */
      cwd?: string | null
      workspaceIsolation?: 'policy' | 'strict'
      provider?: string
      model?: string
      /** Ordered, de-duplicated roster. `[]` explicitly clears it. */
      personaIds?: string[]
      /** Server-bound isolated persona memory identity. */
      memoryNamespace?: string
      preferPromptReact?: boolean
      starred?: boolean
      tags?: string[]
    },
  ): Promise<SessionMeta | null>
}
