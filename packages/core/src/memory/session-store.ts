import type { SessionId } from '../types/common.js'
import type { SessionEvent, SessionMeta, ToolCallEvent, ToolResultEvent } from './types.js'
import type { PaginatedResult, PaginationParams } from '../types/pagination.js'

export interface ISessionStore {
  init?(): Promise<void>
  create(meta: Omit<SessionMeta, 'messageCount' | 'totalTokens' | 'totalCost'>): Promise<SessionMeta>
  get(id: SessionId): Promise<SessionMeta | null>
  list(params?: PaginationParams & {
    query?: string
    workspaceRoot?: string
  }): Promise<PaginatedResult<SessionMeta>>
  delete(id: SessionId): Promise<void>
  appendEvent(sessionId: SessionId, event: SessionEvent): Promise<void>
  getEvents(sessionId: SessionId): Promise<SessionEvent[]>
  updateTitleIfUnchanged?(
    sessionId: SessionId,
    title: string,
    expected: Pick<SessionMeta, 'title' | 'createdAt'>,
  ): Promise<SessionMeta | null>
  getToolResult?(
    sessionId: SessionId,
    toolCallId: string,
    options?: { signal?: AbortSignal },
  ): Promise<{ call: ToolCallEvent; result: ToolResultEvent } | null>
  replaceEvents?(sessionId: SessionId, events: SessionEvent[]): Promise<void>
  updateMeta?(
    sessionId: SessionId,
    patch: {
      title?: string
      status?: SessionMeta['status']
      cwd?: string | null
      workspaceIsolation?: 'policy' | 'strict'
      provider?: string
      model?: string
      personaIds?: string[]
      memoryNamespace?: string
      preferPromptReact?: boolean
      starred?: boolean
      tags?: string[]
    },
  ): Promise<SessionMeta | null>
}
