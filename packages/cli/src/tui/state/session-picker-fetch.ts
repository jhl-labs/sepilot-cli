// Pulled out of App.tsx so the session-picker fetch logic — list +
// (when no search query) augment with workspace-project sessions
// missing from the page — can live as a single async function with a
// proper result type instead of as an inline useEffect.
//
// Two extra wrinkles compared to a straight list call:
//
//   1. When the operator hasn't typed a query, we want to surface the
//      workspace project's recent sessions even if they fall off the
//      list page. We fetch their detail records one-by-one (capped to
//      PROJECT_SESSION_PICKER_FETCH_LIMIT) and merge them in.
//   2. Cancellation: a long-running augment can outlive the picker
//      being closed or another query being typed; the caller passes a
//      cancelled() predicate that we check at every await boundary so
//      a stale fetch never updates the surface.

import type {
  DaemonSessionDetail,
  DaemonSessionMeta,
} from '@sepilotd/api-client'
import {
  formatApiError,
  friendlyErrorMessage,
  isRecoverableStreamDrop,
} from '../../utils/error-message.js'
import type { WorkspaceProject } from '../utils/projects.js'
import { sessionBelongsToWorkspace } from '../utils/session-workspace.js'

const SESSION_PICKER_PAGE_SIZE = 100
const PROJECT_SESSION_PICKER_FETCH_LIMIT = 20

export interface SessionPickerHttpClient {
  sessions(
    query?: string,
    opts?: { perPage?: number; workspaceRoot?: string },
  ): Promise<{ items: DaemonSessionMeta[] }>
  session(id: string): Promise<DaemonSessionDetail>
}

export type SessionPickerFetchResult =
  | { ok: true; items: DaemonSessionMeta[]; cancelled?: false }
  | { ok: true; cancelled: true }
  | { ok: false; error: string }

export function formatSessionPickerError(error: unknown): string {
  const apiError = formatApiError(error)
  if (apiError) {
    return `Couldn't load sessions: ${apiError.split('\n')[0]} — press r to retry`
  }

  const raw = friendlyErrorMessage(error)
  if (
    isRecoverableStreamDrop(error)
    || /unable to connect|computer able to access the url/i.test(raw)
  ) {
    return "Couldn't load sessions: daemon is unreachable — press r to retry"
  }

  const summary = raw.split('\n').find((line) => line.trim().length > 0)?.trim()
    ?? 'Failed to load sessions.'
  return `Couldn't load sessions: ${summary} — press r to retry`
}

export async function loadSessionPickerItems(opts: {
  httpClient: SessionPickerHttpClient
  workspaceRoot: string
  workspaceProject: WorkspaceProject | null
  sessionQuery: string
  cancelled: () => boolean
  refreshProjectSessionIds: (
    workspace: WorkspaceProject,
  ) => Promise<string[]>
}): Promise<SessionPickerFetchResult> {
  const trimmed = opts.sessionQuery.trim()

  let workspaceSessionIds: string[] = []
  if (!trimmed && opts.workspaceProject) {
    workspaceSessionIds = await opts.refreshProjectSessionIds(opts.workspaceProject)
    if (opts.cancelled()) return { ok: true, cancelled: true }
  }

  let listItems: DaemonSessionMeta[]
  try {
    const result = await opts.httpClient.sessions(
      trimmed || undefined,
      {
        perPage: SESSION_PICKER_PAGE_SIZE,
        workspaceRoot: opts.workspaceRoot,
      },
    )
    if (opts.cancelled()) return { ok: true, cancelled: true }
    // Keep the client-side guard for compatibility with older daemons that
    // may ignore an unknown workspaceRoot query parameter.
    listItems = result.items.filter((session) => (
      sessionBelongsToWorkspace(session, opts.workspaceRoot)
    ))
  } catch (error) {
    if (opts.cancelled()) return { ok: true, cancelled: true }
    return {
      ok: false,
      error: formatSessionPickerError(error),
    }
  }

  const sessionById = new Map(
    listItems.map((session) => [session.id, session] as const),
  )

  if (!trimmed && workspaceSessionIds.length > 0) {
    const missingProjectSessionIds = workspaceSessionIds
      .filter((sessionId) => !sessionById.has(sessionId))
      .slice(0, PROJECT_SESSION_PICKER_FETCH_LIMIT)

    if (missingProjectSessionIds.length > 0) {
      const missingSessions = await Promise.all(
        missingProjectSessionIds.map(async (sessionId) => {
          try {
            return await opts.httpClient.session(sessionId)
          } catch {
            return null
          }
        }),
      )
      if (opts.cancelled()) return { ok: true, cancelled: true }
      for (const session of missingSessions) {
        if (session && sessionBelongsToWorkspace(session, opts.workspaceRoot)) {
          sessionById.set(session.id, session)
        }
      }
    }
  }

  const items: DaemonSessionMeta[] = [
    ...listItems,
    ...[...sessionById.values()].filter((session) => (
      !listItems.some((existing) => existing.id === session.id)
    )),
  ]
  return { ok: true, items }
}
