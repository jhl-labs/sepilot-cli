// Pulled out of App.tsx so the "after-the-session-id-changes" registry
// sync effect can read top-down: fetch session metadata best-effort,
// record into the local project-session registry, return the
// resulting session id list. Both daemon and registry IO are
// best-effort — the helper never throws.

import type { DaemonSessionDetail } from '@sepilotd/api-client'
import { recordProjectSessionAccess } from '../project-sessions.js'
import type { WorkspaceProject } from '../utils/projects.js'

export interface ProjectSessionSyncHttpClient {
  session(id: string): Promise<DaemonSessionDetail>
}

export interface ProjectSessionSyncFallback {
  provider: string
  model: string
  projectId: string | null
  projectName: string | null
}

export type ProjectSessionSyncResult =
  | { ok: true; cancelled?: false; projectSessionIds: string[] }
  | { ok: true; cancelled: true }

/**
 * Refresh the local project-session registry for `sessionId` belonging
 * to `workspace`. Tries the daemon for the session's title /
 * provider / model / updatedAt first; falls back to the supplied
 * fallback fields when that call fails. Returns the registry's
 * resulting session id list so the caller can update the picker
 * counter without re-listing the registry.
 *
 * The ok=true / cancelled=true tagged shape mirrors the other
 * bootstrap helpers; cancellation is checked once after the daemon
 * fetch so a stale effect run can drop on the floor before mutating
 * disk.
 */
export async function syncProjectSessionRegistry(opts: {
  httpClient: ProjectSessionSyncHttpClient
  workspace: WorkspaceProject
  sessionId: string
  fallback: ProjectSessionSyncFallback
  cancelled: () => boolean
}): Promise<ProjectSessionSyncResult> {
  let title = `(session ${opts.sessionId.slice(0, 8)})`
  let provider = opts.fallback.provider || 'unknown'
  let model = opts.fallback.model || 'unknown'
  let updatedAt = new Date().toISOString()

  try {
    const session = await opts.httpClient.session(opts.sessionId)
    if (opts.cancelled()) return { ok: true, cancelled: true }
    title = session.title?.trim() || title
    provider = session.provider || provider
    model = session.model || model
    updatedAt = session.updatedAt || updatedAt
  } catch {
    if (opts.cancelled()) return { ok: true, cancelled: true }
    // Best-effort: missing metadata is fine; we still record the access.
  }

  try {
    const registry = await recordProjectSessionAccess(opts.workspace, {
      sessionId: opts.sessionId,
      title,
      provider,
      model,
      updatedAt,
      projectId: opts.fallback.projectId,
      projectName: opts.fallback.projectName,
    })
    if (opts.cancelled()) return { ok: true, cancelled: true }
    return {
      ok: true,
      projectSessionIds: registry.sessions.map((session) => session.sessionId),
    }
  } catch {
    if (opts.cancelled()) return { ok: true, cancelled: true }
    // Disk write failed — return the previous list shape (empty) to
    // signal "no change to record." Caller leaves projectSessionIds
    // untouched in this branch.
    return { ok: true, projectSessionIds: [] }
  }
}
