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
    return { ok: true, projectSessionIds: [] }
  }
}
