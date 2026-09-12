import { resolve } from 'node:path'
import type { DaemonSessionMeta } from '@sepilotd/api-client'

function normalizedWorkspace(path: string): string {
  return resolve(path)
}

/** A resume list contains only sessions carrying this exact workspace capability. */
export function sessionBelongsToWorkspace(
  session: Pick<DaemonSessionMeta, 'cwd'>,
  workspaceRoot: string,
): boolean {
  const sessionWorkspace = session.cwd
  if (!sessionWorkspace) return false
  return normalizedWorkspace(sessionWorkspace) === normalizedWorkspace(workspaceRoot)
}

/**
 * Explicit session IDs may still target legacy, unbound sessions. A bound
 * session, however, must never be loaded into a different workspace surface.
 */
export function sessionWorkspaceLoadError(
  session: Pick<DaemonSessionMeta, 'cwd'>,
  workspaceRoot: string,
): string | null {
  if (!session.cwd || sessionBelongsToWorkspace(session, workspaceRoot)) return null
  return [
    'This session belongs to a different workspace.',
    `Session: ${session.cwd}`,
    `Current: ${normalizedWorkspace(workspaceRoot)}`,
    'Open sepilot from the session workspace, or start a new session here.',
  ].join('\n')
}
