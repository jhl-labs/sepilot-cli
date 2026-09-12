// Pulled out of App.tsx so the connect-on-mount project-discovery
// branch — list -> match-by-workspace-or-session -> auto-create when
// a workspace is detected — can live as a single async helper with a
// proper result type and be unit-tested without rendering App.
//
// Two helper exports also move with this module:
//   - sortProjects: deterministic newest-first ordering (the picker
//     and /project info both rely on it)
//   - upsertProject: dedup + sort, used after createProject and
//     attachSessionToProject responses to refresh the local state

import type { DaemonProject } from '@sepilotd/api-client'
import {
  findProjectBySession,
  matchProjectForWorkspace,
  type WorkspaceProject,
} from '../utils/projects.js'

type ProjectBindings = Record<string, string>

export function sortProjects(projects: DaemonProject[]): DaemonProject[] {
  return [...projects].sort((left, right) => (
    new Date(right.updatedAt).getTime() - new Date(left.updatedAt).getTime()
  ))
}

export function upsertProject(
  projects: DaemonProject[],
  project: DaemonProject,
): DaemonProject[] {
  return sortProjects([
    project,
    ...projects.filter((item) => item.id !== project.id),
  ])
}

export interface ProjectHydrationHttpClient {
  projects(): Promise<DaemonProject[]>
  createProject(input: { name: string }): Promise<DaemonProject>
}

type InitialProjectSelection = {
  project: DaemonProject
  source: 'session' | 'workspace'
  workspaceRootDir?: string
}

export type ProjectHydrationResult =
  | {
      ok: true
      cancelled?: false
      availableProjects: DaemonProject[]
      selection: InitialProjectSelection | null
    }
  | { ok: true; cancelled: true }
  | { ok: false }

/**
 * Resolve the initial project selection at boot time.
 *
 * Steps:
 *   1. fetch and sort the daemon's project list
 *   2. pick the right initial project — match by workspace if we have
 *      one, else find by session id
 *   3. if neither matched and a workspace was detected, create a new
 *      project named after the workspace and use that
 *
 * Returns the available projects so the caller can store them, plus
 * the selection (or null when no workspace and no project matched).
 * A failure anywhere in the chain returns ok=false — project discovery
 * is best-effort and must not block chat boot.
 */
export async function resolveInitialProject(opts: {
  httpClient: ProjectHydrationHttpClient
  detectedWorkspace: WorkspaceProject | null
  initialSessionId?: string
  projectBindings?: ProjectBindings
  cancelled: () => boolean
}): Promise<ProjectHydrationResult> {
  let availableProjects: DaemonProject[]
  try {
    availableProjects = sortProjects(await opts.httpClient.projects())
  } catch {
    return { ok: false }
  }
  if (opts.cancelled()) return { ok: true, cancelled: true }

  const matched = opts.detectedWorkspace
    ? matchProjectForWorkspace(
        availableProjects,
        opts.detectedWorkspace,
        opts.projectBindings,
        opts.initialSessionId,
      )
    : findProjectBySession(availableProjects, opts.initialSessionId)

  if (matched) {
    const linkedToSession = Boolean(
      opts.initialSessionId
      && matched.sessionIds.includes(opts.initialSessionId),
    )
    return {
      ok: true,
      availableProjects,
      selection: {
        project: matched,
        source: linkedToSession ? 'session' : 'workspace',
        workspaceRootDir: opts.detectedWorkspace?.rootDir,
      },
    }
  }

  if (!opts.detectedWorkspace) {
    return { ok: true, availableProjects, selection: null }
  }

  let createdProject: DaemonProject
  try {
    createdProject = await opts.httpClient.createProject({
      name: opts.detectedWorkspace.name,
    })
  } catch {
    return { ok: true, availableProjects, selection: null }
  }
  if (opts.cancelled()) return { ok: true, cancelled: true }

  return {
    ok: true,
    availableProjects: upsertProject(availableProjects, createdProject),
    selection: {
      project: createdProject,
      source: 'workspace',
      workspaceRootDir: opts.detectedWorkspace.rootDir,
    },
  }
}
