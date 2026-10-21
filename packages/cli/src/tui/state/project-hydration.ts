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
