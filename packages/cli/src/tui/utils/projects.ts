import { access, readFile } from 'node:fs/promises'
import { basename, dirname, join, resolve } from 'node:path'
import type { DaemonProject } from '@sepilotd/api-client'

const PROJECT_ROOT_MARKERS = [
  '.git',
  'package.json',
  'pnpm-workspace.yaml',
  'pyproject.toml',
  'Cargo.toml',
  'go.mod',
  'Gemfile',
] as const

export interface WorkspaceProject {
  rootDir: string
  name: string
  marker: string | null
}

export interface ProjectLookupResult {
  match: DaemonProject | null
  ambiguousMatches: DaemonProject[]
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path)
    return true
  } catch {
    return false
  }
}

async function detectProjectRoot(startDir: string): Promise<{
  rootDir: string
  marker: string | null
}> {
  let currentDir = resolve(startDir)

  while (true) {
    for (const marker of PROJECT_ROOT_MARKERS) {
      if (await pathExists(join(currentDir, marker))) {
        return { rootDir: currentDir, marker }
      }
    }

    const parentDir = dirname(currentDir)
    if (parentDir === currentDir) {
      return { rootDir: resolve(startDir), marker: null }
    }
    currentDir = parentDir
  }
}

async function inferWorkspaceProjectName(rootDir: string): Promise<string> {
  try {
    const raw = await readFile(join(rootDir, 'package.json'), 'utf-8')
    const parsed = JSON.parse(raw) as { name?: unknown }
    if (typeof parsed.name === 'string' && parsed.name.trim()) {
      return parsed.name.trim()
    }
  } catch {
    // Fall back to directory name.
  }

  return basename(rootDir) || rootDir
}

export async function detectWorkspaceProject(
  startDir = process.cwd(),
): Promise<WorkspaceProject> {
  const { rootDir, marker } = await detectProjectRoot(startDir)
  return {
    rootDir,
    name: await inferWorkspaceProjectName(rootDir),
    marker,
  }
}

export function findProjectBySession(
  projects: DaemonProject[],
  sessionId?: string | null,
): DaemonProject | null {
  if (!sessionId) return null
  return projects.find((project) => project.sessionIds.includes(sessionId)) ?? null
}

export function matchProjectForWorkspace(
  projects: DaemonProject[],
  workspace: WorkspaceProject,
  projectBindings?: Record<string, string>,
  sessionId?: string | null,
): DaemonProject | null {
  const linkedProject = findProjectBySession(projects, sessionId)
  if (linkedProject) {
    return linkedProject
  }

  const boundProjectId = projectBindings?.[resolve(workspace.rootDir)]
  if (boundProjectId) {
    const boundProject = projects.find((project) => project.id === boundProjectId)
    if (boundProject) {
      return boundProject
    }
  }

  const exactNameMatches = projects.filter((project) => (
    project.name.trim().toLowerCase() === workspace.name.trim().toLowerCase()
  ))

  if (exactNameMatches.length === 1) {
    return exactNameMatches[0] ?? null
  }

  return null
}

export function findProjectByName(
  projects: DaemonProject[],
  query: string,
): ProjectLookupResult {
  const normalizedQuery = query.trim().toLowerCase()
  if (!normalizedQuery) {
    return { match: null, ambiguousMatches: [] }
  }

  const exactMatch = projects.find((project) => (
    project.name.trim().toLowerCase() === normalizedQuery
  ))
  if (exactMatch) {
    return { match: exactMatch, ambiguousMatches: [] }
  }

  const partialMatches = projects.filter((project) => (
    project.name.toLowerCase().includes(normalizedQuery)
  ))

  if (partialMatches.length === 1) {
    return { match: partialMatches[0] ?? null, ambiguousMatches: [] }
  }

  return {
    match: null,
    ambiguousMatches: partialMatches,
  }
}
