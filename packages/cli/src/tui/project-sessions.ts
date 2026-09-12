import { createHash } from 'node:crypto'
import {
  mkdir,
  readFile,
  readdir,
  rm,
  writeFile,
} from 'node:fs/promises'
import { homedir } from 'node:os'
import { basename, join, resolve } from 'node:path'
import type { WorkspaceProject } from './utils/projects.js'

const MAX_PROJECT_SESSIONS = 100

function getProjectSessionsRoot(): string {
  return join(homedir(), '.sepilot', 'projects')
}

export interface ProjectSessionEntry {
  sessionId: string
  title: string
  provider: string
  model: string
  updatedAt: string
  lastUsedAt: string
  projectId?: string | null
  projectName?: string | null
}

interface ProjectSessionManifest {
  version: 2
  workspaceRoot: string
  workspaceName: string
  updatedAt: string
  sessionIds: string[]
}

export interface ProjectSessionRegistry {
  version: 2
  workspaceRoot: string
  workspaceName: string
  updatedAt: string
  sessions: ProjectSessionEntry[]
}

function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    || 'workspace'
}

function workspaceKey(workspace: WorkspaceProject): string {
  const normalizedRoot = resolve(workspace.rootDir)
  const hash = createHash('sha1')
    .update(normalizedRoot)
    .digest('hex')
    .slice(0, 10)
  const base = slugify(workspace.name || basename(normalizedRoot))
  return `${base}-${hash}`
}

function getWorkspaceDirectory(workspace: WorkspaceProject): string {
  return join(
    getProjectSessionsRoot(),
    workspaceKey(workspace),
  )
}

function getLegacyProjectSessionRegistryPath(
  workspace: WorkspaceProject,
): string {
  return join(getWorkspaceDirectory(workspace), 'sessions.json')
}

export function getProjectSessionRegistryPath(
  workspace: WorkspaceProject,
): string {
  return join(getWorkspaceDirectory(workspace), 'project.json')
}

export function getProjectSessionDirectory(
  workspace: WorkspaceProject,
): string {
  return join(getWorkspaceDirectory(workspace), 'sessions')
}

export function getProjectSessionEntryPath(
  workspace: WorkspaceProject,
  sessionId: string,
): string {
  return join(
    getProjectSessionDirectory(workspace),
    `${encodeURIComponent(sessionId)}.json`,
  )
}

function normalizeEntry(entry: ProjectSessionEntry): ProjectSessionEntry | null {
  if (!entry.sessionId?.trim()) {
    return null
  }

  const timestamp = entry.lastUsedAt || entry.updatedAt || new Date().toISOString()

  return {
    sessionId: entry.sessionId.trim(),
    title: entry.title?.trim() || '(untitled)',
    provider: entry.provider?.trim() || 'unknown',
    model: entry.model?.trim() || 'unknown',
    updatedAt: entry.updatedAt || timestamp,
    lastUsedAt: timestamp,
    projectId: entry.projectId?.trim() || null,
    projectName: entry.projectName?.trim() || null,
  }
}

function normalizeRegistry(
  workspace: WorkspaceProject,
  registry: Partial<ProjectSessionRegistry> | null | undefined,
): ProjectSessionRegistry {
  const deduped = new Map<string, ProjectSessionEntry>()

  for (const rawEntry of registry?.sessions ?? []) {
    const entry = normalizeEntry(rawEntry)
    if (!entry) continue
    deduped.set(entry.sessionId, entry)
  }

  const sessions = [...deduped.values()]
    .sort((left, right) => (
      right.lastUsedAt.localeCompare(left.lastUsedAt)
      || right.updatedAt.localeCompare(left.updatedAt)
    ))
    .slice(0, MAX_PROJECT_SESSIONS)

  return {
    version: 2,
    workspaceRoot: resolve(workspace.rootDir),
    workspaceName: workspace.name,
    updatedAt: registry?.updatedAt || new Date().toISOString(),
    sessions,
  }
}

async function readProjectSessionEntry(
  workspace: WorkspaceProject,
  sessionId: string,
): Promise<ProjectSessionEntry | null> {
  try {
    const raw = await readFile(getProjectSessionEntryPath(workspace, sessionId), 'utf8')
    return normalizeEntry(JSON.parse(raw) as ProjectSessionEntry)
  } catch {
    return null
  }
}

async function migrateLegacyRegistry(
  workspace: WorkspaceProject,
): Promise<void> {
  const legacyPath = getLegacyProjectSessionRegistryPath(workspace)
  const manifestPath = getProjectSessionRegistryPath(workspace)

  try {
    await readFile(manifestPath, 'utf8')
    return
  } catch {
    // Continue and attempt legacy migration.
  }

  try {
    const raw = await readFile(legacyPath, 'utf8')
    const legacy = normalizeRegistry(
      workspace,
      JSON.parse(raw) as Partial<ProjectSessionRegistry>,
    )
    await saveProjectSessionRegistry(workspace, legacy)
    await rm(legacyPath, { force: true })
  } catch {
    // No legacy registry to migrate.
  }
}

export async function loadProjectSessionRegistry(
  workspace: WorkspaceProject,
): Promise<ProjectSessionRegistry> {
  await migrateLegacyRegistry(workspace)

  const manifestPath = getProjectSessionRegistryPath(workspace)
  try {
    const raw = await readFile(manifestPath, 'utf8')
    const manifest = JSON.parse(raw) as ProjectSessionManifest
    const sessions = await Promise.all(
      (manifest.sessionIds ?? []).map((sessionId) => readProjectSessionEntry(workspace, sessionId)),
    )
    return normalizeRegistry(workspace, {
      updatedAt: manifest.updatedAt,
      sessions: sessions.filter((entry): entry is ProjectSessionEntry => entry !== null),
    })
  } catch {
    return normalizeRegistry(workspace, null)
  }
}

async function saveProjectSessionRegistry(
  workspace: WorkspaceProject,
  registry: ProjectSessionRegistry,
): Promise<void> {
  const normalized = normalizeRegistry(workspace, registry)
  const workspaceDir = getWorkspaceDirectory(workspace)
  const manifestPath = getProjectSessionRegistryPath(workspace)
  const sessionsDir = getProjectSessionDirectory(workspace)

  await mkdir(workspaceDir, { recursive: true })
  await mkdir(sessionsDir, { recursive: true })

  const manifest: ProjectSessionManifest = {
    version: 2,
    workspaceRoot: normalized.workspaceRoot,
    workspaceName: normalized.workspaceName,
    updatedAt: normalized.updatedAt,
    sessionIds: normalized.sessions.map((session) => session.sessionId),
  }

  await writeFile(
    manifestPath,
    JSON.stringify(manifest, null, 2),
    'utf8',
  )

  await Promise.all(
    normalized.sessions.map((session) => (
      writeFile(
        getProjectSessionEntryPath(workspace, session.sessionId),
        JSON.stringify(session, null, 2),
        'utf8',
      )
    )),
  )

  const activeFiles = new Set(
    normalized.sessions.map((session) => `${encodeURIComponent(session.sessionId)}.json`),
  )
  const existingFiles = await readdir(sessionsDir).catch(() => [])
  await Promise.all(
    existingFiles
      .filter((filename) => filename.endsWith('.json') && !activeFiles.has(filename))
      .map((filename) => rm(join(sessionsDir, filename), { force: true })),
  )
}

export async function recordProjectSessionAccess(
  workspace: WorkspaceProject,
  entry: Omit<ProjectSessionEntry, 'lastUsedAt'> & { lastUsedAt?: string },
): Promise<ProjectSessionRegistry> {
  const current = await loadProjectSessionRegistry(workspace)
  const timestamp = entry.lastUsedAt || new Date().toISOString()
  const next = normalizeRegistry(workspace, {
    ...current,
    updatedAt: timestamp,
    sessions: [
      {
        ...entry,
        lastUsedAt: timestamp,
      },
      ...current.sessions.filter((session) => session.sessionId !== entry.sessionId),
    ],
  })
  await saveProjectSessionRegistry(workspace, next)
  return next
}

export async function removeProjectSession(
  workspace: WorkspaceProject,
  sessionId: string,
): Promise<ProjectSessionRegistry> {
  const current = await loadProjectSessionRegistry(workspace)
  const next = normalizeRegistry(workspace, {
    ...current,
    updatedAt: new Date().toISOString(),
    sessions: current.sessions.filter((session) => session.sessionId !== sessionId),
  })
  await saveProjectSessionRegistry(workspace, next)
  return next
}
