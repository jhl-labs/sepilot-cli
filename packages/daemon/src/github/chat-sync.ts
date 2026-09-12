import { execFile } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { promisify } from 'node:util'
import type { ISessionStore, SessionEvent, SessionMeta } from '@sepilotd/core'
import { sepilotdHome } from '../storage/home.js'
import type { EncryptionManager } from '../security/encryption.js'
import { secureDir, secureFile } from '../utils/secure-file.js'

const execFileAsync = promisify(execFile)

const DEFAULT_BRANCH = 'main'
const CHAT_SYNC_VERSION = 1
const GIT_TIMEOUT_MS = 120_000
const GIT_MAX_BUFFER = 10 * 1024 * 1024

export interface GitHubChatSyncConfig {
  enabled: boolean
  repoFullName: string | null
  branch: string
  lastSyncedAt: number | null
  lastSyncStatus: 'success' | 'error' | null
  lastSyncError: string | null
  lastCommit: string | null
  syncedSessions: number
  updatedAt: number | null
}

export interface UpdateGitHubChatSyncConfigInput {
  enabled?: boolean
  repoFullName?: string | null
  branch?: string
}

export interface GitHubChatSyncResult {
  startedAt: number
  finishedAt: number
  repoFullName: string
  branch: string
  worktree: string
  syncedSessions: number
  changedFiles: number
  pushed: boolean
  commitSha: string | null
}

export interface ExportEncryptedChatSessionsOptions {
  sessions: ISessionStore
  encryption: EncryptionManager
  rootDir: string
  deviceName: string
}

interface ChatSyncManifestEntry {
  file: string
  updatedAt: string
  eventCount: number
}

interface ChatSyncManifest {
  version: number
  kind: 'sepilotd.chat-sync.manifest'
  generatedAt: string
  deviceRef: string
  sessionCount: number
  sessions: ChatSyncManifestEntry[]
}

interface EncryptedSessionEnvelope {
  version: number
  kind: 'sepilotd.chat-session.encrypted'
  algorithm: 'aes-256-gcm'
  updatedAt: string
  eventCount: number
  ciphertext: string
}

interface SessionSnapshot {
  version: number
  exportedAt: string
  session: SessionMeta
  events: SessionEvent[]
}

function githubDir(): string {
  const d = join(sepilotdHome(), 'github')
  mkdirSync(d, { recursive: true, mode: 0o700 })
  secureDir(d)
  return d
}

function configFile(): string {
  return join(githubDir(), 'chat-sync.json')
}

function worktreesDir(): string {
  const d = join(githubDir(), 'chat-sync-worktrees')
  mkdirSync(d, { recursive: true, mode: 0o700 })
  secureDir(d)
  return d
}

function emptyConfig(): GitHubChatSyncConfig {
  return {
    enabled: false,
    repoFullName: null,
    branch: DEFAULT_BRANCH,
    lastSyncedAt: null,
    lastSyncStatus: null,
    lastSyncError: null,
    lastCommit: null,
    syncedSessions: 0,
    updatedAt: null,
  }
}

function loadConfig(): GitHubChatSyncConfig {
  const file = configFile()
  if (!existsSync(file)) return emptyConfig()
  try {
    return normalizeConfig(JSON.parse(readFileSync(file, 'utf-8')) as Partial<GitHubChatSyncConfig>)
  } catch {
    return emptyConfig()
  }
}

function saveConfig(config: GitHubChatSyncConfig): GitHubChatSyncConfig {
  const normalized = normalizeConfig(config)
  writeFileSync(configFile(), JSON.stringify(normalized, null, 2), 'utf-8')
  secureFile(configFile())
  return normalized
}

function normalizeConfig(input: Partial<GitHubChatSyncConfig>): GitHubChatSyncConfig {
  const fallback = emptyConfig()
  const repoFullName =
    typeof input.repoFullName === 'string' && input.repoFullName.trim()
      ? normalizeRepoFullName(input.repoFullName)
      : null
  const branch =
    typeof input.branch === 'string' && input.branch.trim()
      ? normalizeBranch(input.branch)
      : fallback.branch
  return {
    enabled: Boolean(input.enabled),
    repoFullName,
    branch,
    lastSyncedAt: typeof input.lastSyncedAt === 'number' ? input.lastSyncedAt : null,
    lastSyncStatus:
      input.lastSyncStatus === 'success' || input.lastSyncStatus === 'error'
        ? input.lastSyncStatus
        : null,
    lastSyncError: typeof input.lastSyncError === 'string' ? input.lastSyncError : null,
    lastCommit: typeof input.lastCommit === 'string' ? input.lastCommit : null,
    syncedSessions: typeof input.syncedSessions === 'number' ? input.syncedSessions : 0,
    updatedAt: typeof input.updatedAt === 'number' ? input.updatedAt : null,
  }
}

function normalizeRepoFullName(value: string): string {
  const trimmed = value.trim().replace(/^https:\/\/github\.com\//i, '').replace(/\.git$/i, '')
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(trimmed)) {
    throw new Error('GitHub repository must be in owner/repo form.')
  }
  return trimmed
}

function normalizeBranch(value: string): string {
  const trimmed = value.trim()
  if (
    !trimmed ||
    trimmed.startsWith('-') ||
    trimmed.includes('..') ||
    trimmed.includes('//') ||
    trimmed.endsWith('.lock') ||
    !/^[A-Za-z0-9._/-]+$/.test(trimmed)
  ) {
    throw new Error('GitHub chat sync branch contains unsafe characters.')
  }
  return trimmed
}

function repoWorktree(repoFullName: string): string {
  return join(worktreesDir(), repoFullName.replace(/[^A-Za-z0-9_.-]+/g, '__'))
}

function deviceRef(deviceName: string): string {
  return createHash('sha256').update(deviceName || 'default').digest('hex').slice(0, 16)
}

function sessionFileRef(sessionId: string): string {
  return createHash('sha256').update(sessionId).digest('hex')
}

async function listAllSessions(sessions: ISessionStore): Promise<SessionMeta[]> {
  const all: SessionMeta[] = []
  let page = 1
  const perPage = 200
  for (;;) {
    const result = await sessions.list({ page, perPage })
    all.push(...result.items)
    if (!result.hasNextPage) return all
    page += 1
  }
}

async function writeJsonIfChanged(path: string, value: unknown): Promise<boolean> {
  const next = `${JSON.stringify(value, null, 2)}\n`
  return writeTextIfChanged(path, next)
}

async function writeTextIfChanged(path: string, next: string): Promise<boolean> {
  const current = await readFile(path, 'utf-8').catch(() => null)
  if (current === next) return false
  await mkdir(dirname(path), { recursive: true })
  await writeFile(path, next, 'utf-8')
  return true
}

async function writeEncryptedSession(
  rootDir: string,
  prefix: string,
  encryption: EncryptionManager,
  session: SessionMeta,
  events: SessionEvent[],
  exportedAt: string,
): Promise<ChatSyncManifestEntry> {
  const ref = sessionFileRef(session.id)
  const relativeFile = `${prefix}/sessions/${ref.slice(0, 2)}/${ref}.json`
  const target = join(rootDir, relativeFile)
  const snapshot: SessionSnapshot = {
    version: CHAT_SYNC_VERSION,
    exportedAt,
    session,
    events,
  }
  const existing = await readExistingEnvelope(target)
  if (!existing || !encryptedSnapshotMatches(existing, encryption, snapshot)) {
    const envelope: EncryptedSessionEnvelope = {
      version: CHAT_SYNC_VERSION,
      kind: 'sepilotd.chat-session.encrypted',
      algorithm: 'aes-256-gcm',
      updatedAt: session.updatedAt,
      eventCount: events.length,
      ciphertext: encryption.encrypt(JSON.stringify(snapshot)),
    }
    await writeJsonIfChanged(target, envelope)
  }
  return {
    file: relativeFile,
    updatedAt: session.updatedAt,
    eventCount: events.length,
  }
}

function encryptedSnapshotMatches(
  envelope: EncryptedSessionEnvelope,
  encryption: EncryptionManager,
  next: SessionSnapshot,
): boolean {
  if (envelope.updatedAt !== next.session.updatedAt || envelope.eventCount !== next.events.length) {
    return false
  }
  try {
    const current = JSON.parse(encryption.decrypt(envelope.ciphertext)) as Partial<SessionSnapshot>
    return JSON.stringify({ session: current.session, events: current.events })
      === JSON.stringify({ session: next.session, events: next.events })
  } catch {
    return false
  }
}

async function readExistingEnvelope(path: string): Promise<EncryptedSessionEnvelope | null> {
  try {
    const parsed = JSON.parse(await readFile(path, 'utf-8')) as Partial<EncryptedSessionEnvelope>
    if (
      parsed.kind === 'sepilotd.chat-session.encrypted' &&
      typeof parsed.updatedAt === 'string' &&
      typeof parsed.eventCount === 'number'
    ) {
      return parsed as EncryptedSessionEnvelope
    }
  } catch {
    // Missing or invalid existing envelopes are overwritten by the next export.
  }
  return null
}

async function readExistingManifest(path: string): Promise<ChatSyncManifest | null> {
  try {
    const parsed = JSON.parse(await readFile(path, 'utf-8')) as Partial<ChatSyncManifest>
    if (
      parsed.kind === 'sepilotd.chat-sync.manifest' &&
      typeof parsed.generatedAt === 'string' &&
      typeof parsed.deviceRef === 'string' &&
      Array.isArray(parsed.sessions)
    ) {
      return parsed as ChatSyncManifest
    }
  } catch {
    // Missing or invalid manifests are replaced below.
  }
  return null
}

export async function exportEncryptedChatSessions(
  options: ExportEncryptedChatSessionsOptions,
): Promise<ChatSyncManifest> {
  if (!options.encryption.isEnabled()) {
    throw new Error('Chat sync requires the local data encryption key to be loaded.')
  }

  const exportedAt = new Date().toISOString()
  const ref = deviceRef(options.deviceName)
  const prefix = `devices/${ref}`
  await mkdir(options.rootDir, { recursive: true })

  const sessionMetas = await listAllSessions(options.sessions)
  const entries: ChatSyncManifestEntry[] = []
  for (const session of sessionMetas) {
    const events = await options.sessions.getEvents(session.id)
    entries.push(
      await writeEncryptedSession(
        options.rootDir,
        prefix,
        options.encryption,
        session,
        events,
        exportedAt,
      ),
    )
  }

  entries.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
  const manifestPath = join(options.rootDir, prefix, 'manifest.json')
  const existingManifest = await readExistingManifest(manifestPath)
  const entryFiles = new Set(entries.map((entry) => entry.file))
  for (const oldEntry of existingManifest?.sessions ?? []) {
    if (!entryFiles.has(oldEntry.file)) {
      await rm(join(options.rootDir, oldEntry.file), { force: true })
    }
  }
  const manifestChanged =
    !existingManifest ||
    existingManifest.version !== CHAT_SYNC_VERSION ||
    existingManifest.deviceRef !== ref ||
    JSON.stringify(existingManifest.sessions) !== JSON.stringify(entries)
  const manifest: ChatSyncManifest = {
    version: CHAT_SYNC_VERSION,
    kind: 'sepilotd.chat-sync.manifest',
    generatedAt: manifestChanged ? exportedAt : (existingManifest?.generatedAt ?? exportedAt),
    deviceRef: ref,
    sessionCount: entries.length,
    sessions: entries,
  }
  await writeJsonIfChanged(manifestPath, manifest)
  await writeTextIfChanged(
    join(options.rootDir, 'README.md'),
    [
      '# sepilotd encrypted chat sync',
      '',
      'This repository contains encrypted sepilotd chat session payloads only.',
      '',
      '- Keep `~/.sepilotd/security/data.key` private. It is not stored in this repository.',
      '- Each session JSON file contains AES-256-GCM ciphertext and non-content sync metadata.',
      '- GitHub access controls protect repository access; the data key protects message contents.',
      '',
    ].join('\n'),
  )
  return manifest
}

export function getChatSyncConfig(): GitHubChatSyncConfig {
  return loadConfig()
}

export function updateChatSyncConfig(input: UpdateGitHubChatSyncConfigInput): GitHubChatSyncConfig {
  const current = loadConfig()
  const next = normalizeConfig({
    ...current,
    ...('enabled' in input ? { enabled: Boolean(input.enabled) } : {}),
    ...('repoFullName' in input
      ? {
        repoFullName:
          typeof input.repoFullName === 'string' && input.repoFullName.trim()
            ? input.repoFullName
            : null,
      }
      : {}),
    ...('branch' in input && input.branch ? { branch: input.branch } : {}),
    updatedAt: Date.now(),
  })
  return saveConfig(next)
}

export async function runChatSync(input: {
  token: string
  sessions: ISessionStore
  encryption: EncryptionManager
  deviceName: string
}): Promise<GitHubChatSyncResult> {
  const startedAt = Date.now()
  let config = loadConfig()
  if (!config.enabled) throw new Error('GitHub encrypted chat sync is disabled.')
  if (!config.repoFullName) throw new Error('GitHub chat sync repository is not configured.')
  const repoFullName = config.repoFullName
  if (!input.encryption.isEnabled()) {
    throw new Error('GitHub chat sync requires memory.encryption and security/data.key.')
  }

  try {
    const worktree = repoWorktree(repoFullName)
    await prepareGitWorktree({
      worktree,
      repoFullName,
      branch: config.branch,
      token: input.token,
    })
    const manifest = await exportEncryptedChatSessions({
      sessions: input.sessions,
      encryption: input.encryption,
      rootDir: worktree,
      deviceName: input.deviceName,
    })
    await runGit(worktree, ['add', 'README.md', `devices/${manifest.deviceRef}`])
    const changedFiles = await gitChangedFileCount(worktree)
    let pushed = false
    let commitSha: string | null = null
    if (changedFiles > 0) {
      await runGit(worktree, ['commit', '-m', 'Sync encrypted sepilotd chat sessions'])
      commitSha = (await runGit(worktree, ['rev-parse', 'HEAD'])).trim()
      await runGit(worktree, ['push', 'origin', `HEAD:${config.branch}`], {
        token: input.token,
      })
      pushed = true
    } else {
      commitSha = (await runGit(worktree, ['rev-parse', '--verify', 'HEAD']).catch(() => '')).trim() || null
    }

    config = saveConfig({
      ...config,
      lastSyncedAt: Date.now(),
      lastSyncStatus: 'success',
      lastSyncError: null,
      lastCommit: commitSha,
      syncedSessions: manifest.sessionCount,
      updatedAt: Date.now(),
    })
    return {
      startedAt,
      finishedAt: Date.now(),
      repoFullName,
      branch: config.branch,
      worktree,
      syncedSessions: manifest.sessionCount,
      changedFiles,
      pushed,
      commitSha,
    }
  } catch (error) {
    saveConfig({
      ...config,
      lastSyncedAt: Date.now(),
      lastSyncStatus: 'error',
      lastSyncError: error instanceof Error ? error.message : 'GitHub chat sync failed',
      updatedAt: Date.now(),
    })
    throw error
  }
}

async function prepareGitWorktree(input: {
  worktree: string
  repoFullName: string
  branch: string
  token: string
}): Promise<void> {
  mkdirSync(input.worktree, { recursive: true, mode: 0o700 })
  secureDir(input.worktree)
  if (!existsSync(join(input.worktree, '.git'))) {
    await runGit(input.worktree, ['init'])
  }
  await runGit(input.worktree, ['config', 'user.name', 'sepilotd'])
  await runGit(input.worktree, ['config', 'user.email', 'sepilotd@local'])
  const remoteUrl = `https://github.com/${input.repoFullName}.git`
  const hasOrigin = await runGit(input.worktree, ['remote', 'get-url', 'origin']).then(
    () => true,
    () => false,
  )
  await runGit(input.worktree, hasOrigin ? ['remote', 'set-url', 'origin', remoteUrl] : ['remote', 'add', 'origin', remoteUrl])
  const fetched = await runGit(input.worktree, ['fetch', 'origin', input.branch], {
    token: input.token,
  }).then(
    () => true,
    () => false,
  )
  const hasHead = await runGit(input.worktree, ['rev-parse', '--verify', 'HEAD']).then(
    () => true,
    () => false,
  )
  if (fetched) {
    if (hasHead) {
      await runGit(input.worktree, ['checkout', input.branch])
      await runGit(input.worktree, ['merge', '--ff-only', `origin/${input.branch}`])
    } else {
      await runGit(input.worktree, ['checkout', '-B', input.branch, `origin/${input.branch}`])
    }
  } else {
    await runGit(input.worktree, ['checkout', '-B', input.branch])
  }
}

async function gitChangedFileCount(worktree: string): Promise<number> {
  const status = await runGit(worktree, ['status', '--porcelain'])
  return status.split('\n').filter((line) => line.trim()).length
}

async function runGit(
  cwd: string,
  args: string[],
  options: { token?: string } = {},
): Promise<string> {
  const authEnv = options.token
    ? {
      GIT_CONFIG_COUNT: '1',
      GIT_CONFIG_KEY_0: 'http.https://github.com/.extraheader',
      GIT_CONFIG_VALUE_0: `AUTHORIZATION: Bearer ${options.token}`,
    }
    : {}
  const { stdout } = await execFileAsync('git', args, {
    cwd,
    timeout: GIT_TIMEOUT_MS,
    maxBuffer: GIT_MAX_BUFFER,
    env: {
      ...process.env,
      GIT_TERMINAL_PROMPT: '0',
      ...authEnv,
    },
  })
  return String(stdout)
}
