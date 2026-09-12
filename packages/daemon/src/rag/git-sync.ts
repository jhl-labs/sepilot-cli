import { syncWebFolder } from './web-sync.js'
import { createHash } from 'node:crypto'
import { execFile } from 'node:child_process'
import { lstat, readdir, readFile } from 'node:fs/promises'
import { basename, extname, join, relative, resolve, sep } from 'node:path'
import { promisify } from 'node:util'
import type { IDocumentMemoryStore } from '@sepilotd/core'
import type { RagFolder, RagStore } from './store.js'

const execFileAsync = promisify(execFile)

const MAX_FILE_BYTES = 1_000_000

const TEXT_EXTENSIONS = new Set([
  '.c',
  '.cc',
  '.cfg',
  '.conf',
  '.cpp',
  '.cs',
  '.css',
  '.go',
  '.h',
  '.hpp',
  '.html',
  '.java',
  '.js',
  '.json',
  '.jsx',
  '.kt',
  '.lua',
  '.md',
  '.mdx',
  '.php',
  '.py',
  '.rb',
  '.rs',
  '.scss',
  '.sh',
  '.sql',
  '.svelte',
  '.toml',
  '.ts',
  '.tsx',
  '.txt',
  '.vue',
  '.xml',
  '.yaml',
  '.yml',
])

const TEXT_FILENAMES = new Set([
  'dockerfile',
  'gemfile',
  'license',
  'makefile',
  'rakefile',
  'readme',
])

const ALWAYS_EXCLUDED_DIRS = new Set([
  '.git',
  '.hg',
  '.next',
  '.nuxt',
  '.sepilot',
  'build',
  'coverage',
  'dist',
  'node_modules',
  'target',
  'vendor',
])

export interface RagSyncError {
  folderId: string
  path?: string
  error: string
}

export interface RagSyncResult {
  ok: boolean
  folders: number
  indexed: number
  deleted: number
  skipped: number
  errors: RagSyncError[]
}

interface SourceFile {
  relativePath: string
  absolutePath: string
  size: number
}

export async function syncRagSources(input: {
  store: RagStore
  semanticIndex?: IDocumentMemoryStore
}): Promise<RagSyncResult> {
  const folders = input.store
    .listFolders()
    .filter((folder) => (folder.sourceType === 'git' || folder.sourceType === 'web') && folder.path)

  const result: RagSyncResult = {
    ok: true,
    folders: folders.length,
    indexed: 0,
    deleted: 0,
    skipped: 0,
    errors: [],
  }

  for (const folder of folders) {
    try {
      const stats = await (folder.sourceType === 'web' ? syncWebFolder : syncFolder)(
        input.store,
        folder,
        input.semanticIndex,
      )
      result.indexed += stats.indexed
      result.deleted += stats.deleted
      result.skipped += stats.skipped
      input.store.recordFolderSync(folder.id, { status: 'success' })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      input.store.recordFolderSync(folder.id, {
        status: 'error',
        error: message,
      })
      result.errors.push({
        folderId: folder.id,
        path: folder.path,
        error: message,
      })
    }
  }

  result.ok = result.errors.length === 0
  return result
}

async function syncFolder(
  store: RagStore,
  folder: RagFolder,
  semanticIndex?: IDocumentMemoryStore,
): Promise<{ indexed: number; deleted: number; skipped: number }> {
  if (!folder.path) return { indexed: 0, deleted: 0, skipped: 0 }

  const root = resolve(folder.path)
  const files = await collectSourceFiles(root, folder)
  const seen = new Set<string>()
  let indexed = 0
  let skipped = 0

  for (const file of files) {
    const documentId = documentIdForFile(folder.id, file.relativePath)
    seen.add(documentId)

    if (file.size > MAX_FILE_BYTES) {
      skipped++
      continue
    }

    const body = await readFile(file.absolutePath, 'utf-8').catch(() => null)
    if (!body || body.includes('\0')) {
      skipped++
      continue
    }

    const sourceFileId = `git:${folder.id}:${file.relativePath}`
    await store.upsertDocument({
      id: documentId,
      folderId: folder.id,
      title: file.relativePath,
      body,
      path: file.relativePath,
      sourceFileId,
      size: file.size,
    })

    await semanticIndex?.ingestDocument({
      id: documentId,
      title: file.relativePath || basename(file.absolutePath),
      content: body,
      path: file.relativePath,
      sourceFileId,
      tags: ['rag', `rag-folder:${folder.id}`, 'source:git'],
    })
    indexed++
  }

  let deleted = 0
  for (const document of store.listDocuments(folder.id)) {
    if (!document.id.startsWith(`rag:${folder.id}:`)) continue
    if (seen.has(document.id)) continue
    store.removeDocument(document.id)
    await semanticIndex?.deleteDocument(document.id)
    deleted++
  }

  return { indexed, deleted, skipped }
}

async function collectSourceFiles(root: string, folder: RagFolder): Promise<SourceFile[]> {
  const gitFiles = await listGitFiles(root)
  const candidates = gitFiles ?? (await walkFiles(root))
  const files: SourceFile[] = []

  for (const relativePath of candidates) {
    const normalized = normalizeRelativePath(relativePath)
    if (!normalized || !isAllowedPath(normalized, folder)) continue
    const absolutePath = resolve(root, normalized)
    if (!isInside(root, absolutePath)) continue
    const stat = await lstat(absolutePath).catch(() => null)
    if (!stat?.isFile()) continue
    files.push({ relativePath: normalized, absolutePath, size: stat.size })
  }

  return files
}

async function listGitFiles(root: string): Promise<string[] | null> {
  try {
    const { stdout } = await execFileAsync(
      'git',
      ['-C', root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
      { maxBuffer: 10 * 1024 * 1024 },
    )
    return stdout.split('\0').filter(Boolean)
  } catch {
    return null
  }
}

async function walkFiles(root: string, current = root): Promise<string[]> {
  const entries = await readdir(current, { withFileTypes: true })
  const files: string[] = []
  for (const entry of entries) {
    if (entry.isDirectory()) {
      if (ALWAYS_EXCLUDED_DIRS.has(entry.name)) continue
      files.push(...(await walkFiles(root, join(current, entry.name))))
      continue
    }
    if (!entry.isFile()) continue
    files.push(relative(root, join(current, entry.name)))
  }
  return files
}

function normalizeRelativePath(value: string): string {
  return value.split(sep).join('/').replace(/\\/g, '/').replace(/^\/+/, '')
}

function isAllowedPath(path: string, folder: RagFolder): boolean {
  if (path.split('/').some((part) => ALWAYS_EXCLUDED_DIRS.has(part))) {
    return false
  }
  if (!isTextPath(path)) {
    return false
  }
  if (folder.include?.length && !folder.include.some((p) => matchesGlob(path, p))) {
    return false
  }
  if (folder.exclude?.some((p) => matchesGlob(path, p))) {
    return false
  }
  return true
}

function isTextPath(path: string): boolean {
  const ext = extname(path).toLowerCase()
  if (TEXT_EXTENSIONS.has(ext)) return true
  return TEXT_FILENAMES.has(basename(path).toLowerCase())
}

function matchesGlob(path: string, pattern: string): boolean {
  const normalized = normalizeRelativePath(pattern)
  const source = normalized
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/\*\*/g, '__DOUBLE_STAR__')
    .replace(/\*/g, '[^/]*')
    .replace(/__DOUBLE_STAR__/g, '.*')
  return new RegExp(`^${source}$`).test(path)
}

function isInside(root: string, target: string): boolean {
  const rel = relative(resolve(root), resolve(target))
  return rel === '' || (!rel.startsWith('..') && !rel.startsWith('/'))
}

function documentIdForFile(folderId: string, relativePath: string): string {
  const digest = createHash('sha1')
    .update(folderId)
    .update('\0')
    .update(relativePath)
    .digest('hex')
    .slice(0, 20)
  return `rag:${folderId}:${digest}`
}
