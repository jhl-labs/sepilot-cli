import { exec as execCallback } from 'node:child_process'
import { readdir } from 'node:fs/promises'
import { dirname, join, posix, relative, sep } from 'node:path'
import { promisify } from 'node:util'

const execAsync = promisify(execCallback)

const HARDCODED_EXCLUDES = new Set([
  'node_modules',
  '.git',
  'dist',
  'build',
  '.next',
  '.turbo',
  '.cache',
])

export interface FileIndexEntry {
  path: string
  basename: string
  dir: string
  isDirectory: boolean
}

export interface FileIndex {
  rootDir: string
  entries: FileIndexEntry[]
  loadedAt: number
}

const cache = new Map<string, FileIndex>()
const inflight = new Map<string, Promise<FileIndex>>()

function toPosix(value: string): string {
  return sep === '/' ? value : value.split(sep).join('/')
}

function makeEntry(relativePath: string, isDirectory: boolean): FileIndexEntry {
  const path = toPosix(relativePath)
  const dir = dirname(path)
  return {
    path,
    basename: posix.basename(path),
    dir: dir === '.' ? '' : dir,
    isDirectory,
  }
}

function collectGitDirectoryEntries(paths: readonly string[]): FileIndexEntry[] {
  const directories = new Set<string>()

  for (const rawPath of paths) {
    let current = dirname(toPosix(rawPath))
    while (current !== '.' && current !== '') {
      directories.add(current)
      current = dirname(current)
    }
  }

  return [...directories]
    .sort((left, right) => left.localeCompare(right))
    .map((path) => makeEntry(path, true))
}

async function loadFromGit(rootDir: string): Promise<FileIndexEntry[] | null> {
  try {
    const { stdout } = await execAsync('git ls-files -co --exclude-standard', {
      cwd: rootDir,
      maxBuffer: 32 * 1024 * 1024,
    })
    const lines = stdout.split('\n').filter((line) => line.length > 0)
    return [
      ...collectGitDirectoryEntries(lines),
      ...lines.map((line) => makeEntry(line, false)),
    ]
  } catch {
    return null
  }
}

async function loadFromReaddir(rootDir: string): Promise<FileIndexEntry[]> {
  const entries: FileIndexEntry[] = []

  async function walk(currentDir: string): Promise<void> {
    let dirents
    try {
      dirents = await readdir(currentDir, { withFileTypes: true })
    } catch {
      return
    }

    for (const dirent of dirents) {
      if (HARDCODED_EXCLUDES.has(dirent.name)) continue
      const absolute = join(currentDir, dirent.name)
      const rel = relative(rootDir, absolute)
      if (dirent.isDirectory()) {
        entries.push(makeEntry(rel, true))
        await walk(absolute)
      } else if (dirent.isFile()) {
        entries.push(makeEntry(rel, false))
      }
    }
  }

  await walk(rootDir)
  return entries
}

export async function getFileIndex(rootDir: string): Promise<FileIndex> {
  const cached = cache.get(rootDir)
  if (cached) return cached

  const pending = inflight.get(rootDir)
  if (pending) return pending

  const promise = (async () => {
    const fromGit = await loadFromGit(rootDir)
    const entries = fromGit ?? (await loadFromReaddir(rootDir))
    const index: FileIndex = { rootDir, entries, loadedAt: Date.now() }
    cache.set(rootDir, index)
    inflight.delete(rootDir)
    return index
  })()

  inflight.set(rootDir, promise)
  return promise
}

export function invalidateFileIndex(rootDir: string): void {
  cache.delete(rootDir)
  inflight.delete(rootDir)
}

export function __resetFileIndexCacheForTest(): void {
  cache.clear()
  inflight.clear()
}
