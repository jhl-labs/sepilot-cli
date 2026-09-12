import {
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs'
import { dirname, isAbsolute, join, normalize, relative, resolve } from 'node:path'
import { createGzip, gunzipSync } from 'node:zlib'
import { createHash } from 'node:crypto'
import { sepilotdHome } from '../storage/home.js'
import { isSafeId } from '../utils/safe-id.js'
import { secureDir, secureFile } from '../utils/secure-file.js'

export interface BackupItem {
  id: string
  createdAt: number
  sizeBytes: number
  path: string
  sha256: string
}

export interface RestoreResult {
  id: string
  restoredFiles: number
}

function backupDir(): string {
  const d = join(sepilotdHome(), 'backup')
  mkdirSync(d, { recursive: true, mode: 0o700 })
  secureDir(d)
  return d
}

function archiveEntries(): string[] {
  return [
    'config.yaml',
    'security/github-oauth.json',
    'github',
    'team-docs',
    'sessions',
    'memory',
    'persona',
    'snippets',
    'wiki',
    'knowledge',
    'rag',
    'image-gen',
    'prompt-templates',
    'personal-docs',
    'notifications',
    'scheduler',
  ]
}

function backupFilePath(id: string): string {
  if (!isSafeId(id)) {
    throw new Error('invalid backup id')
  }
  const dir = resolve(backupDir())
  const file = resolve(join(dir, `${id}.tar.gz`))
  const rel = relative(dir, file)
  if (rel.startsWith('..') || isAbsolute(rel)) {
    throw new Error('invalid backup path')
  }
  return file
}

function safeRestoreTarget(root: string, entryPath: string): string {
  const normalizedEntry = normalize(entryPath)
  if (
    !entryPath
    || entryPath.includes('\u0000')
    || isAbsolute(entryPath)
    || normalizedEntry === '..'
    || normalizedEntry.startsWith(`..\\`)
    || normalizedEntry.startsWith('../')
  ) {
    throw new Error(`unsafe backup entry: ${entryPath}`)
  }
  const target = resolve(root, entryPath)
  const rel = relative(root, target)
  if (rel.startsWith('..') || isAbsolute(rel)) {
    throw new Error(`unsafe backup entry: ${entryPath}`)
  }
  return target
}

export async function createBackup(): Promise<BackupItem> {
  const id = `backup-${new Date().toISOString().replace(/[:.]/g, '-')}`
  const file = join(backupDir(), `${id}.tar.gz`)
  const home = sepilotdHome()
  const stream = createWriteStream(file)
  const gz = createGzip()
  gz.pipe(stream)
  function addFile(path: string): void {
    const buf = readFileSync(path)
    const header = Buffer.from(
      `${path.slice(home.length + 1)}\u0000${buf.length}\n`,
      'utf-8',
    )
    gz.write(header)
    gz.write(buf)
  }
  function walk(path: string): void {
    if (!existsSync(path)) return
    const st = statSync(path)
    if (st.isDirectory()) {
      for (const name of readdirSync(path)) {
        walk(join(path, name))
      }
      return
    }
    if (st.isFile()) {
      addFile(path)
    }
  }
  for (const sub of archiveEntries()) {
    walk(join(home, sub))
  }
  await new Promise<void>((res, rej) => {
    gz.end()
    stream.on('finish', res)
    stream.on('error', rej)
  })
  // Backups contain config.yaml (provider apiKey) + all session
  // transcripts in cleartext — refuse to leave them at the default
  // 0644 where any other OS user can read them.
  secureFile(file)
  const buf = readFileSync(file)
  const sha = createHash('sha256').update(buf).digest('hex')
  return {
    id,
    createdAt: Date.now(),
    sizeBytes: buf.length,
    path: file,
    sha256: sha,
  }
}

export function listBackups(): BackupItem[] {
  const dir = backupDir()
  if (!existsSync(dir)) return []
  return readdirSync(dir)
    .filter((f) => f.endsWith('.tar.gz'))
    .map((f) => {
      const p = join(dir, f)
      const st = statSync(p)
      const buf = readFileSync(p)
      const sha = createHash('sha256').update(buf).digest('hex')
      return {
        id: f.replace(/\.tar\.gz$/, ''),
        createdAt: st.mtimeMs,
        sizeBytes: st.size,
        path: p,
        sha256: sha,
      }
    })
    .sort((a, b) => b.createdAt - a.createdAt)
}

export function removeBackup(id: string): void {
  const file = backupFilePath(id)
  if (existsSync(file)) unlinkSync(file)
}

export function restoreBackup(id: string): RestoreResult {
  const file = backupFilePath(id)
  if (!existsSync(file)) throw new Error('backup not found')
  const home = resolve(sepilotdHome())
  const archive = gunzipSync(readFileSync(file))
  const entries: Array<{ path: string; data: Buffer }> = []
  let offset = 0
  while (offset < archive.length) {
    const pathEnd = archive.indexOf(0, offset)
    if (pathEnd < 0) throw new Error('corrupt backup archive: missing path separator')
    const sizeEnd = archive.indexOf(10, pathEnd + 1)
    if (sizeEnd < 0) throw new Error('corrupt backup archive: missing size terminator')
    const entryPath = archive.subarray(offset, pathEnd).toString('utf-8')
    const size = Number.parseInt(
      archive.subarray(pathEnd + 1, sizeEnd).toString('utf-8'),
      10,
    )
    if (!Number.isSafeInteger(size) || size < 0) {
      throw new Error(`corrupt backup archive: invalid size for ${entryPath}`)
    }
    const dataStart = sizeEnd + 1
    const dataEnd = dataStart + size
    if (dataEnd > archive.length) {
      throw new Error(`corrupt backup archive: truncated data for ${entryPath}`)
    }
    safeRestoreTarget(home, entryPath)
    entries.push({ path: entryPath, data: Buffer.from(archive.subarray(dataStart, dataEnd)) })
    offset = dataEnd
  }

  for (const entry of entries) {
    const target = safeRestoreTarget(home, entry.path)
    mkdirSync(dirname(target), { recursive: true, mode: 0o700 })
    secureDir(dirname(target))
    writeFileSync(target, entry.data)
    secureFile(target)
  }

  return {
    id,
    restoredFiles: entries.length,
  }
}
