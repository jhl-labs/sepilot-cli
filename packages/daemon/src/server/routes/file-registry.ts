import { randomUUID } from 'node:crypto'
import {
  lstat,
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  unlink,
  writeFile,
} from 'node:fs/promises'
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import type { SessionAttachmentRef } from '@sepilotd/core'
import { isSafeId, isUuid } from '../../utils/safe-id.js'

const UPLOAD_DIR_NAME = 'uploads'
const UPLOAD_METADATA_DIR_NAME = '.metadata'
const UPLOAD_METADATA_VERSION = 1
export const ORPHAN_UPLOAD_GRACE_MS = 24 * 60 * 60 * 1000

export function attachmentAllowedRoots(options: { dataDir?: string; cwd?: string }): string[] {
  const roots: string[] = []
  if (options.dataDir) roots.push(join(options.dataDir, UPLOAD_DIR_NAME))
  if (options.cwd?.trim()) roots.push(resolve(options.cwd))
  return roots
}

export interface UploadedFile {
  id: string
  filename: string
  mimeType: string
  size: number
  path: string
  uploadedAt: string
  sessionId?: string
  kind?: 'upload' | 'voice'
}

interface RegistryEntry extends UploadedFile {
  storedAt: number
  lastAccess: number
}

const uploadedFiles = new Map<string, RegistryEntry>()

interface PersistedUploadedFile {
  version: typeof UPLOAD_METADATA_VERSION
  id: string
  filename: string
  mimeType: string
  size: number
  storedName: string
  uploadedAt: string
}

interface VoiceRetentionConfig {
  ttlMs: number
  maxEntries: number
}

const voiceRetention: VoiceRetentionConfig = {
  ttlMs: 60 * 60 * 1000,
  maxEntries: 200,
}

export function configureVoiceRetention(config: Partial<VoiceRetentionConfig>): void {
  if (typeof config.ttlMs === 'number' && config.ttlMs > 0) voiceRetention.ttlMs = config.ttlMs
  if (typeof config.maxEntries === 'number' && config.maxEntries > 0) {
    voiceRetention.maxEntries = config.maxEntries
  }
}

function unlinkQuietly(path: string): void {
  void unlink(path).catch(() => {})
}

function isExpiredVoiceEntry(entry: RegistryEntry, now: number): boolean {
  return entry.kind === 'voice' && now - entry.lastAccess >= voiceRetention.ttlMs
}

function enforceVoiceRetention(now: number): void {
  for (const [id, entry] of uploadedFiles) {
    if (isExpiredVoiceEntry(entry, now)) {
      uploadedFiles.delete(id)
      unlinkQuietly(entry.path)
    }
  }
  const voiceEntries = [...uploadedFiles.values()].filter((e) => e.kind === 'voice')
  if (voiceEntries.length > voiceRetention.maxEntries) {
    voiceEntries.sort((a, b) => a.lastAccess - b.lastAccess)
    const overflow = voiceEntries.length - voiceRetention.maxEntries
    for (const entry of voiceEntries.slice(0, overflow)) {
      uploadedFiles.delete(entry.id)
      unlinkQuietly(entry.path)
    }
  }
}

export function storeUploadedFile(file: UploadedFile): void {
  const now = Date.now()
  uploadedFiles.set(file.id, { ...file, storedAt: now, lastAccess: now })
  enforceVoiceRetention(now)
}

export function getUploadedFile(id: string): UploadedFile | undefined {
  const entry = uploadedFiles.get(id)
  if (!entry) return undefined
  const now = Date.now()
  if (isExpiredVoiceEntry(entry, now)) {
    uploadedFiles.delete(id)
    unlinkQuietly(entry.path)
    return undefined
  }
  entry.lastAccess = now
  return entry
}

export function deleteUploadedFile(id: string): void {
  uploadedFiles.delete(id)
}

function uploadMetadataDir(uploadDir: string): string {
  return join(uploadDir, UPLOAD_METADATA_DIR_NAME)
}

function uploadMetadataPath(uploadDir: string, id: string): string | null {
  if (!isSafeId(id)) return null
  return join(uploadMetadataDir(uploadDir), `${id}.json`)
}

function directUploadName(uploadDir: string, filePath: string): string | null {
  const relativePath = relative(resolve(uploadDir), resolve(filePath))
  if (
    !relativePath ||
    isAbsolute(relativePath) ||
    relativePath === '..' ||
    relativePath.startsWith(`..${sep}`) ||
    relativePath.includes(sep) ||
    basename(relativePath) !== relativePath
  ) {
    return null
  }
  return relativePath
}

function isPersistedUploadedFile(value: unknown): value is PersistedUploadedFile {
  if (!value || typeof value !== 'object') return false
  const record = value as Partial<PersistedUploadedFile>
  return (
    record.version === UPLOAD_METADATA_VERSION &&
    isUuid(record.id) &&
    typeof record.filename === 'string' &&
    record.filename.length > 0 &&
    record.filename.length <= 512 &&
    typeof record.mimeType === 'string' &&
    record.mimeType.length > 0 &&
    record.mimeType.length <= 255 &&
    typeof record.size === 'number' &&
    Number.isSafeInteger(record.size) &&
    record.size >= 0 &&
    typeof record.storedName === 'string' &&
    record.storedName.length > 0 &&
    basename(record.storedName) === record.storedName &&
    !record.storedName.includes('/') &&
    !record.storedName.includes('\\') &&
    typeof record.uploadedAt === 'string' &&
    Number.isFinite(Date.parse(record.uploadedAt))
  )
}

export async function persistUploadedFileMetadata(
  uploadDir: string,
  file: UploadedFile,
): Promise<void> {
  const metadataPath = uploadMetadataPath(uploadDir, file.id)
  const storedName = directUploadName(uploadDir, file.path)
  if (!metadataPath || !isUuid(file.id) || !storedName) {
    throw new Error('Uploaded file metadata must reference a safe file inside the upload directory')
  }

  const metadata: PersistedUploadedFile = {
    version: UPLOAD_METADATA_VERSION,
    id: file.id,
    filename: file.filename,
    mimeType: file.mimeType,
    size: file.size,
    storedName,
    uploadedAt: file.uploadedAt,
  }
  if (!isPersistedUploadedFile(metadata)) {
    throw new Error('Uploaded file metadata is invalid')
  }
  const metadataDir = uploadMetadataDir(uploadDir)
  await mkdir(metadataDir, { recursive: true })
  const temporaryPath = join(metadataDir, `.${file.id}.${randomUUID()}.tmp`)
  try {
    await writeFile(temporaryPath, JSON.stringify(metadata), {
      encoding: 'utf8',
      flag: 'wx',
      mode: 0o600,
    })
    await rename(temporaryPath, metadataPath)
  } finally {
    await rm(temporaryPath, { force: true }).catch(() => {})
  }
}

export async function deleteUploadedFileMetadata(uploadDir: string, id: string): Promise<void> {
  const metadataPath = uploadMetadataPath(uploadDir, id)
  if (!metadataPath) return
  await rm(metadataPath, { force: true }).catch(() => {})
}

export async function restoreUploadedFiles(uploadDir: string): Promise<number> {
  const metadataDir = uploadMetadataDir(uploadDir)
  let names: string[]
  try {
    names = await readdir(metadataDir)
  } catch {
    return 0
  }

  let restored = 0
  for (const name of names) {
    if (!name.endsWith('.json')) continue
    const metadataPath = join(metadataDir, name)
    try {
      const metadataInfo = await lstat(metadataPath)
      if (!metadataInfo.isFile() || metadataInfo.isSymbolicLink()) continue
      const parsed = JSON.parse(await readFile(metadataPath, 'utf8')) as unknown
      if (!isPersistedUploadedFile(parsed) || name !== `${parsed.id}.json`) continue
      const filePath = join(uploadDir, parsed.storedName)
      if (directUploadName(uploadDir, filePath) !== parsed.storedName) continue
      const fileInfo = await lstat(filePath)
      if (!fileInfo.isFile() || fileInfo.isSymbolicLink() || fileInfo.size !== parsed.size) continue
      const restoredAt = Math.max(
        Date.parse(parsed.uploadedAt),
        metadataInfo.mtimeMs,
        fileInfo.mtimeMs,
      )
      uploadedFiles.set(parsed.id, {
        id: parsed.id,
        filename: parsed.filename,
        mimeType: parsed.mimeType,
        size: fileInfo.size,
        path: filePath,
        uploadedAt: parsed.uploadedAt,
        kind: 'upload',
        storedAt: restoredAt,
        lastAccess: restoredAt,
      })
      restored += 1
    } catch {
    }
  }
  return restored
}

export interface OrphanUploadSweepOptions {
  referencedFileIds: ReadonlySet<string>
  now?: number
  graceMs?: number
  removePath?: (path: string) => Promise<void>
}

export interface UploadedFileRemovalOptions {
  removePath?: (path: string) => Promise<void>
}

export async function removeUploadedFilesByIds(
  fileIds: ReadonlySet<string>,
  options: UploadedFileRemovalOptions = {},
): Promise<UploadedFile[]> {
  const removePath = options.removePath ?? ((path: string) => rm(path, { force: true }))
  const candidates = deleteUploadedFilesByIds(fileIds)
  const results = await Promise.all(
    candidates.map(async (file) => {
      try {
        await removePath(file.path)
      } catch {
        if (!uploadedFiles.has(file.id)) uploadedFiles.set(file.id, file as RegistryEntry)
        return null
      }
      await deleteUploadedFileMetadata(dirname(file.path), file.id)
      return file
    }),
  )
  return results.filter((file): file is UploadedFile => file !== null)
}

export async function sweepOrphanUploadedFiles(
  uploadDir: string,
  options: OrphanUploadSweepOptions,
): Promise<number> {
  const now = options.now ?? Date.now()
  const graceMs = options.graceMs ?? ORPHAN_UPLOAD_GRACE_MS
  const removePath = options.removePath ?? ((path: string) => rm(path, { force: true }))
  if (!Number.isFinite(now) || !Number.isFinite(graceMs) || graceMs < 0) return 0

  const metadataDir = uploadMetadataDir(uploadDir)
  let names: string[]
  try {
    names = await readdir(metadataDir)
  } catch {
    return 0
  }

  let removed = 0
  for (const name of names) {
    if (!name.endsWith('.json')) continue
    const metadataPath = join(metadataDir, name)
    try {
      const metadataInfo = await lstat(metadataPath)
      if (!metadataInfo.isFile() || metadataInfo.isSymbolicLink()) continue

      const parsed = JSON.parse(await readFile(metadataPath, 'utf8')) as unknown
      if (!isPersistedUploadedFile(parsed) || name !== `${parsed.id}.json`) continue
      if (options.referencedFileIds.has(parsed.id)) continue

      const filePath = join(uploadDir, parsed.storedName)
      if (directUploadName(uploadDir, filePath) !== parsed.storedName) continue
      const fileInfo = await lstat(filePath)
      if (!fileInfo.isFile() || fileInfo.isSymbolicLink() || fileInfo.size !== parsed.size) continue

      const entry = uploadedFiles.get(parsed.id)
      if (
        !entry ||
        entry.kind !== 'upload' ||
        resolve(entry.path) !== resolve(filePath) ||
        entry.size !== parsed.size
      ) {
        continue
      }

      const newestTimestamp = Math.max(
        Date.parse(parsed.uploadedAt),
        metadataInfo.mtimeMs,
        fileInfo.mtimeMs,
      )
      if (!Number.isFinite(newestTimestamp) || now - newestTimestamp < graceMs) continue

      const currentEntry = uploadedFiles.get(parsed.id)
      if (
        currentEntry !== entry ||
        now - currentEntry.lastAccess < graceMs ||
        options.referencedFileIds.has(parsed.id)
      ) {
        continue
      }
      const deleted = await removeUploadedFilesByIds(new Set([parsed.id]), { removePath })
      if (deleted.length > 0) removed += 1
    } catch {
    }
  }
  return removed
}

export function resolveSessionAttachmentRefs(fileIds: readonly string[]): SessionAttachmentRef[] {
  const seen = new Set<string>()
  const refs: SessionAttachmentRef[] = []
  for (const id of fileIds) {
    if (seen.has(id) || !isSafeId(id)) continue
    seen.add(id)
    const file = getUploadedFile(id)
    if (!file) continue
    refs.push({
      fileId: file.id,
      mimeType: file.mimeType,
      filename: file.filename,
      size: file.size,
    })
  }
  return refs
}

export function deleteUploadedFilesBySession(sessionId: string): UploadedFile[] {
  if (!sessionId) return []
  const removed: UploadedFile[] = []
  for (const [id, file] of uploadedFiles) {
    if (file.sessionId === sessionId) {
      removed.push(file)
      uploadedFiles.delete(id)
    }
  }
  return removed
}

export function listUploadedFilesBySession(sessionId: string): UploadedFile[] {
  if (!sessionId) return []
  return [...uploadedFiles.values()].filter((file) => file.sessionId === sessionId)
}

export function deleteUploadedFilesByIds(fileIds: ReadonlySet<string>): UploadedFile[] {
  const removed: UploadedFile[] = []
  for (const id of fileIds) {
    const file = uploadedFiles.get(id)
    if (!file) continue
    removed.push(file)
    uploadedFiles.delete(id)
  }
  return removed
}

export function resolveFileIds(
  fileIds: string[],
): Array<{ type: string; path: string; filename: string; trusted: true }> {
  return fileIds
    .map((id) => getUploadedFile(id))
    .filter((file): file is UploadedFile => file !== undefined)
    .map((file) => ({
      type: file.mimeType,
      path: file.path,
      filename: file.filename,
      trusted: true,
    }))
}

export function resolveUploadedFileNames(fileIds: string[]): string[] {
  return fileIds.map((id) => getUploadedFile(id)?.filename ?? id)
}

export async function sweepOrphanVoiceFiles(voiceDir: string): Promise<number> {
  let removed = 0
  let names: string[]
  try {
    names = await readdir(voiceDir)
  } catch {
    return 0
  }
  const now = Date.now()
  const registeredPaths = new Set([...uploadedFiles.values()].map((e) => resolve(e.path)))
  for (const name of names) {
    if (!name.toLowerCase().endsWith('.wav')) continue
    const full = join(voiceDir, name)
    const inRegistry = registeredPaths.has(resolve(full))
    let expired = false
    if (inRegistry) {
      try {
        const info = await stat(full)
        expired = now - info.mtimeMs >= voiceRetention.ttlMs
      } catch {
        continue
      }
      if (!expired) continue
    }
    try {
      await unlink(full)
      removed += 1
    } catch {
    }
  }
  return removed
}
