import { mkdir, readdir, readFile, rm, unlink, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { createHash, randomUUID } from 'node:crypto'
import { closeSync, constants, fstatSync, lstatSync, openSync, readFileSync, realpathSync } from 'node:fs'
import type { EditCheckpointFile, EditCheckpointSummary } from '@sepilotd/core'
import { createLogger } from '../../logger.js'
import { buildUnifiedDiff } from '../../tools/edit-diff.js'

const log = createLogger('edit-snapshots')

const DEFAULT_MAX_COMMITTED_PER_SESSION = 20

interface FileSnapshot {
  path: string
  hadFileBefore: boolean
  contentBefore: Buffer | null
  postHash?: string | null
  postParent?: string
}

interface CheckpointRecord {
  checkpointId: string
  label: string
  status: 'open' | 'committed' | 'reverted'
  files: Map<string, FileSnapshot>
  createdAt: string
  seq: number
  closedAt?: string
  revertedAt?: string
  revertReason?: string
}

interface PersistedManifest {
  checkpointId: string
  label: string
  status: 'committed'
  createdAt: string
  seq: number
  closedAt?: string
  files: Array<{
    path: string
    hadFileBefore: boolean
    blob: string | null
    postHash?: string | null
    postParent?: string
  }>
}

export class RewindConflictError extends Error {
  constructor(readonly paths: string[]) {
    super(`Rewind refused: files changed outside this checkpoint or lack a verified postimage: ${paths.join(', ')}`)
  }
}

function readCurrentContent(path: string): Buffer | null {
  let fd: number | undefined
  try {
    fd = openSync(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0) | (constants.O_NONBLOCK ?? 0))
    const opened = fstatSync(fd)
    const named = lstatSync(path)
    if (!opened.isFile() || !named.isFile() || opened.dev !== named.dev || opened.ino !== named.ino) {
      throw new Error(`Not the same regular file: ${path}`)
    }
    return readFileSync(fd)
  } catch (error) {
    if (fd === undefined && (error as NodeJS.ErrnoException).code === 'ENOENT') return null
    throw error
  } finally {
    if (fd !== undefined) closeSync(fd)
  }
}

function contentHash(content: Buffer | null): string | null {
  return content === null ? null : createHash('sha256').update(content).digest('hex')
}

function currentHash(path: string): string | null {
  return contentHash(readCurrentContent(path))
}

export interface EditSnapshotStoreOptions {
  persistDir?: string
  maxCommittedPerSession?: number
}

export interface RewindFilesResult {
  checkpoint: EditCheckpointSummary
  restoredFiles: string[]
  incompleteHistory: boolean
}

export interface RewindPreview {
  checkpoint: EditCheckpointSummary
  checkpointIds: string[]
  files: Array<{ path: string; operation: 'restore' | 'remove'; diff?: string; conflict: boolean }>
  incompleteHistory: boolean
}

export interface EditCheckpointDeltaFile {
  path: string
  status: 'created' | 'modified' | 'deleted' | 'unchanged'
  bytesBefore: number
  bytesAfter: number
  unifiedDiff?: string
}

function isTextBuffer(content: Buffer | null): boolean {
  return content === null || !content.includes(0)
}

export class EditSnapshotStore {
  private readonly bySession = new Map<string, Map<string, CheckpointRecord>>()
  private readonly persistDir?: string
  private readonly maxCommittedPerSession: number
  private seqCounter = 0
  private pendingPersist: Promise<void> = Promise.resolve()
  private readonly persistFailures = new Map<string, number>()
  private readonly rewinding = new Set<string>()

  constructor(options: EditSnapshotStoreOptions = {}) {
    this.persistDir = options.persistDir
    this.maxCommittedPerSession =
      options.maxCommittedPerSession ?? DEFAULT_MAX_COMMITTED_PER_SESSION
  }

  openCheckpoint(sessionId: string, label: string): string {
    const checkpointId = randomUUID()
    const record: CheckpointRecord = {
      checkpointId,
      label,
      status: 'open',
      files: new Map(),
      createdAt: new Date().toISOString(),
      seq: this.seqCounter++,
    }
    let bucket = this.bySession.get(sessionId)
    if (!bucket) {
      bucket = new Map()
      this.bySession.set(sessionId, bucket)
    }
    bucket.set(checkpointId, record)
    return checkpointId
  }

  async recordPreEdit(
    sessionId: string,
    checkpointId: string,
    path: string,
  ): Promise<void> {
    const record = this.requireOpen(sessionId, checkpointId)
    if (record.files.has(path)) return
    try {
      const buf = await readFile(path)
      record.files.set(path, { path, hadFileBefore: true, contentBefore: buf })
    } catch (err) {
      const code = (err as { code?: string }).code
      if (code !== 'ENOENT') throw err
      record.files.set(path, { path, hadFileBefore: false, contentBefore: null })
    }
  }

  get(sessionId: string, checkpointId: string): EditCheckpointSummary | undefined {
    const record = this.bySession.get(sessionId)?.get(checkpointId)
    return record ? toSummary(record) : undefined
  }

  async inspectCheckpointDelta(
    sessionId: string,
    checkpointId: string,
  ): Promise<EditCheckpointDeltaFile[]> {
    const record = this.requireOpen(sessionId, checkpointId)
    const deltas: EditCheckpointDeltaFile[] = []
    for (const snapshot of record.files.values()) {
      let current: Buffer | null = null
      try {
        current = await readFile(snapshot.path)
      } catch (error) {
        if ((error as { code?: string }).code !== 'ENOENT') throw error
      }
      const unchanged = snapshot.hadFileBefore
        ? current !== null && snapshot.contentBefore?.equals(current) === true
        : current === null
      const status: EditCheckpointDeltaFile['status'] = unchanged
        ? 'unchanged'
        : !snapshot.hadFileBefore
          ? 'created'
          : current === null
            ? 'deleted'
            : 'modified'
      const unifiedDiff = !unchanged
        && isTextBuffer(snapshot.contentBefore)
        && isTextBuffer(current)
        ? buildUnifiedDiff(
            snapshot.path,
            snapshot.contentBefore?.toString('utf-8') ?? '',
            current?.toString('utf-8') ?? '',
          )
        : undefined
      deltas.push({
        path: snapshot.path,
        status,
        bytesBefore: snapshot.contentBefore?.byteLength ?? 0,
        bytesAfter: current?.byteLength ?? 0,
        ...(unifiedDiff ? { unifiedDiff } : {}),
      })
    }
    return deltas.sort((left, right) => left.path.localeCompare(right.path))
  }

  commitCheckpoint(sessionId: string, checkpointId: string): EditCheckpointSummary {
    const record = this.requireOpen(sessionId, checkpointId)
    for (const file of record.files.values()) {
      try {
        file.postHash = currentHash(file.path)
        file.postParent = realpathSync(dirname(file.path))
      } catch { file.postHash = undefined }
    }
    record.status = 'committed'
    record.closedAt = new Date().toISOString()
    const summary = toSummary(record)
    if (this.persistDir && record.files.size > 0) {
      this.pendingPersist = this.pendingPersist
        .then(() => this.persistCommitted(sessionId, record))
        .then(() => this.pruneSession(sessionId))
        .catch((err) => {
          this.persistFailures.set(sessionId, (this.persistFailures.get(sessionId) ?? 0) + 1)
          log.error('failed to persist edit checkpoint', {
            sessionId,
            checkpointId,
            error: err instanceof Error ? err.message : String(err),
          })
        })
        .finally(() => {
          this.bySession.get(sessionId)?.delete(checkpointId)
        })
    } else {
      this.bySession.get(sessionId)?.delete(checkpointId)
    }
    return summary
  }

  async revertCheckpoint(
    sessionId: string,
    checkpointId: string,
    reason: string,
  ): Promise<EditCheckpointSummary> {
    const record = this.requireOpen(sessionId, checkpointId)
    await restoreSnapshots(record.files.values())
    record.status = 'reverted'
    record.revertedAt = new Date().toISOString()
    record.revertReason = reason
    record.closedAt = record.revertedAt
    const summary = toSummary(record)
    this.bySession.get(sessionId)?.delete(checkpointId)
    return summary
  }

  async flush(): Promise<void> {
    await this.pendingPersist
  }

  hasPersistFailures(sessionId: string): boolean {
    return (this.persistFailures.get(sessionId) ?? 0) > 0
  }

  async listCheckpoints(sessionId: string): Promise<EditCheckpointSummary[]> {
    const manifests = await this.loadManifests(sessionId)
    return manifests.map(manifestToSummary)
  }

  async previewRewind(sessionId: string, checkpointId: string): Promise<RewindPreview> {
    const manifests = await this.loadManifests(sessionId)
    const index = manifests.findIndex(manifest => manifest.checkpointId === checkpointId)
    if (index < 0) throw new Error(`Edit checkpoint ${checkpointId} not found for session ${sessionId}`)
    const earliest = new Map<string, { manifest: PersistedManifest; file: PersistedManifest['files'][number] }>()
    const latest = new Map<string, PersistedManifest['files'][number]>()
    for (const manifest of manifests.slice(index)) for (const file of manifest.files) {
      if (!earliest.has(file.path)) earliest.set(file.path, { manifest, file })
      latest.set(file.path, file)
    }
    const files: RewindPreview['files'] = []
    let diffBudget = 128_000
    for (const [path, { manifest, file }] of earliest) {
      const expected = latest.get(path)!
      let conflict = true
      let diff: string | undefined
      try {
        const before = readCurrentContent(path)
        conflict = expected.postHash === undefined || contentHash(before) !== expected.postHash
          || realpathSync(dirname(path)) !== expected.postParent
        if (!conflict && diffBudget > 0) {
          const after = file.hadFileBefore && file.blob
            ? await readFile(join(this.checkpointDir(sessionId, manifest.checkpointId), file.blob)) : null
          if (isTextBuffer(before) && isTextBuffer(after)) {
            diff = buildUnifiedDiff(path, before?.toString('utf8') ?? '', after?.toString('utf8') ?? '')
            if (diff && diff.length > diffBudget) diff = `${diff.slice(0, diffBudget)}\n[preview truncated]`
            diffBudget -= diff?.length ?? 0
          }
        }
      } catch { conflict = true }
      files.push({ path, operation: file.hadFileBefore ? 'restore' : 'remove', conflict, ...(diff ? { diff } : {}) })
    }
    return { checkpoint: manifestToSummary(manifests[index]), checkpointIds: manifests.slice(index).map(manifest => manifest.checkpointId), files, incompleteHistory: this.hasPersistFailures(sessionId) }
  }

  async rewindFiles(sessionId: string, checkpointId: string, expectedCheckpointIds?: string[]): Promise<RewindFilesResult> {
    if (this.rewinding.has(sessionId)) throw new RewindConflictError(['rewind already in progress'])
    this.rewinding.add(sessionId)
    try { return await this.rewindFilesOnce(sessionId, checkpointId, expectedCheckpointIds) }
    finally { this.rewinding.delete(sessionId) }
  }

  private async rewindFilesOnce(sessionId: string, checkpointId: string, expectedCheckpointIds?: string[]): Promise<RewindFilesResult> {
    const manifests = await this.loadManifests(sessionId)
    const targetIndex = manifests.findIndex((m) => m.checkpointId === checkpointId)
    if (targetIndex === -1) {
      throw new Error(`Edit checkpoint ${checkpointId} not found for session ${sessionId}`)
    }
    const selected = manifests.slice(targetIndex)
    if (expectedCheckpointIds && JSON.stringify(expectedCheckpointIds) !== JSON.stringify(selected.map(manifest => manifest.checkpointId))) {
      throw new RewindConflictError(['checkpoint scope changed since preview'])
    }
    if (this.hasPersistFailures(sessionId) || [...(this.bySession.get(sessionId)?.values() ?? [])].some(record => record.status === 'open')) {
      throw new RewindConflictError(['checkpoint history incomplete or edit still active'])
    }
    const latestByPath = new Map<string, PersistedManifest['files'][number]>()
    for (const manifest of selected) for (const file of manifest.files) latestByPath.set(file.path, file)
    const conflicts = [...latestByPath].flatMap(([path, expected]) => {
      try { return expected.postHash !== undefined && currentHash(path) === expected.postHash
        && expected.postParent === realpathSync(dirname(path)) ? [] : [path] }
      catch { return [path] }
    })
    if (conflicts.length) throw new RewindConflictError(conflicts)
    const earliestByPath = new Map<
      string,
      { manifest: PersistedManifest; file: PersistedManifest['files'][number] }
    >()
    for (const manifest of selected) {
      for (const file of manifest.files) {
        if (!earliestByPath.has(file.path)) {
          earliestByPath.set(file.path, { manifest, file })
        }
      }
    }

    const restoreContents = new Map<string, Buffer | null>()
    for (const { manifest, file } of earliestByPath.values()) {
      restoreContents.set(file.path, file.hadFileBefore && file.blob
        ? await readFile(join(this.checkpointDir(sessionId, manifest.checkpointId), file.blob)) : null)
    }
    const restoredFiles: string[] = []
    for (const { file } of earliestByPath.values()) {
      if (file.hadFileBefore && file.blob) {
        const content = restoreContents.get(file.path)!
        await mkdir(dirname(file.path), { recursive: true })
        await writeFile(file.path, content!)
      } else {
        try {
          await unlink(file.path)
        } catch (err) {
          const code = (err as { code?: string }).code
          if (code !== 'ENOENT') throw err
        }
      }
      restoredFiles.push(file.path)
    }

    for (const manifest of selected) {
      await rm(this.checkpointDir(sessionId, manifest.checkpointId), {
        recursive: true,
        force: true,
      })
    }

    return {
      checkpoint: manifestToSummary(manifests[targetIndex]!),
      restoredFiles: restoredFiles.sort(),
      incompleteHistory: this.hasPersistFailures(sessionId),
    }
  }

  dispose(sessionId: string): void {
    this.bySession.delete(sessionId)
  }

  private requireOpen(sessionId: string, checkpointId: string): CheckpointRecord {
    const record = this.bySession.get(sessionId)?.get(checkpointId)
    if (!record) {
      throw new Error(
        `Edit checkpoint ${checkpointId} not found for session ${sessionId}`,
      )
    }
    if (record.status !== 'open') {
      throw new Error(
        `Edit checkpoint ${checkpointId} is ${record.status}`,
      )
    }
    return record
  }

  private sessionDir(sessionId: string): string {
    if (!this.persistDir) throw new Error('EditSnapshotStore has no persistDir')
    return join(this.persistDir, encodeURIComponent(sessionId), 'checkpoints')
  }

  private checkpointDir(sessionId: string, checkpointId: string): string {
    return join(this.sessionDir(sessionId), encodeURIComponent(checkpointId))
  }

  private async persistCommitted(sessionId: string, record: CheckpointRecord): Promise<void> {
    const dir = this.checkpointDir(sessionId, record.checkpointId)
    await mkdir(join(dir, 'blobs'), { recursive: true })
    const files: PersistedManifest['files'] = []
    let blobIndex = 0
    for (const snap of record.files.values()) {
      let blob: string | null = null
      if (snap.hadFileBefore && snap.contentBefore) {
        blob = join('blobs', String(blobIndex++))
        await writeFile(join(dir, blob), snap.contentBefore)
      }
      files.push({ path: snap.path, hadFileBefore: snap.hadFileBefore, blob, postHash: snap.postHash, postParent: snap.postParent })
    }
    const manifest: PersistedManifest = {
      checkpointId: record.checkpointId,
      label: record.label,
      status: 'committed',
      createdAt: record.createdAt,
      seq: record.seq,
      closedAt: record.closedAt,
      files,
    }
    await writeFile(join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2), 'utf-8')
  }

  private async pruneSession(sessionId: string): Promise<void> {
    const manifests = await this.loadPersistedManifests(sessionId)
    const excess = manifests.length - this.maxCommittedPerSession
    if (excess <= 0) return
    for (const manifest of manifests.slice(0, excess)) {
      await rm(this.checkpointDir(sessionId, manifest.checkpointId), {
        recursive: true,
        force: true,
      })
    }
  }

  private async loadManifests(sessionId: string): Promise<PersistedManifest[]> {
    if (!this.persistDir) return []
    await this.flush()
    return this.loadPersistedManifests(sessionId)
  }

  private async loadPersistedManifests(sessionId: string): Promise<PersistedManifest[]> {
    let entries: string[]
    try {
      entries = await readdir(this.sessionDir(sessionId))
    } catch {
      return []
    }
    const manifests: PersistedManifest[] = []
    for (const entry of entries) {
      try {
        const raw = await readFile(
          join(this.sessionDir(sessionId), entry, 'manifest.json'),
          'utf-8',
        )
        manifests.push(JSON.parse(raw) as PersistedManifest)
      } catch {
      }
    }
    manifests.sort((a, b) =>
      a.createdAt === b.createdAt ? a.seq - b.seq : a.createdAt.localeCompare(b.createdAt),
    )
    return manifests
  }
}

async function restoreSnapshots(snapshots: Iterable<FileSnapshot>): Promise<void> {
  for (const snap of snapshots) {
    if (snap.hadFileBefore && snap.contentBefore) {
      await writeFile(snap.path, snap.contentBefore)
    } else {
      try {
        await unlink(snap.path)
      } catch (err) {
        const code = (err as { code?: string }).code
        if (code !== 'ENOENT') throw err
      }
    }
  }
}

function toSummary(record: CheckpointRecord): EditCheckpointSummary {
  const files: EditCheckpointFile[] = Array.from(record.files.values()).map(
    (snap) => ({
      path: snap.path,
      hadFileBefore: snap.hadFileBefore,
      bytesBefore: snap.contentBefore?.byteLength ?? 0,
    }),
  )
  return {
    checkpointId: record.checkpointId,
    label: record.label,
    status: record.status,
    files,
    createdAt: record.createdAt,
    closedAt: record.closedAt,
    revertedAt: record.revertedAt,
    revertReason: record.revertReason,
  }
}

function manifestToSummary(manifest: PersistedManifest): EditCheckpointSummary {
  return {
    checkpointId: manifest.checkpointId,
    label: manifest.label,
    status: manifest.status,
    files: manifest.files.map((file) => ({
      path: file.path,
      hadFileBefore: file.hadFileBefore,
      bytesBefore: 0,
    })),
    createdAt: manifest.createdAt,
    closedAt: manifest.closedAt,
  }
}
