import { createHash } from 'node:crypto'
import { readFile, stat } from 'node:fs/promises'

interface FileFingerprint {
  mtimeMs: number
  size: number
  contentHash: string
  recordedAt: string
}

export interface WorkspaceReadObservation {
  output: string
  evidenceId: string
  contentHash: string
  recordedAt: string
}

export interface WorkspaceReadObservationLookup {
  status: 'hit' | 'miss' | 'invalidated'
  reason: 'not-cached' | 'unchanged' | 'content-changed' | 'unavailable'
  observation?: WorkspaceReadObservation
}

interface CachedReadObservation extends WorkspaceReadObservation {
  path: string
  viewKey: string
  lastUsed: number
}

const MAX_CACHED_READ_OBSERVATIONS_PER_SESSION = 64
const MAX_CACHED_READ_OBSERVATION_CHARS = 512_000
const MAX_CACHED_READ_CHARS_PER_SESSION = 2_000_000

export interface WorkspaceStaleSignal {
  path: string
  baselineMtimeMs: number
  currentMtimeMs: number
  baselineSize: number
  currentSize: number
  baselineContentHash: string
  currentContentHash: string
}

export class WorkspaceMutationTracker {
  private readonly bySession = new Map<string, Map<string, FileFingerprint>>()
  private readonly observationsBySession = new Map<string, Map<string, CachedReadObservation>>()

  async recordRead(sessionId: string, path: string): Promise<void> {
    const fingerprint = await this.fingerprint(path)
    if (!fingerprint) return
    let bucket = this.bySession.get(sessionId)
    if (!bucket) {
      bucket = new Map()
      this.bySession.set(sessionId, bucket)
    }
    bucket.set(path, fingerprint)
  }

  async detectStale(
    sessionId: string,
    path: string,
  ): Promise<WorkspaceStaleSignal | null> {
    const baseline = this.bySession.get(sessionId)?.get(path)
    if (!baseline) return null
    const current = await this.fingerprint(path)
    if (!current) return null
    if (current.contentHash === baseline.contentHash) {
      return null
    }
    return {
      path,
      baselineMtimeMs: baseline.mtimeMs,
      currentMtimeMs: current.mtimeMs,
      baselineSize: baseline.size,
      currentSize: current.size,
      baselineContentHash: baseline.contentHash,
      currentContentHash: current.contentHash,
    }
  }

  /**
   * Record post-write fingerprint so subsequent stale checks compare against
   * the latest agent-authored state, not the pre-edit baseline.
   */
  async recordWrite(sessionId: string, path: string): Promise<void> {
    this.invalidateReadObservations(sessionId, path)
    await this.recordRead(sessionId, path)
  }

  async lookupReadObservation(
    sessionId: string,
    path: string,
    viewKey: string,
  ): Promise<WorkspaceReadObservationLookup> {
    const key = this.observationKey(path, viewKey)
    const cached = this.observationsBySession.get(sessionId)?.get(key)
    if (!cached) return { status: 'miss', reason: 'not-cached' }

    const current = await this.fingerprint(path)
    if (!current) {
      this.invalidateReadObservations(sessionId, path)
      return { status: 'invalidated', reason: 'unavailable' }
    }
    if (current.contentHash !== cached.contentHash) {
      this.invalidateReadObservations(sessionId, path)
      this.sessionFingerprints(sessionId).set(path, current)
      return { status: 'invalidated', reason: 'content-changed' }
    }

    cached.lastUsed = Date.now()
    this.sessionFingerprints(sessionId).set(path, current)
    return {
      status: 'hit',
      reason: 'unchanged',
      observation: {
        output: cached.output,
        evidenceId: cached.evidenceId,
        contentHash: cached.contentHash,
        recordedAt: cached.recordedAt,
      },
    }
  }

  async recordReadObservation(
    sessionId: string,
    path: string,
    viewKey: string,
    output: string,
  ): Promise<WorkspaceReadObservation | null> {
    const fingerprint = await this.fingerprint(path)
    if (!fingerprint) return null
    this.sessionFingerprints(sessionId).set(path, fingerprint)
    if (output.length > MAX_CACHED_READ_OBSERVATION_CHARS) return null

    const recordedAt = new Date().toISOString()
    const evidenceId = `read-${createHash('sha256')
      .update(`${path}\0${viewKey}\0${fingerprint.contentHash}`)
      .digest('hex')
      .slice(0, 16)}`
    const observation: CachedReadObservation = {
      path,
      viewKey,
      output,
      evidenceId,
      contentHash: fingerprint.contentHash,
      recordedAt,
      lastUsed: Date.now(),
    }
    const bucket = this.sessionObservations(sessionId)
    bucket.set(this.observationKey(path, viewKey), observation)
    this.pruneObservationBucket(bucket)
    return observation
  }

  dispose(sessionId: string): void {
    this.bySession.delete(sessionId)
    this.observationsBySession.delete(sessionId)
  }

  private sessionFingerprints(sessionId: string): Map<string, FileFingerprint> {
    let bucket = this.bySession.get(sessionId)
    if (!bucket) {
      bucket = new Map()
      this.bySession.set(sessionId, bucket)
    }
    return bucket
  }

  private sessionObservations(sessionId: string): Map<string, CachedReadObservation> {
    let bucket = this.observationsBySession.get(sessionId)
    if (!bucket) {
      bucket = new Map()
      this.observationsBySession.set(sessionId, bucket)
    }
    return bucket
  }

  private observationKey(path: string, viewKey: string): string {
    return `${path}\0${viewKey}`
  }

  private invalidateReadObservations(sessionId: string, path: string): void {
    const bucket = this.observationsBySession.get(sessionId)
    if (!bucket) return
    for (const [key, entry] of bucket) {
      if (entry.path === path) bucket.delete(key)
    }
    if (bucket.size === 0) this.observationsBySession.delete(sessionId)
  }

  private pruneObservationBucket(bucket: Map<string, CachedReadObservation>): void {
    let totalChars = [...bucket.values()].reduce((sum, entry) => sum + entry.output.length, 0)
    while (
      bucket.size > MAX_CACHED_READ_OBSERVATIONS_PER_SESSION
      || totalChars > MAX_CACHED_READ_CHARS_PER_SESSION
    ) {
      const oldest = [...bucket.entries()]
        .sort((left, right) => left[1].lastUsed - right[1].lastUsed)[0]
      if (!oldest) break
      bucket.delete(oldest[0])
      totalChars -= oldest[1].output.length
    }
  }

  private async fingerprint(path: string): Promise<FileFingerprint | null> {
    try {
      const info = await stat(path)
      const content = await readFile(path)
      return {
        mtimeMs: info.mtimeMs,
        size: info.size,
        contentHash: createHash('sha256').update(content).digest('hex'),
        recordedAt: new Date().toISOString(),
      }
    } catch {
      return null
    }
  }
}

export function describeStaleSignal(signal: WorkspaceStaleSignal): string {
  const sizeDelta = signal.currentSize - signal.baselineSize
  const sizeNote = sizeDelta === 0
    ? 'size unchanged'
    : `size ${sizeDelta > 0 ? '+' : ''}${sizeDelta} bytes`
  return `${signal.path}: changed externally (${sizeNote})`
}
