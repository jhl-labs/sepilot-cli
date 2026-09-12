import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises'
import { isAbsolute, join, relative, resolve } from 'node:path'
import type { Artifact } from '../agent/artifacts.js'
import { assertSafeId } from '../utils/safe-id.js'

export interface RecentArtifactEntry {
  sessionId: string
  artifact: Artifact
  modifiedAt: number
}

function compareArtifactIds(left: string, right: string): number {
  const leftMatch = left.match(/^artifact-(\d+)$/)
  const rightMatch = right.match(/^artifact-(\d+)$/)
  if (leftMatch && rightMatch) {
    return Number(leftMatch[1]) - Number(rightMatch[1])
  }
  return left.localeCompare(right)
}

export class ArtifactStore {
  constructor(private readonly baseDir: string) {}

  private sessionDir(sessionId: string): string {
    assertSafeId(sessionId, 'artifact sessionId')
    const root = resolve(this.baseDir)
    const dir = resolve(join(root, sessionId))
    const relativeDir = relative(root, dir)
    if (relativeDir.startsWith('..') || isAbsolute(relativeDir)) {
      throw new Error('invalid artifact sessionId')
    }
    return dir
  }

  async save(sessionId: string, artifact: Artifact): Promise<Artifact> {
    assertSafeId(artifact.id, 'artifact id')
    const key = assertSafeId(artifact.key ?? artifact.id, 'artifact key')
    const dir = this.sessionDir(sessionId)
    await mkdir(dir, { recursive: true })
    const existing = await this.findByKey(dir, key)
    const saved: Artifact = {
      ...artifact,
      id: existing?.id ?? artifact.id,
      key,
      version: (existing?.version ?? 0) + 1,
    }
    await writeFile(
      join(dir, `${saved.id}.json`),
      JSON.stringify(saved, null, 2),
      'utf-8',
    )
    return saved
  }

  async listBySession(sessionId: string): Promise<Artifact[]> {
    const dir = this.sessionDir(sessionId)
    try {
      const files = await readdir(dir)
      const artifacts: Artifact[] = []

      for (const file of files) {
        if (!file.endsWith('.json')) continue
        const raw = await readFile(join(dir, file), 'utf-8')
        artifacts.push(JSON.parse(raw) as Artifact)
      }

      return artifacts.sort((left, right) =>
        compareArtifactIds(left.id, right.id),
      )
    } catch {
      return []
    }
  }

  private async findByKey(dir: string, key: string): Promise<Artifact | null> {
    try {
      const files = await readdir(dir)
      for (const file of files) {
        if (!file.endsWith('.json')) continue
        const raw = await readFile(join(dir, file), 'utf-8')
        const artifact = JSON.parse(raw) as Artifact
        if ((artifact.key ?? artifact.id) === key) {
          return artifact
        }
      }
    } catch {
      return null
    }
    return null
  }

  /**
   * List recent artifacts across sessions, newest first by file mtime.
   * Artifact payloads do not currently carry createdAt, so filesystem
   * modification time is the only honest global recency signal.
   */
  async listRecent(limit: number): Promise<RecentArtifactEntry[]> {
    const safeLimit = Math.max(1, Math.min(50, Math.floor(limit) || 20))
    const root = resolve(this.baseDir)
    let sessionDirs: string[]
    try {
      sessionDirs = await readdir(root)
    } catch {
      return []
    }

    const candidates: Array<{ sessionId: string; path: string; modifiedAt: number }> = []
    for (const sessionId of sessionDirs) {
      try {
        assertSafeId(sessionId, 'artifact sessionId')
        const dir = this.sessionDir(sessionId)
        const files = await readdir(dir)
        for (const file of files) {
          if (!file.endsWith('.json')) continue
          const artifactId = file.slice(0, -'.json'.length)
          assertSafeId(artifactId, 'artifact id')
          const path = join(dir, file)
          const info = await stat(path)
          if (!info.isFile()) continue
          candidates.push({ sessionId, path, modifiedAt: info.mtimeMs })
        }
      } catch {
        // Ignore malformed or concurrently removed session directories.
      }
    }

    const recent = candidates
      .sort((left, right) => right.modifiedAt - left.modifiedAt)
      .slice(0, safeLimit)
    const entries: RecentArtifactEntry[] = []
    for (const candidate of recent) {
      try {
        entries.push({
          sessionId: candidate.sessionId,
          artifact: JSON.parse(await readFile(candidate.path, 'utf-8')) as Artifact,
          modifiedAt: candidate.modifiedAt,
        })
      } catch {
        // Skip corrupt artifact JSON without failing the whole gallery.
      }
    }
    return entries
  }
}
