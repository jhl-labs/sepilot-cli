import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { SessionMeta } from '@sepilotd/core'

/**
 * Lazy session index — only loads session metadata on demand.
 * Keeps a lightweight in-memory index of IDs + titles.
 */
export class LazySessionIndex {
  private sessionsDir: string
  private lightIndex: Map<string, { title: string; updatedAt: string }> = new Map()
  private fullCache: Map<string, SessionMeta> = new Map()

  constructor(sessionsDir: string) {
    this.sessionsDir = sessionsDir
  }

  async init(): Promise<void> {
    // Load just the light index (IDs + titles) from index.json
    try {
      const data = await readFile(join(this.sessionsDir, 'index.json'), 'utf-8')
      const parsed = JSON.parse(data)
      for (const s of parsed.sessions ?? []) {
        this.lightIndex.set(s.id, { title: s.title, updatedAt: s.updatedAt })
      }
    } catch { /* no index yet */ }
  }

  /** Get count without loading all metadata */
  count(): number {
    return this.lightIndex.size
  }

  /** Get session IDs sorted by updatedAt desc */
  getIds(limit?: number): string[] {
    const sorted = [...this.lightIndex.entries()].sort((a, b) => b[1].updatedAt.localeCompare(a[1].updatedAt))
    return (limit ? sorted.slice(0, limit) : sorted).map(([id]) => id)
  }

  /** Quick title lookup without full load */
  getTitle(id: string): string | undefined {
    return this.lightIndex.get(id)?.title
  }

  /** Register a new session in the light index */
  register(id: string, title: string, updatedAt: string): void {
    this.lightIndex.set(id, { title, updatedAt })
  }

  /** Remove from index */
  remove(id: string): void {
    this.lightIndex.delete(id)
    this.fullCache.delete(id)
  }

  /** Update timestamp */
  touch(id: string, updatedAt: string): void {
    const entry = this.lightIndex.get(id)
    if (entry) entry.updatedAt = updatedAt
  }
}
