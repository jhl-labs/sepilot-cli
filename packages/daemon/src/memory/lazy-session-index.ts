import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { SessionMeta } from '@sepilotd/core'

export class LazySessionIndex {
  private sessionsDir: string
  private lightIndex: Map<string, { title: string; updatedAt: string }> = new Map()
  private fullCache: Map<string, SessionMeta> = new Map()

  constructor(sessionsDir: string) {
    this.sessionsDir = sessionsDir
  }

  async init(): Promise<void> {
    try {
      const data = await readFile(join(this.sessionsDir, 'index.json'), 'utf-8')
      const parsed = JSON.parse(data)
      for (const s of parsed.sessions ?? []) {
        this.lightIndex.set(s.id, { title: s.title, updatedAt: s.updatedAt })
      }
    } catch {}
  }

  count(): number {
    return this.lightIndex.size
  }

  getIds(limit?: number): string[] {
    const sorted = [...this.lightIndex.entries()].sort((a, b) => b[1].updatedAt.localeCompare(a[1].updatedAt))
    return (limit ? sorted.slice(0, limit) : sorted).map(([id]) => id)
  }

  getTitle(id: string): string | undefined {
    return this.lightIndex.get(id)?.title
  }

  register(id: string, title: string, updatedAt: string): void {
    this.lightIndex.set(id, { title, updatedAt })
  }

  remove(id: string): void {
    this.lightIndex.delete(id)
    this.fullCache.delete(id)
  }

  touch(id: string, updatedAt: string): void {
    const entry = this.lightIndex.get(id)
    if (entry) entry.updatedAt = updatedAt
  }
}
