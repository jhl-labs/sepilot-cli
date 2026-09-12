import { retractDerivedMemories } from './retractions.js'
import { createHash, randomUUID } from 'node:crypto'
import { dirname } from 'node:path'
import type { SqliteDatabase } from '../db/sqlite.js'
import { FileMemory } from './file-memory.js'
import { canonicalFileMemoryScopeKey, type ScopedFileMemoryRegistry } from './scoped-file-memory.js'

export const normalizeProjectionText = (text: string) => text.normalize('NFKC').trim().replace(/\s+/g, ' ')
export const projectionItems = (text: string) => text.split('\n')
  .filter((line) => /^\s*-\s+/.test(line)).map((line) => normalizeProjectionText(line.replace(/^\s*-\s+/, '')))

/** SQLite outbox survives crashes between the authoritative write and file IO. */
export class MemoryProjectionSync {
  private flushing?: Promise<void>
  constructor(private readonly db: SqliteDatabase, private readonly registry: ScopedFileMemoryRegistry) {}

  flush(): Promise<void> {
    const next = (this.flushing ?? Promise.resolve()).catch(() => {}).then(() => this.drain())
    this.flushing = next
    void next.finally(() => { if (this.flushing === next) this.flushing = undefined }).catch(() => {})
    return next
  }

  private async drain(): Promise<void> {
    retractDerivedMemories(this.db)
    this.db.prepare(`INSERT INTO memory_projection_changes(memory_id, old_content, old_tags, new_content)
      SELECT id, content, COALESCE(tags, '[]'), NULL FROM memories
      WHERE COALESCE(json_extract(metadata, '$.memory.status'), 'active') != 'active'
        OR json_extract(metadata, '$.memory.validUntil') IS NOT NULL
        OR json_extract(metadata, '$.memory.validFrom') IS NOT NULL
        OR json_extract(metadata, '$.memory.subject') IS NOT NULL
        OR json_extract(metadata, '$.memory.reality') IS NOT NULL`).run()
    for (;;) {
      const change = this.db.prepare('SELECT * FROM memory_projection_changes ORDER BY sequence LIMIT 1').get() as {
        sequence: number; memory_id: string; old_content: string; old_tags: string; new_content: string | null
      } | undefined
      if (!change) return
      const tags = JSON.parse(change.old_tags) as string[]
      // A separate instance bypasses the reverse-sync listener. Writes are
      // awaited, idempotent, and acknowledged only after the projection lands.
      const file = new FileMemory(dirname(this.registry.get(tags).getMemoryPath()))
      for (const section of await file.readMemorySections()) {
        const items = projectionItems(section.content)
        if (!items.includes(normalizeProjectionText(change.old_content))) continue
        const next = section.content.split('\n').flatMap((line) => {
          if (!/^\s*-\s+/.test(line) || normalizeProjectionText(line.replace(/^\s*-\s+/, '')) !== normalizeProjectionText(change.old_content)) return [line]
          return change.new_content ? [`${line.match(/^\s*/)?.[0] ?? ''}- ${change.new_content}`] : []
        }).join('\n').trim()
        if (next) await file.replaceMemorySection(section.title, next)
        else await file.deleteMemorySection(section.title)
      }
      this.db.prepare('DELETE FROM memory_projection_changes WHERE sequence = ?').run(change.sequence)
    }
  }

  /** Exact provenance-preserving reverse sync, never fuzzy deletion. */
  async reconcileFile(key: string, before: string, after: string): Promise<void> {
    const beforeItems = new Set(projectionItems(before))
    const afterItems = new Set(projectionItems(after))
    const added = [...afterItems].filter((item) => !beforeItems.has(item))
    const removed = new Set([...beforeItems].filter((item) => !afterItems.has(item)))
    if (!removed.size) return
    const rows = this.db.prepare('SELECT id, content, tags, source FROM memories').all() as Array<{ id: string; content: string; tags: string; source: string }>
    this.db.transaction(() => {
      for (const row of rows) {
        if (canonicalFileMemoryScopeKey(JSON.parse(row.tags ?? '[]')) !== key) continue
        if (removed.has(normalizeProjectionText(row.content))) {
          const corrected = removed.size === 1 && added.length === 1
          const before = { id: row.id, content: row.content, source: row.source, tags: JSON.parse(row.tags ?? '[]') }
          this.db.prepare(`INSERT INTO memory_audit(id, memory_id, action, actor, reason, before_json, after_json, created_at)
            VALUES(?, ?, ?, 'file-memory', 'Long-term file correction or forgetting', ?, ?, ?)`)
            .run(randomUUID(), row.id, corrected ? 'updated' : 'deleted', JSON.stringify(before), corrected ? JSON.stringify({ ...before, content: added[0] }) : null, new Date().toISOString())
          if (removed.size === 1 && added.length === 1) {
            this.db.prepare(`UPDATE memories SET content = ?, updated_at = datetime('now'),
              embedding = NULL, embedding_model = NULL, embedding_state = 'pending' WHERE id = ?`).run(added[0], row.id)
          } else this.db.prepare('DELETE FROM memories WHERE id = ?').run(row.id)
        }
      }
    })()
    // File writer already persisted after; outbox replay now has no matching
    // old item and cannot recreate the deleted content.
    await this.flush()
  }
}

export function projectionFingerprint(content: string): string {
  return createHash('sha256').update(normalizeProjectionText(content)).digest('hex')
}
