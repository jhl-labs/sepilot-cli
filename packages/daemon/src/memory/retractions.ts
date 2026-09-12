import type { SqliteDatabase } from '../db/sqlite.js'
import { canonicalFileMemoryScopeKey } from './scoped-file-memory.js'

/** Conservative transitive invalidation of inferred dependents, including cycles. */
export function retractDerivedMemories(db: SqliteDatabase): void {
  db.transaction(() => {
    for (;;) {
      const rows = db.prepare(`SELECT DISTINCT m.id, m.tags, r.tags AS source_tags
        FROM memories m, json_each(json_extract(m.metadata, '$.memory.sourceIds')) source
        JOIN memory_retractions r ON r.memory_id = source.value
        WHERE json_extract(m.metadata, '$.memory.origin') != 'user'
          AND json_extract(m.metadata, '$.memory.status') != 'retracted'`).all() as Array<{ id: string; tags: string; source_tags: string }>
      let changed = false
      for (const row of rows) {
        if (canonicalFileMemoryScopeKey(JSON.parse(row.tags ?? '[]')) !== canonicalFileMemoryScopeKey(JSON.parse(row.source_tags))) continue
        db.prepare(`UPDATE memories SET metadata = json_set(metadata, '$.memory.status', 'retracted'),
          updated_at = datetime('now') WHERE id = ?`).run(row.id)
        changed = true
      }
      if (!changed) break
    }
  })()
}
