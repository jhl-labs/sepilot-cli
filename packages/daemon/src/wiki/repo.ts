import type { SqliteDatabase } from '../db/sqlite.js'
import { openDomainDb } from '../storage/domain-db.js'
import type {
  WikiMoveInput,
  WikiNode,
  WikiNodeInput,
} from './schema.js'

interface Row {
  id: string
  parent_id: string | null
  title: string
  icon: string | null
  group_name: string | null
  order: number
  body: string
  updated_at: number
}

function rowToNode(r: Row): WikiNode {
  return {
    id: r.id,
    parentId: r.parent_id,
    title: r.title,
    icon: r.icon,
    group: r.group_name,
    order: r.order,
    body: r.body,
    updatedAt: r.updated_at,
  }
}

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS wiki_nodes (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    title TEXT NOT NULL,
    icon TEXT,
    group_name TEXT,
    "order" INTEGER NOT NULL DEFAULT 0,
    body TEXT NOT NULL DEFAULT '',
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES wiki_nodes(id) ON DELETE CASCADE
  )`,
  ).run()
  db.prepare(
    'CREATE INDEX IF NOT EXISTS wiki_nodes_parent_idx ON wiki_nodes(parent_id, "order")',
  ).run()
  db.prepare(
    `CREATE VIRTUAL TABLE IF NOT EXISTS wiki_nodes_fts USING fts5(
      title, body, content='wiki_nodes', content_rowid='rowid'
    )`,
  ).run()
  // Triggers keep the FTS index in sync after this point. For databases
  // that pre-existed the FTS table, the index is empty until rebuilt; the
  // line below fills it once. The 'rebuild' command is idempotent and
  // cheap on small tables.
  db.prepare(
    `INSERT INTO wiki_nodes_fts(wiki_nodes_fts) VALUES ('rebuild')`,
  ).run()
  db.prepare(
    `CREATE TRIGGER IF NOT EXISTS wiki_nodes_ai AFTER INSERT ON wiki_nodes BEGIN
      INSERT INTO wiki_nodes_fts(rowid, title, body) VALUES (new.rowid, new.title, new.body);
    END;`,
  ).run()
  db.prepare(
    `CREATE TRIGGER IF NOT EXISTS wiki_nodes_ad AFTER DELETE ON wiki_nodes BEGIN
      INSERT INTO wiki_nodes_fts(wiki_nodes_fts, rowid, title, body) VALUES ('delete', old.rowid, old.title, old.body);
    END;`,
  ).run()
  db.prepare(
    `CREATE TRIGGER IF NOT EXISTS wiki_nodes_au AFTER UPDATE ON wiki_nodes BEGIN
      INSERT INTO wiki_nodes_fts(wiki_nodes_fts, rowid, title, body) VALUES ('delete', old.rowid, old.title, old.body);
      INSERT INTO wiki_nodes_fts(rowid, title, body) VALUES (new.rowid, new.title, new.body);
    END;`,
  ).run()
}

export interface WikiSearchHit extends WikiNode {
  /** Snippet of the body around the matching terms (HTML-safe plain text). */
  snippet: string
}

export interface WikiRepo {
  tree(): WikiNode[]
  upsert(input: WikiNodeInput): WikiNode
  remove(id: string): void
  move(id: string, input: WikiMoveInput): WikiNode
  search(query: string, limit?: number): WikiSearchHit[]
  importNodes(nodes: readonly WikiNode[], options: { replace: boolean }): void
}

export function createWikiRepo(): WikiRepo {
  const db = openDomainDb({ name: 'wiki' })
  ensureSchema(db)
  return {
    tree() {
      return (
        db
          .prepare(
            'SELECT id, parent_id, title, icon, group_name, "order", body, updated_at FROM wiki_nodes ORDER BY parent_id, "order"',
          )
          .all() as Row[]
      ).map(rowToNode)
    },
    upsert(input) {
      return db.transaction(() => {
      const id = input.id ?? crypto.randomUUID()
      const parentId = input.parentId ?? null
      let order: number
      const existing = db
        .prepare('SELECT "order", body, updated_at FROM wiki_nodes WHERE id=?')
        .get(id) as { order: number; body: string; updated_at: number } | undefined
      if (input.expectedUpdatedAt !== undefined && existing?.updated_at !== input.expectedUpdatedAt) {
        throw Object.assign(new Error('Document changed on another device. Reload before saving; your draft is preserved.'), { statusCode: 409 })
      }
      const now = Math.max(Date.now(), (existing?.updated_at ?? 0) + 1)
      if (existing) {
        order = existing.order
      } else {
        const maxRow = db
          .prepare(
            'SELECT COALESCE(MAX("order"), -1) AS m FROM wiki_nodes WHERE parent_id IS ?',
          )
          .get(parentId) as { m: number }
        order = maxRow.m + 1
      }
      // Preserve existing body when the caller does not include one — the
      // edit dialog sends only metadata, and clobbering the body to '' on
      // every metadata edit destroys content.
      const body =
        input.body !== undefined ? input.body : (existing?.body ?? '')
      db.prepare(
        `INSERT INTO wiki_nodes (id, parent_id, title, icon, group_name, "order", body, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET parent_id=excluded.parent_id, title=excluded.title, icon=excluded.icon, group_name=excluded.group_name, body=excluded.body, updated_at=excluded.updated_at`,
      ).run(
        id,
        parentId,
        input.title,
        input.icon ?? null,
        input.group ?? null,
        order,
        body,
        now,
      )
      return rowToNode(
        db
          .prepare(
            'SELECT id, parent_id, title, icon, group_name, "order", body, updated_at FROM wiki_nodes WHERE id=?',
          )
          .get(id) as Row,
      )
      })()
    },
    remove(id) {
      db.prepare('DELETE FROM wiki_nodes WHERE id=?').run(id)
    },
    search(query, limit = 50) {
      const q = query.trim()
      if (!q) return []
      // Escape FTS5 syntax characters by quoting each whitespace-separated
      // term so user queries like `name: foo` don't blow up the parser.
      const ftsQuery = q
        .split(/\s+/)
        .filter(Boolean)
        .map((term) => `"${term.replace(/"/g, '""')}"*`)
        .join(' ')
      try {
        const rows = db
          .prepare(
            `SELECT n.id, n.parent_id, n.title, n.icon, n.group_name, n."order", n.body, n.updated_at,
              snippet(wiki_nodes_fts, 1, '', '', '…', 16) AS snippet
            FROM wiki_nodes n
            JOIN wiki_nodes_fts f ON f.rowid = n.rowid
            WHERE wiki_nodes_fts MATCH ?
            ORDER BY rank
            LIMIT ?`,
          )
          .all(ftsQuery, Math.max(1, Math.min(200, limit))) as (Row & {
          snippet: string
        })[]
        return rows.map((r) => ({ ...rowToNode(r), snippet: r.snippet }))
      } catch {
        // FTS5 may reject odd queries; fall back to a LIKE search so the
        // user always gets a deterministic response instead of a 500.
        const like = `%${q.replace(/[%_]/g, (c) => `\\${c}`)}%`
        const rows = db
          .prepare(
            `SELECT id, parent_id, title, icon, group_name, "order", body, updated_at
            FROM wiki_nodes
            WHERE title LIKE ? ESCAPE '\\' OR body LIKE ? ESCAPE '\\'
            ORDER BY updated_at DESC
            LIMIT ?`,
          )
          .all(like, like, Math.max(1, Math.min(200, limit))) as Row[]
        return rows.map((r) => ({
          ...rowToNode(r),
          snippet: r.body.slice(0, 120),
        }))
      }
    },
    move(id, input) {
      const now = Date.now()
      const tx = db.transaction(() => {
        db.prepare(
          'UPDATE wiki_nodes SET "order"="order"+1 WHERE parent_id IS ? AND "order">=?',
        ).run(input.parentId, input.order)
        db.prepare(
          'UPDATE wiki_nodes SET parent_id=?, "order"=?, updated_at=? WHERE id=?',
        ).run(input.parentId, input.order, now, id)
      })
      tx()
      return rowToNode(
        db
          .prepare(
            'SELECT id, parent_id, title, icon, group_name, "order", body, updated_at FROM wiki_nodes WHERE id=?',
          )
          .get(id) as Row,
      )
    },
    importNodes(nodes, options) {
      const write = db.prepare(
        `INSERT INTO wiki_nodes (id, parent_id, title, icon, group_name, "order", body, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          parent_id=excluded.parent_id,
          title=excluded.title,
          icon=excluded.icon,
          group_name=excluded.group_name,
          "order"=excluded."order",
          body=excluded.body,
          updated_at=excluded.updated_at`,
      )
      const tx = db.transaction(() => {
        if (options.replace) db.prepare('DELETE FROM wiki_nodes').run()
        for (const node of nodes) {
          write.run(
            node.id,
            node.parentId,
            node.title,
            node.icon,
            node.group,
            node.order,
            node.body,
            node.updatedAt,
          )
        }
      })
      tx()
    },
  }
}
