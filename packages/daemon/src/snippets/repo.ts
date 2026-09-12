import type { SqliteDatabase } from '../db/sqlite.js'
import { openDomainDb } from '../storage/domain-db.js'
import type { Snippet, SnippetGistLink, SnippetInput } from './schema.js'

interface Row {
  id: string
  title: string
  language: string
  body: string
  tags: string
  gist_id: string | null
  gist_file: string | null
  gist_html_url: string | null
  gist_updated_at: string | null
  gist_synced_at: number | null
}

function rowToSnippet(r: Row): Snippet {
  let tags: string[] = []
  try {
    tags = JSON.parse(r.tags) as string[]
  } catch {
    tags = []
  }
  return {
    id: r.id,
    title: r.title,
    language: r.language,
    body: r.body,
    tags,
    gist: r.gist_id
      ? {
          id: r.gist_id,
          file: r.gist_file ?? '',
          htmlUrl: r.gist_html_url,
          updatedAt: r.gist_updated_at,
          syncedAt: r.gist_synced_at,
        }
      : null,
  }
}

function tableHasColumn(db: SqliteDatabase, table: string, column: string): boolean {
  const rows = db.pragma(`table_info(${table})`) as Array<{ name: string }>
  return rows.some((row) => row.name === column)
}

function ensureColumn(db: SqliteDatabase, table: string, column: string, ddl: string): void {
  if (tableHasColumn(db, table, column)) return
  try {
    db.prepare(`ALTER TABLE ${table} ADD COLUMN ${ddl}`).run()
  } catch (error) {
    // Multiple daemon/test processes can open the same legacy database at
    // once. If another process completed this migration after our PRAGMA,
    // accept only the confirmed duplicate-column race and surface all other
    // schema errors.
    if (
      error instanceof Error
      && /duplicate column name/i.test(error.message)
      && tableHasColumn(db, table, column)
    ) {
      return
    }
    throw error
  }
}

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS snippets (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, language TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
    gist_id TEXT, gist_file TEXT, gist_html_url TEXT,
    gist_updated_at TEXT, gist_synced_at INTEGER
  )`,
  ).run()
  ensureColumn(db, 'snippets', 'gist_id', 'gist_id TEXT')
  ensureColumn(db, 'snippets', 'gist_file', 'gist_file TEXT')
  ensureColumn(db, 'snippets', 'gist_html_url', 'gist_html_url TEXT')
  ensureColumn(db, 'snippets', 'gist_updated_at', 'gist_updated_at TEXT')
  ensureColumn(db, 'snippets', 'gist_synced_at', 'gist_synced_at INTEGER')
  db.prepare(
    `CREATE VIRTUAL TABLE IF NOT EXISTS snippets_fts USING fts5(
    title, body, tags, content='snippets', content_rowid='rowid'
  )`,
  ).run()
  db.prepare(
    `CREATE TRIGGER IF NOT EXISTS snippets_ai AFTER INSERT ON snippets BEGIN
    INSERT INTO snippets_fts(rowid, title, body, tags) VALUES (new.rowid, new.title, new.body, new.tags);
  END;`,
  ).run()
  db.prepare(
    `CREATE TRIGGER IF NOT EXISTS snippets_ad AFTER DELETE ON snippets BEGIN
    INSERT INTO snippets_fts(snippets_fts, rowid, title, body, tags) VALUES ('delete', old.rowid, old.title, old.body, old.tags);
  END;`,
  ).run()
  db.prepare(
    `CREATE TRIGGER IF NOT EXISTS snippets_au AFTER UPDATE ON snippets BEGIN
    INSERT INTO snippets_fts(snippets_fts, rowid, title, body, tags) VALUES ('delete', old.rowid, old.title, old.body, old.tags);
    INSERT INTO snippets_fts(rowid, title, body, tags) VALUES (new.rowid, new.title, new.body, new.tags);
  END;`,
  ).run()
  // Existing databases can predate the FTS table and its synchronization
  // triggers. Rebuild from the content table so those rows are searchable as
  // soon as the upgraded daemon opens the repository. The command is
  // idempotent and snippets is a user-curated, bounded collection.
  db.prepare(
    `INSERT INTO snippets_fts(snippets_fts) VALUES ('rebuild')`,
  ).run()
}

function buildLiteralFtsQuery(query: string): string {
  return query
    .trim()
    .split(/\s+/u)
    .filter((term) => /[\p{L}\p{N}]/u.test(term))
    .map((term) => `"${term.replace(/"/g, '""')}"*`)
    .join(' ')
}

function escapeLikePattern(query: string): string {
  return `%${query.replace(/[\\%_]/g, (character) => `\\${character}`)}%`
}

function searchWithLike(db: SqliteDatabase, query: string): Row[] {
  const pattern = escapeLikePattern(query)
  return db
    .prepare(
      `SELECT ${SNIPPET_COLUMNS} FROM snippets
      WHERE title LIKE ? ESCAPE '\\'
        OR body LIKE ? ESCAPE '\\'
        OR tags LIKE ? ESCAPE '\\'
      ORDER BY updated_at DESC
      LIMIT 200`,
    )
    .all(pattern, pattern, pattern) as Row[]
}

export interface SnippetsRepo {
  list(query?: { q?: string }): Snippet[]
  get(id: string): Snippet | null
  findByGist(gistId: string, file: string): Snippet | null
  upsert(input: SnippetInput): Snippet
  setGistLink(id: string, link: SnippetGistLink): Snippet
  clearGistLink(id: string): Snippet | null
  remove(id: string): void
}

const SNIPPET_COLUMNS = [
  'id',
  'title',
  'language',
  'body',
  'tags',
  'gist_id',
  'gist_file',
  'gist_html_url',
  'gist_updated_at',
  'gist_synced_at',
].join(', ')

export function createSnippetsRepo(): SnippetsRepo {
  const db = openDomainDb({ name: 'snippets' })
  ensureSchema(db)
  return {
    list(query) {
      const q = query?.q?.trim()
      if (!q) {
        return (
          db
            .prepare(
              `SELECT ${SNIPPET_COLUMNS} FROM snippets ORDER BY updated_at DESC`,
            )
            .all() as Row[]
        ).map(rowToSnippet)
      }
      const ftsQuery = buildLiteralFtsQuery(q)
      if (!ftsQuery) {
        return searchWithLike(db, q).map(rowToSnippet)
      }
      let rows: Row[]
      try {
        rows = db
          .prepare(
            `SELECT ${SNIPPET_COLUMNS.split(', ').map((column) => `s.${column}`).join(', ')} FROM snippets s JOIN snippets_fts f ON f.rowid = s.rowid WHERE snippets_fts MATCH ? ORDER BY rank LIMIT 200`,
          )
          .all(ftsQuery) as Row[]
      } catch {
        // FTS5 syntax/tokenizer behavior differs across SQLite builds. A
        // literal LIKE search keeps natural punctuation-heavy queries usable
        // instead of surfacing a 500 from the snippets route.
        rows = searchWithLike(db, q)
      }
      return rows.map(rowToSnippet)
    },
    get(id) {
      const row = db
        .prepare(`SELECT ${SNIPPET_COLUMNS} FROM snippets WHERE id=?`)
        .get(id) as Row | undefined
      return row ? rowToSnippet(row) : null
    },
    findByGist(gistId, file) {
      const row = db
        .prepare(
          `SELECT ${SNIPPET_COLUMNS} FROM snippets WHERE gist_id=? AND gist_file=? LIMIT 1`,
        )
        .get(gistId, file) as Row | undefined
      return row ? rowToSnippet(row) : null
    },
    upsert(input) {
      const now = Date.now()
      const id = input.id ?? crypto.randomUUID()
      db.prepare(
        `INSERT INTO snippets (id, title, language, body, tags, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET title=excluded.title, language=excluded.language, body=excluded.body, tags=excluded.tags, updated_at=excluded.updated_at`,
      ).run(
        id,
        input.title,
        input.language,
        input.body,
        JSON.stringify(input.tags),
        now,
        now,
      )
      return rowToSnippet(
        db
          .prepare(
            `SELECT ${SNIPPET_COLUMNS} FROM snippets WHERE id=?`,
          )
          .get(id) as Row,
      )
    },
    setGistLink(id, link) {
      db.prepare(
        `UPDATE snippets
        SET gist_id=?, gist_file=?, gist_html_url=?, gist_updated_at=?, gist_synced_at=?, updated_at=?
        WHERE id=?`,
      ).run(
        link.id,
        link.file,
        link.htmlUrl,
        link.updatedAt,
        link.syncedAt,
        Date.now(),
        id,
      )
      const row = db
        .prepare(`SELECT ${SNIPPET_COLUMNS} FROM snippets WHERE id=?`)
        .get(id) as Row | undefined
      if (!row) throw new Error(`Snippet not found: ${id}`)
      return rowToSnippet(row)
    },
    clearGistLink(id) {
      db.prepare(
        `UPDATE snippets
        SET gist_id=NULL, gist_file=NULL, gist_html_url=NULL, gist_updated_at=NULL, gist_synced_at=NULL, updated_at=?
        WHERE id=?`,
      ).run(Date.now(), id)
      const row = db
        .prepare(`SELECT ${SNIPPET_COLUMNS} FROM snippets WHERE id=?`)
        .get(id) as Row | undefined
      return row ? rowToSnippet(row) : null
    },
    remove(id) {
      db.prepare('DELETE FROM snippets WHERE id=?').run(id)
    },
  }
}
