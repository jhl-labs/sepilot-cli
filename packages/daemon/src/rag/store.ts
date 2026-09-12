import type { SqliteDatabase } from '../db/sqlite.js'
import { randomUUID } from 'node:crypto'
import { openDomainDb } from '../storage/domain-db.js'
import { chunkText } from './chunker.js'
import { hashEmbedder } from './embedder/hash.js'
import { createMemoryIndex, type MemoryIndex } from './memory-index.js'

export type RagSourceType = 'manual' | 'git' | 'web'
export type RagSyncStatus = 'success' | 'error' | null

export interface RagFolder {
  id: string
  name: string
  documents: number
  sourceType: RagSourceType
  tlsVerify?: boolean
  caCert?: string
  path?: string
  include?: string[]
  exclude?: string[]
  lastSyncedAt: number | null
  lastSyncStatus: RagSyncStatus
  lastSyncError: string | null
}

export interface RagDocument {
  id: string
  folderId: string
  title: string
  path?: string
  sourceFileId?: string
  size?: number
  updatedAt: number
}

export interface RagDocumentContent extends RagDocument {
  body: string
}

export interface SearchHit {
  documentId: string
  folderId: string
  title: string
  score: number
  snippet: string
}

interface DocumentRow {
  id: string
  folder_id: string
  title: string
  body: string
  path: string | null
  source_file_id: string | null
  size: number | null
  updated_at: number
}

interface FolderRow {
  id: string
  name: string
  c: number
  source_type: string | null
  tls_verify: number | null
  ca_cert: string | null
  path: string | null
  include_globs: string | null
  exclude_globs: string | null
  last_synced_at: number | null
  last_sync_status: string | null
  last_sync_error: string | null
}

function tableColumns(db: SqliteDatabase, table: string): Set<string> {
  return new Set(
    (db.prepare(`PRAGMA table_info(${table})`).all() as { name: string }[]).map((row) => row.name),
  )
}

function addColumnIfMissing(
  db: SqliteDatabase,
  table: string,
  column: string,
  definition: string,
): void {
  if (tableColumns(db, table).has(column)) return
  try {
    db.prepare(`ALTER TABLE ${table} ADD COLUMN ${definition}`).run()
  } catch (error) {
    if (error instanceof Error && /duplicate column name/i.test(error.message)) {
      return
    }
    throw error
  }
}

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS folders (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL
  )`,
  ).run()
  db.prepare(
    `CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    folder_id TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE CASCADE
  )`,
  ).run()

  addColumnIfMissing(db, 'folders', 'source_type', "source_type TEXT NOT NULL DEFAULT 'manual'")
  addColumnIfMissing(db, 'folders', 'tls_verify', 'tls_verify INTEGER NOT NULL DEFAULT 1')
  addColumnIfMissing(db, 'folders', 'ca_cert', 'ca_cert TEXT')
  addColumnIfMissing(db, 'folders', 'path', 'path TEXT')
  addColumnIfMissing(db, 'folders', 'include_globs', 'include_globs TEXT')
  addColumnIfMissing(db, 'folders', 'exclude_globs', 'exclude_globs TEXT')
  addColumnIfMissing(db, 'folders', 'last_synced_at', 'last_synced_at INTEGER')
  addColumnIfMissing(db, 'folders', 'last_sync_status', 'last_sync_status TEXT')
  addColumnIfMissing(db, 'folders', 'last_sync_error', 'last_sync_error TEXT')
  addColumnIfMissing(db, 'documents', 'path', 'path TEXT')
  addColumnIfMissing(db, 'documents', 'source_file_id', 'source_file_id TEXT')
  addColumnIfMissing(db, 'documents', 'size', 'size INTEGER')

  db.prepare(`CREATE INDEX IF NOT EXISTS documents_folder_idx ON documents(folder_id)`).run()
  db.prepare(
    `CREATE INDEX IF NOT EXISTS documents_source_file_idx ON documents(source_file_id)`,
  ).run()
}

function parseStringList(raw: string | null): string[] | undefined {
  if (!raw) return undefined
  try {
    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) return undefined
    return parsed.filter((item): item is string => typeof item === 'string')
  } catch {
    return undefined
  }
}

function stringifyStringList(value: string[] | undefined): string | null {
  if (!value?.length) return null
  return JSON.stringify(value)
}

function normalizeSourceType(value: string | null | undefined): RagSourceType {
  return value === 'git' || value === 'web' ? value : 'manual'
}

function normalizeSyncStatus(value: string | null): RagSyncStatus {
  return value === 'success' || value === 'error' ? value : null
}

function mapFolderRow(row: FolderRow): RagFolder {
  return {
    id: row.id,
    name: row.name,
    documents: row.c,
    sourceType: normalizeSourceType(row.source_type),
    tlsVerify: row.tls_verify !== 0,
    caCert: row.ca_cert ?? undefined,
    path: row.path ?? undefined,
    include: parseStringList(row.include_globs),
    exclude: parseStringList(row.exclude_globs),
    lastSyncedAt: row.last_synced_at,
    lastSyncStatus: normalizeSyncStatus(row.last_sync_status),
    lastSyncError: row.last_sync_error,
  }
}

function mapDocumentRow(row: DocumentRow): RagDocument {
  return {
    id: row.id,
    folderId: row.folder_id,
    title: row.title,
    path: row.path ?? undefined,
    sourceFileId: row.source_file_id ?? undefined,
    size: row.size ?? undefined,
    updatedAt: row.updated_at,
  }
}

export interface RagStore {
  listFolders(): RagFolder[]
  upsertFolder(input: {
    id?: string
    name: string
    sourceType?: RagSourceType
    tlsVerify?: boolean
    caCert?: string
    path?: string
    include?: string[]
    exclude?: string[]
  }): RagFolder
  recordFolderSync(
    id: string,
    result: {
      status: Exclude<RagSyncStatus, null>
      syncedAt?: number
      error?: string | null
    },
  ): void
  removeFolder(id: string): void
  listDocuments(folderId: string): RagDocument[]
  getDocument(id: string): RagDocumentContent | null
  upsertDocument(input: {
    id?: string
    folderId: string
    title: string
    body: string
    path?: string
    sourceFileId?: string
    size?: number
  }): Promise<RagDocument>
  removeDocument(id: string): void
  search(query: string, limit: number): Promise<SearchHit[]>
  info(): { engine: 'memory'; dimension: number; documents: number }
}

export function createRagStore(): RagStore {
  const db = openDomainDb({ name: 'rag', filename: 'rag.db' })
  ensureSchema(db)
  const index: MemoryIndex = createMemoryIndex()

  async function indexDocument(documentId: string, body: string): Promise<void> {
    const chunks = chunkText(body, { size: 512, overlap: 64 })
    if (chunks.length === 0) return
    const vectors = await hashEmbedder.embed(chunks.map((c) => c.text))
    index.replaceDocument(
      documentId,
      chunks.map((c, i) => ({
        chunkIndex: c.index,
        text: c.text,
        vector: vectors[i]!,
      })),
    )
  }

  // Rebuild in-memory index from DB on startup.
  const bootstrap = db.prepare('SELECT id, body FROM documents').all() as {
    id: string
    body: string
  }[]
  for (const row of bootstrap) {
    void indexDocument(row.id, row.body)
  }

  return {
    listFolders() {
      const rows = db
        .prepare(
          `SELECT
             f.id,
             f.name,
             f.source_type,
             f.tls_verify,
             f.ca_cert,
             f.path,
             f.include_globs,
             f.exclude_globs,
             f.last_synced_at,
             f.last_sync_status,
             f.last_sync_error,
             COUNT(d.id) AS c
           FROM folders f LEFT JOIN documents d ON d.folder_id = f.id
           GROUP BY
             f.id,
             f.name,
             f.source_type,
             f.tls_verify,
             f.ca_cert,
             f.path,
             f.include_globs,
             f.exclude_globs,
             f.last_synced_at,
             f.last_sync_status,
             f.last_sync_error
           ORDER BY f.name`,
        )
        .all() as FolderRow[]
      return rows.map(mapFolderRow)
    },
    upsertFolder(input) {
      const id = input.id ?? randomUUID()
      const sourceType = input.sourceType ?? (input.path ? 'git' : 'manual')
      db.prepare(
        `INSERT INTO folders (
           id,
           name,
           source_type,
           tls_verify,
           ca_cert,
           path,
           include_globs,
           exclude_globs
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           name=excluded.name,
           source_type=excluded.source_type,
           tls_verify=excluded.tls_verify,
           ca_cert=excluded.ca_cert,
           path=excluded.path,
           include_globs=excluded.include_globs,
           exclude_globs=excluded.exclude_globs`,
      ).run(
        id,
        input.name,
        sourceType,
        input.tlsVerify === false ? 0 : 1,
        input.caCert ?? null,
        input.path ?? null,
        stringifyStringList(input.include),
        stringifyStringList(input.exclude),
      )
      return {
        id,
        name: input.name,
        documents: 0,
        sourceType,
        tlsVerify: input.tlsVerify !== false,
        caCert: input.caCert,
        path: input.path,
        include: input.include?.length ? input.include : undefined,
        exclude: input.exclude?.length ? input.exclude : undefined,
        lastSyncedAt: null,
        lastSyncStatus: null,
        lastSyncError: null,
      }
    },
    recordFolderSync(id, result) {
      db.prepare(
        `UPDATE folders
         SET last_synced_at=?,
             last_sync_status=?,
             last_sync_error=?
         WHERE id=?`,
      ).run(result.syncedAt ?? Date.now(), result.status, result.error ?? null, id)
    },
    removeFolder(id) {
      const docs = db.prepare('SELECT id FROM documents WHERE folder_id=?').all(id) as {
        id: string
      }[]
      for (const d of docs) index.removeDocument(d.id)
      db.prepare('DELETE FROM folders WHERE id=?').run(id)
    },
    listDocuments(folderId) {
      return (
        db
          .prepare(
            `SELECT id, folder_id, title, body, path, source_file_id, size, updated_at
             FROM documents
             WHERE folder_id=?
             ORDER BY updated_at DESC`,
          )
          .all(folderId) as DocumentRow[]
      ).map(mapDocumentRow)
    },
    getDocument(id) {
      const row = db
        .prepare(
          `SELECT id, folder_id, title, body, path, source_file_id, size, updated_at
           FROM documents
           WHERE id=?`,
        )
        .get(id) as DocumentRow | undefined
      if (!row) return null
      return {
        id: row.id,
        folderId: row.folder_id,
        title: row.title,
        path: row.path ?? undefined,
        sourceFileId: row.source_file_id ?? undefined,
        size: row.size ?? undefined,
        body: row.body,
        updatedAt: row.updated_at,
      }
    },
    async upsertDocument(input) {
      const id = input.id ?? randomUUID()
      const now = Date.now()
      db.prepare(
        `INSERT INTO documents (
           id,
           folder_id,
           title,
           body,
           path,
           source_file_id,
           size,
           updated_at
         )
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           folder_id=excluded.folder_id,
           title=excluded.title,
           body=excluded.body,
           path=excluded.path,
           source_file_id=excluded.source_file_id,
           size=excluded.size,
           updated_at=excluded.updated_at`,
      ).run(
        id,
        input.folderId,
        input.title,
        input.body,
        input.path ?? null,
        input.sourceFileId ?? null,
        input.size ?? null,
        now,
      )
      await indexDocument(id, input.body)
      return {
        id,
        folderId: input.folderId,
        title: input.title,
        path: input.path,
        sourceFileId: input.sourceFileId,
        size: input.size,
        updatedAt: now,
      }
    },
    removeDocument(id) {
      db.prepare('DELETE FROM documents WHERE id=?').run(id)
      index.removeDocument(id)
    },
    async search(query, limit) {
      const [qVec] = await hashEmbedder.embed([query])
      const hits = index.search(qVec!, limit)
      if (hits.length === 0) return []
      const byRowid = hits.map((h) => ({ rowid: h.rowid, score: h.score }))
      const docCache = new Map<string, { folderId: string; title: string }>()
      return byRowid.flatMap((h) => {
        const chunk = index.get(h.rowid)
        if (!chunk) return []
        const cached = docCache.get(chunk.documentId)
        if (cached) {
          return [
            {
              documentId: chunk.documentId,
              folderId: cached.folderId,
              title: cached.title,
              score: h.score,
              snippet: chunk.text.slice(0, 240),
            },
          ]
        }
        const row = db
          .prepare('SELECT folder_id, title FROM documents WHERE id=?')
          .get(chunk.documentId) as { folder_id: string; title: string } | undefined
        if (!row) return []
        docCache.set(chunk.documentId, {
          folderId: row.folder_id,
          title: row.title,
        })
        return [
          {
            documentId: chunk.documentId,
            folderId: row.folder_id,
            title: row.title,
            score: h.score,
            snippet: chunk.text.slice(0, 240),
          },
        ]
      })
    },
    info() {
      const snap = index.snapshot()
      return {
        engine: 'memory',
        dimension: snap.dimension,
        documents: snap.documents,
      }
    },
  }
}
