import { memoryResetOwner } from './reset.js'
import { retractDerivedMemories } from './retractions.js'
import { canonicalFileMemoryScopeKey } from './scoped-file-memory.js'
import { MemoryProjectionSync } from './projection-sync.js'
import type { ScopedFileMemoryRegistryImpl } from './scoped-file-memory.js'
import { evidenceForWrite, memoryEvidenceSchema, isEvidenceActive } from './evidence.js'
import { buildMemorySearchFilter } from './search-filter.js'
import { cosineDistance } from './vector-backend.js'
import { mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import { createHash, randomUUID } from 'node:crypto'
import type {
  DocumentIngestInput,
  DocumentUpdateInput,
  DocumentListOptions,
  DocumentSearchOptions,
  MemoryAccessHotEntry,
  MemoryAccessStats,
  MemoryDocument,
  MemoryDocumentChunk,
  MemoryEntry,
  MemoryPinnedEntry,
  MergeMemoriesInput,
  MergeMemoriesResult,
  SemanticSearchOptions,
} from '@sepilotd/core'
import { openDatabase, type SqliteDatabase } from '../db/sqlite.js'
import { chunkDocumentForRag } from './document-chunker.js'
import { enrichDocumentChunk } from './document-citations.js'
import { runMigrations } from './migrations.js'
import { recoverSqliteDatabaseIfCorrupt } from './sqlite-recovery.js'
import { MEMORY_GRAPH_NODE_KINDS } from './types.js'
import {
  blobToFloat32Array,
  createVectorBackend,
  type CustomApiMemoryConfig,
  type MeilisearchMemoryConfig,
  type QdrantMemoryConfig,
  type SearchEngineMemoryConfig,
  type MemoryVectorBackend,
} from './vector-backend.js'
import type {
  MemoryAuditEntry,
  MemoryAuditListOptions,
  MemoryAuditRecordInput,
  MemoryAuditSnapshot,
  MemoryGraphEdge,
  MemoryGraphEvidenceEntry,
  MemoryGraphMaintenanceResult,
  MemoryGraphNode,
  MemoryGraphNodeInput,
  MemoryGraphQualityReport,
  MemoryGraphQualitySignal,
  MemoryGraphRepairDecisionInput,
  MemoryGraphRepairDecisionResult,
  MemoryGraphRepairDecisionSkip,
  MemoryGraphRepairDecisionUpdate,
  MemoryGraphRepairAction,
  MemoryGraphRepairProposal,
  MemoryGraphRepairResult,
  MemoryGraphSearchResult,
  MemoryGraphStats,
  MemoryGraphWikiPage,
  MemoryGraphUpsertInput,
  MemoryLifecycleOptions,
  MemoryLifecycleStatus,
  MemoryMaintenanceOptions,
  MemoryMaintenanceResult,
  MemorySemanticStatus,
  SemanticIndexRuntimeStatus,
  SemanticMemoryStore,
  MemoryVectorBackendKind,
} from './types.js'

export type {
  MemoryAuditEntry,
  MemoryAuditListOptions,
  MemoryAuditRecordInput,
  MemoryAuditSnapshot,
  MemoryGraphEdge,
  MemoryGraphEvidenceEntry,
  MemoryGraphMaintenanceResult,
  MemoryGraphNode,
  MemoryGraphNodeInput,
  MemoryGraphQualityReport,
  MemoryGraphQualitySignal,
  MemoryGraphSearchResult,
  MemoryGraphStats,
  MemoryGraphStore,
  MemoryGraphWikiPage,
  MemoryGraphUpsertInput,
  MemoryLifecycleOptions,
  MemoryLifecycleStatus,
  MemoryMaintenanceOptions,
  MemoryMaintenanceResult,
  MemorySemanticStatus,
  SemanticIndexRuntimeStatus,
  SemanticMemoryStore,
  MemoryVectorBackendKind,
} from './types.js'

export interface MemoryEmbedder {
  providerId: string
  embed(texts: string[], model?: string): Promise<number[][]>
}

// Bounded retry for transient embedding failures. Without it, a single
// transient provider error permanently parks a memory outside semantic search
// (the backfill selector excluded all 'failed' rows forever). A failed row is
// retried up to MAX attempts, each after a cooldown window, then parked.
function resolveEmbedMaxAttempts(): number {
  const raw = Number(process.env.SEPILOTD_EMBED_MAX_ATTEMPTS)
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 5
}

function resolveEmbedRetryMinutes(): number {
  const raw = Number(process.env.SEPILOTD_EMBED_RETRY_MINUTES)
  // 0 is valid (retry on the very next pass); only fall back on unset/invalid.
  return Number.isFinite(raw) && raw >= 0 ? raw : 5
}

interface SqliteSemanticIndexOptions {
  embedder?: MemoryEmbedder
  embeddingModel?: string
  configuredProviderId?: string
  vectorBackend?: 'auto' | MemoryVectorBackendKind
  qdrant?: QdrantMemoryConfig
  opensearch?: SearchEngineMemoryConfig
  elasticsearch?: SearchEngineMemoryConfig
  meilisearch?: MeilisearchMemoryConfig
  customApi?: CustomApiMemoryConfig
}

interface SemanticIndexStateRow {
  provider_id: string | null
  model: string | null
  dimensions: number | null
  status: MemorySemanticStatus
  last_error: string | null
}

interface MemoryRow {
  rowid: number
  id: string
  content: string
}

interface DocumentRow {
  id: string
  title: string
  path: string | null
  mime_type: string | null
  source_file_id: string | null
  tags: string | null
  chunk_count: number
  created_at: string
  updated_at: string
}

interface DocumentChunkRow {
  id: string
  content: string
  tags: string | null
  document_id: string
  chunk_index: number
  chunk_count: number
  chunk_start: number
  chunk_end: number
  chunk_title: string | null
  document_title: string
  path: string | null
  mime_type: string | null
  source_file_id: string | null
}

interface MemorySnapshotRow {
  id: string
  content: string
  source: MemoryEntry['source']
  tags: string | null
}

interface MemoryAuditRow {
  id: string
  memory_id: string
  action: MemoryAuditEntry['action']
  actor: string
  reason: string | null
  before_json: string | null
  after_json: string | null
  created_at: string
}

interface MemoryLifecycleCountRow {
  total: number
  conversation: number
  document: number
  skill: number
  user: number
  staleConversation: number
  lowImportanceConversation: number
  pruneCandidate: number
}

interface MemoryGraphNodeRow {
  id: string
  label: string
  kind: string
  aliases: string | null
  tags: string | null
  evidence_memory_ids: string | null
  confidence: number
  created_at: string
  updated_at: string
  last_seen_at: string
}

interface MemoryGraphEdgeRow {
  id: string
  from_node_id: string
  to_node_id: string
  relation: string
  tags: string | null
  evidence_memory_ids: string | null
  confidence: number
  created_at: string
  updated_at: string
  last_seen_at: string
}

interface MemoryGraphEvidenceStats {
  evidenceIds: string[]
  activeIds: string[]
  inactiveIds: string[]
  missingIds: string[]
}

const RRF_K = 60
const MEMORY_MAINTENANCE_ID = '__memory_maintenance__'
const GRAPH_LOW_CONFIDENCE_THRESHOLD = 0.45
const GRAPH_STALE_AFTER_DAYS = 180
const DOCUMENT_EMBEDDING_BATCH_SIZE = 32
// Ceiling on chunks produced by a single document ingest, so a pathologically
// large document cannot flood the memories table + embedding queue.
const MAX_DOCUMENT_CHUNKS = 2_000

/** Run each attempt in order and take the first that returns any rows. */
function firstNonEmpty<T>(...attempts: Array<() => T[]>): T[] {
  for (const attempt of attempts) {
    const rows = attempt()
    if (rows.length > 0) return rows
  }
  return []
}

export class SqliteSemanticIndex implements SemanticMemoryStore {
  private projectionSync?: MemoryProjectionSync
  async configureFileMemory(registry: ScopedFileMemoryRegistryImpl): Promise<void> {
    this.projectionSync = new MemoryProjectionSync(this.db, registry)
    registry.setProjectionHooks((key, before, after) => this.projectionSync!.reconcileFile(key, before, after), () => this.projectionSync!.flush())
    await this.projectionSync.flush()
  }

  readonly db: SqliteDatabase
  private closed = false
  private readonly embedder?: MemoryEmbedder
  private readonly embeddingModel?: string
  private readonly configuredProviderId?: string
  private readonly vectorBackend: MemoryVectorBackend
  private activeDimensions?: number
  private backfillPromise?: Promise<number>

  constructor(dbPath: string, options: SqliteSemanticIndexOptions = {}) {
    this.db = openDatabase(dbPath)
    this.db.pragma('journal_mode = WAL')
    this.embedder = options.embedder
    this.embeddingModel = options.embeddingModel
    this.configuredProviderId = options.configuredProviderId ?? options.embedder?.providerId
    this.vectorBackend = createVectorBackend({
      db: this.db,
      dbPath,
      requested: options.vectorBackend,
      qdrant: options.qdrant,
      opensearch: options.opensearch,
      elasticsearch: options.elasticsearch,
      meilisearch: options.meilisearch,
      customApi: options.customApi,
    })

    runMigrations(this.db)
  }

  static async create(
    dbPath: string,
    options: SqliteSemanticIndexOptions = {},
  ): Promise<SqliteSemanticIndex> {
    await mkdir(dirname(dbPath), { recursive: true })
    recoverSqliteDatabaseIfCorrupt(dbPath)
    const index = new SqliteSemanticIndex(dbPath, options)
    await index.bootstrap()
    return index
  }

  async getRetractedSourceIds(scopeTags: string[]): Promise<string[]> {
    const rows = this.db.prepare('SELECT tags, source_ids FROM memory_retractions').all() as Array<{ tags: string; source_ids: string }>
    return [...new Set(rows.filter((row) => canonicalFileMemoryScopeKey(JSON.parse(row.tags)) === canonicalFileMemoryScopeKey(scopeTags))
      .flatMap((row) => JSON.parse(row.source_ids) as string[]))]
  }

  async add(
    entry: Omit<MemoryEntry, 'score'>,
    options: { sessionId?: string } = {},
  ): Promise<void> {
    const id = entry.id || randomUUID()
    const previous = await this.get(id)
    const evidence = evidenceForWrite(entry, previous, options.sessionId)
    const entryOwner = memoryResetOwner(entry.tags)
    const reset = this.db.prepare('SELECT MAX(reset_at) AS reset_at FROM memory_resets WHERE owner = ? OR owner = ?').get(entryOwner, entryOwner.startsWith('scope:session:') ? 'global' : entryOwner) as { reset_at: string | null } | undefined
    if (evidence.origin !== 'user' && reset?.reset_at && Date.parse(evidence.observedAt) <= Date.parse(reset.reset_at)) {
      throw new Error('Automatic memory predates the owner memory reset')
    }
    if (!previous && evidence.origin !== 'user') {
      const retired = this.db.prepare(`SELECT memory_id, content, tags FROM memory_retractions
        WHERE memory_id = @id OR content = @content OR memory_id IN (SELECT value FROM json_each(@sources))
          OR EXISTS (SELECT 1 FROM json_each(source_ids) oldSource WHERE oldSource.value IN (SELECT value FROM json_each(@sources)))`)
        .all({ id, content: entry.content, sources: JSON.stringify(evidence.sourceIds) }) as Array<{ memory_id: string; content: string; tags: string }>
      if (retired.some((row) => canonicalFileMemoryScopeKey(JSON.parse(row.tags)) === canonicalFileMemoryScopeKey(entry.tags))) {
        throw new Error('Automatic memory would restore retracted evidence; explicit user correction is required')
      }
    }
    const tags = JSON.stringify(entry.tags ?? [])
    const initialEmbeddingState = this.isSemanticConfigured() ? 'pending' : 'disabled'

    this.db.prepare(`
      INSERT INTO memories (
        id,
        content,
        source,
        tags,
        session_id,
        document_id,
        chunk_index,
        chunk_count,
        chunk_start,
        chunk_end,
        chunk_title,
        metadata,
        updated_at,
        embedding_state,
        embedding_error
      )
      VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, datetime('now'), ?, NULL)
      ON CONFLICT(id) DO UPDATE SET
        content = excluded.content,
        source = excluded.source,
        tags = excluded.tags,
        session_id = excluded.session_id,
        document_id = excluded.document_id,
        chunk_index = excluded.chunk_index,
        chunk_count = excluded.chunk_count,
        chunk_start = excluded.chunk_start,
        chunk_end = excluded.chunk_end,
        chunk_title = excluded.chunk_title,
        metadata = excluded.metadata,
        updated_at = excluded.updated_at,
        embedding_state = excluded.embedding_state,
        embedding_error = NULL,
        embedding = NULL, embedding_model = NULL, embedding_updated_at = NULL
    `).run(
      id,
      entry.content,
      entry.source,
      tags,
      options.sessionId ?? null,
      JSON.stringify({ memory: evidence }),
      initialEmbeddingState,
    )

    retractDerivedMemories(this.db)
    await this.projectionSync?.flush()

    if (!this.canWriteSemanticVectors()) {
      this.persistDerivedStatus()
      return
    }

    const row = this.db.prepare(`
      SELECT rowid, id
      FROM memories
      WHERE id = ?
    `).get(id) as { rowid: number; id: string } | undefined
    if (!row) return

    try {
      const embeddings = await this.embedTexts([entry.content])
      const embedding = embeddings[0]
      this.validateEmbedding(embedding)
      await this.ensureVectorBackendReady(embedding.length)
      await this.writeEmbedding(row.rowid, row.id, embedding)
      this.persistDerivedStatus(null)
    } catch (error) {
      this.markRowsFailed([row.id], this.toErrorMessage(error))
      this.persistDerivedStatus(this.toErrorMessage(error))
    }
  }

  async get(id: string): Promise<MemoryEntry | null> {
    const row = this.db.prepare(`
      SELECT id, content, source, tags
      FROM memories
      WHERE id = ?
    `).get(id) as {
      id: string
      content: string
      source: MemoryEntry['source']
      tags: string | null
    } | undefined

    if (!row) {
      return null
    }

    return this.withEvidence([{
      id: row.id,
      content: row.content,
      source: row.source,
      tags: this.parseTags(row.tags),
    }])[0]
  }

  private withEvidence<T extends MemoryEntry>(entries: T[]): T[] {
    if (!entries.length) return entries
    const rows = this.db.prepare(`SELECT id, metadata, created_at, updated_at FROM memories
      WHERE id IN (SELECT value FROM json_each(?))`).all(JSON.stringify(entries.map((entry) => entry.id))) as Array<{
        id: string; metadata: string | null; created_at: string; updated_at: string
      }>
    const byId = new Map(rows.map((row) => [row.id, row]))
    return entries.map((entry) => {
      const row = byId.get(entry.id)
      if (!row) return entry
      let evidence: MemoryEntry['evidence']
      try {
        const parsed = memoryEvidenceSchema.safeParse(JSON.parse(row.metadata ?? '{}').memory)
        if (parsed.success) evidence = parsed.data
      } catch { /* Legacy document metadata has no memory evidence. */ }
      return { ...entry, ...(evidence ? { evidence } : {}), createdAt: row.created_at, updatedAt: row.updated_at }
    })
  }

  async ingestDocument(input: DocumentIngestInput): Promise<MemoryDocument> {
    const title = input.title.trim()
    const content = input.content.trim()
    if (!title) throw new Error('Document title is required')
    if (!content) throw new Error('Document content is required')

    const tags = JSON.stringify(input.tags ?? [])
    // Content-hash dedup: re-ingesting the same title/tags/content must not
    // multiply rows. When the caller does not pin an explicit id, reuse the id
    // of an identical existing document so the upsert updates it in place.
    const contentHash = createHash('sha256')
      .update(`${title}\n${tags}\n${content}`)
      .digest('hex')
    let id = input.id?.trim()
    if (!id) {
      const existing = this.db.prepare(
        'SELECT id FROM memory_documents WHERE content_hash = ? LIMIT 1',
      ).get(contentHash) as { id: string } | undefined
      id = existing?.id ?? randomUUID()
    }
    const chunks = chunkDocumentForRag({
      title,
      content,
      mimeType: input.mimeType,
    })
    if (chunks.length > MAX_DOCUMENT_CHUNKS) {
      throw new Error(
        `Document produces too many chunks: ${chunks.length} (max ${MAX_DOCUMENT_CHUNKS})`,
      )
    }
    const initialEmbeddingState = this.isSemanticConfigured() ? 'pending' : 'disabled'

    const replaceDocument = this.db.transaction(() => {
      const existingRowids = this.db.prepare(`
        SELECT rowid
        FROM memories
        WHERE document_id = ?
      `).all(id) as Array<{ rowid: number }>

      this.db.prepare(`
        INSERT INTO memory_documents (
          id,
          title,
          path,
          mime_type,
          source_file_id,
          tags,
          chunk_count,
          content,
          content_hash,
          updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          title = excluded.title,
          path = excluded.path,
          mime_type = excluded.mime_type,
          source_file_id = excluded.source_file_id,
          tags = excluded.tags,
          chunk_count = excluded.chunk_count,
          content = excluded.content,
          content_hash = excluded.content_hash,
          updated_at = excluded.updated_at
      `).run(
        id,
        title,
        input.path ?? null,
        input.mimeType ?? null,
        input.sourceFileId ?? null,
        tags,
        chunks.length,
        content,
        contentHash,
      )

      this.db.prepare('DELETE FROM memories WHERE document_id = ?').run(id)

      const insertChunk = this.db.prepare(`
        INSERT INTO memories (
          id,
          content,
          source,
          tags,
          document_id,
          chunk_index,
          chunk_count,
          chunk_start,
          chunk_end,
          chunk_title,
          metadata,
          updated_at,
          embedding_state,
          embedding_error
        )
        VALUES (?, ?, 'document', ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, NULL)
      `)

      for (const chunk of chunks) {
        insertChunk.run(
          `${id}::chunk::${chunk.chunkIndex}`,
          chunk.content,
          tags,
          id,
          chunk.chunkIndex,
          chunk.chunkCount,
          chunk.startOffset,
          chunk.endOffset,
          chunk.chunkTitle ?? null,
          JSON.stringify({
            documentId: id,
            title,
            path: input.path,
            mimeType: input.mimeType,
            sourceFileId: input.sourceFileId,
            chunkIndex: chunk.chunkIndex,
            chunkCount: chunk.chunkCount,
            startOffset: chunk.startOffset,
            endOffset: chunk.endOffset,
            chunkTitle: chunk.chunkTitle,
          }),
          initialEmbeddingState,
        )
      }

      return existingRowids.map((row) => row.rowid)
    })

    const replacedRowids = replaceDocument()
    await this.deleteVectorRows(replacedRowids)

    if (chunks.length === 0 || !this.canWriteSemanticVectors()) {
      this.persistDerivedStatus()
      return (await this.getDocument(id))!
    }

    const rows = this.db.prepare(`
      SELECT rowid, id, content
      FROM memories
      WHERE document_id = ?
      ORDER BY chunk_index ASC
    `).all(id) as MemoryRow[]

    const lastEmbeddingError = await this.embedDocumentRowsInBatches(
      rows,
      DOCUMENT_EMBEDDING_BATCH_SIZE,
    )
    this.persistDerivedStatus(lastEmbeddingError)

    return (await this.getDocument(id))!
  }

  async getDocument(id: string): Promise<MemoryDocument | null> {
    const row = this.db.prepare(`
      SELECT id, title, path, mime_type, source_file_id, tags, chunk_count, created_at, updated_at
      FROM memory_documents
      WHERE id = ?
    `).get(id) as DocumentRow | undefined

    if (!row) return null
    return this.mapDocumentRow(row)
  }

  async listDocuments(options?: DocumentListOptions): Promise<MemoryDocument[]> {
    const limit = Math.max(1, options?.limit ?? 50)
    const query = options?.query?.trim()

    if (!query) {
      const rows = this.db.prepare(`
        SELECT id, title, path, mime_type, source_file_id, tags, chunk_count, created_at, updated_at
        FROM memory_documents
        ORDER BY updated_at DESC, created_at DESC
        LIMIT ?
      `).all(limit) as DocumentRow[]
      return rows.map((row) => this.mapDocumentRow(row))
    }

    const pattern = `%${query.replaceAll('\\', '\\\\').replaceAll('%', '\\%').replaceAll('_', '\\_')}%`
    const rows = this.db.prepare(`
      SELECT id, title, path, mime_type, source_file_id, tags, chunk_count, created_at, updated_at
      FROM memory_documents
      WHERE title LIKE @pattern ESCAPE '\\'
         OR COALESCE(path, '') LIKE @pattern ESCAPE '\\'
         OR COALESCE(tags, '') LIKE @pattern ESCAPE '\\'
      ORDER BY updated_at DESC, created_at DESC
      LIMIT @limit
    `).all({ pattern, limit }) as DocumentRow[]
    return rows.map((row) => this.mapDocumentRow(row))
  }

  async searchDocuments(
    query: string,
    options?: DocumentSearchOptions,
  ): Promise<MemoryDocumentChunk[]> {
    const results = await this.search(query, {
      ...options,
      sources: ['document'],
      documentId: options?.documentId,
    })
    if (results.length === 0) return []

    const ids = results.map((result) => result.id)
    const params: Record<string, unknown> = {}
    const placeholders = ids.map((id, index) => {
      const key = `id_${index}`
      params[key] = id
      return `@${key}`
    })

    const rows = this.db.prepare(`
      SELECT
        m.id,
        m.content,
        m.tags,
        m.document_id,
        m.chunk_index,
        m.chunk_count,
        m.chunk_start,
        m.chunk_end,
        m.chunk_title,
        d.title AS document_title,
        d.path,
        d.mime_type,
        d.source_file_id
      FROM memories m
      JOIN memory_documents d ON d.id = m.document_id
      WHERE m.id IN (${placeholders.join(', ')})
    `).all(params) as DocumentChunkRow[]

    const byId = new Map(rows.map((row) => [row.id, row]))
    return results.flatMap((result) => {
      const row = byId.get(result.id)
      if (!row) return []
      return [this.mapDocumentChunkRow(row, result.score, query)]
    })
  }

  async listDocumentChunks(
    documentId: string,
    limit: number = 5,
  ): Promise<MemoryDocumentChunk[]> {
    const safeLimit = Math.max(1, Math.min(Math.floor(limit), 100))
    const rows = this.db.prepare(`
      SELECT
        m.id,
        m.content,
        m.tags,
        m.document_id,
        m.chunk_index,
        m.chunk_count,
        m.chunk_start,
        m.chunk_end,
        m.chunk_title,
        d.title AS document_title,
        d.path,
        d.mime_type,
        d.source_file_id
      FROM memories m
      JOIN memory_documents d ON d.id = m.document_id
      WHERE m.document_id = ?
      ORDER BY m.chunk_index ASC
      LIMIT ?
    `).all(documentId, safeLimit) as DocumentChunkRow[]
    return rows.map((row) => this.mapDocumentChunkRow(row))
  }

  async updateDocument(
    id: string,
    input: DocumentUpdateInput,
  ): Promise<MemoryDocument | null> {
    const existing = await this.getDocument(id)
    if (!existing) return null

    const sets: string[] = []
    const params: Record<string, unknown> = { id }
    if (typeof input.title === 'string' && input.title.trim().length > 0) {
      sets.push('title = @title')
      params.title = input.title.trim()
    }
    if (input.path !== undefined) {
      sets.push('path = @path')
      params.path = input.path === '' ? null : input.path
    }
    if (input.mimeType !== undefined) {
      sets.push('mime_type = @mimeType')
      params.mimeType = input.mimeType === '' ? null : input.mimeType
    }
    if (input.tags !== undefined) {
      sets.push('tags = @tags')
      params.tags = JSON.stringify(input.tags)
    }
    sets.push("updated_at = datetime('now')")

    if (sets.length === 1) {
      // only updated_at — nothing changed, treat as no-op.
      return existing
    }

    this.db.prepare(`
      UPDATE memory_documents
      SET ${sets.join(', ')}
      WHERE id = @id
    `).run(params)

    // Propagate the new tags / title into the chunk-level memories rows so
    // memory.documents.search and memory.search keep returning consistent
    // metadata.
    if (input.tags !== undefined) {
      this.db.prepare(`
        UPDATE memories
        SET tags = @tags
        WHERE document_id = @id
      `).run({ id, tags: params.tags })
    }

    return this.getDocument(id)
  }

  async deleteDocument(id: string): Promise<boolean> {
    const tx = this.db.transaction((documentId: string) => {
      const rowids = this.db.prepare(`
        SELECT rowid
        FROM memories
        WHERE document_id = ?
      `).all(documentId) as Array<{ rowid: number }>
      const existed = this.db.prepare(`
        SELECT 1 AS found
        FROM memory_documents
        WHERE id = ?
      `).get(documentId) as { found: number } | undefined

      this.db.prepare('DELETE FROM memories WHERE document_id = ?').run(documentId)
      this.db.prepare('DELETE FROM memory_documents WHERE id = ?').run(documentId)
      return {
        existed: Boolean(existed),
        rowids: rowids.map((row) => row.rowid),
      }
    })

    const deleted = tx(id)
    if (deleted.existed) {
      await this.deleteVectorRows(deleted.rowids)
      this.persistDerivedStatus()
    }
    return deleted.existed
  }

  async search(query: string, options?: SemanticSearchOptions): Promise<MemoryEntry[]> {
    retractDerivedMemories(this.db)
    const trimmedQuery = query.trim()
    if (!trimmedQuery) return []

    const limit = options?.limit ?? 10
    const type = options?.type ?? (this.isSemanticQueryable() ? 'hybrid' : 'keyword')

    // When a tag filter is active it is applied in JS after the backend LIMIT,
    // so fetching exactly `limit` candidates under-returns whenever any are
    // filtered out. Over-fetch a wider candidate window, filter, then trim to
    // the requested limit.
    const tagFilterActive = Boolean(options?.tags?.length || options?.excludeTags?.length)
    const fetchLimit = tagFilterActive ? Math.min(Math.max(limit * 4, limit), 500) : limit

    let raw: MemoryEntry[]
    if (type === 'keyword') {
      raw = this.applyTagFilter(this.keywordSearch(trimmedQuery, fetchLimit, options), options)
    } else if (type === 'semantic') {
      raw = this.applyTagFilter(await this.semanticSearch(trimmedQuery, fetchLimit, options), options)
    } else {
      const [semantic, keyword] = await Promise.all([
        this.semanticSearch(trimmedQuery, fetchLimit, options),
        Promise.resolve(this.keywordSearch(trimmedQuery, fetchLimit, options)),
      ])
      raw = this.applyTagFilter(this.mergeHybridResults(semantic, keyword, fetchLimit), options)
    }

    // Nothing matched at all. Rather than answer "no memories" for something
    // that is stored, retry the keyword pass with its looser forms.
    //
    // Korean glues particles onto the stem, so a memory saved as `주 개발 언어는
    // TypeScript` indexes the token `언어는` and a search for `언어` matched
    // nothing — a memory the user had saved a moment earlier came back as zero
    // hits. Recall is also asked in whole sentences ("주력 언어 프로그래밍",
    // "내 노트북 OS") that no single memory contains every word of. Against real
    // stored data those queries go from 0 hits to 8 and 9 this way.
    //
    // Only on an empty result, so every query that already returned something
    // keeps returning exactly what it did: the widened forms trade precision
    // for recall, and that trade is only worth making against nothing.
    // Not for `type: 'semantic'`, which promises vector results only and must
    // stay empty rather than fall back to keyword matching.
    if (raw.length === 0 && type !== 'semantic') {
      raw = this.applyTagFilter(
        this.keywordSearch(trimmedQuery, fetchLimit, { ...options, widenWhenEmpty: true }),
        options,
      )
    }

    // Optional chronological re-sort: relevance backends rank by score, but
    // operators sometimes want newest-first / oldest-first (e.g. "지난주에
    // 결정한 것"). Pull created_at / updated_at for the candidate ids in
    // one batch so we don't N+1 the SQLite layer.
    if (options?.sortBy === 'createdAt' || options?.sortBy === 'updatedAt') {
      const column = options.sortBy === 'updatedAt' ? 'updated_at' : 'created_at'
      const ids = raw.map((entry) => entry.id)
      if (ids.length > 0) {
        const placeholders = ids.map((_, i) => `@id_${i}`).join(', ')
        const params: Record<string, unknown> = {}
        ids.forEach((id, i) => { params[`id_${i}`] = id })
        const rows = this.db.prepare(`
          SELECT id, ${column} AS ts
          FROM memories
          WHERE id IN (${placeholders})
        `).all(params) as Array<{ id: string; ts: string }>
        const tsById = new Map(rows.map((row) => [row.id, row.ts]))
        raw = [...raw].sort((a, b) => {
          const av = Date.parse(tsById.get(a.id) ?? '') || 0
          const bv = Date.parse(tsById.get(b.id) ?? '') || 0
          return bv - av
        })
      }
    }

    return this.withEvidence(raw.slice(0, limit))
  }

  async getMemoryResetAt(scopeTags: string[]): Promise<string | null> {
    const owner = memoryResetOwner(scopeTags)
    const row = this.db.prepare('SELECT MAX(reset_at) AS reset_at FROM memory_resets WHERE owner = ? OR owner = ?').get(owner, owner.startsWith('scope:session:') ? 'global' : owner) as { reset_at: string | null } | undefined
    return row?.reset_at ?? null
  }

  private ownedMemoryRows(scopeTags: string[]) {
    const owner = memoryResetOwner(scopeTags)
    const owns = (raw: string | null): boolean => {
      const tags: unknown = JSON.parse(raw ?? '[]')
      if (!Array.isArray(tags) || !tags.every((tag) => typeof tag === 'string')) throw new Error('Invalid memory ownership metadata')
      const entryOwner = memoryResetOwner(tags)
      return !tags.some((tag) => tag.toLowerCase() === 'scope:public') && (entryOwner === owner || (owner === 'global' && entryOwner.startsWith('scope:session:')))
    }
    // Enumerate authoritative storage without a search/list top-k ceiling. Fail
    // before deleting anything when ownership metadata cannot be interpreted.
    const rows = (this.db.prepare('SELECT id, rowid, tags FROM memories').all() as Array<{ id: string; rowid: number; tags: string }>).filter((row) => owns(row.tags))
    const documents = (this.db.prepare('SELECT id, tags FROM memory_documents').all() as Array<{ id: string; tags: string }>).filter((row) => owns(row.tags))
    return { owner, rows, documents }
  }

  async previewForgetOwnedMemory(scopeTags: string[]) {
    const { rows, documents } = this.ownedMemoryRows(scopeTags)
    return { memories: rows.length, documents: documents.length }
  }

  async forgetOwnedMemory(scopeTags: string[]): Promise<{ memories: number; documents: number; resetAt: string }> {
    const { owner, rows, documents } = this.ownedMemoryRows(scopeTags)
    // External deletion must succeed before authoritative rows disappear, so a
    // failed backend cleanup remains retryable and cannot be reported as success.
    await this.vectorBackend.delete(rows.map((row) => row.rowid))
    const resetAt = new Date().toISOString()
    this.db.transaction(() => {
      this.db.prepare('INSERT OR REPLACE INTO memory_resets(owner, reset_at) VALUES (?, ?)').run(owner, resetAt)
      for (const row of rows) this.db.prepare('DELETE FROM memories WHERE id = ?').run(row.id)
      for (const row of documents) this.db.prepare('DELETE FROM memory_documents WHERE id = ?').run(row.id)
    })()
    retractDerivedMemories(this.db)
    await this.projectionSync?.flush()
    this.persistDerivedStatus()
    return { memories: rows.length, documents: documents.length, resetAt }
  }

  async delete(id: string): Promise<void> {
    const row = this.db.prepare(`
      SELECT rowid
      FROM memories
      WHERE id = ?
    `).get(id) as { rowid: number } | undefined
    this.db.prepare('DELETE FROM memories WHERE id = ?').run(id)
    retractDerivedMemories(this.db)
    await this.projectionSync?.flush()
    await this.deleteVectorRows(row ? [row.rowid] : [])
    this.persistDerivedStatus()
  }

  async deleteBySession(sessionId: string): Promise<number> {
    if (!sessionId) return 0
    const rows = this.db.prepare(`
      SELECT rowid
      FROM memories
      WHERE session_id = ?
    `).all(sessionId) as Array<{ rowid: number }>
    if (rows.length === 0) return 0
    this.db.prepare('DELETE FROM memories WHERE session_id = ?').run(sessionId)
    await this.deleteVectorRows(rows.map((row) => row.rowid))
    this.persistDerivedStatus()
    return rows.length
  }

  async deleteOlderThan(cutoffIso: string): Promise<number> {
    if (!cutoffIso) return 0
    const rows = this.db.prepare(`
      SELECT rowid
      FROM memories
      WHERE updated_at < ?
    `).all(cutoffIso) as Array<{ rowid: number }>
    if (rows.length === 0) return 0
    this.db.prepare('DELETE FROM memories WHERE updated_at < ?').run(cutoffIso)
    await this.deleteVectorRows(rows.map((row) => row.rowid))
    this.persistDerivedStatus()
    return rows.length
  }

  async mergeMemories(input: MergeMemoriesInput): Promise<MergeMemoriesResult> {
    const embeddingState = this.isSemanticConfigured() ? 'pending' : 'disabled'
    // The content update + duplicate deletes run inside one better-sqlite3
    // transaction so a crash cannot leave the kept memory deleted with the
    // merged row unwritten. Vector rows are cleaned up after the transaction
    // commits (mirrors delete()/deleteDocument()), which is best-effort by
    // design — the durable metadata is already consistent.
    const tx = this.db.transaction(() => {
      const kept = this.db.prepare(
        'SELECT rowid FROM memories WHERE id = ?',
      ).get(input.keepId) as { rowid: number } | undefined
      if (!kept) {
        return { merged: false, removed: [], skippedPinned: [], staleRowids: [] as number[] }
      }

      const removed: string[] = []
      const skippedPinned: string[] = []
      const staleRowids: number[] = [kept.rowid]
      for (const removeId of input.removeIds) {
        if (removeId === input.keepId) continue
        const row = this.db.prepare(
          'SELECT rowid, pinned FROM memories WHERE id = ?',
        ).get(removeId) as { rowid: number; pinned: number | null } | undefined
        if (!row) continue
        // Never hard-delete a pinned duplicate — the user asked to keep it.
        if (row.pinned) {
          skippedPinned.push(removeId)
          continue
        }
        this.db.prepare('DELETE FROM memories WHERE id = ?').run(removeId)
        removed.push(removeId)
        staleRowids.push(row.rowid)
      }

      // Update ONLY content/tags/updated_at and invalidate the stale embedding.
      // importance, base_importance, access_count, pinned, and created_at are
      // deliberately left untouched so the merge preserves the kept memory's
      // accumulated signal (the old delete+re-add reset all of these).
      if (input.tags !== undefined) {
        this.db.prepare(`
          UPDATE memories SET
            content = @content,
            tags = @tags,
            updated_at = datetime('now'),
            embedding = NULL,
            embedding_model = NULL,
            embedding_error = NULL,
            embedding_updated_at = NULL,
            embedding_state = @embeddingState
          WHERE id = @id
        `).run({
          id: input.keepId,
          content: input.content,
          tags: JSON.stringify(input.tags),
          embeddingState,
        })
      } else {
        this.db.prepare(`
          UPDATE memories SET
            content = @content,
            updated_at = datetime('now'),
            embedding = NULL,
            embedding_model = NULL,
            embedding_error = NULL,
            embedding_updated_at = NULL,
            embedding_state = @embeddingState
          WHERE id = @id
        `).run({ id: input.keepId, content: input.content, embeddingState })
      }

      return { merged: true, removed, skippedPinned, staleRowids }
    })

    const outcome = tx()
    // Drop vector rows for both the deleted duplicates and the kept memory
    // (its embedding no longer matches the merged content; backfill re-embeds).
    await this.deleteVectorRows(outcome.staleRowids)
    this.persistDerivedStatus()
    return {
      merged: outcome.merged,
      removed: outcome.removed,
      skippedPinned: outcome.skippedPinned,
    }
  }

  async listForScope(scopeTags: string[], options: { limit?: number; offset?: number; includeInactive?: boolean } = {}) {
    retractDerivedMemories(this.db)
    const limit = Math.max(1, Math.min(100, Math.floor(options.limit ?? 20)))
    const offset = Math.max(0, Math.floor(options.offset ?? 0))
    const filter = buildMemorySearchFilter('m', { scopeTags, includeInactive: options.includeInactive,
      excludeTags: options.includeInactive ? undefined : ['archived', 'superseded'] })
    const rows = this.db.prepare(`SELECT m.id, m.content, m.source, m.tags FROM memories m
      WHERE 1=1 ${filter.clause} ORDER BY m.created_at DESC, m.id ASC LIMIT @pageLimit OFFSET @offset`)
      .all({ ...filter.params, pageLimit: limit + 1, offset }) as Array<{ id: string; content: string; source: MemoryEntry['source']; tags: string | null }>
    return { memories: this.withEvidence(rows.slice(0, limit).map((row) => ({ ...row, tags: this.parseTags(row.tags) }))),
      nextOffset: rows.length > limit ? offset + limit : null }
  }

  async listRecent(limit: number): Promise<Array<Omit<MemoryEntry, 'score'>>> {
    const rows = this.db.prepare(`
      SELECT id, content, source, tags
      FROM memories
      ORDER BY created_at DESC
      LIMIT ?
    `).all(limit) as Array<{
      id: string
      content: string
      source: MemoryEntry['source']
      tags: string | null
    }>

    return this.withEvidence(rows.map((row) => ({
      id: row.id,
      content: row.content,
      source: row.source,
      tags: this.parseTags(row.tags),
    })))
  }

  /** Like listRecent but each row also carries created_at / updated_at so
   *  callers (memory.export) can filter by time without a second query.
   *  Optional createdAfter / createdBefore are applied at the SQL layer. */
  async listRecentWithTimestamps(
    limit: number,
    options: { createdAfter?: string; createdBefore?: string } = {},
  ): Promise<Array<Omit<MemoryEntry, 'score'> & { createdAt: string; updatedAt: string }>> {
    const where: string[] = []
    const params: Record<string, unknown> = { limit }
    if (options.createdAfter) {
      where.push('created_at >= @createdAfter')
      params.createdAfter = options.createdAfter
    }
    if (options.createdBefore) {
      where.push('created_at <= @createdBefore')
      params.createdBefore = options.createdBefore
    }
    const clause = where.length > 0 ? `WHERE ${where.join(' AND ')}` : ''
    const rows = this.db.prepare(`
      SELECT id, content, source, tags, created_at, updated_at
      FROM memories
      ${clause}
      ORDER BY created_at DESC
      LIMIT @limit
    `).all(params) as Array<{
      id: string
      content: string
      source: MemoryEntry['source']
      tags: string | null
      created_at: string
      updated_at: string
    }>
    return this.withEvidence(rows.map((row) => ({
      id: row.id,
      content: row.content,
      source: row.source,
      tags: this.parseTags(row.tags),
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    })))
  }

  /**
   * Bump access_count + last_accessed_at for the given ids. Best-effort:
   * unknown ids are silently ignored, no throw, no audit (counter
   * mutations are far too high-volume to log every hit). Duplicate ids
   * within a single call each count — surfacing the same memory twice
   * IS two accesses.
   */
  async recordAccess(ids: string[]): Promise<void> {
    const counts = new Map<string, number>()
    for (const raw of ids) {
      const id = typeof raw === 'string' ? raw.trim() : ''
      if (!id) continue
      counts.set(id, (counts.get(id) ?? 0) + 1)
    }
    if (counts.size === 0) return
    const stmt = this.db.prepare(`
      UPDATE memories
      SET access_count = access_count + ?,
          last_accessed_at = datetime('now')
      WHERE id = ?
    `)
    const tx = this.db.transaction((entries: Array<[string, number]>) => {
      for (const [id, delta] of entries) {
        stmt.run(delta, id)
      }
    })
    tx([...counts.entries()])
  }

  async getAccessStats(id: string): Promise<MemoryAccessStats | null> {
    const row = this.db.prepare(`
      SELECT access_count, last_accessed_at
      FROM memories
      WHERE id = ?
    `).get(id) as { access_count: number | null; last_accessed_at: string | null } | undefined
    if (!row) return null
    return {
      accessCount: row.access_count ?? 0,
      lastAccessedAt: row.last_accessed_at ?? null,
    }
  }

  async listHotMemories(limit: number): Promise<MemoryAccessHotEntry[]> {
    const safeLimit = Math.max(1, Math.min(500, Math.floor(limit) || 10))
    const rows = this.db.prepare(`
      SELECT id, content, source, tags, access_count, last_accessed_at
      FROM memories
      WHERE access_count > 0
      ORDER BY access_count DESC, last_accessed_at DESC
      LIMIT ?
    `).all(safeLimit) as Array<{
      id: string
      content: string
      source: MemoryEntry['source']
      tags: string | null
      access_count: number | null
      last_accessed_at: string | null
    }>
    return rows.map((row) => ({
      id: row.id,
      content: row.content,
      source: row.source,
      tags: this.parseTags(row.tags),
      accessCount: row.access_count ?? 0,
      lastAccessedAt: row.last_accessed_at ?? null,
    }))
  }

  /**
   * Mark a memory as pin so the dreaming prune phase skips it. The
   * column doubles as a recency timestamp (`last_accessed_at` is bumped
   * on pin so a freshly-pinned item is also "warm" for ranking). Pin
   * is idempotent — re-pinning is a no-op except for the timestamp
   * refresh, which lets users revive a memory's freshness without
   * actually accessing it.
   */
  async pin(id: string): Promise<void> {
    const trimmed = id?.trim()
    if (!trimmed) return
    this.db.prepare(`
      UPDATE memories
      SET pinned = 1,
          last_accessed_at = datetime('now')
      WHERE id = ?
    `).run(trimmed)
  }

  async unpin(id: string): Promise<void> {
    const trimmed = id?.trim()
    if (!trimmed) return
    this.db.prepare(`
      UPDATE memories
      SET pinned = 0
      WHERE id = ?
    `).run(trimmed)
  }

  async isPinned(id: string): Promise<boolean | null> {
    const row = this.db.prepare(`
      SELECT pinned FROM memories WHERE id = ?
    `).get(id) as { pinned: number | null } | undefined
    if (!row) return null
    return Boolean(row.pinned)
  }

  async listPinned(limit: number): Promise<MemoryPinnedEntry[]> {
    const safeLimit = Math.max(1, Math.min(500, Math.floor(limit) || 50))
    const rows = this.db.prepare(`
      SELECT id, content, source, tags, last_accessed_at
      FROM memories
      WHERE pinned = 1
      ORDER BY last_accessed_at DESC, id ASC
      LIMIT ?
    `).all(safeLimit) as Array<{
      id: string
      content: string
      source: MemoryEntry['source']
      tags: string | null
      last_accessed_at: string | null
    }>
    return rows.map((row) => ({
      id: row.id,
      content: row.content,
      source: row.source,
      tags: this.parseTags(row.tags),
      pinnedAt: row.last_accessed_at ?? null,
    }))
  }

  async recordAudit(input: MemoryAuditRecordInput): Promise<MemoryAuditEntry> {
    return this.recordAuditSync(input)
  }

  async listAudit(options?: MemoryAuditListOptions): Promise<MemoryAuditEntry[]> {
    const limit = Math.min(Math.max(1, options?.limit ?? 50), 500)
    const memoryId = options?.memoryId?.trim()

    if (memoryId) {
      const rows = this.db.prepare(`
        SELECT id, memory_id, action, actor, reason, before_json, after_json, created_at
        FROM memory_audit
        WHERE memory_id = ?
        ORDER BY created_at DESC, rowid DESC
        LIMIT ?
      `).all(memoryId, limit) as MemoryAuditRow[]
      return rows.map((row) => this.mapAuditRow(row))
    }

    const rows = this.db.prepare(`
      SELECT id, memory_id, action, actor, reason, before_json, after_json, created_at
      FROM memory_audit
      ORDER BY created_at DESC, rowid DESC
      LIMIT ?
    `).all(limit) as MemoryAuditRow[]
    return rows.map((row) => this.mapAuditRow(row))
  }

  async getLifecycleStatus(options?: MemoryLifecycleOptions): Promise<MemoryLifecycleStatus> {
    const staleAfterDays = options?.staleAfterDays ?? 90
    const lowImportance = options?.lowImportance ?? 0.1
    const row = this.db.prepare(`
      SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN source = 'conversation' THEN 1 ELSE 0 END) AS conversation,
        SUM(CASE WHEN source = 'document' THEN 1 ELSE 0 END) AS document,
        SUM(CASE WHEN source = 'skill' THEN 1 ELSE 0 END) AS skill,
        SUM(CASE WHEN source = 'user' THEN 1 ELSE 0 END) AS user,
        SUM(CASE
          WHEN source = 'conversation'
           AND julianday('now') - julianday(created_at) > @staleAfterDays
          THEN 1 ELSE 0 END) AS staleConversation,
        SUM(CASE
          WHEN source = 'conversation'
           AND (importance IS NULL OR importance < @lowImportance)
          THEN 1 ELSE 0 END) AS lowImportanceConversation,
        SUM(CASE
          WHEN source = 'conversation'
           AND julianday('now') - julianday(created_at) > @staleAfterDays
           AND (importance IS NULL OR importance < @lowImportance)
          THEN 1 ELSE 0 END) AS pruneCandidate
      FROM memories
    `).get({ staleAfterDays, lowImportance }) as MemoryLifecycleCountRow

    const audit = this.db.prepare(`
      SELECT MAX(created_at) AS lastAuditAt
      FROM memory_audit
    `).get() as { lastAuditAt: string | null }

    return {
      totalMemories: row.total ?? 0,
      conversationMemories: row.conversation ?? 0,
      documentMemories: row.document ?? 0,
      skillMemories: row.skill ?? 0,
      userMemories: row.user ?? 0,
      staleConversationMemories: row.staleConversation ?? 0,
      lowImportanceConversationMemories: row.lowImportanceConversation ?? 0,
      pruneCandidateMemories: row.pruneCandidate ?? 0,
      pendingEmbeddings: this.getPendingRowCount(),
      failedEmbeddings: this.getFailedRowCount(),
      lastAuditAt: audit.lastAuditAt ?? undefined,
    }
  }

  async runMaintenance(options?: MemoryMaintenanceOptions): Promise<MemoryMaintenanceResult> {
    const staleAfterDays = options?.staleAfterDays ?? 90
    const lowImportance = options?.lowImportance ?? 0.1
    const before = await this.getLifecycleStatus({ staleAfterDays, lowImportance })

    if (options?.dryRun) {
      return {
        dryRun: true,
        importanceUpdated: 0,
        pruned: 0,
        wouldPrune: before.pruneCandidateMemories,
        status: before,
      }
    }

    const actor = options?.actor ?? 'memory-maintenance'
    const reason = options?.reason ?? 'Recalculated importance and pruned stale conversation memories'
    const importanceUpdated = await this.recalculateImportanceScores()
    const pruned = await this.pruneStaleConversationMemoriesForActor({
      maxAgeDays: staleAfterDays,
      maxImportance: lowImportance,
      actor,
      reason,
    })
    const status = await this.getLifecycleStatus({ staleAfterDays, lowImportance })
    this.recordAuditSync({
      memoryId: MEMORY_MAINTENANCE_ID,
      action: 'maintenance',
      actor,
      reason,
    })

    return {
      dryRun: false,
      importanceUpdated,
      pruned,
      wouldPrune: before.pruneCandidateMemories,
      status,
    }
  }

  async recalculateImportanceScores(): Promise<number> {
    // Deterministic recompute from a stable base, NOT a running ratchet. The
    // old form did `importance = importance + age + access` every run, so a
    // 6h+1d schedule drove almost everything to 1.0 within days and importance
    // stopped discriminating (prune became a no-op). Here importance is
    // re-derived each run as clamp(base_importance + recency + access): the
    // base is fixed, created_at is fixed, and access_count only grows, so two
    // consecutive runs with the same access_count produce the same score (no
    // saturation, idempotent). base_importance is seeded from the current
    // importance the first time a row is recomputed.
    const result = this.db.prepare(`
      UPDATE memories SET
        base_importance = COALESCE(base_importance, importance, 0.5),
        importance = MAX(0.0, MIN(1.0,
          COALESCE(base_importance, importance, 0.5)
          + CASE
            WHEN julianday('now') - julianday(created_at) < 7 THEN 0.1
            WHEN julianday('now') - julianday(created_at) > 90 THEN -0.1
            ELSE 0
          END
          + CASE
            WHEN access_count >= 10 THEN 0.20
            WHEN access_count >= 3  THEN 0.10
            WHEN access_count >= 1  THEN 0.05
            ELSE 0
          END
        ))
      WHERE importance IS NOT NULL
    `).run()
    return result.changes
  }

  async pruneStaleConversationMemories(options?: {
    maxAgeDays?: number
    maxImportance?: number
  }): Promise<number> {
    return this.pruneStaleConversationMemoriesForActor({
      maxAgeDays: options?.maxAgeDays,
      maxImportance: options?.maxImportance,
      actor: 'memory-maintenance',
      reason: 'Pruned stale low-importance conversation memory',
    })
  }

  private async pruneStaleConversationMemoriesForActor(options?: {
    maxAgeDays?: number
    maxImportance?: number
    actor?: string
    reason?: string
  }): Promise<number> {
    const maxAgeDays = options?.maxAgeDays ?? 90
    const maxImportance = options?.maxImportance ?? 0.1
    const actor = options?.actor ?? 'memory-maintenance'
    const reason = options?.reason ?? 'Pruned stale low-importance conversation memory'
    // Two layers of prune protection:
    //   1. Hot-memory: access_count > 0 AND last_accessed_at within 30d
    //      → the agent keeps using it, retain.
    //   2. Pinned: row.pinned = 1 → user explicitly marked it
    //      indispensable, retain regardless of age/importance.
    const tx = this.db.transaction((ageDays: number, importance: number) => {
      const rows = this.db.prepare(`
        SELECT rowid, id, content, source, tags
        FROM memories
        WHERE julianday('now') - julianday(created_at) > ?
          AND (importance IS NULL OR importance < ?)
          AND source = 'conversation'
          AND COALESCE(pinned, 0) = 0
          AND (
            COALESCE(access_count, 0) = 0
            OR last_accessed_at IS NULL
            OR julianday('now') - julianday(last_accessed_at) > 30
          )
      `).all(ageDays, importance) as Array<{ rowid: number } & MemorySnapshotRow>
      for (const row of rows) {
        this.recordAuditSync({
          memoryId: row.id,
          action: 'pruned',
          actor,
          reason,
          before: this.mapMemorySnapshotRow(row),
        })
      }
      const result = this.db.prepare(`
        DELETE FROM memories
        WHERE julianday('now') - julianday(created_at) > ?
          AND (importance IS NULL OR importance < ?)
          AND source = 'conversation'
          AND COALESCE(pinned, 0) = 0
          AND (
            COALESCE(access_count, 0) = 0
            OR last_accessed_at IS NULL
            OR julianday('now') - julianday(last_accessed_at) > 30
          )
      `).run(ageDays, importance)
      return { changes: result.changes, rowids: rows.map((row) => row.rowid) }
    })
    const result = tx(maxAgeDays, maxImportance)
    await this.deleteVectorRows(result.rowids)
    this.persistDerivedStatus()
    return result.changes
  }

  async backfill(batchSize: number = 32): Promise<number> {
    if (this.backfillPromise) return this.backfillPromise
    if (!this.canWriteSemanticVectors()) return 0

    this.writeState({
      providerId: this.configuredProviderId ?? null,
      model: this.embeddingModel ?? null,
      dimensions: this.activeDimensions ?? undefined,
      status: 'backfilling',
      lastError: null,
    })

    const run = this.backfillInternal(batchSize)
    this.backfillPromise = run.finally(() => {
      this.backfillPromise = undefined
      this.persistDerivedStatus()
    })

    return this.backfillPromise
  }

  async reindex(batchSize: number = 32): Promise<number> {
    if (this.backfillPromise) {
      throw new Error('Semantic reindex is already in progress')
    }

    await this.prepareForReindex()
    return this.backfill(batchSize)
  }

  startBackgroundBackfill(batchSize: number = 32): void {
    if (this.backfillPromise || !this.canWriteSemanticVectors()) return
    if (this.getPendingRowCount() === 0) {
      this.persistDerivedStatus()
      return
    }

    void this.backfill(batchSize).catch(() => {
      // Backfill is best-effort.
    })
  }

  startBackgroundReindex(batchSize: number = 32): boolean {
    if (this.backfillPromise) return false
    void this.reindex(batchSize).catch((error) => {
      this.writeState({
        status: 'degraded',
        lastError: this.toErrorMessage(error),
      })
    })

    return true
  }

  getStatus(): SemanticIndexRuntimeStatus {
    const state = this.readState()
    const pendingCount = this.isSemanticConfigured() ? this.getPendingRowCount() : 0
    const failedCount = this.isSemanticConfigured() ? this.getFailedRowCount() : 0

    let status: MemorySemanticStatus
    if (!this.isSemanticConfigured()) {
      status = 'disabled'
    } else if (this.isConfiguredStateMismatch(state)) {
      status = 'reindex_required'
    } else if (!this.vectorBackend.available || !this.embedder) {
      status = 'degraded'
    } else if (this.backfillPromise) {
      status = 'backfilling'
    } else if (failedCount > 0) {
      status = 'degraded'
    } else if (pendingCount > 0) {
      status = 'backfilling'
    } else {
      status = 'ready'
    }

    return {
      status,
      configuredProviderId: this.configuredProviderId,
      configuredModel: this.embeddingModel,
      indexedProviderId: state.provider_id ?? undefined,
      indexedModel: state.model ?? undefined,
      dimensions: state.dimensions ?? this.activeDimensions,
      pendingCount,
      failedCount,
      vecAvailable: this.vectorBackend.kind === 'sqlite-vec' && this.vectorBackend.available,
      vectorBackend: this.vectorBackend.kind,
      backendAvailable: this.vectorBackend.available,
      lastError: state.last_error ?? undefined,
    }
  }

  async upsertGraph(input: MemoryGraphUpsertInput): Promise<{ nodesUpserted: number; edgesUpserted: number }> {
    const upsert = this.db.transaction(() => {
      const nodesTouched = new Set<string>()
      const edgesTouched = new Set<string>()

      const ensureNode = (node: MemoryGraphNodeInput): MemoryGraphNode | null => {
        const label = this.normalizeGraphLabel(node.label)
        if (!label) return null
        const kind = this.normalizeGraphNodeKind(node.kind)
        const id = this.memoryGraphNodeId(kind, label)
        const incomingAliases = this.normalizeGraphStringList(node.aliases, 24, 120)
        const incomingTags = this.normalizeGraphStringList(node.tags, 48, 80)
        const incomingEvidence = this.normalizeGraphStringList(node.evidenceMemoryIds, 100, 160)
        const incomingConfidence = this.clampGraphConfidence(node.confidence)
        const existing = this.db.prepare(`
          SELECT id, label, kind, aliases, tags, evidence_memory_ids, confidence, created_at, updated_at, last_seen_at
          FROM memory_graph_nodes
          WHERE id = ?
        `).get(id) as MemoryGraphNodeRow | undefined

        if (existing) {
          const aliases = this.mergeGraphStringLists(
            this.parseStringList(existing.aliases),
            incomingAliases,
          )
          const tags = this.mergeGraphStringLists(
            this.parseStringList(existing.tags),
            incomingTags,
          )
          const evidenceMemoryIds = this.mergeGraphStringLists(
            this.parseStringList(existing.evidence_memory_ids),
            incomingEvidence,
          )
          this.db.prepare(`
            UPDATE memory_graph_nodes
            SET
              label = ?,
              aliases = ?,
              tags = ?,
              evidence_memory_ids = ?,
              confidence = ?,
              updated_at = datetime('now'),
              last_seen_at = datetime('now')
            WHERE id = ?
          `).run(
            label,
            JSON.stringify(aliases),
            JSON.stringify(tags),
            JSON.stringify(evidenceMemoryIds),
            Math.max(Number(existing.confidence) || 0, incomingConfidence),
            id,
          )
        } else {
          this.db.prepare(`
            INSERT INTO memory_graph_nodes (
              id,
              label,
              kind,
              aliases,
              tags,
              evidence_memory_ids,
              confidence,
              created_at,
              updated_at,
              last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), datetime('now'))
          `).run(
            id,
            label,
            kind,
            JSON.stringify(incomingAliases),
            JSON.stringify(incomingTags),
            JSON.stringify(incomingEvidence),
            incomingConfidence,
          )
        }

        nodesTouched.add(id)
        const row = this.db.prepare(`
          SELECT id, label, kind, aliases, tags, evidence_memory_ids, confidence, created_at, updated_at, last_seen_at
          FROM memory_graph_nodes
          WHERE id = ?
        `).get(id) as MemoryGraphNodeRow
        return this.mapGraphNodeRow(row)
      }

      for (const node of input.nodes ?? []) {
        ensureNode(node)
      }

      for (const edge of input.edges ?? []) {
        const relation = this.normalizeGraphRelation(edge.relation)
        if (!relation) continue
        const from = ensureNode({
          label: edge.fromLabel,
          kind: edge.fromKind,
          evidenceMemoryIds: edge.evidenceMemoryIds,
        })
        const to = ensureNode({
          label: edge.toLabel,
          kind: edge.toKind,
          evidenceMemoryIds: edge.evidenceMemoryIds,
        })
        if (!from || !to) continue

        const id = this.memoryGraphEdgeId(from.id, relation, to.id)
        const incomingTags = this.normalizeGraphStringList(edge.tags, 48, 80)
        const incomingEvidence = this.normalizeGraphStringList(edge.evidenceMemoryIds, 100, 160)
        const incomingConfidence = this.clampGraphConfidence(edge.confidence)
        const existing = this.db.prepare(`
          SELECT id, from_node_id, to_node_id, relation, tags, evidence_memory_ids, confidence, created_at, updated_at, last_seen_at
          FROM memory_graph_edges
          WHERE id = ?
        `).get(id) as MemoryGraphEdgeRow | undefined

        if (existing) {
          const tags = this.mergeGraphStringLists(
            this.parseStringList(existing.tags),
            incomingTags,
          )
          const evidenceMemoryIds = this.mergeGraphStringLists(
            this.parseStringList(existing.evidence_memory_ids),
            incomingEvidence,
          )
          this.db.prepare(`
            UPDATE memory_graph_edges
            SET
              tags = ?,
              evidence_memory_ids = ?,
              confidence = ?,
              updated_at = datetime('now'),
              last_seen_at = datetime('now')
            WHERE id = ?
          `).run(
            JSON.stringify(tags),
            JSON.stringify(evidenceMemoryIds),
            Math.max(Number(existing.confidence) || 0, incomingConfidence),
            id,
          )
        } else {
          this.db.prepare(`
            INSERT INTO memory_graph_edges (
              id,
              from_node_id,
              to_node_id,
              relation,
              tags,
              evidence_memory_ids,
              confidence,
              created_at,
              updated_at,
              last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), datetime('now'))
          `).run(
            id,
            from.id,
            to.id,
            relation,
            JSON.stringify(incomingTags),
            JSON.stringify(incomingEvidence),
            incomingConfidence,
          )
        }
        edgesTouched.add(id)
      }

      return {
        nodesUpserted: nodesTouched.size,
        edgesUpserted: edgesTouched.size,
      }
    })

    return upsert()
  }

  async searchGraph(
    query: string,
    options: { limit?: number; kind?: string } = {},
  ): Promise<MemoryGraphSearchResult> {
    const limit = this.clampGraphLimit(options.limit, 20)
    const trimmed = query.trim().toLowerCase()
    const nodeParams: Record<string, unknown> = { limit }
    const nodeConditions: string[] = []
    if (trimmed) {
      nodeParams.query = `%${trimmed}%`
      nodeConditions.push(`(
        lower(label) LIKE @query
        OR lower(aliases) LIKE @query
        OR lower(tags) LIKE @query
      )`)
    }
    if (typeof options.kind === 'string' && options.kind.trim().length > 0) {
      nodeParams.kind = this.normalizeGraphNodeKind(options.kind)
      nodeConditions.push('kind = @kind')
    }

    const nodeRows = this.db.prepare(`
      SELECT id, label, kind, aliases, tags, evidence_memory_ids, confidence, created_at, updated_at, last_seen_at
      FROM memory_graph_nodes
      ${nodeConditions.length > 0 ? `WHERE ${nodeConditions.join(' AND ')}` : ''}
      ORDER BY last_seen_at DESC, updated_at DESC
      LIMIT @limit
    `).all(nodeParams) as MemoryGraphNodeRow[]
    const nodesById = new Map(nodeRows.map((row) => [row.id, this.mapGraphNodeRow(row)]))

    const edgeParams: Record<string, unknown> = { limit }
    const edgeConditions: string[] = []
    const nodeIds = [...nodesById.keys()]
    if (nodeIds.length > 0) {
      const placeholders = nodeIds.map((id, index) => {
        const key = `node_${index}`
        edgeParams[key] = id
        return `@${key}`
      }).join(', ')
      edgeConditions.push(`(from_node_id IN (${placeholders}) OR to_node_id IN (${placeholders}))`)
    }
    if (trimmed) {
      edgeParams.query = `%${trimmed}%`
      edgeConditions.push(`(
        lower(relation) LIKE @query
        OR lower(tags) LIKE @query
        OR lower(evidence_memory_ids) LIKE @query
      )`)
    }

    const edgeRows = this.db.prepare(`
      SELECT id, from_node_id, to_node_id, relation, tags, evidence_memory_ids, confidence, created_at, updated_at, last_seen_at
      FROM memory_graph_edges
      ${edgeConditions.length > 0 ? `WHERE ${edgeConditions.join(' OR ')}` : ''}
      ORDER BY last_seen_at DESC, updated_at DESC
      LIMIT @limit
    `).all(edgeParams) as MemoryGraphEdgeRow[]
    const edges = edgeRows.map((row) => this.mapGraphEdgeRow(row))

    const missingNodeIds = new Set<string>()
    for (const edge of edges) {
      if (!nodesById.has(edge.fromNodeId)) missingNodeIds.add(edge.fromNodeId)
      if (!nodesById.has(edge.toNodeId)) missingNodeIds.add(edge.toNodeId)
    }
    if (missingNodeIds.size > 0) {
      const params: Record<string, unknown> = {}
      const placeholders = [...missingNodeIds].map((id, index) => {
        const key = `id_${index}`
        params[key] = id
        return `@${key}`
      }).join(', ')
      const rows = this.db.prepare(`
        SELECT id, label, kind, aliases, tags, evidence_memory_ids, confidence, created_at, updated_at, last_seen_at
        FROM memory_graph_nodes
        WHERE id IN (${placeholders})
      `).all(params) as MemoryGraphNodeRow[]
      for (const row of rows) {
        nodesById.set(row.id, this.mapGraphNodeRow(row))
      }
    }

    return {
      nodes: [...nodesById.values()],
      edges,
    }
  }

  async getGraphNeighbors(
    nodeId: string,
    options: { limit?: number; direction?: 'in' | 'out' | 'both' } = {},
  ): Promise<{ node: MemoryGraphNode | null; edges: MemoryGraphEdge[]; nodes: MemoryGraphNode[] }> {
    const nodeRow = this.db.prepare(`
      SELECT id, label, kind, aliases, tags, evidence_memory_ids, confidence, created_at, updated_at, last_seen_at
      FROM memory_graph_nodes
      WHERE id = ?
    `).get(nodeId) as MemoryGraphNodeRow | undefined
    if (!nodeRow) {
      return { node: null, edges: [], nodes: [] }
    }

    const limit = this.clampGraphLimit(options.limit, 20)
    const direction = options.direction ?? 'both'
    const condition = direction === 'in'
      ? 'to_node_id = @nodeId'
      : direction === 'out'
        ? 'from_node_id = @nodeId'
        : '(from_node_id = @nodeId OR to_node_id = @nodeId)'

    const edgeRows = this.db.prepare(`
      SELECT id, from_node_id, to_node_id, relation, tags, evidence_memory_ids, confidence, created_at, updated_at, last_seen_at
      FROM memory_graph_edges
      WHERE ${condition}
      ORDER BY confidence DESC, last_seen_at DESC
      LIMIT @limit
    `).all({ nodeId, limit }) as MemoryGraphEdgeRow[]
    const edges = edgeRows.map((row) => this.mapGraphEdgeRow(row))

    const neighborIds = new Set<string>()
    for (const edge of edges) {
      if (edge.fromNodeId !== nodeId) neighborIds.add(edge.fromNodeId)
      if (edge.toNodeId !== nodeId) neighborIds.add(edge.toNodeId)
    }

    let nodes: MemoryGraphNode[] = []
    if (neighborIds.size > 0) {
      const params: Record<string, unknown> = {}
      const placeholders = [...neighborIds].map((id, index) => {
        const key = `id_${index}`
        params[key] = id
        return `@${key}`
      }).join(', ')
      const rows = this.db.prepare(`
        SELECT id, label, kind, aliases, tags, evidence_memory_ids, confidence, created_at, updated_at, last_seen_at
        FROM memory_graph_nodes
        WHERE id IN (${placeholders})
      `).all(params) as MemoryGraphNodeRow[]
      nodes = rows.map((row) => this.mapGraphNodeRow(row))
    }

    return {
      node: this.mapGraphNodeRow(nodeRow),
      edges,
      nodes,
    }
  }

  async getGraphWikiPage(input: {
    query?: string
    nodeId?: string
    limit?: number
    evidenceLimit?: number
  }): Promise<MemoryGraphWikiPage> {
    const limit = this.clampGraphLimit(input.limit, 12)
    const evidenceLimit = this.clampGraphLimit(input.evidenceLimit, 12)
    let nodeId = typeof input.nodeId === 'string' ? input.nodeId.trim() : ''
    const query = typeof input.query === 'string' ? input.query.trim() : ''

    if (!nodeId && query) {
      const search = await this.searchGraph(query, { limit: 1 })
      nodeId = search.nodes[0]?.id ?? ''
    }
    if (!nodeId) {
      return { query: query || undefined, node: null, relationships: [], evidence: [] }
    }

    const graph = await this.getGraphNeighbors(nodeId, { limit, direction: 'both' })
    if (!graph.node) {
      return { query: query || undefined, node: null, relationships: [], evidence: [] }
    }

    const nodesById = new Map(graph.nodes.map((node) => [node.id, node]))
    const relationships = graph.edges
      .map((edge) => {
        const direction = edge.fromNodeId === graph.node?.id ? 'out' : 'in'
        const relatedNodeId = direction === 'out' ? edge.toNodeId : edge.fromNodeId
        const relatedNode = nodesById.get(relatedNodeId)
        if (!relatedNode) return null
        return {
          direction,
          edge,
          node: relatedNode,
        } satisfies MemoryGraphWikiPage['relationships'][number]
      })
      .filter((entry): entry is MemoryGraphWikiPage['relationships'][number] => entry !== null)

    const evidenceIds = this.mergeGraphStringLists(
      graph.node.evidenceMemoryIds,
      ...relationships.map((entry) => entry.edge.evidenceMemoryIds),
    ).slice(0, evidenceLimit)

    return {
      query: query || undefined,
      node: graph.node,
      relationships,
      evidence: this.getEvidenceMemories(evidenceIds),
      quality: this.buildGraphWikiPageQuality(graph.node, relationships),
    }
  }

  async runGraphMaintenance(
    options: { pruneMissingEvidence?: boolean } = {},
  ): Promise<MemoryGraphMaintenanceResult> {
    if (options.pruneMissingEvidence === false) {
      const stats = await this.getGraphStats()
      return {
        checkedNodes: stats.nodes,
        checkedEdges: stats.edges,
        prunedNodes: 0,
        prunedEdges: 0,
      }
    }

    const tx = this.db.transaction(() => {
      const edgeRows = this.db.prepare(`
        SELECT id, from_node_id, to_node_id, relation, tags, evidence_memory_ids, confidence, created_at, updated_at, last_seen_at
        FROM memory_graph_edges
      `).all() as MemoryGraphEdgeRow[]
      const nodeRows = this.db.prepare(`
        SELECT id, label, kind, aliases, tags, evidence_memory_ids, confidence, created_at, updated_at, last_seen_at
        FROM memory_graph_nodes
      `).all() as MemoryGraphNodeRow[]

      let prunedEdges = 0
      for (const edge of edgeRows) {
        if (this.hasActiveEvidence(this.parseStringList(edge.evidence_memory_ids))) continue
        this.db.prepare('DELETE FROM memory_graph_edges WHERE id = ?').run(edge.id)
        prunedEdges += 1
      }

      let prunedNodes = 0
      for (const node of nodeRows) {
        const degree = this.db.prepare(`
          SELECT COUNT(*) AS count
          FROM memory_graph_edges
          WHERE from_node_id = ? OR to_node_id = ?
        `).get(node.id, node.id) as { count: number }
        if ((degree.count ?? 0) > 0) continue
        if (this.hasActiveEvidence(this.parseStringList(node.evidence_memory_ids))) continue
        this.db.prepare('DELETE FROM memory_graph_nodes WHERE id = ?').run(node.id)
        prunedNodes += 1
      }

      return {
        checkedNodes: nodeRows.length,
        checkedEdges: edgeRows.length,
        prunedNodes,
        prunedEdges,
      }
    })

    return tx()
  }

  async getGraphStats(): Promise<MemoryGraphStats> {
    const nodeRow = this.db.prepare(`
      SELECT COUNT(*) AS count, MAX(updated_at) AS updatedAt
      FROM memory_graph_nodes
    `).get() as { count: number; updatedAt: string | null }
    const edgeRow = this.db.prepare(`
      SELECT COUNT(*) AS count, MAX(updated_at) AS updatedAt
      FROM memory_graph_edges
    `).get() as { count: number; updatedAt: string | null }
    const updatedTimes = [nodeRow.updatedAt, edgeRow.updatedAt]
      .filter((value): value is string => typeof value === 'string' && value.length > 0)
      .sort((left, right) => Date.parse(right) - Date.parse(left))

    return {
      nodes: nodeRow.count ?? 0,
      edges: edgeRow.count ?? 0,
      lastUpdatedAt: updatedTimes[0],
    }
  }

  async inspectGraphQuality(
    options: {
      limit?: number
      signalLimit?: number
      lowConfidenceThreshold?: number
      staleAfterDays?: number
    } = {},
  ): Promise<MemoryGraphQualityReport> {
    const limit = this.clampGraphLimit(options.limit, 200, 1000)
    const signalLimit = this.clampGraphLimit(options.signalLimit, 50, 500)
    const lowConfidenceThreshold = this.clampGraphQualityThreshold(
      options.lowConfidenceThreshold,
      GRAPH_LOW_CONFIDENCE_THRESHOLD,
    )
    const staleAfterDays = this.clampGraphDays(options.staleAfterDays, GRAPH_STALE_AFTER_DAYS)
    const stats = await this.getGraphStats()

    const nodeRows = this.db.prepare(`
      SELECT id, label, kind, aliases, tags, evidence_memory_ids, confidence, created_at, updated_at, last_seen_at
      FROM memory_graph_nodes
      ORDER BY confidence ASC, last_seen_at ASC
      LIMIT @limit
    `).all({ limit }) as MemoryGraphNodeRow[]
    const edgeRows = this.db.prepare(`
      SELECT id, from_node_id, to_node_id, relation, tags, evidence_memory_ids, confidence, created_at, updated_at, last_seen_at
      FROM memory_graph_edges
      ORDER BY confidence ASC, last_seen_at ASC
      LIMIT @limit
    `).all({ limit }) as MemoryGraphEdgeRow[]

    const signals: MemoryGraphQualitySignal[] = []
    const report: MemoryGraphQualityReport = {
      generatedAt: new Date().toISOString(),
      stats,
      scannedNodes: nodeRows.length,
      scannedEdges: edgeRows.length,
      lowConfidenceNodes: 0,
      lowConfidenceEdges: 0,
      thinEvidenceNodes: 0,
      thinEvidenceEdges: 0,
      missingEvidenceNodes: 0,
      missingEvidenceEdges: 0,
      inactiveEvidenceNodes: 0,
      inactiveEvidenceEdges: 0,
      staleNodes: 0,
      staleEdges: 0,
      orphanNodes: 0,
      contradictions: 0,
      signals,
    }

    const pushSignal = (signal: MemoryGraphQualitySignal) => {
      if (signals.length < signalLimit) signals.push(signal)
    }

    for (const row of nodeRows) {
      const node = this.mapGraphNodeRow(row)
      const evidence = this.getEvidenceStats(node.evidenceMemoryIds)
      const stale = this.isGraphEntryStale(node.lastSeenAt, staleAfterDays)
      const degree = this.getGraphNodeDegree(node.id)

      if (node.confidence < lowConfidenceThreshold) {
        report.lowConfidenceNodes += 1
        pushSignal({
          code: 'low_confidence',
          severity: 'warning',
          subjectType: 'node',
          subjectId: node.id,
          message: `Graph node "${node.label}" has low confidence ${node.confidence.toFixed(2)}.`,
        })
      }
      if (evidence.activeIds.length === 0) {
        report.thinEvidenceNodes += 1
        pushSignal({
          code: 'thin_evidence',
          severity: 'critical',
          subjectType: 'node',
          subjectId: node.id,
          message: `Graph node "${node.label}" has no active evidence memories.`,
          evidenceMemoryIds: evidence.evidenceIds,
        })
      }
      if (evidence.missingIds.length > 0) {
        report.missingEvidenceNodes += 1
        pushSignal({
          code: 'missing_evidence',
          severity: 'warning',
          subjectType: 'node',
          subjectId: node.id,
          message: `Graph node "${node.label}" references ${evidence.missingIds.length} missing evidence memories.`,
          evidenceMemoryIds: evidence.missingIds,
        })
      }
      if (evidence.inactiveIds.length > 0) {
        report.inactiveEvidenceNodes += 1
        pushSignal({
          code: 'inactive_evidence',
          severity: 'info',
          subjectType: 'node',
          subjectId: node.id,
          message: `Graph node "${node.label}" still cites ${evidence.inactiveIds.length} archived or superseded evidence memories.`,
          evidenceMemoryIds: evidence.inactiveIds,
        })
      }
      if (stale) {
        report.staleNodes += 1
        pushSignal({
          code: 'stale',
          severity: 'info',
          subjectType: 'node',
          subjectId: node.id,
          message: `Graph node "${node.label}" has not been seen for at least ${staleAfterDays} days.`,
        })
      }
      if (degree === 0 && evidence.activeIds.length === 0) {
        report.orphanNodes += 1
        pushSignal({
          code: 'orphan',
          severity: 'warning',
          subjectType: 'node',
          subjectId: node.id,
          message: `Graph node "${node.label}" is orphaned and has no active evidence.`,
          evidenceMemoryIds: evidence.evidenceIds,
        })
      }
    }

    for (const row of edgeRows) {
      const edge = this.mapGraphEdgeRow(row)
      const evidence = this.getEvidenceStats(edge.evidenceMemoryIds)

      if (edge.confidence < lowConfidenceThreshold) {
        report.lowConfidenceEdges += 1
        pushSignal({
          code: 'low_confidence',
          severity: 'warning',
          subjectType: 'edge',
          subjectId: edge.id,
          message: `Graph edge ${edge.id} has low confidence ${edge.confidence.toFixed(2)}.`,
        })
      }
      if (evidence.activeIds.length === 0) {
        report.thinEvidenceEdges += 1
        pushSignal({
          code: 'thin_evidence',
          severity: 'critical',
          subjectType: 'edge',
          subjectId: edge.id,
          message: `Graph edge ${edge.id} has no active evidence memories.`,
          evidenceMemoryIds: evidence.evidenceIds,
        })
      }
      if (evidence.missingIds.length > 0) {
        report.missingEvidenceEdges += 1
        pushSignal({
          code: 'missing_evidence',
          severity: 'warning',
          subjectType: 'edge',
          subjectId: edge.id,
          message: `Graph edge ${edge.id} references ${evidence.missingIds.length} missing evidence memories.`,
          evidenceMemoryIds: evidence.missingIds,
        })
      }
      if (evidence.inactiveIds.length > 0) {
        report.inactiveEvidenceEdges += 1
        pushSignal({
          code: 'inactive_evidence',
          severity: 'info',
          subjectType: 'edge',
          subjectId: edge.id,
          message: `Graph edge ${edge.id} still cites ${evidence.inactiveIds.length} archived or superseded evidence memories.`,
          evidenceMemoryIds: evidence.inactiveIds,
        })
      }
      if (this.isGraphEntryStale(edge.lastSeenAt, staleAfterDays)) {
        report.staleEdges += 1
        pushSignal({
          code: 'stale',
          severity: 'info',
          subjectType: 'edge',
          subjectId: edge.id,
          message: `Graph edge ${edge.id} has not been seen for at least ${staleAfterDays} days.`,
        })
      }
      if (this.isContradictionEdge(edge)) {
        report.contradictions += 1
        pushSignal({
          code: 'contradiction',
          severity: 'warning',
          subjectType: 'edge',
          subjectId: edge.id,
          message: `Graph edge ${edge.id} records a contradiction relationship.`,
          evidenceMemoryIds: edge.evidenceMemoryIds,
        })
      }
    }

    return report
  }

  async repairGraphQuality(
    options: {
      dryRun?: boolean
      limit?: number
      signalLimit?: number
      lowConfidenceThreshold?: number
      staleAfterDays?: number
    } = {},
  ): Promise<MemoryGraphRepairResult> {
    const report = await this.inspectGraphQuality({
      limit: options.limit,
      signalLimit: options.signalLimit,
      lowConfidenceThreshold: options.lowConfidenceThreshold,
      staleAfterDays: options.staleAfterDays,
    })
    const proposals = this.buildGraphRepairProposals(report)
    const dryRun = options.dryRun !== false
    const hasSafeProposal = proposals.some((proposal) => proposal.safeToApply)
    const maintenance = !dryRun && hasSafeProposal
      ? await this.runGraphMaintenance({ pruneMissingEvidence: true })
      : {
          checkedNodes: 0,
          checkedEdges: 0,
          prunedNodes: 0,
          prunedEdges: 0,
        }

    return {
      dryRun,
      generatedAt: new Date().toISOString(),
      report,
      proposals,
      applied: {
        checkedNodes: maintenance.checkedNodes,
        checkedEdges: maintenance.checkedEdges,
        prunedNodes: maintenance.prunedNodes,
        prunedEdges: maintenance.prunedEdges,
        skippedUnsafe: proposals.filter((proposal) => !proposal.safeToApply).length,
      },
    }
  }

  async applyGraphRepairDecision(
    input: MemoryGraphRepairDecisionInput,
  ): Promise<MemoryGraphRepairDecisionResult> {
    const action = input.action ?? 'supersede_memories'
    if (action !== 'supersede_memories') {
      throw new Error(`Unsupported graph repair decision action: ${action}`)
    }
    const dryRun = input.dryRun !== false
    const winnerMemoryId = input.winnerMemoryId.trim()
    const supersededMemoryIds = this.mergeGraphStringLists(input.supersededMemoryIds)
    const updated: MemoryGraphRepairDecisionUpdate[] = []
    const skipped: MemoryGraphRepairDecisionSkip[] = []
    const emptyMaintenance: MemoryGraphMaintenanceResult = {
      checkedNodes: 0,
      checkedEdges: 0,
      prunedNodes: 0,
      prunedEdges: 0,
    }

    const resultBase = (): Omit<MemoryGraphRepairDecisionResult, 'maintenance'> => ({
      dryRun,
      generatedAt: new Date().toISOString(),
      action,
      proposalId: input.proposalId,
      winnerMemoryId,
      updated,
      skipped,
    })

    const winner = winnerMemoryId ? await this.get(winnerMemoryId) : null
    if (!winner) {
      skipped.push({
        id: winnerMemoryId || '(missing winnerMemoryId)',
        reason: 'winner_not_found',
        message: 'Winner memory was not found; no repair decision was applied.',
      })
      return { ...resultBase(), maintenance: emptyMaintenance }
    }
    if (this.isInactiveEvidenceMemory(winner)) {
      skipped.push({
        id: winner.id,
        reason: 'winner_inactive',
        message: 'Winner memory is archived or already superseded; no repair decision was applied.',
      })
      return { ...resultBase(), maintenance: emptyMaintenance }
    }

    for (const loserId of supersededMemoryIds) {
      if (loserId === winnerMemoryId) {
        skipped.push({
          id: loserId,
          reason: 'winner_in_superseded_ids',
          message: 'Winner memory cannot also be superseded.',
        })
        continue
      }

      const loser = await this.get(loserId)
      if (!loser) {
        skipped.push({
          id: loserId,
          reason: 'memory_not_found',
          message: 'Superseded memory was not found.',
        })
        continue
      }
      if (loser.tags.some((tag) => tag.toLowerCase() === `superseded-by:${winnerMemoryId.toLowerCase()}`)) {
        skipped.push({
          id: loserId,
          reason: 'already_superseded_by_winner',
          message: 'Memory is already superseded by the requested winner.',
        })
        continue
      }
      if (loser.tags.some((tag) => tag.toLowerCase().startsWith('superseded-by:'))) {
        skipped.push({
          id: loserId,
          reason: 'already_superseded',
          message: 'Memory is already superseded by another winner.',
        })
        continue
      }

      const nextTags = this.mergeGraphStringLists(loser.tags, [
        'superseded',
        `superseded-by:${winnerMemoryId}`,
      ])
      if (tagsEqual(nextTags, loser.tags)) {
        skipped.push({
          id: loserId,
          reason: 'no_change',
          message: 'Repair decision would not change this memory.',
        })
        continue
      }

      updated.push({
        id: loser.id,
        beforeTags: loser.tags,
        afterTags: nextTags,
      })

      if (dryRun) continue
      await this.add({
        id: loser.id,
        content: loser.content,
        source: loser.source,
        tags: nextTags,
      })
      try {
        await this.recordAudit({
          memoryId: loser.id,
          action: 'updated',
          actor: input.actor ?? 'memory.graph.repair',
          reason: input.reason ?? `Graph repair decision superseded by ${winnerMemoryId}`,
          before: {
            id: loser.id,
            content: loser.content,
            source: loser.source,
            tags: loser.tags,
          },
          after: {
            id: loser.id,
            content: loser.content,
            source: loser.source,
            tags: nextTags,
          },
        })
      } catch {
        // Repair should not become half-applied because the audit sink failed.
      }
    }

    const maintenance = !dryRun && updated.length > 0
      ? await this.runGraphMaintenance({ pruneMissingEvidence: true })
      : emptyMaintenance
    return { ...resultBase(), maintenance }
  }

  close(): void {
    this.closed = true
    // Fold the WAL back into the main db so the -wal file does not accrue
    // across restarts. Best-effort; never block close on a checkpoint error.
    try {
      this.db.pragma('wal_checkpoint(TRUNCATE)')
    } catch {
      /* ignore checkpoint failure */
    }
    this.db.close()
  }

  private async bootstrap(): Promise<void> {
    const state = this.readState()

    if (!this.isSemanticConfigured()) {
      this.writeState({
        status: 'disabled',
        lastError: null,
      })
      return
    }

    if (this.isConfiguredStateMismatch(state)) {
      this.writeState({
        status: 'reindex_required',
        lastError: `Embedding configuration changed from ${state.provider_id}/${state.model} to ${this.configuredProviderId}/${this.embeddingModel}. Reindex required.`,
      })
      return
    }

    if (!this.vectorBackend.available) {
      this.writeState({
        providerId: this.configuredProviderId ?? null,
        model: this.embeddingModel ?? null,
        status: 'degraded',
        lastError: `${this.vectorBackend.kind} backend unavailable`,
      })
      return
    }

    if (!this.embedder) {
      this.writeState({
        providerId: this.configuredProviderId ?? null,
        model: this.embeddingModel ?? null,
        status: 'degraded',
        lastError: 'Embedding provider unavailable or does not support embeddings',
      })
      return
    }

    const dimensions = state.dimensions ?? this.detectCachedEmbeddingDimensions()
    if (dimensions) {
      try {
        await this.ensureVectorBackendReady(dimensions)
        if (this.embeddingModel) {
          await this.vectorBackend.restoreFromCache(this.embeddingModel, dimensions)
        }
      } catch (error) {
        this.writeState({
          providerId: this.configuredProviderId ?? null,
          model: this.embeddingModel ?? null,
          dimensions,
          status: 'degraded',
          lastError: this.toErrorMessage(error),
        })
        return
      }
    }

    this.persistDerivedStatus(null)
  }

  private readState(): SemanticIndexStateRow {
    const row = this.db.prepare(`
      SELECT provider_id, model, dimensions, status, last_error
      FROM semantic_index_state
      WHERE singleton = 1
    `).get() as SemanticIndexStateRow | undefined

    return row ?? {
      provider_id: null,
      model: null,
      dimensions: null,
      status: 'disabled',
      last_error: null,
    }
  }

  private writeState(update: {
    providerId?: string | null
    model?: string | null
    dimensions?: number | null
    status?: MemorySemanticStatus
    lastError?: string | null
  }): void {
    const current = this.readState()
    const next = {
      provider_id: update.providerId === undefined ? current.provider_id : update.providerId,
      model: update.model === undefined ? current.model : update.model,
      dimensions: update.dimensions === undefined ? current.dimensions : update.dimensions,
      status: update.status ?? current.status,
      last_error: update.lastError === undefined ? current.last_error : update.lastError,
    }

    this.db.prepare(`
      INSERT INTO semantic_index_state (
        singleton,
        provider_id,
        model,
        dimensions,
        status,
        last_error,
        updated_at
      )
      VALUES (1, ?, ?, ?, ?, ?, datetime('now'))
      ON CONFLICT(singleton) DO UPDATE SET
        provider_id = excluded.provider_id,
        model = excluded.model,
        dimensions = excluded.dimensions,
        status = excluded.status,
        last_error = excluded.last_error,
        updated_at = excluded.updated_at
    `).run(
      next.provider_id,
      next.model,
      next.dimensions,
      next.status,
      next.last_error,
    )
  }

  private persistDerivedStatus(lastError?: string | null): void {
    const status = this.getStatus()
    this.writeState({
      providerId: this.isSemanticConfigured() ? (this.configuredProviderId ?? null) : undefined,
      model: this.isSemanticConfigured() ? (this.embeddingModel ?? null) : undefined,
      dimensions: this.activeDimensions ?? undefined,
      status: status.status,
      lastError: lastError === undefined ? (status.lastError ?? null) : lastError,
    })
  }

  private isSemanticConfigured(): boolean {
    return Boolean(this.configuredProviderId && this.embeddingModel)
  }

  private isConfiguredStateMismatch(state: SemanticIndexStateRow): boolean {
    if (!this.isSemanticConfigured()) return false
    if (!state.provider_id || !state.model) return false
    return state.provider_id !== this.configuredProviderId || state.model !== this.embeddingModel
  }

  private canWriteSemanticVectors(): boolean {
    const state = this.readState()
    return Boolean(
      this.isSemanticConfigured()
      && this.vectorBackend.available
      && this.embedder
      && !this.isConfiguredStateMismatch(state),
    )
  }

  private async prepareForReindex(): Promise<void> {
    if (!this.isSemanticConfigured()) {
      throw new Error('Semantic search is disabled because no embedding provider/model is configured')
    }
    if (!this.vectorBackend.available) {
      throw new Error(`${this.vectorBackend.kind} backend unavailable`)
    }
    if (!this.embedder) {
      throw new Error('Embedding provider unavailable or does not support embeddings')
    }

    // Atomic-swap-lite: probe the embedder BEFORE destroying the existing
    // index. If the provider is actually down (not just object-present), this
    // throws here and the old vectors/blobs stay queryable, instead of clearing
    // first and then discovering the provider cannot rebuild — which would kill
    // search until the provider recovers.
    const probe = await this.embedTexts(['reindex readiness probe'])
    this.validateEmbeddings(probe)

    await this.vectorBackend.clear()

    const reset = this.db.transaction(() => {
      this.activeDimensions = undefined

      this.db.prepare(`
        UPDATE memories
        SET embedding = NULL,
            embedding_model = NULL,
            embedding_state = 'pending',
            embedding_error = NULL,
            embedding_updated_at = NULL
        WHERE content IS NOT NULL
          AND trim(content) != ''
      `).run()

      this.db.prepare(`
        UPDATE memories
        SET embedding = NULL,
            embedding_model = NULL,
            embedding_state = 'disabled',
            embedding_error = NULL,
            embedding_updated_at = NULL
        WHERE content IS NULL
          OR trim(content) = ''
      `).run()

      this.writeState({
        providerId: this.configuredProviderId ?? null,
        model: this.embeddingModel ?? null,
        dimensions: null,
        status: 'backfilling',
        lastError: null,
      })
    })

    reset()
  }

  private isSemanticQueryable(): boolean {
    const status = this.getStatus().status
    return Boolean(
      this.vectorBackend.available
      && this.activeDimensions
      && status !== 'disabled'
      && status !== 'reindex_required',
    )
  }

  private async ensureVectorBackendReady(dimensions: number): Promise<void> {
    if (!this.vectorBackend.available) return
    if (dimensions <= 0) {
      throw new Error('Embedding dimension must be greater than zero')
    }

    const state = this.readState()
    if (state.dimensions && state.dimensions !== dimensions) {
      this.writeState({
        status: 'reindex_required',
        lastError: `Embedding dimension changed from ${state.dimensions} to ${dimensions}. Reindex required.`,
      })
      throw new Error(`Embedding dimension mismatch: expected ${state.dimensions}, received ${dimensions}`)
    }

    await this.vectorBackend.ensureReady(dimensions)
    this.activeDimensions = dimensions

    this.writeState({
      providerId: this.configuredProviderId ?? null,
      model: this.embeddingModel ?? null,
      dimensions,
      status: 'backfilling',
      lastError: null,
    })
  }

  private detectCachedEmbeddingDimensions(): number | undefined {
    if (!this.embeddingModel) return undefined

    const row = this.db.prepare(`
      SELECT embedding
      FROM memories
      WHERE embedding IS NOT NULL
        AND embedding_model = ?
      LIMIT 1
    `).get(this.embeddingModel) as { embedding: Uint8Array } | undefined

    if (!row?.embedding) return undefined
    try {
      return blobToFloat32Array(row.embedding).length
    } catch {
      return undefined
    }
  }

  private getNeedsEmbeddingWhereClause(): string {
    // A row needs (re)embedding when it is missing/stale, OR when a prior
    // embedding failed but is still within the bounded-retry budget and past
    // its cooldown window (transient errors recover instead of parking forever).
    const maxAttempts = resolveEmbedMaxAttempts()
    return `
      content IS NOT NULL
      AND trim(content) != ''
      AND (
        (
          COALESCE(embedding_state, 'pending') != 'failed'
          AND (
            embedding IS NULL
            OR embedding_model IS NULL
            OR embedding_model != @model
            OR embedding_updated_at IS NULL
            OR julianday(embedding_updated_at) < julianday(updated_at)
          )
        )
        OR (
          embedding_state = 'failed'
          AND COALESCE(embedding_attempts, 0) < ${maxAttempts}
          AND (
            embedding_next_retry_at IS NULL
            OR julianday(embedding_next_retry_at) <= julianday('now')
          )
        )
      )
    `
  }

  private getPendingRowCount(): number {
    if (!this.embeddingModel) return 0

    const row = this.db.prepare(`
      SELECT COUNT(*) AS count
      FROM memories
      WHERE ${this.getNeedsEmbeddingWhereClause()}
    `).get({ model: this.embeddingModel }) as { count: number }

    return row.count
  }

  private getFailedRowCount(): number {
    const row = this.db.prepare(`
      SELECT COUNT(*) AS count
      FROM memories
      WHERE embedding_state = 'failed'
    `).get() as { count: number }

    return row.count
  }

  private async backfillInternal(batchSize: number): Promise<number> {
    if (!this.embeddingModel) return 0

    let total = 0

    while (!this.closed) {
      const rows = this.db.prepare(`
        SELECT rowid, id, content
        FROM memories
        WHERE ${this.getNeedsEmbeddingWhereClause()}
        ORDER BY updated_at ASC, rowid ASC
        LIMIT @limit
      `).all({
        model: this.embeddingModel,
        limit: batchSize,
      }) as MemoryRow[]

      if (rows.length === 0) return total

      const processed = await this.embedAndWriteWithSplitRetry(rows)
      total += processed
      if (processed === 0) {
        // The whole batch failed and every row was marked for a later retry
        // (cooldown); stop this pass instead of re-selecting the same rows.
        return total
      }
    }

    return total
  }

  // Embed and persist a batch, bisecting on failure so a single bad row does
  // not fail its whole batch. On singleton failure the row is marked failed
  // (bounded retry). Returns the number of rows successfully embedded.
  private async embedAndWriteWithSplitRetry(rows: MemoryRow[]): Promise<number> {
    if (rows.length === 0) return 0

    let embeddings: number[][]
    let dimensions: number
    try {
      embeddings = await this.embedTexts(rows.map((row) => row.content))
      if (embeddings.length !== rows.length) {
        throw new Error(`Embedding provider returned ${embeddings.length} vectors for ${rows.length} inputs`)
      }
      dimensions = this.validateEmbeddings(embeddings)
    } catch (error) {
      const message = this.toErrorMessage(error)
      if (rows.length === 1) {
        this.markRowsFailed([rows[0]!.id], message)
        this.writeState({
          providerId: this.configuredProviderId ?? null,
          model: this.embeddingModel ?? null,
          dimensions: this.activeDimensions ?? undefined,
          status: 'degraded',
          lastError: message,
        })
        return 0
      }
      const mid = Math.floor(rows.length / 2)
      const left = await this.embedAndWriteWithSplitRetry(rows.slice(0, mid))
      const right = await this.embedAndWriteWithSplitRetry(rows.slice(mid))
      return left + right
    }

    await this.ensureVectorBackendReady(dimensions)
    for (let index = 0; index < rows.length; index++) {
      await this.writeEmbedding(rows[index]!.rowid, rows[index]!.id, embeddings[index]!)
    }
    return rows.length
  }

  private async embedTexts(texts: string[]): Promise<number[][]> {
    if (!this.embedder || !this.embeddingModel) {
      throw new Error('Semantic embeddings are not configured')
    }

    const vectors = await this.embedder.embed(texts, this.embeddingModel)
    if (!Array.isArray(vectors)) {
      throw new Error('Embedding provider returned a non-array response')
    }

    return vectors
  }

  private async embedDocumentRowsInBatches(
    rows: MemoryRow[],
    batchSize: number,
  ): Promise<string | null> {
    let expectedDimensions: number | undefined
    let lastError: string | null = null

    for (let start = 0; start < rows.length; start += batchSize) {
      const batch = rows.slice(start, start + batchSize)
      try {
        const embeddings = await this.embedTexts(batch.map((row) => row.content))
        if (embeddings.length !== batch.length) {
          throw new Error(`Embedding provider returned ${embeddings.length} vectors for ${batch.length} chunks`)
        }
        const dimensions = this.validateEmbeddings(embeddings)
        if (expectedDimensions !== undefined && dimensions !== expectedDimensions) {
          throw new Error(`Embedding provider returned inconsistent vector dimensions: expected ${expectedDimensions}, received ${dimensions}`)
        }
        expectedDimensions = dimensions
        await this.ensureVectorBackendReady(dimensions)
        for (let index = 0; index < batch.length; index++) {
          await this.writeEmbedding(batch[index].rowid, batch[index].id, embeddings[index])
        }
      } catch (error) {
        lastError = this.toErrorMessage(error)
        this.markRowsFailed(batch.map((row) => row.id), lastError)
      }
    }

    return lastError
  }

  private validateEmbeddings(vectors: number[][]): number {
    if (vectors.length === 0) {
      throw new Error('Embedding provider returned no vectors')
    }

    const dimensions = this.validateEmbedding(vectors[0])
    for (const vector of vectors.slice(1)) {
      if (this.validateEmbedding(vector) !== dimensions) {
        throw new Error('Embedding provider returned inconsistent vector dimensions')
      }
    }

    return dimensions
  }

  private validateEmbedding(vector: number[] | undefined): number {
    if (!vector || vector.length === 0) {
      throw new Error('Embedding provider returned an empty vector')
    }
    if (!vector.every((value) => Number.isFinite(value))) {
      throw new Error('Embedding provider returned a non-finite vector value')
    }
    return vector.length
  }

  private async writeEmbedding(rowid: number, id: string, embedding: number[]): Promise<void> {
    if (!this.embeddingModel || !this.activeDimensions) return

    const typedEmbedding = new Float32Array(embedding)
    this.db.prepare(`
      UPDATE memories
      SET embedding = ?,
          embedding_model = ?,
          embedding_state = 'pending',
          embedding_error = NULL,
          embedding_updated_at = datetime('now')
      WHERE id = ?
    `).run(typedEmbedding, this.embeddingModel, id)

    await this.vectorBackend.upsert(rowid, typedEmbedding, this.embeddingModel)

    this.db.prepare(`
      UPDATE memories
      SET embedding_state = 'ready',
          embedding_error = NULL,
          embedding_attempts = 0,
          embedding_next_retry_at = NULL,
          embedding_updated_at = datetime('now')
      WHERE id = ?
    `).run(id)
  }

  private async deleteVectorRows(rowids: number[]): Promise<void> {
    if (rowids.length === 0) return
    try {
      await this.vectorBackend.delete(rowids)
    } catch (error) {
      this.writeState({
        providerId: this.configuredProviderId ?? null,
        model: this.embeddingModel ?? null,
        dimensions: this.activeDimensions ?? undefined,
        status: 'degraded',
        lastError: this.toErrorMessage(error),
      })
    }
  }

  private markRowsFailed(ids: string[], error: string): void {
    if (ids.length === 0) return

    // Increment the attempt counter and schedule the next retry a cooldown
    // window out, so a transient failure is retried by a later backfill pass
    // (bounded by SEPILOTD_EMBED_MAX_ATTEMPTS) rather than parked permanently.
    const retryMinutes = resolveEmbedRetryMinutes()
    const updateFailure = this.db.prepare(`
      UPDATE memories
      SET embedding_state = 'failed',
          embedding_error = @message,
          embedding_attempts = COALESCE(embedding_attempts, 0) + 1,
          embedding_next_retry_at = datetime('now', @retry)
      WHERE id = @id
    `)

    const tx = this.db.transaction((memoryIds: string[], message: string) => {
      for (const id of memoryIds) {
        updateFailure.run({ message, retry: `+${retryMinutes} minutes`, id })
      }
    })

    tx(ids, error)
  }

  /**
   * Build the FTS5 MATCH expression, matching each term as a prefix.
   *
   * FTS5's default tokenizer splits on whitespace and punctuation, which suits
   * languages that separate words with spaces. Korean glues particles onto the
   * stem, so a memory stored as `주 개발 언어는 TypeScript` indexes the token
   * `언어는`, and the quoted-exact query `언어` matched nothing — recall of a
   * memory the user had just saved silently returned zero hits while the same
   * text was plainly there (`언어*` finds it, `언어` does not).
   *
   * The trailing `*` costs nothing for languages that already matched — an
   * exact term is a prefix of itself — and makes agglutinative queries work
   * without re-indexing every memory under a CJK-aware tokenizer.
   *
   * `prefix` and `joiner` pick how strict the match is. See
   * {@link keywordSearch}: the exact, all-terms form runs first and the looser
   * forms only when it finds nothing.
   */
  private escapeFts5(
    query: string,
    options: { prefix?: boolean; joiner?: ' ' | ' OR ' } = {},
  ): string {
    const suffix = options.prefix ? '*' : ''
    return query
      .split(/\s+/)
      .filter(Boolean)
      .map((word) => `"${word.replace(/"/g, '""')}"${suffix}`)
      .join(options.joiner ?? ' ')
  }

  private keywordSearch(
    query: string,
    limit: number,
    options?: SemanticSearchOptions,
  ): MemoryEntry[] {
    const filter = this.buildSearchFilter('m', options)
    try {
      const match = (expression: string) => this.db.prepare(`
        SELECT m.id, m.content, m.source, m.tags, rank AS score
        FROM memories_fts f
        JOIN memories m ON m.rowid = f.rowid
        WHERE memories_fts MATCH @query
        ${filter.clause}
        ORDER BY rank
        LIMIT @limit
      `).all({
        query: expression,
        limit,
        ...filter.params,
      }) as Array<{
        id: string
        content: string
        source: MemoryEntry['source']
        tags: string | null
        score: number
      }>

      const exact = this.escapeFts5(query)
      if (exact) {
        // Widen only when the stricter form found nothing, so a query that
        // already worked keeps returning exactly what it did.
        //
        // Exact whole terms, all required, is right for a language that puts
        // spaces between words. Korean glues particles onto the stem, so a
        // memory stored as `주 개발 언어는 TypeScript` indexes `언어는` and a
        // search for `언어` matched nothing — recall of a memory the user had
        // just saved returned zero hits while the text was plainly there. And
        // recall is asked in whole sentences ("주력 언어 프로그래밍", "내 노트북
        // OS") that no single memory contains every word of. Against real data
        // those two went from 0 hits to 8 and 9 through these fallbacks.
        const matched = options?.widenWhenEmpty
          ? firstNonEmpty(
              () => match(exact),
              () => match(this.escapeFts5(query, { prefix: true })),
              () => match(this.escapeFts5(query, { prefix: true, joiner: ' OR ' })),
            )
          : match(exact)

        return matched.map((row) => ({
          id: row.id,
          content: row.content,
          source: row.source,
          tags: this.parseTags(row.tags),
          score: this.normalizeKeywordScore(row.score),
        }))
      }
    } catch {
      // Fall back to LIKE if the FTS parser rejects the query.
    }

    const rows = this.db.prepare(`
      SELECT id, content, source, tags
      FROM memories m
      WHERE content LIKE @pattern
      ${filter.clause}
      LIMIT @limit
    `).all({
      pattern: `%${query}%`,
      limit,
      ...filter.params,
    }) as Array<{
      id: string
      content: string
      source: MemoryEntry['source']
      tags: string | null
    }>

    return rows.map((row) => ({
      id: row.id,
      content: row.content,
      source: row.source,
      tags: this.parseTags(row.tags),
      score: 0.25,
    }))
  }

  private async semanticSearch(
    query: string,
    limit: number,
    options?: SemanticSearchOptions,
  ): Promise<MemoryEntry[]> {
    if (!this.isSemanticQueryable() || !this.activeDimensions || !this.embeddingModel) return []

    let queryEmbedding: number[]
    try {
      const embeddings = await this.embedTexts([query])
      queryEmbedding = embeddings[0]
      this.validateEmbedding(queryEmbedding)
      if (queryEmbedding.length !== this.activeDimensions) {
        throw new Error(`Query embedding dimension mismatch: expected ${this.activeDimensions}, received ${queryEmbedding.length}`)
      }
    } catch (error) {
      this.writeState({
        status: 'degraded',
        lastError: this.toErrorMessage(error),
      })
      return []
    }

    const filter = this.buildSearchFilter('m', options)
    let matches: Array<{ memoryRowid: number; distance: number }>
    try {
      const needsEligibleRanking = options?.scopeTags !== undefined || options?.tags?.length
        || options?.excludeTags?.length || options?.sources?.length || options?.documentId
        || options?.createdAfter || options?.createdBefore || options?.asOf
        || this.db.prepare(`SELECT 1 FROM memories WHERE
          json_extract(metadata, '$.memory.validFrom') IS NOT NULL OR
          json_extract(metadata, '$.memory.validUntil') IS NOT NULL OR
          json_extract(metadata, '$.memory.status') != 'active' LIMIT 1`).get()
      if (!needsEligibleRanking) {
        matches = await this.vectorBackend.search(new Float32Array(queryEmbedding), limit, this.embeddingModel)
      } else {
      // Every backend maintains the authoritative SQLite embedding cache.
      // Rank eligible rows locally when filtering; global KNN followed by a
      // filter cannot guarantee scoped recall, regardless of over-fetch size.
      const statement = this.db.prepare(`
        SELECT m.rowid, m.embedding FROM memories m
        WHERE m.rowid > @cursor AND m.embedding IS NOT NULL AND m.embedding_model = @model
          AND m.embedding_state = 'ready' ${filter.clause}
        ORDER BY m.rowid LIMIT 256
      `)
      const vector = new Float32Array(queryEmbedding)
      matches = []
      let cursor = 0
      for (;;) {
        const rows = statement.all({ ...filter.params, model: this.embeddingModel, cursor }) as Array<{ rowid: number; embedding: Uint8Array }>
        if (!rows.length) break
        for (const row of rows) {
          const embedding = blobToFloat32Array(row.embedding)
          if (embedding.length !== vector.length) continue
          const distance = cosineDistance(vector, embedding)
          // Orthogonal/opposite vectors provide no positive semantic evidence.
          if (distance < 1) matches.push({ memoryRowid: row.rowid, distance })
        }
        matches.sort((a, b) => a.distance - b.distance)
        matches = matches.slice(0, limit)
        cursor = rows[rows.length - 1].rowid
        if (rows.length < 256) break
        await new Promise<void>((resolve) => setImmediate(resolve))
      }
      }
    } catch (error) {
      this.writeState({
        status: 'degraded',
        lastError: this.toErrorMessage(error),
      })
      return []
    }
    if (matches.length === 0) return []

    const matchValues = matches.map((_match, index) => `(@rowid_${index}, @distance_${index})`).join(', ')
    const params: Record<string, unknown> = { ...filter.params }
    for (const [index, match] of matches.entries()) {
      params[`rowid_${index}`] = match.memoryRowid
      params[`distance_${index}`] = match.distance
    }

    const rows = this.db.prepare(`
      WITH knn_matches(memory_rowid, distance) AS (
        VALUES ${matchValues}
      )
      SELECT m.id, m.content, m.source, m.tags, knn_matches.distance
      FROM knn_matches
      JOIN memories m ON m.rowid = knn_matches.memory_rowid
      WHERE m.embedding_state = 'ready'
      ${filter.clause}
      ORDER BY knn_matches.distance ASC
      LIMIT @limit
    `).all({
      ...params,
      limit,
    }) as Array<{
      id: string
      content: string
      source: MemoryEntry['source']
      tags: string | null
      distance: number
    }>

    return rows
      .map((row) => ({
        id: row.id,
        content: row.content,
        source: row.source,
        tags: this.parseTags(row.tags),
        score: this.normalizeSemanticScore(row.distance),
      }))
      .filter((row) => options?.minScore === undefined || (row.score ?? 0) >= options.minScore)
      .slice(0, limit)
  }

  private mergeHybridResults(
    semantic: MemoryEntry[],
    keyword: MemoryEntry[],
    limit: number,
  ): MemoryEntry[] {
    const merged = new Map<string, MemoryEntry>()
    const scores = new Map<string, number>()

    const applyRanking = (results: MemoryEntry[]) => {
      for (const [index, result] of results.entries()) {
        const score = 1 / (RRF_K + index + 1)
        const existing = merged.get(result.id)
        merged.set(result.id, existing ?? result)
        scores.set(result.id, (scores.get(result.id) ?? 0) + score)
      }
    }

    applyRanking(semantic)
    applyRanking(keyword)

    return Array.from(merged.values())
      .map((result) => ({
        ...result,
        score: scores.get(result.id) ?? result.score ?? 0,
      }))
      .sort((left, right) => (right.score ?? 0) - (left.score ?? 0))
      .slice(0, limit)
  }

  private applyTagFilter(
    results: MemoryEntry[],
    options?: SemanticSearchOptions,
  ): MemoryEntry[] {
    const requireTags = options?.tags
    const excludeTags = options?.excludeTags
    const logic = options?.tagsLogic ?? 'and'
    if (!requireTags?.length && !excludeTags?.length) return results
    return results.filter((result) => {
      if (requireTags?.length) {
        const matched = logic === 'or'
          ? requireTags.some((tag) => result.tags.includes(tag))
          : requireTags.every((tag) => result.tags.includes(tag))
        if (!matched) return false
      }
      if (excludeTags?.length && excludeTags.some((tag) => result.tags.includes(tag))) {
        return false
      }
      return true
    })
  }

  private buildSearchFilter(
    alias: string,
    options?: SemanticSearchOptions,
  ): { clause: string; params: Record<string, unknown> } {
    const memoryFilter = buildMemorySearchFilter(alias, options)
    const conditions: string[] = []
    const params: Record<string, unknown> = { ...memoryFilter.params }

    if (options?.sources?.length) {
      const placeholders = options.sources.map((source, index) => {
        const key = `source_${index}`
        params[key] = source
        return `@${key}`
      })
      conditions.push(`${alias}.source IN (${placeholders.join(', ')})`)
    }

    if (options?.documentId) {
      params.documentId = options.documentId
      conditions.push(`${alias}.document_id = @documentId`)
    }

    if (options?.createdAfter) {
      params.createdAfter = options.createdAfter
      conditions.push(`${alias}.created_at >= @createdAfter`)
    }

    if (options?.createdBefore) {
      params.createdBefore = options.createdBefore
      conditions.push(`${alias}.created_at <= @createdBefore`)
    }

    return {
      clause: `${memoryFilter.clause} ${conditions.length > 0 ? ` AND ${conditions.join(' AND ')}` : ''}`,
      params,
    }
  }

  /** Subscribe to live audit events. Returns an unsubscribe handle.
   *  Subscribers fire AFTER the audit row has been persisted. */
  subscribeAudit(listener: (entry: MemoryAuditEntry) => void): () => void {
    this.auditSubscribers.add(listener)
    return () => {
      this.auditSubscribers.delete(listener)
    }
  }

  private auditSubscribers = new Set<(entry: MemoryAuditEntry) => void>()

  private recordAuditSync(input: MemoryAuditRecordInput): MemoryAuditEntry {
    const id = randomUUID()
    const createdAt = new Date().toISOString()
    const before = input.before ? this.normalizeMemorySnapshot(input.before) : undefined
    const after = input.after ? this.normalizeMemorySnapshot(input.after) : undefined

    this.db.prepare(`
      INSERT INTO memory_audit (
        id,
        memory_id,
        action,
        actor,
        reason,
        before_json,
        after_json,
        created_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      id,
      input.memoryId,
      input.action,
      input.actor,
      input.reason ?? null,
      before ? JSON.stringify(before) : null,
      after ? JSON.stringify(after) : null,
      createdAt,
    )

    const entry: MemoryAuditEntry = {
      id,
      memoryId: input.memoryId,
      action: input.action,
      actor: input.actor,
      reason: input.reason,
      before,
      after,
      createdAt,
    }
    if (this.auditSubscribers.size > 0) {
      // Snapshot subscribers — listeners are free to add/remove themselves.
      for (const listener of Array.from(this.auditSubscribers)) {
        try {
          listener(entry)
        } catch {
          // A failing listener must not block the audit write.
        }
      }
    }
    return entry
  }

  private mapAuditRow(row: MemoryAuditRow): MemoryAuditEntry {
    return {
      id: row.id,
      memoryId: row.memory_id,
      action: row.action,
      actor: row.actor,
      reason: row.reason ?? undefined,
      before: this.parseMemorySnapshot(row.before_json),
      after: this.parseMemorySnapshot(row.after_json),
      createdAt: row.created_at,
    }
  }

  private mapMemorySnapshotRow(row: MemorySnapshotRow): MemoryAuditSnapshot {
    return this.normalizeMemorySnapshot({
      id: row.id,
      content: row.content,
      source: row.source,
      tags: this.parseTags(row.tags),
    })
  }

  private normalizeMemorySnapshot(entry: MemoryAuditSnapshot): MemoryAuditSnapshot {
    const evidence = memoryEvidenceSchema.safeParse(entry.evidence)
    return {
      ...(evidence.success ? { evidence: evidence.data } : {}),
      id: entry.id,
      content: entry.content,
      source: entry.source,
      tags: entry.tags ?? [],
    }
  }

  private parseMemorySnapshot(raw: string | null | undefined): MemoryAuditSnapshot | undefined {
    if (!raw) return undefined
    try {
      const parsed = JSON.parse(raw) as Partial<MemoryAuditSnapshot>
      if (
        typeof parsed.id === 'string'
        && typeof parsed.content === 'string'
        && typeof parsed.source === 'string'
        && Array.isArray(parsed.tags)
      ) {
        return {
          ...(memoryEvidenceSchema.safeParse(parsed.evidence).success ? { evidence: parsed.evidence } : {}),
          id: parsed.id,
          content: parsed.content,
          source: parsed.source as MemoryEntry['source'],
          tags: parsed.tags.filter((tag): tag is string => typeof tag === 'string'),
        }
      }
    } catch {
      return undefined
    }
    return undefined
  }

  private mapDocumentRow(row: DocumentRow): MemoryDocument {
    return {
      id: row.id,
      title: row.title,
      path: row.path ?? undefined,
      mimeType: row.mime_type ?? undefined,
      sourceFileId: row.source_file_id ?? undefined,
      tags: this.parseTags(row.tags),
      chunkCount: row.chunk_count,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    }
  }

  private mapDocumentChunkRow(
    row: DocumentChunkRow,
    score?: number,
    query = '',
  ): MemoryDocumentChunk {
    return enrichDocumentChunk({
      id: row.id,
      content: row.content,
      source: 'document',
      tags: this.parseTags(row.tags),
      score,
      documentId: row.document_id,
      documentTitle: row.document_title,
      documentPath: row.path ?? undefined,
      mimeType: row.mime_type ?? undefined,
      sourceFileId: row.source_file_id ?? undefined,
      chunkIndex: row.chunk_index,
      chunkCount: row.chunk_count,
      startOffset: row.chunk_start,
      endOffset: row.chunk_end,
      chunkTitle: row.chunk_title ?? undefined,
    }, query)
  }

  private mapGraphNodeRow(row: MemoryGraphNodeRow): MemoryGraphNode {
    const kind = this.normalizeGraphNodeKind(row.kind)
    return {
      id: row.id,
      label: row.label,
      kind,
      aliases: this.parseStringList(row.aliases),
      tags: this.parseStringList(row.tags),
      evidenceMemoryIds: this.parseStringList(row.evidence_memory_ids),
      confidence: this.clampGraphConfidence(row.confidence),
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      lastSeenAt: row.last_seen_at,
    }
  }

  private mapGraphEdgeRow(row: MemoryGraphEdgeRow): MemoryGraphEdge {
    return {
      id: row.id,
      fromNodeId: row.from_node_id,
      toNodeId: row.to_node_id,
      relation: row.relation,
      tags: this.parseStringList(row.tags),
      evidenceMemoryIds: this.parseStringList(row.evidence_memory_ids),
      confidence: this.clampGraphConfidence(row.confidence),
      createdAt: row.created_at,
      updatedAt: row.updated_at,
      lastSeenAt: row.last_seen_at,
    }
  }

  private getEvidenceMemories(ids: string[]): MemoryGraphEvidenceEntry[] {
    if (ids.length === 0) return []
    const params: Record<string, unknown> = {}
    const placeholders = ids.map((id, index) => {
      const key = `id_${index}`
      params[key] = id
      return `@${key}`
    }).join(', ')
    const rows = this.db.prepare(`
      SELECT id, content, source, tags
      FROM memories
      WHERE id IN (${placeholders})
    `).all(params) as Array<{
      id: string
      content: string
      source: MemoryEntry['source']
      tags: string | null
    }>
    const byId = new Map(rows.map((row) => [row.id, row]))
    return this.withEvidence(ids.flatMap((id) => {
      const row = byId.get(id)
      if (!row) return []
      return [{
        id: row.id,
        content: row.content,
        source: row.source,
        tags: this.parseTags(row.tags),
      }]
    }))
  }

  private getEvidenceStats(ids: string[]): MemoryGraphEvidenceStats {
    const evidenceIds = this.mergeGraphStringLists(ids)
    if (evidenceIds.length === 0) {
      return { evidenceIds: [], activeIds: [], inactiveIds: [], missingIds: [] }
    }
    const memories = this.getEvidenceMemories(evidenceIds)
    const byId = new Map(memories.map((memory) => [memory.id, memory]))
    const activeIds: string[] = []
    const inactiveIds: string[] = []
    const missingIds: string[] = []

    for (const id of evidenceIds) {
      const memory = byId.get(id)
      if (!memory) {
        missingIds.push(id)
        continue
      }
      if (this.isInactiveEvidenceMemory(memory)) {
        inactiveIds.push(id)
      } else {
        activeIds.push(id)
      }
    }

    return { evidenceIds, activeIds, inactiveIds, missingIds }
  }

  private hasActiveEvidence(ids: string[]): boolean {
    if (ids.length === 0) return false
    return this.getEvidenceStats(ids).activeIds.length > 0
  }

  private isInactiveEvidenceMemory(memory: MemoryGraphEvidenceEntry): boolean {
    return !isEvidenceActive(memory) || memory.tags.some((tag) => {
      const lower = tag.toLowerCase()
      return lower === 'archived'
        || lower.startsWith('archived-by:')
        || lower === 'superseded'
        || lower.startsWith('superseded-by:')
    })
  }

  private buildGraphWikiPageQuality(
    node: MemoryGraphNode,
    relationships: MemoryGraphWikiPage['relationships'],
  ): MemoryGraphWikiPage['quality'] {
    const evidenceStats = this.getEvidenceStats(this.mergeGraphStringLists(
      node.evidenceMemoryIds,
      ...relationships.map((relationship) => relationship.edge.evidenceMemoryIds),
    ))
    const confidences = [node.confidence, ...relationships.map((relationship) => relationship.edge.confidence)]
    const confidence = confidences.reduce((sum, value) => sum + value, 0) / Math.max(1, confidences.length)
    const contradictionCount = relationships.filter((relationship) =>
      this.isContradictionEdge(relationship.edge),
    ).length
    const staleRelationshipCount = relationships.filter((relationship) =>
      this.isGraphEntryStale(relationship.edge.lastSeenAt, GRAPH_STALE_AFTER_DAYS),
    ).length
    const signals: MemoryGraphQualitySignal[] = []

    if (node.confidence < GRAPH_LOW_CONFIDENCE_THRESHOLD) {
      signals.push({
        code: 'low_confidence',
        severity: 'warning',
        subjectType: 'node',
        subjectId: node.id,
        message: `Graph page root "${node.label}" has low confidence ${node.confidence.toFixed(2)}.`,
      })
    }
    if (evidenceStats.activeIds.length === 0) {
      signals.push({
        code: 'thin_evidence',
        severity: 'critical',
        subjectType: 'page',
        subjectId: node.id,
        message: `Graph page "${node.label}" has no active evidence memories.`,
        evidenceMemoryIds: evidenceStats.evidenceIds,
      })
    }
    if (evidenceStats.missingIds.length > 0) {
      signals.push({
        code: 'missing_evidence',
        severity: 'warning',
        subjectType: 'page',
        subjectId: node.id,
        message: `Graph page "${node.label}" references ${evidenceStats.missingIds.length} missing evidence memories.`,
        evidenceMemoryIds: evidenceStats.missingIds,
      })
    }
    if (evidenceStats.inactiveIds.length > 0) {
      signals.push({
        code: 'inactive_evidence',
        severity: 'info',
        subjectType: 'page',
        subjectId: node.id,
        message: `Graph page "${node.label}" cites ${evidenceStats.inactiveIds.length} archived or superseded evidence memories.`,
        evidenceMemoryIds: evidenceStats.inactiveIds,
      })
    }
    if (contradictionCount > 0) {
      signals.push({
        code: 'contradiction',
        severity: 'warning',
        subjectType: 'page',
        subjectId: node.id,
        message: `Graph page "${node.label}" includes ${contradictionCount} contradiction relationship(s).`,
      })
    }
    if (this.isGraphEntryStale(node.lastSeenAt, GRAPH_STALE_AFTER_DAYS)) {
      signals.push({
        code: 'stale',
        severity: 'info',
        subjectType: 'node',
        subjectId: node.id,
        message: `Graph page root "${node.label}" has not been seen for at least ${GRAPH_STALE_AFTER_DAYS} days.`,
      })
    }

    const evidenceScore = evidenceStats.activeIds.length === 0
      ? 0
      : Math.min(1, evidenceStats.activeIds.length / 3)
    const stalePenalty = staleRelationshipCount > 0 ? Math.min(0.2, staleRelationshipCount * 0.04) : 0
    const contradictionPenalty = Math.min(0.25, contradictionCount * 0.08)
    const missingPenalty = Math.min(0.2, evidenceStats.missingIds.length * 0.05)
    const inactivePenalty = Math.min(0.1, evidenceStats.inactiveIds.length * 0.03)
    const score = this.clampGraphConfidence(
      confidence * 0.55
        + evidenceScore * 0.35
        + 0.1
        - stalePenalty
        - contradictionPenalty
        - missingPenalty
        - inactivePenalty,
    )

    return {
      score,
      confidence,
      evidenceCount: evidenceStats.evidenceIds.length,
      activeEvidenceCount: evidenceStats.activeIds.length,
      inactiveEvidenceCount: evidenceStats.inactiveIds.length,
      missingEvidenceCount: evidenceStats.missingIds.length,
      relationshipCount: relationships.length,
      contradictionCount,
      staleRelationshipCount,
      signals,
    }
  }

  private buildGraphRepairProposals(report: MemoryGraphQualityReport): MemoryGraphRepairProposal[] {
    type SignalGroup = {
      subjectType: MemoryGraphRepairProposal['subjectType']
      subjectId?: string
      signals: MemoryGraphQualitySignal[]
    }

    const groups = new Map<string, SignalGroup>()
    for (const signal of report.signals) {
      const key = `${signal.subjectType}:${signal.subjectId ?? signal.message}`
      const existing = groups.get(key)
      if (existing) {
        existing.signals.push(signal)
      } else {
        groups.set(key, {
          subjectType: signal.subjectType,
          subjectId: signal.subjectId,
          signals: [signal],
        })
      }
    }

    return [...groups.values()]
      .map((group, index) => this.buildGraphRepairProposal(group, index))
      .sort((left, right) =>
        graphSeverityRank(right.severity) - graphSeverityRank(left.severity)
        || Number(right.safeToApply) - Number(left.safeToApply)
        || left.id.localeCompare(right.id),
      )
  }

  private buildGraphRepairProposal(
    group: {
      subjectType: MemoryGraphRepairProposal['subjectType']
      subjectId?: string
      signals: MemoryGraphQualitySignal[]
    },
    index: number,
  ): MemoryGraphRepairProposal {
    const codes = this.mergeGraphStringLists(
      group.signals.map((signal) => signal.code),
    ) as MemoryGraphRepairProposal['signalCodes']
    const evidenceMemoryIds = this.mergeGraphStringLists(
      ...group.signals.map((signal) => signal.evidenceMemoryIds ?? []),
    )
    const severity = group.signals
      .map((signal) => signal.severity)
      .sort((left, right) => graphSeverityRank(right) - graphSeverityRank(left))[0] ?? 'info'
    const has = (code: MemoryGraphQualitySignal['code']) => codes.includes(code)
    const safePrune = (group.subjectType === 'edge' && has('thin_evidence'))
      || (group.subjectType === 'node' && has('orphan'))
    const action: MemoryGraphRepairAction = safePrune
      ? 'prune_unbacked_graph_entry'
      : has('contradiction')
        ? 'review_contradiction'
        : has('missing_evidence') || has('inactive_evidence') || has('thin_evidence')
          ? 'relink_or_add_evidence'
          : 'refresh_graph_entry'
    const safeToApply = action === 'prune_unbacked_graph_entry'
    const subject = group.subjectId
      ? `${group.subjectType} ${group.subjectId}`
      : group.subjectType

    return {
      id: `graph-repair:${group.subjectType}:${group.subjectId ?? index}:${codes.join('+')}`,
      action,
      subjectType: group.subjectType,
      subjectId: group.subjectId,
      severity,
      safeToApply,
      reason: graphRepairReason(action, subject, codes),
      signalCodes: codes,
      evidenceMemoryIds,
    }
  }

  private getGraphNodeDegree(nodeId: string): number {
    const row = this.db.prepare(`
      SELECT COUNT(*) AS count
      FROM memory_graph_edges
      WHERE from_node_id = ? OR to_node_id = ?
    `).get(nodeId, nodeId) as { count: number } | undefined
    return row?.count ?? 0
  }

  private isContradictionEdge(edge: MemoryGraphEdge): boolean {
    const relation = edge.relation.toLowerCase()
    return relation === 'contradicts'
      || relation === 'conflicts_with'
      || edge.tags.some((tag) => {
        const lower = tag.toLowerCase()
        return lower === 'contradiction' || lower === 'conflict'
      })
  }

  private isGraphEntryStale(value: string, staleAfterDays: number): boolean {
    const timestamp = Date.parse(value)
    if (!Number.isFinite(timestamp)) return false
    const ageMs = Date.now() - timestamp
    return ageMs > staleAfterDays * 24 * 60 * 60 * 1000
  }

  private memoryGraphNodeId(kind: string, label: string): string {
    return `kg-node:${createHash('sha256')
      .update(`${kind}:${label.toLowerCase()}`)
      .digest('hex')
      .slice(0, 24)}`
  }

  private memoryGraphEdgeId(fromNodeId: string, relation: string, toNodeId: string): string {
    return `kg-edge:${createHash('sha256')
      .update(`${fromNodeId}:${relation}:${toNodeId}`)
      .digest('hex')
      .slice(0, 24)}`
  }

  private normalizeGraphLabel(value: unknown): string {
    if (typeof value !== 'string') return ''
    return value.replace(/\s+/g, ' ').trim().slice(0, 180)
  }

  private normalizeGraphRelation(value: unknown): string {
    if (typeof value !== 'string') return 'related_to'
    const normalized = value
      .trim()
      .toLowerCase()
      .replace(/[\s-]+/g, '_')
      .replace(/[^\p{L}\p{N}_:.]+/gu, '_')
      .replace(/_+/g, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 80)
    return normalized || 'related_to'
  }

  private normalizeGraphNodeKind(value: unknown): MemoryGraphNode['kind'] {
    if (typeof value !== 'string') return 'other'
    const normalized = value.trim().toLowerCase()
    return (MEMORY_GRAPH_NODE_KINDS as readonly string[]).includes(normalized)
      ? normalized as MemoryGraphNode['kind']
      : 'other'
  }

  private normalizeGraphStringList(values: unknown, maxItems: number, maxLength: number): string[] {
    if (!Array.isArray(values)) return []
    const out: string[] = []
    const seen = new Set<string>()
    for (const value of values) {
      if (typeof value !== 'string') continue
      const normalized = value.replace(/\s+/g, ' ').trim().slice(0, maxLength)
      const key = normalized.toLowerCase()
      if (!key || seen.has(key)) continue
      seen.add(key)
      out.push(normalized)
      if (out.length >= maxItems) break
    }
    return out
  }

  private mergeGraphStringLists(...lists: string[][]): string[] {
    const out: string[] = []
    const seen = new Set<string>()
    for (const list of lists) {
      for (const value of list) {
        const normalized = value.replace(/\s+/g, ' ').trim()
        const key = normalized.toLowerCase()
        if (!key || seen.has(key)) continue
        seen.add(key)
        out.push(normalized)
      }
    }
    return out
  }

  private parseStringList(raw: string | null | undefined): string[] {
    try {
      const parsed = JSON.parse(raw || '[]')
      return Array.isArray(parsed)
        ? parsed.filter((value): value is string => typeof value === 'string')
        : []
    } catch {
      return []
    }
  }

  private clampGraphConfidence(value: unknown): number {
    const numeric = typeof value === 'number' ? value : Number(value)
    if (!Number.isFinite(numeric)) return 0.5
    return Math.max(0, Math.min(1, numeric))
  }

  private clampGraphLimit(value: unknown, fallback: number, max = 100): number {
    const numeric = typeof value === 'number' ? value : Number(value)
    if (!Number.isFinite(numeric) || numeric <= 0) return fallback
    return Math.max(1, Math.min(Math.floor(numeric), max))
  }

  private clampGraphQualityThreshold(value: unknown, fallback: number): number {
    const numeric = typeof value === 'number' ? value : Number(value)
    if (!Number.isFinite(numeric)) return fallback
    return Math.max(0, Math.min(1, numeric))
  }

  private clampGraphDays(value: unknown, fallback: number): number {
    const numeric = typeof value === 'number' ? value : Number(value)
    if (!Number.isFinite(numeric) || numeric <= 0) return fallback
    return Math.max(1, Math.min(Math.floor(numeric), 3650))
  }

  private parseTags(raw: string | null | undefined): string[] {
    try {
      const parsed = JSON.parse(raw || '[]')
      return Array.isArray(parsed)
        ? parsed.filter((tag): tag is string => typeof tag === 'string')
        : []
    } catch {
      return []
    }
  }

  private normalizeSemanticScore(distance: unknown): number {
    const value = typeof distance === 'number' ? distance : Number(distance)
    if (!Number.isFinite(value)) return 0
    return 1 / (1 + Math.max(value, 0))
  }

  private normalizeKeywordScore(rank: unknown): number {
    const value = typeof rank === 'number' ? rank : Number(rank)
    if (!Number.isFinite(value)) return 0.25
    return 1 / (1 + Math.abs(value))
  }

  private toErrorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error)
  }
}

function graphSeverityRank(severity: MemoryGraphQualitySignal['severity']): number {
  if (severity === 'critical') return 3
  if (severity === 'warning') return 2
  return 1
}

function graphRepairReason(
  action: MemoryGraphRepairAction,
  subject: string,
  signalCodes: readonly string[],
): string {
  const signalText = signalCodes.join(', ')
  switch (action) {
    case 'prune_unbacked_graph_entry':
      return `Prune ${subject} because graph quality signals show no active backing evidence (${signalText}).`
    case 'review_contradiction':
      return `Review ${subject} with its cited memories before deciding which fact supersedes the other (${signalText}).`
    case 'relink_or_add_evidence':
      return `Relink ${subject} to active evidence or add a fresh memory before trusting this graph fact (${signalText}).`
    case 'refresh_graph_entry':
      return `Refresh ${subject} from recent evidence before relying on it (${signalText}).`
  }
}

function tagsEqual(left: string[], right: string[]): boolean {
  if (left.length !== right.length) return false
  return left.every((tag, index) => tag === right[index])
}
