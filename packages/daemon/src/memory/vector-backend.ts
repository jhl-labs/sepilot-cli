import { createHash } from 'node:crypto'
import { setTimeout as sleep } from 'node:timers/promises'
import * as sqliteVec from 'sqlite-vec'
import type { SqliteDatabase } from '../db/sqlite.js'
import { createLogger } from '../logger.js'
import type { MemoryVectorBackendKind } from './types.js'

const log = createLogger('memory:vector-backend')

export interface VectorSearchMatch {
  memoryRowid: number
  distance: number
}

export interface QdrantMemoryConfig {
  url: string
  apiKey?: string
  collection?: string
  timeoutMs?: number
}

export interface SearchEngineMemoryConfig {
  url: string
  index?: string
  apiKey?: string
  username?: string
  password?: string
  timeoutMs?: number
}

export interface MeilisearchMemoryConfig {
  url: string
  index?: string
  apiKey?: string
  embedder?: string
  timeoutMs?: number
}

// Wall-clock cap for a single external vector-backend HTTP request. Without it,
// a hung backend (qdrant/opensearch/ES/meili down or slow) makes memory search
// hang forever: the throw never fires, so the keyword fallback never runs and a
// hybrid Promise.all blocks the whole call. Default 15s, env-tunable.
export function resolveVectorBackendTimeoutMs(configured?: number): number {
  if (configured !== undefined && configured > 0) return configured
  const env = Number(process.env.SEPILOTD_VECTOR_BACKEND_TIMEOUT_MS)
  return Number.isFinite(env) && env > 0 ? env : 15_000
}

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { ...init, signal: controller.signal })
  } finally {
    clearTimeout(timer)
  }
}

export interface MemoryHttpAuthConfig {
  type?: 'none' | 'bearer' | 'api-key' | 'basic'
  headerName?: string
  username?: string
  password?: string
}

export interface CustomApiMemoryConfig {
  url: string
  apiKey?: string
  auth?: MemoryHttpAuthConfig
  headers?: Record<string, string>
  healthPath?: string
  configurePath?: string
  upsertPath?: string
  deletePath?: string
  searchPath?: string
  clearPath?: string
  timeoutMs?: number
}

export interface CreateVectorBackendOptions {
  db: SqliteDatabase
  dbPath: string
  requested?: 'auto' | MemoryVectorBackendKind
  qdrant?: QdrantMemoryConfig
  opensearch?: SearchEngineMemoryConfig
  elasticsearch?: SearchEngineMemoryConfig
  meilisearch?: MeilisearchMemoryConfig
  customApi?: CustomApiMemoryConfig
}

export interface MemoryVectorBackend {
  readonly kind: MemoryVectorBackendKind
  readonly available: boolean
  ensureReady(dimensions: number): Promise<void>
  clear(): Promise<void>
  restoreFromCache(model: string, dimensions: number): Promise<void>
  upsert(memoryRowid: number, embedding: Float32Array, model: string): Promise<void>
  delete(memoryRowids: number[]): Promise<void>
  search(queryEmbedding: Float32Array, limit: number, model: string): Promise<VectorSearchMatch[]>
}

function hasHeader(headers: Record<string, string>, name: string): boolean {
  const lowerName = name.toLowerCase()
  return Object.keys(headers).some((key) => key.toLowerCase() === lowerName)
}

function setHeaderIfMissing(
  headers: Record<string, string>,
  name: string,
  value: string,
): void {
  if (!hasHeader(headers, name)) {
    headers[name] = value
  }
}

export function buildMemoryHttpHeaders({
  apiKey,
  auth,
  contentType,
  headers: customHeaders,
}: {
  apiKey?: string
  auth?: MemoryHttpAuthConfig
  contentType?: string
  headers?: Record<string, string>
}): Record<string, string> {
  const headers: Record<string, string> = { ...(customHeaders ?? {}) }
  const authType = auth?.type ?? (apiKey ? 'bearer' : 'none')

  if (authType === 'bearer' && apiKey) {
    setHeaderIfMissing(headers, auth?.headerName ?? 'authorization', `Bearer ${apiKey}`)
  } else if (authType === 'api-key' && apiKey) {
    setHeaderIfMissing(headers, auth?.headerName ?? 'x-api-key', apiKey)
  } else if (authType === 'basic') {
    const username = auth?.username
    const password = auth?.password
    if (username && password) {
      setHeaderIfMissing(
        headers,
        auth?.headerName ?? 'authorization',
        `Basic ${Buffer.from(`${username}:${password}`).toString('base64')}`,
      )
    }
  }

  if (contentType) {
    setHeaderIfMissing(headers, 'content-type', contentType)
  }

  return headers
}

export function cosineDistance(left: Float32Array, right: Float32Array): number {
  let dot = 0
  let leftNorm = 0
  let rightNorm = 0
  const length = Math.min(left.length, right.length)
  for (let index = 0; index < length; index++) {
    dot += left[index] * right[index]
    leftNorm += left[index] * left[index]
    rightNorm += right[index] * right[index]
  }
  if (leftNorm === 0 || rightNorm === 0) return 1
  const similarity = dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm))
  return 1 - similarity
}

export function blobToFloat32Array(value: unknown): Float32Array {
  if (value instanceof Float32Array) return value
  if (value instanceof Uint8Array) {
    return new Float32Array(
      value.buffer.slice(
        value.byteOffset,
        value.byteOffset + value.byteLength,
      ),
    )
  }
  if (value instanceof ArrayBuffer) return new Float32Array(value.slice(0))
  throw new Error('Unsupported embedding blob type')
}

export class SqliteVecBackend implements MemoryVectorBackend {
  readonly kind = 'sqlite-vec' as const
  readonly available: boolean
  private activeDimensions?: number

  constructor(private readonly db: SqliteDatabase) {
    try {
      const override = process.env.SEPILOTD_SQLITE_VEC_PATH?.trim()
      if (override) {
        // Compiled single-file binary: the vec0 loadable lives at a temp path,
        // not under node_modules/sqlite-vec-*. Load it directly.
        this.db.loadExtension(override)
      } else {
        // SqliteDatabase is a better-sqlite3 instance at runtime; sqlite-vec needs the concrete type
        sqliteVec.load(this.db as never)
      }
      this.available = this.probe()
    } catch (err) {
      // Loading the loadable extension failed (missing/garbage path, unsupported
      // platform, or this SQLite build forbids extension loading). Degrade to the
      // same "vector search unavailable" state as when sqlite-vec isn't installed —
      // createVectorBackend() then falls back to the sqlite-scan backend — instead
      // of letting the exception crash daemon startup.
      log.warn('sqlite-vec extension load failed; vector search disabled (falling back to sqlite-scan)', {
        error: err instanceof Error ? err.message : String(err),
        path: process.env.SEPILOTD_SQLITE_VEC_PATH?.trim() || undefined,
      })
      this.available = false
    }
  }

  async ensureReady(dimensions: number): Promise<void> {
    if (!this.available) return
    if (this.activeDimensions === dimensions && this.tableExists()) return

    await this.clear()
    this.db.exec(`
      CREATE VIRTUAL TABLE memories_vec USING vec0(
        memory_rowid INTEGER PRIMARY KEY,
        embedding float[${dimensions}] distance_metric=cosine
      )
    `)
    this.db.exec(`
      CREATE TRIGGER memories_vec_ad AFTER DELETE ON memories BEGIN
        DELETE FROM memories_vec WHERE memory_rowid = old.rowid;
      END
    `)
    this.activeDimensions = dimensions
  }

  async clear(): Promise<void> {
    this.db.exec('DROP TRIGGER IF EXISTS memories_vec_ad')
    this.db.exec('DROP TABLE IF EXISTS memories_vec')
    this.activeDimensions = undefined
  }

  async restoreFromCache(model: string, dimensions: number): Promise<void> {
    if (!this.available) return
    await this.ensureReady(dimensions)
    if (!this.tableExists()) return
    this.db.prepare('DELETE FROM memories_vec').run()
    const rows = this.db.prepare(`
      SELECT rowid, embedding
      FROM memories
      WHERE embedding IS NOT NULL
        AND embedding_model = ?
    `).all(model) as Array<{
      rowid: number
      embedding: Uint8Array
    }>
    const insert = this.db.prepare(`
      INSERT INTO memories_vec (memory_rowid, embedding)
      VALUES (?, ?)
    `)
    for (const row of rows) {
      const embedding = blobToFloat32Array(row.embedding)
      if (embedding.length !== dimensions) continue
      insert.run(BigInt(row.rowid), embedding)
    }
    this.activeDimensions = dimensions
  }

  async upsert(memoryRowid: number, embedding: Float32Array): Promise<void> {
    if (!this.available || !this.tableExists()) return

    const vectorRowid = BigInt(memoryRowid)
    const updated = this.db.prepare(`
      UPDATE memories_vec
      SET embedding = ?
      WHERE memory_rowid = ?
    `).run(embedding, vectorRowid)

    if (updated.changes === 0) {
      this.db.prepare(`
        INSERT INTO memories_vec (memory_rowid, embedding)
        VALUES (?, ?)
      `).run(vectorRowid, embedding)
    }
  }

  async delete(memoryRowids: number[]): Promise<void> {
    if (!this.available || !this.tableExists() || memoryRowids.length === 0) return
    const statement = this.db.prepare('DELETE FROM memories_vec WHERE memory_rowid = ?')
    const tx = this.db.transaction((rowids: number[]) => {
      for (const rowid of rowids) {
        statement.run(BigInt(rowid))
      }
    })
    tx(memoryRowids)
  }

  async search(queryEmbedding: Float32Array, limit: number, model: string): Promise<VectorSearchMatch[]> {
    if (!this.available || !this.tableExists()) {
      return this.searchByScan(queryEmbedding, limit, model)
    }

    try {
      const matches = this.db.prepare(`
        WITH knn_matches AS (
          SELECT memory_rowid, distance
          FROM memories_vec
          WHERE embedding MATCH ?
            AND k = ?
        )
        SELECT memory_rowid, distance
        FROM knn_matches
        ORDER BY distance ASC
      `).all(queryEmbedding, limit) as Array<{
        memory_rowid: number
        distance: number
      }>

      if (matches.length > 0) {
        return matches.map((row) => ({
          memoryRowid: row.memory_rowid,
          distance: row.distance,
        }))
      }
    } catch {
      // Fall back to a deterministic scan when sqlite-vec search is unavailable.
    }

    return this.searchByScan(queryEmbedding, limit, model)
  }

  private tableExists(): boolean {
    const row = this.db.prepare(`
      SELECT 1 AS found
      FROM sqlite_master
      WHERE type = 'table' AND name = 'memories_vec'
    `).get() as { found: number } | undefined
    return Boolean(row)
  }

  private probe(): boolean {
    const probeTable = '__memories_vec_probe__'

    try {
      this.db.exec(`DROP TABLE IF EXISTS ${probeTable}`)
      this.db.exec(`
        CREATE VIRTUAL TABLE ${probeTable} USING vec0(
          memory_rowid INTEGER PRIMARY KEY,
          embedding float[1] distance_metric=cosine
        )
      `)
      this.db.prepare(`
        INSERT INTO ${probeTable} (memory_rowid, embedding)
        VALUES (?, ?)
      `).run(1n, new Float32Array([1]))
      const rows = this.db.prepare(`
        WITH knn_matches AS (
          SELECT memory_rowid, distance
          FROM ${probeTable}
          WHERE embedding MATCH ?
            AND k = ?
        )
        SELECT memory_rowid, distance
        FROM knn_matches
      `).all(new Float32Array([1]), 1) as VectorSearchMatch[]
      return rows.length === 1
    } catch {
      return false
    } finally {
      this.db.exec(`DROP TABLE IF EXISTS ${probeTable}`)
    }
  }

  private searchByScan(queryEmbedding: Float32Array, limit: number, model: string): VectorSearchMatch[] {
    const rows = this.db.prepare(`
      SELECT rowid, embedding
      FROM memories
      WHERE embedding IS NOT NULL
        AND embedding_model = ?
    `).all(model) as Array<{
      rowid: number
      embedding: Uint8Array
    }>

    return rows
      .flatMap((row) => {
        const candidate = blobToFloat32Array(row.embedding)
        if (candidate.length !== queryEmbedding.length) return []
        return [{
          memoryRowid: row.rowid,
          distance: cosineDistance(queryEmbedding, candidate),
        }]
      })
      .sort((left, right) => left.distance - right.distance)
      .slice(0, limit)
  }
}

export class SqliteScanBackend implements MemoryVectorBackend {
  readonly kind = 'sqlite-scan' as const
  readonly available = true

  constructor(private readonly db: SqliteDatabase) {}

  async ensureReady(_dimensions: number): Promise<void> {}

  async clear(): Promise<void> {}

  async restoreFromCache(_model: string, _dimensions: number): Promise<void> {}

  async upsert(_memoryRowid: number, _embedding: Float32Array): Promise<void> {}

  async delete(_memoryRowids: number[]): Promise<void> {}

  async search(queryEmbedding: Float32Array, limit: number, model: string): Promise<VectorSearchMatch[]> {
    const rows = this.db.prepare(`
      SELECT rowid, embedding
      FROM memories
      WHERE embedding IS NOT NULL
        AND embedding_model = ?
    `).all(model) as Array<{
      rowid: number
      embedding: Uint8Array
    }>

    return rows
      .flatMap((row) => {
        const candidate = blobToFloat32Array(row.embedding)
        if (candidate.length !== queryEmbedding.length) return []
        return [{
          memoryRowid: row.rowid,
          distance: cosineDistance(queryEmbedding, candidate),
        }]
      })
      .sort((left, right) => left.distance - right.distance)
      .slice(0, limit)
  }
}

class QdrantBackend implements MemoryVectorBackend {
  readonly kind = 'qdrant' as const
  private availability: boolean
  private activeDimensions?: number
  private readonly collection: string

  constructor(
    private readonly db: SqliteDatabase,
    private readonly dbPath: string,
    private readonly config?: QdrantMemoryConfig,
  ) {
    this.collection = this.resolveCollectionName()
    this.availability = Boolean(config?.url)
  }

  get available(): boolean {
    return this.availability
  }

  async ensureReady(dimensions: number): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }
    if (this.activeDimensions === dimensions) return

    await this.clear()
    await this.request('PUT', `/collections/${encodeURIComponent(this.collection)}`, {
      vectors: {
        size: dimensions,
        distance: 'Cosine',
      },
    })
    this.activeDimensions = dimensions
  }

  async clear(): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }

    await this.request(
      'DELETE',
      `/collections/${encodeURIComponent(this.collection)}`,
      undefined,
      { allow404: true },
    )
    this.activeDimensions = undefined
  }

  async restoreFromCache(model: string, dimensions: number): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }

    await this.clear()
    await this.ensureReady(dimensions)

    const rows = this.db.prepare(`
      SELECT rowid, embedding
      FROM memories
      WHERE embedding IS NOT NULL
        AND embedding_model = ?
    `).all(model) as Array<{
      rowid: number
      embedding: Uint8Array
    }>

    const batchSize = 64
    for (let start = 0; start < rows.length; start += batchSize) {
      const batch = rows.slice(start, start + batchSize)
      const points = batch.flatMap((row) => {
        const embedding = blobToFloat32Array(row.embedding)
        if (embedding.length !== dimensions) return []
        return [{
          id: row.rowid,
          vector: Array.from(embedding),
          payload: { model },
        }]
      })
      if (points.length === 0) continue
      await this.upsertPoints(points)
    }
  }

  async upsert(memoryRowid: number, embedding: Float32Array, model: string): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }
    await this.ensureReady(embedding.length)
    await this.upsertPoints([{
      id: memoryRowid,
      vector: Array.from(embedding),
      payload: { model },
    }])
  }

  async delete(memoryRowids: number[]): Promise<void> {
    if (!this.config?.url || memoryRowids.length === 0) return
    await this.request(
      'POST',
      `/collections/${encodeURIComponent(this.collection)}/points/delete?wait=true`,
      { points: memoryRowids },
      { allow404: true },
    )
  }

  async search(queryEmbedding: Float32Array, limit: number, model: string): Promise<VectorSearchMatch[]> {
    if (!this.config?.url) return []

    const response = await this.request<{
      result?: Array<{
        id: number | string
        score: number
      }>
    }>(
      'POST',
      `/collections/${encodeURIComponent(this.collection)}/points/search`,
      {
        vector: Array.from(queryEmbedding),
        limit,
        with_payload: false,
        with_vector: false,
        filter: {
          must: [
            {
              key: 'model',
              match: { value: model },
            },
          ],
        },
      },
      { allow404: true },
    )

    return (response?.result ?? [])
      .flatMap((match) => {
        const id = typeof match.id === 'number' ? match.id : Number(match.id)
        if (!Number.isFinite(id)) return []
        return [{
          memoryRowid: id,
          distance: Math.max(0, 1 - match.score),
        }]
      })
      .sort((left, right) => left.distance - right.distance)
      .slice(0, limit)
  }

  private async upsertPoints(points: Array<{
    id: number
    vector: number[]
    payload: { model: string }
  }>): Promise<void> {
    await this.request(
      'PUT',
      `/collections/${encodeURIComponent(this.collection)}/points?wait=true`,
      { points },
    )
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    options?: { allow404?: boolean },
  ): Promise<T | undefined> {
    if (!this.config?.url) {
      this.availability = false
      throw new Error('Qdrant backend is not configured')
    }

    const headers: Record<string, string> = {}
    if (body !== undefined) headers['content-type'] = 'application/json'
    if (this.config.apiKey) headers['api-key'] = this.config.apiKey

    let response: Response
    try {
      response = await fetchWithTimeout(
        new URL(path, this.config.url).toString(),
        {
          method,
          headers,
          body: body === undefined ? undefined : JSON.stringify(body),
        },
        resolveVectorBackendTimeoutMs(this.config.timeoutMs),
      )
    } catch (error) {
      this.availability = false
      throw new Error(`Qdrant request failed: ${error instanceof Error ? error.message : String(error)}`)
    }

    if (options?.allow404 && response.status === 404) {
      this.availability = true
      return undefined
    }

    const text = await response.text()
    const payload = text ? JSON.parse(text) as T & { status?: { error?: string } } : undefined
    if (!response.ok) {
      this.availability = false
      const message = payload && typeof payload === 'object' && 'status' in payload
        ? payload.status?.error
        : undefined
      throw new Error(`Qdrant request failed with ${response.status}${message ? `: ${message}` : ''}`)
    }

    this.availability = true
    return payload
  }

  private resolveCollectionName(): string {
    const configured = this.config?.collection?.trim()
    if (configured) return configured

    const suffix = createHash('sha1')
      .update(this.dbPath)
      .digest('hex')
      .slice(0, 12)

    return `sepilotd-memory-${suffix}`
  }
}

interface SearchHitPayload {
  result?: {
    hits?: {
      hits?: Array<{
        _id?: string
        _score?: number
      }>
    }
  }
  hits?: {
    hits?: Array<{
      _id?: string
      _score?: number
    }>
  }
}

abstract class JsonHttpBackendBase {
  protected availability: boolean
  protected readonly timeoutMs: number

  constructor(protected readonly url?: string, timeoutMs?: number) {
    this.availability = Boolean(url)
    this.timeoutMs = resolveVectorBackendTimeoutMs(timeoutMs)
  }

  get available(): boolean {
    return this.availability
  }

  protected async request<T>(
    method: string,
    baseUrl: string,
    path: string,
    body?: unknown,
    options?: {
      allowStatuses?: number[]
      headers?: Record<string, string>
      rawBody?: string
    },
  ): Promise<T | undefined> {
    const headers = { ...(options?.headers ?? {}) }
    const payload = options?.rawBody ?? (body === undefined ? undefined : JSON.stringify(body))
    if (payload !== undefined && !headers['content-type']) {
      headers['content-type'] = options?.rawBody ? 'application/x-ndjson' : 'application/json'
    }

    let response: Response
    try {
      response = await fetchWithTimeout(
        new URL(path, baseUrl).toString(),
        {
          method,
          headers,
          body: payload,
        },
        this.timeoutMs,
      )
    } catch (error) {
      this.availability = false
      throw new Error(`${this.constructor.name} request failed: ${error instanceof Error ? error.message : String(error)}`)
    }

    if (options?.allowStatuses?.includes(response.status)) {
      this.availability = true
      return undefined
    }

    const text = await response.text()
    const parsed = text ? JSON.parse(text) as T & {
      error?: { reason?: string; type?: string }
      message?: string
      status?: { error?: string }
      taskUid?: number
      task?: { uid?: number }
      uid?: number
    } : undefined

    if (!response.ok) {
      this.availability = false
      const message = parsed && typeof parsed === 'object'
        ? ('message' in parsed && typeof parsed.message === 'string'
          ? parsed.message
          : 'error' in parsed && parsed.error && typeof parsed.error === 'object'
            ? parsed.error.reason ?? parsed.error.type
            : 'status' in parsed && parsed.status && typeof parsed.status === 'object'
              ? parsed.status.error
              : undefined)
        : undefined
      throw new Error(`${this.constructor.name} request failed with ${response.status}${message ? `: ${message}` : ''}`)
    }

    this.availability = true
    return parsed
  }
}

abstract class SearchEngineBackendBase extends JsonHttpBackendBase implements MemoryVectorBackend {
  protected activeDimensions?: number
  protected readonly indexName: string

  constructor(
    readonly kind: MemoryVectorBackendKind,
    protected readonly db: SqliteDatabase,
    protected readonly dbPath: string,
    protected readonly config?: SearchEngineMemoryConfig,
  ) {
    super(config?.url, config?.timeoutMs)
    this.indexName = this.resolveIndexName()
  }

  async ensureReady(dimensions: number): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }
    if (this.activeDimensions === dimensions) return

    await this.clear()
    await this.createIndex(dimensions)
    await this.waitForIndexReady()
    this.activeDimensions = dimensions
  }

  async clear(): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }
    await this.request(
      'DELETE',
      this.config.url,
      `/${encodeURIComponent(this.indexName)}`,
      undefined,
      {
        allowStatuses: [404],
        headers: this.headers(),
      },
    )
    this.activeDimensions = undefined
  }

  async restoreFromCache(model: string, dimensions: number): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }

    await this.clear()
    await this.ensureReady(dimensions)
    const rows = this.db.prepare(`
      SELECT rowid, embedding
      FROM memories
      WHERE embedding IS NOT NULL
        AND embedding_model = ?
    `).all(model) as Array<{
      rowid: number
      embedding: Uint8Array
    }>

    for (const row of rows) {
      const embedding = blobToFloat32Array(row.embedding)
      if (embedding.length !== dimensions) continue
      await this.upsert(row.rowid, embedding, model)
    }
  }

  async upsert(memoryRowid: number, embedding: Float32Array, model: string): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }
    await this.ensureReady(embedding.length)
    await this.indexDocument(memoryRowid, embedding, model)
  }

  async delete(memoryRowids: number[]): Promise<void> {
    if (!this.config?.url || memoryRowids.length === 0) return
    for (const rowid of memoryRowids) {
      await this.deleteDocument(rowid)
    }
  }

  async search(queryEmbedding: Float32Array, limit: number, model: string): Promise<VectorSearchMatch[]> {
    if (!this.config?.url) return []
    const hits = await this.searchDocuments(queryEmbedding, limit, model)
    return hits
      .flatMap((hit, index) => {
        const id = hit._id === undefined ? Number.NaN : Number(hit._id)
        if (!Number.isFinite(id)) return []
        return [{
          memoryRowid: id,
          distance: hit._score && Number.isFinite(hit._score)
            ? Math.max(0, (1 / Math.max(hit._score, 0.000001)) - 1)
            : index,
        }]
      })
      .sort((left, right) => left.distance - right.distance)
      .slice(0, limit)
  }

  protected headers(): Record<string, string> {
    const headers: Record<string, string> = {}
    if (this.config?.apiKey) {
      headers.authorization = this.apiKeyHeaderValue(this.config.apiKey)
    } else if (this.config?.username && this.config?.password) {
      headers.authorization = `Basic ${Buffer.from(`${this.config.username}:${this.config.password}`).toString('base64')}`
    }
    return headers
  }

  protected resolveIndexName(prefix: string = 'sepilotd-memory'): string {
    const configured = this.config?.index?.trim().toLowerCase()
    if (configured) return configured

    const suffix = createHash('sha1')
      .update(this.dbPath)
      .digest('hex')
      .slice(0, 12)

    return `${prefix}-${suffix}`
  }

  protected abstract apiKeyHeaderValue(apiKey: string): string
  protected abstract createIndex(dimensions: number): Promise<void>
  protected async indexDocument(memoryRowid: number, embedding: Float32Array, model: string): Promise<void> {
    if (!this.config?.url) return
    await this.request(
      'PUT',
      this.config.url,
      `/${encodeURIComponent(this.indexName)}/_doc/${memoryRowid}?refresh=true`,
      {
        model,
        embedding: Array.from(embedding),
      },
      { headers: this.headers() },
    )
  }

  protected async deleteDocument(memoryRowid: number): Promise<void> {
    if (!this.config?.url) return
    await this.request(
      'DELETE',
      this.config.url,
      `/${encodeURIComponent(this.indexName)}/_doc/${memoryRowid}?refresh=true`,
      undefined,
      { allowStatuses: [404], headers: this.headers() },
    )
  }

  protected async waitForIndexReady(): Promise<void> {}
  protected abstract searchDocuments(
    queryEmbedding: Float32Array,
    limit: number,
    model: string,
  ): Promise<Array<{ _id?: string; _score?: number }>>
}

class OpenSearchBackend extends SearchEngineBackendBase {
  constructor(
    db: SqliteDatabase,
    dbPath: string,
    config?: SearchEngineMemoryConfig,
  ) {
    super('opensearch', db, dbPath, config)
  }

  protected apiKeyHeaderValue(apiKey: string): string {
    return `Bearer ${apiKey}`
  }

  protected async createIndex(dimensions: number): Promise<void> {
    if (!this.config?.url) return
    await this.request(
      'PUT',
      this.config.url,
      `/${encodeURIComponent(this.indexName)}`,
      {
        settings: {
          index: {
            knn: true,
          },
        },
        mappings: {
          properties: {
            model: { type: 'keyword' },
            embedding: {
              type: 'knn_vector',
              dimension: dimensions,
              space_type: 'cosinesimil',
            },
          },
        },
      },
      { headers: this.headers() },
    )
  }

  protected async searchDocuments(
    queryEmbedding: Float32Array,
    limit: number,
    model: string,
  ): Promise<Array<{ _id?: string; _score?: number }>> {
    if (!this.config?.url) return []
    const response = await this.request<SearchHitPayload>(
      'POST',
      this.config.url,
      `/${encodeURIComponent(this.indexName)}/_search`,
      {
        size: limit,
        _source: false,
        query: {
          knn: {
            embedding: {
              vector: Array.from(queryEmbedding),
              k: limit,
              filter: {
                term: {
                  model: {
                    value: model,
                  },
                },
              },
            },
          },
        },
      },
      { allowStatuses: [404], headers: this.headers() },
    )

    return response?.hits?.hits ?? response?.result?.hits?.hits ?? []
  }
}

class ElasticsearchBackend extends SearchEngineBackendBase {
  constructor(
    db: SqliteDatabase,
    dbPath: string,
    config?: SearchEngineMemoryConfig,
  ) {
    super('elasticsearch', db, dbPath, config)
  }

  protected apiKeyHeaderValue(apiKey: string): string {
    return `ApiKey ${apiKey}`
  }

  protected async createIndex(dimensions: number): Promise<void> {
    if (!this.config?.url) return
    await this.request(
      'PUT',
      this.config.url,
      `/${encodeURIComponent(this.indexName)}`,
      {
        mappings: {
          properties: {
            model: { type: 'keyword' },
            embedding: {
              type: 'dense_vector',
              dims: dimensions,
              similarity: 'cosine',
            },
          },
        },
      },
      { headers: this.headers() },
    )
  }

  protected override async waitForIndexReady(): Promise<void> {
    if (!this.config?.url) return
    const response = await this.request<{
      status?: string
      timed_out?: boolean
    }>(
      'GET',
      this.config.url,
      `/_cluster/health/${encodeURIComponent(this.indexName)}?wait_for_status=yellow&timeout=120s`,
      undefined,
      { headers: this.headers() },
    )

    if (response?.timed_out || (response?.status !== 'yellow' && response?.status !== 'green')) {
      throw new Error(`Elasticsearch index ${this.indexName} did not become ready`)
    }
  }

  protected async searchDocuments(
    queryEmbedding: Float32Array,
    limit: number,
    model: string,
  ): Promise<Array<{ _id?: string; _score?: number }>> {
    if (!this.config?.url) return []
    const response = await this.request<SearchHitPayload>(
      'POST',
      this.config.url,
      `/${encodeURIComponent(this.indexName)}/_search`,
      {
        size: limit,
        _source: false,
        knn: {
          field: 'embedding',
          query_vector: Array.from(queryEmbedding),
          k: limit,
          num_candidates: Math.max(limit * 4, 50),
          filter: {
            term: {
              model,
            },
          },
        },
      },
      { allowStatuses: [404], headers: this.headers() },
    )

    return response?.hits?.hits ?? response?.result?.hits?.hits ?? []
  }
}

class MeilisearchBackend extends JsonHttpBackendBase implements MemoryVectorBackend {
  readonly kind = 'meilisearch' as const
  private activeDimensions?: number
  private readonly indexName: string
  private readonly embedderName: string

  constructor(
    private readonly db: SqliteDatabase,
    private readonly dbPath: string,
    private readonly config?: MeilisearchMemoryConfig,
  ) {
    super(config?.url, config?.timeoutMs)
    this.indexName = this.resolveIndexName()
    this.embedderName = config?.embedder?.trim() || 'sepilotd_memory'
  }

  async ensureReady(dimensions: number): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }
    if (this.activeDimensions === dimensions) return

    await this.clear()
    await this.createIndex()
    await this.updateSettings(dimensions)
    this.activeDimensions = dimensions
  }

  async clear(): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }

    await this.request(
      'DELETE',
      this.config.url,
      `/indexes/${encodeURIComponent(this.indexName)}`,
      undefined,
      {
        allowStatuses: [404],
        headers: this.headers(),
      },
    )
    this.activeDimensions = undefined
  }

  async restoreFromCache(model: string, dimensions: number): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }

    await this.clear()
    await this.ensureReady(dimensions)
    const rows = this.db.prepare(`
      SELECT rowid, embedding
      FROM memories
      WHERE embedding IS NOT NULL
        AND embedding_model = ?
    `).all(model) as Array<{
      rowid: number
      embedding: Uint8Array
    }>

    const batchSize = 64
    for (let start = 0; start < rows.length; start += batchSize) {
      const batch = rows.slice(start, start + batchSize)
      const documents = batch.flatMap((row) => {
        const embedding = blobToFloat32Array(row.embedding)
        if (embedding.length !== dimensions) return []
        return [{
          id: row.rowid,
          model,
          _vectors: {
            [this.embedderName]: Array.from(embedding),
          },
        }]
      })
      if (documents.length === 0) continue
      await this.addDocuments(documents)
    }
  }

  async upsert(memoryRowid: number, embedding: Float32Array, model: string): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }
    await this.ensureReady(embedding.length)
    await this.addDocuments([{
      id: memoryRowid,
      model,
      _vectors: {
        [this.embedderName]: Array.from(embedding),
      },
    }])
  }

  async delete(memoryRowids: number[]): Promise<void> {
    if (!this.config?.url || memoryRowids.length === 0) return
    const task = await this.request<{ taskUid?: number }>(
      'POST',
      this.config.url,
      `/indexes/${encodeURIComponent(this.indexName)}/documents/delete-batch`,
      memoryRowids,
      {
        allowStatuses: [404],
        headers: this.headers(),
      },
    )
    if (task?.taskUid) await this.waitForTask(task.taskUid)
  }

  async search(queryEmbedding: Float32Array, limit: number, model: string): Promise<VectorSearchMatch[]> {
    if (!this.config?.url) return []
    const response = await this.request<{
      hits?: Array<{
        id: number | string
      }>
    }>(
      'POST',
      this.config.url,
      `/indexes/${encodeURIComponent(this.indexName)}/search`,
      {
        limit,
        vector: Array.from(queryEmbedding),
        filter: `model = "${model.replace(/"/g, '\\"')}"`,
        hybrid: {
          embedder: this.embedderName,
          semanticRatio: 1,
        },
      },
      {
        allowStatuses: [404],
        headers: this.headers(),
      },
    )

    return (response?.hits ?? [])
      .flatMap((hit, index) => {
        const id = typeof hit.id === 'number' ? hit.id : Number(hit.id)
        if (!Number.isFinite(id)) return []
        return [{
          memoryRowid: id,
          distance: index,
        }]
      })
      .slice(0, limit)
  }

  private async createIndex(): Promise<void> {
    if (!this.config?.url) return
    const task = await this.request<{ taskUid?: number }>(
      'POST',
      this.config.url,
      '/indexes',
      {
        uid: this.indexName,
        primaryKey: 'id',
      },
      {
        allowStatuses: [409],
        headers: this.headers(),
      },
    )
    if (task?.taskUid) await this.waitForTask(task.taskUid)
  }

  private async updateSettings(dimensions: number): Promise<void> {
    if (!this.config?.url) return
    const task = await this.request<{ taskUid?: number }>(
      'PATCH',
      this.config.url,
      `/indexes/${encodeURIComponent(this.indexName)}/settings`,
      {
        filterableAttributes: ['model'],
        embedders: {
          [this.embedderName]: {
            source: 'userProvided',
            dimensions,
          },
        },
      },
      { headers: this.headers() },
    )
    if (task?.taskUid) await this.waitForTask(task.taskUid)
  }

  private async addDocuments(documents: Array<Record<string, unknown>>): Promise<void> {
    if (!this.config?.url) return
    const task = await this.request<{ taskUid?: number }>(
      'POST',
      this.config.url,
      `/indexes/${encodeURIComponent(this.indexName)}/documents`,
      documents,
      { headers: this.headers() },
    )
    if (task?.taskUid) await this.waitForTask(task.taskUid)
  }

  private async waitForTask(taskUid: number): Promise<void> {
    if (!this.config?.url) return
    for (let attempt = 0; attempt < 60; attempt++) {
      const task = await this.request<{
        status?: string
        error?: { message?: string; code?: string }
      }>(
        'GET',
        this.config.url,
        `/tasks/${taskUid}`,
        undefined,
        { headers: this.headers() },
      )
      if (task?.status === 'succeeded') return
      if (task?.status === 'failed') {
        throw new Error(`Meilisearch task ${taskUid} failed${task.error?.message ? `: ${task.error.message}` : ''}`)
      }
      await sleep(25)
    }
    throw new Error(`Meilisearch task ${taskUid} timed out`)
  }

  private headers(): Record<string, string> {
    return this.config?.apiKey
      ? { authorization: `Bearer ${this.config.apiKey}` }
      : {}
  }

  private resolveIndexName(): string {
    const configured = this.config?.index?.trim().toLowerCase()
    if (configured) return configured

    const suffix = createHash('sha1')
      .update(this.dbPath)
      .digest('hex')
      .slice(0, 12)

    return `sepilotd-memory-${suffix}`
  }
}

class CustomApiBackend extends JsonHttpBackendBase implements MemoryVectorBackend {
  readonly kind = 'custom-api' as const

  constructor(
    private readonly db: SqliteDatabase,
    private readonly config?: CustomApiMemoryConfig,
  ) {
    super(config?.url)
  }

  async ensureReady(dimensions: number): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }
    if (!this.config.configurePath) return
    await this.requestJson('POST', this.config.configurePath, { dimensions })
  }

  async clear(): Promise<void> {
    if (!this.config?.url || !this.config.clearPath) return
    await this.requestJson('POST', this.config.clearPath, {})
  }

  async restoreFromCache(model: string, dimensions: number): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }
    await this.ensureReady(dimensions)
    const rows = this.db.prepare(`
      SELECT rowid, embedding
      FROM memories
      WHERE embedding IS NOT NULL
        AND embedding_model = ?
    `).all(model) as Array<{
      rowid: number
      embedding: Uint8Array
    }>

    for (const row of rows) {
      const embedding = blobToFloat32Array(row.embedding)
      if (embedding.length !== dimensions) continue
      await this.upsert(row.rowid, embedding, model)
    }
  }

  async upsert(memoryRowid: number, embedding: Float32Array, model: string): Promise<void> {
    if (!this.config?.url) {
      this.availability = false
      return
    }
    await this.ensureReady(embedding.length)
    const metadata = this.readMemoryMetadata(memoryRowid)
    await this.requestJson('POST', this.config.upsertPath ?? '/vectors/upsert', {
      id: memoryRowid,
      memoryRowid,
      vector: Array.from(embedding),
      model,
      metadata,
    })
  }

  async delete(memoryRowids: number[]): Promise<void> {
    if (!this.config?.url || memoryRowids.length === 0) return
    await this.requestJson('POST', this.config.deletePath ?? '/vectors/delete', {
      ids: memoryRowids,
      memoryRowids,
    })
  }

  async search(queryEmbedding: Float32Array, limit: number, model: string): Promise<VectorSearchMatch[]> {
    if (!this.config?.url) return []
    const response = await this.requestJson<{
      matches?: unknown[]
      results?: unknown[]
      hits?: unknown[]
    }>('POST', this.config.searchPath ?? '/vectors/search', {
      vector: Array.from(queryEmbedding),
      model,
      limit,
    })
    const rawHits = response?.matches ?? response?.results ?? response?.hits ?? []
    return rawHits
      .flatMap((hit, index) => {
        if (!hit || typeof hit !== 'object') return []
        const record = hit as Record<string, unknown>
        const rawId = record.memoryRowid ?? record.rowid ?? record.id
        const id = typeof rawId === 'number' ? rawId : Number(rawId)
        if (!Number.isFinite(id)) return []
        const score = typeof record.score === 'number' ? record.score : undefined
        const distance = typeof record.distance === 'number'
          ? record.distance
          : score === undefined
            ? index
            : Math.max(0, 1 - score)
        return [{
          memoryRowid: id,
          distance,
        }]
      })
      .sort((left, right) => left.distance - right.distance)
      .slice(0, limit)
  }

  private readMemoryMetadata(memoryRowid: number): Record<string, unknown> | null {
    const row = this.db.prepare(`
      SELECT
        m.id,
        m.content,
        m.source,
        m.tags,
        m.document_id,
        m.chunk_index,
        m.chunk_title,
        d.title AS document_title,
        d.path AS document_path,
        d.source_file_id
      FROM memories m
      LEFT JOIN memory_documents d ON d.id = m.document_id
      WHERE m.rowid = ?
    `).get(memoryRowid) as
      | {
        id: string
        content: string
        source: string
        tags: string | null
        document_id: string | null
        chunk_index: number | null
        chunk_title: string | null
        document_title: string | null
        document_path: string | null
        source_file_id: string | null
      }
      | undefined
    if (!row) return null
    return {
      memoryId: row.id,
      content: row.content,
      source: row.source,
      tags: this.parseTags(row.tags),
      documentId: row.document_id,
      chunkIndex: row.chunk_index,
      chunkTitle: row.chunk_title,
      documentTitle: row.document_title,
      documentPath: row.document_path,
      sourceFileId: row.source_file_id,
    }
  }

  private parseTags(raw: string | null): string[] {
    if (!raw) return []
    try {
      const parsed = JSON.parse(raw) as unknown
      return Array.isArray(parsed)
        ? parsed.filter((item): item is string => typeof item === 'string')
        : []
    } catch {
      return []
    }
  }

  private async requestJson<T = unknown>(
    method: string,
    path: string,
    body: unknown,
  ): Promise<T | undefined> {
    if (!this.config?.url) {
      this.availability = false
      throw new Error('Custom API vector backend is not configured')
    }
    const headers = buildMemoryHttpHeaders({
      apiKey: this.config.apiKey,
      auth: this.config.auth,
      contentType: 'application/json',
      headers: this.config.headers,
    })
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.config.timeoutMs ?? 15_000)
    try {
      const response = await fetch(new URL(path, this.config.url).toString(), {
        method,
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      })
      const text = await response.text()
      const parsed = text ? JSON.parse(text) as T & { message?: string; error?: unknown } : undefined
      if (!response.ok) {
        this.availability = false
        const message = parsed && typeof parsed === 'object' && 'message' in parsed
          ? parsed.message
          : undefined
        throw new Error(`Custom API vector request failed with ${response.status}${message ? `: ${message}` : ''}`)
      }
      this.availability = true
      return parsed
    } catch (error) {
      this.availability = false
      throw new Error(`Custom API vector request failed: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      clearTimeout(timer)
    }
  }
}

export function createVectorBackend(
  options: CreateVectorBackendOptions,
): MemoryVectorBackend {
  const requested = options.requested ?? 'auto'

  if (requested === 'sqlite-vec') {
    return new SqliteVecBackend(options.db)
  }

  if (requested === 'sqlite-scan') {
    return new SqliteScanBackend(options.db)
  }

  if (requested === 'qdrant') {
    return new QdrantBackend(options.db, options.dbPath, options.qdrant)
  }

  if (requested === 'opensearch') {
    return new OpenSearchBackend(options.db, options.dbPath, options.opensearch)
  }

  if (requested === 'elasticsearch') {
    return new ElasticsearchBackend(options.db, options.dbPath, options.elasticsearch)
  }

  if (requested === 'meilisearch') {
    return new MeilisearchBackend(options.db, options.dbPath, options.meilisearch)
  }

  if (requested === 'custom-api') {
    return new CustomApiBackend(options.db, options.customApi)
  }

  const vecBackend = new SqliteVecBackend(options.db)
  if (vecBackend.available) return vecBackend
  return new SqliteScanBackend(options.db)
}
