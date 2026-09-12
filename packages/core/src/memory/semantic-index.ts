export interface SemanticSearchOptions {
  /** Server-derived ownership ACL; undefined is an administrative unfiltered read. */
  scopeTags?: string[]
  /** Temporal query instant; defaults to now. */
  asOf?: string
  /** Explicit historical inspection, including retracted or expired facts. */
  includeInactive?: boolean
  type?: 'semantic' | 'keyword' | 'hybrid'
  limit?: number
  minScore?: number
  /**
   * Retry the keyword pass with looser matching (term prefixes, then any term
   * instead of all) — set internally for the retry a search makes when nothing
   * matched at all, so a stored memory is not reported as missing. Trades
   * precision for recall, so it is never applied to a search that already
   * returned something.
   */
  widenWhenEmpty?: boolean
  /** Required tags. With the default `tagsLogic: 'and'` a memory must
   *  carry every tag in this array; with `'or'` it must carry at least
   *  one. Match is exact and case-sensitive either way. */
  tags?: string[]
  /** How `tags` is combined. 'and' (default) requires every tag;
   *  'or' requires at least one. */
  tagsLogic?: 'and' | 'or'
  /** Forbidden tags. A memory carrying any of these is filtered out
   *  (NOT-ANY semantics). Example: drop archived/superseded entries
   *  via excludeTags: ['archived']. Applied after `tags`. */
  excludeTags?: string[]
  sources?: MemoryEntry['source'][]
  documentId?: string
  /** Inclusive lower bound on created_at. ISO-8601 string. */
  createdAfter?: string
  /** Inclusive upper bound on created_at. ISO-8601 string. */
  createdBefore?: string
  /** Result ordering. 'score' (default) keeps the search backend's
   *  similarity ranking. 'createdAt' / 'updatedAt' overrides it with a
   *  chronological sort (descending: newest first). */
  sortBy?: 'score' | 'createdAt' | 'updatedAt'
}

export interface MemoryEvidence {
  /** Explicit claim classification; omitted for unclassified legacy memories. */
  subject?: 'user' | 'persona' | 'relationship'
  reality?: 'real' | 'fictional'
  kind: 'semantic' | 'episodic' | 'procedural'
  origin: 'user' | 'observed' | 'inferred'
  observedAt: string
  validFrom?: string
  validUntil?: string
  status: 'active' | 'candidate' | 'retracted'
  /** Original event or memory ids; retained across corrections. */
  sourceIds: string[]
}

export interface MemoryEntry {
  id: string
  content: string
  source: 'conversation' | 'document' | 'skill' | 'user'
  tags: string[]
  score?: number
  evidence?: MemoryEvidence
  createdAt?: string
  updatedAt?: string
}

export interface MemoryDocument {
  id: string
  title: string
  path?: string
  mimeType?: string
  sourceFileId?: string
  tags: string[]
  chunkCount: number
  createdAt: string
  updatedAt: string
}

export interface MemoryDocumentChunk extends MemoryEntry {
  documentId: string
  documentTitle: string
  documentPath?: string
  mimeType?: string
  sourceFileId?: string
  chunkIndex: number
  chunkCount: number
  startOffset: number
  endOffset: number
  chunkTitle?: string
  snippet?: string
  citationLabel?: string
}

export interface MemoryContextItem {
  id: string
  kind: 'memory' | 'document'
  source: MemoryEntry['source']
  title: string
  snippet: string
  citationLabel: string
  score?: number
  documentId?: string
  documentTitle?: string
  documentPath?: string
}

export interface DocumentIngestInput {
  id?: string
  title: string
  content: string
  path?: string
  mimeType?: string
  sourceFileId?: string
  tags?: string[]
}

export interface DocumentSearchOptions extends SemanticSearchOptions {
  documentId?: string
}

export interface DocumentListOptions {
  query?: string
  limit?: number
}

export interface ISemanticIndex {
  getMemoryResetAt?(scopeTags: string[]): Promise<string | null>
  getRetractedSourceIds?(scopeTags: string[]): Promise<string[]>
  add(entry: Omit<MemoryEntry, 'score'>): Promise<void>
  get(id: string): Promise<MemoryEntry | null>
  search(query: string, options?: SemanticSearchOptions): Promise<MemoryEntry[]>
  delete(id: string): Promise<void>
}

export interface MemoryAuditEvent {
  id: string
  memoryId: string
  action: 'created' | 'updated' | 'deleted' | 'pruned' | 'maintenance'
  actor: string
  reason?: string
  before?: Omit<MemoryEntry, 'score'>
  after?: Omit<MemoryEntry, 'score'>
  createdAt: string
}

/** Implementations may emit audit events via subscribeAudit so dashboards
 *  and SSE routes can listen live. Optional: not every store needs to
 *  expose this surface. */
export interface ISubscribableMemoryStore {
  subscribeAudit(listener: (entry: MemoryAuditEvent) => void): () => void
}

/** Surface that memory.export and admin tooling rely on: list recent
 *  entries with their created/updated timestamps so callers can window
 *  by time without an extra round-trip. */
export interface IListRecentWithTimestampsStore {
  listRecentWithTimestamps(
    limit: number,
    options?: { createdAfter?: string; createdBefore?: string },
  ): Promise<Array<Omit<MemoryEntry, 'score'> & { createdAt: string; updatedAt: string }>>
}

export interface DocumentUpdateInput {
  title?: string
  path?: string
  mimeType?: string
  tags?: string[]
}

export interface IDocumentMemoryStore {
  ingestDocument(input: DocumentIngestInput): Promise<MemoryDocument>
  listDocuments(options?: DocumentListOptions): Promise<MemoryDocument[]>
  getDocument(id: string): Promise<MemoryDocument | null>
  searchDocuments(query: string, options?: DocumentSearchOptions): Promise<MemoryDocumentChunk[]>
  deleteDocument(id: string): Promise<boolean>
  /** Update document metadata (title/path/mimeType/tags). Chunks and
   *  embeddings are preserved. Returns the updated document, or null if
   *  the id doesn't exist. */
  updateDocument(id: string, input: DocumentUpdateInput): Promise<MemoryDocument | null>
  /** Return the first `limit` chunks of a document in order, without
   *  hitting the search backend. Used for previews / debugging. */
  listDocumentChunks(documentId: string, limit?: number): Promise<MemoryDocumentChunk[]>
}

export interface MergeMemoriesInput {
  /** The memory to keep; its pin/importance/access/created_at are preserved. */
  keepId: string
  /** New (merged) content for the kept memory. */
  content: string
  /** Optional replacement tags for the kept memory (unchanged when omitted). */
  tags?: string[]
  /** Duplicate memories to fold in and delete. Pinned ids are refused. */
  removeIds: string[]
}

export interface MergeMemoriesResult {
  /** True when the kept memory existed and was updated. */
  merged: boolean
  /** Ids that were actually deleted. */
  removed: string[]
  /** removeIds that were pinned and therefore NOT deleted. */
  skippedPinned: string[]
}

export interface IDreamingMemoryStore extends ISemanticIndex {
  listRecent(limit: number): Promise<Array<Omit<MemoryEntry, 'score'>>>
  recalculateImportanceScores(): Promise<number>
  /**
   * Fold duplicate memories into `keepId` inside a single transaction: update
   * the kept memory's content (preserving its pin/importance/access/created_at)
   * and delete the non-pinned duplicates atomically. A crash mid-merge leaves
   * the kept memory intact rather than deleting it before the merged row is
   * written. Pinned duplicates are never hard-deleted.
   */
  mergeMemories(input: MergeMemoriesInput): Promise<MergeMemoriesResult>
  pruneStaleConversationMemories(options?: {
    maxAgeDays?: number
    maxImportance?: number
  }): Promise<number>
}

export interface MemoryAccessStats {
  accessCount: number
  lastAccessedAt: string | null
}

export interface MemoryAccessHotEntry extends Omit<MemoryEntry, 'score'> {
  accessCount: number
  lastAccessedAt: string | null
}

/**
 * Access-counter surface. Tools and the channel pipeline call
 * recordAccess(ids) after a retrieval surfaced a memory to either the
 * agent or a downstream user. The counter is used by dreaming's
 * importance-recalculation phase to bias retention toward memories the
 * agent actually keeps reaching for, and by the /memory/hot route +
 * memory.access.hot tool to surface the top-N most-used memories.
 *
 * Implementations should be best-effort and idempotent: callers fire
 * recordAccess from background or non-critical paths and do not await
 * its success. Missing ids must be silently ignored, never throw.
 */
export interface IAccessTrackingMemoryStore {
  recordAccess(ids: string[]): Promise<void>
  getAccessStats(id: string): Promise<MemoryAccessStats | null>
  listHotMemories(limit: number): Promise<MemoryAccessHotEntry[]>
}

export interface MemoryPinnedEntry extends Omit<MemoryEntry, 'score'> {
  pinnedAt: string | null
}

/**
 * Pin a memory to mark it prune-immune. The dreaming pipeline's stale
 * cleanup phase skips pinned rows regardless of age, importance, or
 * access count. Used by `/memory_pin` and the `memory.pin` tool — the
 * agent or user pins something they explicitly never want forgotten
 * (an SOP, a critical preference, a long-running ticket reference).
 *
 * Pin/unpin are idempotent and silent on missing ids (consistent with
 * recordAccess). isPinned returns null if the id doesn't exist.
 */
export interface IPinningMemoryStore {
  pin(id: string): Promise<void>
  unpin(id: string): Promise<void>
  isPinned(id: string): Promise<boolean | null>
  listPinned(limit: number): Promise<MemoryPinnedEntry[]>
}
