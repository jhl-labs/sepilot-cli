export interface SemanticSearchOptions {
  scopeTags?: string[]
  asOf?: string
  includeInactive?: boolean
  type?: 'semantic' | 'keyword' | 'hybrid'
  limit?: number
  minScore?: number
  widenWhenEmpty?: boolean
  tags?: string[]
  tagsLogic?: 'and' | 'or'
  excludeTags?: string[]
  sources?: MemoryEntry['source'][]
  documentId?: string
  createdAfter?: string
  createdBefore?: string
  sortBy?: 'score' | 'createdAt' | 'updatedAt'
}

export interface MemoryEvidence {
  subject?: 'user' | 'persona' | 'relationship'
  reality?: 'real' | 'fictional'
  kind: 'semantic' | 'episodic' | 'procedural'
  origin: 'user' | 'observed' | 'inferred'
  observedAt: string
  validFrom?: string
  validUntil?: string
  status: 'active' | 'candidate' | 'retracted'
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

export interface ISubscribableMemoryStore {
  subscribeAudit(listener: (entry: MemoryAuditEvent) => void): () => void
}

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
  updateDocument(id: string, input: DocumentUpdateInput): Promise<MemoryDocument | null>
  listDocumentChunks(documentId: string, limit?: number): Promise<MemoryDocumentChunk[]>
}

export interface MergeMemoriesInput {
  keepId: string
  content: string
  tags?: string[]
  removeIds: string[]
}

export interface MergeMemoriesResult {
  merged: boolean
  removed: string[]
  skippedPinned: string[]
}

export interface IDreamingMemoryStore extends ISemanticIndex {
  listRecent(limit: number): Promise<Array<Omit<MemoryEntry, 'score'>>>
  recalculateImportanceScores(): Promise<number>
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

export interface IAccessTrackingMemoryStore {
  recordAccess(ids: string[]): Promise<void>
  getAccessStats(id: string): Promise<MemoryAccessStats | null>
  listHotMemories(limit: number): Promise<MemoryAccessHotEntry[]>
}

export interface MemoryPinnedEntry extends Omit<MemoryEntry, 'score'> {
  pinnedAt: string | null
}

export interface IPinningMemoryStore {
  pin(id: string): Promise<void>
  unpin(id: string): Promise<void>
  isPinned(id: string): Promise<boolean | null>
  listPinned(limit: number): Promise<MemoryPinnedEntry[]>
}
