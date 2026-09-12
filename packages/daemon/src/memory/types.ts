import type {
  IAccessTrackingMemoryStore,
  IDocumentMemoryStore,
  IDreamingMemoryStore,
  IListRecentWithTimestampsStore,
  IPinningMemoryStore,
  ISubscribableMemoryStore,
  MemoryEntry,
} from '@sepilotd/core'

export type MemoryVectorBackendKind =
  | 'sqlite-vec'
  | 'sqlite-scan'
  | 'qdrant'
  | 'opensearch'
  | 'elasticsearch'
  | 'meilisearch'
  | 'custom-api'

export type MemorySemanticStatus =
  | 'disabled'
  | 'ready'
  | 'backfilling'
  | 'degraded'
  | 'reindex_required'

export interface SemanticIndexRuntimeStatus {
  status: MemorySemanticStatus
  configuredProviderId?: string
  configuredModel?: string
  indexedProviderId?: string
  indexedModel?: string
  dimensions?: number
  pendingCount: number
  failedCount: number
  vecAvailable: boolean
  vectorBackend: MemoryVectorBackendKind
  backendAvailable: boolean
  lastError?: string
}

export type MemoryAuditAction =
  | 'created'
  | 'updated'
  | 'deleted'
  | 'pruned'
  | 'maintenance'

export type MemoryAuditSnapshot = Omit<MemoryEntry, 'score'>

export interface MemoryAuditEntry {
  id: string
  memoryId: string
  action: MemoryAuditAction
  actor: string
  reason?: string
  before?: MemoryAuditSnapshot
  after?: MemoryAuditSnapshot
  createdAt: string
}

export interface MemoryAuditListOptions {
  memoryId?: string
  limit?: number
}

export interface MemoryLifecycleOptions {
  staleAfterDays?: number
  lowImportance?: number
}

export interface MemoryLifecycleStatus {
  totalMemories: number
  conversationMemories: number
  documentMemories: number
  skillMemories: number
  userMemories: number
  staleConversationMemories: number
  lowImportanceConversationMemories: number
  pruneCandidateMemories: number
  pendingEmbeddings: number
  failedEmbeddings: number
  lastAuditAt?: string
}

export interface MemoryAuditRecordInput {
  memoryId: string
  action: MemoryAuditAction
  actor: string
  reason?: string
  before?: MemoryAuditSnapshot
  after?: MemoryAuditSnapshot
}

export interface MemoryMaintenanceOptions extends MemoryLifecycleOptions {
  dryRun?: boolean
  actor?: string
  reason?: string
}

export interface MemoryMaintenanceResult {
  dryRun: boolean
  importanceUpdated: number
  pruned: number
  wouldPrune: number
  status: MemoryLifecycleStatus
}

export const MEMORY_GRAPH_NODE_KINDS = [
  'person',
  'project',
  'preference',
  'tool',
  'topic',
  'decision',
  'constraint',
  'fact',
  'place',
  'organization',
  'other',
] as const

export type MemoryGraphNodeKind = typeof MEMORY_GRAPH_NODE_KINDS[number]

export interface MemoryGraphNode {
  id: string
  label: string
  kind: MemoryGraphNodeKind
  aliases: string[]
  tags: string[]
  evidenceMemoryIds: string[]
  confidence: number
  createdAt: string
  updatedAt: string
  lastSeenAt: string
}

export interface MemoryGraphEdge {
  id: string
  fromNodeId: string
  toNodeId: string
  relation: string
  tags: string[]
  evidenceMemoryIds: string[]
  confidence: number
  createdAt: string
  updatedAt: string
  lastSeenAt: string
}

export interface MemoryGraphNodeInput {
  label: string
  kind?: MemoryGraphNodeKind | string
  aliases?: string[]
  tags?: string[]
  evidenceMemoryIds?: string[]
  confidence?: number
}

export interface MemoryGraphEdgeInput {
  fromLabel: string
  fromKind?: MemoryGraphNodeKind | string
  toLabel: string
  toKind?: MemoryGraphNodeKind | string
  relation: string
  tags?: string[]
  evidenceMemoryIds?: string[]
  confidence?: number
}

export interface MemoryGraphUpsertInput {
  nodes?: MemoryGraphNodeInput[]
  edges?: MemoryGraphEdgeInput[]
}

export interface MemoryGraphSearchResult {
  nodes: MemoryGraphNode[]
  edges: MemoryGraphEdge[]
}

export interface MemoryGraphStats {
  nodes: number
  edges: number
  lastUpdatedAt?: string
}

export interface MemoryGraphEvidenceEntry {
  evidence?: MemoryEntry['evidence']
  id: string
  content: string
  source: MemoryEntry['source']
  tags: string[]
}

export interface MemoryGraphWikiPageRelationship {
  direction: 'in' | 'out'
  edge: MemoryGraphEdge
  node: MemoryGraphNode
}

export type MemoryGraphQualitySeverity = 'info' | 'warning' | 'critical'

export type MemoryGraphQualitySignalCode =
  | 'low_confidence'
  | 'thin_evidence'
  | 'missing_evidence'
  | 'inactive_evidence'
  | 'contradiction'
  | 'stale'
  | 'orphan'

export interface MemoryGraphQualitySignal {
  code: MemoryGraphQualitySignalCode
  severity: MemoryGraphQualitySeverity
  subjectType: 'page' | 'node' | 'edge'
  subjectId?: string
  message: string
  evidenceMemoryIds?: string[]
}

export interface MemoryGraphWikiPageQuality {
  score: number
  confidence: number
  evidenceCount: number
  activeEvidenceCount: number
  inactiveEvidenceCount: number
  missingEvidenceCount: number
  relationshipCount: number
  contradictionCount: number
  staleRelationshipCount: number
  signals: MemoryGraphQualitySignal[]
}

export interface MemoryGraphWikiPage {
  query?: string
  node: MemoryGraphNode | null
  relationships: MemoryGraphWikiPageRelationship[]
  evidence: MemoryGraphEvidenceEntry[]
  quality?: MemoryGraphWikiPageQuality
}

export interface MemoryGraphMaintenanceResult {
  checkedNodes: number
  checkedEdges: number
  prunedNodes: number
  prunedEdges: number
}

export interface MemoryGraphQualityReport {
  generatedAt: string
  stats: MemoryGraphStats
  scannedNodes: number
  scannedEdges: number
  lowConfidenceNodes: number
  lowConfidenceEdges: number
  thinEvidenceNodes: number
  thinEvidenceEdges: number
  missingEvidenceNodes: number
  missingEvidenceEdges: number
  inactiveEvidenceNodes: number
  inactiveEvidenceEdges: number
  staleNodes: number
  staleEdges: number
  orphanNodes: number
  contradictions: number
  signals: MemoryGraphQualitySignal[]
}

export type MemoryGraphRepairAction =
  | 'prune_unbacked_graph_entry'
  | 'relink_or_add_evidence'
  | 'refresh_graph_entry'
  | 'review_contradiction'

export interface MemoryGraphRepairProposal {
  id: string
  action: MemoryGraphRepairAction
  subjectType: 'node' | 'edge' | 'page'
  subjectId?: string
  severity: MemoryGraphQualitySeverity
  safeToApply: boolean
  reason: string
  signalCodes: MemoryGraphQualitySignalCode[]
  evidenceMemoryIds: string[]
}

export interface MemoryGraphRepairAppliedSummary {
  checkedNodes: number
  checkedEdges: number
  prunedNodes: number
  prunedEdges: number
  skippedUnsafe: number
}

export interface MemoryGraphRepairResult {
  dryRun: boolean
  generatedAt: string
  report: MemoryGraphQualityReport
  proposals: MemoryGraphRepairProposal[]
  applied: MemoryGraphRepairAppliedSummary
}

export type MemoryGraphRepairDecisionAction = 'supersede_memories'

export interface MemoryGraphRepairDecisionInput {
  action?: MemoryGraphRepairDecisionAction
  winnerMemoryId: string
  supersededMemoryIds: string[]
  proposalId?: string
  dryRun?: boolean
  reason?: string
  actor?: string
}

export interface MemoryGraphRepairDecisionUpdate {
  id: string
  beforeTags: string[]
  afterTags: string[]
}

export interface MemoryGraphRepairDecisionSkip {
  id: string
  reason:
    | 'winner_not_found'
    | 'winner_inactive'
    | 'memory_not_found'
    | 'winner_in_superseded_ids'
    | 'already_superseded'
    | 'already_superseded_by_winner'
    | 'no_change'
  message: string
}

export interface MemoryGraphRepairDecisionResult {
  dryRun: boolean
  generatedAt: string
  action: MemoryGraphRepairDecisionAction
  proposalId?: string
  winnerMemoryId: string
  updated: MemoryGraphRepairDecisionUpdate[]
  skipped: MemoryGraphRepairDecisionSkip[]
  maintenance: MemoryGraphMaintenanceResult
}

export interface MemoryGraphStore {
  upsertGraph(input: MemoryGraphUpsertInput): Promise<{ nodesUpserted: number; edgesUpserted: number }>
  searchGraph(query: string, options?: { limit?: number; kind?: string }): Promise<MemoryGraphSearchResult>
  getGraphNeighbors(
    nodeId: string,
    options?: { limit?: number; direction?: 'in' | 'out' | 'both' },
  ): Promise<{ node: MemoryGraphNode | null; edges: MemoryGraphEdge[]; nodes: MemoryGraphNode[] }>
  getGraphWikiPage(input: {
    query?: string
    nodeId?: string
    limit?: number
    evidenceLimit?: number
  }): Promise<MemoryGraphWikiPage>
  runGraphMaintenance(options?: { pruneMissingEvidence?: boolean }): Promise<MemoryGraphMaintenanceResult>
  inspectGraphQuality(options?: {
    limit?: number
    signalLimit?: number
    lowConfidenceThreshold?: number
    staleAfterDays?: number
  }): Promise<MemoryGraphQualityReport>
  repairGraphQuality(options?: {
    dryRun?: boolean
    limit?: number
    signalLimit?: number
    lowConfidenceThreshold?: number
    staleAfterDays?: number
  }): Promise<MemoryGraphRepairResult>
  applyGraphRepairDecision(input: MemoryGraphRepairDecisionInput): Promise<MemoryGraphRepairDecisionResult>
  getGraphStats(): Promise<MemoryGraphStats>
}

export interface SemanticMemoryStore
  extends IDreamingMemoryStore,
    IDocumentMemoryStore,
    MemoryGraphStore,
    ISubscribableMemoryStore,
    IListRecentWithTimestampsStore,
    IAccessTrackingMemoryStore,
    IPinningMemoryStore {
  listForScope?(scopeTags: string[], options?: { limit?: number; offset?: number; includeInactive?: boolean }): Promise<{ memories: Array<Omit<MemoryEntry, 'score'>>; nextOffset: number | null }>
  previewForgetOwnedMemory?(scopeTags: string[]): Promise<{ memories: number; documents: number }>
  forgetOwnedMemory?(scopeTags: string[]): Promise<{ memories: number; documents: number; resetAt: string }>
  configureFileMemory?(registry: import('./scoped-file-memory.js').ScopedFileMemoryRegistryImpl): Promise<void>

  getStatus(): SemanticIndexRuntimeStatus
  getLifecycleStatus(options?: MemoryLifecycleOptions): Promise<MemoryLifecycleStatus>
  runMaintenance(options?: MemoryMaintenanceOptions): Promise<MemoryMaintenanceResult>
  recordAudit(input: MemoryAuditRecordInput): Promise<MemoryAuditEntry>
  listAudit(options?: MemoryAuditListOptions): Promise<MemoryAuditEntry[]>
  /**
   * Delete every memory row (and its vector rows) derived from a given session.
   * Used by complete session deletion so a removed session leaves no orphaned
   * memory entries or embeddings behind. Returns the number of rows removed.
   */
  deleteBySession(sessionId: string): Promise<number>
  /**
   * Delete every memory row (and its vector rows) last updated before an ISO
   * cutoff. Used by the retention sweeper to age out old memory. Returns the
   * number of rows removed.
   */
  deleteOlderThan(cutoffIso: string): Promise<number>
  backfill(batchSize?: number): Promise<number>
  reindex(batchSize?: number): Promise<number>
  startBackgroundBackfill(batchSize?: number): void
  startBackgroundReindex(batchSize?: number): boolean
  close(): void
}
