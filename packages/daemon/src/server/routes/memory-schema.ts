import { memoryEvidenceSchema } from '../../memory/evidence.js'
import { z } from 'zod'

export const memorySourceSchema = z.enum(['conversation', 'document', 'skill', 'user'])

export const memorySearchTypeSchema = z.enum(['semantic', 'keyword', 'hybrid'])

export const memoryEntrySchema = z.object({
  id: z.string(),
  evidence: memoryEvidenceSchema.optional(),
  content: z.string(),
  source: memorySourceSchema,
  tags: z.array(z.string()),
  score: z.number().optional(),
})

export const memoryRecentEntrySchema = memoryEntrySchema.omit({ score: true }).extend({
  createdAt: z.string().optional(),
  updatedAt: z.string().optional(),
})

const memoryAuditSnapshotSchema = memoryEntrySchema.omit({ score: true })

export const memoryDocumentSchema = z.object({
  id: z.string(),
  title: z.string(),
  path: z.string().optional(),
  mimeType: z.string().optional(),
  sourceFileId: z.string().optional(),
  tags: z.array(z.string()),
  chunkCount: z.number().int(),
  createdAt: z.string(),
  updatedAt: z.string(),
})

export const memoryDocumentChunkSchema = memoryEntrySchema.extend({
  documentId: z.string(),
  documentTitle: z.string(),
  documentPath: z.string().optional(),
  mimeType: z.string().optional(),
  sourceFileId: z.string().optional(),
  chunkIndex: z.number().int(),
  chunkCount: z.number().int(),
  startOffset: z.number().int(),
  endOffset: z.number().int(),
  chunkTitle: z.string().optional(),
  snippet: z.string().optional(),
  citationLabel: z.string().optional(),
})

export const memoryCreateRequestSchema = z.object({
  evidence: memoryEvidenceSchema.optional(),
  content: z.string().min(1),
  source: memorySourceSchema.optional(),
  tags: z.array(z.string()).optional(),
  reason: z.string().trim().min(1).optional(),
})

export const memoryEntryIdParamsSchema = z.object({
  id: z.string().min(1),
})

export const memoryUpdateRequestSchema = z.object({
  evidence: memoryEvidenceSchema.optional(),
  content: z.string().min(1),
  source: memorySourceSchema.optional(),
  tags: z.array(z.string()).optional(),
  reason: z.string().trim().min(1).optional(),
})

export const memoryDeleteQuerySchema = z.object({
  reason: z.string().trim().min(1).optional(),
  includeAllScopes: z.union([z.boolean(), z.string()])
    .transform((value) => value === true || value === 'true' || value === '1')
    .optional(),
})

export const memoryDeleteRequestSchema = memoryDeleteQuerySchema.extend({
  id: z.string().trim().min(1),
})

export const memoryAuditActionSchema = z.enum([
  'created',
  'updated',
  'deleted',
  'pruned',
  'maintenance',
])

export const memoryAuditEntrySchema = z.object({
  id: z.string(),
  memoryId: z.string(),
  action: memoryAuditActionSchema,
  actor: z.string(),
  reason: z.string().optional(),
  before: memoryAuditSnapshotSchema.optional(),
  after: memoryAuditSnapshotSchema.optional(),
  createdAt: z.string(),
})

export const memoryAuditQuerySchema = z.object({
  memoryId: z.string().min(1).optional(),
  limit: z.coerce.number().int().min(1).max(500).optional(),
  action: z.union([
    memoryAuditActionSchema,
    z.array(memoryAuditActionSchema),
    z.string(),
  ]).optional(),
  actor: z.string().min(1).optional(),
  since: z.string().optional(),
  until: z.string().optional(),
  includeAllScopes: z.union([z.boolean(), z.string()])
    .transform((value) => value === true || value === 'true' || value === '1')
    .optional(),
})

export const memorySecurityAuditQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(500).optional(),
  since: z.string().optional(),
  actor: z.string().min(1).optional(),
  authKind: z.string().min(1).optional(),
  route: z.string().min(1).optional(),
})

export const memoryScopeTransferRequestSchema = z.object({
  target: z.string().trim().min(1),
  dryRun: z.boolean().optional(),
  includeFile: z.boolean().optional(),
  includeReminders: z.boolean().optional(),
  reason: z.string().trim().min(1).optional(),
  confirmGlobal: z.boolean().optional(),
  ids: z.array(z.string().trim().min(1)).optional(),
  limit: z.number().int().min(1).max(10_000).optional(),
})

export const memoryLifecycleQuerySchema = z.object({
  staleAfterDays: z.coerce.number().int().min(1).optional(),
  lowImportance: z.coerce.number().min(0).max(1).optional(),
})

export const memoryLifecycleStatusSchema = z.object({
  totalMemories: z.number().int(),
  conversationMemories: z.number().int(),
  documentMemories: z.number().int(),
  skillMemories: z.number().int(),
  userMemories: z.number().int(),
  staleConversationMemories: z.number().int(),
  lowImportanceConversationMemories: z.number().int(),
  pruneCandidateMemories: z.number().int(),
  pendingEmbeddings: z.number().int(),
  failedEmbeddings: z.number().int(),
  lastAuditAt: z.string().optional(),
})

export const memoryMaintenanceRequestSchema = z.object({
  maxAgeDays: z.number().int().min(1).optional(),
  maxImportance: z.number().min(0).max(1).optional(),
  dryRun: z.boolean().optional(),
  reason: z.string().trim().min(1).optional(),
}).default({})

export const memoryMaintenanceResultSchema = z.object({
  dryRun: z.boolean(),
  importanceUpdated: z.number().int(),
  pruned: z.number().int(),
  wouldPrune: z.number().int(),
  status: memoryLifecycleStatusSchema,
})

// Querystring helpers — Fastify delivers `?tags=a&tags=b` as an array, but
// `?tags=a` as a single string. Accept either and normalize to string[].
const csvOrArray = z.union([z.string(), z.array(z.string())])
  .transform((value) => {
    if (Array.isArray(value)) {
      return value.flatMap((entry) => entry.split(',')).map((entry) => entry.trim()).filter(Boolean)
    }
    return value.split(',').map((entry) => entry.trim()).filter(Boolean)
  })

const memorySourceArraySchema = z.union([
  z.array(z.enum(['conversation', 'document', 'skill', 'user'])),
  z.string().transform((value) =>
    value.split(',').map((entry) => entry.trim()).filter(Boolean) as Array<'conversation' | 'document' | 'skill' | 'user'>,
  ),
])

const booleanQuery = z.union([z.boolean(), z.string()])
  .transform((value) => value === true || value === 'true' || value === '1')

export const memorySearchQuerySchema = z.object({
  asOf: z.string().datetime({ offset: true }).optional(),
  includeInactive: booleanQuery.optional(),
  query: z.string(),
  limit: z.coerce.number().int().min(1).optional(),
  type: memorySearchTypeSchema.optional(),
  sources: memorySourceArraySchema.optional(),
  tags: csvOrArray.optional(),
  tagsLogic: z.enum(['and', 'or']).optional(),
  excludeTags: csvOrArray.optional(),
  minScore: z.coerce.number().min(0).max(1).optional(),
  createdAfter: z.string().optional(),
  createdBefore: z.string().optional(),
  includeAllScopes: booleanQuery.optional(),
  includeSuperseded: booleanQuery.optional(),
  includeArchived: booleanQuery.optional(),
  sortBy: z.enum(['score', 'createdAt', 'updatedAt']).optional(),
})

export const memoryRecentQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(500).optional(),
  sources: memorySourceArraySchema.optional(),
  tags: csvOrArray.optional(),
  tagsLogic: z.enum(['and', 'or']).optional(),
  excludeTags: csvOrArray.optional(),
  createdAfter: z.string().optional(),
  createdBefore: z.string().optional(),
  includeAllScopes: booleanQuery.optional(),
  includeSuperseded: booleanQuery.optional(),
  includeArchived: booleanQuery.optional(),
  sortBy: z.enum(['createdAt', 'updatedAt']).optional(),
})

export const memoryPinnedQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(500).optional(),
  includeAllScopes: booleanQuery.optional(),
})

export const memoryGraphPageQuerySchema = z.object({
  query: z.string().trim().min(1).optional(),
  id: z.string().trim().min(1).optional(),
  limit: z.coerce.number().int().min(1).max(100).optional(),
  evidenceLimit: z.coerce.number().int().min(1).max(100).optional(),
  includeAllScopes: booleanQuery.optional(),
}).refine((value) => Boolean(value.query || value.id), {
  message: 'query or id is required',
})

export const memoryGraphAuditQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(1000).optional(),
  signalLimit: z.coerce.number().int().min(1).max(500).optional(),
  lowConfidenceThreshold: z.coerce.number().min(0).max(1).optional(),
  staleAfterDays: z.coerce.number().int().min(1).max(3650).optional(),
  includeAllScopes: booleanQuery.optional(),
})

export const memoryGraphRepairRequestSchema = z.object({
  dryRun: z.boolean().optional(),
  limit: z.number().int().min(1).max(1000).optional(),
  signalLimit: z.number().int().min(1).max(500).optional(),
  lowConfidenceThreshold: z.number().min(0).max(1).optional(),
  staleAfterDays: z.number().int().min(1).max(3650).optional(),
  includeAllScopes: z.boolean().optional(),
  reason: z.string().trim().min(1).optional(),
}).default({})

export const memoryGraphRepairDecisionRequestSchema = z.object({
  action: z.literal('supersede_memories').optional(),
  winnerMemoryId: z.string().trim().min(1),
  supersededMemoryIds: z.array(z.string().trim().min(1)).min(1).max(100),
  proposalId: z.string().trim().min(1).optional(),
  dryRun: z.boolean().optional(),
  includeAllScopes: z.boolean().optional(),
  reason: z.string().trim().min(1).optional(),
})

export const memoryDocumentIdParamsSchema = z.object({
  id: z.string().min(1),
})

export const memoryDocumentIngestRequestSchema = z.object({
  id: z.string().min(1).optional(),
  title: z.string().min(1),
  content: z.string().min(1),
  path: z.string().optional(),
  mimeType: z.string().optional(),
  sourceFileId: z.string().optional(),
  tags: z.array(z.string()).optional(),
})

export const memoryDocumentSearchQuerySchema = z.object({
  query: z.string().min(1),
  limit: z.coerce.number().int().min(1).optional(),
  type: memorySearchTypeSchema.optional(),
  documentId: z.string().min(1).optional(),
  sources: memorySourceArraySchema.optional(),
  tags: csvOrArray.optional(),
  tagsLogic: z.enum(['and', 'or']).optional(),
  excludeTags: csvOrArray.optional(),
  minScore: z.coerce.number().min(0).max(1).optional(),
  createdAfter: z.string().optional(),
  createdBefore: z.string().optional(),
  includeAllScopes: booleanQuery.optional(),
  sortBy: z.enum(['score', 'createdAt', 'updatedAt']).optional(),
})

export const memoryDocumentListQuerySchema = z.object({
  query: z.string().min(1).optional(),
  limit: z.coerce.number().int().min(1).optional(),
  includeAllScopes: booleanQuery.optional(),
})

export const memoryHotQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(200).optional(),
  includeAllScopes: z.union([z.boolean(), z.string()])
    .transform((value) => value === true || value === 'true' || value === '1')
    .optional(),
})

export const memorySemanticStatusSchema = z.object({
  status: z.enum(['disabled', 'ready', 'backfilling', 'degraded', 'reindex_required']),
  configuredProviderId: z.string().optional(),
  configuredModel: z.string().optional(),
  indexedProviderId: z.string().optional(),
  indexedModel: z.string().optional(),
  dimensions: z.number().int().optional(),
  pendingCount: z.number().int(),
  failedCount: z.number().int(),
  vecAvailable: z.boolean(),
  vectorBackend: z.enum(['sqlite-vec', 'sqlite-scan', 'qdrant', 'opensearch', 'elasticsearch', 'meilisearch', 'custom-api']),
  backendAvailable: z.boolean(),
  lastError: z.string().optional(),
})

export const memoryFileSectionParamsSchema = z.object({
  sectionTitle: z.string().min(1),
})

export const memoryFileSectionUpdateSchema = z.object({
  content: z.string(),
})

export type MemoryCreateBody = z.infer<typeof memoryCreateRequestSchema>
export type MemoryEntryIdParams = z.infer<typeof memoryEntryIdParamsSchema>
export type MemoryUpdateBody = z.infer<typeof memoryUpdateRequestSchema>
export type MemoryDeleteQuery = z.infer<typeof memoryDeleteQuerySchema>
export type MemoryDeleteBody = z.infer<typeof memoryDeleteRequestSchema>
export type MemoryAuditQuery = z.infer<typeof memoryAuditQuerySchema>
export type MemorySecurityAuditQuery = z.infer<typeof memorySecurityAuditQuerySchema>
export type MemoryScopeTransferBody = z.infer<typeof memoryScopeTransferRequestSchema>
export type MemoryLifecycleQuery = z.infer<typeof memoryLifecycleQuerySchema>
export type MemoryMaintenanceBody = z.infer<typeof memoryMaintenanceRequestSchema>
export type MemorySearchQuery = z.infer<typeof memorySearchQuerySchema>
export type MemoryRecentQuery = z.infer<typeof memoryRecentQuerySchema>
export type MemoryPinnedQuery = z.infer<typeof memoryPinnedQuerySchema>
export type MemoryGraphPageQuery = z.infer<typeof memoryGraphPageQuerySchema>
export type MemoryGraphAuditQuery = z.infer<typeof memoryGraphAuditQuerySchema>
export type MemoryGraphRepairBody = z.infer<typeof memoryGraphRepairRequestSchema>
export type MemoryGraphRepairDecisionBody = z.infer<typeof memoryGraphRepairDecisionRequestSchema>
export type MemoryDocumentIdParams = z.infer<typeof memoryDocumentIdParamsSchema>
export type MemoryDocumentIngestBody = z.infer<typeof memoryDocumentIngestRequestSchema>
export type MemoryDocumentSearchQuery = z.infer<typeof memoryDocumentSearchQuerySchema>
export type MemoryDocumentListQuery = z.infer<typeof memoryDocumentListQuerySchema>
export type MemoryHotQuery = z.infer<typeof memoryHotQuerySchema>
export type MemoryFileSectionParams = z.infer<typeof memoryFileSectionParamsSchema>
export type MemoryFileSectionUpdateBody = z.infer<typeof memoryFileSectionUpdateSchema>
