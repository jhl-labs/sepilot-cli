import { looksSensitive } from '../memory/sensitive.js'
import { memoryEvidenceSchema, memoryEvidenceInputSchema } from '../memory/evidence.js'
import type {
  DocumentIngestInput,
  DocumentListOptions,
  DocumentSearchOptions,
  DocumentUpdateInput,
  IDocumentMemoryStore,
  ISemanticIndex,
  MemoryAccessHotEntry,
  MemoryAccessStats,
  MemoryDocument,
  MemoryDocumentChunk,
  MemoryEntry,
  MemoryPinnedEntry,
  SemanticSearchOptions,
} from '@sepilotd/core'
import type { FileMemory, FileMemorySection } from '../memory/file-memory.js'
import type { ScopedFileMemoryRegistry } from '../memory/scoped-file-memory.js'
import type {
  MemoryAuditEntry,
  MemoryAuditListOptions,
  MemoryAuditRecordInput,
  MemoryGraphEdge,
  MemoryGraphNode,
  MemoryGraphQualityReport,
  MemoryGraphRepairDecisionResult,
  MemoryGraphRepairResult,
  MemoryGraphStore,
  MemoryGraphWikiPage,
  MemoryLifecycleOptions,
  MemoryLifecycleStatus,
  MemoryMaintenanceOptions,
  MemoryMaintenanceResult,
} from '../memory/types.js'
import {
  attachScopeTags,
  isMemoryArchived,
  isMemoryVisibleInScope,
  isMemoryWritableInScope,
  isMemorySuperseded,
  parseScopeFromTags,
} from '../memory/scope.js'
import { parseReminderTime } from '../memory/reminders.js'
import { parseRelativePastTimestamp } from '../memory/time-relative.js'
import { rememberUserMemory } from '../memory/user-memory.js'
import {
  explicitlyRequestsAppIndex,
  filterAppIndexEntries,
  isAppIndexMemoryEntry,
} from '../memory/internal-app-index.js'
import type { ToolDefinitionRuntime, ToolExecutionContext, ToolResult } from './registry.js'

export interface MemoryRememberToolDeps {
  fileMemory?: Pick<FileMemory, 'mergeMemorySectionItems'>
  fileMemoryRegistry?: ScopedFileMemoryRegistry
  semanticIndex?: Pick<ISemanticIndex, 'add'>
}

export interface MemoryForgetToolDeps {
  forgetAll?: (scopeTags: string[], options?: { dryRun?: boolean }) => Promise<unknown>
  fileMemory?: Pick<
    FileMemory,
    'deleteMemorySection' | 'removeMemorySectionItem' | 'readMemorySections'
  >
  fileMemoryRegistry?: ScopedFileMemoryRegistry
}

export interface MemoryListToolDeps {
  semanticIndex?: Pick<import('../memory/types.js').SemanticMemoryStore, 'listForScope'>
  fileMemory?: Pick<FileMemory, 'readMemorySections' | 'readMemorySection'>
  fileMemoryRegistry?: ScopedFileMemoryRegistry
}

export interface MemorySearchToolDeps {
  semanticIndex?: Pick<ISemanticIndex, 'search'> & {
    recordAccess?(ids: string[]): Promise<void>
  }
}

export interface MemoryGraphToolDeps {
  semanticIndex?: Pick<
    MemoryGraphStore,
    'searchGraph' | 'getGraphNeighbors' | 'getGraphWikiPage' | 'getGraphStats'
  > & Partial<Pick<MemoryGraphStore, 'inspectGraphQuality' | 'repairGraphQuality' | 'applyGraphRepairDecision'>>
    & Partial<Pick<ISemanticIndex, 'get'>>
}

export interface MemoryUpdateToolDeps {
  semanticIndex?: Pick<ISemanticIndex, 'add' | 'get'> & {
    recordAudit(input: MemoryAuditRecordInput): Promise<MemoryAuditEntry>
  }
}

export interface MemoryDocumentsToolDeps {
  documentStore?: Pick<
    IDocumentMemoryStore,
    'ingestDocument' | 'searchDocuments' | 'listDocuments' | 'getDocument' | 'deleteDocument' | 'updateDocument' | 'listDocumentChunks'
  >
}

export interface MemoryAuditToolDeps {
  semanticIndex?: {
    listAudit(options?: MemoryAuditListOptions): Promise<MemoryAuditEntry[]>
  }
}

export interface MemoryMaintenanceToolDeps {
  semanticIndex?: {
    getLifecycleStatus(options?: MemoryLifecycleOptions): Promise<MemoryLifecycleStatus>
    runMaintenance(options?: MemoryMaintenanceOptions): Promise<MemoryMaintenanceResult>
  }
}

export interface MemoryExportImportToolDeps {
  fileMemory?: Pick<
    FileMemory,
    'readMemorySections' | 'replaceMemorySection' | 'mergeMemorySectionItems'
  >
  fileMemoryRegistry?: ScopedFileMemoryRegistry
  semanticIndex?: {
    listRecent(limit: number): Promise<Array<Omit<MemoryEntry, 'score'>>>
    listRecentWithTimestamps?(
      limit: number,
      options?: { createdAfter?: string; createdBefore?: string },
    ): Promise<Array<Omit<MemoryEntry, 'score'> & { createdAt: string; updatedAt: string }>>
    add(entry: Omit<MemoryEntry, 'score'>): Promise<void>
    get(id: string): Promise<MemoryEntry | null>
  }
  reminders?: {
    listForScope(scopeTags: string[]): Promise<Array<{
      id: string
      dueAt: string
      content: string
      scopeTags: string[]
      channelType?: string
      chatId?: string
    }>>
    add(input: {
      dueAt: string
      content: string
      scopeTags: string[]
      channelType?: string
      chatId?: string
    }): Promise<{ id: string }>
  }
}

export interface MemoryTagSuggestToolDeps {
  /** Function that calls the LLM and returns 3-5 lowercase topic tags. */
  suggestTags?: (content: string) => Promise<string[]>
}

export interface MemoryMergeToolDeps {
  semanticIndex?: {
    get(id: string): Promise<MemoryEntry | null>
    add(entry: Omit<MemoryEntry, 'score'>): Promise<void>
    recordAudit(input: MemoryAuditRecordInput): Promise<MemoryAuditEntry>
  }
  summarize?: (contents: string[], options?: { instruction?: string; targetChars?: number }) => Promise<string | null>
}

export interface MemorySummarizeToolDeps {
  semanticIndex?: {
    get(id: string): Promise<MemoryEntry | null>
    listRecent(limit: number): Promise<Array<Omit<MemoryEntry, 'score'>>>
  }
  /** Returns an LLM summary for the supplied memory contents. Null when
   *  no provider is configured. */
  summarize?: (contents: string[], options?: { instruction?: string; targetChars?: number }) => Promise<string | null>
  /** Optional file-memory registry — when present, callers may pass
   *  saveTo to land the summary in a daily-note section. */
  fileMemoryRegistry?: ScopedFileMemoryRegistry
  fileMemory?: Pick<FileMemory, 'appendToDailySection' | 'getDailyNotePath'>
}

export interface MemoryDiffToolDeps {
  semanticIndex?: Pick<ISemanticIndex, 'get'>
}

export interface MemoryTagRenameToolDeps {
  semanticIndex?: {
    listRecent(limit: number): Promise<Array<Omit<MemoryEntry, 'score'>>>
    add(entry: Omit<MemoryEntry, 'score'>): Promise<void>
  }
}

export interface MemoryRelatedToolDeps {
  semanticIndex?: Pick<ISemanticIndex, 'get' | 'search'> & {
    recordAccess?(ids: string[]): Promise<void>
  }
}

export interface MemoryTagListToolDeps {
  semanticIndex?: {
    listRecent(limit: number): Promise<Array<Omit<MemoryEntry, 'score'>>>
  }
}

export interface MemorySearchByTagToolDeps {
  semanticIndex?: {
    listRecent(limit: number): Promise<Array<Omit<MemoryEntry, 'score'>>>
    recordAccess?(ids: string[]): Promise<void>
  }
}

export interface MemoryAccessToolDeps {
  semanticIndex?: {
    get?(id: string): Promise<MemoryEntry | null>
    recordAccess(ids: string[]): Promise<void>
    getAccessStats(id: string): Promise<MemoryAccessStats | null>
    listHotMemories(limit: number): Promise<MemoryAccessHotEntry[]>
  }
}

export interface MemoryPinToolDeps {
  semanticIndex?: Pick<ISemanticIndex, 'get'> & {
    pin(id: string): Promise<void>
    unpin(id: string): Promise<void>
    isPinned(id: string): Promise<boolean | null>
    listPinned(limit: number): Promise<MemoryPinnedEntry[]>
    recordAudit?(input: MemoryAuditRecordInput): Promise<MemoryAuditEntry>
  }
}

export interface MemoryHistoryToolDeps {
  semanticIndex?: Pick<ISemanticIndex, 'get'> & {
    listAudit(options?: MemoryAuditListOptions): Promise<MemoryAuditEntry[]>
  }
}

export interface MemoryConflictsToolDeps {
  /** Returns contradiction candidates without mutating memory. */
  findRecentContradictions?: (options?: { maxPairs?: number }) => Promise<Array<{
    winnerId: string
    supersededIds: string[]
    primaryId: string
    candidateIds: string[]
  }>>
}

export interface MemoryContextSnapshotToolDeps {
  fileMemory?: Pick<FileMemory, 'getPromptContext' | 'readMemorySections' | 'getDailyNotePath'>
  fileMemoryRegistry?: ScopedFileMemoryRegistry
  semanticIndex?: Pick<ISemanticIndex, 'search'> & Partial<Pick<MemoryGraphStore, 'searchGraph' | 'getGraphStats'>>
  reminders?: {
    listForScope(scopeTags: string[]): Promise<Array<{
      id: string
      dueAt: string
      content: string
      scopeTags: string[]
      channelType?: string
      chatId?: string
      firedAt?: string
      cancelledAt?: string
    }>>
  }
}

export interface MemoryUsageToolDeps {
  semanticIndex?: {
    listRecent(limit: number): Promise<Array<Omit<MemoryEntry, 'score'>>>
    listAudit?(options?: MemoryAuditListOptions): Promise<MemoryAuditEntry[]>
  }
  reminders?: {
    listForScope(scopeTags: string[]): Promise<Array<{
      id: string
      dueAt: string
      content: string
      scopeTags: string[]
      channelType?: string
      chatId?: string
      firedAt?: string
      cancelledAt?: string
    }>>
  }
  fileMemoryRegistry?: ScopedFileMemoryRegistry
  fileMemory?: Pick<FileMemory, 'listDailyNoteDates' | 'readMemorySections'>
}

export interface MemoryReminderToolDeps {
  reminders?: {
    add(input: {
      dueAt: string
      content: string
      scopeTags: string[]
      channelType?: string
      chatId?: string
    }): Promise<{ id: string; dueAt: string; content: string; scopeTags: string[] }>
    listForScope(scopeTags: string[]): Promise<Array<{
      id: string
      dueAt: string
      content: string
      scopeTags: string[]
      channelType?: string
      chatId?: string
    }>>
    cancel(id: string, reason?: string): Promise<{ id: string } | null>
  }
}

export interface MemoryDailyToolDeps {
  fileMemory?: Pick<
    FileMemory,
    | 'readDailyNote'
    | 'readDailySection'
    | 'appendToDailySection'
    | 'replaceDailySection'
    | 'getDailyNotePath'
    | 'listDailyNoteDates'
    | 'searchDailyNotes'
  > & Partial<Pick<FileMemory, 'getDailyNoteReadPath' | 'searchDailyNotesPage'>>
  fileMemoryRegistry?: ScopedFileMemoryRegistry
}

const ALLOWED_MEMORY_SOURCES: ReadonlyArray<MemoryEntry['source']> = [
  'user',
  'conversation',
  'document',
  'skill',
]
const SEARCH_DEFAULT_LIMIT = 8
const SEARCH_MAX_LIMIT = 50
const SEARCH_SNIPPET_MAX_CHARS = 320
// Upper bound on a single memory.documents.ingest content payload, so an
// arbitrarily large document cannot flood chunking + the embedding queue.
const MAX_DOCUMENT_INGEST_BYTES = 4 * 1024 * 1024 // 4MB

function isInternalAppDocument(doc: Pick<MemoryDocument, 'id' | 'path' | 'tags'>): boolean {
  return isAppIndexMemoryEntry({
    source: 'document',
    id: doc.id,
    path: doc.path,
    tags: doc.tags,
  })
}

function canTagFilterReturnInternalAppIndex(
  tags: readonly string[],
  tagsLogic: 'and' | 'or' = 'and',
): boolean {
  if (tags.length === 0) return true
  const canMatchInternalAppTag = (tag: string): boolean =>
    tag === 'app' || tag.startsWith('app:') || tag.startsWith('kind:')
  return tagsLogic === 'or'
    ? tags.some(canMatchInternalAppTag)
    : tags.every(canMatchInternalAppTag)
}

export function createMemoryRememberTool(deps: MemoryRememberToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.remember',
    description: 'Persist an explicit user-requested memory, preference, stable personal fact, durable project fact, or reusable instruction for future sessions. Use knowledge.save instead when the user explicitly chooses the personal Wiki / permanent knowledge destination; file/document skills handle authored documents. Ask about the destination if context leaves it ambiguous. Use memory.update with an existing id for corrections, rather than storing a contradictory new fact. For a known temporary preference, supply validUntil. Do not store secrets, credentials, live prices, or volatile facts as durable memory.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        evidence: memoryEvidenceInputSchema,
        subject: memoryEvidenceInputSchema.properties.subject,
        reality: memoryEvidenceInputSchema.properties.reality,
        validFrom: { type: 'string', format: 'date-time', description: 'Optional inclusive start of validity, full ISO timestamp with timezone. Mutually exclusive with evidence.' },
        validUntil: { type: 'string', format: 'date-time', description: 'Optional exclusive expiry, full ISO timestamp with timezone. Mutually exclusive with evidence.' },
        content: {
          type: 'string',
          description: 'The durable fact or preference to remember, without the surrounding "remember this" wording.',
        },
        tags: {
          type: 'array',
          items: { type: 'string' },
          description: 'Optional user tags such as preference, portfolio, project, or contact. Never copy system tags from search results: scope:*, archived*, superseded*, compressed-from:* are runtime-managed.',
        },
      },
      required: ['content'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const hasValidity = input.validFrom !== undefined || input.validUntil !== undefined || input.subject !== undefined || input.reality !== undefined
      if (hasValidity && input.evidence !== undefined) return { status: 'error', code: 'INVALID_INPUT_PERMANENT', output: 'Use top-level classification/validity fields or evidence, not both.', durationMs: Date.now() - startedAt }
      const evidenceInput = hasValidity ? { kind: 'semantic', origin: 'user', observedAt: new Date().toISOString(), status: 'active', sourceIds: [], validFrom: input.validFrom, validUntil: input.validUntil, subject: input.subject, reality: input.reality } : input.evidence
      const evidence = evidenceInput === undefined ? undefined : memoryEvidenceSchema.safeParse(evidenceInput)
      if (evidence && !evidence.success) return { status: 'error', code: 'INVALID_INPUT_PERMANENT', output: 'Invalid memory evidence or validity interval.', durationMs: Date.now() - startedAt }
      const content = typeof input.content === 'string' ? input.content : ''
      const userTags = Array.isArray(input.tags)
        ? input.tags.filter((tag): tag is string => typeof tag === 'string')
        : []
      const forbidden = findForbiddenUserTags(userTags)
      if (forbidden.length > 0) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: `Reserved/system tags cannot be set directly: ${forbidden.join(', ')}. Scope ownership is assigned automatically; lifecycle tags (archived/superseded) are managed by dedicated operations.`,
          durationMs: Date.now() - startedAt,
        }
      }
      const tags = attachScopeTags(userTags, context?.scopeTags)
      const fileMemory = resolveFileMemory(deps, context?.scopeTags)
      const result = await rememberUserMemory({
        content,
        evidence: evidence?.data,
        tags,
        fileMemory,
        semanticIndex: deps.semanticIndex,
      })

      if (result.status === 'partial') return { status: 'error', code: 'MEMORY_PARTIALLY_SAVED', output: JSON.stringify({ ...result, retryable: true }), durationMs: Date.now() - startedAt }
      if (result.status === 'saved') {
        return {
          status: 'success',
          output: JSON.stringify({
            remembered: true,
            id: result.id,
            content: result.content,
            fileMemoryAdded: result.fileMemoryAdded ?? 0,
            semanticMemorySaved: result.semanticMemorySaved ?? false,
            warnings: result.errors ?? [],
          }),
          durationMs: Date.now() - startedAt,
        }
      }

      const message = result.status === 'too-long' ? 'Memory exceeds 1200 characters. Split it into complete facts without dropping qualifications; nothing was stored.' : result.status === 'sensitive'
        ? 'Sensitive credential-like content was not stored in memory. Use the secret vault/config instead.'
        : result.status === 'empty'
          ? 'No memory content was provided.'
          : result.status === 'unavailable' && !fileMemory && !deps.semanticIndex
            ? 'No persistent memory store is available.'
            : `Memory could not be saved. ${(result.errors ?? []).join('; ')}`

      return {
        status: 'error',
        code: result.status === 'sensitive' ? 'SENSITIVE_MEMORY_USER' : 'MEMORY_UNAVAILABLE_USER',
        output: message,
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryListTool(deps: MemoryListToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.list',
    description: 'Inspect saved memory: file sections plus a scoped, paginated semantic inventory with IDs and evidence, including facts that are not copied into long-term notes. Use includeInactive to inspect expired or candidate memories. A section filter inspects only that file section.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        limit: { type: 'integer', minimum: 1, maximum: 100, description: 'Semantic entries per page (default 20).' },
        offset: { type: 'integer', minimum: 0, description: 'Use nextOffset from the previous page.' },
        includeInactive: { type: 'boolean', description: 'Include expired, candidate, retracted and archived entries for inspection.' },
        section: {
          type: 'string',
          description: 'Optional section title to fetch a single section. Omit to list all sections.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const fileMemory = resolveFileMemory(deps, context?.scopeTags)
      if (!fileMemory && !deps.semanticIndex?.listForScope) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'No persistent memory store is available.',
          durationMs: Date.now() - startedAt,
        }
      }

      const sectionFilter = typeof input.section === 'string' ? input.section.trim() : ''

      if (sectionFilter) {
        const body = await fileMemory?.readMemorySection(sectionFilter)
        if (!body) {
          return {
            status: 'success',
            output: JSON.stringify({ section: sectionFilter, found: false, items: [] }),
            durationMs: Date.now() - startedAt,
          }
        }
        return {
          status: 'success',
          output: JSON.stringify({
            section: sectionFilter,
            found: true,
            items: extractBulletItems(body),
            content: body,
          }),
          durationMs: Date.now() - startedAt,
        }
      }

      const sections: FileMemorySection[] = await fileMemory?.readMemorySections() ?? []
      const inventory = await deps.semanticIndex?.listForScope?.(context?.scopeTags ?? [], {
        limit: typeof input.limit === 'number' && Number.isFinite(input.limit) ? input.limit : 20,
        offset: typeof input.offset === 'number' && Number.isFinite(input.offset) ? input.offset : 0,
        includeInactive: input.includeInactive === true,
      })
      return {
        status: 'success',
        output: JSON.stringify({
          memories: inventory?.memories,
          nextOffset: inventory?.nextOffset,
          sections: sections.map((section) => ({
            title: section.title,
            items: extractBulletItems(section.content),
            content: section.content,
          })),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryForgetTool(deps: MemoryForgetToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.forget',
    description: 'Delete durable user memory. For ALL owned memory use {all:true}: clears long-term notes, daily journals, learned facts, procedures and ingested documents. Use all:true,dryRun:true to inspect counts and retained stores without deleting anything; a preview is a snapshot, not an authorization requirement. The result identifies retained stores. Otherwise either remove a single bullet item from a section (provide `section` and `item`) or clear an entire section (provide only `section`). Call only when the user explicitly asks to delete, clear, forget, or 비워 their memory. This does not affect the current conversation history.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        dryRun: { type: 'boolean', description: 'With all:true, return a non-destructive reset preview. Default false.' },
        all: { type: 'boolean', const: true, description: 'Delete all memory owned by the caller across memory stores. Mutually exclusive with section/item.' },
        section: {
          type: 'string',
          description: 'Section title to target (e.g. "Stable Facts", "Open Loop Queue", "User Memory").',
        },
        item: {
          type: 'string',
          description: 'Optional. The exact bullet text to remove (substring match, case-insensitive). Omit to clear the whole section.',
        },
      },
      anyOf: [{ required: ['section'] }, { required: ['all'] }],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (input.dryRun !== undefined && (typeof input.dryRun !== 'boolean' || input.all !== true)) {
        return { status: 'error', code: 'INVALID_INPUT_PERMANENT', output: 'dryRun requires all:true and a boolean value.', durationMs: Date.now() - startedAt }
      }
      if (input.all !== undefined) {
        if (input.all !== true || input.section !== undefined || input.item !== undefined) {
          return { status: 'error', code: 'INVALID_INPUT_PERMANENT', output: 'Use all:true alone, or section with optional item.', durationMs: Date.now() - startedAt }
        }
        if (!deps.forgetAll) return { status: 'error', code: 'MEMORY_UNAVAILABLE_USER', output: 'Complete memory reset is unavailable; no deletion was performed.', durationMs: Date.now() - startedAt }
        try {
          const result = await deps.forgetAll(context?.scopeTags ?? [], { dryRun: input.dryRun === true })
          return { status: 'success', output: JSON.stringify(result), durationMs: Date.now() - startedAt }
        } catch (error) {
          return { status: 'error', code: 'MEMORY_RESET_FAILED', output: `Memory reset did not complete and may be partial. Retry the same operation. ${String(error)}`, durationMs: Date.now() - startedAt }
        }
      }
      const fileMemory = resolveFileMemory(deps, context?.scopeTags)
      if (!fileMemory) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'No persistent memory store is available.',
          durationMs: Date.now() - startedAt,
        }
      }

      const section = typeof input.section === 'string' ? input.section.trim() : ''
      if (!section) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A section title is required.',
          durationMs: Date.now() - startedAt,
        }
      }

      const itemRaw = typeof input.item === 'string' ? input.item.trim() : ''

      if (itemRaw) {
        const result = await fileMemory.removeMemorySectionItem(section, itemRaw)
        if (result.ambiguous) return { status: 'error', code: 'MEMORY_ITEM_AMBIGUOUS_USER', output: 'Several items match. Use memory.list and provide the exact item text; nothing was deleted.', durationMs: Date.now() - startedAt }
        if (!result.removed) {
          return {
            status: 'error',
            code: 'MEMORY_ITEM_NOT_FOUND_USER',
            output: `No matching item was found in section "${section}". Use memory.list to inspect available items.`,
            durationMs: Date.now() - startedAt,
          }
        }
        return {
          status: 'success',
          output: JSON.stringify({
            section,
            removedItem: itemRaw,
            sectionDeleted: result.sectionDeleted,
            remainingItems: result.remainingItems,
          }),
          durationMs: Date.now() - startedAt,
        }
      }

      const deleted = await fileMemory.deleteMemorySection(section)
      if (!deleted) {
        return {
          status: 'error',
          code: 'MEMORY_SECTION_NOT_FOUND_USER',
          output: `Section "${section}" was not found in durable memory.`,
          durationMs: Date.now() - startedAt,
        }
      }
      return {
        status: 'success',
        output: JSON.stringify({ section, sectionDeleted: true }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemorySearchTool(deps: MemorySearchToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.search',
    description: 'Semantic + keyword search over durable memory (user-saved facts, dreaming-consolidated knowledge, ingested document chunks). Use this whenever the user mentions earlier context, asks "do you remember", references a past decision, or you need to verify what is already stored before answering. Returns top hits with snippet, source, tags, and score.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory', key: (input) => String(input.query ?? '') },
    inputSchema: {
      type: 'object',
      properties: {
        asOf: { type: 'string', format: 'date-time', description: 'Optional full ISO date-time with timezone for historical recall. Omit for the current state; the runtime uses its current clock. Do not supply words such as "now" or invent a timestamp.' },
        includeInactive: { type: 'boolean', description: 'Inspect expired, retracted, or candidate memories explicitly.' },
        query: {
          type: 'string',
          description: 'Natural-language query. Phrase as you would when asking a colleague to recall something — full questions are fine.',
        },
        limit: {
          type: 'number',
          description: `Maximum number of hits to return (default ${SEARCH_DEFAULT_LIMIT}, max ${SEARCH_MAX_LIMIT}).`,
        },
        sources: {
          type: 'array',
          items: { type: 'string', enum: [...ALLOWED_MEMORY_SOURCES] },
          description: 'Restrict to particular sources. Defaults to all sources.',
        },
        tags: {
          type: 'array',
          items: { type: 'string' },
          description: 'Restrict to entries that carry every supplied tag (default AND match, exact, case-sensitive). See tagsLogic to switch to OR.',
        },
        tagsLogic: {
          type: 'string',
          enum: ['and', 'or'],
          description: 'How tags are combined. "and" (default) requires every tag; "or" requires at least one. Useful when the user means "memories about deployment OR k8s".',
        },
        excludeTags: {
          type: 'array',
          items: { type: 'string' },
          description: 'Drop entries that carry any of these tags. Useful to exclude scope, source, or lifecycle markers (e.g. exclude `auto:dreaming` to skip auto-generated summaries).',
        },
        type: {
          type: 'string',
          enum: ['semantic', 'keyword', 'hybrid'],
          description: 'Search mode. Defaults to hybrid (recommended).',
        },
        minScore: {
          type: 'number',
          description: 'Drop hits below this score (0-1).',
        },
        includeAllScopes: {
          type: 'boolean',
          description: 'Include memories from other users / channels too. Default false (return only the active scope + global memories).',
        },
        includeSuperseded: {
          type: 'boolean',
          description: 'Include entries marked superseded by dreaming contradiction-resolution. Default false. Use only when investigating history.',
        },
        includeArchived: {
          type: 'boolean',
          description: 'Include entries archived by dreaming compression (the original long content; the summary stays visible by default). Default false.',
        },
        createdAfter: {
          type: 'string',
          description: 'Only return memories created at or after this timestamp. Accepts ISO-8601 ("2026-05-01T00:00:00Z"), English relative ("yesterday", "5d ago", "2 hours ago", "last week"), or Korean relative ("어제", "5일 전", "지난주").',
        },
        createdBefore: {
          type: 'string',
          description: 'Only return memories created at or before this timestamp. Same formats as createdAfter.',
        },
        sortBy: {
          type: 'string',
          enum: ['score', 'createdAt', 'updatedAt'],
          description: 'Result ordering. Default `score` (similarity rank). `createdAt`/`updatedAt` overrides with newest-first.',
        },
        excludeIds: {
          type: 'array',
          items: { type: 'string' },
          description: 'Memory ids to drop from the results (e.g. ones the user already saw or is editing).',
        },
      },
      required: ['query'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }

      const query = typeof input.query === 'string' ? input.query.trim() : ''
      if (!query) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A query is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      const scopeBypassError = rejectScopedIncludeAll(input, context, startedAt)
      if (scopeBypassError) return scopeBypassError
      const includeSuperseded = input.includeSuperseded === true
      const includeArchived = input.includeArchived === true
      const includeAllScopes = input.includeAllScopes === true

      const options: SemanticSearchOptions = {}
      if (input.includeInactive === true) options.includeInactive = true
      if (typeof input.asOf === 'string') {
        if (!Number.isFinite(Date.parse(input.asOf))) return { status: 'error', code: 'INVALID_INPUT_PERMANENT', output: 'asOf must be an ISO timestamp.', durationMs: Date.now() - startedAt }
        options.asOf = new Date(input.asOf).toISOString()
      }
      const limitInput = typeof input.limit === 'number' ? input.limit : Number(input.limit)
      if (Number.isFinite(limitInput) && limitInput > 0) {
        options.limit = Math.min(Math.floor(limitInput), SEARCH_MAX_LIMIT)
      } else {
        options.limit = SEARCH_DEFAULT_LIMIT
      }
      // Over-fetch a bit so the post-filter still has enough hits to return.
      const requestedLimit = options.limit

      const sources = Array.isArray(input.sources)
        ? input.sources
            .filter((entry): entry is string => typeof entry === 'string')
            .filter((entry): entry is MemoryEntry['source'] =>
              (ALLOWED_MEMORY_SOURCES as readonly string[]).includes(entry),
            )
        : []
      if (sources.length > 0) {
        options.sources = sources
      }

      const tags = Array.isArray(input.tags)
        ? input.tags
            .filter((entry): entry is string => typeof entry === 'string')
            .map((entry) => entry.trim())
            .filter((entry) => entry.length > 0)
        : []
      if (tags.length > 0) {
        options.tags = tags
      }
      const includeAppIndex = explicitlyRequestsAppIndex({ tags })
      const tagsLogic = input.tagsLogic === 'or' || input.tagsLogic === 'and'
        ? input.tagsLogic
        : undefined
      if (tagsLogic) {
        options.tagsLogic = tagsLogic
      }
      const mayReturnAppIndex = !includeAppIndex
        && (!options.sources || options.sources.includes('document'))
        && canTagFilterReturnInternalAppIndex(tags, tagsLogic)
      if (
        (context?.scopeTags && context.scopeTags.length > 0 && !includeAllScopes)
        || mayReturnAppIndex
      ) {
        options.limit = Math.min(requestedLimit * 4, SEARCH_MAX_LIMIT)
      }

      const excludeTags = Array.isArray(input.excludeTags)
        ? input.excludeTags
            .filter((entry): entry is string => typeof entry === 'string')
            .map((entry) => entry.trim())
            .filter((entry) => entry.length > 0)
        : []
      if (excludeTags.length > 0) {
        options.excludeTags = excludeTags
      }

      if (typeof input.type === 'string'
          && (input.type === 'semantic' || input.type === 'keyword' || input.type === 'hybrid')) {
        options.type = input.type
      }

      if (typeof input.minScore === 'number' && Number.isFinite(input.minScore)) {
        options.minScore = input.minScore
      }

      if (typeof input.sortBy === 'string'
          && (input.sortBy === 'score' || input.sortBy === 'createdAt' || input.sortBy === 'updatedAt')) {
        options.sortBy = input.sortBy
      }

      if (typeof input.createdAfter === 'string' && input.createdAfter.trim().length > 0) {
        const parsed = parseRelativePastTimestamp(input.createdAfter)
        if (!parsed) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: `createdAfter "${input.createdAfter}" is not a recognised timestamp. Use ISO-8601 (2026-05-01T00:00:00Z), a relative phrase ("yesterday", "5d ago", "last week"), or a Korean form ("어제", "5일 전", "지난주").`,
            durationMs: Date.now() - startedAt,
          }
        }
        options.createdAfter = parsed.toISOString()
      }
      if (typeof input.createdBefore === 'string' && input.createdBefore.trim().length > 0) {
        const parsed = parseRelativePastTimestamp(input.createdBefore)
        if (!parsed) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: `createdBefore "${input.createdBefore}" is not a recognised timestamp. Use ISO-8601, a relative phrase ("yesterday", "5d ago"), or Korean ("어제", "5일 전").`,
            durationMs: Date.now() - startedAt,
          }
        }
        options.createdBefore = parsed.toISOString()
      }

      let entries: MemoryEntry[]
      try {
        if (!includeAllScopes && context?.scopeTags) options.scopeTags = context.scopeTags
        entries = await deps.semanticIndex.search(query, options)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_SEARCH_FAILED_TRANSIENT',
          output: `Memory search failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }

      let scopeFilteredOut = 0
      let supersededFilteredOut = 0
      let archivedFilteredOut = 0
      let excludedFilteredOut = 0
      let visible = filterAppIndexEntries(entries, includeAppIndex)

      const excludeIds = Array.isArray(input.excludeIds)
        ? new Set(input.excludeIds.filter((id): id is string => typeof id === 'string').map((id) => id.trim()).filter(Boolean))
        : new Set<string>()
      if (excludeIds.size > 0) {
        visible = visible.filter((entry) => {
          if (!excludeIds.has(entry.id)) return true
          excludedFilteredOut += 1
          return false
        })
      }
      if (!includeSuperseded) {
        visible = visible.filter((entry) => {
          if (!isMemorySuperseded(entry.tags)) return true
          supersededFilteredOut += 1
          return false
        })
      }
      if (!includeArchived) {
        visible = visible.filter((entry) => {
          if (!isMemoryArchived(entry.tags)) return true
          archivedFilteredOut += 1
          return false
        })
      }
      if (!includeAllScopes && context?.scopeTags && context.scopeTags.length > 0) {
        visible = visible.filter((entry) => {
          if (isMemoryVisibleInScope(entry.tags, context.scopeTags)) return true
          scopeFilteredOut += 1
          return false
        })
      }
      visible = visible.slice(0, requestedLimit)

      const hits = visible.map((entry) => ({
        id: entry.id,
        snippet: truncateSnippet(entry.content),
        evidence: entry.evidence,
        createdAt: entry.createdAt,
        updatedAt: entry.updatedAt,
        source: entry.source,
        tags: entry.tags,
        score: entry.score,
      }))

      // Best-effort: bump access counter for the ids actually surfaced to
      // the agent. We don't await; failure is silent.
      if (visible.length > 0 && deps.semanticIndex.recordAccess) {
        void deps.semanticIndex.recordAccess(visible.map((entry) => entry.id)).catch(() => {})
      }

      return {
        status: 'success',
        output: JSON.stringify({
          query,
          limit: requestedLimit,
          totalHits: hits.length,
          scopeFilteredOut: context?.scopeTags?.length ? 0 : scopeFilteredOut,
          supersededFilteredOut,
          archivedFilteredOut,
          excludedFilteredOut,
          hits,
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryGraphSearchTool(deps: MemoryGraphToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.graph.search',
    description: 'Search the durable memory knowledge graph extracted by dreaming. Returns entity nodes, typed relationships, confidence, tags, and evidence memory ids. Use when a question is about relationships, preferences, projects, decisions, or "what do you know about X". Read-only.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory.graph', key: (input) => String(input.query ?? '') },
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Optional text to match against graph labels, aliases, tags, relations, or evidence ids. Omit or pass empty string for recent graph items.',
        },
        kind: {
          type: 'string',
          description: 'Optional node kind filter, e.g. person, project, preference, tool, topic, decision, constraint, fact, organization, place.',
        },
        limit: {
          type: 'number',
          description: 'Maximum graph nodes/edges to return (default 20, max 100).',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const scopeError = rejectScopedGraphAccess(context, startedAt)
      if (scopeError) return scopeError
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_GRAPH_UNAVAILABLE_USER',
          output: 'Memory graph is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }

      const query = typeof input.query === 'string' ? input.query.trim() : ''
      const kind = typeof input.kind === 'string' && input.kind.trim().length > 0
        ? input.kind.trim()
        : undefined
      const limitInput = typeof input.limit === 'number' ? input.limit : Number(input.limit)
      const limit = Number.isFinite(limitInput) && limitInput > 0
        ? Math.min(Math.floor(limitInput), 100)
        : 20

      try {
        const [stats, graph] = await Promise.all([
          deps.semanticIndex.getGraphStats(),
          deps.semanticIndex.searchGraph(query, { kind, limit }),
        ])
        return {
          status: 'success',
          output: JSON.stringify({
            query,
            kind,
            limit,
            stats,
            nodes: graph.nodes.map(serializeMemoryGraphNode),
            edges: graph.edges.map(serializeMemoryGraphEdge),
          }),
          durationMs: Date.now() - startedAt,
        }
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_GRAPH_SEARCH_FAILED_TRANSIENT',
          output: `Memory graph search failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
    },
  }
}

export function createMemoryGraphNeighborsTool(deps: MemoryGraphToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.graph.neighbors',
    description: 'Return neighboring graph nodes and relationships for a memory graph node id from memory.graph.search. Use to inspect why an entity is connected to preferences, tools, projects, decisions, or constraints. Read-only.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory.graph', key: (input) => String(input.id ?? '') },
    inputSchema: {
      type: 'object',
      properties: {
        id: {
          type: 'string',
          description: 'Memory graph node id returned by memory.graph.search.',
        },
        direction: {
          type: 'string',
          enum: ['in', 'out', 'both'],
          description: 'Relationship direction to traverse. Default both.',
        },
        limit: {
          type: 'number',
          description: 'Maximum neighboring edges to return (default 20, max 100).',
        },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const scopeError = rejectScopedGraphAccess(context, startedAt)
      if (scopeError) return scopeError
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_GRAPH_UNAVAILABLE_USER',
          output: 'Memory graph is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }

      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (!id) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A graph node id is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      const direction = input.direction === 'in' || input.direction === 'out' || input.direction === 'both'
        ? input.direction
        : 'both'
      const limitInput = typeof input.limit === 'number' ? input.limit : Number(input.limit)
      const limit = Number.isFinite(limitInput) && limitInput > 0
        ? Math.min(Math.floor(limitInput), 100)
        : 20

      try {
        const graph = await deps.semanticIndex.getGraphNeighbors(id, { direction, limit })
        return {
          status: 'success',
          output: JSON.stringify({
            id,
            direction,
            limit,
            node: graph.node ? serializeMemoryGraphNode(graph.node) : null,
            nodes: graph.nodes.map(serializeMemoryGraphNode),
            edges: graph.edges.map(serializeMemoryGraphEdge),
          }),
          durationMs: Date.now() - startedAt,
        }
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_GRAPH_NEIGHBORS_FAILED_TRANSIENT',
          output: `Memory graph neighbors failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
    },
  }
}

export function createMemoryGraphPageTool(deps: MemoryGraphToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.graph.page',
    description: 'Return an entity-centric wiki page from the durable memory graph: canonical node, aliases, incoming/outgoing relationships, and evidence memory snippets. Use this before answering relationship-heavy recall questions or when the user asks what the daemon knows about a person, project, tool, preference, decision, or topic. Read-only.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory.graph', key: (input) => String(input.id ?? input.query ?? '') },
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Entity/topic text to resolve to a graph node. Required unless id is supplied.',
        },
        id: {
          type: 'string',
          description: 'Exact graph node id from memory.graph.search. Takes precedence over query.',
        },
        limit: {
          type: 'number',
          description: 'Maximum neighboring relationships to include (default 12, max 100).',
        },
        evidenceLimit: {
          type: 'number',
          description: 'Maximum evidence memory snippets to include (default 12, max 100).',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const scopeError = rejectScopedGraphAccess(context, startedAt)
      if (scopeError) return scopeError
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_GRAPH_UNAVAILABLE_USER',
          output: 'Memory graph is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }

      const id = typeof input.id === 'string' ? input.id.trim() : ''
      const query = typeof input.query === 'string' ? input.query.trim() : ''
      if (!id && !query) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A graph node id or query is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      const limit = clampToolLimit(input.limit, 12, 100)
      const evidenceLimit = clampToolLimit(input.evidenceLimit, 12, 100)

      try {
        const page = await deps.semanticIndex.getGraphWikiPage({
          nodeId: id || undefined,
          query: query || undefined,
          limit,
          evidenceLimit,
        })
        return {
          status: 'success',
          output: JSON.stringify(serializeMemoryGraphWikiPage(page)),
          durationMs: Date.now() - startedAt,
        }
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_GRAPH_PAGE_FAILED_TRANSIENT',
          output: `Memory graph page failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
    },
  }
}

export function createMemoryGraphAuditTool(deps: MemoryGraphToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.graph.audit',
    description: 'Inspect durable memory graph quality without mutating it. Returns low-confidence, thin-evidence, missing-evidence, stale, orphan, and contradiction signals so the agent can decide whether graph facts are trustworthy. Read-only.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory.graph' },
    inputSchema: {
      type: 'object',
      properties: {
        limit: {
          type: 'number',
          description: 'Maximum nodes and edges to scan for quality signals (default 200, max 1000).',
        },
        signalLimit: {
          type: 'number',
          description: 'Maximum signals to return (default 50, max 500).',
        },
        lowConfidenceThreshold: {
          type: 'number',
          description: 'Confidence below this threshold is reported as low confidence (default 0.45).',
        },
        staleAfterDays: {
          type: 'number',
          description: 'Last-seen age threshold for stale nodes/edges (default 180 days).',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const scopeError = rejectScopedGraphAccess(context, startedAt)
      if (scopeError) return scopeError
      if (!deps.semanticIndex?.inspectGraphQuality) {
        return {
          status: 'error',
          code: 'MEMORY_GRAPH_UNAVAILABLE_USER',
          output: 'Memory graph quality inspection is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }

      const lowConfidenceInput = typeof input.lowConfidenceThreshold === 'number'
        ? input.lowConfidenceThreshold
        : Number(input.lowConfidenceThreshold)
      const staleInput = typeof input.staleAfterDays === 'number'
        ? input.staleAfterDays
        : Number(input.staleAfterDays)

      try {
        const report = await deps.semanticIndex.inspectGraphQuality({
          limit: clampToolLimit(input.limit, 200, 1000),
          signalLimit: clampToolLimit(input.signalLimit, 50, 500),
          lowConfidenceThreshold: Number.isFinite(lowConfidenceInput)
            ? Math.max(0, Math.min(1, lowConfidenceInput))
            : undefined,
          staleAfterDays: Number.isFinite(staleInput) && staleInput > 0
            ? Math.min(Math.floor(staleInput), 3650)
            : undefined,
        })
        return {
          status: 'success',
          output: JSON.stringify(serializeMemoryGraphQualityReport(report)),
          durationMs: Date.now() - startedAt,
        }
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_GRAPH_AUDIT_FAILED_TRANSIENT',
          output: `Memory graph audit failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
    },
  }
}

export function createMemoryGraphRepairTool(deps: MemoryGraphToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.graph.repair',
    description: 'Build a daemon-wide memory graph repair plan from quality signals. Defaults to dryRun=true. With dryRun=false it only applies deterministic safe pruning for graph entries with no active evidence; contradiction/evidence relink work remains a proposal for memory.graph.repair.apply.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory.graph' },
    inputSchema: {
      type: 'object',
      properties: {
        dryRun: {
          type: 'boolean',
          description: 'Preview only by default. Set false only after the user asks to apply safe graph cleanup.',
        },
        limit: {
          type: 'number',
          description: 'Maximum nodes and edges to scan for quality signals (default 200, max 1000).',
        },
        signalLimit: {
          type: 'number',
          description: 'Maximum quality signals to convert into repair proposals (default 50, max 500).',
        },
        lowConfidenceThreshold: {
          type: 'number',
          description: 'Confidence below this threshold is reported as low confidence (default 0.45).',
        },
        staleAfterDays: {
          type: 'number',
          description: 'Last-seen age threshold for stale nodes/edges (default 180 days).',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const scopeError = rejectScopedGraphAccess(context, startedAt)
      if (scopeError) return scopeError
      if (!deps.semanticIndex?.repairGraphQuality) {
        return {
          status: 'error',
          code: 'MEMORY_GRAPH_UNAVAILABLE_USER',
          output: 'Memory graph repair is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }

      const lowConfidenceInput = typeof input.lowConfidenceThreshold === 'number'
        ? input.lowConfidenceThreshold
        : Number(input.lowConfidenceThreshold)
      const staleInput = typeof input.staleAfterDays === 'number'
        ? input.staleAfterDays
        : Number(input.staleAfterDays)

      try {
        const result = await deps.semanticIndex.repairGraphQuality({
          dryRun: input.dryRun !== false,
          limit: clampToolLimit(input.limit, 200, 1000),
          signalLimit: clampToolLimit(input.signalLimit, 50, 500),
          lowConfidenceThreshold: Number.isFinite(lowConfidenceInput)
            ? Math.max(0, Math.min(1, lowConfidenceInput))
            : undefined,
          staleAfterDays: Number.isFinite(staleInput) && staleInput > 0
            ? Math.min(Math.floor(staleInput), 3650)
            : undefined,
        })
        return {
          status: 'success',
          output: JSON.stringify(serializeMemoryGraphRepairResult(result)),
          durationMs: Date.now() - startedAt,
        }
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_GRAPH_REPAIR_FAILED_TRANSIENT',
          output: `Memory graph repair failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
    },
  }
}

export function createMemoryGraphRepairApplyTool(deps: MemoryGraphToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.graph.repair.apply',
    description: 'Apply an explicit memory graph repair decision. Use only after inspecting memory.graph.repair proposals and identifying the winner memory plus loser memories. Defaults to dryRun=true; with dryRun=false it tags loser memories as superseded-by:<winner> and then prunes unsupported graph facts.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory.graph' },
    inputSchema: {
      type: 'object',
      properties: {
        winnerMemoryId: {
          type: 'string',
          description: 'Memory id that should remain active as the winning fact.',
        },
        supersededMemoryIds: {
          type: 'array',
          items: { type: 'string' },
          description: 'Memory ids to mark as superseded by winnerMemoryId.',
        },
        proposalId: {
          type: 'string',
          description: 'Optional repair proposal id from memory.graph.repair.',
        },
        dryRun: {
          type: 'boolean',
          description: 'Preview only by default. Set false only after the user explicitly approves this exact winner/loser decision.',
        },
        reason: {
          type: 'string',
          description: 'Short audit note explaining the repair decision.',
        },
      },
      required: ['winnerMemoryId', 'supersededMemoryIds'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const scopedGraphError = rejectScopedGraphAccess(context, startedAt)
      if (scopedGraphError) return scopedGraphError
      if (!deps.semanticIndex?.applyGraphRepairDecision) {
        return {
          status: 'error',
          code: 'MEMORY_GRAPH_UNAVAILABLE_USER',
          output: 'Memory graph repair decisions are not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }

      const winnerMemoryId = typeof input.winnerMemoryId === 'string' ? input.winnerMemoryId.trim() : ''
      const supersededMemoryIds = Array.isArray(input.supersededMemoryIds)
        ? input.supersededMemoryIds
            .filter((id): id is string => typeof id === 'string')
            .map((id) => id.trim())
            .filter(Boolean)
        : []
      if (!winnerMemoryId || supersededMemoryIds.length === 0) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'winnerMemoryId and at least one supersededMemoryIds entry are required.',
          durationMs: Date.now() - startedAt,
        }
      }

      const scopeError = await validateGraphRepairDecisionScope(
        deps,
        context,
        winnerMemoryId,
        supersededMemoryIds,
        startedAt,
      )
      if (scopeError) return scopeError

      try {
        const result = await deps.semanticIndex.applyGraphRepairDecision({
          winnerMemoryId,
          supersededMemoryIds,
          proposalId: typeof input.proposalId === 'string' && input.proposalId.trim().length > 0
            ? input.proposalId.trim()
            : undefined,
          dryRun: input.dryRun !== false,
          reason: typeof input.reason === 'string' && input.reason.trim().length > 0
            ? input.reason.trim()
            : undefined,
          actor: 'agent',
        })
        return {
          status: 'success',
          output: JSON.stringify(serializeMemoryGraphRepairDecisionResult(result)),
          durationMs: Date.now() - startedAt,
        }
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_GRAPH_REPAIR_APPLY_FAILED_TRANSIENT',
          output: `Memory graph repair decision failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
    },
  }
}

export function createMemoryDailySearchTool(deps: MemoryDailyToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.daily.search',
    description: 'Grep across daily notes (case-insensitive) and return matching lines with their date and section. Use to recover a specific note when the user remembers the topic but not the date. Read-only.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory.daily', key: (input) => String(input.query ?? '') },
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Substring to search for (case-insensitive).' },
        maxMatches: { type: 'number', description: 'Max matching lines to return (default 25, max 200).' },
        withinDays: { type: 'integer', minimum: 1, description: 'Optional limit on recorded dates, not elapsed days. Default searches all history in bounded pages.' },
        cursor: { type: 'string', description: 'nextCursor from the previous page. Continue even if that page has no matches. Restart after concurrent journal edits.' },
        since: { type: 'string', description: 'Inclusive YYYY-MM-DD lower date boundary.' },
        until: { type: 'string', description: 'Inclusive YYYY-MM-DD upper date boundary.' },
        contextLines: { type: 'number', description: 'Include N surrounding lines as contextBefore / contextAfter (default 0, max 10).' },
      },
      required: ['query'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const fileMemory = resolveFileMemory(deps, context?.scopeTags)
      if (!fileMemory || typeof fileMemory.searchDailyNotes !== 'function') {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'No persistent memory store is available.',
          durationMs: Date.now() - startedAt,
        }
      }
      const query = typeof input.query === 'string' ? input.query.trim() : ''
      if (!query) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A query is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      const maxMatches = typeof input.maxMatches === 'number' ? input.maxMatches : undefined
      const withinDays = typeof input.withinDays === 'number' ? input.withinDays : undefined
      const contextLines = typeof input.contextLines === 'number' ? input.contextLines : undefined
      let hits: Awaited<ReturnType<NonNullable<typeof fileMemory.searchDailyNotes>>>
      let page: { nextCursor: string | null; scannedDates: number } | undefined
      try {
        if (fileMemory.searchDailyNotesPage) {
          for (const key of ['cursor', 'since', 'until'] as const) {
            if (input[key] !== undefined && typeof input[key] !== 'string') throw new Error(`${key} must be a string`)
          }
          const result = await fileMemory.searchDailyNotesPage(query, { maxMatches, withinDays, contextLines,
            cursor: input.cursor as string | undefined, since: input.since as string | undefined, until: input.until as string | undefined })
          hits = result.matches
          page = result
        } else {
          if (input.cursor !== undefined || input.since !== undefined || input.until !== undefined) throw new Error('This memory store does not support paged date-range search')
          hits = await fileMemory.searchDailyNotes(query, { maxMatches, withinDays, contextLines })
        }
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_DAILY_SEARCH_FAILED_TRANSIENT',
          output: `Daily-note search failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
      return {
        status: 'success',
        output: JSON.stringify({
          query,
          totalMatches: hits.length,
          matches: hits,
          ...(page ? { nextCursor: page.nextCursor, scannedDates: page.scannedDates, complete: page.nextCursor === null } : { complete: false, legacyWindow: withinDays ?? 90 }),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryDailyListTool(deps: MemoryDailyToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.daily.list',
    description: 'Enumerate the dates of recent daily notes (most recent first) so you can decide which one to memory.daily.read. Pulls from the active scope\'s file-memory bucket; returns just the YYYY-MM-DD keys plus a path. Optional since/until narrows to a date range — accepts ISO/YYYY-MM-DD or relative phrases ("last week", "5d ago", "지난주").',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory.daily' },
    inputSchema: {
      type: 'object',
      properties: {
        limit: {
          type: 'number',
          description: 'Maximum daily-note dates to return (default 14, max 90). Applied after the since/until filter.',
        },
        since: {
          type: 'string',
          description: 'Only return notes on or after this date. YYYY-MM-DD, ISO-8601, or relative ("last week", "5d ago", "지난주").',
        },
        until: {
          type: 'string',
          description: 'Only return notes on or before this date. Same formats as since.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const fileMemory = resolveFileMemory(deps, context?.scopeTags)
      if (!fileMemory) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'No persistent memory store is available.',
          durationMs: Date.now() - startedAt,
        }
      }
      const limit = typeof input.limit === 'number' ? input.limit : 14

      let sinceKey: string | undefined
      if (typeof input.since === 'string' && input.since.trim()) {
        const parsed = parseRelativePastTimestamp(input.since)
        if (!parsed) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: `since "${input.since}" is not a recognised date (YYYY-MM-DD, ISO-8601, "last week", "지난주").`,
            durationMs: Date.now() - startedAt,
          }
        }
        sinceKey = formatDateKey(startOfDay(parsed))
      }
      let untilKey: string | undefined
      if (typeof input.until === 'string' && input.until.trim()) {
        const parsed = parseRelativePastTimestamp(input.until)
        if (!parsed) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: `until "${input.until}" is not a recognised date (YYYY-MM-DD, ISO-8601, "yesterday", "어제").`,
            durationMs: Date.now() - startedAt,
          }
        }
        untilKey = formatDateKey(startOfDay(parsed))
      }

      let dateKeys: string[]
      try {
        // Over-fetch when filtering so the post-filter still hits `limit`.
        const fetchLimit = (sinceKey !== undefined || untilKey !== undefined)
          ? Math.min(Math.max(limit, 14) * 6, 365)
          : limit
        dateKeys = await fileMemory.listDailyNoteDates(fetchLimit)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_DAILY_LIST_FAILED_TRANSIENT',
          output: `Failed to list daily notes: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }

      let filtered = dateKeys
      if (sinceKey !== undefined || untilKey !== undefined) {
        filtered = dateKeys.filter((dateKey) => {
          const date = parseDailyDate(dateKey)
          if (!date) return false
          if (sinceKey !== undefined && dateKey < sinceKey) return false
          if (untilKey !== undefined && dateKey > untilKey) return false
          return true
        })
      }
      filtered = filtered.slice(0, limit)

      return {
        status: 'success',
        output: JSON.stringify({
          totalDates: filtered.length,
          dates: await Promise.all(filtered.map(async (dateKey) => {
            const date = parseDailyDate(dateKey)
            return {
              date: dateKey,
              path: date ? await fileMemory.getDailyNoteReadPath?.(date) ?? fileMemory.getDailyNotePath(date) : undefined,
            }
          })),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryDailyReadTool(deps: MemoryDailyToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.daily.read',
    description: 'Read today\'s, yesterday\'s, or any specific date\'s daily note (or a single section within it). Use when the user asks "what did we do yesterday", references an Open Loop item, or you want to see your prior reflection notes.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory.daily' },
    inputSchema: {
      type: 'object',
      properties: {
        date: {
          type: 'string',
          description: 'today (default), yesterday, or YYYY-MM-DD.',
        },
        section: {
          type: 'string',
          description: 'Optional ## section title to read only that part.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const fileMemory = resolveFileMemory(deps, context?.scopeTags)
      if (!fileMemory) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'No persistent memory store is available.',
          durationMs: Date.now() - startedAt,
        }
      }

      const date = parseDailyDate(input.date)
      if (!date) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'date must be "today", "yesterday", or YYYY-MM-DD.',
          durationMs: Date.now() - startedAt,
        }
      }

      const section = typeof input.section === 'string' ? input.section.trim() : ''
      const path = await fileMemory.getDailyNoteReadPath?.(date) ?? fileMemory.getDailyNotePath(date)
      if (section) {
        const body = await fileMemory.readDailySection(section, date)
        return {
          status: 'success',
          output: JSON.stringify({
            date: formatDateKey(date),
            path,
            section,
            found: Boolean(body),
            content: body ?? '',
          }),
          durationMs: Date.now() - startedAt,
        }
      }

      const note = await fileMemory.readDailyNote(date)
      return {
        status: 'success',
        output: JSON.stringify({
          date: formatDateKey(date),
          path,
          found: Boolean(note),
          content: note ?? '',
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryDailyAppendTool(deps: MemoryDailyToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.daily.append',
    description: 'Append a paragraph or bullet to a section of a daily note. Use to record decisions, open questions, reflections, or backlog items as you work. Creates the section if missing.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory.daily' },
    inputSchema: {
      type: 'object',
      properties: {
        section: {
          type: 'string',
          description: 'Section title (e.g. "Open Loop Queue", "Reflection Ledger", "Tasks").',
        },
        entry: {
          type: 'string',
          description: 'Entry text. Can be multi-line. Will be appended below the existing content with a blank line separator.',
        },
        date: {
          type: 'string',
          description: 'today (default), yesterday, or YYYY-MM-DD.',
        },
      },
      required: ['section', 'entry'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const fileMemory = resolveFileMemory(deps, context?.scopeTags)
      if (!fileMemory) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'No persistent memory store is available.',
          durationMs: Date.now() - startedAt,
        }
      }

      const section = typeof input.section === 'string' ? input.section.trim() : ''
      const entry = typeof input.entry === 'string' ? input.entry.trim() : ''
      if (!section || !entry) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'Both section and entry are required.',
          durationMs: Date.now() - startedAt,
        }
      }

      const date = parseDailyDate(input.date)
      if (!date) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'date must be "today", "yesterday", or YYYY-MM-DD.',
          durationMs: Date.now() - startedAt,
        }
      }

      await fileMemory.appendToDailySection(section, entry, date)
      return {
        status: 'success',
        output: JSON.stringify({
          date: formatDateKey(date),
          section,
          appendedChars: entry.length,
          path: fileMemory.getDailyNotePath(date),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryDailyReplaceTool(deps: MemoryDailyToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.daily.replace',
    description: 'Replace a section of a daily note with new content. Use when consolidating, summarizing, or correcting a section in place. Empty content has no effect (use memory.forget-style cleanup elsewhere).',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory.daily' },
    inputSchema: {
      type: 'object',
      properties: {
        section: { type: 'string', description: 'Section title to overwrite.' },
        content: { type: 'string', description: 'New section body (markdown allowed).' },
        date: {
          type: 'string',
          description: 'today (default), yesterday, or YYYY-MM-DD.',
        },
      },
      required: ['section', 'content'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const fileMemory = resolveFileMemory(deps, context?.scopeTags)
      if (!fileMemory) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'No persistent memory store is available.',
          durationMs: Date.now() - startedAt,
        }
      }

      const section = typeof input.section === 'string' ? input.section.trim() : ''
      const content = typeof input.content === 'string' ? input.content.trim() : ''
      if (!section) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A section title is required.',
          durationMs: Date.now() - startedAt,
        }
      }

      const date = parseDailyDate(input.date)
      if (!date) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'date must be "today", "yesterday", or YYYY-MM-DD.',
          durationMs: Date.now() - startedAt,
        }
      }

      if (!content) {
        return {
          status: 'success',
          output: JSON.stringify({
            date: formatDateKey(date),
            section,
            replaced: false,
            note: 'Empty content was ignored. Use a non-empty body to overwrite the section.',
          }),
          durationMs: Date.now() - startedAt,
        }
      }

      await fileMemory.replaceDailySection(section, content, date)
      return {
        status: 'success',
        output: JSON.stringify({
          date: formatDateKey(date),
          section,
          replaced: true,
          chars: content.length,
          path: fileMemory.getDailyNotePath(date),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export interface MemorySectionToolDeps {
  fileMemory?: Pick<FileMemory, 'replaceMemorySection' | 'readMemorySection' | 'getMemoryPath'>
  fileMemoryRegistry?: ScopedFileMemoryRegistry
}

export function createMemorySectionReplaceTool(deps: MemorySectionToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.section.replace',
    description:
      'Rewrite a section of long-term memory (MEMORY.md) wholesale with new content — the counterpart of memory.daily.replace for the durable file rather than a daily note. ' +
      'Use when consolidating or correcting a whole section in place, e.g. refreshing a "User Profile" / "Stable Facts" section after reflecting on recent conversations. ' +
      'For appending a single new fact, prefer memory.remember; for deleting a section or one bullet, use memory.forget. Empty content is ignored (it will not blank the section).',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory.file' },
    inputSchema: {
      type: 'object',
      properties: {
        section: { type: 'string', description: 'Section title to overwrite (e.g. "User Profile").' },
        content: { type: 'string', description: 'New section body (markdown allowed). Bullets recommended for facts.' },
      },
      required: ['section', 'content'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const fileMemory = resolveFileMemory(deps, context?.scopeTags)
      if (!fileMemory) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'No persistent memory store is available.',
          durationMs: Date.now() - startedAt,
        }
      }

      const section = typeof input.section === 'string' ? input.section.trim() : ''
      const content = typeof input.content === 'string' ? input.content.trim() : ''
      if (!section) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A section title is required.',
          durationMs: Date.now() - startedAt,
        }
      }

      if (!content) {
        return {
          status: 'success',
          output: JSON.stringify({
            section,
            replaced: false,
            note: 'Empty content was ignored. Use a non-empty body to overwrite the section, or memory.forget to clear it.',
          }),
          durationMs: Date.now() - startedAt,
        }
      }

      const previous = await fileMemory.readMemorySection(section)
      await fileMemory.replaceMemorySection(section, content)
      return {
        status: 'success',
        output: JSON.stringify({
          section,
          replaced: true,
          chars: content.length,
          previousChars: previous?.length ?? 0,
          created: previous == null,
          path: fileMemory.getMemoryPath(),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryUpdateTool(deps: MemoryUpdateToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.update',
    description: 'Edit an existing memory in place by ID, preserving its semantic identity, embeddings continuity, and audit chain. Use when correcting a fact you previously stored, refining wording, or adjusting tags — call memory.search first to obtain the id. Do NOT use this for unrelated new facts (use memory.remember).',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        evidence: memoryEvidenceInputSchema,
        id: {
          type: 'string',
          description: 'Memory id (returned by memory.search hits[].id).',
        },
        content: {
          type: 'string',
          description: 'Replacement content. Omit to keep the existing content.',
        },
        tags: {
          type: 'array',
          items: { type: 'string' },
          description: 'Replace user tags only. Omit when changing content without retagging. System tags returned by search (scope:*, archived*, superseded*, compressed-from:*) are preserved by the runtime and must not be copied into inputs. Mutually exclusive with addTags / removeTags.',
        },
        addTags: {
          type: 'array',
          items: { type: 'string' },
          description: 'User tags to add. Do not supply runtime-managed scope:*, archived*, superseded*, or compressed-from:* tags.',
        },
        removeTags: {
          type: 'array',
          items: { type: 'string' },
          description: 'User tags to remove. Runtime-managed scope and lifecycle tags cannot be removed here.',
        },
        reason: {
          type: 'string',
          description: 'Short note recorded in the audit log (why this change was made).',
        },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const evidence = input.evidence === undefined ? undefined : memoryEvidenceSchema.safeParse(input.evidence)
      if (evidence && !evidence.success) return { status: 'error', code: 'INVALID_INPUT_PERMANENT', output: 'Invalid memory evidence or validity interval.', durationMs: Date.now() - startedAt }
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }

      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (!id) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A memory id is required.',
          durationMs: Date.now() - startedAt,
        }
      }

      const existing = await deps.semanticIndex.get(id)
      if (!existing) {
        return {
          status: 'error',
          code: 'MEMORY_ID_NOT_FOUND_USER',
          output: `No memory with id "${id}" was found.`,
          durationMs: Date.now() - startedAt,
        }
      }

      if (!isMemoryWritableInScope(existing.tags, context?.scopeTags)) {
        return {
          status: 'error',
          code: 'MEMORY_SCOPE_MISMATCH_USER',
          output: 'This memory belongs to a different user/channel scope and cannot be edited from here.',
          durationMs: Date.now() - startedAt,
        }
      }

      const explicitTags = Array.isArray(input.tags)
        ? input.tags.filter((tag): tag is string => typeof tag === 'string').map((tag) => tag.trim()).filter(Boolean)
        : null
      const addTags = Array.isArray(input.addTags)
        ? input.addTags.filter((tag): tag is string => typeof tag === 'string').map((tag) => tag.trim()).filter(Boolean)
        : []
      const removeTags = Array.isArray(input.removeTags)
        ? input.removeTags.filter((tag): tag is string => typeof tag === 'string').map((tag) => tag.trim()).filter(Boolean)
        : []

      if (explicitTags && (addTags.length > 0 || removeTags.length > 0)) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: '`tags` is mutually exclusive with `addTags` / `removeTags`.',
          durationMs: Date.now() - startedAt,
        }
      }

      // Reject foreign-scope / lifecycle tag injection from user-supplied tags.
      // Without this, `tags: ['scope:user:<other>']` re-owned the memory into
      // another scope, and `archived` / `superseded` arbitrarily hid it.
      const forbiddenUpdateTags = findForbiddenUserTags([...(explicitTags ?? []), ...addTags, ...removeTags])
      if (forbiddenUpdateTags.length > 0) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: `Reserved/system tags cannot be set directly: ${forbiddenUpdateTags.join(', ')}. Scope ownership is preserved automatically; lifecycle tags are managed by dedicated operations.`,
          durationMs: Date.now() - startedAt,
        }
      }

      const existingScopeTags = (existing.tags ?? []).filter(isReservedTag)
      let nextTags: string[]
      if (explicitTags) {
        // Preserve existing scope tags so the agent can't accidentally strip
        // ownership when rewriting tag list.
        nextTags = uniqueOrdered([...explicitTags, ...existingScopeTags])
      } else {
        const removeSet = new Set(removeTags.map((tag) => tag.toLowerCase()))
        const kept = (existing.tags ?? []).filter((tag) => {
          if (isReservedTag(tag)) return true
          return !removeSet.has(tag.toLowerCase())
        })
        nextTags = uniqueOrdered([...kept, ...addTags])
      }

      const nextContent = typeof input.content === 'string' && input.content.trim().length > 0
        ? input.content.trim()
        : existing.content

      if (nextContent !== existing.content && looksSensitive(nextContent)) {
        return { status: 'error', code: 'SENSITIVE_MEMORY_USER', output: 'Sensitive credential-like content was not stored. Use the secret vault/config instead.', durationMs: Date.now() - startedAt }
      }

      const unchanged = input.evidence === undefined && nextContent === existing.content
        && tagsEqual(nextTags, existing.tags ?? [])
      if (unchanged) {
        return {
          status: 'success',
          output: JSON.stringify({
            id: existing.id,
            updated: false,
            reason: 'No changes detected.',
          }),
          durationMs: Date.now() - startedAt,
        }
      }

      try {
        await deps.semanticIndex.add({
          id: existing.id,
          evidence: evidence?.data ?? (nextContent !== existing.content && existing.evidence
            ? { ...existing.evidence, observedAt: new Date().toISOString() } : undefined),
          content: nextContent,
          source: existing.source,
          tags: nextTags,
        })
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_UPDATE_FAILED_TRANSIENT',
          output: `Failed to persist update: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }

      try {
        await deps.semanticIndex.recordAudit({
          memoryId: existing.id,
          action: 'updated',
          actor: 'agent',
          reason: typeof input.reason === 'string' && input.reason.trim().length > 0
            ? input.reason.trim()
            : undefined,
          before: {
            evidence: existing.evidence,
            id: existing.id,
            content: existing.content,
            source: existing.source,
            tags: existing.tags ?? [],
          },
          after: {
            evidence: (await deps.semanticIndex.get(existing.id))?.evidence,
            id: existing.id,
            content: nextContent,
            source: existing.source,
            tags: nextTags,
          },
        })
      } catch {
        // Audit failure shouldn't break the update — surface a warning instead.
      }

      return {
        status: 'success',
        output: JSON.stringify({
          id: existing.id,
          updated: true,
          before: {
            content: existing.content,
            tags: existing.tags ?? [],
          },
          after: {
            content: nextContent,
            tags: nextTags,
          },
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryDocumentsIngestTool(
  deps: MemoryDocumentsToolDeps,
): ToolDefinitionRuntime {
  return {
    name: 'memory.documents.ingest',
    description: 'Ingest a document (markdown, text, or any text-extracted content) into the durable knowledge store so future memory.documents.search and memory.search calls can retrieve it. Provide a clear title — it appears in citations.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory.documents' },
    inputSchema: {
      type: 'object',
      properties: {
        title: { type: 'string', description: 'Human-readable title for citations.' },
        content: { type: 'string', description: 'Full document content.' },
        path: { type: 'string', description: 'Optional source path (file or URL).' },
        mimeType: { type: 'string', description: 'Optional MIME type, e.g. text/markdown.' },
        tags: {
          type: 'array',
          items: { type: 'string' },
          description: 'Optional tags applied to every chunk.',
        },
      },
      required: ['title', 'content'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.documentStore) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Document memory store is not configured.',
          durationMs: Date.now() - startedAt,
        }
      }

      const title = typeof input.title === 'string' ? input.title.trim() : ''
      const content = typeof input.content === 'string' ? input.content : ''
      if (!title || !content.trim()) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'Both title and non-empty content are required.',
          durationMs: Date.now() - startedAt,
        }
      }

      const contentBytes = Buffer.byteLength(content, 'utf8')
      if (contentBytes > MAX_DOCUMENT_INGEST_BYTES) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: `Document content is too large: ${contentBytes} bytes (max ${MAX_DOCUMENT_INGEST_BYTES}). Split it into smaller documents.`,
          durationMs: Date.now() - startedAt,
        }
      }

      const userTags = Array.isArray(input.tags)
        ? input.tags.filter((tag): tag is string => typeof tag === 'string').map((t) => t.trim()).filter(Boolean)
        : []
      const tags = attachScopeTags(userTags, context?.scopeTags)
      const ingestInput: DocumentIngestInput = {
        title,
        content,
        path: typeof input.path === 'string' ? input.path : undefined,
        mimeType: typeof input.mimeType === 'string' ? input.mimeType : undefined,
        tags: tags.length > 0 ? tags : undefined,
      }

      let document: MemoryDocument
      try {
        document = await deps.documentStore.ingestDocument(ingestInput)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_INGEST_FAILED_TRANSIENT',
          output: `Document ingest failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }

      return {
        status: 'success',
        output: JSON.stringify({
          documentId: document.id,
          title: document.title,
          chunkCount: document.chunkCount,
          tags: document.tags,
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryDocumentsSearchTool(
  deps: MemoryDocumentsToolDeps,
): ToolDefinitionRuntime {
  return {
    name: 'memory.documents.search',
    description: 'Search ingested documents for chunks relevant to a question. Returns chunk snippets with citation labels you can quote (e.g. "see [docTitle §3]"). Prefer this over memory.search when the user is asking about content that was explicitly ingested as a document.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory.documents', key: (input) => String(input.query ?? '') },
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Natural-language query.' },
        limit: { type: 'number', description: `Default ${SEARCH_DEFAULT_LIMIT}, max ${SEARCH_MAX_LIMIT}.` },
        documentId: { type: 'string', description: 'Restrict to a single document.' },
        tags: { type: 'array', items: { type: 'string' }, description: 'Restrict by tags.' },
        type: { type: 'string', enum: ['semantic', 'keyword', 'hybrid'] },
        minScore: { type: 'number' },
        sortBy: {
          type: 'string',
          enum: ['score', 'createdAt', 'updatedAt'],
          description: 'Result ordering. Default `score`.',
        },
        includeAllScopes: {
          type: 'boolean',
          description: 'Include documents ingested by other users / channels too. Default false.',
        },
      },
      required: ['query'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.documentStore) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Document memory store is not configured.',
          durationMs: Date.now() - startedAt,
        }
      }

      const query = typeof input.query === 'string' ? input.query.trim() : ''
      if (!query) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A query is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      const scopeBypassError = rejectScopedIncludeAll(input, context, startedAt)
      if (scopeBypassError) return scopeBypassError
      const includeAllScopes = input.includeAllScopes === true

      const options: DocumentSearchOptions = {}
      const limitInput = typeof input.limit === 'number' ? input.limit : Number(input.limit)
      const requestedLimit = Number.isFinite(limitInput) && limitInput > 0
        ? Math.min(Math.floor(limitInput), SEARCH_MAX_LIMIT)
        : SEARCH_DEFAULT_LIMIT
      options.limit = requestedLimit

      if (typeof input.documentId === 'string' && input.documentId.trim().length > 0) {
        options.documentId = input.documentId.trim()
      }
      const tags = Array.isArray(input.tags)
        ? input.tags.filter((tag): tag is string => typeof tag === 'string').map((t) => t.trim()).filter(Boolean)
        : []
      if (tags.length > 0) options.tags = tags

      const includeAppIndex = explicitlyRequestsAppIndex({
        tags,
        documentId: options.documentId,
      })
      const mayReturnAppIndex = !includeAppIndex
        && !options.documentId
        && canTagFilterReturnInternalAppIndex(tags)
      if (
        (context?.scopeTags && context.scopeTags.length > 0 && !includeAllScopes)
        || mayReturnAppIndex
      ) {
        options.limit = Math.min(requestedLimit * 4, SEARCH_MAX_LIMIT)
      }

      if (typeof input.type === 'string'
          && (input.type === 'semantic' || input.type === 'keyword' || input.type === 'hybrid')) {
        options.type = input.type
      }
      if (typeof input.minScore === 'number' && Number.isFinite(input.minScore)) {
        options.minScore = input.minScore
      }
      if (typeof input.sortBy === 'string'
          && (input.sortBy === 'score' || input.sortBy === 'createdAt' || input.sortBy === 'updatedAt')) {
        options.sortBy = input.sortBy
      }

      let chunks: MemoryDocumentChunk[]
      try {
        if (!includeAllScopes && context?.scopeTags) options.scopeTags = context.scopeTags
        chunks = await deps.documentStore.searchDocuments(query, options)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_SEARCH_FAILED_TRANSIENT',
          output: `Document search failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }

      let scopeFilteredOut = 0
      let visible = filterAppIndexEntries(chunks, includeAppIndex)
      if (!includeAllScopes && context?.scopeTags && context.scopeTags.length > 0) {
        visible = visible.filter((chunk) => {
          if (isMemoryVisibleInScope(chunk.tags, context.scopeTags)) return true
          scopeFilteredOut += 1
          return false
        })
      }
      visible = visible.slice(0, requestedLimit)

      return {
        status: 'success',
        output: JSON.stringify({
          query,
          totalHits: visible.length,
          scopeFilteredOut: context?.scopeTags?.length ? 0 : scopeFilteredOut,
          hits: visible.map((chunk) => ({
            id: chunk.id,
            documentId: chunk.documentId,
            documentTitle: chunk.documentTitle,
            documentPath: chunk.documentPath,
            citationLabel: chunk.citationLabel,
            chunkIndex: chunk.chunkIndex,
            chunkCount: chunk.chunkCount,
            chunkTitle: chunk.chunkTitle,
            snippet: truncateSnippet(chunk.snippet ?? chunk.content),
            tags: chunk.tags,
            score: chunk.score,
          })),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryDocumentsPreviewTool(
  deps: MemoryDocumentsToolDeps,
): ToolDefinitionRuntime {
  return {
    name: 'memory.documents.preview',
    description: 'Return the first N chunks of an ingested document in order so you can confirm the content without running a search. Use after memory.documents.list / memory.documents.get to verify which document the user means before memory.documents.delete or memory.documents.update.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory.documents' },
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Document id.' },
        limit: { type: 'number', description: 'Max chunks to preview (default 5, max 30).' },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.documentStore || typeof deps.documentStore.listDocumentChunks !== 'function') {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Document memory store is not configured.',
          durationMs: Date.now() - startedAt,
        }
      }
      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (!id) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A document id is required.',
          durationMs: Date.now() - startedAt,
        }
      }

      const doc = await deps.documentStore.getDocument(id)
      if (!doc) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_NOT_FOUND_USER',
          output: `Document ${id} was not found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (isInternalAppDocument(doc)) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_NOT_FOUND_USER',
          output: `Document ${id} was not found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      // Hide cross-scope documents from preview just like get/search.
      if (context?.scopeTags && context.scopeTags.length > 0
        && !isMemoryVisibleInScope(doc.tags, context.scopeTags)) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_NOT_FOUND_USER',
          output: `Document ${id} was not found.`,
          durationMs: Date.now() - startedAt,
        }
      }

      const limitInput = typeof input.limit === 'number' ? input.limit : 5
      const limit = Math.max(1, Math.min(Math.floor(limitInput), 30))
      const chunks = await deps.documentStore.listDocumentChunks(id, limit)
      return {
        status: 'success',
        output: JSON.stringify({
          documentId: doc.id,
          title: doc.title,
          totalChunkCount: doc.chunkCount,
          previewChunkCount: chunks.length,
          chunks: chunks.map((chunk) => ({
            id: chunk.id,
            chunkIndex: chunk.chunkIndex,
            chunkTitle: chunk.chunkTitle,
            citationLabel: chunk.citationLabel,
            snippet: truncateSnippet(chunk.snippet ?? chunk.content),
          })),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryDocumentsGetTool(
  deps: MemoryDocumentsToolDeps,
): ToolDefinitionRuntime {
  return {
    name: 'memory.documents.get',
    description: 'Fetch a single ingested document\'s metadata by id (title, path, mimeType, tags, chunkCount, timestamps). Use after memory.documents.list to confirm the right document, or before memory.documents.delete / memory.documents.update.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory.documents' },
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Document id.' },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.documentStore) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Document memory store is not configured.',
          durationMs: Date.now() - startedAt,
        }
      }
      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (!id) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A document id is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      const doc = await deps.documentStore.getDocument(id)
      if (!doc) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_NOT_FOUND_USER',
          output: `Document ${id} was not found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (isInternalAppDocument(doc)) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_NOT_FOUND_USER',
          output: `Document ${id} was not found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      // Hide cross-scope documents from view.
      if (context?.scopeTags && context.scopeTags.length > 0
        && !isMemoryVisibleInScope(doc.tags, context.scopeTags)) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_NOT_FOUND_USER',
          output: `Document ${id} was not found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      return {
        status: 'success',
        output: JSON.stringify({
          id: doc.id,
          title: doc.title,
          path: doc.path,
          mimeType: doc.mimeType,
          tags: doc.tags,
          chunkCount: doc.chunkCount,
          createdAt: doc.createdAt,
          updatedAt: doc.updatedAt,
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryDocumentsUpdateTool(
  deps: MemoryDocumentsToolDeps,
): ToolDefinitionRuntime {
  return {
    name: 'memory.documents.update',
    description: 'Edit document metadata in place (title, path, mimeType, tags) without re-ingesting the chunks/embeddings. Use to fix a typo in the title, attach more tags, or update the source path. Refuses to touch documents owned by another user/channel scope.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory.documents' },
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Document id.' },
        title: { type: 'string', description: 'New title (non-empty).' },
        path: { type: 'string', description: 'New path. Pass empty string to clear.' },
        mimeType: { type: 'string', description: 'New MIME type. Pass empty string to clear.' },
        tags: {
          type: 'array',
          items: { type: 'string' },
          description: 'Replace tags entirely. Mutually exclusive with addTags / removeTags.',
        },
        addTags: {
          type: 'array',
          items: { type: 'string' },
          description: 'Tags to add to the existing set.',
        },
        removeTags: {
          type: 'array',
          items: { type: 'string' },
          description: 'Tags to remove from the existing set.',
        },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.documentStore) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Document memory store is not configured.',
          durationMs: Date.now() - startedAt,
        }
      }
      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (!id) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A document id is required.',
          durationMs: Date.now() - startedAt,
        }
      }

      const existing = await deps.documentStore.getDocument(id)
      if (!existing) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_NOT_FOUND_USER',
          output: `Document ${id} was not found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (isInternalAppDocument(existing)) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_NOT_FOUND_USER',
          output: `Document ${id} was not found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (context?.scopeTags && context.scopeTags.length > 0
        && !isMemoryWritableInScope(existing.tags, context.scopeTags)) {
        return {
          status: 'error',
          code: 'MEMORY_SCOPE_MISMATCH_USER',
          output: 'This document belongs to a different user/channel scope and cannot be updated from here.',
          durationMs: Date.now() - startedAt,
        }
      }

      const explicitTags = Array.isArray(input.tags)
        ? input.tags.filter((tag): tag is string => typeof tag === 'string').map((tag) => tag.trim()).filter(Boolean)
        : null
      const addTags = Array.isArray(input.addTags)
        ? input.addTags.filter((tag): tag is string => typeof tag === 'string').map((tag) => tag.trim()).filter(Boolean)
        : []
      const removeTags = Array.isArray(input.removeTags)
        ? input.removeTags.filter((tag): tag is string => typeof tag === 'string').map((tag) => tag.trim()).filter(Boolean)
        : []
      if (explicitTags && (addTags.length > 0 || removeTags.length > 0)) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: '`tags` is mutually exclusive with `addTags` / `removeTags`.',
          durationMs: Date.now() - startedAt,
        }
      }

      // Preserve scope tags so the caller can't strip ownership.
      const existingScopeTags = (existing.tags ?? []).filter(isReservedTag)
      let nextTags: string[] | undefined
      if (explicitTags) {
        nextTags = uniqueOrdered([...explicitTags, ...existingScopeTags])
      } else if (addTags.length > 0 || removeTags.length > 0) {
        const removeSet = new Set(removeTags.map((tag) => tag.toLowerCase()))
        const kept = (existing.tags ?? []).filter((tag) => {
          if (isReservedTag(tag)) return true
          return !removeSet.has(tag.toLowerCase())
        })
        nextTags = uniqueOrdered([...kept, ...addTags])
      }

      const update: DocumentUpdateInput = {}
      if (typeof input.title === 'string' && input.title.trim().length > 0) {
        update.title = input.title.trim()
      }
      if (typeof input.path === 'string') {
        update.path = input.path
      }
      if (typeof input.mimeType === 'string') {
        update.mimeType = input.mimeType
      }
      if (nextTags) {
        update.tags = nextTags
      }

      if (Object.keys(update).length === 0) {
        return {
          status: 'success',
          output: JSON.stringify({ id, updated: false, reason: 'No changes detected.' }),
          durationMs: Date.now() - startedAt,
        }
      }

      let updated: MemoryDocument | null
      try {
        updated = await deps.documentStore.updateDocument(id, update)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_UPDATE_FAILED_TRANSIENT',
          output: `Document update failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (!updated) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_NOT_FOUND_USER',
          output: `Document ${id} was not found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      return {
        status: 'success',
        output: JSON.stringify({
          id: updated.id,
          updated: true,
          before: {
            title: existing.title,
            path: existing.path,
            mimeType: existing.mimeType,
            tags: existing.tags,
          },
          after: {
            title: updated.title,
            path: updated.path,
            mimeType: updated.mimeType,
            tags: updated.tags,
          },
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryDocumentsDeleteTool(
  deps: MemoryDocumentsToolDeps & {
    documentStore?: Pick<IDocumentMemoryStore, 'getDocument' | 'deleteDocument'>
  },
): ToolDefinitionRuntime {
  return {
    name: 'memory.documents.delete',
    description: 'Delete an ingested document (and all of its chunks) by id. Use after memory.documents.list. Refuses to delete documents owned by another user/channel scope.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory.documents' },
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Document id (from memory.documents.list).' },
        reason: { type: 'string', description: 'Optional note for the audit log.' },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.documentStore) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Document memory store is not configured.',
          durationMs: Date.now() - startedAt,
        }
      }
      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (!id) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A document id is required.',
          durationMs: Date.now() - startedAt,
        }
      }

      const doc = await deps.documentStore.getDocument(id)
      if (!doc) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_NOT_FOUND_USER',
          output: `Document ${id} was not found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (isInternalAppDocument(doc)) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_NOT_FOUND_USER',
          output: `Document ${id} was not found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (!isMemoryWritableInScope(doc.tags, context?.scopeTags)) {
        return {
          status: 'error',
          code: 'MEMORY_SCOPE_MISMATCH_USER',
          output: 'This document belongs to a different user/channel scope and cannot be deleted from here.',
          durationMs: Date.now() - startedAt,
        }
      }

      let removed = false
      try {
        removed = await deps.documentStore.deleteDocument(id)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_DELETE_FAILED_TRANSIENT',
          output: `Document delete failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (!removed) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_NOT_FOUND_USER',
          output: `Document ${id} was not found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      return {
        status: 'success',
        output: JSON.stringify({
          id,
          title: doc.title,
          chunkCount: doc.chunkCount,
          deleted: true,
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryDocumentsListTool(
  deps: MemoryDocumentsToolDeps,
): ToolDefinitionRuntime {
  return {
    name: 'memory.documents.list',
    description: 'List ingested documents (id, title, chunk count, tags). Use this before memory.documents.search when you need to discover what documents exist or find a specific document by title.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory.documents' },
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Optional title fragment.' },
        limit: { type: 'number', description: `Default 50, max ${SEARCH_MAX_LIMIT * 4}.` },
        sortBy: {
          type: 'string',
          enum: ['createdAt', 'updatedAt', 'title', 'chunkCount'],
          description: 'Sort key. Default `updatedAt` (newest first); `title` is alphabetical, the rest sort descending.',
        },
        includeAllScopes: {
          type: 'boolean',
          description: 'Admin-only: include documents owned by other scopes. Default false.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.documentStore) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Document memory store is not configured.',
          durationMs: Date.now() - startedAt,
        }
      }
      const scopeBypassError = rejectScopedIncludeAll(input, context, startedAt)
      if (scopeBypassError) return scopeBypassError
      const includeAllScopes = input.includeAllScopes === true
      const scopeTags = context?.scopeTags ?? []

      const options: DocumentListOptions = {}
      if (typeof input.query === 'string' && input.query.trim().length > 0) {
        options.query = input.query.trim()
      }
      const limit = typeof input.limit === 'number' ? input.limit : Number(input.limit)
      const requestedLimit = Number.isFinite(limit) && limit > 0
        ? Math.min(Math.floor(limit), SEARCH_MAX_LIMIT * 4)
        : 50
      options.limit = Math.min(requestedLimit * 4, SEARCH_MAX_LIMIT * 4)

      let documents: MemoryDocument[]
      try {
        documents = await deps.documentStore.listDocuments(options)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_DOCUMENT_LIST_FAILED_TRANSIENT',
          output: `Document list failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
      documents = documents.filter((doc) => !isInternalAppDocument(doc))
      let scopeFilteredOut = 0
      if (!includeAllScopes && scopeTags.length > 0) {
        documents = documents.filter((doc) => {
          if (isMemoryVisibleInScope(doc.tags, scopeTags)) return true
          scopeFilteredOut += 1
          return false
        })
      }

      const sortBy = typeof input.sortBy === 'string' ? input.sortBy : 'updatedAt'
      const sorted = [...documents].sort((a, b) => {
        switch (sortBy) {
          case 'createdAt':
            return Date.parse(b.createdAt ?? '') - Date.parse(a.createdAt ?? '')
          case 'title':
            return (a.title ?? '').localeCompare(b.title ?? '')
          case 'chunkCount':
            return (b.chunkCount ?? 0) - (a.chunkCount ?? 0)
          default:
            return Date.parse(b.updatedAt ?? '') - Date.parse(a.updatedAt ?? '')
        }
      })

      return {
        status: 'success',
        output: JSON.stringify({
          totalDocuments: Math.min(sorted.length, requestedLimit),
          scopeFilteredOut: context?.scopeTags?.length ? 0 : scopeFilteredOut,
          documents: sorted.slice(0, requestedLimit).map((doc) => ({
            id: doc.id,
            title: doc.title,
            path: doc.path,
            mimeType: doc.mimeType,
            tags: doc.tags,
            chunkCount: doc.chunkCount,
            createdAt: doc.createdAt,
            updatedAt: doc.updatedAt,
          })),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryAuditTool(deps: MemoryAuditToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.audit',
    description: 'Inspect the audit trail for durable memory entries (created/updated/deleted/pruned). Use to investigate why a memory looks different than expected, to verify your last edit landed, or when the user asks "when did this memory change". Optionally filter by memoryId; default returns the most-recent 50 entries.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        memoryId: { type: 'string', description: 'Restrict to a single memory id.' },
        limit: { type: 'number', description: 'Max entries (default 50, max 500).' },
        includeAllScopes: {
          type: 'boolean',
          description: 'Include audit entries for memories owned by other scopes. Default false.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }

      const options: MemoryAuditListOptions = {}
      if (typeof input.memoryId === 'string' && input.memoryId.trim().length > 0) {
        options.memoryId = input.memoryId.trim()
      }
      const limit = typeof input.limit === 'number' ? input.limit : Number(input.limit)
      options.limit = Number.isFinite(limit) && limit > 0
        ? Math.min(Math.floor(limit), 500)
        : 50
      const scopeBypassError = rejectScopedIncludeAll(input, context, startedAt)
      if (scopeBypassError) return scopeBypassError
      const includeAllScopes = input.includeAllScopes === true

      let entries: MemoryAuditEntry[]
      try {
        entries = await deps.semanticIndex.listAudit(options)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_AUDIT_FAILED_TRANSIENT',
          output: `Audit query failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }

      let scopeFilteredOut = 0
      let visible = entries
      if (!includeAllScopes && context?.scopeTags && context.scopeTags.length > 0) {
        visible = entries.filter((entry) => {
          const ownerTags = entry.after?.tags ?? entry.before?.tags
          if (isMemoryVisibleInScope(ownerTags, context.scopeTags)) return true
          scopeFilteredOut += 1
          return false
        })
      }

      const summary = visible.reduce<Record<string, number>>((acc, entry) => {
        acc[entry.action] = (acc[entry.action] ?? 0) + 1
        return acc
      }, {})

      return {
        status: 'success',
        output: JSON.stringify({
          totalEntries: visible.length,
          scopeFilteredOut: context?.scopeTags?.length ? 0 : scopeFilteredOut,
          summary,
          entries: visible.map((entry) => ({
            id: entry.id,
            memoryId: entry.memoryId,
            action: entry.action,
            actor: entry.actor,
            reason: entry.reason,
            createdAt: entry.createdAt,
            beforeSnippet: entry.before?.content ? truncateSnippet(entry.before.content) : undefined,
            afterSnippet: entry.after?.content ? truncateSnippet(entry.after.content) : undefined,
          })),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryMaintenanceTool(deps: MemoryMaintenanceToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.maintenance',
    description: 'Inspect and (with explicit confirmation) prune the durable-memory lifecycle: stale conversation memories beyond a configurable age, low-importance entries, and re-scored importance values. ALWAYS run with confirm=false (default) first to preview wouldPrune counts; only run with confirm=true after the user explicitly approves the numbers (e.g. "정리해", "prune them"). Maintenance is daemon-wide and is NOT scoped per user — refuse to run a non-dry maintenance unless the requesting user is the daemon operator.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        staleAfterDays: {
          type: 'number',
          description: 'Conversation memories older than this many days become prune candidates. Default 90.',
        },
        lowImportance: {
          type: 'number',
          description: 'Conversation memories with importance below this score become prune candidates. Default 0.1, range 0-1.',
        },
        confirm: {
          type: 'boolean',
          description: 'Set to true to actually run the prune (after a dryRun preview). Default false (dryRun only).',
        },
        reason: {
          type: 'string',
          description: 'Human-readable note recorded in the audit log when confirm=true.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }

      const confirm = input.confirm === true
      // Scope guard (symmetric with rejectScopedIncludeAll): maintenance prune is
      // daemon-wide and NOT scoped per user, so a destructive run (confirm:true)
      // from a scoped, non-operator context would prune *every* user's memories.
      // Enforce operator-only in code — the prompt-only guard in the description
      // was not binding. Dry-run previews (confirm:false) stay allowed.
      if (confirm && context?.scopeTags && context.scopeTags.length > 0) {
        return {
          status: 'error',
          code: 'MEMORY_SCOPE_BYPASS_DENIED_USER',
          output: 'memory.maintenance prune (confirm:true) is daemon-wide and only allowed from an unscoped/operator execution context; a scoped agent cannot prune other users\' memories. Run with confirm:false for a scoped-safe preview.',
          durationMs: Date.now() - startedAt,
        }
      }

      const options: MemoryMaintenanceOptions = {}
      if (typeof input.staleAfterDays === 'number' && input.staleAfterDays > 0) {
        // Floor at 1 day: a fractional value (e.g. 0.001) would make almost
        // every conversation memory a stale prune candidate — an accidental or
        // hostile full wipe. Never prune below a one-day age.
        options.staleAfterDays = Math.max(1, input.staleAfterDays)
      }
      if (typeof input.lowImportance === 'number'
          && input.lowImportance >= 0 && input.lowImportance <= 1) {
        options.lowImportance = input.lowImportance
      }
      options.dryRun = !confirm
      if (confirm) {
        options.actor = 'agent'
        if (typeof input.reason === 'string' && input.reason.trim().length > 0) {
          options.reason = input.reason.trim()
        }
      }

      let result: MemoryMaintenanceResult
      try {
        result = await deps.semanticIndex.runMaintenance(options)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_MAINTENANCE_FAILED_TRANSIENT',
          output: `Maintenance failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }

      return {
        status: 'success',
        output: JSON.stringify({
          dryRun: result.dryRun,
          importanceUpdated: result.importanceUpdated,
          pruned: result.pruned,
          wouldPrune: result.wouldPrune,
          appliedOptions: {
            staleAfterDays: options.staleAfterDays ?? 90,
            lowImportance: options.lowImportance ?? 0.1,
          },
          status: {
            totalMemories: result.status.totalMemories,
            conversationMemories: result.status.conversationMemories,
            documentMemories: result.status.documentMemories,
            userMemories: result.status.userMemories,
            staleConversationMemories: result.status.staleConversationMemories,
            lowImportanceConversationMemories: result.status.lowImportanceConversationMemories,
            pruneCandidateMemories: result.status.pruneCandidateMemories,
            pendingEmbeddings: result.status.pendingEmbeddings,
            failedEmbeddings: result.status.failedEmbeddings,
            lastAuditAt: result.status.lastAuditAt,
          },
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryContextSnapshotTool(
  deps: MemoryContextSnapshotToolDeps,
): ToolDefinitionRuntime {
  return {
    name: 'memory.context.snapshot',
    description: 'Return a snapshot of what the agent currently has in its working memory context: long-term file memory sections, today/yesterday daily notes, top semantic search hits for an optional query, and upcoming reminders. Use to answer "what do you remember about me right now" or for transparency when the user wonders what context the agent is operating on. Read-only.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Optional natural-language query for the semantic-retrieval portion of the snapshot. Omit to skip retrieval and only return file/reminders state.',
        },
        retrieveLimit: {
          type: 'number',
          description: 'How many semantic hits to include when query is provided. Default 5, max 20.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const scopeTags = context?.scopeTags ?? []
      const fileMemory = resolveFileMemory(deps, scopeTags)

      const snapshot: Record<string, unknown> = {
        scopeTags,
      }

      if (fileMemory) {
        try {
          const ctx = await fileMemory.getPromptContext()
          snapshot.fileMemory = {
            longTermMemory: ctx.longTermMemory ?? '',
            todayNote: ctx.todayNote ?? '',
            yesterdayNote: ctx.yesterdayNote ?? '',
            todayNotePath: fileMemory.getDailyNotePath(new Date()),
            sections: (await fileMemory.readMemorySections()).map((section) => section.title),
          }
        } catch (error) {
          snapshot.fileMemory = { error: error instanceof Error ? error.message : String(error) }
        }
      }

      if (deps.semanticIndex && typeof input.query === 'string' && input.query.trim().length > 0) {
        const limitInput = typeof input.retrieveLimit === 'number' ? input.retrieveLimit : 5
        const limit = Math.max(1, Math.min(Math.floor(limitInput), 20))
        try {
          const fetchLimit = scopeTags.length > 0 ? Math.min(limit * 4, 80) : limit
          const raw = await deps.semanticIndex.search(input.query.trim(), {
            scopeTags: scopeTags.length ? scopeTags : undefined,
            limit: fetchLimit,
            type: 'hybrid',
          })
          const visible = raw.filter((entry) =>
            !isMemoryArchived(entry.tags)
            && !isMemorySuperseded(entry.tags)
            && (scopeTags.length === 0 || isMemoryVisibleInScope(entry.tags, scopeTags)),
          ).slice(0, limit)
          snapshot.relevantHits = visible.map((entry) => ({
            id: entry.id,
            snippet: truncateSnippet(entry.content),
            source: entry.source,
            tags: entry.tags,
            score: entry.score,
          }))
        } catch (error) {
          snapshot.relevantHits = { error: error instanceof Error ? error.message : String(error) }
        }
      }

      // Graph stats/search currently describe the daemon-wide graph. Do not
      // expose them through a scoped snapshot until the graph has a scoped
      // projection equivalent to semantic memory filtering.
      if (scopeTags.length === 0 && deps.semanticIndex?.getGraphStats) {
        try {
          const graphSummary: Record<string, unknown> = {
            stats: await deps.semanticIndex.getGraphStats(),
          }
          if (
            deps.semanticIndex.searchGraph
            && typeof input.query === 'string'
            && input.query.trim().length > 0
          ) {
            const limitInput = typeof input.retrieveLimit === 'number' ? input.retrieveLimit : 5
            const limit = Math.max(1, Math.min(Math.floor(limitInput), 20))
            const graph = await deps.semanticIndex.searchGraph(input.query.trim(), { limit })
            graphSummary.hits = {
              nodes: graph.nodes.map(serializeMemoryGraphNode),
              edges: graph.edges.map(serializeMemoryGraphEdge),
            }
          }
          snapshot.memoryGraph = graphSummary
        } catch (error) {
          snapshot.memoryGraph = { error: error instanceof Error ? error.message : String(error) }
        }
      }

      if (deps.reminders) {
        try {
          const list = await deps.reminders.listForScope(scopeTags)
          const horizonMs = Date.now() + 7 * 24 * 60 * 60 * 1000
          const upcoming = list
            .filter((reminder) => !reminder.firedAt && !reminder.cancelledAt)
            .filter((reminder) => Date.parse(reminder.dueAt) <= horizonMs)
            .sort((a, b) => Date.parse(a.dueAt) - Date.parse(b.dueAt))
            .slice(0, 10)
          snapshot.upcomingReminders = upcoming.map((reminder) => ({
            id: reminder.id,
            dueAt: reminder.dueAt,
            content: reminder.content,
          }))
        } catch (error) {
          snapshot.upcomingReminders = { error: error instanceof Error ? error.message : String(error) }
        }
      }

      return {
        status: 'success',
        output: JSON.stringify(snapshot),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryUsageTool(deps: MemoryUsageToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.usage',
    description: 'Summarise the active scope\'s memory state: counts by source, pending reminders, recent audit activity by action. Read-only. Use to answer "내 메모리 상태 보여줘" / "what do I have stored" without dumping full content.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory' },
    inputSchema: { type: 'object', properties: {} },
    async execute(_input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const scopeTags = context?.scopeTags ?? []
      const summary: Record<string, unknown> = {
        scopeTags,
      }

      if (deps.semanticIndex) {
        try {
          const all = await deps.semanticIndex.listRecent(5000)
          const visible = scopeTags.length > 0
            ? all.filter((entry) => isMemoryVisibleInScope(entry.tags, scopeTags))
            : all
          const bySource: Record<string, number> = {}
          let archivedCount = 0
          let supersededCount = 0
          for (const entry of visible) {
            bySource[entry.source] = (bySource[entry.source] ?? 0) + 1
            if (isMemoryArchived(entry.tags)) archivedCount += 1
            if (isMemorySuperseded(entry.tags)) supersededCount += 1
          }
          summary.semantic = {
            totalVisible: visible.length,
            bySource,
            archived: archivedCount,
            superseded: supersededCount,
          }
        } catch (error) {
          summary.semantic = { error: error instanceof Error ? error.message : String(error) }
        }

        if (typeof deps.semanticIndex.listAudit === 'function') {
          try {
            const audit = await deps.semanticIndex.listAudit({ limit: 200 })
            const visibleAudit = scopeTags.length > 0
              ? audit.filter((entry) => {
                  const ownerTags = entry.after?.tags ?? entry.before?.tags
                  return isMemoryVisibleInScope(ownerTags, scopeTags)
                })
              : audit
            const byAction = visibleAudit.reduce<Record<string, number>>((acc, entry) => {
              acc[entry.action] = (acc[entry.action] ?? 0) + 1
              return acc
            }, {})
            summary.audit = {
              recentEntries: visibleAudit.length,
              byAction,
            }
          } catch (error) {
            summary.audit = { error: error instanceof Error ? error.message : String(error) }
          }
        }
      }

      if (deps.reminders) {
        try {
          const list = await deps.reminders.listForScope(scopeTags)
          summary.reminders = {
            pending: list.filter((reminder) => !reminder.firedAt && !reminder.cancelledAt).length,
          }
        } catch (error) {
          summary.reminders = { error: error instanceof Error ? error.message : String(error) }
        }
      }

      const usageFileMemory = resolveFileMemory(deps, scopeTags)
      if (usageFileMemory) {
        try {
          const dates = typeof usageFileMemory.listDailyNoteDates === 'function'
            ? await usageFileMemory.listDailyNoteDates(90)
            : []
          const sections = typeof usageFileMemory.readMemorySections === 'function'
            ? await usageFileMemory.readMemorySections()
            : []
          summary.fileMemory = {
            longTermSections: sections.length,
            dailyNotesLast90Days: dates.length,
            mostRecentDailyNote: dates[0],
          }
        } catch (error) {
          summary.fileMemory = { error: error instanceof Error ? error.message : String(error) }
        }
      }

      return {
        status: 'success',
        output: JSON.stringify(summary),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

const RESERVED_TAG_PREFIXES = ['scope:', 'archived', 'superseded', 'compressed-from:', 'archived-by:', 'superseded-by:']

function isReservedTag(tag: string): boolean {
  const lower = tag.toLowerCase()
  return RESERVED_TAG_PREFIXES.some((prefix) => lower === prefix.replace(/:$/, '') || lower.startsWith(prefix))
}

/**
 * Reserved/system tags a user-facing write (memory.remember / memory.update)
 * must never set directly: `scope:*` ownership tags (foreign-scope injection)
 * and lifecycle tags like `archived` / `superseded` / `archived-by:*`
 * (arbitrary hide/supersede). Scope tags are attached by the runtime from the
 * execution context; lifecycle tags are managed by dedicated operations.
 */
function findForbiddenUserTags(tags: readonly string[]): string[] {
  return tags.filter((tag) => isReservedTag(tag))
}

export function createMemoryTagRenameTool(deps: MemoryTagRenameToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.tag.rename',
    description: 'Bulk-rename a tag across the active scope\'s memories (e.g. "preference" → "user-pref"). Always run with dryRun:true (default) first to preview the affected entries; only confirm:true after the user explicitly approves the count. Reserved tags (scope:*, archived, superseded, compressed-from:*) cannot be renamed.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        from: { type: 'string', description: 'Tag to rename (case-insensitive match).' },
        to: { type: 'string', description: 'New tag value (will be lowercased and trimmed).' },
        confirm: {
          type: 'boolean',
          description: 'Set to true to actually apply. Default false (dryRun preview).',
        },
        reason: {
          type: 'string',
          description: 'Optional human-readable note recorded in the audit log when applied.',
        },
      },
      required: ['from', 'to'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }
      const from = typeof input.from === 'string' ? input.from.trim() : ''
      const to = typeof input.to === 'string' ? input.to.trim() : ''
      if (!from || !to) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'Both `from` and `to` are required.',
          durationMs: Date.now() - startedAt,
        }
      }
      if (from.toLowerCase() === to.toLowerCase()) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: '`from` and `to` are equivalent; nothing to rename.',
          durationMs: Date.now() - startedAt,
        }
      }
      if (isReservedTag(from) || isReservedTag(to)) {
        return {
          status: 'error',
          code: 'MEMORY_RESERVED_TAG_USER',
          output: 'Reserved tags (scope:*, archived, superseded, compressed-from:*, …) cannot be renamed.',
          durationMs: Date.now() - startedAt,
        }
      }

      const confirm = input.confirm === true
      const dryRun = !confirm
      const fromLower = from.toLowerCase()
      const toCanonical = to.toLowerCase()
      const scopeTags = context?.scopeTags ?? []

      let all: Array<Omit<MemoryEntry, 'score'>>
      try {
        all = await deps.semanticIndex.listRecent(5000)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_TAG_RENAME_FAILED_TRANSIENT',
          output: `Failed to list semantic memory: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }

      const candidates = all.filter((entry) => {
        if (scopeTags.length > 0 && !isMemoryWritableInScope(entry.tags, scopeTags)) return false
        return entry.tags.some((tag) => tag.toLowerCase() === fromLower)
      })

      const affectedIds = candidates.map((entry) => entry.id)

      if (dryRun) {
        return {
          status: 'success',
          output: JSON.stringify({
            dryRun: true,
            wouldRename: affectedIds.length,
            affectedIds: affectedIds.slice(0, 50),
            from,
            to,
          }),
          durationMs: Date.now() - startedAt,
        }
      }

      let renamed = 0
      const warnings: string[] = []
      for (const entry of candidates) {
        const nextTags = uniqueOrdered([
          ...entry.tags
            .filter((tag) => tag.toLowerCase() !== fromLower)
            .filter((tag) => tag.toLowerCase() !== toCanonical),
          to,
        ])
        try {
          await deps.semanticIndex.add({
            id: entry.id,
            content: entry.content,
            source: entry.source,
            tags: nextTags,
          })
          renamed += 1
        } catch (error) {
          warnings.push(`${entry.id}: ${error instanceof Error ? error.message : String(error)}`)
        }
      }

      return {
        status: 'success',
        output: JSON.stringify({
          dryRun: false,
          renamed,
          from,
          to,
          reason: typeof input.reason === 'string' && input.reason.trim().length > 0
            ? input.reason.trim()
            : undefined,
          warnings,
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryTagListTool(deps: MemoryTagListToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.tag.list',
    description: 'List all tags used by memories in the active scope, with their counts. Use to answer "what tags do I have" or to find tag-rename candidates. System/scope tags (scope:*, archived, superseded, compressed-from:*, archived-by:*, superseded-by:*) are hidden by default.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        limit: { type: 'number', description: 'Max distinct tags to return (default 50, max 500).' },
        includeReserved: {
          type: 'boolean',
          description: 'Include scope:*, archived, superseded, compressed-from:* internal markers. Default false.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }
      const limitInput = typeof input.limit === 'number' ? input.limit : 50
      const limit = Math.max(1, Math.min(Math.floor(limitInput), 500))
      const includeReserved = input.includeReserved === true
      const scopeTags = context?.scopeTags ?? []

      let all: Array<Omit<MemoryEntry, 'score'>>
      try {
        all = await deps.semanticIndex.listRecent(5000)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_TAG_LIST_FAILED_TRANSIENT',
          output: `Failed to list semantic memory: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
      const visible = scopeTags.length > 0
        ? all.filter((entry) => isMemoryVisibleInScope(entry.tags, scopeTags))
        : all
      const counts = new Map<string, number>()
      for (const entry of visible) {
        for (const tag of entry.tags ?? []) {
          if (!includeReserved && isReservedTag(tag)) continue
          if (!includeReserved && tag === 'explicit-memory') continue
          const normalized = tag.trim()
          if (!normalized) continue
          counts.set(normalized, (counts.get(normalized) ?? 0) + 1)
        }
      }
      const sorted = Array.from(counts.entries())
        .map(([tag, count]) => ({ tag, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, limit)
      return {
        status: 'success',
        output: JSON.stringify({
          totalDistinctTags: sorted.length,
          tags: sorted,
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemorySearchByTagTool(
  deps: MemorySearchByTagToolDeps,
): ToolDefinitionRuntime {
  return {
    name: 'memory.search.by_tag',
    description: 'Return memories that carry every supplied tag (case-insensitive AND match). Use when the user asks "show me everything tagged X" without a natural-language query. Read-only, scope-aware. archived/superseded entries are hidden by default.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        tags: {
          type: 'array',
          items: { type: 'string' },
          description: 'One or more tags. All must match (AND).',
        },
        limit: { type: 'number', description: 'Max hits (default 20, max 100).' },
        includeAllScopes: { type: 'boolean', description: 'Admin opt-in to bypass scope filter. Default false.' },
      },
      required: ['tags'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }
      const rawTags = Array.isArray(input.tags) ? input.tags : []
      const tags = rawTags
        .filter((tag): tag is string => typeof tag === 'string')
        .map((tag) => tag.trim().toLowerCase())
        .filter(Boolean)
      if (tags.length === 0) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'At least one tag is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      const limitInput = typeof input.limit === 'number' ? input.limit : 20
      const limit = Math.max(1, Math.min(Math.floor(limitInput), 100))
      const scopeTags = context?.scopeTags ?? []
      const scopeBypassError = rejectScopedIncludeAll(input, context, startedAt)
      if (scopeBypassError) return scopeBypassError
      const includeAllScopes = input.includeAllScopes === true

      let all: Array<Omit<MemoryEntry, 'score'>>
      try {
        all = await deps.semanticIndex.listRecent(5000)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_SEARCH_FAILED_TRANSIENT',
          output: `Failed to scan memory: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }

      const matched = all.filter((entry) => {
        if (isMemoryArchived(entry.tags) || isMemorySuperseded(entry.tags)) return false
        if (scopeTags.length > 0 && !includeAllScopes
          && !isMemoryVisibleInScope(entry.tags, scopeTags)) return false
        const lower = (entry.tags ?? []).map((tag) => tag.toLowerCase())
        return tags.every((tag) => lower.includes(tag))
      }).slice(0, limit)

      if (matched.length > 0 && deps.semanticIndex.recordAccess) {
        void deps.semanticIndex.recordAccess(matched.map((entry) => entry.id)).catch(() => {})
      }

      return {
        status: 'success',
        output: JSON.stringify({
          query: { tags: rawTags },
          totalHits: matched.length,
          hits: matched.map((entry) => ({
            id: entry.id,
            snippet: truncateSnippet(entry.content),
            source: entry.source,
            tags: entry.tags,
          })),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemorySearchRelatedTool(
  deps: MemoryRelatedToolDeps,
): ToolDefinitionRuntime {
  return {
    name: 'memory.search.related',
    description: 'Find memories similar to a given memory id (uses the source memory\'s content as the query). Use to suggest "you might also remember…" or to discover memories that should be merged. Read-only; result excludes the source memory itself, archived/superseded entries, and entries outside the caller scope.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory', key: (input) => String(input.id ?? '') },
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Source memory id.' },
        limit: { type: 'number', description: 'Max related hits (default 5, max 20).' },
        type: { type: 'string', enum: ['semantic', 'keyword', 'hybrid'] },
        minScore: { type: 'number', description: 'Drop hits below this score (0-1).' },
        sortBy: {
          type: 'string',
          enum: ['score', 'createdAt', 'updatedAt'],
          description: 'Result ordering. Default `score`.',
        },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }
      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (!id) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A source memory id is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      const source = await deps.semanticIndex.get(id)
      if (!source) {
        return {
          status: 'error',
          code: 'MEMORY_ID_NOT_FOUND_USER',
          output: `No memory with id "${id}" was found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      const scopeTags = context?.scopeTags ?? []
      if (scopeTags.length > 0 && !isMemoryVisibleInScope(source.tags, scopeTags)) {
        return {
          status: 'error',
          code: 'MEMORY_ID_NOT_FOUND_USER',
          output: `No memory with id "${id}" was found.`,
          durationMs: Date.now() - startedAt,
        }
      }

      const limitInput = typeof input.limit === 'number' ? input.limit : 5
      const limit = Math.max(1, Math.min(Math.floor(limitInput), 20))
      const fetchLimit = Math.min(limit * 4 + 1, 80)

      const searchOptions: SemanticSearchOptions = {
        limit: fetchLimit,
        type: (typeof input.type === 'string'
          && (input.type === 'semantic' || input.type === 'keyword' || input.type === 'hybrid'))
          ? input.type
          : 'hybrid',
      }
      if (typeof input.minScore === 'number' && Number.isFinite(input.minScore)) {
        searchOptions.minScore = input.minScore
      }
      if (typeof input.sortBy === 'string'
          && (input.sortBy === 'score' || input.sortBy === 'createdAt' || input.sortBy === 'updatedAt')) {
        searchOptions.sortBy = input.sortBy
      }

      let raw: MemoryEntry[]
      try {
        if (scopeTags.length) searchOptions.scopeTags = scopeTags
        raw = await deps.semanticIndex.search(source.content, searchOptions)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_SEARCH_FAILED_TRANSIENT',
          output: `Related search failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }

      const visible = raw.filter((entry) => entry.id !== id)
        .filter((entry) => !isMemoryArchived(entry.tags))
        .filter((entry) => !isMemorySuperseded(entry.tags))
        .filter((entry) => scopeTags.length === 0 || isMemoryVisibleInScope(entry.tags, scopeTags))
        .slice(0, limit)

      if (visible.length > 0 && deps.semanticIndex.recordAccess) {
        void deps.semanticIndex.recordAccess(visible.map((entry) => entry.id)).catch(() => {})
      }

      return {
        status: 'success',
        output: JSON.stringify({
          sourceId: id,
          totalHits: visible.length,
          hits: visible.map((entry) => ({
            id: entry.id,
            snippet: truncateSnippet(entry.content),
            source: entry.source,
            tags: entry.tags,
            score: entry.score,
          })),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryHistoryTool(deps: MemoryHistoryToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.history',
    description: 'Return the audit history of a single memory id alongside its current state in one call. Use to investigate "show me everything that happened to this memory". Read-only.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Memory id.' },
        limit: { type: 'number', description: 'Max audit entries (default 50, max 500).' },
        since: { type: 'string', description: 'Only return audit entries with createdAt at or after this ISO-8601 timestamp.' },
        until: { type: 'string', description: 'Only return audit entries with createdAt at or before this ISO-8601 timestamp.' },
        actions: {
          type: 'array',
          items: { type: 'string', enum: ['created', 'updated', 'deleted', 'pruned', 'maintenance'] },
          description: 'Restrict to specific audit actions.',
        },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }
      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (!id) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A memory id is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      // Validate optional time-range filters before doing any work.
      let sinceMs: number | undefined
      let untilMs: number | undefined
      if (typeof input.since === 'string' && input.since.trim().length > 0) {
        const parsed = Date.parse(input.since)
        if (Number.isNaN(parsed)) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: `since "${input.since}" is not a valid timestamp.`,
            durationMs: Date.now() - startedAt,
          }
        }
        sinceMs = parsed
      }
      if (typeof input.until === 'string' && input.until.trim().length > 0) {
        const parsed = Date.parse(input.until)
        if (Number.isNaN(parsed)) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: `until "${input.until}" is not a valid timestamp.`,
            durationMs: Date.now() - startedAt,
          }
        }
        untilMs = parsed
      }
      const actionsFilter = Array.isArray(input.actions)
        ? new Set(input.actions.filter((a): a is string => typeof a === 'string'))
        : null

      const current = await deps.semanticIndex.get(id)
      const scopeTags = context?.scopeTags ?? []
      // If the memory exists and is cross-scope, refuse without leaking content.
      if (current && scopeTags.length > 0 && !isMemoryVisibleInScope(current.tags, scopeTags)) {
        return {
          status: 'error',
          code: 'MEMORY_ID_NOT_FOUND_USER',
          output: `No memory with id "${id}" was found.`,
          durationMs: Date.now() - startedAt,
        }
      }

      const limitInput = typeof input.limit === 'number' ? input.limit : 50
      const limit = Math.max(1, Math.min(Math.floor(limitInput), 500))
      // Over-fetch when post-filtering will trim the result so we don't
      // return a short page just because most matches landed outside the
      // time/action window.
      const willPostFilter = sinceMs !== undefined || untilMs !== undefined || actionsFilter !== null
      const fetchLimit = willPostFilter ? Math.min(limit * 4, 2000) : limit

      let entries: MemoryAuditEntry[] = []
      try {
        entries = await deps.semanticIndex.listAudit({ memoryId: id, limit: fetchLimit })
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_AUDIT_FAILED_TRANSIENT',
          output: `Audit query failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (sinceMs !== undefined) {
        entries = entries.filter((entry) => Date.parse(entry.createdAt) >= sinceMs!)
      }
      if (untilMs !== undefined) {
        entries = entries.filter((entry) => Date.parse(entry.createdAt) <= untilMs!)
      }
      if (actionsFilter) {
        entries = entries.filter((entry) => actionsFilter.has(entry.action))
      }
      entries = entries.slice(0, limit)

      // For deleted memories, we still surface scope-mismatched ones as
      // not-found so the audit doesn't leak content owned by another user.
      if (!current && scopeTags.length > 0 && entries.length > 0) {
        const visible = entries.filter((entry) => {
          const ownerTags = entry.after?.tags ?? entry.before?.tags
          return isMemoryVisibleInScope(ownerTags, scopeTags)
        })
        if (visible.length === 0) {
          return {
            status: 'error',
            code: 'MEMORY_ID_NOT_FOUND_USER',
            output: `No memory with id "${id}" was found.`,
            durationMs: Date.now() - startedAt,
          }
        }
        entries = visible
      }

      return {
        status: 'success',
        output: JSON.stringify({
          memoryId: id,
          present: Boolean(current),
          currentSnippet: current ? truncateSnippet(current.content) : undefined,
          currentTags: current?.tags,
          totalEntries: entries.length,
          entries: entries.map((entry) => ({
            id: entry.id,
            action: entry.action,
            actor: entry.actor,
            reason: entry.reason,
            createdAt: entry.createdAt,
            beforeSnippet: entry.before?.content ? truncateSnippet(entry.before.content) : undefined,
            afterSnippet: entry.after?.content ? truncateSnippet(entry.after.content) : undefined,
          })),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryConflictsFindTool(
  deps: MemoryConflictsToolDeps,
): ToolDefinitionRuntime {
  return {
    name: 'memory.conflicts.find',
    description: 'Scan recent memories for contradiction pairs and return them WITHOUT mutating anything (dreaming consolidation auto-supersedes; this tool just surfaces). Use when the user asks "내 메모리에 모순 있어?" / "are any of my memories contradicting each other?". Read-only; calls the LLM, so use sparingly.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        maxPairs: {
          type: 'number',
          description: 'Maximum contradiction verdicts to return (default 10, max 25).',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (context?.scopeTags && context.scopeTags.length > 0) {
        return {
          status: 'error',
          code: 'MEMORY_SCOPE_BYPASS_DENIED_USER',
          output: 'Conflict scans require an unscoped/admin execution context until contradiction detection is scope-aware.',
          durationMs: Date.now() - startedAt,
        }
      }
      if (!deps.findRecentContradictions) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Contradiction detector is not configured (no LLM provider for dreaming).',
          durationMs: Date.now() - startedAt,
        }
      }
      const maxPairs = typeof input.maxPairs === 'number' && Number.isFinite(input.maxPairs)
        ? input.maxPairs
        : undefined
      let pairs: Array<{
        winnerId: string
        supersededIds: string[]
        primaryId: string
        candidateIds: string[]
      }>
      try {
        pairs = await deps.findRecentContradictions({ maxPairs })
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_CONFLICTS_FAILED_TRANSIENT',
          output: `Conflict scan failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
      return {
        status: 'success',
        output: JSON.stringify({
          totalPairs: pairs.length,
          pairs: pairs.map((entry) => ({
            winnerId: entry.winnerId,
            supersededIds: entry.supersededIds,
            primaryId: entry.primaryId,
            candidateIds: entry.candidateIds,
          })),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryDiffTool(deps: MemoryDiffToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.diff',
    description: 'Compare two memory entries by id and return the differences (content line-by-line, tags added/removed, source change). Use to investigate "why does this look different than I remember" or to verify a memory.update landed correctly. Read-only.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        leftId: { type: 'string', description: 'First memory id (the "old" or "expected" side).' },
        rightId: { type: 'string', description: 'Second memory id (the "new" or "actual" side).' },
      },
      required: ['leftId', 'rightId'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }
      const leftId = typeof input.leftId === 'string' ? input.leftId.trim() : ''
      const rightId = typeof input.rightId === 'string' ? input.rightId.trim() : ''
      if (!leftId || !rightId) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'Both leftId and rightId are required.',
          durationMs: Date.now() - startedAt,
        }
      }
      if (leftId === rightId) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'leftId and rightId must differ.',
          durationMs: Date.now() - startedAt,
        }
      }
      const [left, right] = await Promise.all([
        deps.semanticIndex.get(leftId),
        deps.semanticIndex.get(rightId),
      ])
      if (!left) {
        return {
          status: 'error',
          code: 'MEMORY_ID_NOT_FOUND_USER',
          output: `No memory with id "${leftId}" was found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (!right) {
        return {
          status: 'error',
          code: 'MEMORY_ID_NOT_FOUND_USER',
          output: `No memory with id "${rightId}" was found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      // Scope guard: caller must be able to see BOTH sides.
      const scopeTags = context?.scopeTags
      if (scopeTags && scopeTags.length > 0) {
        if (!isMemoryVisibleInScope(left.tags, scopeTags)
          || !isMemoryVisibleInScope(right.tags, scopeTags)) {
          return {
            status: 'error',
            code: 'MEMORY_SCOPE_MISMATCH_USER',
            output: 'One of the memories belongs to a different user/channel scope and is not visible from here.',
            durationMs: Date.now() - startedAt,
          }
        }
      }

      const leftTags = new Set((left.tags ?? []).map((tag) => tag.toLowerCase()))
      const rightTags = new Set((right.tags ?? []).map((tag) => tag.toLowerCase()))
      const tagsAdded = (right.tags ?? []).filter((tag) => !leftTags.has(tag.toLowerCase()))
      const tagsRemoved = (left.tags ?? []).filter((tag) => !rightTags.has(tag.toLowerCase()))

      const linesL = left.content.split(/\r?\n/)
      const linesR = right.content.split(/\r?\n/)
      const linesLSet = new Set(linesL)
      const linesRSet = new Set(linesR)
      const addedLines = linesR.filter((line) => !linesLSet.has(line))
      const removedLines = linesL.filter((line) => !linesRSet.has(line))

      const sourceChanged = left.source !== right.source

      return {
        status: 'success',
        output: JSON.stringify({
          leftId,
          rightId,
          identical: !sourceChanged
            && tagsAdded.length === 0
            && tagsRemoved.length === 0
            && addedLines.length === 0
            && removedLines.length === 0,
          source: sourceChanged
            ? { left: left.source, right: right.source }
            : undefined,
          tags: {
            added: tagsAdded,
            removed: tagsRemoved,
          },
          content: {
            added: addedLines.slice(0, 50),
            removed: removedLines.slice(0, 50),
            truncated: addedLines.length > 50 || removedLines.length > 50,
          },
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemorySummarizeTool(deps: MemorySummarizeToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.summarize',
    description: 'Compress several memories into a single LLM-generated summary. Pass either ids (explicit list — usually obtained from memory.search / by_tag / related) or tags (caller scope, AND match). Optional instruction tells the LLM what angle to take. Read-only: just returns the summary, never persists it.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory.summarize' },
    inputSchema: {
      type: 'object',
      properties: {
        ids: {
          type: 'array',
          items: { type: 'string' },
          description: 'Explicit memory ids to summarise (capped at 50).',
        },
        tags: {
          type: 'array',
          items: { type: 'string' },
          description: 'AND-match tag filter when ids is omitted. Restricted to caller scope.',
        },
        instruction: {
          type: 'string',
          description: 'Free-form instruction, e.g. "Summarise as a 5-bullet weekly digest" or "최근 결정만 한국어로 요약".',
        },
        targetChars: {
          type: 'number',
          description: 'Approximate target length of the summary (default 600, max 4000).',
        },
        limit: {
          type: 'number',
          description: 'When using tags, max memories to feed the LLM (default 20, max 50).',
        },
        saveTo: {
          type: 'object',
          description: 'When set, append the summary to a daily-note section after generation. {date?: today/yesterday/YYYY-MM-DD, section: required}.',
          properties: {
            section: { type: 'string' },
            date: { type: 'string' },
          },
          required: ['section'],
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex || !deps.summarize) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Memory summariser is not configured (no LLM provider for dreaming).',
          durationMs: Date.now() - startedAt,
        }
      }

      const idsInput = Array.isArray(input.ids) ? input.ids : []
      const tagsInput = Array.isArray(input.tags) ? input.tags : []
      const ids = idsInput
        .filter((id): id is string => typeof id === 'string')
        .map((id) => id.trim())
        .filter(Boolean)
        .slice(0, 50)
      const tags = tagsInput
        .filter((tag): tag is string => typeof tag === 'string')
        .map((tag) => tag.trim().toLowerCase())
        .filter(Boolean)

      if (ids.length === 0 && tags.length === 0) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'Provide either `ids` or `tags`.',
          durationMs: Date.now() - startedAt,
        }
      }
      const scopeTags = context?.scopeTags ?? []
      const limitInput = typeof input.limit === 'number' ? input.limit : 20
      const limit = Math.max(1, Math.min(Math.floor(limitInput), 50))

      const collected: Array<Omit<MemoryEntry, 'score'>> = []
      const sourceIds: string[] = []

      if (ids.length > 0) {
        for (const id of ids) {
          let entry: MemoryEntry | null
          try {
            entry = await deps.semanticIndex.get(id)
          } catch {
            continue
          }
          if (!entry) continue
          if (scopeTags.length > 0 && !isMemoryVisibleInScope(entry.tags, scopeTags)) continue
          if (isMemoryArchived(entry.tags) || isMemorySuperseded(entry.tags)) continue
          collected.push(entry)
          sourceIds.push(entry.id)
        }
      } else {
        let all: Array<Omit<MemoryEntry, 'score'>>
        try {
          all = await deps.semanticIndex.listRecent(5000)
        } catch (error) {
          return {
            status: 'error',
            code: 'MEMORY_SUMMARIZE_FAILED_TRANSIENT',
            output: `Failed to scan memory: ${error instanceof Error ? error.message : String(error)}`,
            durationMs: Date.now() - startedAt,
          }
        }
        for (const entry of all) {
          if (collected.length >= limit) break
          if (isMemoryArchived(entry.tags) || isMemorySuperseded(entry.tags)) continue
          if (scopeTags.length > 0 && !isMemoryVisibleInScope(entry.tags, scopeTags)) continue
          const lower = (entry.tags ?? []).map((tag) => tag.toLowerCase())
          if (!tags.every((tag) => lower.includes(tag))) continue
          collected.push(entry)
          sourceIds.push(entry.id)
        }
      }

      if (collected.length === 0) {
        return {
          status: 'error',
          code: 'MEMORY_SUMMARIZE_NO_INPUT_USER',
          output: 'No memories matched the supplied ids / tags within the active scope.',
          durationMs: Date.now() - startedAt,
        }
      }

      const targetChars = typeof input.targetChars === 'number' ? input.targetChars : 600
      const instruction = typeof input.instruction === 'string' ? input.instruction : undefined

      let summary: string | null
      try {
        summary = await deps.summarize(
          collected.map((entry) => entry.content),
          { instruction, targetChars },
        )
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_SUMMARIZE_FAILED_TRANSIENT',
          output: `Summariser failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (!summary) {
        return {
          status: 'error',
          code: 'MEMORY_SUMMARIZE_FAILED_TRANSIENT',
          output: 'LLM did not return a summary.',
          durationMs: Date.now() - startedAt,
        }
      }

      // Optional saveTo: append the summary to a daily-note section.
      let savedTo: { path?: string; section?: string; date?: string } | undefined
      const saveTo = (input.saveTo && typeof input.saveTo === 'object') ? input.saveTo as { date?: unknown; section?: unknown } : undefined
      if (saveTo) {
        const fileMemory = resolveFileMemory(deps, context?.scopeTags)
        if (!fileMemory || typeof fileMemory.appendToDailySection !== 'function') {
          return {
            status: 'error',
            code: 'MEMORY_UNAVAILABLE_USER',
            output: 'saveTo was supplied but file memory is not configured.',
            durationMs: Date.now() - startedAt,
          }
        }
        const section = typeof saveTo.section === 'string' ? saveTo.section.trim() : ''
        if (!section) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: 'saveTo.section is required when saveTo is set.',
            durationMs: Date.now() - startedAt,
          }
        }
        const date = parseDailyDate(saveTo.date)
        if (!date) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: 'saveTo.date must be "today", "yesterday", or YYYY-MM-DD.',
            durationMs: Date.now() - startedAt,
          }
        }
        try {
          await fileMemory.appendToDailySection(section, summary, date)
          savedTo = {
            section,
            date: formatDateKey(date),
            path: fileMemory.getDailyNotePath(date),
          }
        } catch (error) {
          return {
            status: 'error',
            code: 'MEMORY_SUMMARIZE_FAILED_TRANSIENT',
            output: `saveTo append failed: ${error instanceof Error ? error.message : String(error)}`,
            durationMs: Date.now() - startedAt,
          }
        }
      }

      return {
        status: 'success',
        output: JSON.stringify({
          totalSourceMemories: collected.length,
          sourceIds,
          summary,
          savedTo,
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryMergeTool(deps: MemoryMergeToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.merge',
    description: 'Manually combine 2+ memories into a single LLM-merged entry. Pass `ids` (the memories to combine, capped at 8) and optionally `winnerId` (which id keeps the new content; default first id). Other ids gain `archived` + `archived-by:<winnerId>` so audit history is preserved. ALWAYS run with confirm:false (default) first to preview the merged text — only confirm:true actually writes. Refuses cross-scope ids and reserved/system memories.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        ids: {
          type: 'array',
          items: { type: 'string' },
          description: '2-8 memory ids to merge.',
        },
        winnerId: {
          type: 'string',
          description: 'Which id keeps the merged content. Defaults to ids[0].',
        },
        instruction: {
          type: 'string',
          description: 'Optional steering for the merge (language, focus, format).',
        },
        confirm: {
          type: 'boolean',
          description: 'Default false (dryRun preview). True applies the merge.',
        },
        reason: {
          type: 'string',
          description: 'Recorded in the audit log when confirm:true.',
        },
      },
      required: ['ids'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex || !deps.summarize) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Memory merger is not configured (no LLM provider for dreaming).',
          durationMs: Date.now() - startedAt,
        }
      }
      const ids = (Array.isArray(input.ids) ? input.ids : [])
        .filter((id): id is string => typeof id === 'string')
        .map((id) => id.trim())
        .filter(Boolean)
      if (ids.length < 2 || ids.length > 8) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'Provide 2-8 ids to merge.',
          durationMs: Date.now() - startedAt,
        }
      }
      const dedup = Array.from(new Set(ids))
      if (dedup.length !== ids.length) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'ids must be unique.',
          durationMs: Date.now() - startedAt,
        }
      }
      const winnerInput = typeof input.winnerId === 'string' ? input.winnerId.trim() : ''
      const winnerId = winnerInput || ids[0]
      if (!ids.includes(winnerId)) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'winnerId must be one of the supplied ids.',
          durationMs: Date.now() - startedAt,
        }
      }

      const scopeTags = context?.scopeTags ?? []
      const entries: MemoryEntry[] = []
      for (const id of ids) {
        const entry = await deps.semanticIndex.get(id)
        if (!entry) {
          return {
            status: 'error',
            code: 'MEMORY_ID_NOT_FOUND_USER',
            output: `No memory with id "${id}" was found.`,
            durationMs: Date.now() - startedAt,
          }
        }
        if (scopeTags.length > 0 && !isMemoryWritableInScope(entry.tags, scopeTags)) {
          return {
            status: 'error',
            code: 'MEMORY_SCOPE_MISMATCH_USER',
            output: `Memory ${id} belongs to a different scope.`,
            durationMs: Date.now() - startedAt,
          }
        }
        if (isMemoryArchived(entry.tags) || isMemorySuperseded(entry.tags)) {
          return {
            status: 'error',
            code: 'MEMORY_RESERVED_TAG_USER',
            output: `Memory ${id} is archived/superseded; restore or use memory.update.`,
            durationMs: Date.now() - startedAt,
          }
        }
        entries.push(entry)
      }

      const winner = entries.find((entry) => entry.id === winnerId)!
      const losers = entries.filter((entry) => entry.id !== winnerId)
      const instruction = typeof input.instruction === 'string'
        ? input.instruction
        : 'Merge these memories into a single coherent entry. Keep all distinct facts; drop duplicates.'

      const merged = await deps.summarize(
        entries.map((entry) => entry.content),
        { instruction, targetChars: 800 },
      )
      if (!merged) {
        return {
          status: 'error',
          code: 'MEMORY_MERGE_FAILED_TRANSIENT',
          output: 'LLM did not return a merged text.',
          durationMs: Date.now() - startedAt,
        }
      }

      const confirm = input.confirm === true
      // Union of all non-system tags.
      const tagUnion = uniqueOrdered(
        entries.flatMap((entry) => entry.tags ?? []).filter((tag) => !isMemoryArchived([tag]) && !isMemorySuperseded([tag])),
      )

      if (!confirm) {
        return {
          status: 'success',
          output: JSON.stringify({
            dryRun: true,
            winnerId,
            losers: losers.map((entry) => entry.id),
            mergedPreview: truncateSnippet(merged),
            unionTagPreview: tagUnion.slice(0, 20),
          }),
          durationMs: Date.now() - startedAt,
        }
      }

      const reason = typeof input.reason === 'string' && input.reason.trim().length > 0
        ? input.reason.trim()
        : `manual merge of ${ids.join(', ')}`

      try {
        await deps.semanticIndex.add({
          id: winnerId,
          content: merged,
          source: winner.source,
          tags: tagUnion,
        })
        await deps.semanticIndex.recordAudit({
          memoryId: winnerId,
          action: 'updated',
          actor: 'agent',
          reason,
          before: {
            id: winnerId,
            content: winner.content,
            source: winner.source,
            tags: winner.tags ?? [],
          },
          after: {
            id: winnerId,
            content: merged,
            source: winner.source,
            tags: tagUnion,
          },
        })
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_MERGE_FAILED_TRANSIENT',
          output: `winner update failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }

      const archivedIds: string[] = []
      for (const loser of losers) {
        const archivedTags = uniqueOrdered([
          ...(loser.tags ?? []),
          'archived',
          `archived-by:${winnerId}`,
        ])
        try {
          await deps.semanticIndex.add({
            id: loser.id,
            content: loser.content,
            source: loser.source,
            tags: archivedTags,
          })
          await deps.semanticIndex.recordAudit({
            memoryId: loser.id,
            action: 'updated',
            actor: 'agent',
            reason: `archived as part of merge into ${winnerId}`,
            before: {
              id: loser.id,
              content: loser.content,
              source: loser.source,
              tags: loser.tags ?? [],
            },
            after: {
              id: loser.id,
              content: loser.content,
              source: loser.source,
              tags: archivedTags,
            },
          })
          archivedIds.push(loser.id)
        } catch {
          // continue
        }
      }

      return {
        status: 'success',
        output: JSON.stringify({
          dryRun: false,
          winnerId,
          archivedIds,
          mergedSnippet: truncateSnippet(merged),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryTagSuggestTool(deps: MemoryTagSuggestToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.tag.suggest',
    description: 'Ask the LLM to suggest 3-5 short topic tags for a piece of content. Use BEFORE memory.remember when the user has not provided tags themselves and the content is non-trivial — it raises later memory.search recall. Read-only: just returns suggestions, does not store anything.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory.tag.suggest', key: (input) => String(input.content ?? '') },
    inputSchema: {
      type: 'object',
      properties: {
        content: {
          type: 'string',
          description: 'The memory content you would save with memory.remember. The model decides language (Korean/English) from the content.',
        },
      },
      required: ['content'],
    },
    async execute(input): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.suggestTags) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Tag suggestion provider is not configured (no LLM provider for dreaming).',
          durationMs: Date.now() - startedAt,
        }
      }
      const content = typeof input.content === 'string' ? input.content.trim() : ''
      if (!content) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'content is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      let tags: string[]
      try {
        tags = await deps.suggestTags(content)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_TAG_SUGGEST_FAILED_TRANSIENT',
          output: `Tag suggestion failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
      return {
        status: 'success',
        output: JSON.stringify({
          totalSuggestions: tags.length,
          tags,
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryRemindAtTool(deps: MemoryReminderToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.remind_at',
    description: 'Schedule a future reminder. The daemon delivers the message back through the same channel/user that asked, at the requested time. Use when the user says "remind me at <time>", "X시에 알려줘", "1시간 후에 알려줘", etc. Reminders are scoped to the active user/channel — other people will not be notified.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory.reminders' },
    inputSchema: {
      type: 'object',
      properties: {
        when: {
          type: 'string',
          description: 'When to fire the reminder. Accepts: relative durations like "30m" / "2h" / "1d" / "in 2 hours", "tomorrow" (defaults to 9am), "today" (next hour), or an ISO-8601 timestamp like "2026-05-09T15:00:00".',
        },
        content: {
          type: 'string',
          description: 'The message to send when the reminder fires.',
        },
      },
      required: ['when', 'content'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.reminders) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Reminder store is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }

      const when = typeof input.when === 'string' ? input.when : ''
      const content = typeof input.content === 'string' ? input.content.trim() : ''
      if (!when || !content) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'Both `when` and non-empty `content` are required.',
          durationMs: Date.now() - startedAt,
        }
      }

      const dueAt = parseReminderTime(when)
      if (!dueAt) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: `Could not parse "when" value "${when}". Use a relative duration (e.g. "30m"), "tomorrow", or an ISO-8601 timestamp.`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (dueAt.getTime() <= Date.now()) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: `Reminder time ${dueAt.toISOString()} is in the past.`,
          durationMs: Date.now() - startedAt,
        }
      }

      const scopeTags = context?.scopeTags ?? []
      const scope = parseScopeFromTags(scopeTags)

      // Prefer the live channelContext (carries the channel handle that just
      // delivered this turn) over scope tags — scope tags are stable but a
      // channelContext.chatKey is what RemindersScheduler needs to route the
      // delivery back to the same chat. Fall back to the scope tags so REST
      // callers without channelContext keep working.
      const ctxChannel = context?.channelContext
      const channelType = ctxChannel?.channel ?? scope.channelType
      const chatId = ctxChannel?.chatKey
        ? extractChatIdFromKey(ctxChannel.chatKey, ctxChannel.channel)
        : scope.chatId

      const reminder = await deps.reminders.add({
        dueAt: dueAt.toISOString(),
        content,
        scopeTags,
        channelType,
        chatId,
      })

      return {
        status: 'success',
        output: JSON.stringify({
          reminderId: reminder.id,
          dueAt: reminder.dueAt,
          content: reminder.content,
          delivery: channelType && chatId
            ? { channelType, chatId }
            : { warning: 'No channel scope detected — reminder is stored but cannot be auto-delivered.' },
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

/**
 * channelContext.chatKey is the canonical session key the channel pipeline
 * uses internally — for most channels it's already `<channelType>:<channelId>`.
 * Pull just the channelId portion so RemindersScheduler can address the
 * channel SDK with a plain chat id.
 */
function extractChatIdFromKey(chatKey: string, channelType?: string): string {
  if (!chatKey) return chatKey
  if (channelType) {
    const prefix = `${channelType}:`
    if (chatKey.startsWith(prefix)) return chatKey.slice(prefix.length)
  }
  return chatKey
}

export function createMemoryRemindersListTool(deps: MemoryReminderToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.reminders.list',
    description: 'List pending (not yet fired or cancelled) reminders for the active user/channel scope.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory.reminders' },
    inputSchema: { type: 'object', properties: {} },
    async execute(_input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.reminders) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Reminder store is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }
      const scopeTags = context?.scopeTags ?? []
      const reminders = await deps.reminders.listForScope(scopeTags)
      return {
        status: 'success',
        output: JSON.stringify({
          totalPending: reminders.length,
          reminders: reminders.map((reminder) => ({
            id: reminder.id,
            dueAt: reminder.dueAt,
            content: reminder.content,
            channelType: reminder.channelType,
            chatId: reminder.chatId,
          })),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryRemindersCancelTool(deps: MemoryReminderToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.reminders.cancel',
    description: 'Cancel a pending reminder by id. Use after memory.reminders.list to find the id.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory.reminders' },
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Reminder id (from memory.reminders.list).' },
        reason: { type: 'string', description: 'Optional human note recorded with the cancellation.' },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.reminders) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Reminder store is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }
      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (!id) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A reminder id is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      // Confirm the reminder belongs to the caller scope before cancelling.
      const scopeTags = context?.scopeTags ?? []
      const ours = await deps.reminders.listForScope(scopeTags)
      const match = ours.find((reminder) => reminder.id === id)
      if (!match) {
        return {
          status: 'error',
          code: 'MEMORY_REMINDER_NOT_FOUND_USER',
          output: `Reminder ${id} was not found in the active scope.`,
          durationMs: Date.now() - startedAt,
        }
      }
      const cancelled = await deps.reminders.cancel(
        id,
        typeof input.reason === 'string' && input.reason.trim().length > 0
          ? input.reason.trim()
          : undefined,
      )
      if (!cancelled) {
        return {
          status: 'error',
          code: 'MEMORY_REMINDER_NOT_FOUND_USER',
          output: `Reminder ${id} could not be cancelled (already fired or cancelled).`,
          durationMs: Date.now() - startedAt,
        }
      }
      return {
        status: 'success',
        output: JSON.stringify({ id: cancelled.id, cancelled: true }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

interface MemoryExportPayload {
  version: '1.0'
  exportedAt: string
  scope?: {
    userId?: string
    channelType?: string
    chatId?: string
    sessionId?: string
  }
  fileMemory?: { sections: Array<{ title: string; content: string }> }
  semanticEntries?: Array<{ id: string; content: string; source: MemoryEntry['source']; tags: string[]; evidence?: MemoryEntry['evidence'] }>
  reminders?: Array<{
    id?: string
    dueAt: string
    content: string
    scopeTags: string[]
    channelType?: string
    chatId?: string
  }>
}

const SEMANTIC_LIST_LIMIT = 5000

function rejectScopedGraphAccess(
  context: ToolExecutionContext | undefined,
  startedAt: number,
): ToolResult | null {
  if (!context?.scopeTags || context.scopeTags.length === 0) return null
  return {
    status: 'error',
    code: 'MEMORY_SCOPE_BYPASS_DENIED_USER',
    output: 'Memory graph tools require an unscoped/admin execution context until graph projection is scope-aware.',
    durationMs: Date.now() - startedAt,
  }
}

function rejectScopedIncludeAll(
  input: Record<string, unknown>,
  context: ToolExecutionContext | undefined,
  startedAt: number,
): ToolResult | null {
  if (input.includeAllScopes !== true || !context?.scopeTags || context.scopeTags.length === 0) {
    return null
  }
  return {
    status: 'error',
    code: 'MEMORY_SCOPE_BYPASS_DENIED_USER',
    output: 'includeAllScopes is only allowed from an unscoped/admin execution context.',
    durationMs: Date.now() - startedAt,
  }
}

export function createMemoryExportTool(deps: MemoryExportImportToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.export',
    description: 'Dump the durable memory visible to the active scope as a single JSON document. Used for backup, migration, or audit. Includes long-term file sections, scoped semantic entries, and pending reminders. Pass includeAllScopes:true ONLY for an admin export — that pulls in other users\' memories.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        includeFile: { type: 'boolean', description: 'Include MEMORY.md sections (default true).' },
        includeSemantic: { type: 'boolean', description: 'Include semantic memory entries (default true).' },
        includeReminders: { type: 'boolean', description: 'Include pending reminders (default true).' },
        includeAllScopes: { type: 'boolean', description: 'Admin-only: include other scopes\' memories (default false).' },
        createdAfter: {
          type: 'string',
          description: 'Optional ISO-8601 lower bound on semantic entry created_at — only memories created on or after this date are exported.',
        },
        createdBefore: {
          type: 'string',
          description: 'Optional ISO-8601 upper bound on semantic entry created_at.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      const includeFile = input.includeFile !== false
      const includeSemantic = input.includeSemantic !== false
      const includeReminders = input.includeReminders !== false
      const scopeBypassError = rejectScopedIncludeAll(input, context, startedAt)
      if (scopeBypassError) return scopeBypassError
      const includeAllScopes = input.includeAllScopes === true
      const scopeTags = context?.scopeTags ?? []

      // Validate optional time-range filters before doing any work.
      let createdAfterIso: string | undefined
      let createdBeforeIso: string | undefined
      if (typeof input.createdAfter === 'string' && input.createdAfter.trim().length > 0) {
        const parsed = new Date(input.createdAfter)
        if (Number.isNaN(parsed.valueOf())) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: `createdAfter "${input.createdAfter}" is not a valid timestamp.`,
            durationMs: Date.now() - startedAt,
          }
        }
        createdAfterIso = parsed.toISOString()
      }
      if (typeof input.createdBefore === 'string' && input.createdBefore.trim().length > 0) {
        const parsed = new Date(input.createdBefore)
        if (Number.isNaN(parsed.valueOf())) {
          return {
            status: 'error',
            code: 'INVALID_INPUT_PERMANENT',
            output: `createdBefore "${input.createdBefore}" is not a valid timestamp.`,
            durationMs: Date.now() - startedAt,
          }
        }
        createdBeforeIso = parsed.toISOString()
      }

      const payload: MemoryExportPayload = {
        version: '1.0',
        exportedAt: new Date().toISOString(),
        scope: parseScopeFromTags(scopeTags),
      }

      const exportFileMemory = resolveFileMemory(deps, scopeTags)
      if (includeFile && exportFileMemory) {
        const sections = await exportFileMemory.readMemorySections()
        payload.fileMemory = {
          sections: sections.map((s) => ({ title: s.title, content: s.content })),
        }
      }

      if (includeSemantic && deps.semanticIndex) {
        let all: Array<Omit<MemoryEntry, 'score'>> = []
        try {
          if ((createdAfterIso || createdBeforeIso) && typeof deps.semanticIndex.listRecentWithTimestamps === 'function') {
            // Use the time-range-aware variant when available so the SQL
            // backend can prune at the source.
            all = await deps.semanticIndex.listRecentWithTimestamps(SEMANTIC_LIST_LIMIT, {
              createdAfter: createdAfterIso,
              createdBefore: createdBeforeIso,
            })
          } else {
            all = await deps.semanticIndex.listRecent(SEMANTIC_LIST_LIMIT)
          }
        } catch (error) {
          return {
            status: 'error',
            code: 'MEMORY_EXPORT_FAILED_TRANSIENT',
            output: `Failed to list semantic memory: ${error instanceof Error ? error.message : String(error)}`,
            durationMs: Date.now() - startedAt,
          }
        }
        const filtered = (includeAllScopes || scopeTags.length === 0)
          ? all
          : all.filter((entry) => isMemoryVisibleInScope(entry.tags, scopeTags))
        payload.semanticEntries = filtered.map((entry) => ({
          evidence: entry.evidence,
          id: entry.id,
          content: entry.content,
          source: entry.source,
          tags: entry.tags,
        }))
      }

      if (includeReminders && deps.reminders) {
        const list = await deps.reminders.listForScope(includeAllScopes ? [] : scopeTags)
        payload.reminders = list.map((reminder) => ({
          id: reminder.id,
          dueAt: reminder.dueAt,
          content: reminder.content,
          scopeTags: reminder.scopeTags,
          channelType: reminder.channelType,
          chatId: reminder.chatId,
        }))
      }

      const json = JSON.stringify(payload, null, 2)
      return {
        status: 'success',
        output: JSON.stringify({
          bytes: json.length,
          counts: {
            fileSections: payload.fileMemory?.sections.length ?? 0,
            semanticEntries: payload.semanticEntries?.length ?? 0,
            reminders: payload.reminders?.length ?? 0,
          },
          payload: json,
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryImportTool(deps: MemoryExportImportToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.import',
    description: 'Restore a memory.export JSON dump. Default conflict policy is "skip" (existing semantic entries with the same id are kept); use "replace" to overwrite. Always preview with dryRun:true first to see counts before applying.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'sequential', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        data: { type: 'string', description: 'Either the raw JSON string emitted by memory.export, or the parsed payload object.' },
        conflict: {
          type: 'string',
          enum: ['skip', 'replace'],
          description: 'How to handle pre-existing semantic memory ids. Default skip.',
        },
        dryRun: { type: 'boolean', description: 'Preview counts without applying. Default false.' },
        applyIds: {
          type: 'array',
          items: { type: 'string' },
          description: 'Optional whitelist: only semantic entries whose id is in this list are applied. Use after a dryRun to commit a subset. File-memory sections and reminders are always processed (use includeFile / includeReminders to gate those if needed).',
        },
      },
      required: ['data'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      let payload: MemoryExportPayload
      try {
        payload = parseExportPayload(input.data)
      } catch (error) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: `Could not parse import payload: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }

      const conflict = input.conflict === 'replace' ? 'replace' : 'skip'
      const dryRun = input.dryRun === true
      const scopeTags = context?.scopeTags ?? []

      const stats = {
        fileSectionsApplied: 0,
        semanticAdded: 0,
        semanticReplaced: 0,
        semanticSkipped: 0,
        remindersAdded: 0,
        warnings: [] as string[],
        // Populated only when dryRun=true so the caller can preview which
        // ids would be added vs. replaced vs. skipped (cross-scope or
        // already-present under conflict=skip).
        addIds: [] as string[],
        replaceIds: [] as string[],
        skipIds: [] as string[],
      }

      const importFileMemory = resolveFileMemory(deps, scopeTags)
      if (payload.fileMemory && Array.isArray(payload.fileMemory.sections) && importFileMemory) {
        for (const section of payload.fileMemory.sections) {
          if (!section.title || !section.content) continue
          if (!dryRun) {
            try {
              const items = section.content
                .split('\n')
                .map((line) => line.trim())
                .filter((line) => line.startsWith('- '))
                .map((line) => line.slice(2).trim())
                .filter((line) => line.length > 0)
              if (items.length > 0) {
                await importFileMemory.mergeMemorySectionItems(section.title, items)
              } else {
                await importFileMemory.replaceMemorySection(section.title, section.content)
              }
            } catch (error) {
              stats.warnings.push(`fileSection ${section.title}: ${error instanceof Error ? error.message : String(error)}`)
              continue
            }
          }
          stats.fileSectionsApplied += 1
        }
      } else if (payload.fileMemory && !importFileMemory) {
        stats.warnings.push('file memory store unavailable; fileMemory section ignored')
      }

      const applyIdsFilter = Array.isArray(input.applyIds)
        ? new Set(
            input.applyIds
              .filter((id): id is string => typeof id === 'string')
              .map((id) => id.trim())
              .filter(Boolean),
          )
        : null
      if (Array.isArray(payload.semanticEntries) && deps.semanticIndex) {
        for (const entry of payload.semanticEntries) {
          if (!entry.id || !entry.content) continue
          if (applyIdsFilter && !applyIdsFilter.has(entry.id)) continue
          // Strip any scope:* tags from the source before re-tagging with the
          // importer scope. Keeping the source's scope tags would let an
          // imported dump re-materialize memories tagged for *another* user's
          // scope, making them visible cross-scope.
          const sourceTagsWithoutScope = (entry.tags ?? []).filter(
            (tag) => typeof tag === 'string' && !tag.toLowerCase().startsWith('scope:'),
          )
          const importTags = attachScopeTags(sourceTagsWithoutScope, scopeTags)
          // Coerce an unknown/hostile source to a safe allowed value.
          const importSource: MemoryEntry['source'] =
            (ALLOWED_MEMORY_SOURCES as readonly string[]).includes(entry.source)
              ? entry.source
              : 'user'
          let existing: MemoryEntry | null = null
          try {
            existing = await deps.semanticIndex.get(entry.id)
          } catch {
            existing = null
          }
          if (existing) {
            if (conflict === 'skip') {
              stats.semanticSkipped += 1
              if (dryRun) stats.skipIds.push(entry.id)
              continue
            }
            // Refuse to replace a memory belonging to another scope.
            if (!isMemoryWritableInScope(existing.tags, scopeTags)) {
              stats.semanticSkipped += 1
              if (dryRun) stats.skipIds.push(entry.id)
              stats.warnings.push(`semantic ${entry.id}: scope mismatch (kept existing)`)
              continue
            }
            if (!dryRun) {
              try {
                await deps.semanticIndex.add({
                  id: entry.id,
                  evidence: entry.evidence,
                  content: entry.content,
                  source: importSource,
                  tags: importTags,
                })
              } catch (error) {
                stats.warnings.push(`semantic ${entry.id}: ${error instanceof Error ? error.message : String(error)}`)
                continue
              }
            }
            stats.semanticReplaced += 1
            if (dryRun) stats.replaceIds.push(entry.id)
          } else {
            if (!dryRun) {
              try {
                await deps.semanticIndex.add({
                  id: entry.id,
                  evidence: entry.evidence,
                  content: entry.content,
                  source: importSource,
                  tags: importTags,
                })
              } catch (error) {
                stats.warnings.push(`semantic ${entry.id}: ${error instanceof Error ? error.message : String(error)}`)
                continue
              }
            }
            stats.semanticAdded += 1
            if (dryRun) stats.addIds.push(entry.id)
          }
        }
      } else if (payload.semanticEntries && !deps.semanticIndex) {
        stats.warnings.push('semantic store unavailable; semanticEntries ignored')
      }

      if (Array.isArray(payload.reminders) && deps.reminders) {
        const nowMs = Date.now()
        for (const reminder of payload.reminders) {
          if (!reminder.dueAt || !reminder.content) continue
          if (Date.parse(reminder.dueAt) <= nowMs) {
            stats.warnings.push(`reminder due in the past dropped (${reminder.dueAt})`)
            continue
          }
          if (!dryRun) {
            try {
              await deps.reminders.add({
                dueAt: reminder.dueAt,
                content: reminder.content,
                // Strip foreign scope tags before re-tagging with importer scope.
                scopeTags: attachScopeTags(
                  (reminder.scopeTags ?? []).filter(
                    (tag) => typeof tag === 'string' && !tag.toLowerCase().startsWith('scope:'),
                  ),
                  scopeTags,
                ),
                channelType: reminder.channelType,
                chatId: reminder.chatId,
              })
            } catch (error) {
              stats.warnings.push(`reminder: ${error instanceof Error ? error.message : String(error)}`)
              continue
            }
          }
          stats.remindersAdded += 1
        }
      } else if (payload.reminders && !deps.reminders) {
        stats.warnings.push('reminders store unavailable; reminders ignored')
      }

      // Trim id lists when not in dryRun so the response stays compact.
      const responseStats: Record<string, unknown> = { ...stats }
      if (!dryRun) {
        delete responseStats.addIds
        delete responseStats.replaceIds
        delete responseStats.skipIds
      }
      return {
        status: 'success',
        output: JSON.stringify({
          dryRun,
          conflict,
          source: {
            version: payload.version,
            exportedAt: payload.exportedAt,
            scope: payload.scope,
          },
          ...responseStats,
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryAccessTouchTool(deps: MemoryAccessToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.access.touch',
    description: 'Mark one or more memories as recently accessed. Bumps an internal access counter and last_accessed_at. The dreaming pipeline uses these signals when recalculating importance, and the prune phase protects rows with non-zero access counters from stale cleanup. Useful when you used a memory off-band (e.g. quoted it directly without searching). The counter mutation is silent and not recorded in the audit log.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        ids: {
          type: 'array',
          items: { type: 'string' },
          description: 'One or more memory ids to record an access for. Unknown ids are silently ignored.',
        },
      },
      required: ['ids'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }
      const ids = Array.isArray(input.ids)
        ? input.ids
            .filter((id): id is string => typeof id === 'string')
            .map((id) => id.trim())
            .filter((id) => id.length > 0)
        : []
      if (ids.length === 0) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'At least one memory id is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      const scopeTags = context?.scopeTags ?? []
      let writableIds = ids
      if (scopeTags.length > 0) {
        if (!deps.semanticIndex.get) {
          return {
            status: 'error',
            code: 'MEMORY_SCOPE_BYPASS_DENIED_USER',
            output: 'Scoped access tracking requires ownership lookup support.',
            durationMs: Date.now() - startedAt,
          }
        }
        const entries = await Promise.all(ids.map((id) => deps.semanticIndex!.get!(id)))
        writableIds = ids.filter((_id, index) => {
          const entry = entries[index]
          return Boolean(entry && isMemoryWritableInScope(entry.tags, scopeTags))
        })
      }
      try {
        if (writableIds.length > 0) await deps.semanticIndex.recordAccess(writableIds)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_ACCESS_FAILED_TRANSIENT',
          output: `recordAccess failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
      return {
        status: 'success',
        output: JSON.stringify({ touched: writableIds.length, ids: writableIds }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryAccessHotTool(deps: MemoryAccessToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.access.hot',
    description: 'Return the most-frequently-accessed memories (by retrieval hit count and last access time). Useful for "what do you keep coming back to?" or for an admin dashboard. Read-only, scope-aware. Hits with access_count = 0 are excluded.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        limit: {
          type: 'number',
          description: 'Max hits to return (default 20, max 100).',
        },
        includeAllScopes: {
          type: 'boolean',
          description: 'Admin opt-in to bypass scope filter. Default false.',
        },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }
      const limitInput = typeof input.limit === 'number' ? input.limit : 20
      const limit = Math.max(1, Math.min(Math.floor(limitInput), 100))
      const scopeTags = context?.scopeTags ?? []
      const scopeBypassError = rejectScopedIncludeAll(input, context, startedAt)
      if (scopeBypassError) return scopeBypassError
      const includeAllScopes = input.includeAllScopes === true

      let raw: MemoryAccessHotEntry[]
      try {
        raw = await deps.semanticIndex.listHotMemories(includeAllScopes ? limit : limit * 4)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_ACCESS_FAILED_TRANSIENT',
          output: `listHotMemories failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }

      const visible = raw
        .filter((entry) => !isMemoryArchived(entry.tags) && !isMemorySuperseded(entry.tags))
        .filter((entry) => includeAllScopes
          || scopeTags.length === 0
          || isMemoryVisibleInScope(entry.tags, scopeTags))
        .slice(0, limit)

      return {
        status: 'success',
        output: JSON.stringify({
          totalHits: visible.length,
          hits: visible.map((entry) => ({
            id: entry.id,
            snippet: truncateSnippet(entry.content),
            source: entry.source,
            tags: entry.tags,
            accessCount: entry.accessCount,
            lastAccessedAt: entry.lastAccessedAt,
          })),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryPinTool(deps: MemoryPinToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.pin',
    description: 'Mark a memory as prune-immune. The dreaming pipeline\'s stale-cleanup phase skips pinned rows regardless of age, importance, or access frequency. Use for SOPs, critical preferences, long-running ticket references — anything the user explicitly never wants forgotten. Idempotent. Pin also bumps last_accessed_at to refresh the row\'s "warm" state.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Memory id to pin.' },
        reason: { type: 'string', description: 'Optional audit-log reason ("user requested", "SOP").' },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }
      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (!id) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A memory id is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      const existing = await deps.semanticIndex.get(id)
      if (!existing) {
        return {
          status: 'error',
          code: 'MEMORY_ID_NOT_FOUND_USER',
          output: `No memory with id "${id}" was found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      const scopeTags = context?.scopeTags ?? []
      if (scopeTags.length > 0 && !isMemoryWritableInScope(existing.tags, scopeTags)) {
        return {
          status: 'error',
          code: 'MEMORY_ID_NOT_FOUND_USER',
          output: `No memory with id "${id}" was found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      try {
        await deps.semanticIndex.pin(id)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_PIN_FAILED_TRANSIENT',
          output: `pin failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (deps.semanticIndex.recordAudit) {
        await deps.semanticIndex.recordAudit({
          memoryId: id,
          action: 'updated',
          actor: 'memory.pin',
          reason: typeof input.reason === 'string' && input.reason.trim() ? input.reason.trim() : 'pinned',
          before: existing,
          after: existing,
        }).catch(() => {})
      }
      return {
        status: 'success',
        output: JSON.stringify({ id, pinned: true }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryUnpinTool(deps: MemoryPinToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.unpin',
    description: 'Clear the pin flag on a memory so the dreaming prune phase can reclaim it on a future cleanup run. Use when the user says the entry is no longer indispensable. Idempotent (unpinning an already-unpinned memory is a no-op).',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string' },
        reason: { type: 'string' },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }
      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (!id) {
        return {
          status: 'error',
          code: 'INVALID_INPUT_PERMANENT',
          output: 'A memory id is required.',
          durationMs: Date.now() - startedAt,
        }
      }
      const existing = await deps.semanticIndex.get(id)
      if (!existing) {
        return {
          status: 'error',
          code: 'MEMORY_ID_NOT_FOUND_USER',
          output: `No memory with id "${id}" was found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      const scopeTags = context?.scopeTags ?? []
      if (scopeTags.length > 0 && !isMemoryWritableInScope(existing.tags, scopeTags)) {
        return {
          status: 'error',
          code: 'MEMORY_ID_NOT_FOUND_USER',
          output: `No memory with id "${id}" was found.`,
          durationMs: Date.now() - startedAt,
        }
      }
      try {
        await deps.semanticIndex.unpin(id)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_PIN_FAILED_TRANSIENT',
          output: `unpin failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
      if (deps.semanticIndex.recordAudit) {
        await deps.semanticIndex.recordAudit({
          memoryId: id,
          action: 'updated',
          actor: 'memory.unpin',
          reason: typeof input.reason === 'string' && input.reason.trim() ? input.reason.trim() : 'unpinned',
          before: existing,
          after: existing,
        }).catch(() => {})
      }
      return {
        status: 'success',
        output: JSON.stringify({ id, pinned: false }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

export function createMemoryPinnedListTool(deps: MemoryPinToolDeps): ToolDefinitionRuntime {
  return {
    name: 'memory.pinned.list',
    description: 'List the memories currently pinned (prune-immune). Read-only, scope-aware. Useful for "what have I pinned?" reviews and for surfacing entries that should be unpinned when the user moves on.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'memory' },
    inputSchema: {
      type: 'object',
      properties: {
        limit: { type: 'number', description: 'Max hits (default 50, max 500).' },
        includeAllScopes: { type: 'boolean', description: 'Admin opt-in. Default false.' },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!deps.semanticIndex) {
        return {
          status: 'error',
          code: 'MEMORY_UNAVAILABLE_USER',
          output: 'Semantic memory is not configured on this daemon.',
          durationMs: Date.now() - startedAt,
        }
      }
      const limitInput = typeof input.limit === 'number' ? input.limit : 50
      const limit = Math.max(1, Math.min(Math.floor(limitInput), 500))
      const scopeTags = context?.scopeTags ?? []
      const scopeBypassError = rejectScopedIncludeAll(input, context, startedAt)
      if (scopeBypassError) return scopeBypassError
      const includeAllScopes = input.includeAllScopes === true
      const fetchLimit = (!includeAllScopes && scopeTags.length > 0) ? Math.min(limit * 4, 500) : limit
      let raw: MemoryPinnedEntry[]
      try {
        raw = await deps.semanticIndex.listPinned(fetchLimit)
      } catch (error) {
        return {
          status: 'error',
          code: 'MEMORY_PIN_FAILED_TRANSIENT',
          output: `listPinned failed: ${error instanceof Error ? error.message : String(error)}`,
          durationMs: Date.now() - startedAt,
        }
      }
      const visible = raw
        .filter((entry) => !isMemoryArchived(entry.tags) && !isMemorySuperseded(entry.tags))
        .filter((entry) => includeAllScopes
          || scopeTags.length === 0
          || isMemoryVisibleInScope(entry.tags, scopeTags))
        .slice(0, limit)
      return {
        status: 'success',
        output: JSON.stringify({
          totalHits: visible.length,
          hits: visible.map((entry) => ({
            id: entry.id,
            snippet: truncateSnippet(entry.content),
            source: entry.source,
            tags: entry.tags,
            pinnedAt: entry.pinnedAt,
          })),
        }),
        durationMs: Date.now() - startedAt,
      }
    },
  }
}

function parseExportPayload(input: unknown): MemoryExportPayload {
  if (input === null || input === undefined) {
    throw new Error('payload is empty')
  }
  let parsed: unknown
  if (typeof input === 'object') {
    // Object payloads previously bypassed the version/shape gate entirely — a
    // malformed object (or one with non-array sections) then threw uncaught
    // inside the import loop. Validate it the same as a parsed string.
    parsed = input
  } else if (typeof input === 'string') {
    const trimmed = input.trim()
    if (!trimmed) throw new Error('payload is empty')
    parsed = JSON.parse(trimmed)
  } else {
    throw new Error(`expected string or object, got ${typeof input}`)
  }
  if (!parsed || typeof parsed !== 'object') {
    throw new Error('payload is not an object')
  }
  const candidate = parsed as Record<string, unknown>
  if (candidate.version !== '1.0') {
    throw new Error(`unsupported export version ${String(candidate.version)}`)
  }
  // Validate the optional section shapes up front so a non-array section can
  // never throw an uncaught error mid-import (crashing the tool call).
  if (candidate.semanticEntries !== undefined && !Array.isArray(candidate.semanticEntries)) {
    throw new Error('semanticEntries must be an array')
  }
  if (candidate.reminders !== undefined && !Array.isArray(candidate.reminders)) {
    throw new Error('reminders must be an array')
  }
  if (candidate.fileMemory !== undefined) {
    const fileMemory = candidate.fileMemory
    if (!fileMemory || typeof fileMemory !== 'object'
      || !Array.isArray((fileMemory as Record<string, unknown>).sections)) {
      throw new Error('fileMemory.sections must be an array')
    }
  }
  return parsed as MemoryExportPayload
}

function resolveFileMemory<TFile>(
  deps: { fileMemoryRegistry?: ScopedFileMemoryRegistry; fileMemory?: TFile },
  scopeTags: string[] | undefined,
): TFile | undefined {
  if (deps.fileMemoryRegistry) {
    return deps.fileMemoryRegistry.get(scopeTags) as unknown as TFile
  }
  return deps.fileMemory
}

function uniqueOrdered(values: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of values) {
    const key = value.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(value)
  }
  return out
}

function tagsEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  const sortedA = [...a].sort()
  const sortedB = [...b].sort()
  return sortedA.every((value, index) => value === sortedB[index])
}

function extractBulletItems(content: string): string[] {
  return content
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.startsWith('- '))
    .map((line) => line.slice(2).trim())
    .filter((line) => line.length > 0)
}

function truncateSnippet(content: string): string {
  const trimmed = content.trim().replace(/\s+/g, ' ')
  if (trimmed.length <= SEARCH_SNIPPET_MAX_CHARS) {
    return trimmed
  }
  return `${trimmed.slice(0, SEARCH_SNIPPET_MAX_CHARS - 1).trimEnd()}…`
}

function serializeMemoryGraphNode(node: MemoryGraphNode): Record<string, unknown> {
  return {
    id: node.id,
    label: node.label,
    kind: node.kind,
    aliases: node.aliases,
    tags: node.tags,
    evidenceMemoryIds: node.evidenceMemoryIds,
    confidence: node.confidence,
    createdAt: node.createdAt,
    updatedAt: node.updatedAt,
    lastSeenAt: node.lastSeenAt,
  }
}

function serializeMemoryGraphEdge(edge: MemoryGraphEdge): Record<string, unknown> {
  return {
    id: edge.id,
    fromNodeId: edge.fromNodeId,
    toNodeId: edge.toNodeId,
    relation: edge.relation,
    tags: edge.tags,
    evidenceMemoryIds: edge.evidenceMemoryIds,
    confidence: edge.confidence,
    createdAt: edge.createdAt,
    updatedAt: edge.updatedAt,
    lastSeenAt: edge.lastSeenAt,
  }
}

function serializeMemoryGraphWikiPage(page: MemoryGraphWikiPage): Record<string, unknown> {
  return {
    query: page.query,
    node: page.node ? serializeMemoryGraphNode(page.node) : null,
    relationships: page.relationships.map((relationship) => ({
      direction: relationship.direction,
      edge: serializeMemoryGraphEdge(relationship.edge),
      node: serializeMemoryGraphNode(relationship.node),
    })),
    evidence: page.evidence.map((entry) => ({
      id: entry.id,
      snippet: truncateSnippet(entry.content),
      source: entry.source,
      tags: entry.tags,
    })),
    quality: page.quality ? serializeMemoryGraphWikiPageQuality(page.quality) : undefined,
  }
}

function serializeMemoryGraphWikiPageQuality(
  quality: NonNullable<MemoryGraphWikiPage['quality']>,
): Record<string, unknown> {
  return {
    score: quality.score,
    confidence: quality.confidence,
    evidenceCount: quality.evidenceCount,
    activeEvidenceCount: quality.activeEvidenceCount,
    inactiveEvidenceCount: quality.inactiveEvidenceCount,
    missingEvidenceCount: quality.missingEvidenceCount,
    relationshipCount: quality.relationshipCount,
    contradictionCount: quality.contradictionCount,
    staleRelationshipCount: quality.staleRelationshipCount,
    signals: quality.signals,
  }
}

function serializeMemoryGraphQualityReport(report: MemoryGraphQualityReport): Record<string, unknown> {
  return {
    generatedAt: report.generatedAt,
    stats: report.stats,
    scannedNodes: report.scannedNodes,
    scannedEdges: report.scannedEdges,
    lowConfidenceNodes: report.lowConfidenceNodes,
    lowConfidenceEdges: report.lowConfidenceEdges,
    thinEvidenceNodes: report.thinEvidenceNodes,
    thinEvidenceEdges: report.thinEvidenceEdges,
    missingEvidenceNodes: report.missingEvidenceNodes,
    missingEvidenceEdges: report.missingEvidenceEdges,
    inactiveEvidenceNodes: report.inactiveEvidenceNodes,
    inactiveEvidenceEdges: report.inactiveEvidenceEdges,
    staleNodes: report.staleNodes,
    staleEdges: report.staleEdges,
    orphanNodes: report.orphanNodes,
    contradictions: report.contradictions,
    signals: report.signals,
  }
}

async function validateGraphRepairDecisionScope(
  deps: MemoryGraphToolDeps,
  context: ToolExecutionContext | undefined,
  winnerMemoryId: string,
  supersededMemoryIds: string[],
  startedAt: number,
): Promise<ToolResult | null> {
  const scopeTags = context?.scopeTags ?? []
  if (scopeTags.length === 0) return null
  if (!deps.semanticIndex?.get) {
    return {
      status: 'error',
      code: 'MEMORY_SCOPE_CHECK_UNAVAILABLE_USER',
      output: 'Scoped graph repair decisions require memory scope checks, but this daemon does not expose them.',
      durationMs: Date.now() - startedAt,
    }
  }

  const winner = await deps.semanticIndex.get(winnerMemoryId)
  if (winner && !isMemoryVisibleInScope(winner.tags, scopeTags)) {
    return {
      status: 'error',
      code: 'MEMORY_SCOPE_MISMATCH_USER',
      output: 'The winner memory belongs to a different user/channel scope.',
      durationMs: Date.now() - startedAt,
    }
  }

  for (const id of supersededMemoryIds) {
    const entry = await deps.semanticIndex.get(id)
    if (!entry) continue
    if (!isMemoryWritableInScope(entry.tags, scopeTags)) {
      return {
        status: 'error',
        code: 'MEMORY_SCOPE_MISMATCH_USER',
        output: 'A superseded memory belongs to a different user/channel scope.',
        durationMs: Date.now() - startedAt,
      }
    }
  }
  return null
}

function serializeMemoryGraphRepairResult(result: MemoryGraphRepairResult): Record<string, unknown> {
  return {
    dryRun: result.dryRun,
    generatedAt: result.generatedAt,
    report: serializeMemoryGraphQualityReport(result.report),
    proposals: result.proposals,
    applied: result.applied,
  }
}

function serializeMemoryGraphRepairDecisionResult(
  result: MemoryGraphRepairDecisionResult,
): Record<string, unknown> {
  return {
    dryRun: result.dryRun,
    generatedAt: result.generatedAt,
    action: result.action,
    proposalId: result.proposalId,
    winnerMemoryId: result.winnerMemoryId,
    updated: result.updated,
    skipped: result.skipped,
    maintenance: result.maintenance,
  }
}

function clampToolLimit(value: unknown, fallback: number, max: number): number {
  const numeric = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numeric) || numeric <= 0) return fallback
  return Math.max(1, Math.min(Math.floor(numeric), max))
}

const ISO_DATE_PATTERN = /^(\d{4})-(\d{2})-(\d{2})$/

function parseDailyDate(input: unknown, now: Date = new Date()): Date | null {
  const today = startOfDay(now)
  if (input === undefined || input === null) {
    return today
  }
  if (typeof input !== 'string') {
    return null
  }
  const value = input.trim().toLowerCase()
  if (!value || value === 'today' || value === '오늘') {
    return today
  }
  if (value === 'yesterday' || value === '어제') {
    const yesterday = new Date(today)
    yesterday.setDate(yesterday.getDate() - 1)
    return yesterday
  }
  if (value === 'tomorrow' || value === '내일') {
    const tomorrow = new Date(today)
    tomorrow.setDate(tomorrow.getDate() + 1)
    return tomorrow
  }
  const match = value.match(ISO_DATE_PATTERN)
  if (!match) {
    return null
  }
  const [, year, month, day] = match
  const numericYear = Number(year)
  const numericMonth = Number(month)
  const numericDay = Number(day)
  const parsed = new Date(numericYear, numericMonth - 1, numericDay)
  if (
    Number.isNaN(parsed.valueOf())
    || parsed.getFullYear() !== numericYear
    || parsed.getMonth() !== numericMonth - 1
    || parsed.getDate() !== numericDay
  ) {
    return null
  }
  return startOfDay(parsed)
}

function startOfDay(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate())
}

function formatDateKey(date: Date): string {
  const year = String(date.getFullYear())
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}
