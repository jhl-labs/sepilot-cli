import type { AuditEvent, MemoryEntry, SemanticSearchOptions } from '@sepilotd/core'
import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify'
import '../fastify-types.js'
import {
  attachScopeTags,
  deriveExtensionMemoryScopeTags,
  deriveScopeTags,
  hasScopeTag,
  isMemoryArchived,
  isMemorySuperseded,
  isMemoryVisibleInScope,
  isMemoryWritableInScope,
  parseScopeFromTags,
  withLegacyGlobalMemoryRead,
} from '../../memory/scope.js'
import { createLogger } from '../../logger.js'
import { parseReminderTime } from '../../memory/reminders.js'
import { parseRelativePastTimestamp } from '../../memory/time-relative.js'
import {
  explicitlyRequestsAppIndex,
  filterAppIndexEntries,
  isAppIndexMemoryEntry,
} from '../../memory/internal-app-index.js'
import type {
  MemoryGraphQualityReport,
  MemoryGraphRepairDecisionResult,
  MemoryGraphRepairResult,
} from '../../memory/types.js'
import { resolveScopedFileMemoryReadView } from '../../memory/scoped-file-memory-read-view.js'
import type { RuntimeServices } from '../runtime.js'
import { zodRequestValidation } from './utils.js'
import {
  memoryCreateRequestSchema,
  memoryDeleteRequestSchema,
  memoryAuditQuerySchema,
  memoryDocumentIdParamsSchema,
  memoryDocumentIngestRequestSchema,
  memoryDocumentListQuerySchema,
  memoryDocumentSearchQuerySchema,
  memoryDeleteQuerySchema,
  memoryEntryIdParamsSchema,
  memoryHotQuerySchema,
  memoryGraphAuditQuerySchema,
  memoryGraphPageQuerySchema,
  memoryGraphRepairDecisionRequestSchema,
  memoryGraphRepairRequestSchema,
  memorySecurityAuditQuerySchema,
  memoryScopeTransferRequestSchema,
  memoryFileSectionParamsSchema,
  memoryFileSectionUpdateSchema,
  memoryLifecycleQuerySchema,
  memoryMaintenanceRequestSchema,
  memoryPinnedQuerySchema,
  memoryRecentQuerySchema,
  memorySearchQuerySchema,
  memoryUpdateRequestSchema,
  type MemoryAuditQuery,
  type MemoryCreateBody,
  type MemoryDeleteBody,
  type MemoryDeleteQuery,
  type MemoryDocumentIdParams,
  type MemoryDocumentIngestBody,
  type MemoryDocumentListQuery,
  type MemoryDocumentSearchQuery,
  type MemoryGraphAuditQuery,
  type MemoryGraphPageQuery,
  type MemoryGraphRepairBody,
  type MemoryGraphRepairDecisionBody,
  type MemoryHotQuery,
  type MemoryEntryIdParams,
  type MemorySecurityAuditQuery,
  type MemoryScopeTransferBody,
  type MemoryFileSectionParams,
  type MemoryFileSectionUpdateBody,
  type MemoryLifecycleQuery,
  type MemoryMaintenanceBody,
  type MemoryPinnedQuery,
  type MemoryRecentQuery,
  type MemorySearchQuery,
  type MemoryUpdateBody,
} from './memory-schema.js'

export { memoryOpenApiComponents, memoryOpenApiOverrides } from './memory-openapi.js'

const log = createLogger('memory:routes')

type MemoryScopeAuditRuntime = Pick<Partial<RuntimeServices>, 'auditLogger' | 'config'> | undefined

const MEMORY_SCOPE_BYPASS_DENIED_EVENT = 'memory.scope_bypass.denied'
const MEMORY_SCOPE_TAG_PREFIXES = [
  'scope:user:',
  'scope:channel:',
  'scope:session:',
  'scope:group:',
]

const EXTENSION_DENIED_MEMORY_ADMIN_PATHS = [
  '/api/v1/memory/scopes',
  '/api/v1/memory/health',
  '/api/v1/memory/maintenance',
  '/api/v1/memory/lifecycle',
  '/api/v1/memory/reindex',
  '/api/v1/memory/dreaming',
  '/api/v1/memory/export',
  '/api/v1/memory/import',
  '/api/v1/memory/graph/page',
  '/api/v1/memory/graph/audit',
  '/api/v1/memory/graph/repair',
] as const

type MemorySearchQueryBase = Pick<
  MemorySearchQuery,
  | 'limit'
  | 'type'
  | 'sources'
  | 'tags'
  | 'tagsLogic'
  | 'excludeTags'
  | 'minScore'
  | 'createdAfter'
  | 'createdBefore'
  | 'sortBy'
>

interface ResolvedMemorySearch<TOptions extends SemanticSearchOptions> {
  requestedLimit: number
  searchOptions: TOptions
}

/**
 * Pull caller scope out of HTTP headers / query params. Surfaces (web,
 * desktop, extension) should send their session-bound user/channel id
 * via the X-Memory-Scope-* headers; the routes then attach scope tags
 * to writes and filter reads to the caller's bucket. When no scope is
 * provided, the caller is treated as global (legacy behavior).
 */
function readRequestScope(request: FastifyRequest): string[] {
  if (request.authContext?.kind === 'extension') {
    return deriveExtensionMemoryScopeTags(request.authContext.tokenId)
  }
  const headerValue = (key: string): string | undefined => {
    const value = request.headers[key]
    if (typeof value === 'string') return value.trim() || undefined
    if (Array.isArray(value)) return value[0]?.trim() || undefined
    return undefined
  }
  const queryValue = (key: string): string | undefined => {
    const rawQuery = request.query
    const queryRecord = rawQuery as Record<string, unknown> | undefined
    const value = queryRecord?.[key]
    return typeof value === 'string' && value.trim().length > 0 ? value.trim() : undefined
  }

  const userId = headerValue('x-memory-scope-user-id') ?? queryValue('scopeUserId')
  const channelType = headerValue('x-memory-scope-channel-type') ?? queryValue('scopeChannelType')
  const chatId = headerValue('x-memory-scope-channel-id') ?? queryValue('scopeChannelId')
  const sessionId = headerValue('x-memory-scope-session-id') ?? queryValue('scopeSessionId')
  const rawGroups = headerValue('x-memory-scope-groups') ?? queryValue('scopeGroups')
  const groupIds = rawGroups
    ? rawGroups.split(',').map((entry) => entry.trim()).filter(Boolean)
    : undefined

  const scopeTags = deriveScopeTags({ userId, channelType, chatId, sessionId, groupIds })
  return withLegacyGlobalMemoryRead(scopeTags)
}

function isIncludeAllScopesRequested(value: unknown): boolean {
  return value === true || value === 'true' || value === '1'
}

function isMemoryScopeTag(tag: string): boolean {
  const lower = tag.toLowerCase()
  return MEMORY_SCOPE_TAG_PREFIXES.some((prefix) => lower.startsWith(prefix))
}

async function recordDeniedScopeBypass(
  runtime: MemoryScopeAuditRuntime,
  request: FastifyRequest,
  scopeTags: string[],
): Promise<void> {
  if (!runtime?.auditLogger) return
  const auth = request.authContext
  try {
    await runtime.auditLogger.log({
      timestamp: new Date().toISOString(),
      event: MEMORY_SCOPE_BYPASS_DENIED_EVENT,
      device: runtime.config?.device?.name ?? 'unknown-device',
      route: request.routeOptions.url ?? request.url.split('?', 1)[0],
      method: request.method,
      actor: auth?.kind === 'extension'
        ? `api:extension:${auth.tokenId}`
        : `api:${auth?.kind ?? 'anonymous'}`,
      authKind: auth?.kind ?? 'anonymous',
      scopeTags,
      requested: 'includeAllScopes',
      reason: 'extension_scope_bypass_denied',
      ...(auth?.kind === 'extension'
        ? {
            tokenId: auth.tokenId,
            label: auth.label,
            tokenScopes: auth.scopes,
          }
        : {}),
    } satisfies AuditEvent)
  } catch (error) {
    log.warn('Failed to record denied memory scope bypass', {
      error: error instanceof Error ? error.message : String(error),
    })
  }
}

async function resolveIncludeAllScopes(
  runtime: MemoryScopeAuditRuntime,
  request: FastifyRequest,
  reply: FastifyReply,
  value: unknown,
  scopeTags: string[],
): Promise<boolean | null> {
  if (!isIncludeAllScopesRequested(value)) return false
  if (request.authContext?.kind === 'extension') {
    await recordDeniedScopeBypass(runtime, request, scopeTags)
    reply.status(403).send({
      error: {
        code: 'SCOPE_BYPASS_DENIED',
        message: 'includeAllScopes requires the daemon master token or an unscoped local operator context.',
      },
    })
    return null
  }
  return true
}

function applyRelativeTimestampFilter(
  searchOptions: SemanticSearchOptions,
  field: 'createdAfter' | 'createdBefore',
  value: string | undefined,
  reply: FastifyReply,
): boolean {
  if (!value) return true
  const parsed = parseRelativePastTimestamp(value)
  if (!parsed) {
    reply.status(400).send({
      error: { code: 'INVALID_INPUT', message: `${field} "${value}" is not a recognised timestamp (ISO-8601, "yesterday", "5d ago", "지난주").` },
    })
    return false
  }
  searchOptions[field] = parsed.toISOString()
  return true
}

function resolveMemorySearchOptions<TOptions extends SemanticSearchOptions>(
  query: MemorySearchQueryBase,
  reply: FastifyReply,
  scopeTags: string[],
  includeAllScopes: boolean,
  options: TOptions,
): ResolvedMemorySearch<TOptions> | null {
  const requestedLimit = query.limit ?? 10
  const fetchLimit = scopeTags.length > 0 && !includeAllScopes
    ? Math.min(requestedLimit * 4, 100)
    : requestedLimit

  if (!includeAllScopes && scopeTags.length) options.scopeTags = scopeTags
  options.limit = fetchLimit
  if (query.type) options.type = query.type
  if (query.sources && query.sources.length > 0) options.sources = query.sources
  if (query.tags && query.tags.length > 0) options.tags = query.tags
  if (query.tagsLogic) options.tagsLogic = query.tagsLogic
  if (query.excludeTags && query.excludeTags.length > 0) options.excludeTags = query.excludeTags
  if (typeof query.minScore === 'number') options.minScore = query.minScore
  if (!applyRelativeTimestampFilter(options, 'createdAfter', query.createdAfter, reply)) return null
  if (!applyRelativeTimestampFilter(options, 'createdBefore', query.createdBefore, reply)) return null
  if (query.sortBy) options.sortBy = query.sortBy

  return { requestedLimit, searchOptions: options }
}


function matchesMemoryTagFilters(
  entryTags: string[] | undefined,
  query: Pick<MemoryRecentQuery, 'tags' | 'tagsLogic' | 'excludeTags'>,
): boolean {
  const tagSet = new Set((entryTags ?? []).map((tag) => tag.toLowerCase()))
  const requiredTags = (query.tags ?? []).map((tag) => tag.toLowerCase())
  const excludedTags = (query.excludeTags ?? []).map((tag) => tag.toLowerCase())

  if (excludedTags.some((tag) => tagSet.has(tag))) return false
  if (requiredTags.length === 0) return true
  if (query.tagsLogic === 'or') return requiredTags.some((tag) => tagSet.has(tag))
  return requiredTags.every((tag) => tagSet.has(tag))
}

// Fetch, then apply post-LIMIT JS filters (scope/superseded/archived/tags),
// growing the fetch window until at least `requestedLimit` in-scope rows are
// collected or the underlying source is exhausted. A fixed over-fetch factor
// silently under-returns whenever many candidates are filtered out.
async function collectWithGrowingWindow<T>(options: {
  requestedLimit: number
  initialFetch: number
  maxFetch: number
  fetch: (limit: number) => Promise<T[]>
  filter: (rows: T[]) => T[]
}): Promise<T[]> {
  let fetchLimit = Math.max(options.initialFetch, options.requestedLimit)
  let lastRawCount = -1
  // Bounded by maxFetch; each pass at most doubles the window.
  for (;;) {
    const raw = await options.fetch(fetchLimit)
    const filtered = options.filter(raw)
    if (
      filtered.length >= options.requestedLimit
      || raw.length < fetchLimit // source exhausted
      || raw.length === lastRawCount // widening produced no new rows
      || fetchLimit >= options.maxFetch
    ) {
      return filtered.slice(0, options.requestedLimit)
    }
    lastRawCount = raw.length
    fetchLimit = Math.min(fetchLimit * 2, options.maxFetch)
  }
}

function includeInternalAppIndex(
  query: Pick<MemoryRecentQuery | MemorySearchQuery | MemoryDocumentSearchQuery, 'tags'> & {
    documentId?: string
  },
): boolean {
  return explicitlyRequestsAppIndex({
    tags: query.tags,
    documentId: query.documentId,
  })
}

function isInternalAppIndexDocument(document: { id: string; path?: string; tags: string[] }): boolean {
  return isAppIndexMemoryEntry({
    source: 'document',
    id: document.id,
    tags: document.tags,
    path: document.path,
  })
}

function toMemoryAuditSnapshot(entry: MemoryEntry): Omit<MemoryEntry, 'score'> {
  return {
    evidence: entry.evidence,
    id: entry.id,
    content: entry.content,
    source: entry.source,
    tags: entry.tags,
  }
}

async function deleteMemoryEntryById(
  runtime: RuntimeServices,
  request: FastifyRequest,
  reply: FastifyReply,
  id: string,
  reason: string | undefined,
  includeAllScopesRequested: unknown,
): Promise<unknown> {
  const scopeTags = readRequestScope(request)
  const includeAllScopes = await resolveIncludeAllScopes(
    runtime,
    request,
    reply,
    includeAllScopesRequested,
    scopeTags,
  )
  if (includeAllScopes === null) return

  const existing = await runtime.semanticIndex.get(id)
  if (!existing) {
    return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Memory not found' } })
  }
  if (!includeAllScopes && scopeTags.length > 0 && !isMemoryWritableInScope(existing.tags, scopeTags)) {
    return reply.status(403).send({
      error: {
        code: 'SCOPE_MISMATCH',
        message: 'Memory belongs to a different user/channel scope.',
      },
    })
  }
  await runtime.semanticIndex.delete(existing.id)
  await runtime.semanticIndex.recordAudit({
    memoryId: existing.id,
    action: 'deleted',
    actor: 'api',
    reason,
    before: toMemoryAuditSnapshot(existing),
  })
  return reply.status(204).send()
}

export async function memoryRoutes(app: FastifyInstance) {
  const runtime = app.runtime

  // Extension tokens receive a private server-derived memory bucket. Global
  // inventory, migration, repair, and bulk transfer endpoints remain master
  // operator capabilities because they reveal or mutate other principals.
  app.addHook('preHandler', async (request, reply) => {
    if (request.authContext?.kind !== 'extension') return
    // Use Fastify's canonical matched route, not the raw URL. Fastify accepts
    // encoded segments such as `%73copes` for the static `scopes` route while
    // request.url preserves the attacker-controlled encoded spelling.
    const path = request.routeOptions.url ?? ''
    const denied = EXTENSION_DENIED_MEMORY_ADMIN_PATHS.some(
      (prefix) => path === prefix || path.startsWith(`${prefix}/`),
    )
    if (!denied) return
    return reply.status(403).send({
      error: {
        code: 'MEMORY_ADMIN_DENIED',
        message: 'This memory administration endpoint requires the daemon master token.',
      },
    })
  })

  app.get('/memory/status', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    return { data: runtime.semanticIndex.getStatus() }
  })

  app.get<{ Querystring: MemoryLifecycleQuery }>('/memory/lifecycle', {
    preValidation: zodRequestValidation({
      query: {
        schema: memoryLifecycleQuerySchema,
        message: 'Invalid memory lifecycle query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const query = request.query
    return {
      data: await runtime.semanticIndex.getLifecycleStatus({
        staleAfterDays: query.staleAfterDays,
        lowImportance: query.lowImportance,
      }),
    }
  })

  app.get<{ Querystring: MemoryAuditQuery }>('/memory/audit', {
    preValidation: zodRequestValidation({
      query: {
        schema: memoryAuditQuerySchema,
        message: 'Invalid memory audit query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const query = request.query
    const scopeTags = readRequestScope(request)
    const includeAllScopes = await resolveIncludeAllScopes(runtime, request, reply, query.includeAllScopes, scopeTags)
    if (includeAllScopes === null) return

    // Normalise the action filter into a Set so the post-filter is O(1).
    const actionFilter = (() => {
      if (!query.action) return null
      if (Array.isArray(query.action)) return new Set(query.action)
      if (typeof query.action === 'string') {
        return new Set(query.action.split(',').map((s) => s.trim()).filter(Boolean))
      }
      return null
    })()

    const sinceMs = query.since ? Date.parse(query.since) : NaN
    const untilMs = query.until ? Date.parse(query.until) : NaN
    if (query.since && Number.isNaN(sinceMs)) {
      return reply.status(400).send({
        error: { code: 'INVALID_INPUT', message: `since "${query.since}" is not a valid timestamp.` },
      })
    }
    if (query.until && Number.isNaN(untilMs)) {
      return reply.status(400).send({
        error: { code: 'INVALID_INPUT', message: `until "${query.until}" is not a valid timestamp.` },
      })
    }

    // Over-fetch a bit when filters will trim, so we don't return short
    // pages just because most matches landed outside the post-filter.
    const requestedLimit = query.limit ?? 50
    const willPostFilter = Boolean(actionFilter || query.actor || query.since || query.until)
      || (scopeTags.length > 0 && !includeAllScopes)
    const fetchLimit = willPostFilter
      ? Math.min(requestedLimit * 4, 500)
      : requestedLimit

    const auditEntries = await runtime.semanticIndex.listAudit({
      memoryId: query.memoryId,
      limit: fetchLimit,
    })

    let visible = auditEntries
    if (actionFilter) {
      visible = visible.filter((entry) => actionFilter.has(entry.action))
    }
    if (query.actor) {
      visible = visible.filter((entry) => entry.actor === query.actor)
    }
    if (Number.isFinite(sinceMs)) {
      visible = visible.filter((entry) => Date.parse(entry.createdAt) >= sinceMs)
    }
    if (Number.isFinite(untilMs)) {
      visible = visible.filter((entry) => Date.parse(entry.createdAt) <= untilMs)
    }
    if (scopeTags.length > 0 && !includeAllScopes) {
      visible = visible.filter((entry) => {
        const ownerTags = entry.after?.tags ?? entry.before?.tags
        return isMemoryVisibleInScope(ownerTags, scopeTags)
      })
    }

    return { data: visible.slice(0, requestedLimit) }
  })

  app.get<{ Querystring: MemorySecurityAuditQuery }>('/memory/security-audit', {
    preValidation: zodRequestValidation({
      query: {
        schema: memorySecurityAuditQuerySchema,
        message: 'Invalid memory security audit query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    if (request.authContext?.kind === 'extension') {
      return reply.status(403).send({
        error: {
          code: 'SCOPE_AUDIT_DENIED',
          message: 'Memory security audit history requires the daemon master token or an unscoped local operator context.',
        },
      })
    }
    const query = request.query
    const sinceMs = query.since ? Date.parse(query.since) : NaN
    if (query.since && Number.isNaN(sinceMs)) {
      return reply.status(400).send({
        error: { code: 'INVALID_INPUT', message: `since "${query.since}" is not a valid timestamp.` },
      })
    }

    const limit = query.limit ?? 100
    const auditEvents = runtime.auditLogger
      ? await runtime.auditLogger.query({
          event: MEMORY_SCOPE_BYPASS_DENIED_EVENT,
          since: query.since,
          limit: Math.min(limit * 4, 1000),
        })
      : []
    const filtered = auditEvents
      .filter((event) => !query.actor || event.actor === query.actor)
      .filter((event) => !query.authKind || event.authKind === query.authKind)
      .filter((event) => !query.route || event.route === query.route)
      .sort((left, right) => right.timestamp.localeCompare(left.timestamp))
      .slice(0, limit)

    return {
      data: filtered,
      meta: {
        limit,
        returned: filtered.length,
      },
    }
  })

  app.post<{ Body: MemoryMaintenanceBody }>('/memory/maintenance', {
    preValidation: zodRequestValidation({
      body: {
        schema: memoryMaintenanceRequestSchema,
        message: 'Invalid memory maintenance request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const body = request.body
    return {
      data: await runtime.semanticIndex.runMaintenance({
        staleAfterDays: body.maxAgeDays,
        lowImportance: body.maxImportance,
        dryRun: body.dryRun,
        actor: 'api',
        reason: body.reason,
      }),
    }
  })

  // Resolve the FileMemory bucket the caller's scope writes to. Falls back
  // to the legacy global bucket when no scope is provided.
  const resolveScopedFileMemory = (request: FastifyRequest) => {
    const scopeTags = readRequestScope(request)
    if (scopeTags.length > 0 && runtime?.fileMemoryRegistry) {
      return runtime.fileMemoryRegistry.get(scopeTags)
    }
    return runtime?.fileMemory
  }

  app.get('/memory/file', async (request, reply) => {
    const scopeTags = readRequestScope(request)
    const fileMemory = resolveScopedFileMemoryReadView(
      runtime?.fileMemory,
      runtime?.fileMemoryRegistry,
      scopeTags,
    )
    if (!fileMemory) {
      return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'File memory is not initialized' } })
    }
    const now = new Date()
    const yesterday = new Date(now)
    yesterday.setDate(yesterday.getDate() - 1)
    const [longTermMemory, todayNote, yesterdayNote, sections] = await Promise.all([
      fileMemory.readMemory(),
      fileMemory.readDailyNote(now),
      fileMemory.readDailyNote(yesterday),
      fileMemory.readMemorySections(),
    ])
    return {
      data: {
        memoryPath: fileMemory.getMemoryPath(),
        todayNotePath: fileMemory.getDailyNotePath(now),
        yesterdayNotePath: fileMemory.getDailyNotePath(yesterday),
        longTermMemory: longTermMemory ?? '',
        todayNote: todayNote ?? '',
        yesterdayNote: yesterdayNote ?? '',
        sections,
      },
    }
  })

  app.put<{ Params: MemoryFileSectionParams; Body: MemoryFileSectionUpdateBody }>('/memory/file/sections/:sectionTitle', {
    preValidation: zodRequestValidation({
      params: {
        schema: memoryFileSectionParamsSchema,
        message: 'Invalid memory file section',
      },
      body: {
        schema: memoryFileSectionUpdateSchema,
        message: 'Invalid memory file section update body',
      },
    }),
  }, async (request, reply) => {
    const fileMemory = resolveScopedFileMemory(request)
    if (!fileMemory) {
      return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'File memory is not initialized' } })
    }
    const { sectionTitle } = request.params
    const { content } = request.body
    const title = sectionTitle.trim()
    if (content.trim()) {
      await fileMemory.replaceMemorySection(title, content)
    } else {
      await fileMemory.deleteMemorySection(title)
    }
    const section = await fileMemory.readMemorySection(title)
    return {
      data: {
        title,
        content: section ?? '',
        deleted: !section,
      },
    }
  })

  app.delete<{ Params: MemoryFileSectionParams }>('/memory/file/sections/:sectionTitle', {
    preValidation: zodRequestValidation({
      params: {
        schema: memoryFileSectionParamsSchema,
        message: 'Invalid memory file section',
      },
    }),
  }, async (request, reply) => {
    const fileMemory = resolveScopedFileMemory(request)
    if (!fileMemory) {
      return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'File memory is not initialized' } })
    }
    const { sectionTitle } = request.params
    const deleted = await fileMemory.deleteMemorySection(sectionTitle.trim())
    return { data: { deleted } }
  })

  app.get<{ Querystring: MemoryRecentQuery }>('/memory/recent', {
    preValidation: zodRequestValidation({
      query: {
        schema: memoryRecentQuerySchema,
        message: 'Invalid memory recent query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const query = request.query
    const scopeTags = readRequestScope(request)
    const includeAllScopes = await resolveIncludeAllScopes(runtime, request, reply, query.includeAllScopes, scopeTags)
    if (includeAllScopes === null) return
    const includeSuperseded = query.includeSuperseded === true
    const includeArchived = query.includeArchived === true
    const requestedLimit = query.limit ?? 50

    const timeFilters: SemanticSearchOptions = {}
    if (!applyRelativeTimestampFilter(timeFilters, 'createdAfter', query.createdAfter, reply)) return
    if (!applyRelativeTimestampFilter(timeFilters, 'createdBefore', query.createdBefore, reply)) return

    const willPostFilter =
      Boolean(query.sources?.length || query.tags?.length || query.excludeTags?.length)
      || !includeSuperseded
      || !includeArchived
      || !includeInternalAppIndex(query)
      || (scopeTags.length > 0 && !includeAllScopes)

    const applyRecentFilters = <T extends { source: MemoryEntry['source']; tags: string[] }>(rows: T[]): T[] => {
      let recent = rows
      if (query.sources && query.sources.length > 0) {
        const sources = new Set(query.sources)
        recent = recent.filter((entry) => sources.has(entry.source))
      }
      if (!includeSuperseded) {
        recent = recent.filter((entry) => !isMemorySuperseded(entry.tags))
      }
      if (!includeArchived) {
        recent = recent.filter((entry) => !isMemoryArchived(entry.tags))
      }
      recent = recent.filter((entry) => matchesMemoryTagFilters(entry.tags, query))
      recent = filterAppIndexEntries(recent, includeInternalAppIndex(query))
      if (scopeTags.length > 0 && !includeAllScopes) {
        recent = recent.filter((entry) => isMemoryVisibleInScope(entry.tags, scopeTags))
      }
      return recent
    }

    const fetchRecent = (limit: number) =>
      runtime.semanticIndex.listRecentWithTimestamps(limit, {
        createdAfter: timeFilters.createdAfter,
        createdBefore: timeFilters.createdBefore,
      })

    let recent: Awaited<ReturnType<typeof fetchRecent>>
    if (query.sortBy === 'updatedAt') {
      // A chronological re-sort needs a broad candidate pool, so keep the fixed
      // large over-fetch and sort the full filtered set before trimming.
      const fetchLimit = Math.min(Math.max(requestedLimit * 4, 200), 5000)
      recent = applyRecentFilters(await fetchRecent(fetchLimit))
      recent = [...recent].sort((left, right) =>
        (Date.parse(right.updatedAt) || 0) - (Date.parse(left.updatedAt) || 0),
      )
    } else if (willPostFilter) {
      // Grow the fetch window until enough in-scope rows are collected instead
      // of under-returning after a fixed over-fetch.
      recent = await collectWithGrowingWindow({
        requestedLimit,
        initialFetch: Math.min(Math.max(requestedLimit * 4, 200), 5000),
        maxFetch: 5000,
        fetch: fetchRecent,
        filter: applyRecentFilters,
      })
    } else {
      recent = await fetchRecent(requestedLimit)
    }

    return { data: recent.slice(0, requestedLimit) }
  })

  app.get<{ Querystring: MemorySearchQuery }>('/memory/search', {
    preValidation: zodRequestValidation({
      query: {
        schema: memorySearchQuerySchema,
        message: 'Invalid memory search query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const query = request.query
    const scopeTags = readRequestScope(request)
    const includeAllScopes = await resolveIncludeAllScopes(runtime, request, reply, query.includeAllScopes, scopeTags)
    if (includeAllScopes === null) return
    const includeSuperseded = query.includeSuperseded === true
    const includeArchived = query.includeArchived === true
    const resolvedSearch = resolveMemorySearchOptions<SemanticSearchOptions>(
      query,
      reply,
      scopeTags,
      includeAllScopes,
      {},
    )
    if (!resolvedSearch) return
    const { requestedLimit, searchOptions } = resolvedSearch
    searchOptions.scopeTags = includeAllScopes || scopeTags.length === 0 ? undefined : scopeTags
    searchOptions.asOf = query.asOf
    searchOptions.includeInactive = query.includeInactive
    const willPostFilterSearch =
      !includeSuperseded
      || !includeArchived
      || !includeInternalAppIndex(query)
      || (scopeTags.length > 0 && !includeAllScopes)

    const applySearchFilters = (entries: MemoryEntry[]): MemoryEntry[] => {
      let filtered = entries
      if (!includeSuperseded) {
        filtered = filtered.filter((entry) => !isMemorySuperseded(entry.tags))
      }
      if (!includeArchived) {
        filtered = filtered.filter((entry) => !isMemoryArchived(entry.tags))
      }
      filtered = filterAppIndexEntries(filtered, includeInternalAppIndex(query))
      if (scopeTags.length > 0 && !includeAllScopes) {
        filtered = filtered.filter((entry) => isMemoryVisibleInScope(entry.tags, scopeTags))
      }
      return filtered
    }

    const data = willPostFilterSearch
      ? await collectWithGrowingWindow<MemoryEntry>({
          requestedLimit,
          initialFetch: Math.min(Math.max(requestedLimit * 4, requestedLimit), 500),
          maxFetch: 500,
          fetch: (limit) =>
            runtime.semanticIndex.search(query.query, { ...searchOptions, limit }),
          filter: applySearchFilters,
        })
      : (await runtime.semanticIndex.search(query.query, searchOptions)).slice(0, requestedLimit)

    return { data }
  })

  app.get<{ Querystring: MemoryGraphPageQuery }>('/memory/graph/page', {
    preValidation: zodRequestValidation({
      query: {
        schema: memoryGraphPageQuerySchema,
        message: 'Invalid memory graph page query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const query = request.query
    const scopeTags = readRequestScope(request)
    if (scopeTags.length > 0) {
      return reply.status(403).send({
        error: {
          code: 'MEMORY_GRAPH_SCOPE_UNSUPPORTED',
          message: 'Memory graph pages require an unscoped administrator until graph projection is scope-aware.',
        },
      })
    }
    const page = await runtime.semanticIndex.getGraphWikiPage({
      query: query.query,
      nodeId: query.id,
      limit: query.limit,
      evidenceLimit: query.evidenceLimit,
    })
    return { data: page }
  })

  app.get<{ Querystring: MemoryGraphAuditQuery }>('/memory/graph/audit', {
    preValidation: zodRequestValidation({
      query: {
        schema: memoryGraphAuditQuerySchema,
        message: 'Invalid memory graph audit query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const query = request.query
    const scopeTags = readRequestScope(request)
    const includeAllScopes = await resolveIncludeAllScopes(runtime, request, reply, query.includeAllScopes, scopeTags)
    if (includeAllScopes === null) return
    if (scopeTags.length > 0 && !includeAllScopes) {
      return reply.status(403).send({
        error: {
          code: 'SCOPE_BYPASS_REQUIRED',
          message: 'Graph audit is global; includeAllScopes requires the daemon master token or an unscoped local operator context.',
        },
      })
    }

    const report: MemoryGraphQualityReport = await runtime.semanticIndex.inspectGraphQuality({
      limit: query.limit,
      signalLimit: query.signalLimit,
      lowConfidenceThreshold: query.lowConfidenceThreshold,
      staleAfterDays: query.staleAfterDays,
    })
    return { data: report }
  })

  app.post<{ Body: MemoryGraphRepairBody }>('/memory/graph/repair', {
    preValidation: zodRequestValidation({
      body: {
        schema: memoryGraphRepairRequestSchema,
        message: 'Invalid memory graph repair request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const body = request.body
    const scopeTags = readRequestScope(request)
    const includeAllScopes = await resolveIncludeAllScopes(runtime, request, reply, body.includeAllScopes, scopeTags)
    if (includeAllScopes === null) return
    if (scopeTags.length > 0 && !includeAllScopes) {
      return reply.status(403).send({
        error: {
          code: 'SCOPE_BYPASS_REQUIRED',
          message: 'Graph repair is global; includeAllScopes requires the daemon master token or an unscoped local operator context.',
        },
      })
    }

    const result: MemoryGraphRepairResult = await runtime.semanticIndex.repairGraphQuality({
      dryRun: body.dryRun,
      limit: body.limit,
      signalLimit: body.signalLimit,
      lowConfidenceThreshold: body.lowConfidenceThreshold,
      staleAfterDays: body.staleAfterDays,
    })
    if (!result.dryRun && (result.applied.prunedNodes > 0 || result.applied.prunedEdges > 0)) {
      await runtime.semanticIndex.recordAudit({
        memoryId: '__memory_graph__',
        action: 'maintenance',
        actor: 'api',
        reason: body.reason ?? 'memory graph safe repair',
      })
    }
    return { data: result }
  })

  app.post<{ Body: MemoryGraphRepairDecisionBody }>('/memory/graph/repair/decision', {
    preValidation: zodRequestValidation({
      body: {
        schema: memoryGraphRepairDecisionRequestSchema,
        message: 'Invalid memory graph repair decision request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const body = request.body
    const scopeTags = readRequestScope(request)
    const includeAllScopes = await resolveIncludeAllScopes(runtime, request, reply, body.includeAllScopes, scopeTags)
    if (includeAllScopes === null) return

    if (scopeTags.length > 0 && !includeAllScopes) {
      const ids = [body.winnerMemoryId, ...body.supersededMemoryIds]
      for (const id of ids) {
        const entry = await runtime.semanticIndex.get(id)
        if (!entry) continue
        const visible = id === body.winnerMemoryId
          ? isMemoryVisibleInScope(entry.tags, scopeTags)
          : isMemoryWritableInScope(entry.tags, scopeTags)
        if (!visible) {
          return reply.status(403).send({
            error: {
              code: 'SCOPE_MISMATCH',
              message: 'Graph repair decision references a memory from a different user/channel scope.',
            },
          })
        }
      }
    }

    const result: MemoryGraphRepairDecisionResult = await runtime.semanticIndex.applyGraphRepairDecision({
      action: body.action,
      winnerMemoryId: body.winnerMemoryId,
      supersededMemoryIds: body.supersededMemoryIds,
      proposalId: body.proposalId,
      dryRun: body.dryRun,
      reason: body.reason,
      actor: 'api',
    })
    return { data: result }
  })

  app.get<{ Querystring: MemoryDocumentSearchQuery }>('/memory/documents/search', {
    preValidation: zodRequestValidation({
      query: {
        schema: memoryDocumentSearchQuerySchema,
        message: 'Invalid memory document search query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const query = request.query
    const scopeTags = readRequestScope(request)
    const includeAllScopes = await resolveIncludeAllScopes(runtime, request, reply, query.includeAllScopes, scopeTags)
    if (includeAllScopes === null) return
    const resolvedSearch = resolveMemorySearchOptions(
      query,
      reply,
      scopeTags,
      includeAllScopes,
      { documentId: query.documentId } as SemanticSearchOptions,
    )
    if (!resolvedSearch) return
    const { requestedLimit, searchOptions } = resolvedSearch
    if (!includeInternalAppIndex(query)) {
      searchOptions.limit = Math.min(Math.max((searchOptions.limit ?? requestedLimit) * 4, requestedLimit), 500)
    }

    const raw = await runtime.semanticIndex.searchDocuments(query.query, searchOptions)
    let filtered = filterAppIndexEntries(raw, includeInternalAppIndex(query))
    if (scopeTags.length > 0 && !includeAllScopes) {
      filtered = filtered.filter((chunk) => isMemoryVisibleInScope(chunk.tags, scopeTags))
    }
    return { data: filtered.slice(0, requestedLimit) }
  })

  app.get<{ Querystring: MemoryDocumentListQuery }>('/memory/documents', {
    preValidation: zodRequestValidation({
      query: {
        schema: memoryDocumentListQuerySchema,
        message: 'Invalid memory document list query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const query = request.query
    const scopeTags = readRequestScope(request)
    const includeAllScopes = await resolveIncludeAllScopes(runtime, request, reply, query.includeAllScopes, scopeTags)
    if (includeAllScopes === null) return
    const requestedLimit = query.limit ?? 50
    const fetchLimit = Math.min(requestedLimit * 4, 500)
    const documents = await runtime.semanticIndex.listDocuments({
      query: query.query,
      limit: fetchLimit,
    })
    let filtered = documents.filter((doc) => !isInternalAppIndexDocument(doc))
    if (scopeTags.length > 0 && !includeAllScopes) {
      filtered = filtered.filter((doc) => isMemoryVisibleInScope(doc.tags, scopeTags))
    }
    return { data: filtered.slice(0, requestedLimit) }
  })

  app.post<{ Body: MemoryCreateBody }>('/memory', {
    preValidation: zodRequestValidation({
      body: {
        schema: memoryCreateRequestSchema,
        message: 'Invalid memory create request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const body = request.body
    const { content, source, tags } = body
    const scopeTags = readRequestScope(request)
    const finalTags = attachScopeTags(tags ?? [], scopeTags)
    const { randomUUID } = await import('node:crypto')
    const id = randomUUID()
    await runtime.semanticIndex.add({
      id,
      content,
      source: source ?? 'user',
      evidence: body.evidence,
      tags: finalTags,
    })
    const created = await runtime.semanticIndex.get(id)
    await runtime.semanticIndex.recordAudit({
      memoryId: id,
      action: 'created',
      actor: 'api',
      reason: body.reason,
      after: created ? toMemoryAuditSnapshot(created) : {
        id,
        content,
        source: source ?? 'user',
        tags: finalTags,
      },
    })
    return { data: { id } }
  })

  app.get<{ Querystring: MemoryPinnedQuery }>('/memory/pinned', {
    preValidation: zodRequestValidation({
      query: {
        schema: memoryPinnedQuerySchema,
        message: 'Invalid memory pinned query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const query = request.query
    const scopeTags = readRequestScope(request)
    const includeAllScopes = await resolveIncludeAllScopes(runtime, request, reply, query.includeAllScopes, scopeTags)
    if (includeAllScopes === null) return
    const requestedLimit = query.limit ?? 50
    const fetchLimit = Math.min(requestedLimit * 4, 500)
    let pinned = await runtime.semanticIndex.listPinned(fetchLimit)
    pinned = pinned.filter((entry) => !isAppIndexMemoryEntry(entry))
    if (scopeTags.length > 0 && !includeAllScopes) {
      pinned = pinned.filter((entry) => isMemoryVisibleInScope(entry.tags, scopeTags))
    }
    return { data: pinned.slice(0, requestedLimit) }
  })

  app.post<{ Params: MemoryEntryIdParams }>('/memory/:id/pin', {
    preValidation: zodRequestValidation({
      params: {
        schema: memoryEntryIdParamsSchema,
        message: 'Invalid memory id',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const existing = await runtime.semanticIndex.get(params.id)
    if (!existing) {
      return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Memory not found' } })
    }
    const scopeTags = readRequestScope(request)
    if (scopeTags.length > 0 && !isMemoryWritableInScope(existing.tags, scopeTags)) {
      return reply.status(403).send({
        error: {
          code: 'SCOPE_MISMATCH',
          message: 'Memory belongs to a different user/channel scope.',
        },
      })
    }
    await runtime.semanticIndex.pin(existing.id)
    const after = await runtime.semanticIndex.get(existing.id)
    await runtime.semanticIndex.recordAudit({
      memoryId: existing.id,
      action: 'updated',
      actor: 'api',
      reason: 'Pinned from REST API',
      before: toMemoryAuditSnapshot(existing),
      after: after ? toMemoryAuditSnapshot(after) : toMemoryAuditSnapshot(existing),
    })
    return { data: { pinned: true } }
  })

  app.delete<{ Params: MemoryEntryIdParams }>('/memory/:id/pin', {
    preValidation: zodRequestValidation({
      params: {
        schema: memoryEntryIdParamsSchema,
        message: 'Invalid memory id',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const existing = await runtime.semanticIndex.get(params.id)
    if (!existing) {
      return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Memory not found' } })
    }
    const scopeTags = readRequestScope(request)
    if (scopeTags.length > 0 && !isMemoryWritableInScope(existing.tags, scopeTags)) {
      return reply.status(403).send({
        error: {
          code: 'SCOPE_MISMATCH',
          message: 'Memory belongs to a different user/channel scope.',
        },
      })
    }
    await runtime.semanticIndex.unpin(existing.id)
    const after = await runtime.semanticIndex.get(existing.id)
    await runtime.semanticIndex.recordAudit({
      memoryId: existing.id,
      action: 'updated',
      actor: 'api',
      reason: 'Unpinned from REST API',
      before: toMemoryAuditSnapshot(existing),
      after: after ? toMemoryAuditSnapshot(after) : toMemoryAuditSnapshot(existing),
    })
    return { data: { pinned: false } }
  })

  app.put<{ Params: MemoryEntryIdParams; Body: MemoryUpdateBody }>('/memory/:id', {
    preValidation: zodRequestValidation({
      params: {
        schema: memoryEntryIdParamsSchema,
        message: 'Invalid memory id',
      },
      body: {
        schema: memoryUpdateRequestSchema,
        message: 'Invalid memory update request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const body = request.body
    const existing = await runtime.semanticIndex.get(params.id)
    if (!existing) {
      return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Memory not found' } })
    }
    const scopeTags = readRequestScope(request)
    if (scopeTags.length > 0 && !isMemoryWritableInScope(existing.tags, scopeTags)) {
      return reply.status(403).send({
        error: {
          code: 'SCOPE_MISMATCH',
          message: 'Memory belongs to a different user/channel scope.',
        },
      })
    }
    // Preserve existing scope tags so callers can't strip ownership.
    const existingScopeTags = (existing.tags ?? []).filter((tag) =>
      tag.toLowerCase().startsWith('scope:'),
    )
    const baseTags = body.tags ?? existing.tags
    const finalTags = attachScopeTags(baseTags, existingScopeTags)
    await runtime.semanticIndex.add({
      id: existing.id,
      content: body.content,
      evidence: body.evidence,
      source: body.source ?? existing.source,
      tags: finalTags,
    })
    const updated = await runtime.semanticIndex.get(existing.id)
    await runtime.semanticIndex.recordAudit({
      memoryId: existing.id,
      action: 'updated',
      actor: 'api',
      reason: body.reason,
      before: toMemoryAuditSnapshot(existing),
      after: updated ? toMemoryAuditSnapshot(updated) : {
        ...toMemoryAuditSnapshot(existing),
        content: body.content,
        source: body.source ?? existing.source,
        tags: finalTags,
      },
    })
    return { data: updated ?? { ...existing, content: body.content } }
  })

  app.delete<{ Params: MemoryEntryIdParams; Querystring: MemoryDeleteQuery }>('/memory/:id', {
    preValidation: zodRequestValidation({
      params: {
        schema: memoryEntryIdParamsSchema,
        message: 'Invalid memory id',
      },
      query: {
        schema: memoryDeleteQuerySchema,
        message: 'Invalid memory delete query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const query = request.query
    return deleteMemoryEntryById(
      runtime,
      request,
      reply,
      params.id,
      query.reason,
      query.includeAllScopes,
    )
  })

  app.post<{ Body: MemoryDeleteBody }>('/memory/delete', {
    preValidation: zodRequestValidation({
      body: {
        schema: memoryDeleteRequestSchema,
        message: 'Invalid memory delete request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const body = request.body
    return deleteMemoryEntryById(
      runtime,
      request,
      reply,
      body.id,
      body.reason,
      body.includeAllScopes,
    )
  })

  app.post<{ Body: MemoryDocumentIngestBody }>('/memory/documents', {
    preValidation: zodRequestValidation({
      body: {
        schema: memoryDocumentIngestRequestSchema,
        message: 'Invalid memory document ingest request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const scopeTags = readRequestScope(request)
    const body = request.body
    const finalTags = attachScopeTags(body.tags ?? [], scopeTags)
    const document = await runtime.semanticIndex.ingestDocument({
      ...body,
      tags: finalTags.length > 0 ? finalTags : undefined,
    })
    return { data: document }
  })

  app.get<{ Params: MemoryDocumentIdParams; Querystring: { includeAllScopes?: string | boolean } }>('/memory/documents/:id', {
    preValidation: zodRequestValidation({
      params: {
        schema: memoryDocumentIdParamsSchema,
        message: 'Invalid memory document id',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const query = request.query
    const scopeTags = readRequestScope(request)
    const includeAllScopes = await resolveIncludeAllScopes(runtime, request, reply, query.includeAllScopes, scopeTags)
    if (includeAllScopes === null) return
    const document = await runtime.semanticIndex.getDocument(params.id)
    if (!document) {
      return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Document not found' } })
    }
    if (isInternalAppIndexDocument(document)) {
      return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Document not found' } })
    }
    if (scopeTags.length > 0 && !includeAllScopes && !isMemoryVisibleInScope(document.tags, scopeTags)) {
      return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Document not found' } })
    }
    return { data: document }
  })

  app.delete<{ Params: MemoryDocumentIdParams }>('/memory/documents/:id', {
    preValidation: zodRequestValidation({
      params: {
        schema: memoryDocumentIdParamsSchema,
        message: 'Invalid memory document id',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const scopeTags = readRequestScope(request)
    const document = await runtime.semanticIndex.getDocument(params.id)
    if (!document || isInternalAppIndexDocument(document)) {
      return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Document not found' } })
    }
    if (scopeTags.length > 0 && !isMemoryWritableInScope(document.tags, scopeTags)) {
      return reply.status(403).send({
        error: {
          code: 'SCOPE_MISMATCH',
          message: 'Document belongs to a different user/channel scope.',
        },
      })
    }
    const deleted = await runtime.semanticIndex.deleteDocument(params.id)
    if (!deleted) {
      return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Document not found' } })
    }
    return reply.status(204).send()
  })

  app.post('/memory/reindex', async (_request, reply) => {
    if (!runtime) {
      return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    }

    const status = runtime.semanticIndex.getStatus()
    if (status.status === 'disabled') {
      return reply.status(409).send({
        error: {
          code: 'INVALID_STATE',
          message: 'Semantic memory is disabled. Configure memory.embeddingProvider and memory.embeddingModel first.',
        },
      })
    }
    if (status.status === 'backfilling') {
      return reply.status(409).send({
        error: {
          code: 'INVALID_STATE',
          message: 'Semantic memory is already backfilling.',
        },
      })
    }

    try {
      const started = runtime.semanticIndex.startBackgroundReindex()
      if (!started) {
        return reply.status(409).send({
          error: {
            code: 'INVALID_STATE',
            message: 'Semantic reindex is already in progress.',
          },
        })
      }

      return reply.status(202).send({
        data: {
          started: true,
          status: runtime.semanticIndex.getStatus(),
        },
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      const statusCode = /unavailable/i.test(message) ? 503 : 409
      return reply.status(statusCode).send({
        error: {
          code: statusCode === 503 ? 'SERVICE_UNAVAILABLE' : 'INVALID_STATE',
          message,
        },
      })
    }
  })

  // ─── Audit live stream (SSE) ────────────────────────────────────
  app.get('/memory/audit/stream', async (request, reply) => {
    if (!runtime?.semanticIndex || typeof runtime.semanticIndex.subscribeAudit !== 'function') {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Audit stream is not available' },
      })
    }
    const scopeTags = readRequestScope(request)
    const rawQuery = request.query
    const queryRaw = (rawQuery as Record<string, unknown> | undefined) ?? {}
    const includeAllScopes = await resolveIncludeAllScopes(runtime, request, reply, queryRaw.includeAllScopes, scopeTags)
    if (includeAllScopes === null) return
    const actionsParam = typeof queryRaw.actions === 'string' ? queryRaw.actions : undefined
    const actorsParam = typeof queryRaw.actors === 'string' ? queryRaw.actors : undefined
    const memoryIdsParam = typeof queryRaw.memoryIds === 'string' ? queryRaw.memoryIds : undefined
    const actionsFilter = actionsParam
      ? new Set(actionsParam.split(',').map((s) => s.trim()).filter(Boolean))
      : null
    const actorsFilter = actorsParam
      ? new Set(actorsParam.split(',').map((s) => s.trim()).filter(Boolean))
      : null
    const memoryIdsFilter = memoryIdsParam
      ? new Set(memoryIdsParam.split(',').map((s) => s.trim()).filter(Boolean))
      : null

    reply.raw.setHeader('Content-Type', 'text/event-stream')
    reply.raw.setHeader('Cache-Control', 'no-cache')
    reply.raw.setHeader('Connection', 'keep-alive')
    reply.raw.flushHeaders?.()
    reply.raw.write(': memory audit stream\n\n')

    const send = (event: string, data: unknown) => {
      try {
        reply.raw.write(`event: ${event}\n`)
        reply.raw.write(`data: ${JSON.stringify(data)}\n\n`)
      } catch {
        // Connection torn down — clean up below.
      }
    }

    const unsubscribe = runtime.semanticIndex.subscribeAudit((entry) => {
        if (actionsFilter && !actionsFilter.has(entry.action)) return
        if (actorsFilter && !actorsFilter.has(entry.actor)) return
        if (memoryIdsFilter && !memoryIdsFilter.has(entry.memoryId)) return
        if (!includeAllScopes && scopeTags.length > 0) {
          const ownerTags = entry.after?.tags ?? entry.before?.tags
          if (!isMemoryVisibleInScope(ownerTags, scopeTags)) return
        }
        send('audit', {
          id: entry.id,
          memoryId: entry.memoryId,
          action: entry.action,
          actor: entry.actor,
          reason: entry.reason,
          createdAt: entry.createdAt,
        })
      })

    // Periodic keepalive comment so intermediaries / proxies don't drop us.
    const keepalive = setInterval(() => {
      try { reply.raw.write(': keepalive\n\n') } catch { /* ignore */ }
    }, 15_000)
    keepalive.unref?.()

    request.raw.on('close', () => {
      clearInterval(keepalive)
      unsubscribe()
      try { reply.raw.end() } catch { /* ignore */ }
    })
  })

  // ─── Scopes (admin) ─────────────────────────────────────────────
  // POST /api/v1/memory/scopes/:scope/transfer — admin migration.
  // Moves every semantic memory tagged with the source scope onto the
  // target scope (re-tagging via UPSERT so ids/embeddings stay), shifts
  // pending reminders' scope tags, and optionally renames the file
  // memory bucket directory.
  app.post<{
    Params: { scope: string }
    Body: MemoryScopeTransferBody
  }>('/memory/scopes/:scope/transfer', {
    preValidation: zodRequestValidation({
      body: {
        schema: memoryScopeTransferRequestSchema,
        message: 'Invalid memory scope transfer request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    const params = request.params
    const body = request.body
    const fromRaw = params.scope?.trim() ?? ''
    const targetRaw = (body?.target ?? '').trim()
    if (!fromRaw || !targetRaw) {
      return reply.status(400).send({
        error: { code: 'INVALID_INPUT', message: 'Both source scope and `target` are required.' },
      })
    }
    const normalizeScope = (raw: string, options?: { allowGlobal?: boolean }) => {
      const lower = raw.toLowerCase()
      if (options?.allowGlobal && (lower === 'global' || lower === 'scope:global')) {
        return 'global'
      }
      const tag = lower.startsWith('scope:') ? lower : `scope:${lower}`
      if (!isMemoryScopeTag(tag)) return null
      return tag
    }
    const fromTag = normalizeScope(fromRaw, { allowGlobal: true })
    const toTag = normalizeScope(targetRaw)
    if (!fromTag || !toTag) {
      return reply.status(400).send({
        error: { code: 'INVALID_INPUT', message: 'Source must be global or a user/channel/session/group tag; target must be a user/channel/session/group tag.' },
      })
    }
    if (fromTag === toTag) {
      return reply.status(400).send({
        error: { code: 'INVALID_INPUT', message: 'Source and target scopes must differ.' },
      })
    }

    const dryRun = body?.dryRun !== false  // default true to be safe
    const includeFile = body?.includeFile === true
    const includeReminders = body?.includeReminders !== false  // default true
    const fromGlobal = fromTag === 'global'
    const requestedIds = new Set((body?.ids ?? []).map((id) => id.trim()).filter(Boolean))
    const requestedLimit = typeof body?.limit === 'number' && Number.isFinite(body.limit)
      ? Math.min(Math.max(Math.floor(body.limit), 1), 10_000)
      : undefined

    if (!dryRun && fromGlobal && body?.confirmGlobal !== true) {
      return reply.status(400).send({
        error: {
          code: 'CONFIRMATION_REQUIRED',
          message: 'Migrating global memories requires confirmGlobal:true after reviewing a dry run.',
        },
      })
    }

    // Stage 1: enumerate semantic memories.
    const all = await runtime.semanticIndex.listRecent(10000)
    const matchedBeforeLimit = all.filter((entry) => {
      if (requestedIds.size > 0 && !requestedIds.has(entry.id)) return false
      if (fromGlobal) return !hasScopeTag(entry.tags)
      return entry.tags.some((tag) => tag.toLowerCase() === fromTag)
    })
    const matched = requestedLimit
      ? matchedBeforeLimit.slice(0, requestedLimit)
      : matchedBeforeLimit

    // Stage 2: count reminders.
    let pendingReminders: Array<{ id: string; scopeTags: string[]; dueAt: string; content: string; channelType?: string; chatId?: string }> = []
    if (includeReminders && runtime.remindersStore) {
      try {
        const list = await runtime.remindersStore.list()
        pendingReminders = list
          .filter((reminder) => !reminder.firedAt && !reminder.cancelledAt)
          .filter((reminder) => {
            if (fromGlobal) return !hasScopeTag(reminder.scopeTags)
            return reminder.scopeTags?.some((tag) => tag.toLowerCase() === fromTag)
          })
      } catch {
        // best-effort
      }
    }

    // Determine bucket keys for optional file rename.
    const fromBucketKey = (() => {
      if (fromGlobal) return undefined
      if (fromTag.startsWith('scope:user:')) return `user-${fromTag.slice('scope:user:'.length)}`
      if (fromTag.startsWith('scope:channel:')) return `channel-${fromTag.slice('scope:channel:'.length).replace(':', '-')}`
      return undefined
    })()
    const toBucketKey = (() => {
      if (toTag.startsWith('scope:user:')) return `user-${toTag.slice('scope:user:'.length)}`
      if (toTag.startsWith('scope:channel:')) return `channel-${toTag.slice('scope:channel:'.length).replace(':', '-')}`
      return undefined
    })()
    const canMoveFile = includeFile && fromBucketKey && toBucketKey

    if (dryRun) {
      return {
        data: {
          dryRun: true,
          globalSource: fromGlobal,
          fromScope: fromTag,
          toScope: toTag,
          matchedSemanticMemories: matchedBeforeLimit.length,
          wouldRetagSemanticMemories: matched.length,
          wouldUpdateReminders: pendingReminders.length,
          wouldMoveFileBucket: canMoveFile ? `${fromBucketKey} → ${toBucketKey}` : null,
          limited: matched.length < matchedBeforeLimit.length,
          requiresConfirmGlobal: fromGlobal,
          sampleSemanticIds: matched.slice(0, 50).map((entry) => entry.id),
          sampleReminderIds: pendingReminders.slice(0, 50).map((reminder) => reminder.id),
        },
      }
    }

    // Stage 1 apply: re-tag semantic memories via UPSERT.
    let retaggedSemantic = 0
    for (const entry of matched) {
      const nextTags = Array.from(new Set([
        ...(fromGlobal ? entry.tags : entry.tags.filter((tag) => tag.toLowerCase() !== fromTag)),
        toTag,
      ]))
      try {
        await runtime.semanticIndex.add({
          id: entry.id,
          content: entry.content,
          source: entry.source,
          tags: nextTags,
        })
        await runtime.semanticIndex.recordAudit({
          memoryId: entry.id,
          action: 'updated',
          actor: 'api',
          reason: body?.reason ?? `transfer ${fromTag} → ${toTag}`,
          before: {
            id: entry.id,
            content: entry.content,
            source: entry.source,
            tags: entry.tags,
          },
          after: {
            id: entry.id,
            content: entry.content,
            source: entry.source,
            tags: nextTags,
          },
        })
        retaggedSemantic += 1
      } catch {
        // continue
      }
    }

    // Stage 2 apply: re-issue reminders with the target scope (cancel + add).
    let retaggedReminders = 0
    if (includeReminders && runtime.remindersStore) {
      for (const reminder of pendingReminders) {
        try {
          const newScope = Array.from(new Set([
            ...(fromGlobal ? (reminder.scopeTags ?? []) : reminder.scopeTags.filter((tag) => tag.toLowerCase() !== fromTag)),
            toTag,
          ]))
          await runtime.remindersStore.cancel(reminder.id, `transfer ${fromTag} → ${toTag}`)
          await runtime.remindersStore.add({
            dueAt: reminder.dueAt,
            content: reminder.content,
            scopeTags: newScope,
            channelType: reminder.channelType,
            chatId: reminder.chatId,
          })
          retaggedReminders += 1
        } catch {
          // continue
        }
      }
    }

    // Stage 3: optional file bucket rename.
    let fileBucketMoved = false
    if (canMoveFile && runtime.fileMemoryRegistry?.renameScope) {
      try {
        fileBucketMoved = await runtime.fileMemoryRegistry.renameScope(fromBucketKey!, toBucketKey!)
      } catch {
        // best-effort
      }
    }

    return {
      data: {
        dryRun: false,
        globalSource: fromGlobal,
        fromScope: fromTag,
        toScope: toTag,
        matchedSemanticMemories: matchedBeforeLimit.length,
        retaggedSemanticMemories: retaggedSemantic,
        retaggedReminders,
        fileBucketMoved,
      },
    }
  })

  app.delete<{
    Params: { scope: string }
    Querystring: { confirm?: string | boolean; reason?: string; includeFile?: string | boolean }
  }>('/memory/scopes/:scope', async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    const deleteParams = request.params
    const deleteQuery = request.query
    const rawScope = deleteParams.scope?.trim() ?? ''
    if (!rawScope) {
      return reply.status(400).send({
        error: { code: 'INVALID_INPUT', message: 'Scope tag is required.' },
      })
    }
    // Accept either the full canonical tag (scope:user:42) or shorthand
    // forms (user:42 / channel:telegram:99). Normalize to the canonical
    // form for matching.
    const lower = rawScope.toLowerCase()
    const scopeTag = lower.startsWith('scope:') ? lower : `scope:${lower}`
    if (!isMemoryScopeTag(scopeTag)) {
      return reply.status(400).send({
        error: { code: 'INVALID_INPUT', message: `Scope ${rawScope} must target a user, channel, session, or group.` },
      })
    }

    const confirm = deleteQuery.confirm === true
      || deleteQuery.confirm === 'true'
      || deleteQuery.confirm === '1'

    // Phase 1: count + (if confirm) delete semantic memories.
    const allMemories = await runtime.semanticIndex.listRecent(10000)
    const matched = allMemories.filter((entry) =>
      entry.tags.some((tag) => tag.toLowerCase() === scopeTag),
    )

    // Phase 2: count reminders.
    let matchedReminders: Array<{ id: string }> = []
    if (runtime.remindersStore) {
      try {
        const all = await runtime.remindersStore.list()
        matchedReminders = all
          .filter((reminder) => !reminder.firedAt && !reminder.cancelledAt)
          .filter((reminder) =>
            reminder.scopeTags?.some((tag) => tag.toLowerCase() === scopeTag),
          )
      } catch {
        // best-effort; swallow
      }
    }

    const includeFile = deleteQuery.includeFile === true
      || deleteQuery.includeFile === 'true'
      || deleteQuery.includeFile === '1'

    // Determine the file-bucket key (user-42 / channel-telegram-99 / group-eng)
    // from the scope tag for the optional bucket cleanup.
    // Sanitize the trailing segment so caller-supplied scope tags can never
    // contain `/`, `\\`, or `..` that would escape the scopes directory in
    // `ScopedFileMemoryRegistryImpl.{deleteScope,renameScope}`.
    const sanitizeBucketSegment = (value: string): string =>
      value.toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '')
    const fileBucketKey = (() => {
      if (!includeFile || !runtime.fileMemoryRegistry) return undefined
      if (scopeTag.startsWith('scope:user:')) {
        const segment = sanitizeBucketSegment(scopeTag.slice('scope:user:'.length))
        return segment ? `user-${segment}` : undefined
      }
      if (scopeTag.startsWith('scope:channel:')) {
        const rest = scopeTag.slice('scope:channel:'.length).replace(':', '-')
        const segment = sanitizeBucketSegment(rest)
        return segment ? `channel-${segment}` : undefined
      }
      // groups don't have their own file bucket today.
      return undefined
    })()

    if (!confirm) {
      return {
        data: {
          dryRun: true,
          scope: scopeTag,
          wouldDeleteSemanticMemories: matched.length,
          wouldCancelReminders: matchedReminders.length,
          wouldDeleteFileBucket: fileBucketKey ?? null,
        },
      }
    }

    let deletedSemantic = 0
    for (const entry of matched) {
      try {
        await runtime.semanticIndex.delete(entry.id)
        await runtime.semanticIndex.recordAudit({
          memoryId: entry.id,
          action: 'deleted',
          actor: 'api',
          reason: deleteQuery.reason ?? `bulk-delete scope ${scopeTag}`,
          before: {
            id: entry.id,
            content: entry.content,
            source: entry.source,
            tags: entry.tags,
          },
        })
        deletedSemantic += 1
      } catch {
        // continue with the rest
      }
    }
    let cancelledReminders = 0
    if (runtime.remindersStore) {
      for (const reminder of matchedReminders) {
        try {
          const result = await runtime.remindersStore.cancel(reminder.id, `bulk-delete scope ${scopeTag}`)
          if (result) cancelledReminders += 1
        } catch {
          // continue
        }
      }
    }
    let fileBucketDeleted = false
    if (fileBucketKey && runtime.fileMemoryRegistry?.deleteScope) {
      try {
        fileBucketDeleted = await runtime.fileMemoryRegistry.deleteScope(fileBucketKey)
      } catch {
        // best-effort
      }
    }

    return {
      data: {
        dryRun: false,
        scope: scopeTag,
        deletedSemanticMemories: deletedSemantic,
        cancelledReminders,
        fileBucketDeleted,
      },
    }
  })

  app.get('/memory/scopes', async (_request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    // Collect file-memory bucket keys.
    const fileScopes = runtime.fileMemoryRegistry?.list().map((entry) => entry.key)
      ?? (runtime.fileMemory ? ['global'] : [])

    // Aggregate semantic memory scope tags.
    const counts: Record<string, number> = {}
    let untagged = 0
    try {
      const all = await runtime.semanticIndex.listRecent(5000)
      for (const entry of all) {
        const scopeTags = (entry.tags ?? []).filter((tag) =>
          isMemoryScopeTag(tag),
        )
        if (scopeTags.length === 0) {
          untagged += 1
          continue
        }
        for (const tag of scopeTags) {
          counts[tag] = (counts[tag] ?? 0) + 1
        }
      }
    } catch {
      // semantic store unavailable — return what we have from fileMemoryRegistry.
    }

    const semanticScopes = Object.entries(counts)
      .map(([scope, count]) => ({ scope, count }))
      .sort((a, b) => b.count - a.count)

    // Reminders pending per scope (best-effort).
    const reminderCounts: Record<string, number> = {}
    if (runtime.remindersStore) {
      try {
        const list = await runtime.remindersStore.list()
        for (const reminder of list) {
          if (reminder.firedAt || reminder.cancelledAt) continue
          for (const tag of reminder.scopeTags ?? []) {
            const lower = tag.toLowerCase()
            if (isMemoryScopeTag(lower)) {
              reminderCounts[tag] = (reminderCounts[tag] ?? 0) + 1
            }
          }
        }
      } catch { /* ignore */ }
    }

    return {
      data: {
        fileScopes,
        semanticScopes,
        untaggedSemanticEntries: untagged,
        pendingReminders: Object.entries(reminderCounts)
          .map(([scope, count]) => ({ scope, count }))
          .sort((a, b) => b.count - a.count),
      },
    }
  })

  // ─── Dreaming ───────────────────────────────────────────────────
  app.post('/memory/dreaming/run', async (_request, reply) => {
    if (!runtime?.dreaming) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Dreaming engine is not initialized' },
      })
    }
    if (typeof runtime.dreaming.consolidate !== 'function') {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Dreaming consolidate is not available' },
      })
    }
    try {
      const result = await runtime.dreaming.consolidate()
      if (result.skipped) {
        return reply.status(409).send({
          error: { code: 'INVALID_STATE', message: 'Dreaming consolidation already in progress.' },
        })
      }
      return { data: { result, lastRun: runtime.dreaming.getLastRun?.() } }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      return reply.status(500).send({
        error: { code: 'DREAMING_FAILED', message },
      })
    }
  })

  app.get('/memory/dreaming/last', async (_request, reply) => {
    if (!runtime?.dreaming) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Dreaming engine is not initialized' },
      })
    }
    const snapshot = runtime.dreaming.getLastRun?.()
    if (!snapshot) {
      return { data: null }
    }
    return { data: snapshot }
  })

  // ─── Access counter (hot memories + manual touch) ────────────────
  // GET /api/v1/memory/hot — top-N memories by access_count.
  // Scope-aware: visible memories only by default. archived/superseded
  // are filtered out. Use ?includeAllScopes=true for admin overview.
  app.get<{ Querystring: MemoryHotQuery }>(
    '/memory/hot',
    {
      preValidation: zodRequestValidation({
        query: { schema: memoryHotQuerySchema, message: 'Invalid memory hot query' },
      }),
    },
    async (request, reply) => {
      if (!runtime) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
        })
      }
      if (typeof runtime.semanticIndex.listHotMemories !== 'function') {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Access counter is not available on this index' },
        })
      }
      const scopeTags = readRequestScope(request)
      const query = request.query
      const includeAllScopes = await resolveIncludeAllScopes(runtime, request, reply, query.includeAllScopes, scopeTags)
      if (includeAllScopes === null) return
      const limit = Math.max(1, Math.min(query.limit ?? 20, 200))
      // Over-fetch to compensate for scope/lifecycle filters.
      const fetchLimit = (!includeAllScopes && scopeTags.length > 0) ? Math.min(limit * 4, 500) : limit
      const raw = await runtime.semanticIndex.listHotMemories(fetchLimit)
      const visible = raw
        .filter((entry) => !isMemoryArchived(entry.tags) && !isMemorySuperseded(entry.tags))
        .filter((entry) => includeAllScopes
          || scopeTags.length === 0
          || isMemoryVisibleInScope(entry.tags, scopeTags))
        .slice(0, limit)
      return {
        data: visible.map((entry) => ({
          id: entry.id,
          content: entry.content,
          source: entry.source,
          tags: entry.tags,
          accessCount: entry.accessCount,
          lastAccessedAt: entry.lastAccessedAt,
        })),
      }
    },
  )

  // POST /api/v1/memory/:id/access — manually bump access counter.
  // Used by surfaces (web/desktop) when a memory is shown to the user
  // outside a search call (e.g. opened from a saved-list shortcut).
  app.post<{ Params: MemoryEntryIdParams }>(
    '/memory/:id/access',
    {
      preValidation: zodRequestValidation({
        params: { schema: memoryEntryIdParamsSchema, message: 'Invalid memory id' },
      }),
    },
    async (request, reply) => {
      if (!runtime) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
        })
      }
      if (typeof runtime.semanticIndex.recordAccess !== 'function') {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Access counter is not available on this index' },
        })
      }
      const params = request.params
      const id = params.id
      const scopeTags = readRequestScope(request)
      // An access bump changes ranking/retention state, so visibility alone is
      // insufficient (public/shared memories may be readable but not owned).
      const existing = await runtime.semanticIndex.get(id)
      if (!existing) {
        return reply.status(404).send({
          error: { code: 'NOT_FOUND', message: `Memory "${id}" not found.` },
        })
      }
      if (scopeTags.length > 0 && !isMemoryWritableInScope(existing.tags, scopeTags)) {
        return reply.status(404).send({
          error: { code: 'NOT_FOUND', message: `Memory "${id}" not found.` },
        })
      }
      await runtime.semanticIndex.recordAccess([id])
      const stats = typeof runtime.semanticIndex.getAccessStats === 'function'
        ? await runtime.semanticIndex.getAccessStats(id)
        : null
      return { data: { id, accessCount: stats?.accessCount ?? null, lastAccessedAt: stats?.lastAccessedAt ?? null } }
    },
  )

  // GET /api/v1/memory/:id/access — peek at the counter without bumping.
  app.get<{ Params: MemoryEntryIdParams }>(
    '/memory/:id/access',
    {
      preValidation: zodRequestValidation({
        params: { schema: memoryEntryIdParamsSchema, message: 'Invalid memory id' },
      }),
    },
    async (request, reply) => {
      if (!runtime) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
        })
      }
      if (typeof runtime.semanticIndex.getAccessStats !== 'function') {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Access counter is not available on this index' },
        })
      }
      const params = request.params
      const id = params.id
      const scopeTags = readRequestScope(request)
      const existing = await runtime.semanticIndex.get(id)
      if (!existing) {
        return reply.status(404).send({
          error: { code: 'NOT_FOUND', message: `Memory "${id}" not found.` },
        })
      }
      if (scopeTags.length > 0 && !isMemoryVisibleInScope(existing.tags, scopeTags)) {
        return reply.status(404).send({
          error: { code: 'NOT_FOUND', message: `Memory "${id}" not found.` },
        })
      }
      const stats = await runtime.semanticIndex.getAccessStats(id)
      return { data: { id, accessCount: stats?.accessCount ?? 0, lastAccessedAt: stats?.lastAccessedAt ?? null } }
    },
  )

  // ─── Health ─────────────────────────────────────────────────────
  app.get('/memory/health', async (_request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    const semanticStatus = runtime.semanticIndex.getStatus()
    let lifecycle: Record<string, number | undefined> = {}
    try {
      const result = await runtime.semanticIndex.getLifecycleStatus({})
      lifecycle = {
        totalMemories: result.totalMemories,
        conversationMemories: result.conversationMemories,
        documentMemories: result.documentMemories,
        userMemories: result.userMemories,
        staleConversationMemories: result.staleConversationMemories,
        pruneCandidateMemories: result.pruneCandidateMemories,
        pendingEmbeddings: result.pendingEmbeddings,
        failedEmbeddings: result.failedEmbeddings,
      }
    } catch (error) {
      lifecycle = { errorMessage: undefined }
      void error
    }

    const fileMemoryScopes = runtime.fileMemoryRegistry?.list().map((entry) => entry.key)
      ?? (runtime.fileMemory ? ['global'] : [])

    let pendingReminders = 0
    if (runtime.remindersStore) {
      try {
        const all = await runtime.remindersStore.list()
        pendingReminders = all.filter((reminder) => !reminder.firedAt && !reminder.cancelledAt).length
      } catch {
        pendingReminders = -1
      }
    }

    return {
      data: {
        semantic: {
          status: semanticStatus.status,
          model: semanticStatus.indexedModel ?? semanticStatus.configuredModel,
          backend: semanticStatus.vectorBackend,
          backendAvailable: semanticStatus.backendAvailable,
          pending: semanticStatus.pendingCount,
          failed: semanticStatus.failedCount,
          lastError: semanticStatus.lastError,
        },
        lifecycle,
        fileMemory: {
          scopeCount: fileMemoryScopes.length,
          scopes: fileMemoryScopes.slice(0, 50),
        },
        reminders: {
          configured: Boolean(runtime.remindersStore),
          pending: pendingReminders,
        },
      },
    }
  })

  // ─── Reminders ───────────────────────────────────────────────────
  app.get('/memory/reminders', async (request, reply) => {
    if (!runtime?.remindersStore) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Reminders store is not initialized' },
      })
    }
    const scopeTags = readRequestScope(request)
    const list = scopeTags.length > 0
      ? await runtime.remindersStore.listForScope(scopeTags)
      : await runtime.remindersStore.list()
    return {
      data: list
        .filter((reminder) => !reminder.firedAt && !reminder.cancelledAt)
        .map((reminder) => ({
          id: reminder.id,
          dueAt: reminder.dueAt,
          content: reminder.content,
          channelType: reminder.channelType,
          chatId: reminder.chatId,
          createdAt: reminder.createdAt,
        })),
    }
  })

  app.post<{ Body: { when: string; content: string } }>('/memory/reminders', async (request, reply) => {
    if (!runtime?.remindersStore) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Reminders store is not initialized' },
      })
    }
    const body = request.body ?? ({} as { when?: unknown; content?: unknown })
    const when = typeof body.when === 'string' ? body.when : ''
    const content = typeof body.content === 'string' ? body.content.trim() : ''
    if (!when || !content) {
      return reply.status(400).send({
        error: { code: 'INVALID_INPUT', message: 'Both `when` and non-empty `content` are required.' },
      })
    }
    const dueAt = parseReminderTime(when)
    if (!dueAt) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_INPUT',
          message: `Could not parse "when" value "${when}". Use a relative duration, "tomorrow", or an ISO-8601 timestamp.`,
        },
      })
    }
    if (dueAt.getTime() <= Date.now()) {
      return reply.status(400).send({
        error: { code: 'INVALID_INPUT', message: `Reminder time ${dueAt.toISOString()} is in the past.` },
      })
    }
    const scopeTags = readRequestScope(request)
    const scope = parseScopeFromTags(scopeTags)
    const reminder = await runtime.remindersStore.add({
      dueAt: dueAt.toISOString(),
      content,
      scopeTags,
      channelType: scope.channelType,
      chatId: scope.chatId,
    })
    return {
      data: {
        id: reminder.id,
        dueAt: reminder.dueAt,
        content: reminder.content,
        delivery: scope.channelType && scope.chatId
          ? { channelType: scope.channelType, chatId: scope.chatId }
          : { warning: 'No channel scope detected — reminder is stored but cannot be auto-delivered.' },
      },
    }
  })

  // ─── Export / Import ────────────────────────────────────────────
  app.get<{ Querystring: { includeAllScopes?: string | boolean; includeFile?: string | boolean; includeSemantic?: string | boolean; includeReminders?: string | boolean; createdAfter?: string; createdBefore?: string } }>(
    '/memory/export',
    async (request, reply) => {
      if (!runtime) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
        })
      }
      const query = request.query ?? {}
      const includeFile = query.includeFile !== false && query.includeFile !== 'false'
      const includeSemantic = query.includeSemantic !== false && query.includeSemantic !== 'false'
      const includeReminders = query.includeReminders !== false && query.includeReminders !== 'false'
      const scopeTags = readRequestScope(request)
      const includeAllScopes = await resolveIncludeAllScopes(runtime, request, reply, query.includeAllScopes, scopeTags)
      if (includeAllScopes === null) return

      const payload: Record<string, unknown> = {
        version: '1.0',
        exportedAt: new Date().toISOString(),
        scope: parseScopeFromTags(scopeTags),
      }

      if (includeFile) {
        const fileMemory = (scopeTags.length > 0 && runtime.fileMemoryRegistry)
          ? runtime.fileMemoryRegistry.get(scopeTags)
          : runtime.fileMemory
        if (fileMemory) {
          const sections = await fileMemory.readMemorySections()
          payload.fileMemory = {
            sections: sections.map((s) => ({ title: s.title, content: s.content })),
          }
        }
      }

      if (includeSemantic) {
        let createdAfterIso: string | undefined
        let createdBeforeIso: string | undefined
        if (query.createdAfter) {
          const parsed = new Date(query.createdAfter)
          if (Number.isNaN(parsed.valueOf())) {
            return reply.status(400).send({
              error: { code: 'INVALID_INPUT', message: `createdAfter "${query.createdAfter}" is not a valid timestamp.` },
            })
          }
          createdAfterIso = parsed.toISOString()
        }
        if (query.createdBefore) {
          const parsed = new Date(query.createdBefore)
          if (Number.isNaN(parsed.valueOf())) {
            return reply.status(400).send({
              error: { code: 'INVALID_INPUT', message: `createdBefore "${query.createdBefore}" is not a valid timestamp.` },
            })
          }
          createdBeforeIso = parsed.toISOString()
        }

        let all: Array<{ id: string; content: string; source: MemoryEntry['source']; tags: string[]; evidence?: MemoryEntry['evidence'] }> = []
        if ((createdAfterIso || createdBeforeIso)
          && typeof runtime.semanticIndex.listRecentWithTimestamps === 'function') {
          all = await runtime.semanticIndex.listRecentWithTimestamps(5000, {
            createdAfter: createdAfterIso,
            createdBefore: createdBeforeIso,
          })
        } else {
          all = await runtime.semanticIndex.listRecent(5000)
        }
        const filtered = (scopeTags.length > 0 && !includeAllScopes)
          ? all.filter((entry) => isMemoryVisibleInScope(entry.tags, scopeTags))
          : all
        payload.semanticEntries = filtered.map((entry) => ({
          evidence: entry.evidence,
          id: entry.id,
          content: entry.content,
          source: entry.source,
          tags: entry.tags,
        }))
      }

      if (includeReminders && runtime.remindersStore) {
        const list = scopeTags.length > 0 && !includeAllScopes
          ? await runtime.remindersStore.listForScope(scopeTags)
          : await runtime.remindersStore.list()
        payload.reminders = list
          .filter((reminder) => !reminder.firedAt && !reminder.cancelledAt)
          .map((reminder) => ({
            id: reminder.id,
            dueAt: reminder.dueAt,
            content: reminder.content,
            scopeTags: reminder.scopeTags,
            channelType: reminder.channelType,
            chatId: reminder.chatId,
          }))
      }

      return { data: payload }
    },
  )

  app.post<{ Body: { data?: unknown; conflict?: 'skip' | 'replace'; dryRun?: boolean; applyIds?: string[] } }>(
    '/memory/import',
    async (request, reply) => {
      if (!runtime) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
        })
      }
      const body = request.body ?? ({} as { data?: unknown; conflict?: 'skip' | 'replace'; dryRun?: boolean; applyIds?: string[] })
      const raw = body.data
      if (raw === undefined || raw === null) {
        return reply.status(400).send({
          error: { code: 'INVALID_INPUT', message: 'Missing `data` field with the export payload.' },
        })
      }
      let payload: {
        version?: string
        fileMemory?: { sections: Array<{ title: string; content: string }> }
        semanticEntries?: Array<{ id: string; content: string; source: MemoryEntry['source']; tags: string[]; evidence?: MemoryEntry['evidence'] }>
        reminders?: Array<{ dueAt: string; content: string; scopeTags?: string[]; channelType?: string; chatId?: string }>
      }
      try {
        payload = typeof raw === 'string' ? JSON.parse(raw) : (raw as never)
      } catch (error) {
        return reply.status(400).send({
          error: {
            code: 'INVALID_INPUT',
            message: `Could not parse import payload: ${error instanceof Error ? error.message : String(error)}`,
          },
        })
      }
      if (payload.version !== '1.0') {
        return reply.status(400).send({
          error: { code: 'INVALID_INPUT', message: `Unsupported export version ${payload.version}` },
        })
      }
      const conflict = body.conflict === 'replace' ? 'replace' : 'skip'
      const dryRun = body.dryRun === true
      const scopeTags = readRequestScope(request)
      const stats = {
        fileSectionsApplied: 0,
        semanticAdded: 0,
        semanticReplaced: 0,
        semanticSkipped: 0,
        remindersAdded: 0,
        warnings: [] as string[],
      }

      if (payload.fileMemory) {
        const fileMemory = (scopeTags.length > 0 && runtime.fileMemoryRegistry)
          ? runtime.fileMemoryRegistry.get(scopeTags)
          : runtime.fileMemory
        if (!fileMemory) {
          stats.warnings.push('file memory store unavailable; fileMemory section ignored')
        } else {
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
                  await fileMemory.mergeMemorySectionItems(section.title, items)
                } else {
                  await fileMemory.replaceMemorySection(section.title, section.content)
                }
              } catch (error) {
                stats.warnings.push(`fileSection ${section.title}: ${error instanceof Error ? error.message : String(error)}`)
                continue
              }
            }
            stats.fileSectionsApplied += 1
          }
        }
      }

      const applyIdsFilter = Array.isArray(body.applyIds)
        ? new Set(body.applyIds.filter((id): id is string => typeof id === 'string').map((id) => id.trim()).filter(Boolean))
        : null
      if (payload.semanticEntries) {
        for (const entry of payload.semanticEntries) {
          if (!entry.id || !entry.content) continue
          if (applyIdsFilter && !applyIdsFilter.has(entry.id)) continue
          const importTags = attachScopeTags(entry.tags ?? [], scopeTags)
          let existing: MemoryEntry | null = null
          try {
            existing = await runtime.semanticIndex.get(entry.id)
          } catch { /* fresh insert */ }
          if (existing) {
            if (conflict === 'skip') {
              stats.semanticSkipped += 1
              continue
            }
            if (scopeTags.length > 0 && !isMemoryWritableInScope(existing.tags, scopeTags)) {
              stats.semanticSkipped += 1
              stats.warnings.push(`semantic ${entry.id}: scope mismatch (kept existing)`)
              continue
            }
            if (!dryRun) {
              try {
                await runtime.semanticIndex.add({
                  id: entry.id,
                  evidence: entry.evidence,
                  content: entry.content,
                  source: entry.source,
                  tags: importTags,
                })
              } catch (error) {
                stats.warnings.push(`semantic ${entry.id}: ${error instanceof Error ? error.message : String(error)}`)
                continue
              }
            }
            stats.semanticReplaced += 1
          } else {
            if (!dryRun) {
              try {
                await runtime.semanticIndex.add({
                  id: entry.id,
                  evidence: entry.evidence,
                  content: entry.content,
                  source: entry.source,
                  tags: importTags,
                })
              } catch (error) {
                stats.warnings.push(`semantic ${entry.id}: ${error instanceof Error ? error.message : String(error)}`)
                continue
              }
            }
            stats.semanticAdded += 1
          }
        }
      }

      if (payload.reminders && runtime.remindersStore) {
        const nowMs = Date.now()
        for (const reminder of payload.reminders) {
          if (!reminder.dueAt || !reminder.content) continue
          if (Date.parse(reminder.dueAt) <= nowMs) {
            stats.warnings.push(`reminder due in the past dropped (${reminder.dueAt})`)
            continue
          }
          if (!dryRun) {
            try {
              await runtime.remindersStore.add({
                dueAt: reminder.dueAt,
                content: reminder.content,
                scopeTags: attachScopeTags(reminder.scopeTags ?? [], scopeTags),
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
      } else if (payload.reminders && !runtime.remindersStore) {
        stats.warnings.push('reminders store unavailable; reminders ignored')
      }

      return { data: { dryRun, conflict, ...stats } }
    },
  )

  app.delete<{ Params: { id: string }; Querystring: { reason?: string } }>(
    '/memory/reminders/:id',
    async (request, reply) => {
      if (!runtime?.remindersStore) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Reminders store is not initialized' },
        })
      }
      const reminderParams = request.params
      const reminderQuery = request.query
      const id = reminderParams.id?.trim()
      if (!id) {
        return reply.status(400).send({
          error: { code: 'INVALID_INPUT', message: 'A reminder id is required.' },
        })
      }
      const scopeTags = readRequestScope(request)
      // Confirm ownership before cancelling.
      if (scopeTags.length > 0) {
        const visible = await runtime.remindersStore.listForScope(scopeTags)
        if (!visible.some((reminder) => reminder.id === id)) {
          return reply.status(403).send({
            error: {
              code: 'SCOPE_MISMATCH',
              message: 'Reminder belongs to a different user/channel scope.',
            },
          })
        }
      }
      const cancelled = await runtime.remindersStore.cancel(id, reminderQuery.reason)
      if (!cancelled) {
        return reply.status(404).send({
          error: { code: 'NOT_FOUND', message: 'Reminder not found or already fired/cancelled.' },
        })
      }
      return reply.status(204).send()
    },
  )
}
