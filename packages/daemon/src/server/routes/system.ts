import { link, mkdir, open, readFile, rm } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import type { FastifyInstance } from 'fastify'
import type { ISessionStore, SessionEvent, SessionMeta } from '@sepilotd/core'
import { externalNotificationRelayConfigured } from '../../notifications/publish.js'
import {
  createNotificationsRepo,
  type NotificationItem,
} from '../../notifications/repo.js'
import { createNotificationRelayOutbox } from '../../notifications/relay-outbox.js'
import {
  createSchedulerDeliveryOutbox,
  type SchedulerDeliveryRecord,
} from '../../scheduler/delivery-outbox.js'
import type { JobRun, JobStore, ScheduledJob } from '../../scheduler/job-store.js'
import { resolveRequestSurface } from '../request-surface.js'
import {
  JPAD_PUBLICATION_METADATA_KEY,
  parseJpadPublicationToolMetadata,
} from '../../tools/jpad.js'
import { buildSessionInteractionSummary } from '../session-interactions.js'
import type { RuntimeServices } from '../runtime/types.js'
import { summarizeChannelStatus } from './channels-internals.js'
import '../fastify-types.js'

const BASE_ASSISTANT_TOOL_NAMES = [
  'assistant.status',
  'workspace.prepare',
  'fs.read',
  'fs.write',
  'terminal.run',
] as const

const REQUIRED_ASSISTANT_SKILL_IDS = [
  'monitor-infrastructure',
  'research-to-jpad',
] as const

const ASSISTANT_OPERATION_CORRELATION_LIMIT = 10
const ASSISTANT_JPAD_SESSION_SCAN_LIMIT = 50
const ASSISTANT_JPAD_SESSION_READ_CONCURRENCY = 4
const ASSISTANT_CORRELATION_ID_LIMIT = 512
const UNSAFE_ASSISTANT_CORRELATION_ID_PATTERN = /[\u0000-\u001f\u007f]/

type AssistantRuntimeView = Pick<
  RuntimeServices,
  'config' | 'toolRegistry' | 'skillRegistry' | 'jobStore' | 'sessions' | 'channels'
>

export interface AssistantRuntimeStatusOptions {
  surface?: string | null
  includeInteractions?: boolean
  includeJpadPublications?: boolean
}

export interface AssistantJpadPublicationCorrelation {
  sessionId: string
  sessionUpdatedAt: string
  toolCallId: string
  operation: 'create' | 'update'
  outcome: 'confirmed' | 'unconfirmed'
  pageId: string | null
  completedAt: number
}

export interface AssistantJpadPublicationEvidence {
  status: 'available' | 'partial' | 'unavailable'
  items: AssistantJpadPublicationCorrelation[]
}

export interface AssistantSchedulerRunCorrelation {
  jobId: string
  jobName: string | null
  run: Pick<JobRun, 'id' | 'status' | 'attempt' | 'startedAt' | 'finishedAt' | 'durationMs'>
  notifications: Array<{
    id: string
    createdAt: number
    relayDelivery: NotificationItem['relayDelivery']
    relayProviderDelivery: NotificationItem['relayProviderDelivery']
  }>
  channelDeliveries: Array<Pick<
    SchedulerDeliveryRecord,
    'id' | 'status' | 'channelType' | 'attempt' | 'deliveredAt'
  >>
}

export interface AssistantChannelReadiness {
  type: string
  configured: boolean
  configuredCount: number
  enabled: boolean
  enabledCount: number
  activeCount: number
  status: 'connected' | 'disconnected' | 'connecting' | 'error' | 'not_configured'
}

export interface AssistantSchedulerJobInventory {
  total: number
  enabled: number
  disabled: number
  running: number
  failed: number
  enabledFailed: number
}

interface AssistantToolCallEvidence {
  tool: string
  eventIndex: number
  ambiguous: boolean
}

function safeAssistantCorrelationId(value: string): string | null {
  const trimmed = value.trim()
  if (
    !trimmed
    || trimmed.length > ASSISTANT_CORRELATION_ID_LIMIT
    || UNSAFE_ASSISTANT_CORRELATION_ID_PATTERN.test(trimmed)
  ) {
    return null
  }
  return trimmed
}

function buildAssistantChannelReadiness(
  runtime: AssistantRuntimeView,
): AssistantChannelReadiness[] {
  const configuredCount = new Map<string, number>()
  const enabledCount = new Map<string, number>()
  for (const channel of runtime.config.channels ?? []) {
    configuredCount.set(channel.type, (configuredCount.get(channel.type) ?? 0) + 1)
    if (channel.enabled) {
      enabledCount.set(channel.type, (enabledCount.get(channel.type) ?? 0) + 1)
    }
  }

  const activeByType = new Map<string, string[]>()
  for (const channel of runtime.channels ?? []) {
    const statuses = activeByType.get(channel.type) ?? []
    statuses.push(channel.getStatus())
    activeByType.set(channel.type, statuses)
  }

  return [...new Set([...configuredCount.keys(), ...activeByType.keys()])]
    .sort((left, right) => left.localeCompare(right))
    .map((type) => {
      const configured = configuredCount.get(type) ?? 0
      const enabled = enabledCount.get(type) ?? 0
      const statuses = activeByType.get(type) ?? []
      return {
        type,
        configured: configured > 0,
        configuredCount: configured,
        enabled: enabled > 0,
        enabledCount: enabled,
        activeCount: statuses.length,
        status: summarizeChannelStatus(statuses),
      }
    })
}

function buildAssistantSchedulerJobInventory(
  runtime: AssistantRuntimeView,
): AssistantSchedulerJobInventory {
  const jobs = runtime.jobStore.list()
  const enabledFailed = jobs.filter((job) => (
    job.enabled && (job.status === 'failed' || Boolean(job.lastError))
  )).length
  return {
    total: jobs.length,
    enabled: jobs.filter((job) => job.enabled).length,
    disabled: jobs.filter((job) => !job.enabled).length,
    running: jobs.filter((job) => job.status === 'running').length,
    failed: jobs.filter((job) => job.status === 'failed').length,
    enabledFailed,
  }
}

function toolCallsById(events: readonly SessionEvent[]): Map<string, AssistantToolCallEvidence> {
  const calls = new Map<string, AssistantToolCallEvidence>()
  events.forEach((event, eventIndex) => {
    if (event.type !== 'tool_call') return
    const previous = calls.get(event.id)
    if (previous) {
      previous.ambiguous = true
      return
    }
    calls.set(event.id, { tool: event.tool, eventIndex, ambiguous: false })
  })
  return calls
}

async function mapWithBoundedConcurrency<T, R>(
  values: readonly T[],
  concurrency: number,
  mapper: (value: T, index: number) => Promise<R>,
): Promise<R[]> {
  if (values.length === 0) return []
  const results = new Array<R>(values.length)
  const workerCount = Math.max(1, Math.min(concurrency, values.length))
  let nextIndex = 0

  await Promise.all(Array.from({ length: workerCount }, async () => {
    while (nextIndex < values.length) {
      const index = nextIndex
      nextIndex += 1
      results[index] = await mapper(values[index]!, index)
    }
  }))
  return results
}

/**
 * Project recent JPAD publication receipts from durable session journals.
 * Only a successful result from the matching built-in mutation tool is
 * accepted; arbitrary plugin metadata and model-authored output cannot forge
 * a publication. Content, workspace ids, URLs, ETags, and credentials are not
 * returned by this operator surface.
 */
export async function buildAssistantJpadPublicationEvidence(input: {
  sessions: Pick<ISessionStore, 'list' | 'getEvents'> | null | undefined
  limit?: number
  sessionScanLimit?: number
}): Promise<AssistantJpadPublicationEvidence> {
  if (!input.sessions) return { status: 'unavailable', items: [] }
  const limit = Math.max(
    1,
    Math.min(input.limit ?? ASSISTANT_OPERATION_CORRELATION_LIMIT, 50),
  )
  const sessionScanLimit = Math.max(
    1,
    Math.min(input.sessionScanLimit ?? ASSISTANT_JPAD_SESSION_SCAN_LIMIT, 200),
  )

  let sessionItems: SessionMeta[]
  try {
    // This operator surface promises recent evidence, not an unbounded
    // historical text search. A query against an encrypted store must decrypt
    // every journal before filtering and then this function would read the
    // matching journals again. Bound the authoritative window first instead.
    sessionItems = (await input.sessions.list({
      page: 1,
      perPage: sessionScanLimit,
    })).items
  } catch {
    return { status: 'unavailable', items: [] }
  }

  // Session journals are independent durable evidence sources. Read a small,
  // bounded number concurrently while preserving list order in the result
  // array; a large or unavailable journal must not serialize every other
  // readiness item behind it or erase successful sibling evidence.
  const sessionEvidence = await mapWithBoundedConcurrency(
    sessionItems,
    ASSISTANT_JPAD_SESSION_READ_CONCURRENCY,
    async (session) => {
      try {
        return { session, events: await input.sessions!.getEvents(session.id) }
      } catch {
        return { session, events: null }
      }
    },
  )

  let partial = false
  const items: AssistantJpadPublicationCorrelation[] = []
  for (const evidence of sessionEvidence) {
    const { session, events } = evidence
    if (!events) {
      partial = true
      continue
    }
    const calls = toolCallsById(events)
    for (const [eventIndex, event] of events.entries()) {
      if (event.type !== 'tool_result' || event.status !== 'success') continue
      const metadata = parseJpadPublicationToolMetadata(
        event.metadata?.[JPAD_PUBLICATION_METADATA_KEY],
      )
      if (!metadata) continue
      const expectedTool = metadata.operation === 'create'
        ? 'jpad.pages.create'
        : 'jpad.pages.update'
      const call = calls.get(event.toolCallId)
      if (
        !call
        || call.ambiguous
        || call.tool !== expectedTool
        || call.eventIndex >= eventIndex
      ) {
        continue
      }
      const sessionId = safeAssistantCorrelationId(session.id)
      const toolCallId = safeAssistantCorrelationId(event.toolCallId)
      if (!sessionId || !toolCallId) continue
      items.push({
        sessionId,
        sessionUpdatedAt: session.updatedAt,
        toolCallId,
        operation: metadata.operation,
        outcome: metadata.outcome,
        pageId: metadata.pageId,
        completedAt: metadata.completedAt,
      })
    }
  }

  items.sort((left, right) => right.completedAt - left.completedAt)
  return {
    status: partial ? 'partial' : 'available',
    items: items.slice(0, limit),
  }
}

/**
 * Join persisted scheduler execution, notification, relay-acceptance, and
 * channel-delivery evidence by the run id generated at execution start.
 * Deliberately return no output text, notification body/title, target, or
 * channel id: this authenticated readiness surface needs correlation state,
 * not user content or destination metadata.
 */
export function buildAssistantSchedulerRunCorrelations(input: {
  jobStore: Pick<JobStore, 'list' | 'listRecentRuns'> | null | undefined
  notifications: NotificationItem[]
  channelDeliveries: SchedulerDeliveryRecord[]
  limit?: number
}): AssistantSchedulerRunCorrelation[] {
  if (!input.jobStore) return []
  const limit = Math.max(
    1,
    Math.min(input.limit ?? ASSISTANT_OPERATION_CORRELATION_LIMIT, 50),
  )
  const jobsById = new Map<string, ScheduledJob>(
    input.jobStore.list().map((job) => [job.id, job]),
  )
  const correlationKey = (jobId: string, runId: string) => `${jobId}\u0000${runId}`
  const notificationsByRun = new Map<string, NotificationItem[]>()
  for (const notification of input.notifications) {
    const correlation = notification.correlation
    if (correlation?.kind !== 'scheduler' || !correlation.runId) continue
    const key = correlationKey(correlation.jobId, correlation.runId)
    const current = notificationsByRun.get(key) ?? []
    current.push(notification)
    notificationsByRun.set(key, current)
  }
  const deliveriesByRun = new Map<string, SchedulerDeliveryRecord[]>()
  for (const delivery of input.channelDeliveries) {
    if (!delivery.runId) continue
    const key = correlationKey(delivery.jobId, delivery.runId)
    const current = deliveriesByRun.get(key) ?? []
    current.push(delivery)
    deliveriesByRun.set(key, current)
  }

  return input.jobStore.listRecentRuns(limit).map((run) => ({
    jobId: run.jobId,
    jobName: jobsById.get(run.jobId)?.name ?? null,
    run: {
      id: run.id,
      status: run.status,
      attempt: run.attempt,
      startedAt: run.startedAt,
      finishedAt: run.finishedAt,
      durationMs: run.durationMs,
    },
    notifications: (notificationsByRun.get(correlationKey(run.jobId, run.id)) ?? [])
      .sort((left, right) => right.createdAt - left.createdAt)
      .map((notification) => ({
        id: notification.id,
        createdAt: notification.createdAt,
        relayDelivery: notification.relayDelivery,
        relayProviderDelivery: notification.relayProviderDelivery,
      })),
    channelDeliveries: (deliveriesByRun.get(correlationKey(run.jobId, run.id)) ?? [])
      .sort((left, right) => right.createdAt - left.createdAt)
      .map((delivery) => ({
        id: delivery.id,
        status: delivery.status,
        channelType: delivery.channelType,
        attempt: delivery.attempt,
        deliveredAt: delivery.deliveredAt,
      })),
  }))
}

interface ScopeFile {
  /** Stable per-install user scope id. Generated on first read, then
   *  reused forever. Hex characters only. */
  userId: string
  /** ISO-8601 timestamp of the initial generation. */
  createdAt: string
}

const scopeFileLoads = new Map<string, Promise<ScopeFile>>()

function parseScopeFile(raw: string, scopePath: string): ScopeFile {
  const parsed = JSON.parse(raw) as Partial<ScopeFile>
  if (!parsed || typeof parsed.userId !== 'string' || parsed.userId.trim().length === 0) {
    throw new Error(`Invalid scope file: ${scopePath}`)
  }
  return {
    userId: parsed.userId.trim(),
    createdAt: typeof parsed.createdAt === 'string'
      ? parsed.createdAt
      : new Date().toISOString(),
  }
}

async function readScopeFile(scopePath: string): Promise<ScopeFile | null> {
  try {
    return parseScopeFile(await readFile(scopePath, 'utf8'), scopePath)
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null
    throw error
  }
}

/**
 * Publish a fully-written scope file without replacing a winner created by a
 * concurrent daemon process. A hard link is the commit point: the temporary
 * file remains invisible at `scopePath` until its contents have been flushed,
 * and `link` fails with EEXIST instead of overwriting an existing identity.
 */
async function createScopeFile(scopePath: string): Promise<ScopeFile> {
  const fresh: ScopeFile = {
    userId: randomUUID().replace(/-/g, ''),
    createdAt: new Date().toISOString(),
  }
  const tempPath = `${scopePath}.${process.pid}.${randomUUID()}.tmp`

  await mkdir(dirname(scopePath), { recursive: true })
  try {
    const handle = await open(tempPath, 'wx', 0o600)
    try {
      await handle.writeFile(JSON.stringify(fresh, null, 2), 'utf8')
      await handle.sync()
    } finally {
      await handle.close()
    }

    try {
      await link(tempPath, scopePath)
      return fresh
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error
      const winner = await readScopeFile(scopePath)
      if (!winner) {
        throw new Error(`Scope file disappeared during concurrent creation: ${scopePath}`)
      }
      return winner
    }
  } finally {
    await rm(tempPath, { force: true })
  }
}

/**
 * Read or generate the per-install scope identity file.
 *
 * Surfaces (web/desktop) call GET /api/v1/system/scope on boot and
 * attach the returned `userId` as `X-Memory-Scope-User-Id` so memory
 * reads and writes are isolated per install. The id is a random UUID;
 * we deliberately do NOT derive it from the OS username/host so the
 * shipped artifacts never carry that information.
 */
async function readOrCreateScopeFile(dataDir: string): Promise<ScopeFile> {
  const scopePath = resolve(join(dataDir, 'scope.json'))
  const existingLoad = scopeFileLoads.get(scopePath)
  if (existingLoad) return existingLoad

  const load = (async () => {
    const existing = await readScopeFile(scopePath)
    return existing ?? createScopeFile(scopePath)
  })()
  scopeFileLoads.set(scopePath, load)
  void load.catch(() => {
    if (scopeFileLoads.get(scopePath) === load) scopeFileLoads.delete(scopePath)
  })
  return load
}

export async function buildAssistantRuntimeStatus(
  runtime: AssistantRuntimeView,
  options: AssistantRuntimeStatusOptions = {},
) {
  const searchProvider = runtime.config.webSearch.provider
  const aiSearchConfigured = Boolean(
    runtime.config.webSearch.endpoint?.trim()
    || process.env.AI_SEARCH_URL?.trim(),
  )

  const [requiredSkills, interactionSummary] = await Promise.all([
    Promise.all(
      REQUIRED_ASSISTANT_SKILL_IDS.map(async (id) => {
        const registered = await runtime.skillRegistry.get(id)
        return {
          id,
          available: registered !== null,
          enabled: registered?.metadata.enabled !== false && registered !== null,
          tools: registered?.metadata.tools ?? [],
        }
      }),
    ),
    options.includeInteractions
      ? buildSessionInteractionSummary(runtime).catch(() => undefined)
      : Promise.resolve(undefined),
  ])
  const requiredToolNames = new Set<string>(BASE_ASSISTANT_TOOL_NAMES)
  for (const skill of requiredSkills) {
    for (const tool of skill.tools) requiredToolNames.add(tool)
  }
  const notificationsRepo = createNotificationsRepo()
  const surface = options.surface ?? null
  const visibleNotifications = notificationsRepo.list({ surface })
  const notificationInventory = notificationsRepo.inventory({ surface })
  const latestNotifyRelayDelivery = notificationsRepo.latestRelayDelivery({ surface })
  const notifyRelayProviderDelivery = notificationsRepo.relayProviderSummary()
  const notifyRelayOutbox = createNotificationRelayOutbox().summary()
  const schedulerDeliveryOutbox = createSchedulerDeliveryOutbox()
  const schedulerRunCorrelations = buildAssistantSchedulerRunCorrelations({
    jobStore: runtime.jobStore,
    notifications: visibleNotifications,
    channelDeliveries: schedulerDeliveryOutbox.listRecent(200),
  })
  const jpadPublicationEvidence = options.includeJpadPublications
    ? await buildAssistantJpadPublicationEvidence({ sessions: runtime.sessions })
    : undefined

  return {
    generatedAt: new Date().toISOString(),
    integrations: {
      notifyRelay: {
        configured: externalNotificationRelayConfigured(),
        deliveryMode: 'durable_outbox' as const,
        latestDelivery: latestNotifyRelayDelivery,
        outbox: notifyRelayOutbox,
        providerDelivery: notifyRelayProviderDelivery,
      },
      jpad: {
        configured: Boolean(process.env.JPAD_PERSONAL_API_TOKEN?.trim()),
      },
      aiSearch: {
        configured: aiSearchConfigured,
        selected: searchProvider === 'ai-search'
          || (searchProvider === 'auto' && aiSearchConfigured),
        provider: searchProvider,
      },
      scheduler: {
        enabled: runtime.config.scheduler?.enabled !== false,
        cliSurfaceEnabled: runtime.config.scheduler?.surfaces?.cli === true,
      },
    },
    operations: {
      notifications: {
        ...notificationInventory,
        listed: visibleNotifications.length,
        truncated: notificationInventory.total > visibleNotifications.length,
      },
      ...(interactionSummary ? { interactions: interactionSummary } : {}),
      channels: buildAssistantChannelReadiness(runtime),
      schedulerJobs: buildAssistantSchedulerJobInventory(runtime),
      schedulerDeliveryOutbox: schedulerDeliveryOutbox.summary(),
      schedulerRuns: schedulerRunCorrelations,
      ...(jpadPublicationEvidence ? { jpadPublications: jpadPublicationEvidence } : {}),
    },
    skills: requiredSkills.map(({ id, available, enabled }) => ({
      id,
      available,
      enabled,
    })),
    tools: [...requiredToolNames].sort().map((name) => ({
      name,
      enabled: runtime.toolRegistry.isEnabled(name),
    })),
  }
}

/**
 * System routes — graceful daemon control surface.
 *
 *   POST /api/v1/system/shutdown
 *     Triggers the same teardown path as SIGTERM. Auth-required (master
 *     token), so a hostile origin cannot DoS the daemon by looping the
 *     endpoint.
 *
 *   GET  /api/v1/system/clients
 *     Snapshot of active WS/SSE connections plus the idle window. Used by
 *     the desktop tray ("3 connected") and by the idle reaper. Returns
 *     `count: 0` when the registry isn't wired (e.g. a hand-built test
 *     Fastify instance).
 */
export async function systemRoutes(app: FastifyInstance) {
  app.get('/system/clients', async () => {
    const registry = app.connectionRegistry
    if (!registry) return { data: { count: 0, byKind: { ws: 0, sse: 0 }, idleMs: 0, clients: [] } }
    return { data: registry.snapshot() }
  })

  // Authenticated, non-secret readiness snapshot for assistant surfaces.
  // Keep this separate from unauthenticated /health: even boolean credential
  // presence and operator-selected integrations are private configuration.
  app.get('/system/assistant', async (request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    return {
      data: await buildAssistantRuntimeStatus(runtime, {
        surface: resolveRequestSurface(request),
        includeInteractions: true,
        includeJpadPublications: true,
      }),
    }
  })

  // GET /api/v1/system/scope — return the daemon's per-install memory
  // scope id. Surfaces (web/desktop) attach it as X-Memory-Scope-User-Id
  // on every memory call so the daemon can isolate reads/writes by user.
  // First call lazily generates ~/.sepilotd/scope.json.
  app.get('/system/scope', async (_request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    try {
      const scope = await readOrCreateScopeFile(runtime.dataDir)
      return { data: { userId: scope.userId, createdAt: scope.createdAt } }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      return reply.status(500).send({
        error: { code: 'SCOPE_FILE_FAILED', message },
      })
    }
  })

  app.post('/system/shutdown', async (_request, reply) => {
    const controller = app.shutdownController
    if (!controller) {
      return reply.status(503).send({
        error: {
          code: 'SHUTDOWN_UNAVAILABLE',
          message: 'Shutdown controller is not wired',
        },
      })
    }

    // Reply *before* tearing down the HTTP server. We schedule the actual
    // shutdown on next tick so the response bytes flush; otherwise the caller
    // sees ECONNRESET and can't tell whether the daemon honored the request.
    reply.send({ data: { status: 'shutting-down' } })
    setImmediate(() => {
      void controller.shutdown('HTTP /system/shutdown').catch(() => {
        // performShutdown swallows component errors; if it throws here it's
        // catastrophic and the OS will reap us soon enough.
      })
    })
  })
}
