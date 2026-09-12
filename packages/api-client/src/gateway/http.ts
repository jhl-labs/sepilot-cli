import type {
  Board,
  Card,
  CardUpdate,
  Comment,
  CommentInput,
  CreateCardInput,
  CreateTicketInput,
  Job,
  JobFilter,
  JobHandle,
  JobInput,
  JobResult,
  JobStatus,
  KnowledgeDocument,
  KnowledgeSearchOptions,
  PaginatedResult,
  SyncResult,
  Ticket,
  TicketEvent,
  TicketUpdate,
} from '@sepilotd/core'
import { ApiHttpClient } from '../http.js'
import type { ApiEnvelope } from '../daemon/types.js'
import { reconnectingJsonSseStream } from '../daemon/stream.js'

export interface GatewayClientOptions {
  baseUrl?: string
  token?: string | null
}

export interface GatewayRequestControlOptions {
  signal?: AbortSignal
}

export interface GatewayTicketFilter {
  status?: string
  labels?: string
  assignee?: string
  query?: string
  page?: number
  perPage?: number
}

export interface GatewayWatchStreamOptions {
  signal?: AbortSignal
  lastEventId?: string
}

export interface GatewayWatchEventsOptions extends GatewayWatchStreamOptions {
  reconnect?: boolean
  maxReconnects?: number
  minRetryMs?: number
  maxRetryMs?: number
}

export interface GatewayHealthComponent {
  status: 'ok' | 'degraded' | 'error'
  [key: string]: unknown
}

export interface GatewayTicketWatchRevalidateHealth extends GatewayHealthComponent {
  staleHeartbeatTakeovers: number
  recentStaleHeartbeatTakeover: boolean
  lastStaleHeartbeatTakeoverAt: string | null
  lastStaleHeartbeatTakeoverKey: string | null
  leaseTtlMs: number
  leaseRetryMs: number
  staleHeartbeatMs: number
}

export interface GatewayHealthInfo {
  status: 'ok' | 'degraded' | 'error'
  version: string
  uptime: number
  timestamp: string
  components: Record<string, GatewayHealthComponent | undefined> & {
    github?: GatewayHealthComponent
    ticketWatchRevalidate?: GatewayTicketWatchRevalidateHealth
  }
}

export interface GatewayOpenApiDocument {
  openapi: string
  info: {
    title: string
    version: string
    description?: string
  }
  paths: Record<string, unknown>
  components?: Record<string, unknown>
  [key: string]: unknown
}

export interface GatewayDelegationLease {
  delegationId: string
  claimId: string
  targetDevice: string
  generation: number
  claimedAt: string
  expiresAt: string
}

export interface GatewayDelegationClaimResult {
  claimed: boolean
  activeClaim: GatewayDelegationLease | null
}

export interface GatewayDelegationClaimReleaseResult {
  released: boolean
  activeClaim: GatewayDelegationLease | null
}

export interface GatewayTicketsWatchSnapshot {
  type: 'snapshot'
  tickets: PaginatedResult<Ticket>
}

export interface GatewayTicketsWatchDelta {
  type: 'delta'
  action: 'upsert' | 'remove'
  reason: TicketEvent['type']
  ticket: Ticket
}

export interface GatewayTicketsWatchWindowDelta {
  type: 'delta'
  action: 'window'
  reason: TicketEvent['type']
  tickets: Ticket[]
}

export interface GatewayTicketsWatchHeartbeat {
  type: 'heartbeat'
  timestamp: string
}

export type GatewayTicketsWatchPayload =
  | GatewayTicketsWatchSnapshot
  | GatewayTicketsWatchDelta
  | GatewayTicketsWatchWindowDelta
  | GatewayTicketsWatchHeartbeat

export function applyGatewayTicketsWatchPayload(
  current: Ticket[],
  event: GatewayTicketsWatchPayload,
): Ticket[] {
  if (event.type === 'snapshot') {
    return event.tickets.items
  }

  if (event.type === 'heartbeat') {
    return current
  }

  if (event.action === 'window') {
    return event.tickets
  }

  if (event.action === 'remove') {
    return current.filter((ticket) => ticket.id !== event.ticket.id)
  }

  const nextTickets = [...current]
  const existingIndex = nextTickets.findIndex(
    (ticket) => ticket.id === event.ticket.id,
  )

  if (existingIndex >= 0) {
    nextTickets[existingIndex] = event.ticket
  } else {
    nextTickets.push(event.ticket)
  }

  nextTickets.sort((left, right) => {
    const byUpdatedAt = right.updatedAt.localeCompare(left.updatedAt)
    if (byUpdatedAt !== 0) {
      return byUpdatedAt
    }
    return left.id.localeCompare(right.id)
  })

  return nextTickets
}

export type GatewayTicketComment = Comment

export type GatewayTicketCommentAction = 'created' | 'edited' | 'deleted'

export interface GatewayTicketCommentDelta extends GatewayTicketComment {
  action: GatewayTicketCommentAction
}

export interface GatewayTicketCommentsWatchSnapshot {
  type: 'snapshot'
  ticketId: string
  comments: GatewayTicketComment[]
}

export interface GatewayTicketCommentsWatchDelta {
  type: 'delta'
  ticketId: string
  comment: GatewayTicketCommentDelta
}

export interface GatewayTicketCommentsWatchHeartbeat {
  type: 'heartbeat'
  ticketId: string
  timestamp: string
}

export type GatewayTicketCommentsWatchPayload =
  | GatewayTicketCommentsWatchSnapshot
  | GatewayTicketCommentsWatchDelta
  | GatewayTicketCommentsWatchHeartbeat

export function applyGatewayTicketCommentsWatchPayload(
  current: GatewayTicketComment[],
  event: GatewayTicketCommentsWatchPayload,
): GatewayTicketComment[] {
  if (event.type === 'snapshot') {
    return event.comments
  }

  if (event.type === 'heartbeat') {
    return current
  }

  if (event.comment.action === 'deleted') {
    return current.filter((comment) => comment.id !== event.comment.id)
  }

  const { action: _action, ...nextComment } = event.comment
  const nextComments = [...current]
  const existingIndex = nextComments.findIndex(
    (comment) => comment.id === nextComment.id,
  )

  if (existingIndex >= 0) {
    nextComments[existingIndex] = nextComment
    return nextComments
  }

  nextComments.push(nextComment)
  nextComments.sort((left, right) => {
    const byTimestamp = left.createdAt.localeCompare(right.createdAt)
    if (byTimestamp !== 0) {
      return byTimestamp
    }
    return left.id.localeCompare(right.id)
  })
  return nextComments
}

export const DEFAULT_GATEWAY_BASE_URL = 'http://127.0.0.1:17610'

function parseOptions(
  options?: string | GatewayClientOptions,
): Required<GatewayClientOptions> {
  if (typeof options === 'string') {
    return { baseUrl: options, token: null }
  }
  return {
    baseUrl: options?.baseUrl ?? DEFAULT_GATEWAY_BASE_URL,
    token: options?.token ?? null,
  }
}

function apiPath(path: string): string {
  return `/api/v1${path.startsWith('/') ? path : `/${path}`}`
}

/**
 * Thrown when a gateway delegation response envelope does not match the
 * expected shape. Surfacing a named error (instead of letting a downstream
 * property access throw an opaque TypeError) makes malformed/negotiation
 * mismatches visible at the boundary.
 */
export class GatewayResponseError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'GatewayResponseError'
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseDelegationLease(value: unknown): GatewayDelegationLease | null {
  if (value === null || value === undefined) return null
  if (!isRecord(value)) {
    throw new GatewayResponseError('gateway delegation lease is not an object')
  }
  const { delegationId, claimId, targetDevice, generation, claimedAt, expiresAt } = value
  if (
    typeof delegationId !== 'string'
    || typeof claimId !== 'string'
    || typeof targetDevice !== 'string'
    || typeof generation !== 'number'
    || typeof claimedAt !== 'string'
    || typeof expiresAt !== 'string'
  ) {
    throw new GatewayResponseError('gateway delegation lease has missing or mistyped fields')
  }
  return { delegationId, claimId, targetDevice, generation, claimedAt, expiresAt }
}

function parseDelegationClaimResult(value: unknown): GatewayDelegationClaimResult {
  if (!isRecord(value) || typeof value.claimed !== 'boolean') {
    throw new GatewayResponseError('gateway delegation claim result is malformed')
  }
  return { claimed: value.claimed, activeClaim: parseDelegationLease(value.activeClaim) }
}

function parseDelegationReleaseResult(value: unknown): GatewayDelegationClaimReleaseResult {
  if (!isRecord(value) || typeof value.released !== 'boolean') {
    throw new GatewayResponseError('gateway delegation release result is malformed')
  }
  return { released: value.released, activeClaim: parseDelegationLease(value.activeClaim) }
}

function parseDelegationComment(value: unknown): Comment {
  if (!isRecord(value)) {
    throw new GatewayResponseError('gateway delegation comment is not an object')
  }
  const { id, ticketId, body, type, author, createdAt } = value
  if (
    typeof id !== 'string'
    || typeof ticketId !== 'string'
    || typeof body !== 'string'
    || typeof type !== 'string'
    || typeof author !== 'string'
    || typeof createdAt !== 'string'
  ) {
    throw new GatewayResponseError('gateway delegation comment has missing or mistyped fields')
  }
  return { id, ticketId, body, type: type as Comment['type'], author, createdAt }
}

function ticketFilterPath(basePath: string, filter: GatewayTicketFilter = {}): string {
  const qs = new URLSearchParams()
  if (filter.status) qs.set('status', filter.status)
  if (filter.labels) qs.set('labels', filter.labels)
  if (filter.assignee) qs.set('assignee', filter.assignee)
  if (filter.query) qs.set('query', filter.query)
  if (filter.page != null) qs.set('page', String(filter.page))
  if (filter.perPage != null) qs.set('perPage', String(filter.perPage))
  return qs.size > 0 ? `${basePath}?${qs.toString()}` : basePath
}

function gatewayWatchHeaders(options?: GatewayWatchStreamOptions): HeadersInit | undefined {
  if (!options?.lastEventId) {
    return undefined
  }
  return {
    'Last-Event-ID': options.lastEventId,
  }
}

export class GatewayClient {
  readonly baseUrl: string
  private readonly transport: ApiHttpClient

  constructor(options?: string | GatewayClientOptions) {
    const resolved = parseOptions(options)
    this.baseUrl = resolved.baseUrl
    this.transport = new ApiHttpClient({
      baseUrl: this.baseUrl,
      token: resolved.token,
    })
  }

  async health(options?: GatewayRequestControlOptions): Promise<boolean> {
    try {
      return (await this.healthInfo(options)) !== null
    } catch {
      return false
    }
  }

  async healthInfo(
    options?: GatewayRequestControlOptions,
  ): Promise<GatewayHealthInfo | null> {
    try {
      const response = await this.transport.fetch(apiPath('/health'), {
        method: 'GET',
        signal: options?.signal,
      })
      if (!response.ok) {
        return null
      }

      const envelope = await response.json() as ApiEnvelope<GatewayHealthInfo>
      return envelope.data
    } catch {
      return null
    }
  }

  async openApi(): Promise<GatewayOpenApiDocument> {
    return this.transport.get<GatewayOpenApiDocument>(apiPath('/openapi.json'))
  }

  async listTickets(filter: GatewayTicketFilter = {}): Promise<PaginatedResult<Ticket>> {
    const { data } = await this.transport.get<ApiEnvelope<PaginatedResult<Ticket>>>(
      apiPath(ticketFilterPath('/tickets', filter)),
    )
    return data
  }

  watchTicketsStream(
    filter: GatewayTicketFilter = {},
    options?: GatewayWatchStreamOptions,
  ): Promise<Response> {
    const path = ticketFilterPath('/tickets/watch', filter)
    return this.transport.fetch(apiPath(path), {
      method: 'GET',
      signal: options?.signal,
      headers: gatewayWatchHeaders(options),
    })
  }

  async *watchTicketsEvents(
    filter: GatewayTicketFilter = {},
    options: GatewayWatchEventsOptions = {},
  ): AsyncIterable<GatewayTicketsWatchPayload> {
    for await (const event of reconnectingJsonSseStream<GatewayTicketsWatchPayload>(
      ({ signal, lastEventId }) =>
        this.watchTicketsStream(filter, { signal, lastEventId }),
      options,
    )) {
      yield event.data
    }
  }

  async getTicket(id: string): Promise<Ticket> {
    const { data } = await this.transport.get<ApiEnvelope<Ticket>>(
      apiPath(`/tickets/${encodeURIComponent(id)}`),
    )
    return data
  }

  async createTicket(input: CreateTicketInput): Promise<Ticket> {
    const { data } = await this.transport.post<ApiEnvelope<Ticket>>(
      apiPath('/tickets'),
      input,
    )
    return data
  }

  async updateTicket(id: string, update: TicketUpdate): Promise<Ticket> {
    const { data } = await this.transport.put<ApiEnvelope<Ticket>>(
      apiPath(`/tickets/${encodeURIComponent(id)}`),
      update,
    )
    return data
  }

  async getComments(ticketId: string): Promise<Comment[]> {
    const { data } = await this.transport.get<ApiEnvelope<Comment[]>>(
      apiPath(`/tickets/${encodeURIComponent(ticketId)}/comments`),
    )
    return data
  }

  watchTicketCommentsStream(
    ticketId: string,
    options?: GatewayWatchStreamOptions,
  ): Promise<Response> {
    return this.transport.fetch(
      apiPath(`/tickets/${encodeURIComponent(ticketId)}/comments/watch`),
      {
        method: 'GET',
        signal: options?.signal,
        headers: gatewayWatchHeaders(options),
      },
    )
  }

  async *watchTicketCommentsEvents(
    ticketId: string,
    options: GatewayWatchEventsOptions = {},
  ): AsyncIterable<GatewayTicketCommentsWatchPayload> {
    for await (const event of reconnectingJsonSseStream<GatewayTicketCommentsWatchPayload>(
      ({ signal, lastEventId }) =>
        this.watchTicketCommentsStream(ticketId, { signal, lastEventId }),
      options,
    )) {
      yield event.data
    }
  }

  async addComment(ticketId: string, comment: CommentInput): Promise<Comment> {
    const { data } = await this.transport.post<ApiEnvelope<Comment>>(
      apiPath(`/tickets/${encodeURIComponent(ticketId)}/comments`),
      comment,
    )
    return data
  }

  async getDelegationComments(): Promise<Comment[]> {
    const { data } = await this.transport.get<ApiEnvelope<unknown>>(
      apiPath('/delegations/comments'),
    )
    if (!Array.isArray(data)) {
      throw new GatewayResponseError('gateway delegation comments response is not an array')
    }
    return data.map(parseDelegationComment)
  }

  async addDelegationComment(comment: CommentInput): Promise<Comment> {
    const { data } = await this.transport.post<ApiEnvelope<unknown>>(
      apiPath('/delegations/comments'),
      comment,
    )
    return parseDelegationComment(data)
  }

  async getDelegationClaim(
    delegationId: string,
  ): Promise<GatewayDelegationLease | null> {
    const { data } = await this.transport.get<ApiEnvelope<unknown>>(
      apiPath(`/delegations/${encodeURIComponent(delegationId)}/claim`),
    )
    if (!isRecord(data)) {
      throw new GatewayResponseError('gateway delegation claim response is malformed')
    }
    return parseDelegationLease(data.activeClaim)
  }

  async claimDelegation(
    delegationId: string,
    input: {
      claimId: string
      targetDevice: string
      ttlMs?: number
      generation?: number
    },
  ): Promise<GatewayDelegationClaimResult> {
    const { data } = await this.transport.post<ApiEnvelope<unknown>>(
      apiPath(`/delegations/${encodeURIComponent(delegationId)}/claim`),
      input,
    )
    return parseDelegationClaimResult(data)
  }

  async releaseDelegationClaim(
    delegationId: string,
    input: {
      claimId: string
      targetDevice?: string
    },
  ): Promise<GatewayDelegationClaimReleaseResult> {
    const qs = new URLSearchParams({ claimId: input.claimId })
    if (input.targetDevice) {
      qs.set('targetDevice', input.targetDevice)
    }
    const { data } = await this.transport.delete<ApiEnvelope<unknown>>(
      apiPath(`/delegations/${encodeURIComponent(delegationId)}/claim?${qs.toString()}`),
    )
    return parseDelegationReleaseResult(data)
  }

  async dispatchJob(input: JobInput): Promise<JobHandle> {
    const { data } = await this.transport.post<ApiEnvelope<JobHandle>>(
      apiPath('/jobs'),
      input,
    )
    return data
  }

  async getJobStatus(jobId: string): Promise<JobStatus> {
    const { data } = await this.transport.get<ApiEnvelope<JobStatus>>(
      apiPath(`/jobs/${encodeURIComponent(jobId)}`),
    )
    return data
  }

  async getJobResult(jobId: string): Promise<JobResult> {
    const { data } = await this.transport.get<ApiEnvelope<JobResult>>(
      apiPath(`/jobs/${encodeURIComponent(jobId)}/result`),
    )
    return data
  }

  async cancelJob(jobId: string): Promise<{ cancelled: true }> {
    const { data } = await this.transport.delete<ApiEnvelope<{ cancelled: true }>>(
      apiPath(`/jobs/${encodeURIComponent(jobId)}`),
    )
    return data
  }

  async listJobs(filter: JobFilter = {}): Promise<PaginatedResult<Job>> {
    const qs = new URLSearchParams()
    if (filter.workflow) qs.set('workflow', filter.workflow)
    if (filter.status?.length) qs.set('status', filter.status.join(','))
    if (filter.pagination?.page != null) qs.set('page', String(filter.pagination.page))
    if (filter.pagination?.perPage != null) qs.set('perPage', String(filter.pagination.perPage))

    const path = qs.size > 0 ? `/jobs?${qs.toString()}` : '/jobs'
    const { data } = await this.transport.get<ApiEnvelope<PaginatedResult<Job>>>(
      apiPath(path),
    )
    return data
  }

  async searchKnowledge(
    query: string,
    options: KnowledgeSearchOptions = {},
  ): Promise<KnowledgeDocument[]> {
    const qs = new URLSearchParams({ query })
    if (options.limit != null) qs.set('limit', String(options.limit))
    if (options.type) qs.set('type', options.type)

    const { data } = await this.transport.get<ApiEnvelope<KnowledgeDocument[]>>(
      apiPath(`/knowledge/search?${qs.toString()}`),
    )
    return data
  }

  async getDocument(path: string): Promise<KnowledgeDocument> {
    const { data } = await this.transport.get<ApiEnvelope<KnowledgeDocument>>(
      apiPath(`/knowledge/${encodeURIComponent(path)}`),
    )
    return data
  }

  async updateDocument(path: string, content: string): Promise<KnowledgeDocument> {
    const { data } = await this.transport.put<ApiEnvelope<KnowledgeDocument>>(
      apiPath(`/knowledge/${encodeURIComponent(path)}`),
      { content },
    )
    return data
  }

  async syncKnowledge(direction: 'push' | 'pull' | 'both'): Promise<SyncResult> {
    const { data } = await this.transport.post<ApiEnvelope<SyncResult>>(
      apiPath('/knowledge/sync'),
      { direction },
    )
    return data
  }

  async getBoard(boardId: string): Promise<Board> {
    const { data } = await this.transport.get<ApiEnvelope<Board>>(
      apiPath(`/boards/${encodeURIComponent(boardId)}`),
    )
    return data
  }

  async listCards(
    boardId: string,
    filter: { columnId?: string; assignee?: string; ticketId?: string } = {},
  ): Promise<Card[]> {
    const qs = new URLSearchParams()
    if (filter.columnId) qs.set('columnId', filter.columnId)
    if (filter.assignee) qs.set('assignee', filter.assignee)
    if (filter.ticketId) qs.set('ticketId', filter.ticketId)
    const path = qs.size > 0
      ? `/boards/${encodeURIComponent(boardId)}/cards?${qs.toString()}`
      : `/boards/${encodeURIComponent(boardId)}/cards`

    const { data } = await this.transport.get<ApiEnvelope<Card[]>>(apiPath(path))
    return data
  }

  async createCard(boardId: string, input: CreateCardInput): Promise<Card> {
    const { data } = await this.transport.post<ApiEnvelope<Card>>(
      apiPath(`/boards/${encodeURIComponent(boardId)}/cards`),
      input,
    )
    return data
  }

  async moveCard(cardId: string, columnId: string): Promise<Card> {
    const { data } = await this.transport.put<ApiEnvelope<Card>>(
      apiPath(`/cards/${encodeURIComponent(cardId)}/move`),
      { columnId },
    )
    return data
  }

  async updateCard(cardId: string, update: CardUpdate): Promise<Card> {
    const { data } = await this.transport.put<ApiEnvelope<Card>>(
      apiPath(`/cards/${encodeURIComponent(cardId)}`),
      update,
    )
    return data
  }
}
