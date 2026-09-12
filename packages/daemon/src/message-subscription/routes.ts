import { createHash, randomUUID } from 'node:crypto'
import type { FastifyInstance } from 'fastify'
import {
  AckPolicy,
  connect,
  DeliverPolicy,
  type Consumer,
  type NatsConnection,
} from 'nats'
import WebSocket, { type RawData } from 'ws'
import { z } from 'zod'
import { publishNotification } from '../notifications/broker.js'
import { createNotificationsRepo } from '../notifications/repo.js'
import { bindCapability } from '../server/capabilities/bind.js'
import {
  buildSseResponseHeaders,
  registerSseDisconnectHandler,
} from '../server/sse-response.js'
import {
  DEFAULT_MESSAGE_SUBSCRIPTION_CONFIG,
  MessageSubscriptionConfigInput,
  type MessageStatus,
  type MessageSubscriptionConfig,
} from './schema.js'
import { createMessageSubscriptionRepo } from './repo.js'

const MESSAGE_SUBSCRIPTION_BACKGROUND_TICK_MS = 1000
type MessageSubscriptionRepo = ReturnType<typeof createMessageSubscriptionRepo>
type NotificationsRepo = ReturnType<typeof createNotificationsRepo>
type MessageSubscriptionRefreshResult =
  Awaited<ReturnType<typeof refreshMessages>>

interface WebhookBrokerMessage {
  id?: string
  source?: string
  eventType?: string
  timestamp?: number
  headers?: Record<string, string>
  payload?: Record<string, unknown>
}

export interface MessageSubscriptionRouteDeps {
  connectNats?: typeof connect
}

function buildOverview(repo: MessageSubscriptionRepo) {
  return {
    config: repo.getConfig(),
    queueStatus: repo.queueStatus(),
    status: repo.getSubscriptionStatus(),
    recentMessages: repo.listMessages(undefined, 10),
  }
}

function latestMessageByStatus(
  repo: MessageSubscriptionRepo,
) {
  return {
    pending: repo.listMessages('pending', 5),
    processing: repo.listMessages('processing', 5),
    completed: repo.listMessages('completed', 5),
    failed: repo.listMessages('failed', 5),
  }
}

function normalizeHeaders(
  config: MessageSubscriptionConfig,
): HeadersInit {
  return {
    accept: 'application/json',
    ...(config.authToken
      ? {
          authorization: `Bearer ${config.authToken}`,
        }
      : {}),
    ...config.customHeaders,
  }
}

function normalizeWebSocketHeaders(
  config: MessageSubscriptionConfig,
): Record<string, string> {
  return {
    ...(config.authToken
      ? {
          authorization: `Bearer ${config.authToken}`,
        }
      : {}),
    ...config.customHeaders,
  }
}

function mapNatsSourceToMessageType(
  source?: string,
): 'github_webhook' | 'community_post' | 'custom' {
  if (source?.trim().toLowerCase() === 'github') {
    return 'github_webhook'
  }
  return 'custom'
}

function normalizeNatsPayload(
  value: unknown,
  subject: string,
): unknown {
  if (typeof value !== 'object' || value === null) return value

  const record = value as Record<string, unknown>
  const hasBrokerShape =
    'payload' in record || 'eventType' in record || 'headers' in record
  if (!hasBrokerShape) return value

  const broker = record as WebhookBrokerMessage
  const payload =
    typeof broker.payload === 'object' && broker.payload !== null
      ? broker.payload
      : {}
  const source =
    typeof broker.source === 'string' && broker.source.trim().length > 0
      ? broker.source.trim()
      : 'nats'
  const eventType =
    typeof broker.eventType === 'string' ? broker.eventType : ''
  const title =
    typeof payload.title === 'string' && payload.title.trim().length > 0
      ? payload.title
      : typeof payload.subject === 'string' && payload.subject.trim().length > 0
        ? payload.subject
        : eventType
          ? `${source}: ${eventType}`
          : subject
  const body =
    typeof payload.body === 'string'
      ? payload.body
      : typeof payload.summary === 'string'
        ? payload.summary
        : title
  const content =
    typeof payload.content === 'string'
      ? payload.content
      : typeof payload.body === 'string'
        ? payload.body
        : JSON.stringify(payload)

  return {
    id: typeof broker.id === 'string' ? broker.id : null,
    type: mapNatsSourceToMessageType(source),
    source,
    title,
    body,
    content,
    metadata: {
      subject,
      eventType,
      headers: broker.headers ?? null,
      ...payload,
    },
    timestamp:
      typeof broker.timestamp === 'number' && Number.isFinite(broker.timestamp)
        ? Math.round(broker.timestamp)
        : Date.now(),
  }
}

function normalizeMessage(
  value: unknown,
  index: number,
): {
  id: string | null
  type: 'github_webhook' | 'community_post' | 'custom'
  source: string
  title: string
  body: string
  content: string
  metadata: Record<string, unknown>
  timestamp: number
} {
  const record =
    typeof value === 'object' && value !== null
      ? (value as Record<string, unknown>)
      : {}

  const type =
    record.type === 'github_webhook'
    || record.type === 'community_post'
    || record.type === 'custom'
      ? record.type
      : 'custom'
  const source =
    typeof record.source === 'string' && record.source.trim().length > 0
      ? record.source.trim()
      : 'external'
  const title =
    typeof record.title === 'string' && record.title.trim().length > 0
      ? record.title.trim()
      : `Message ${index + 1}`
  const body =
    typeof record.body === 'string'
      ? record.body
      : typeof record.summary === 'string'
        ? record.summary
        : typeof record.message === 'string'
          ? record.message
          : typeof record.text === 'string'
            ? record.text
        : ''
  const content =
    typeof record.content === 'string'
      ? record.content
      : body || JSON.stringify(record)
  const metadata =
    typeof record.metadata === 'object' && record.metadata !== null
      ? (record.metadata as Record<string, unknown>)
      : {}
  const timestamp =
    typeof record.timestamp === 'number' && Number.isFinite(record.timestamp)
      ? Math.round(record.timestamp)
      : Date.now()

  return {
    id: typeof record.id === 'string' ? record.id : null,
    type,
    source,
    title,
    body,
    content,
    metadata,
    timestamp,
  }
}

function hashMessage(input: {
  id: string | null
  type: string
  source: string
  title: string
  body: string
  content: string
  timestamp: number
}): string {
  return createHash('sha256')
    .update(
      JSON.stringify([
        input.id,
        input.type,
        input.source,
        input.title,
        input.body,
        input.content,
        input.timestamp,
      ]),
    )
    .digest('hex')
}

async function fetchExternalMessages(
  config: MessageSubscriptionConfig,
): Promise<unknown[]> {
  if (config.connectionType !== 'polling') {
    throw new Error('Manual refresh is only supported for polling subscriptions')
  }
  if (!config.pollingUrl.trim()) {
    throw new Error('Polling URL is required')
  }

  const response = await fetch(config.pollingUrl, {
    headers: normalizeHeaders(config),
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(
      `Polling request failed (${response.status}): ${text || response.statusText}`,
    )
  }

  const payload = (await response.json()) as unknown
  if (Array.isArray(payload)) return payload
  if (
    typeof payload === 'object'
    && payload !== null
    && Array.isArray((payload as { messages?: unknown[] }).messages)
  ) {
    return (payload as { messages: unknown[] }).messages
  }
  return []
}

function normalizeIncomingPayload(
  payload: unknown,
  sourceHint = 'external',
): unknown[] {
  if (Array.isArray(payload)) return payload
  if (
    typeof payload === 'object'
    && payload !== null
    && Array.isArray((payload as { messages?: unknown[] }).messages)
  ) {
    return (payload as { messages: unknown[] }).messages
  }
  if (typeof payload === 'string') {
    return [{
      source: sourceHint,
      title: 'Message',
      body: payload,
      content: payload,
      timestamp: Date.now(),
    }]
  }
  if (typeof payload === 'number' || typeof payload === 'boolean') {
    return [{
      source: sourceHint,
      title: 'Message',
      body: String(payload),
      content: String(payload),
      timestamp: Date.now(),
    }]
  }
  if (payload == null) return []
  return [payload]
}

function insertMessages(
  repo: ReturnType<typeof createMessageSubscriptionRepo>,
  payload: unknown,
  sourceHint?: string,
): number {
  let inserted = 0
  const messages = normalizeIncomingPayload(payload, sourceHint)
  for (const [index, candidate] of messages.entries()) {
    const normalized = normalizeMessage(candidate, index)
    const hash = hashMessage(normalized)
    const added = repo.insertMessage({
      hash,
      id: normalized.id,
      type: normalized.type,
      source: normalized.source,
      title: normalized.title,
      body: normalized.body,
      content: normalized.content,
      metadata: normalized.metadata,
      timestamp: normalized.timestamp,
      queuedAt: Date.now(),
      status: 'pending',
      processedAt: null,
      error: null,
      retryCount: 0,
      conversationId: null,
    })
    if (added) inserted += 1
  }
  return inserted
}

function finalizeSuccessfulIngest(
  repo: ReturnType<typeof createMessageSubscriptionRepo>,
  notificationsRepo: ReturnType<typeof createNotificationsRepo>,
  config: MessageSubscriptionConfig,
  inserted: number,
) {
  const processed = processPendingMessages(
    repo,
    notificationsRepo,
    config,
  )
  repo.cleanup(config.retentionDays)
  repo.trim(config.maxQueueSize)
  repo.setSubscriptionStatus({
    isConnected: true,
    lastPolled: Date.now(),
    lastError: null,
  })

  return {
    success: true,
    count: inserted,
    processed,
    overview: buildOverview(repo),
    groups: latestMessageByStatus(repo),
  }
}

function decodeWebSocketFrame(data: RawData): string {
  if (typeof data === 'string') return data
  if (Array.isArray(data)) {
    return Buffer.concat(
      data.map((chunk) =>
        Buffer.isBuffer(chunk)
          ? chunk
          : Buffer.from(new Uint8Array(chunk)),
      ),
    ).toString('utf-8')
  }
  if (Buffer.isBuffer(data)) return data.toString('utf-8')
  return Buffer.from(data).toString('utf-8')
}

function parseWebSocketPayload(data: RawData): unknown {
  const text = decodeWebSocketFrame(data).trim()
  if (!text) return []
  try {
    return JSON.parse(text) as unknown
  } catch {
    return text
  }
}

function formatWebSocketClose(code: number, reason: Buffer): string {
  const reasonText = reason.toString('utf-8').trim()
  if (reasonText) return `WebSocket closed (${code}): ${reasonText}`
  return `WebSocket closed (${code})`
}

function processPendingMessages(
  repo: MessageSubscriptionRepo,
  notificationsRepo: NotificationsRepo,
  config: MessageSubscriptionConfig,
  options?: { force?: boolean },
): number {
  if (!config.autoProcess && !options?.force) return 0

  const notificationEnabled = config.showNotification
    && (
      notificationsRepo
        .channelSettings()
        .find((entry) => entry.id === 'message-subscription')?.enabled ?? true
    )

  const pending = repo.listMessages('pending', config.maxQueueSize)
  for (const item of pending) {
    repo.updateMessage(item.hash, {
      status: 'processing',
      processedAt: null,
      error: null,
    })

    repo.updateMessage(item.hash, {
      status: 'completed',
      processedAt: Date.now(),
      error: null,
    })

    if (notificationEnabled) {
      const notification = notificationsRepo.upsert({
        title: `[${item.source}] ${item.title}`,
        body: item.body || item.content.slice(0, 180),
        url: null,
      })
      publishNotification(notification)
    }
  }
  return pending.length
}

async function refreshMessages(
  repo: MessageSubscriptionRepo,
  notificationsRepo: NotificationsRepo,
  config: MessageSubscriptionConfig,
) {
  try {
    const payload = await fetchExternalMessages(config)
    const inserted = insertMessages(repo, payload)
    return finalizeSuccessfulIngest(
      repo,
      notificationsRepo,
      config,
      inserted,
    )
  } catch (error) {
    repo.setSubscriptionStatus({
      isConnected: false,
      lastPolled: Date.now(),
      lastError: error instanceof Error ? error.message : 'Refresh failed',
    })
    return {
      success: false,
      count: 0,
      error: error instanceof Error ? error.message : 'Refresh failed',
      overview: buildOverview(repo),
      groups: latestMessageByStatus(repo),
    }
  }
}

function shouldRunBackgroundRefresh(
  config: MessageSubscriptionConfig,
  lastPolled: number | null,
): boolean {
  if (!config.enabled) return false
  if (config.connectionType !== 'polling') return false
  if (!config.pollingUrl.trim()) return false
  if (lastPolled === null) return true
  return Date.now() - lastPolled >= config.pollingInterval
}

function shouldRunBackgroundNatsFetch(
  config: MessageSubscriptionConfig,
  lastPolled: number | null,
): boolean {
  if (!config.enabled) return false
  if (config.connectionType !== 'nats') return false
  if (!config.natsUrl.trim()) return false
  if (lastPolled === null) return true
  return Date.now() - lastPolled >= config.pollingInterval
}

export async function registerMessageSubscriptionRoutes(
  app: FastifyInstance,
  deps: MessageSubscriptionRouteDeps = {},
): Promise<void> {
  const repo = createMessageSubscriptionRepo()
  const notificationsRepo = createNotificationsRepo()
  const connectNats = deps.connectNats ?? connect
  type MessageSubscriptionWatchPayload =
    | {
        type: 'snapshot'
        overview: ReturnType<typeof buildOverview>
        messages: ReturnType<MessageSubscriptionRepo['listMessages']>
        groups: ReturnType<typeof latestMessageByStatus>
      }
    | { type: 'heartbeat'; timestamp: string }
  const watchSubscribers = new Set<
    (payload: MessageSubscriptionWatchPayload) => void
  >()
  let refreshTask: Promise<MessageSubscriptionRefreshResult> | null = null
  let websocket: WebSocket | null = null
  let websocketConnectTask: Promise<void> | null = null
  let websocketReconnectAt = 0
  let natsConnection: NatsConnection | null = null
  let natsConsumer: Consumer | null = null
  let natsConnectTask: Promise<void> | null = null
  let natsFetchTask: Promise<MessageSubscriptionRefreshResult> | null = null
  let natsReconnectAt = 0
  let natsGeneration = 0
  if (!notificationsRepo.channelSettings().some((item) => item.id === 'message-subscription')) {
    notificationsRepo.setChannel('message-subscription', true)
  }

  function buildWatchSnapshot(): MessageSubscriptionWatchPayload {
    return {
      type: 'snapshot',
      overview: buildOverview(repo),
      messages: repo.listMessages(undefined, 30),
      groups: latestMessageByStatus(repo),
    }
  }

  function publishWatchSnapshot(): void {
    if (watchSubscribers.size === 0) return
    const payload = buildWatchSnapshot()
    for (const subscriber of watchSubscribers) subscriber(payload)
  }

  function websocketReconnectDelayMs(config: MessageSubscriptionConfig): number {
    return Math.max(1000, config.retryDelay)
  }

  function natsReconnectDelayMs(config: MessageSubscriptionConfig): number {
    return Math.max(1000, config.retryDelay)
  }

  async function closeNatsConnection(
    connection: NatsConnection,
  ): Promise<void> {
    try {
      if (!connection.isClosed()) {
        await connection.drain()
        return
      }
    } catch {
      // Fall back to a hard close below.
    }

    try {
      await connection.close()
    } catch {
      // Ignore close races.
    }
  }

  function disconnectWebSocket(): void {
    const active = websocket
    const activeTask = websocketConnectTask
    websocket = null
    websocketConnectTask = null
    void activeTask?.catch(() => {})
    if (!active) return

    try {
      if (
        active.readyState === WebSocket.OPEN
        || active.readyState === WebSocket.CONNECTING
      ) {
        active.close()
      } else {
        active.terminate()
      }
    } catch {
      // Ignore close races.
    }
  }

  async function disconnectNats(): Promise<void> {
    natsGeneration += 1
    const activeConnection = natsConnection
    const activeTask = natsConnectTask
    natsConnection = null
    natsConsumer = null
    natsConnectTask = null
    void activeTask?.catch(() => {})
    if (!activeConnection) return
    await closeNatsConnection(activeConnection)
  }

  async function ensureWebSocketConnection(
    options?: { forceReconnect?: boolean },
  ): Promise<void> {
    const config = repo.getConfig()
    if (!config.enabled || config.connectionType !== 'websocket') {
      disconnectWebSocket()
      return
    }

    const websocketUrl = config.websocketUrl.trim()
    if (!websocketUrl) {
      disconnectWebSocket()
      throw new Error('WebSocket URL is required')
    }

    if (options?.forceReconnect) {
      websocketReconnectAt = 0
      disconnectWebSocket()
    }

    if (
      websocket
      && (
        websocket.readyState === WebSocket.OPEN
        || websocket.readyState === WebSocket.CONNECTING
      )
    ) {
      await websocketConnectTask
      return
    }

    if (websocketConnectTask) {
      await websocketConnectTask
      return
    }

    if (Date.now() < websocketReconnectAt) {
      return
    }

    const headers = normalizeWebSocketHeaders(config)
    const socket = new WebSocket(
      websocketUrl,
      Object.keys(headers).length > 0
        ? { headers }
        : undefined,
    )
    websocket = socket

    const connectTask = new Promise<void>((resolve, reject) => {
      let settled = false

      const settleResolve = () => {
        if (settled) return
        settled = true
        resolve()
      }

      const settleReject = (error: Error) => {
        if (settled) return
        settled = true
        reject(error)
      }

      socket.on('open', () => {
        if (websocket !== socket) {
          settleResolve()
          return
        }

        websocketReconnectAt = 0
        repo.setSubscriptionStatus({
          isConnected: true,
          lastPolled: Date.now(),
          lastError: null,
        })
        publishWatchSnapshot()
        settleResolve()
      })

      socket.on('message', (data) => {
        if (websocket !== socket) return

        try {
          const payload = parseWebSocketPayload(data)
          const currentConfig = repo.getConfig()
          const inserted = insertMessages(repo, payload, 'websocket')
          void finalizeSuccessfulIngest(
            repo,
            notificationsRepo,
            currentConfig,
            inserted,
          )
          publishWatchSnapshot()
        } catch (error) {
          repo.setSubscriptionStatus({
            isConnected: true,
            lastPolled: Date.now(),
            lastError:
              error instanceof Error
                ? error.message
                : 'Failed to process WebSocket payload',
          })
          publishWatchSnapshot()
        }
      })

      socket.on('error', (error) => {
        if (websocket !== socket) return

        const message =
          error instanceof Error
            ? error.message
            : 'WebSocket connection failed'
        websocketReconnectAt = Date.now() + websocketReconnectDelayMs(config)
        repo.setSubscriptionStatus({
          isConnected: false,
          lastError: message,
        })
        publishWatchSnapshot()
        settleReject(new Error(message))
      })

      socket.on('close', (code, reason) => {
        if (websocket === socket) {
          websocket = null

          const nextConfig = repo.getConfig()
          if (
            nextConfig.enabled
            && nextConfig.connectionType === 'websocket'
          ) {
            websocketReconnectAt =
              Date.now() + websocketReconnectDelayMs(nextConfig)
          }

          repo.setSubscriptionStatus({
            isConnected: false,
            lastError:
              nextConfig.enabled && nextConfig.connectionType === 'websocket'
                ? formatWebSocketClose(code, reason)
                : null,
          })
          publishWatchSnapshot()
        }

        settleReject(new Error(formatWebSocketClose(code, reason)))
      })
    })

    websocketConnectTask = connectTask
    try {
      await connectTask
    } finally {
      if (websocketConnectTask === connectTask) {
        websocketConnectTask = null
      }
    }
  }

  async function ensureNatsConnection(
    options?: { forceReconnect?: boolean },
  ): Promise<void> {
    const config = repo.getConfig()
    if (!config.enabled || config.connectionType !== 'nats') {
      await disconnectNats()
      return
    }

    const natsUrl = config.natsUrl.trim()
    if (!natsUrl) {
      await disconnectNats()
      throw new Error('NATS URL is required')
    }

    if (options?.forceReconnect) {
      natsReconnectAt = 0
      await disconnectNats()
    }

    if (natsConnection && natsConsumer && !natsConnection.isClosed()) {
      return
    }

    if (natsConnectTask) {
      await natsConnectTask
      return
    }

    if (Date.now() < natsReconnectAt) {
      return
    }

    const generation = natsGeneration + 1
    natsGeneration = generation
    const streamName =
      config.natsStreamName.trim()
      || DEFAULT_MESSAGE_SUBSCRIPTION_CONFIG.natsStreamName
    const subject =
      config.natsSubject.trim()
      || DEFAULT_MESSAGE_SUBSCRIPTION_CONFIG.natsSubject

    const connectTask = (async () => {
      const connectionOptions: {
        servers: string
        maxReconnectAttempts: number
        reconnectTimeWait: number
        user?: string
        pass?: string
      } = {
        servers: natsUrl,
        maxReconnectAttempts: 10,
        reconnectTimeWait: 2000,
      }

      if (
        config.natsConsumerId.trim()
        && config.natsConsumerSecret.trim()
      ) {
        connectionOptions.user = config.natsConsumerId.trim()
        connectionOptions.pass = config.natsConsumerSecret.trim()
      }

      let connection: NatsConnection | null = null
      try {
        connection = await connectNats(connectionOptions)
        if (natsGeneration !== generation) {
          await closeNatsConnection(connection)
          return
        }

        const jetStream = connection.jetstream()
        const manager = await connection.jetstreamManager()
        try {
          await manager.streams.info(streamName)
        } catch {
          throw new Error(
            `NATS stream "${streamName}" was not found. Confirm the broker stream exists.`,
          )
        }

        const consumerName = `sepilot-${process.pid}-${Date.now()}`
        await manager.consumers.add(streamName, {
          durable_name: consumerName,
          filter_subject: subject,
          ack_policy: AckPolicy.Explicit,
          deliver_policy: DeliverPolicy.New,
        })
        const consumer = await jetStream.consumers.get(
          streamName,
          consumerName,
        )
        if (natsGeneration !== generation) {
          await closeNatsConnection(connection)
          return
        }

        natsConnection = connection
        natsConsumer = consumer
        natsReconnectAt = 0
        repo.setSubscriptionStatus({
          isConnected: true,
          lastError: null,
        })
        publishWatchSnapshot()

        void connection.closed().then((reason) => {
          if (natsConnection !== connection) return

          natsConnection = null
          natsConsumer = null

          const nextConfig = repo.getConfig()
          if (nextConfig.enabled && nextConfig.connectionType === 'nats') {
            natsReconnectAt =
              Date.now() + natsReconnectDelayMs(nextConfig)
          }

          repo.setSubscriptionStatus({
            isConnected: false,
            lastError:
              nextConfig.enabled && nextConfig.connectionType === 'nats'
                ? reason instanceof Error
                  ? reason.message
                  : 'NATS connection closed'
                : null,
          })
          publishWatchSnapshot()
        })
      } catch (error) {
        if (connection) {
          await closeNatsConnection(connection)
        }
        throw error
      }
    })()

    natsConnectTask = connectTask
    try {
      await connectTask
    } catch (error) {
      natsReconnectAt = Date.now() + natsReconnectDelayMs(config)
      throw error
    } finally {
      if (natsConnectTask === connectTask) {
        natsConnectTask = null
      }
    }
  }

  async function fetchNatsMessages(
    mode: 'manual' | 'background',
  ): Promise<MessageSubscriptionRefreshResult | null> {
    const config = repo.getConfig()
    if (
      mode === 'background'
      && !shouldRunBackgroundNatsFetch(
        config,
        repo.getSubscriptionStatus().lastPolled,
      )
    ) {
      return null
    }

    if (natsFetchTask) {
      return natsFetchTask
    }

    const task = (async () => {
      try {
        await ensureNatsConnection({
          forceReconnect: mode === 'manual',
        })
        if (!natsConnection || natsConnection.isClosed() || !natsConsumer) {
          throw new Error(
            repo.getSubscriptionStatus().lastError
              ?? 'NATS consumer is not connected',
          )
        }

        let inserted = 0
        const messages = await natsConsumer.fetch({
          max_messages: config.natsBatchSize,
          expires: config.natsFetchTimeout,
        })

        for await (const message of messages) {
          try {
            const payload = normalizeNatsPayload(
              message.json<unknown>(),
              message.subject,
            )
            inserted += insertMessages(repo, payload, 'nats')
            message.ack()
          } catch {
            message.nak()
          }
        }

        return finalizeSuccessfulIngest(
          repo,
          notificationsRepo,
          repo.getConfig(),
          inserted,
        )
      } catch (error) {
        const message =
          error instanceof Error ? error.message : 'NATS fetch failed'
        repo.setSubscriptionStatus({
          isConnected: Boolean(natsConnection && !natsConnection.isClosed()),
          lastPolled: Date.now(),
          lastError: message,
        })
        return {
          success: false,
          count: 0,
          error: message,
          overview: buildOverview(repo),
          groups: latestMessageByStatus(repo),
        }
      }
    })()

    natsFetchTask = task
    try {
      return await task
    } finally {
      if (natsFetchTask === task) {
        natsFetchTask = null
      }
    }
  }

  async function syncRealtimeSubscription(
    options?: { forceReconnect?: boolean },
  ): Promise<void> {
    const config = repo.getConfig()

    if (!config.enabled) {
      disconnectWebSocket()
      await disconnectNats()
      websocketReconnectAt = 0
      natsReconnectAt = 0
      repo.setSubscriptionStatus({
        isConnected: false,
        lastError: null,
      })
      publishWatchSnapshot()
      return
    }

    if (config.connectionType === 'polling') {
      disconnectWebSocket()
      await disconnectNats()
      websocketReconnectAt = 0
      natsReconnectAt = 0
      return
    }

    if (config.connectionType === 'nats') {
      disconnectWebSocket()
      websocketReconnectAt = 0
      try {
        await ensureNatsConnection(options)
      } catch (error) {
        repo.setSubscriptionStatus({
          isConnected: false,
          lastError:
            error instanceof Error
              ? error.message
              : 'NATS connection failed',
        })
        publishWatchSnapshot()
      }
      return
    }

    await disconnectNats()
    natsReconnectAt = 0
    try {
      await ensureWebSocketConnection(options)
    } catch (error) {
      repo.setSubscriptionStatus({
        isConnected: false,
        lastError:
          error instanceof Error
            ? error.message
            : 'WebSocket connection failed',
      })
      publishWatchSnapshot()
    }
  }

  async function runRefresh(
    mode: 'manual' | 'background',
  ): Promise<MessageSubscriptionRefreshResult | null> {
    const config = repo.getConfig()
    if (config.connectionType === 'websocket') {
      await syncRealtimeSubscription({
        forceReconnect: mode === 'manual',
      })
      const status = repo.getSubscriptionStatus()
      const result = {
        success: status.isConnected || !status.lastError,
        count: 0,
        processed: 0,
        error: status.lastError ?? undefined,
        overview: buildOverview(repo),
        groups: latestMessageByStatus(repo),
      }
      publishWatchSnapshot()
      return result
    }

    if (config.connectionType === 'nats') {
      const result = await fetchNatsMessages(mode)
      if (result) publishWatchSnapshot()
      return result
    }

    if (
      mode === 'background'
      && !shouldRunBackgroundRefresh(
        config,
        repo.getSubscriptionStatus().lastPolled,
      )
    ) {
      return null
    }

    if (refreshTask) {
      return refreshTask
    }

    const task = refreshMessages(repo, notificationsRepo, config)
    refreshTask = task
    try {
      const result = await task
      publishWatchSnapshot()
      return result
    } finally {
      if (refreshTask === task) refreshTask = null
    }
  }

  const timer = setInterval(() => {
    void runRefresh('background')
    void syncRealtimeSubscription()
  }, MESSAGE_SUBSCRIPTION_BACKGROUND_TICK_MS)
  app.addHook('onClose', async () => {
    clearInterval(timer)
    disconnectWebSocket()
    const pendingNatsConnectTask = natsConnectTask
    await disconnectNats()
    await refreshTask?.catch(() => {})
    await websocketConnectTask?.catch(() => {})
    await pendingNatsConnectTask?.catch(() => {})
    await natsFetchTask?.catch(() => {})
  })

  await bindCapability(
    app,
    {
      name: 'message-subscription',
      version: '1',
      methods: [
        { method: 'GET', path: '/message-subscription' },
        { method: 'PUT', path: '/message-subscription' },
        { method: 'POST', path: '/message-subscription/start' },
        { method: 'POST', path: '/message-subscription/stop' },
        { method: 'POST', path: '/message-subscription/refresh' },
        { method: 'POST', path: '/message-subscription/process' },
        { method: 'GET', path: '/message-subscription/watch' },
        { method: 'GET', path: '/message-subscription/messages' },
        { method: 'POST', path: '/message-subscription/messages/:hash/reprocess' },
        { method: 'DELETE', path: '/message-subscription/messages/:hash' },
      ],
    },
    async (a) => {
      a.get('/message-subscription', async () => buildOverview(repo))

      a.get('/message-subscription/watch', async (req, reply) => {
        reply.hijack()
        reply.raw.writeHead(200, buildSseResponseHeaders(req, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
          'X-Request-ID': req.requestId ?? randomUUID(),
        }))

        let closed = false
        const send = (payload: MessageSubscriptionWatchPayload) => {
          if (closed) return
          reply.raw.write(
            `event: message-subscription\ndata: ${JSON.stringify(payload)}\n\n`,
          )
        }
        const heartbeat = setInterval(() => {
          send({ type: 'heartbeat', timestamp: new Date().toISOString() })
        }, 15_000)
        heartbeat.unref?.()
        const close = () => {
          if (closed) return
          closed = true
          watchSubscribers.delete(send)
          clearInterval(heartbeat)
          if (!reply.raw.destroyed && !reply.raw.writableEnded) {
            reply.raw.end()
          }
        }

        watchSubscribers.add(send)
        registerSseDisconnectHandler(req, reply, close)
        send(buildWatchSnapshot())
      })

      a.put('/message-subscription', async (req, reply) => {
        const parsed = MessageSubscriptionConfigInput.safeParse(req.body)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }

        repo.saveConfig({
          ...DEFAULT_MESSAGE_SUBSCRIPTION_CONFIG,
          ...parsed.data,
        })
        processPendingMessages(repo, notificationsRepo, repo.getConfig())
        if (
          repo.getConfig().enabled
          && repo.getConfig().connectionType === 'nats'
        ) {
          await runRefresh('manual')
        } else {
          await syncRealtimeSubscription({ forceReconnect: true })
        }
        publishWatchSnapshot()
        return buildOverview(repo)
      })

      a.post('/message-subscription/start', async () => {
        repo.start()
        repo.setSubscriptionStatus({
          isConnected: false,
          lastError: null,
        })
        processPendingMessages(repo, notificationsRepo, repo.getConfig())
        if (repo.getConfig().connectionType === 'nats') {
          await runRefresh('manual')
        } else {
          await syncRealtimeSubscription({ forceReconnect: true })
        }
        publishWatchSnapshot()
        return buildOverview(repo)
      })

      a.post('/message-subscription/stop', async () => {
        repo.stop()
        disconnectWebSocket()
        await disconnectNats()
        websocketReconnectAt = 0
        natsReconnectAt = 0
        repo.setSubscriptionStatus({
          isConnected: false,
          lastError: null,
        })
        publishWatchSnapshot()
        return buildOverview(repo)
      })

      a.post('/message-subscription/refresh', async () => {
        return runRefresh('manual')
      })

      a.post('/message-subscription/process', async () => {
        const config = repo.getConfig()
        const processed = processPendingMessages(
          repo,
          notificationsRepo,
          config,
          { force: true },
        )
        repo.cleanup(config.retentionDays)
        repo.trim(config.maxQueueSize)
        publishWatchSnapshot()
        return {
          success: true,
          processed,
          overview: buildOverview(repo),
          groups: latestMessageByStatus(repo),
        }
      })

      a.get('/message-subscription/messages', async (req, reply) => {
        const parsed = z
          .object({
            status: z
              .enum(['pending', 'processing', 'completed', 'failed'])
              .optional(),
            limit: z.coerce.number().int().min(1).max(100).optional(),
          })
          .safeParse(req.query)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        return repo.listMessages(parsed.data.status as MessageStatus | undefined, parsed.data.limit)
      })

      a.post('/message-subscription/messages/:hash/reprocess', async (req, reply) => {
        const parsed = z.object({ hash: z.string().min(1) }).safeParse(req.params)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        const item = repo.reprocess(parsed.data.hash)
        if (!item) {
          void reply.status(404).send({
            code: 'NOT_FOUND',
            message: 'Message not found',
            retriable: false,
          })
          return reply
        }
        const config = repo.getConfig()
        processPendingMessages(repo, notificationsRepo, config)
        publishWatchSnapshot()
        return repo.getMessage(parsed.data.hash)
      })

      a.delete('/message-subscription/messages/:hash', async (req, reply) => {
        const parsed = z.object({ hash: z.string().min(1) }).safeParse(req.params)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        repo.remove(parsed.data.hash)
        publishWatchSnapshot()
        return { ok: true }
      })
    },
  )
}

export const __testables = {
  buildOverview,
  decodeWebSocketFrame,
  fetchExternalMessages,
  finalizeSuccessfulIngest,
  formatWebSocketClose,
  hashMessage,
  insertMessages,
  latestMessageByStatus,
  mapNatsSourceToMessageType,
  normalizeHeaders,
  normalizeIncomingPayload,
  normalizeMessage,
  normalizeNatsPayload,
  normalizeWebSocketHeaders,
  parseWebSocketPayload,
  processPendingMessages,
  refreshMessages,
  shouldRunBackgroundNatsFetch,
  shouldRunBackgroundRefresh,
}
