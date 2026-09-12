import { createHash } from 'node:crypto'
import { createLogger } from '../logger.js'

const logger = createLogger('notify-relay-push')
const NOTIFY_RELAY_TIMEOUT_MS = 10_000

export type NotifyRelayMessageType = 'alert' | 'deployment' | 'incident' | 'report' | 'custom'
export type NotifyRelayPriority = 'low' | 'normal' | 'high' | 'critical'

export interface NotifyRelayOptions {
  baseUrl: string
  destinationId: string
  token: string
  service?: string
  environment?: string
  fetchImpl?: typeof fetch
}

export interface NotifyRelayNotification {
  id: string
  type: NotifyRelayMessageType
  priority: NotifyRelayPriority
  title: string
  text: string
}

export type NotifyRelayDeliveryStatus =
  | 'accepted'
  | 'pending_review'
  | 'rejected'
  | 'unreachable'
  | 'invalid_response'

export const notifyRelayMessageStatuses = [
  'received',
  'accepted',
  'pending_review',
  'approved',
  'rejected',
  'denied',
  'queued',
  'delivering',
  'delivered',
  'delivery_failed',
  'dead_letter',
  'expired',
] as const

export type NotifyRelayMessageStatus = (typeof notifyRelayMessageStatuses)[number]

export type NotifyRelayStatusLookupOutcome =
  | 'observed'
  | 'rejected'
  | 'unreachable'
  | 'invalid_response'

export interface NotifyRelayDeliveryResult {
  status: NotifyRelayDeliveryStatus
  messageId: string | null
  httpStatus: number | null
  errorCode: string | null
  attemptedAt: number
  completedAt: number
}

export interface NotifyRelayStatusLookupResult {
  outcome: NotifyRelayStatusLookupOutcome
  messageId: string
  messageStatus: NotifyRelayMessageStatus | null
  httpStatus: number | null
  errorCode: string | null
  attemptedAt: number
  completedAt: number
}

export function normalizeNotifyRelayErrorCode(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const normalized = value.trim()
  return /^[a-z][a-z0-9_]{0,79}$/.test(normalized) ? normalized : null
}

function parseErrorCode(value: unknown): string | null {
  if (!value || typeof value !== 'object') return null
  const error = (value as { error?: unknown }).error
  if (!error || typeof error !== 'object') return null
  return normalizeNotifyRelayErrorCode((error as { code?: unknown }).code)
}

function parseAcceptedResponse(value: unknown): {
  messageId: string
  status: 'accepted' | 'pending_review'
} | null {
  if (!value || typeof value !== 'object') return null
  const data = (value as { data?: unknown }).data
  if (!data || typeof data !== 'object') return null
  const messageId = (data as { message_id?: unknown }).message_id
  const status = (data as { status?: unknown }).status
  if (
    typeof messageId !== 'string'
    || messageId.trim().length === 0
    || messageId.length > 200
    || (status !== 'accepted' && status !== 'pending_review')
  ) {
    return null
  }
  return { messageId: messageId.trim(), status }
}

const notifyRelayMessageStatusSet = new Set<string>(notifyRelayMessageStatuses)

function parseStatusResponse(value: unknown, expectedMessageId: string): NotifyRelayMessageStatus | null {
  if (!value || typeof value !== 'object') return null
  const data = (value as { data?: unknown }).data
  if (!data || typeof data !== 'object') return null
  const messageId = (data as { message_id?: unknown }).message_id
  const status = (data as { status?: unknown }).status
  if (
    messageId !== expectedMessageId
    || typeof status !== 'string'
    || !notifyRelayMessageStatusSet.has(status)
  ) {
    return null
  }
  return status as NotifyRelayMessageStatus
}

function stripTrailingSlash(value: string): string {
  return value.replace(/\/+$/, '')
}

function limit(value: string, max: number): string {
  return value.trim().slice(0, max)
}

export class NotifyRelay {
  private readonly baseUrl: string
  private readonly destinationId: string
  private readonly token: string
  private readonly service: string
  private readonly environment: string
  private readonly fetchImpl: typeof fetch

  constructor(options: NotifyRelayOptions) {
    this.baseUrl = stripTrailingSlash(options.baseUrl)
    this.destinationId = options.destinationId
    this.token = options.token
    this.service = limit(options.service ?? 'sepilotd', 120)
    this.environment = limit(options.environment ?? 'dev', 120)
    this.fetchImpl = options.fetchImpl ?? fetch
  }

  endpoint(): string {
    return `${this.baseUrl}/v1/messages`
  }

  statusEndpoint(messageId: string): string {
    return `${this.endpoint()}/${encodeURIComponent(messageId)}`
  }

  idempotencyKey(notificationId: string): string {
    const digest = createHash('sha256').update(notificationId).digest('hex').slice(0, 32)
    return `sepilotd:${digest}`
  }

  async publish(input: NotifyRelayNotification): Promise<NotifyRelayDeliveryResult> {
    const attemptedAt = Date.now()
    try {
      const response = await this.fetchImpl(this.endpoint(), {
        method: 'POST',
        signal: AbortSignal.timeout(NOTIFY_RELAY_TIMEOUT_MS),
        headers: {
          Authorization: `Bearer ${this.token}`,
          'Content-Type': 'application/json',
          'Idempotency-Key': this.idempotencyKey(input.id),
        },
        body: JSON.stringify({
          destination_id: this.destinationId,
          type: input.type,
          priority: input.priority,
          title: limit(input.title, 200),
          text: limit(input.text, 8_000),
          metadata: {
            service: this.service,
            environment: this.environment,
          },
        }),
      })
      if (!response.ok) {
        const errorCode = parseErrorCode(await response.json().catch(() => null))
        logger.warn('notify relay rejected notification', {
          status: response.status,
          errorCode,
        })
        return {
          status: 'rejected',
          messageId: null,
          httpStatus: response.status,
          errorCode,
          attemptedAt,
          completedAt: Date.now(),
        }
      }
      const accepted = parseAcceptedResponse(await response.json().catch(() => null))
      if (!accepted || response.status !== 202) {
        logger.warn('notify relay returned an invalid acceptance response', {
          status: response.status,
        })
        return {
          status: 'invalid_response',
          messageId: null,
          httpStatus: response.status,
          errorCode: null,
          attemptedAt,
          completedAt: Date.now(),
        }
      }
      return {
        status: accepted.status,
        messageId: accepted.messageId,
        httpStatus: response.status,
        errorCode: null,
        attemptedAt,
        completedAt: Date.now(),
      }
    } catch (error) {
      logger.warn('notify relay unreachable', {
        error: error instanceof Error ? error.message : String(error),
      })
      return {
        status: 'unreachable',
        messageId: null,
        httpStatus: null,
        errorCode: null,
        attemptedAt,
        completedAt: Date.now(),
      }
    }
  }

  async status(messageId: string): Promise<NotifyRelayStatusLookupResult> {
    const attemptedAt = Date.now()
    try {
      const response = await this.fetchImpl(this.statusEndpoint(messageId), {
        method: 'GET',
        signal: AbortSignal.timeout(NOTIFY_RELAY_TIMEOUT_MS),
        headers: {
          Authorization: `Bearer ${this.token}`,
        },
      })
      if (!response.ok) {
        const errorCode = parseErrorCode(await response.json().catch(() => null))
        logger.warn('notify relay status lookup rejected', {
          status: response.status,
          errorCode,
        })
        return {
          outcome: 'rejected',
          messageId,
          messageStatus: null,
          httpStatus: response.status,
          errorCode,
          attemptedAt,
          completedAt: Date.now(),
        }
      }
      const messageStatus = parseStatusResponse(
        await response.json().catch(() => null),
        messageId,
      )
      if (!messageStatus || response.status !== 200) {
        logger.warn('notify relay returned an invalid status response', {
          status: response.status,
        })
        return {
          outcome: 'invalid_response',
          messageId,
          messageStatus: null,
          httpStatus: response.status,
          errorCode: null,
          attemptedAt,
          completedAt: Date.now(),
        }
      }
      return {
        outcome: 'observed',
        messageId,
        messageStatus,
        httpStatus: response.status,
        errorCode: null,
        attemptedAt,
        completedAt: Date.now(),
      }
    } catch (error) {
      logger.warn('notify relay status lookup unreachable', {
        error: error instanceof Error ? error.message : String(error),
      })
      return {
        outcome: 'unreachable',
        messageId,
        messageStatus: null,
        httpStatus: null,
        errorCode: null,
        attemptedAt,
        completedAt: Date.now(),
      }
    }
  }
}

export function notifyRelayFromEnv(
  env: NodeJS.ProcessEnv,
  fetchImpl?: typeof fetch,
): NotifyRelay | null {
  const baseUrl = env.NOTIFY_RELAY_BASE_URL?.trim()
  const destinationId = env.NOTIFY_RELAY_DESTINATION_ID?.trim()
  const token = env.NOTIFY_RELAY_TOKEN?.trim()
  if (!baseUrl || !destinationId || !token) return null

  return new NotifyRelay({
    baseUrl,
    destinationId,
    token,
    service: env.NOTIFY_RELAY_SERVICE?.trim() || undefined,
    environment: env.NOTIFY_RELAY_ENVIRONMENT?.trim() || undefined,
    fetchImpl,
  })
}
