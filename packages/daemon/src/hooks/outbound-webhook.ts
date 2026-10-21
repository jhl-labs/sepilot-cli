import type {
  IAuditLogger,
  HookEvent,
  HookPayload,
  HookResult,
  IHookHandler,
  IHookRegistry,
} from '@sepilotd/core'
import { createHash, randomUUID } from 'node:crypto'
import { DEFAULT_OUTBOUND_WEBHOOK_RETRY_CONFIG } from '../config/schema.js'
import type { HookRegistry } from '../hook/registry.js'
import { createLogger } from '../logger.js'
import { DAEMON_VERSION } from '../version.js'
import { isPermanentDeliveryError, postWithEgressGuard } from '../utils/egress-post.js'

const logger = createLogger('webhook-out')
const OUTBOUND_WEBHOOK_TIMEOUT_MS = 10_000

export const OUTBOUND_WEBHOOK_DELIVERY_AUDIT_EVENT = 'hook.outbound_webhook.delivery'

export interface OutboundWebhookConfig {
  enabled?: boolean
  url: string
  events: readonly HookEvent[]
  secret?: string
  headers?: Readonly<Record<string, string>>
  retry?: {
    maxAttempts?: number
    backoffMs?: number
  }
}

export interface OutboundWebhookObserverOptions {
  auditLogger?: IAuditLogger
  deviceName?: string
}

export interface OutboundWebhookDeliveryResult {
  timestamp: string
  deliveryId: string
  webhookId: string
  url: string
  hookEvent: HookEvent
  deliveryStatus: 'success' | 'error'
  attemptCount: number
  statusCode?: number
  durationMs: number
  error?: string
  sessionId?: string
  replayedFromDeliveryId?: string
}

interface OutboundWebhookDeliveryAuditEvent
  extends Omit<OutboundWebhookDeliveryResult, 'sessionId'> {
  timestamp: string
  event: typeof OUTBOUND_WEBHOOK_DELIVERY_AUDIT_EVENT
  device: string
  session?: string
  payload?: HookPayload
}

export interface OutboundWebhookDeliveryOptions {
  deliveryId?: string
  replayedFromDeliveryId?: string
}

export function outboundWebhookIdFromUrl(url: string): string {
  const parsed = new URL(url)
  const base = `webhook:${parsed.hostname}${parsed.pathname}`.replace(/[^a-zA-Z0-9:/._-]/g, '_')
  if (parsed.protocol === 'https:' && !parsed.port && !parsed.search && !parsed.username && !parsed.password && !parsed.hash) {
    return base
  }

  const hash = createHash('sha256')
    .update(parsed.href)
    .digest('base64url')
    .slice(0, 10)
  return `${base}:${hash}`
}

function isRetryableStatusCode(statusCode: number | undefined): boolean {
  return statusCode === 429 || (typeof statusCode === 'number' && statusCode >= 500)
}

function isRetryableError(error: unknown): boolean {
  if (isPermanentDeliveryError(error)) {
    return false
  }
  const statusCode = typeof error === 'object' && error && 'statusCode' in error
    && typeof (error as { statusCode?: unknown }).statusCode === 'number'
    ? (error as { statusCode: number }).statusCode
    : undefined
  return statusCode == null || isRetryableStatusCode(statusCode)
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function resolveRetryPolicy(
  config: OutboundWebhookConfig,
): Required<NonNullable<OutboundWebhookConfig['retry']>> {
  return {
    maxAttempts: config.retry?.maxAttempts
      ?? DEFAULT_OUTBOUND_WEBHOOK_RETRY_CONFIG.maxAttempts,
    backoffMs: config.retry?.backoffMs
      ?? DEFAULT_OUTBOUND_WEBHOOK_RETRY_CONFIG.backoffMs,
  }
}

function extractSessionId(payload: HookPayload): string | undefined {
  if (!payload.data || typeof payload.data !== 'object') {
    return undefined
  }
  const sessionId = (payload.data as Record<string, unknown>).sessionId
  return typeof sessionId === 'string' && sessionId.length > 0
    ? sessionId
    : undefined
}

export class OutboundWebhookHandler implements IHookHandler {
  readonly id: string
  readonly priority = 100
  private config: OutboundWebhookConfig
  private readonly observer: OutboundWebhookObserverOptions

  constructor(
    config: OutboundWebhookConfig,
    observer: OutboundWebhookObserverOptions = {},
  ) {
    this.config = config
    this.id = outboundWebhookIdFromUrl(config.url)
    this.observer = observer
  }

  async handle(payload: HookPayload): Promise<HookResult> {
    void deliverOutboundWebhook(this.config, payload, this.observer)
      .then((result) => {
        if (result.deliveryStatus !== 'error') {
          return
        }
        logger.warn(`Webhook delivery failed: ${result.error ?? 'unknown error'}`, {
          url: this.config.url,
        })
      })
      .catch((error) => {
        logger.warn(`Webhook delivery failed: ${error instanceof Error ? error.message : String(error)}`, {
          url: this.config.url,
        })
      })
    return { action: 'continue' }
  }
}

async function logDelivery(
  payload: HookPayload,
  observer: OutboundWebhookObserverOptions,
  record: Omit<OutboundWebhookDeliveryAuditEvent, 'timestamp' | 'event' | 'device' | 'session'>,
): Promise<void> {
  if (!observer.auditLogger) {
    return
  }

  await observer.auditLogger.log({
    timestamp: new Date().toISOString(),
    event: OUTBOUND_WEBHOOK_DELIVERY_AUDIT_EVENT,
    device: observer.deviceName ?? 'unknown',
    session: extractSessionId(payload),
    ...record,
  })
}

export async function deliverOutboundWebhook(
  config: OutboundWebhookConfig,
  payload: HookPayload,
  observer: OutboundWebhookObserverOptions = {},
  options: OutboundWebhookDeliveryOptions = {},
): Promise<OutboundWebhookDeliveryResult> {
  const deliveryId = options.deliveryId ?? randomUUID()
  const startedAt = Date.now()
  const retryPolicy = resolveRetryPolicy(config)
  const outboundBody = JSON.stringify({
    event: payload.event,
    timestamp: new Date().toISOString(),
    data: payload.data,
  })
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'User-Agent': `sepilotd/${DAEMON_VERSION}`,
    ...config.headers,
  }

  if (config.secret) {
    const { createHmac } = await import('node:crypto')
    headers['X-Signature'] = createHmac('sha256', config.secret).update(outboundBody).digest('hex')
  }
  headers['X-SePilot-Delivery-Id'] = deliveryId
  if (options.replayedFromDeliveryId) {
    headers['X-SePilot-Replayed-From-Delivery-Id'] = options.replayedFromDeliveryId
  }

  let result: OutboundWebhookDeliveryResult | null = null

  for (let attempt = 1; attempt <= retryPolicy.maxAttempts; attempt += 1) {
    try {
      const res = await postWithEgressGuard(config.url, {
        headers: {
          ...headers,
          'X-SePilot-Delivery-Attempt': String(attempt),
        },
        body: outboundBody,
        signal: AbortSignal.timeout(OUTBOUND_WEBHOOK_TIMEOUT_MS),
      })

      if (!res.ok) {
        const error = new Error(`HTTP ${res.status}`)
        ;(error as Error & { statusCode?: number }).statusCode = res.status
        throw error
      }

      result = {
        timestamp: new Date().toISOString(),
        deliveryId,
        webhookId: outboundWebhookIdFromUrl(config.url),
        url: config.url,
        hookEvent: payload.event,
        deliveryStatus: 'success',
        attemptCount: attempt,
        statusCode: res.status,
        durationMs: Date.now() - startedAt,
        sessionId: extractSessionId(payload),
        ...(options.replayedFromDeliveryId
          ? { replayedFromDeliveryId: options.replayedFromDeliveryId }
          : {}),
      }
      break
    } catch (error) {
      const retryable = isRetryableError(error)
      if (retryable && attempt < retryPolicy.maxAttempts) {
        logger.warn('Retrying outbound webhook delivery', {
          url: config.url,
          deliveryId,
          attempt,
        })
        await sleep(retryPolicy.backoffMs * 2 ** Math.max(0, attempt - 1))
        continue
      }

      const message = error instanceof Error ? error.message : String(error)
      const statusCode = typeof error === 'object' && error && 'statusCode' in error
        && typeof (error as { statusCode?: unknown }).statusCode === 'number'
        ? (error as { statusCode: number }).statusCode
        : undefined
      result = {
        timestamp: new Date().toISOString(),
        deliveryId,
        webhookId: outboundWebhookIdFromUrl(config.url),
        url: config.url,
        hookEvent: payload.event,
        deliveryStatus: 'error',
        attemptCount: attempt,
        statusCode,
        durationMs: Date.now() - startedAt,
        error: message,
        sessionId: extractSessionId(payload),
        ...(options.replayedFromDeliveryId
          ? { replayedFromDeliveryId: options.replayedFromDeliveryId }
          : {}),
      }
      break
    }
  }

  if (!result) {
    throw new Error('Outbound webhook delivery did not produce a result')
  }

  await logDelivery(payload, observer, {
    deliveryId: result.deliveryId,
    webhookId: result.webhookId,
    url: result.url,
    hookEvent: result.hookEvent,
    deliveryStatus: result.deliveryStatus,
    attemptCount: result.attemptCount,
    statusCode: result.statusCode,
    durationMs: result.durationMs,
    error: result.error,
    payload,
    replayedFromDeliveryId: result.replayedFromDeliveryId,
  })

  return result
}

export function registerOutboundWebhooks(
  hookRegistry: IHookRegistry,
  webhooks: readonly OutboundWebhookConfig[],
  observer: OutboundWebhookObserverOptions = {},
): void {
  for (const config of webhooks) {
    if (config.enabled === false) continue
    const handler = new OutboundWebhookHandler(config, observer)
    for (const event of config.events) {
      hookRegistry.register(event, handler)
    }
  }
}

export function replaceOutboundWebhooks(
  hookRegistry: HookRegistry,
  webhooks: readonly OutboundWebhookConfig[],
  observer: OutboundWebhookObserverOptions = {},
): void {
  hookRegistry.removeHandlers((_event, handler) =>
    handler instanceof OutboundWebhookHandler,
  )
  registerOutboundWebhooks(hookRegistry, webhooks, observer)
}
