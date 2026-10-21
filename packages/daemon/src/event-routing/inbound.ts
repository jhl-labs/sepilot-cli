import { createHash } from 'node:crypto'
import type { RoutedEventInput } from './types.js'

export const DEFAULT_EVENT_TYPE_HEADERS = ['x-github-event', 'x-gitea-event', 'x-gitlab-event', 'x-webhook-event']
export const DEFAULT_DELIVERY_ID_HEADERS = ['x-github-delivery', 'x-gitea-delivery', 'x-gitlab-event-uuid', 'x-webhook-delivery', 'x-request-id']

interface EndpointHeaders {
  eventTypeHeader?: string
  deliveryIdHeader?: string
}

function firstHeader(headers: Record<string, string>, names: readonly string[]): string | undefined {
  for (const name of names) {
    const value = headers[name.toLowerCase()]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return undefined
}

export function webhookEventSource(normalizedPath: string): string {
  return `webhook:${normalizedPath.replace(/^\/hook\//, '')}`
}

export function deriveWebhookEventType(endpoint: EndpointHeaders, headers: Record<string, string>, body: unknown): string {
  if (endpoint.eventTypeHeader) return firstHeader(headers, [endpoint.eventTypeHeader]) ?? 'unknown'
  const fromHeader = firstHeader(headers, DEFAULT_EVENT_TYPE_HEADERS)
  if (fromHeader) return fromHeader
  const bodyType = body && typeof body === 'object' ? (body as Record<string, unknown>).type : undefined
  return typeof bodyType === 'string' && bodyType ? bodyType : 'unknown'
}

export function buildWebhookEvent(input: {
  endpoint: EndpointHeaders
  normalizedPath: string
  headers: Record<string, string>
  body: Record<string, unknown>
  rawBody: string
  now?: number
}): RoutedEventInput {
  const source = webhookEventSource(input.normalizedPath)
  const deliveryId = input.endpoint.deliveryIdHeader
    ? firstHeader(input.headers, [input.endpoint.deliveryIdHeader])
    : firstHeader(input.headers, DEFAULT_DELIVERY_ID_HEADERS)
  const dedupeKey = deliveryId
    ? `${source}:${deliveryId}`
    : `${source}:sha256:${createHash('sha256').update(input.rawBody).digest('hex')}`
  return {
    source,
    type: deriveWebhookEventType(input.endpoint, input.headers, input.body),
    subject: null,
    occurredAt: input.now ?? Date.now(),
    dedupeKey,
    payload: input.body,
  }
}
