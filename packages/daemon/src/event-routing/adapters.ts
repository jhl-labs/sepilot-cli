import { createHmac } from 'node:crypto'
import type { IChannel } from '@sepilotd/core'
import { splitChannelMessage } from '../channels/message-chunker.js'
import { publishStoredNotification } from '../notifications/publish.js'
import { isPermanentDeliveryError, postWithEgressGuard } from '../utils/egress-post.js'
import { REDACTED, type EventDestination, type MessengerPreset } from './config-schema.js'
import { EVENT_ROUTING_ORIGIN, EVENT_ROUTING_ORIGIN_HEADER } from './origin.js'
import type { DeliveryOutcome, RenderedMessage } from './types.js'

export const MESSENGER_TEXT_LIMITS: Record<MessengerPreset, number> = {
  mattermost: 16_000,
  slack: 40_000,
  discord: 2_000,
}

export const DELIVERY_REQUEST_TIMEOUT_MS = 10_000

export interface DeliveryContext {
  deliveryId: string
  attempt: number
  signal: AbortSignal
}

export type DeliverFn = (destination: EventDestination, message: RenderedMessage, ctx: DeliveryContext) => Promise<DeliveryOutcome>

export interface AdapterDeps {
  post?: typeof postWithEgressGuard
  channels: () => readonly IChannel[]
  notify?: (input: { title: string; body: string; topic: string }) => void
}

export function messengerBodies(preset: MessengerPreset, text: string): Record<string, string>[] {
  const key = preset === 'discord' ? 'content' : 'text'
  return splitChannelMessage(text, { limit: MESSENGER_TEXT_LIMITS[preset] }).map((chunk) => ({ [key]: chunk }))
}

export function httpEnvelope(message: RenderedMessage): Record<string, unknown> {
  return { event: message.event, rule: message.ruleId, title: message.title, text: message.text }
}

function outcomeFromStatus(response: { ok: boolean; status: number | undefined }): DeliveryOutcome {
  if (response.ok) return { ok: true }
  const status = response.status
  const retryable = status === undefined || status === 429 || status >= 500
  return { ok: false, retryable, error: `HTTP ${status ?? 'unknown'}` }
}

function outcomeFromError(error: unknown): DeliveryOutcome {
  const message = error instanceof Error ? error.message : String(error)
  return { ok: false, retryable: !isPermanentDeliveryError(error), error: message }
}

function requestSignal(deliverySignal: AbortSignal): AbortSignal {
  return AbortSignal.any([deliverySignal, AbortSignal.timeout(DELIVERY_REQUEST_TIMEOUT_MS)])
}

export function redactDeliveryError(destination: EventDestination, message: string): string {
  if (destination.kind !== 'messenger-webhook' && destination.kind !== 'http') return message
  const secrets = new Set<string>([destination.url])
  try {
    const parsed = new URL(destination.url)
    const pathAndQuery = `${parsed.pathname}${parsed.search}`
    if (pathAndQuery.length > 1) secrets.add(pathAndQuery)
  } catch {
  }
  if (destination.kind === 'http') {
    if (destination.secret) secrets.add(destination.secret)
    for (const value of Object.values(destination.headers)) secrets.add(value)
  }
  let redacted = message
  for (const secret of secrets) {
    if (secret.length < 4) continue
    redacted = redacted.split(secret).join(REDACTED)
  }
  return redacted
}

export function createDeliver(deps: AdapterDeps): DeliverFn {
  const post = deps.post ?? postWithEgressGuard
  const notify = deps.notify ?? ((input) => { publishStoredNotification(input) })

  const postJson = async (url: string, headers: Record<string, string>, body: string, signal: AbortSignal): Promise<DeliveryOutcome> => {
    try {
      return outcomeFromStatus(await post(url, { headers: { 'content-type': 'application/json', ...headers }, body, signal: requestSignal(signal) }))
    } catch (error) {
      return outcomeFromError(error)
    }
  }

  return async (destination, message, ctx) => {
    switch (destination.kind) {
      case 'messenger-webhook': {
        for (const payload of messengerBodies(destination.preset, message.text)) {
          const outcome = await postJson(destination.url, {}, JSON.stringify(payload), ctx.signal)
          if (!outcome.ok) return { ...outcome, error: redactDeliveryError(destination, outcome.error) }
        }
        return { ok: true }
      }
      case 'http': {
        const body = JSON.stringify(httpEnvelope(message))
        const custom = Object.entries(destination.headers)
          .filter(([name]) => name.toLowerCase() !== EVENT_ROUTING_ORIGIN_HEADER)
        const headers: Record<string, string> = {
          ...Object.fromEntries(custom),
          [EVENT_ROUTING_ORIGIN_HEADER]: EVENT_ROUTING_ORIGIN,
          'x-sepilot-delivery-id': ctx.deliveryId,
          'x-sepilot-delivery-attempt': String(ctx.attempt),
        }
        if (destination.secret) {
          headers['x-sepilot-signature'] = `sha256=${createHmac('sha256', destination.secret).update(body).digest('hex')}`
        }
        const outcome = await postJson(destination.url, headers, body, ctx.signal)
        return outcome.ok ? outcome : { ...outcome, error: redactDeliveryError(destination, outcome.error) }
      }
      case 'channel': {
        const channel = deps.channels().find((candidate) => candidate.type === destination.channelType)
        if (!channel) return { ok: false, retryable: true, error: `channel ${destination.channelType} unavailable` }
        try {
          await channel.sendMessage({ id: destination.channelTarget, type: 'channel' }, { text: message.text, format: 'markdown' })
          return { ok: true }
        } catch (error) {
          return outcomeFromError(error)
        }
      }
      case 'notification': {
        try {
          notify({ title: message.title, body: message.text, topic: destination.topic ?? 'events' })
          return { ok: true }
        } catch (error) {
          return outcomeFromError(error)
        }
      }
    }
  }
}
