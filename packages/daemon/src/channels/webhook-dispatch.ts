import { createHash } from 'node:crypto'
import type { IChannel } from '@sepilotd/core'
import { createLogger } from '../logger.js'
import {
  createChannelIngressQueue,
  type ChannelIngressInput,
  type ChannelIngressQueue,
  type ChannelIngressRecord,
} from './ingress-queue.js'
import type { SlackChannel } from './slack.js'
import type { DiscordChannel } from './discord.js'
import type { LINEChannel } from './line.js'
import type { TeamsChannel } from './teams.js'
import type { WhatsAppChannel } from './whatsapp.js'
import type { MattermostChannel } from './mattermost.js'

const log = createLogger('channel:webhook-dispatch')
const dispatchers = new WeakMap<object, ChannelWebhookDispatcher>()

export interface ChannelWebhookRuntime {
  channels: IChannel[]
}

export interface ChannelWebhookDispatcher {
  queue: ChannelIngressQueue
  enqueue(input: ChannelIngressInput): ChannelIngressRecord
  drainOnce(limit?: number): Promise<number>
  start(): void
  stop(): void
}

function stableHash(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex')
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

export function webhookIdempotencyKey(channelType: string, body: Record<string, unknown>): string {
  if (channelType === 'slack') {
    const event = body.event && typeof body.event === 'object' && !Array.isArray(body.event)
      ? body.event as Record<string, unknown>
      : {}
    return [
      'slack',
      stringValue(body.event_id) ?? stringValue(event.client_msg_id) ?? stringValue(event.ts),
      stringValue(event.channel),
      stringValue(event.user),
    ].filter(Boolean).join(':') || `slack:${stableHash(body)}`
  }
  if (channelType === 'discord') {
    return `discord:${stringValue(body.id) ?? stableHash(body)}`
  }
  if (channelType === 'line') {
    const ids = arrayValue(body.events)
      .map((event) => event && typeof event === 'object' && !Array.isArray(event)
        ? stringValue((event as Record<string, unknown>).message && typeof (event as Record<string, unknown>).message === 'object'
          ? ((event as Record<string, unknown>).message as Record<string, unknown>).id
          : undefined) ?? stringValue((event as Record<string, unknown>).replyToken)
        : undefined)
      .filter((id): id is string => Boolean(id))
    return ids.length ? `line:${ids.join(':')}` : `line:${stableHash(body)}`
  }
  if (channelType === 'teams') {
    return `teams:${stringValue(body.id) ?? stableHash(body)}`
  }
  if (channelType === 'whatsapp') {
    const ids: string[] = []
    for (const entry of arrayValue(body.entry)) {
      const changes = entry && typeof entry === 'object' && !Array.isArray(entry)
        ? arrayValue((entry as Record<string, unknown>).changes)
        : []
      for (const change of changes) {
        const value = change && typeof change === 'object' && !Array.isArray(change)
          ? (change as Record<string, unknown>).value
          : undefined
        const messages = value && typeof value === 'object' && !Array.isArray(value)
          ? arrayValue((value as Record<string, unknown>).messages)
          : []
        for (const msg of messages) {
          if (msg && typeof msg === 'object' && !Array.isArray(msg)) {
            const id = stringValue((msg as Record<string, unknown>).id)
            if (id) ids.push(id)
          }
        }
      }
    }
    return ids.length ? `whatsapp:${ids.join(':')}` : `whatsapp:${stableHash(body)}`
  }
  if (channelType === 'mattermost') {
    return [
      'mattermost',
      stringValue(body.post_id) ?? stringValue(body.trigger_id),
      stringValue(body.channel_id),
      stringValue(body.user_id),
    ].filter(Boolean).join(':') || `mattermost:${stableHash(body)}`
  }
  return `${channelType}:${stableHash(body)}`
}

function channelFor(runtime: ChannelWebhookRuntime, channelType: string): IChannel {
  const channel = runtime.channels.find((candidate) => candidate.type === channelType)
  if (!channel) throw new Error(`channel ${channelType} unavailable`)
  return channel
}

async function processRecord(runtime: ChannelWebhookRuntime, record: ChannelIngressRecord): Promise<void> {
  const payload = record.payload
  const body = payload.body && typeof payload.body === 'object' && !Array.isArray(payload.body)
    ? payload.body as Record<string, unknown>
    : {}

  if (record.kind === 'slack.event') {
    await (channelFor(runtime, 'slack') as SlackChannel).handleEvent(body)
    return
  }
  if (record.kind === 'discord.interaction') {
    await (channelFor(runtime, 'discord') as DiscordChannel).handleInteraction(body)
    return
  }
  if (record.kind === 'line.webhook') {
    await (channelFor(runtime, 'line') as LINEChannel).handleWebhook(
      body as never,
      stringValue(payload.rawBody) ?? JSON.stringify(body),
      stringValue(payload.signature) ?? '',
    )
    return
  }
  if (record.kind === 'teams.activity') {
    await (channelFor(runtime, 'teams') as TeamsChannel).handleActivity(body as never)
    return
  }
  if (record.kind === 'whatsapp.webhook') {
    await (channelFor(runtime, 'whatsapp') as WhatsAppChannel).handleWebhook(body as never)
    return
  }
  if (record.kind === 'mattermost.webhook') {
    await (channelFor(runtime, 'mattermost') as MattermostChannel).handleWebhook(body)
    return
  }
}

export function getChannelWebhookDispatcher(runtime: ChannelWebhookRuntime): ChannelWebhookDispatcher {
  const existing = dispatchers.get(runtime)
  if (existing) return existing

  const queue = createChannelIngressQueue()
  let timer: NodeJS.Timeout | null = null
  let draining = false
  let drainScheduled = false

  const drainOnce = async (limit = 20): Promise<number> => {
    if (draining) return 0
    // The 5s interval is unref'd but outlives the runtime in long-lived
    // worker processes (tests, embedded daemons): once the ingress database
    // has been closed by shutdown, a late tick must stop the dispatcher
    // instead of throwing "database connection is not open" as an unhandled
    // timer error.
    if (!queue.isOpen()) {
      dispatcher.stop()
      return 0
    }
    draining = true
    try {
      return await queue.drainDue((record) => processRecord(runtime, record), limit)
    } finally {
      draining = false
    }
  }

  const dispatcher: ChannelWebhookDispatcher = {
    queue,
    enqueue(input) {
      const record = queue.enqueue(input)
      if (!drainScheduled && record.status !== 'done') {
        drainScheduled = true
        setImmediate(() => {
          void drainOnce()
            .catch((err) => log.warn('channel webhook drain failed', { error: String(err) }))
            .finally(() => { drainScheduled = false })
        })
      }
      return record
    },
    drainOnce,
    start() {
      if (timer) return
      void drainOnce()
      timer = setInterval(() => { void drainOnce() }, 5_000)
      timer.unref?.()
    },
    stop() {
      if (timer) clearInterval(timer)
      timer = null
    },
  }
  dispatchers.set(runtime, dispatcher)
  return dispatcher
}

export function startChannelWebhookDispatcher(runtime: ChannelWebhookRuntime): void {
  getChannelWebhookDispatcher(runtime).start()
}
