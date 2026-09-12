import type { IChannel, ChannelType, ChannelTarget, ChannelMessage, ChannelStatus, IncomingMessage, Disposable } from '@sepilotd/core'
import { createHmac, timingSafeEqual } from 'node:crypto'
import { splitChannelMessage } from './message-chunker.js'
import { UNSUPPORTED_MEDIA_MESSAGE } from './media-ack.js'

const SLACK_MESSAGE_CHAR_LIMIT = 38000

export interface SlackChannelConfig {
  botToken: string
  signingSecret: string
  allowedChannels?: string[]
  allowedUsers?: string[]
}

export class SlackChannel implements IChannel {
  readonly id = 'slack'
  readonly type: ChannelType = 'slack'
  private config: SlackChannelConfig
  private status: ChannelStatus = 'disconnected'
  private handlers: Array<(msg: IncomingMessage) => Promise<void>> = []

  constructor(config: SlackChannelConfig) {
    this.config = config
  }

  async start(): Promise<void> {
    this.status = 'connected'
  }

  async stop(): Promise<void> {
    this.status = 'disconnected'
  }

  getStatus(): ChannelStatus {
    return this.status
  }

  onMessage(handler: (msg: IncomingMessage) => Promise<void>): Disposable {
    this.handlers.push(handler)
    return {
      dispose: () => {
        const i = this.handlers.indexOf(handler)
        if (i >= 0) this.handlers.splice(i, 1)
      },
    }
  }

  async sendMessage(target: ChannelTarget, msg: ChannelMessage): Promise<void> {
    for (const text of splitChannelMessage(msg.text, { limit: SLACK_MESSAGE_CHAR_LIMIT })) {
      const response = await fetch('https://slack.com/api/chat.postMessage', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.config.botToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ channel: target.id, text }),
      })
      if (!response.ok) {
        const body = await response.text().catch(() => '')
        throw new Error(`Slack API error: ${response.status}${body ? ` ${body}` : ''}`)
      }
      const body = await response.json().catch(() => undefined)
      if (!isSlackOkResponse(body)) {
        const error = readSlackError(body) ?? 'unknown_error'
        throw new Error(`Slack API error: ${error}`)
      }
    }
  }

  /** Process incoming Slack event (called by webhook route) */
  async handleEvent(body: Record<string, unknown>): Promise<void> {
    if (body.type === 'url_verification') return // Challenge handled by route

    const event = body.event as Record<string, unknown> | undefined
    if (event?.type === 'message' && !event.bot_id) {
      // Ignore edited/deleted/system message subtypes: `message_changed` and
      // `message_deleted` nest the real content under `event.message` and would
      // otherwise dispatch with empty text and an undefined sender. `file_share`
      // is the one subtype we still process (handled by the media-ack check).
      const subtype = event.subtype as string | undefined
      if (subtype && subtype !== 'file_share') return

      const channel = event.channel as string
      const user = event.user as string
      const ts = event.ts as string
      const text = (event.text as string) ?? ''
      const conversationType = event.channel_type as string | undefined
      // Preserve thread context so a reply stays in the originating thread
      // instead of collapsing to the channel root.
      const threadTs = typeof event.thread_ts === 'string' ? event.thread_ts : undefined

      if (this.config.allowedChannels?.length && !this.config.allowedChannels.includes(channel)) return
      if (this.config.allowedUsers?.length && !this.config.allowedUsers.includes(user)) return

      // Inbound media (file uploads) can't be processed yet. Ack instead of
      // silently dropping so the sender knows the upload wasn't handled.
      const files = event.files as unknown[] | undefined
      if ((Array.isArray(files) && files.length > 0) && text.trim().length === 0) {
        try {
          await this.sendMessage({ id: channel, type: 'channel' }, { text: UNSUPPORTED_MEDIA_MESSAGE })
        } catch { /* best-effort ack */ }
        return
      }

      const msg: IncomingMessage = {
        channelType: 'slack',
        channelId: channel,
        messageId: ts,
        text,
        sender: { id: user, name: user, type: 'user' },
        timestamp: new Date(parseFloat(ts) * 1000).toISOString(),
        ...(threadTs ? { replyTo: threadTs } : {}),
        raw: {
          conversationType,
          threadTs,
        },
      }
      for (const h of this.handlers) {
        try { await h(msg) } catch { /* handler errors are silently ignored */ }
      }
    }
  }

  canVerifySignature(): boolean {
    return typeof this.config.signingSecret === 'string'
      && this.config.signingSecret.trim().length > 0
  }

  verifySignature(
    body: string,
    headers: Record<string, string>,
    maxSkewSeconds = 300,
  ): boolean {
    try {
      const timestamp = headers['x-slack-request-timestamp']
      const signature = headers['x-slack-signature']
      if (!this.canVerifySignature() || !timestamp || !signature) return false

      const requestTimestamp = Number.parseInt(timestamp, 10)
      if (!Number.isFinite(requestTimestamp)) return false
      const nowSeconds = Math.floor(Date.now() / 1000)
      if (Math.abs(nowSeconds - requestTimestamp) > maxSkewSeconds) {
        return false
      }

      const baseString = `v0:${timestamp}:${body}`
      const expected = 'v0=' + createHmac('sha256', this.config.signingSecret).update(baseString).digest('hex')
      return timingSafeEqual(Buffer.from(expected), Buffer.from(signature))
    } catch {
      return false
    }
  }
}

function isSlackOkResponse(value: unknown): value is { ok: true } {
  return Boolean(
    value
      && typeof value === 'object'
      && 'ok' in value
      && value.ok === true,
  )
}

function readSlackError(value: unknown): string | undefined {
  if (!value || typeof value !== 'object' || !('error' in value)) {
    return undefined
  }
  return typeof value.error === 'string' ? value.error : undefined
}
