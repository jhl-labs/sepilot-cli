import { createHmac, timingSafeEqual } from 'node:crypto'
import type { IChannel, ChannelType, ChannelTarget, ChannelMessage, ChannelStatus, IncomingMessage, Disposable } from '@sepilotd/core'
import { createLogger } from '../logger.js'
import { UNSUPPORTED_MEDIA_MESSAGE } from './media-ack.js'

const log = createLogger('line')

export interface LINEChannelConfig {
  /** LINE Channel Access Token (long-lived) */
  channelAccessToken: string
  /** LINE Channel Secret (for signature verification) */
  channelSecret: string
  /** Allowed user IDs (empty = allow all) */
  allowedUsers?: string[]
  rateLimitPerMinute?: number
}

/**
 * LINE Messaging API channel.
 *
 * Uses the LINE Messaging API directly via fetch (no SDK dependency).
 * Receives messages via webhook (POST /webhooks/line).
 * Sends replies via the LINE Reply/Push API.
 */
export class LINEChannel implements IChannel {
  readonly id = 'line'
  readonly type: ChannelType = 'line'
  private config: LINEChannelConfig
  private status: ChannelStatus = 'disconnected'
  private handlers: Array<(msg: IncomingMessage) => Promise<void>> = []
  private allowedUsers: Set<string>
  private messageCount = new Map<string, { count: number; resetAt: number }>()

  constructor(config: LINEChannelConfig) {
    this.config = config
    this.allowedUsers = new Set(config.allowedUsers ?? [])
  }

  async start(): Promise<void> {
    this.status = 'connected'
    log.info('LINE channel started (webhook mode)')
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
        const idx = this.handlers.indexOf(handler)
        if (idx >= 0) this.handlers.splice(idx, 1)
      },
    }
  }

  async sendMessage(target: ChannelTarget, msg: ChannelMessage): Promise<void> {
    const url = 'https://api.line.me/v2/bot/message/push'
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.config.channelAccessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        to: target.id,
        messages: [{ type: 'text', text: msg.text }],
      }),
    })

    if (!response.ok) {
      const error = await response.text()
      log.error('LINE send failed', { error })
      throw new Error(`LINE API error: ${response.status}`)
    }
  }

  canVerifySignature(): boolean {
    return typeof this.config.channelSecret === 'string'
      && this.config.channelSecret.trim().length > 0
  }

  /** Verify LINE webhook signature */
  verifySignature(body: string, signature: string): boolean {
    if (!this.canVerifySignature()) {
      return false
    }
    const hash = createHmac('SHA256', this.config.channelSecret)
      .update(body)
      .digest('base64')
    // timingSafeEqual on equal-length Buffers — string `===` leaks the
    // common-prefix length under a remote-timing attack.
    const expected = Buffer.from(hash)
    const provided = Buffer.from(signature)
    if (expected.length !== provided.length) return false
    return timingSafeEqual(expected, provided)
  }

  /** Handle incoming LINE webhook event */
  async handleWebhook(body: LINEWebhookBody, rawBody: string, signature: string): Promise<boolean> {
    if (!this.verifySignature(rawBody, signature)) {
      log.warn('LINE webhook signature verification failed')
      return false
    }

    for (const event of body.events ?? []) {
      if (event.type !== 'message') continue

      const userId = event.source?.userId ?? ''

      // Non-text inbound (image / audio / video / file / location / sticker)
      // can't be processed yet. Ack via the reply token instead of silently
      // dropping it.
      if (event.message?.type !== 'text') {
        if (this.allowedUsers.size > 0 && !this.allowedUsers.has(userId)) continue
        if (!this.checkRateLimit(userId)) continue
        if (event.replyToken) {
          try {
            await this.reply(event.replyToken, UNSUPPORTED_MEDIA_MESSAGE)
          } catch { /* best-effort ack */ }
        }
        continue
      }

      // Check allowlist
      if (this.allowedUsers.size > 0 && !this.allowedUsers.has(userId)) {
        log.warn('LINE message from non-allowed user', { userId })
        continue
      }

      // Rate limit
      if (!this.checkRateLimit(userId)) continue

      // Get user profile for display name
      let displayName = userId
      try {
        displayName = await this.getUserName(userId)
      } catch {
        // Use userId as fallback
      }

      const incoming: IncomingMessage = {
        channelType: 'line',
        channelId: event.source?.groupId ?? event.source?.roomId ?? userId,
        messageId: event.message.id,
        sender: {
          id: userId,
          name: displayName,
          type: 'user',
        },
        text: event.message.text ?? '',
        timestamp: new Date(event.timestamp).toISOString(),
        raw: {
          replyToken: event.replyToken,
          sourceType: event.source?.type,
        },
      }

      for (const handler of this.handlers) {
        try {
          await handler(incoming)
        } catch (err) {
          log.error('LINE handler error', { error: String(err) })
        }
      }
    }

    return true
  }

  /** Reply using the reply token (more cost-efficient than push) */
  async reply(replyToken: string, text: string): Promise<void> {
    const url = 'https://api.line.me/v2/bot/message/reply'
    await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.config.channelAccessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        replyToken,
        messages: [{ type: 'text', text }],
      }),
    })
  }

  private async getUserName(userId: string): Promise<string> {
    const response = await fetch(`https://api.line.me/v2/bot/profile/${encodeURIComponent(userId)}`, {
      headers: { 'Authorization': `Bearer ${this.config.channelAccessToken}` },
    })
    if (!response.ok) return userId
    const data = await response.json() as { displayName: string }
    return data.displayName
  }

  private checkRateLimit(userId: string): boolean {
    const limit = this.config.rateLimitPerMinute ?? 30
    const now = Date.now()
    const entry = this.messageCount.get(userId)

    if (!entry || now > entry.resetAt) {
      this.messageCount.set(userId, { count: 1, resetAt: now + 60000 })
      return true
    }

    if (entry.count >= limit) return false
    entry.count++
    return true
  }
}

interface LINEWebhookBody {
  events?: Array<{
    type: string
    replyToken?: string
    timestamp: number
    source?: {
      type: string
      userId?: string
      groupId?: string
      roomId?: string
    }
    message?: {
      id: string
      type: string
      text?: string
    }
  }>
}
