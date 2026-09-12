import type { IChannel, ChannelType, ChannelTarget, ChannelMessage, ChannelStatus, IncomingMessage, Disposable } from '@sepilotd/core'
import { createHmac, timingSafeEqual } from 'node:crypto'
import { createLogger } from '../logger.js'
import { UNSUPPORTED_MEDIA_MESSAGE } from './media-ack.js'

const log = createLogger('whatsapp')

export interface WhatsAppChannelConfig {
  /** Phone number ID from Meta Business */
  phoneNumberId: string
  /** WhatsApp Business API access token */
  accessToken: string
  /** Meta app secret used for webhook signature verification */
  appSecret?: string
  /** Webhook verify token */
  verifyToken: string
  /** Allowed phone numbers (E.164 format) */
  allowedNumbers?: string[]
  rateLimitPerMinute?: number
}

/**
 * WhatsApp Business API channel.
 *
 * Uses the official WhatsApp Cloud API (Meta Business Platform).
 * Requires:
 * 1. Meta Business account with WhatsApp Business API access
 * 2. Phone number registered with the API
 * 3. Webhook endpoint configured to receive messages
 *
 * Messages are received via webhook (POST /webhooks/whatsapp)
 * and sent via the Cloud API.
 */
export class WhatsAppChannel implements IChannel {
  readonly id = 'whatsapp'
  readonly type: ChannelType = 'whatsapp'
  private config: WhatsAppChannelConfig
  private status: ChannelStatus = 'disconnected'
  private handlers: Array<(msg: IncomingMessage) => Promise<void>> = []
  private allowedNumbers: Set<string>
  private messageCount = new Map<string, { count: number; resetAt: number }>()

  constructor(config: WhatsAppChannelConfig) {
    this.config = config
    this.allowedNumbers = new Set(config.allowedNumbers ?? [])
  }

  async start(): Promise<void> {
    this.status = 'connected'
    log.info('WhatsApp channel started (webhook mode)')
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
    const url = `https://graph.facebook.com/v21.0/${this.config.phoneNumberId}/messages`
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.config.accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messaging_product: 'whatsapp',
        to: target.id,
        type: 'text',
        text: { body: msg.text },
      }),
    })

    if (!response.ok) {
      const error = await response.text()
      log.error('WhatsApp send failed', { error, target: target.id })
      throw new Error(`WhatsApp API error: ${response.status}`)
    }
  }

  /** Verify webhook challenge from Meta */
  verifyWebhook(mode: string, token: string, challenge: string): string | null {
    if (mode !== 'subscribe') return null
    if (typeof this.config.verifyToken !== 'string' || !this.config.verifyToken) {
      return null
    }
    const incoming = Buffer.from(token)
    const expected = Buffer.from(this.config.verifyToken)
    if (incoming.length !== expected.length) return null
    if (!timingSafeEqual(incoming, expected)) return null
    return challenge
  }

  canVerifyWebhookChallenge(): boolean {
    return typeof this.config.verifyToken === 'string'
      && this.config.verifyToken.trim().length > 0
  }

  canVerifySignature(): boolean {
    return typeof this.config.appSecret === 'string' && this.config.appSecret.length > 0
  }

  isWebhookVerificationReady(): boolean {
    return this.canVerifyWebhookChallenge() && this.canVerifySignature()
  }

  verifySignature(rawBody: string, signature: string | undefined): boolean {
    try {
      if (!this.config.appSecret || !rawBody || !signature?.startsWith('sha256=')) {
        return false
      }

      const expected = Buffer.from(
        `sha256=${createHmac('sha256', this.config.appSecret).update(rawBody).digest('hex')}`,
        'utf8',
      )
      const actual = Buffer.from(signature, 'utf8')
      return expected.length === actual.length && timingSafeEqual(expected, actual)
    } catch {
      return false
    }
  }

  /** Handle incoming webhook event from Meta */
  async handleWebhook(body: WhatsAppWebhookBody): Promise<void> {
    const entries = body.entry ?? []

    for (const entry of entries) {
      const changes = entry.changes ?? []
      for (const change of changes) {
        if (change.field !== 'messages') continue
        const messages = change.value?.messages ?? []

        for (const msg of messages) {
          const from = msg.from
          // Check allowlist
          if (this.allowedNumbers.size > 0 && !this.allowedNumbers.has(from)) {
            log.warn('WhatsApp message from non-allowed number', { from })
            continue
          }

          // Rate limit
          if (!this.checkRateLimit(from)) continue

          // Non-text inbound (image / audio / document / location / …) can't
          // be processed yet. Ack instead of silently dropping.
          if (msg.type !== 'text' || !msg.text?.body) {
            try {
              await this.sendMessage({ id: from, type: 'user' }, { text: UNSUPPORTED_MEDIA_MESSAGE })
            } catch { /* best-effort ack */ }
            continue
          }

          const contact = change.value?.contacts?.find((c) => c.wa_id === from)

          const incoming: IncomingMessage = {
            channelType: 'whatsapp',
            channelId: from,
            messageId: msg.id,
            sender: {
              id: from,
              name: contact?.profile?.name ?? from,
              type: 'user',
            },
            text: msg.text.body,
            timestamp: new Date(parseInt(msg.timestamp) * 1000).toISOString(),
          }

          for (const handler of this.handlers) {
            try {
              await handler(incoming)
            } catch (err) {
              log.error('WhatsApp handler error', { error: String(err) })
            }
          }
        }
      }
    }
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

interface WhatsAppWebhookBody {
  object?: string
  entry?: Array<{
    id: string
    changes?: Array<{
      field: string
      value?: {
        messaging_product?: string
        contacts?: Array<{ wa_id: string; profile?: { name: string } }>
        messages?: Array<{
          id: string
          from: string
          timestamp: string
          type: string
          text?: { body: string }
        }>
      }
    }>
  }>
}
