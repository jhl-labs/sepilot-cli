import type {
  ChannelMessage,
  ChannelStatus,
  ChannelTarget,
  ChannelType,
  Disposable,
  IChannel,
  IncomingMessage,
} from '@sepilotd/core'
import { timingSafeEqual } from 'node:crypto'

export interface MattermostChannelConfig {
  serverUrl: string
  botToken: string
  webhookToken: string
  allowedTeams?: string[]
  allowedChannels?: string[]
  allowedUsers?: string[]
  /**
   * The bot's own user id (and/or username). Posts authored by the bot are
   * dropped so the bot cannot trigger itself in a channel it also posts to
   * (infinite self-reply loop).
   */
  botUserId?: string
  botUsername?: string
}

export interface MattermostWebhookResponse {
  response_type: 'ephemeral'
  text: string
}

export class MattermostChannel implements IChannel {
  readonly id = 'mattermost'
  readonly type: ChannelType = 'mattermost'
  private config: MattermostChannelConfig
  private status: ChannelStatus = 'disconnected'
  private handlers: Array<(msg: IncomingMessage) => Promise<void>> = []

  constructor(config: MattermostChannelConfig) {
    this.config = {
      ...config,
      serverUrl: config.serverUrl.replace(/\/+$/, ''),
      botToken: config.botToken.trim(),
      webhookToken: config.webhookToken.trim(),
    }
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
    const body: Record<string, string> = {
      channel_id: target.id,
      message: msg.text,
    }
    const replyTo = msg.replyTo?.trim()
    if (replyTo) {
      body.root_id = replyTo
    }

    const response = await fetch(`${this.config.serverUrl}/api/v4/posts`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.config.botToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })
    if (!response.ok) {
      const detail = await response.text().catch(() => '')
      throw new Error(`Mattermost API error: ${response.status}${detail ? ` ${detail.slice(0, 200)}` : ''}`)
    }
  }

  canVerifyWebhookToken(): boolean {
    return this.config.webhookToken.trim().length > 0
  }

  verifyWebhookToken(body: Record<string, unknown>): boolean {
    if (!this.canVerifyWebhookToken() || typeof body.token !== 'string') {
      return false
    }

    const incoming = Buffer.from(body.token)
    const expected = Buffer.from(this.config.webhookToken)
    return incoming.length === expected.length && timingSafeEqual(incoming, expected)
  }

  async handleWebhook(
    body: Record<string, unknown>,
  ): Promise<MattermostWebhookResponse> {
    const teamId = stringValue(body.team_id)
    const teamName = stringValue(body.team_domain) ?? stringValue(body.team_name)
    const channelId = stringValue(body.channel_id)
    const channelName = stringValue(body.channel_name)
    const userId = stringValue(body.user_id)
    const userName = stringValue(body.user_name) ?? userId
    const postId = stringValue(body.post_id)
    const triggerId = stringValue(body.trigger_id)
    const rootId = stringValue(body.root_id)
    const messageId = postId
      ?? triggerId
      ?? `${channelId ?? 'unknown'}:${Date.now()}`
    const replyTo = rootId ?? postId
    const text = stringValue(body.text) ?? ''

    if (!channelId || !userId) {
      return mattermostAck()
    }
    // Self-loop guard: never process the bot's own posts.
    if (
      (this.config.botUserId && userId === this.config.botUserId)
      || (this.config.botUsername && userName === this.config.botUsername)
    ) {
      return mattermostAck()
    }
    if (!this.isAllowed(this.config.allowedTeams, teamId, teamName)) {
      return mattermostAck()
    }
    if (!this.isAllowed(this.config.allowedChannels, channelId, channelName)) {
      return mattermostAck()
    }
    if (this.config.allowedUsers?.length && !this.config.allowedUsers.includes(userId)) {
      return mattermostAck()
    }

    const msg: IncomingMessage = {
      channelType: 'mattermost',
      channelId,
      messageId,
      text,
      sender: { id: userId, name: userName ?? userId, type: 'user' },
      timestamp: new Date().toISOString(),
      ...(replyTo ? { replyTo } : {}),
      raw: body,
    }

    for (const h of this.handlers) {
      try { await h(msg) } catch { /* handler errors are silently ignored */ }
    }

    return mattermostAck()
  }

  private isAllowed(
    allowed: string[] | undefined,
    id: string | undefined,
    name: string | undefined,
  ): boolean {
    if (!allowed?.length) return true
    return Boolean((id && allowed.includes(id)) || (name && allowed.includes(name)))
  }
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

function mattermostAck(): MattermostWebhookResponse {
  return {
    response_type: 'ephemeral',
    text: 'Received. sepilotd will reply in this channel when the run finishes.',
  }
}
