import type { IChannel, ChannelType, ChannelTarget, ChannelMessage, ChannelStatus, IncomingMessage, Disposable } from '@sepilotd/core'
import { createPublicKey, verify as verifySignature } from 'node:crypto'
import { splitChannelMessage } from './message-chunker.js'

const DISCORD_MESSAGE_CHAR_LIMIT = 1900

export interface DiscordChannelConfig {
  botToken: string
  applicationId: string
  publicKey: string
  allowedGuilds?: string[]
  allowedChannels?: string[]
  allowedUsers?: string[]
}

export interface DiscordInteractionResponse {
  type: number
}

const DISCORD_INTERACTION_TTL_MS = 14 * 60 * 1000

export class DiscordChannel implements IChannel {
  readonly id = 'discord'
  readonly type: ChannelType = 'discord'
  private config: DiscordChannelConfig
  private status: ChannelStatus = 'disconnected'
  private handlers: Array<(msg: IncomingMessage) => Promise<void>> = []
  private pendingInteractions = new Map<string, { token: string; createdAt: number }>()

  constructor(config: DiscordChannelConfig) {
    this.config = config
  }

  private rememberInteraction(channelId: string, token: string): void {
    const now = Date.now()
    for (const [key, entry] of this.pendingInteractions) {
      if (now - entry.createdAt > DISCORD_INTERACTION_TTL_MS) {
        this.pendingInteractions.delete(key)
      }
    }
    this.pendingInteractions.set(channelId, { token, createdAt: now })
  }

  private takeFreshInteractionToken(channelId: string): string | null {
    const entry = this.pendingInteractions.get(channelId)
    if (!entry) return null
    this.pendingInteractions.delete(channelId)
    if (Date.now() - entry.createdAt > DISCORD_INTERACTION_TTL_MS) return null
    return entry.token
  }

  private async resolveInteractionRejection(token: string): Promise<void> {
    try {
      await this.editInteractionOriginal(
        token,
        'You are not authorized to use this command here.',
      )
    } catch {}
  }

  private async editInteractionOriginal(token: string, text: string): Promise<void> {
    const chunks = splitChannelMessage(text, { limit: DISCORD_MESSAGE_CHAR_LIMIT })
    const base = `https://discord.com/api/v10/webhooks/${this.config.applicationId}/${token}`
    const [first, ...rest] = chunks.length > 0 ? chunks : ['']
    const patch = await fetch(`${base}/messages/@original`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: first }),
    })
    if (!patch.ok) {
      const body = await patch.text().catch(() => '')
      throw new Error(`Discord API error: ${patch.status}${body ? ` ${body}` : ''}`)
    }
    for (const content of rest) {
      const followup = await fetch(base, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      })
      if (!followup.ok) {
        const body = await followup.text().catch(() => '')
        throw new Error(`Discord API error: ${followup.status}${body ? ` ${body}` : ''}`)
      }
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
    const interactionToken = this.takeFreshInteractionToken(target.id)
    if (interactionToken) {
      await this.editInteractionOriginal(interactionToken, msg.text)
      return
    }

    for (const content of splitChannelMessage(msg.text, { limit: DISCORD_MESSAGE_CHAR_LIMIT })) {
      const response = await fetch(`https://discord.com/api/v10/channels/${target.id}/messages`, {
        method: 'POST',
        headers: {
          'Authorization': `Bot ${this.config.botToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content }),
      })
      if (!response.ok) {
        const body = await response.text().catch(() => '')
        throw new Error(`Discord API error: ${response.status}${body ? ` ${body}` : ''}`)
      }
    }
  }

  canVerifyInteractionSignature(): boolean {
    return typeof this.config.publicKey === 'string'
      && /^[0-9a-f]{64}$/i.test(this.config.publicKey)
  }

  verifyInteractionSignature(
    rawBody: string,
    signature: string,
    timestamp: string,
    maxSkewSeconds = 300,
  ): boolean {
    try {
      if (
        !this.canVerifyInteractionSignature()
        || !rawBody
        || !signature
        || !timestamp
      ) {
        return false
      }

      const requestTimestamp = Number.parseInt(timestamp, 10)
      if (!Number.isFinite(requestTimestamp)) {
        return false
      }
      const nowSeconds = Math.floor(Date.now() / 1000)
      if (Math.abs(nowSeconds - requestTimestamp) > maxSkewSeconds) {
        return false
      }

      const publicKeyBytes = Buffer.from(this.config.publicKey, 'hex')
      if (publicKeyBytes.length !== 32) {
        return false
      }

      const publicKeyDerPrefix = Buffer.from('302a300506032b6570032100', 'hex')
      const publicKey = createPublicKey({
        key: Buffer.concat([publicKeyDerPrefix, publicKeyBytes]),
        format: 'der',
        type: 'spki',
      })

      return verifySignature(
        null,
        Buffer.from(`${timestamp}${rawBody}`, 'utf8'),
        publicKey,
        Buffer.from(signature, 'hex'),
      )
    } catch {
      return false
    }
  }

  async handleInteraction(body: Record<string, unknown>): Promise<DiscordInteractionResponse> {
    if (body.type === 1) return { type: 1 }

    if (body.type === 2) {
      const data = body.data as Record<string, unknown> | undefined
      const options = data?.options as Array<{ value: string }> | undefined
      const member = body.member as Record<string, unknown> | undefined
      const memberUser = member?.user as Record<string, unknown> | undefined
      const user = body.user as Record<string, unknown> | undefined
      const guildId = body.guild_id as string | undefined
      const channelId = (body.channel_id as string) ?? ''
      const userId = (memberUser?.id as string) ?? (user?.id as string) ?? ''
      const interactionToken = typeof body.token === 'string' ? body.token : undefined

      if (this.config.allowedGuilds?.length && (!guildId || !this.config.allowedGuilds.includes(guildId))) {
        if (interactionToken) await this.resolveInteractionRejection(interactionToken)
        return { type: 5 }
      }
      if (this.config.allowedChannels?.length && !this.config.allowedChannels.includes(channelId)) {
        if (interactionToken) await this.resolveInteractionRejection(interactionToken)
        return { type: 5 }
      }
      if (this.config.allowedUsers?.length && !this.config.allowedUsers.includes(userId)) {
        if (interactionToken) await this.resolveInteractionRejection(interactionToken)
        return { type: 5 }
      }

      if (interactionToken && channelId) {
        this.rememberInteraction(channelId, interactionToken)
      }

      const msg: IncomingMessage = {
        channelType: 'discord',
        channelId,
        messageId: body.id as string,
        text: options?.map(o => o.value).join(' ') ?? '',
        sender: {
          id: userId,
          name: (memberUser?.username as string) ?? (user?.username as string) ?? userId,
          type: 'user',
        },
        timestamp: new Date().toISOString(),
        raw: {
          guildId,
          isDirectMessage: !guildId,
        },
      }
      for (const h of this.handlers) {
        try { await h(msg) } catch {}
      }
      return { type: 5 }
    }

    return { type: 1 }
  }
}
