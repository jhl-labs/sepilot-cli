import type { ChannelTarget, IncomingMessage } from '@sepilotd/core'

export interface ChannelMessageNormalizerOptions {
  sharedGroupContext?: boolean | (() => boolean)
}

export interface NormalizedIncomingChannelMessage {
  message: IncomingMessage
  replayKey?: string
  sessionKey?: string
  replyTarget?: ChannelTarget
  replyToken?: string
  receiveOnly: boolean
}

export class ChannelMessageNormalizer {
  constructor(private readonly options: ChannelMessageNormalizerOptions = {}) {}

  normalize(message: IncomingMessage): NormalizedIncomingChannelMessage {
    const serviceUrl = this.readRawString(message, 'serviceUrl')
    const conversationId = this.readRawString(message, 'conversationId')
    const replyToken = this.readRawString(message, 'replyToken')

    return {
      message,
      replayKey: this.buildReplayKey(message, serviceUrl, conversationId),
      sessionKey: this.buildSessionKey(message, serviceUrl, conversationId),
      replyTarget: this.buildReplyTarget(message, serviceUrl, conversationId),
      replyToken,
      receiveOnly: message.channelType === 'webhook',
    }
  }

  private buildReplyTarget(
    message: IncomingMessage,
    serviceUrl?: string,
    conversationId?: string,
  ): ChannelTarget | undefined {
    switch (message.channelType) {
      case 'github-issue':
      case 'webchat':
        return { id: message.messageId, type: 'channel' }
      case 'teams':
        if (serviceUrl && conversationId) {
          return {
            id: `${serviceUrl}|${conversationId}`,
            type: 'channel',
          }
        }
        return { id: message.channelId, type: 'channel' }
      case 'webhook':
        return undefined
      default:
        return { id: message.channelId, type: 'channel' }
    }
  }

  private buildReplayKey(
    message: IncomingMessage,
    serviceUrl?: string,
    conversationId?: string,
  ): string | undefined {
    if (!this.isReplayProtected(message)) {
      return undefined
    }

    const messageKey = message.messageId || message.replyTo || message.timestamp
    if (!messageKey) {
      return undefined
    }

    if (message.channelType === 'teams') {
      return [
        message.channelType,
        serviceUrl ?? 'unknown',
        conversationId ?? message.channelId,
        messageKey,
      ].join(':')
    }

    return `${message.channelType}:${message.channelId}:${messageKey}`
  }

  private buildSessionKey(
    message: IncomingMessage,
    serviceUrl?: string,
    conversationId?: string,
  ): string | undefined {
    let baseKey: string | undefined
    switch (message.channelType) {
      case 'webhook':
        return undefined
      case 'github-issue':
        baseKey = `${message.channelType}:${message.channelId}:${message.messageId}`
        break
      case 'teams':
        if (serviceUrl && conversationId) {
          baseKey = `${message.channelType}:${serviceUrl}:${conversationId}`
          break
        }
        baseKey = `${message.channelType}:${message.channelId}`
        break
      default:
        baseKey = `${message.channelType}:${message.channelId}`
    }

    if (
      !baseKey
      || this.sharedGroupContextEnabled()
      || !this.isGroupConversation(message)
    ) {
      return baseKey
    }

    return `${baseKey}:sender:${message.sender.id}`
  }

  private isReplayProtected(message: IncomingMessage): boolean {
    switch (message.channelType) {
      case 'slack':
      case 'discord':
      case 'mattermost':
      case 'webhook':
      case 'whatsapp':
      case 'teams':
      case 'line':
        return true
      default:
        return false
    }
  }

  private readRawString(
    message: IncomingMessage,
    key: string,
  ): string | undefined {
    const value = message.raw?.[key]
    return typeof value === 'string' ? value : undefined
  }

  private readRawBoolean(
    message: IncomingMessage,
    key: string,
  ): boolean | undefined {
    const value = message.raw?.[key]
    return typeof value === 'boolean' ? value : undefined
  }

  private sharedGroupContextEnabled(): boolean {
    const configured = this.options.sharedGroupContext
    return typeof configured === 'function' ? configured() : configured === true
  }

  private isGroupConversation(message: IncomingMessage): boolean {
    switch (message.channelType) {
      case 'telegram': {
        const chatType = this.readRawString(message, 'chatType')?.toLowerCase()
        return chatType === 'group' || chatType === 'supergroup' || chatType === 'channel'
      }
      case 'slack': {
        const conversationType = this.readRawString(message, 'conversationType')
          ?? this.readRawString(message, 'channelType')
        if (conversationType) {
          const normalized = conversationType.toLowerCase()
          return normalized === 'channel' || normalized === 'group' || normalized === 'mpim'
        }
        if (message.channelId.startsWith('D')) return false
        return message.channelId.startsWith('C') || message.channelId.startsWith('G')
      }
      case 'discord': {
        const direct = this.readRawBoolean(message, 'isDirectMessage')
        if (direct !== undefined) return !direct
        return Boolean(this.readRawString(message, 'guildId'))
      }
      case 'line': {
        const sourceType = this.readRawString(message, 'sourceType')?.toLowerCase()
        return sourceType === 'group' || sourceType === 'room'
      }
      case 'teams': {
        const group = this.readRawBoolean(message, 'isGroup')
        if (group !== undefined) return group
        const conversationType = this.readRawString(message, 'conversationType')?.toLowerCase()
        return conversationType === 'channel' || conversationType === 'groupchat'
      }
      case 'mattermost': {
        const channelType = this.readRawString(message, 'channel_type')
          ?? this.readRawString(message, 'channelType')
        if (!channelType) return false
        const normalized = channelType.toLowerCase()
        return normalized === 'o' || normalized === 'p' || normalized === 'channel' || normalized === 'private'
      }
      default:
        return false
    }
  }
}
