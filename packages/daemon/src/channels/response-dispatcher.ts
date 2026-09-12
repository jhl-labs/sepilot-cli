import type { ChannelAttachment, IChannel } from '@sepilotd/core'
import { createLogger } from '../logger.js'
import type { NormalizedIncomingChannelMessage } from './normalizer.js'

const log = createLogger('router')

export interface ChannelResponseDispatchOptions {
  replyToMessageId?: string
  attachments?: ChannelAttachment[]
}

export class ChannelResponseDispatcher {
  async send(
    channel: IChannel,
    normalized: NormalizedIncomingChannelMessage,
    text: string,
    options: ChannelResponseDispatchOptions = {},
  ): Promise<boolean> {
    const { message, receiveOnly, replyTarget, replyToken } = normalized

    if (receiveOnly) {
      log.info('Skipping outbound response for receive-only webhook channel', {
        channelType: message.channelType,
        messageId: message.messageId,
      })
      return false
    }

    if (replyToken && 'reply' in channel && typeof channel.reply === 'function') {
      await channel.reply(replyToken, text)
      return true
    }

    if (!replyTarget) return false

    const replyTo = message.channelType === 'mattermost'
      ? message.replyTo
      : options.replyToMessageId

    await channel.sendMessage(replyTarget, {
      text,
      format: 'markdown',
      replyTo,
      attachments: options.attachments,
    })
    return true
  }
}
