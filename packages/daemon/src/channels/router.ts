import type { Disposable, IChannel, IncomingMessage } from '@sepilotd/core'
import { createLogger } from '../logger.js'
import type { ChannelPipelineCapabilities } from '../server/runtime/capabilities.js'
import { ChannelMessagePipeline } from './pipeline.js'

const log = createLogger('router')

export class ChannelRouter {
  private readonly runtime: ChannelPipelineCapabilities
  private readonly pipeline: ChannelMessagePipeline
  private readonly subscriptions = new Map<IChannel, Disposable>()

  constructor(runtime: ChannelPipelineCapabilities) {
    this.runtime = runtime
    this.pipeline = new ChannelMessagePipeline(
      runtime,
      this.emitChannelHook.bind(this),
    )
  }

  private async emitChannelHook(
    msg: IncomingMessage,
    data: Record<string, unknown>,
  ): Promise<void> {
    try {
      await this.runtime.hookRegistry?.trigger({
        event: 'post:channel:msg',
        data: {
          message: {
            channelType: msg.channelType,
            channelId: msg.channelId,
            messageId: msg.messageId,
            senderId: msg.sender.id,
            senderName: msg.sender.name,
            timestamp: msg.timestamp,
            text: msg.text,
          },
          ...data,
        },
      })
    } catch (error) {
      log.warn('Channel hook failed', {
        channelType: msg.channelType,
        messageId: msg.messageId,
        error: String(error),
      })
    }
  }

  private async preChannelMsgAborted(msg: IncomingMessage): Promise<boolean> {
    if (!this.runtime.hookRegistry) return false
    try {
      const result = await this.runtime.hookRegistry.trigger({
        event: 'pre:channel:msg',
        data: {
          message: {
            channelType: msg.channelType,
            channelId: msg.channelId,
            messageId: msg.messageId,
            senderId: msg.sender.id,
            senderName: msg.sender.name,
            timestamp: msg.timestamp,
            text: msg.text,
          },
        },
      })
      return result.action === 'abort'
    } catch (error) {
      log.warn('pre:channel:msg hook failed', {
        channelType: msg.channelType,
        messageId: msg.messageId,
        error: String(error),
      })
      return false
    }
  }

  wireChannel(channel: IChannel): void {
    if (this.subscriptions.has(channel)) {
      return
    }

    const subscription = channel.onMessage(async (msg: IncomingMessage) => {
      log.info(`Message received on ${msg.channelType}`, { sender: msg.sender.id, text: msg.text.slice(0, 80) })
      if (await this.preChannelMsgAborted(msg)) {
        log.info('Channel message blocked by pre:channel:msg hook', {
          channelType: msg.channelType,
          messageId: msg.messageId,
        })
        return
      }
      await this.pipeline.handle(channel, msg)
    })
    this.subscriptions.set(channel, subscription)
  }

  unwireChannel(channel: IChannel): void {
    const subscription = this.subscriptions.get(channel)
    subscription?.dispose()
    this.subscriptions.delete(channel)
  }

  wireAll(): void {
    for (const channel of this.runtime.channels) {
      this.wireChannel(channel)
    }
  }
}
