import type { Disposable } from '../disposable.js'
import type { ChannelType, ChannelTarget, ChannelMessage, ChannelStatus, IncomingMessage, ChannelActivity } from './types.js'

export interface IChannel {
  readonly id: string
  readonly type: ChannelType
  start(): Promise<void>
  stop(): Promise<void>
  getStatus(): ChannelStatus
  onMessage(handler: (msg: IncomingMessage) => Promise<void>): Disposable
  sendMessage(target: ChannelTarget, msg: ChannelMessage): Promise<void>
  sendActivity?(target: ChannelTarget, activity: ChannelActivity): Promise<void>
}
