import type { IChannel, ChannelType, ChannelTarget, ChannelMessage, ChannelStatus, IncomingMessage, Disposable } from '@sepilotd/core'

export class WebchatChannel implements IChannel {
  readonly id = 'webchat'
  readonly type: ChannelType = 'webchat'
  private status: ChannelStatus = 'disconnected'
  private handlers: Array<(msg: IncomingMessage) => Promise<void>> = []
  private pendingResponses = new Map<string, (msg: ChannelMessage) => void>()

  async start(): Promise<void> {
    this.status = 'connected'
  }

  async stop(): Promise<void> {
    this.status = 'disconnected'
    this.pendingResponses.clear()
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
    const resolver = this.pendingResponses.get(target.id)
    if (resolver) {
      resolver(msg)
      this.pendingResponses.delete(target.id)
    }
  }

  /** Inject a message from the web UI */
  async injectMessage(msg: IncomingMessage): Promise<void> {
    for (const h of this.handlers) {
      try { await h(msg) } catch { /* handler errors are silently ignored */ }
    }
  }

  /** Wait for a response to a specific message */
  waitForResponse(messageId: string, timeoutMs?: number): Promise<ChannelMessage> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingResponses.delete(messageId)
        reject(new Error('Response timeout'))
      }, timeoutMs ?? 60000)

      this.pendingResponses.set(messageId, (msg) => {
        clearTimeout(timer)
        resolve(msg)
      })
    })
  }
}
