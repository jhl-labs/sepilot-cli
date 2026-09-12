import type { IncomingMessage } from '@sepilotd/core'
import { createLogger } from '../logger.js'
import { triggerDreamingSessionEnd } from '../memory/dreaming.js'
import { deriveChannelScopeTags } from '../memory/scope.js'
import type { ChannelPipelineCapabilities } from '../server/runtime/capabilities.js'
import type { DefaultProvider } from './session-resolver.js'

const log = createLogger('router')

export interface ChannelAgentResponse {
  responseText: string
  responseDelivered: boolean
}

export type ChannelHookEmitter = (
  msg: IncomingMessage,
  data: Record<string, unknown>,
) => Promise<void>

export class ChannelPostProcessor {
  constructor(
    private readonly runtime: ChannelPipelineCapabilities,
    private readonly emitChannelHook: ChannelHookEmitter,
  ) {}

  async complete(
    message: IncomingMessage,
    sessionId: string,
    provider: DefaultProvider,
    response: ChannelAgentResponse,
  ): Promise<void> {
    await this.runtime.auditLogger.log({
      timestamp: new Date().toISOString(),
      event: 'channel.message.processed',
      device: this.runtime.config.device.name,
      session: sessionId,
      channel: message.channelType,
      sender: message.sender.id,
    })

    if (
      message.channelType === 'github-issue'
      && response.responseText
      && response.responseDelivered
    ) {
      try {
        await this.runtime.gatewayClient.updateTicket(message.messageId, {
          status: 'closed',
        })
        log.info(`Auto-closed issue #${message.messageId}`)
      } catch (err) {
        // Auto-close is best-effort (the message was processed
        // successfully), but a silent failure means the operator
        // sees a still-open issue with no idea why the OpenClaw
        // integration didn't close it. Surface the failure so
        // gateway/auth/permission problems are diagnosable.
        log.warn(`auto-close failed for issue #${message.messageId}`, {
          channelType: message.channelType,
          messageId: message.messageId,
          error: err instanceof Error ? err.message : String(err),
        })
      }
    }

    triggerDreamingSessionEnd(
      this.runtime.dreaming,
      sessionId,
      'channel',
      deriveChannelScopeTags({
        channelType: message.channelType,
        channelId: message.channelId,
        senderId: message.sender.id,
        sessionId,
      }),
    )

    await this.emitChannelHook(message, {
      status: 'processed',
      sessionId,
      provider: provider.id,
      responseText: response.responseText,
      responseDelivered: response.responseDelivered,
    })

    log.info(`Done processing ${message.channelType} message`, { session: sessionId })
  }
}
