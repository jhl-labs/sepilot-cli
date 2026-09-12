import type { IncomingMessage } from '@sepilotd/core'
import type { ChannelPipelineCapabilities } from '../server/runtime/capabilities.js'

export interface ChannelAccessDecision {
  allowed: boolean
  autonomy?: ChannelPipelineCapabilities['autonomy']
  reason?: 'acl_denied'
}

export class ChannelAccessController {
  constructor(private readonly runtime: ChannelPipelineCapabilities) {}

  evaluate(msg: IncomingMessage): ChannelAccessDecision {
    if (!this.runtime.channelAcl.isAllowed(msg.channelType, msg.sender.id)) {
      return {
        allowed: false,
        reason: 'acl_denied',
      }
    }

    return {
      allowed: true,
      autonomy: this.runtime.channelAcl.getEffectiveAutonomy(
        msg.channelType,
        this.runtime.autonomy,
      ),
    }
  }
}
