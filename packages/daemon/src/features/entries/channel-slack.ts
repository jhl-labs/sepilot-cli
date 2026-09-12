import { SlackChannel } from '../../channels/slack.js'
import type { ChannelFactoryRegistry } from '../../server/runtime/channels.js'

export function registerChannelFactory(registry: ChannelFactoryRegistry): void {
  registry.register('slack', (channel) => {
    const channelConfig = channel.config as Record<string, unknown> | undefined
    return typeof channelConfig?.botToken === 'string' &&
      typeof channelConfig?.signingSecret === 'string'
      ? {
          channel: new SlackChannel({
            botToken: channelConfig.botToken,
            signingSecret: channelConfig.signingSecret,
            allowedChannels: Array.isArray(channelConfig?.allowedChannels)
              ? channelConfig.allowedChannels.filter(
                  (value): value is string => typeof value === 'string',
                )
              : [],
            allowedUsers: Array.isArray(channelConfig?.allowedUsers)
              ? channelConfig.allowedUsers.filter(
                  (value): value is string => typeof value === 'string',
                )
              : [],
          }),
        }
      : { skipReason: 'botToken and signingSecret are required' }
  })
}
