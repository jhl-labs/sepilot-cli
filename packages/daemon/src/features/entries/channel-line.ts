import { LINEChannel } from '../../channels/line.js'
import type { ChannelFactoryRegistry } from '../../server/runtime/channels.js'
import { readRateLimitPerMinute } from './channel-shared.js'

export function registerChannelFactory(registry: ChannelFactoryRegistry): void {
  registry.register('line', (channel) => {
    const channelConfig = channel.config as Record<string, unknown> | undefined
    return typeof channelConfig?.channelAccessToken === 'string' &&
      typeof channelConfig?.channelSecret === 'string'
      ? {
          channel: new LINEChannel({
            channelAccessToken: channelConfig.channelAccessToken,
            channelSecret: channelConfig.channelSecret,
            allowedUsers: Array.isArray(channelConfig?.allowedUsers)
              ? channelConfig.allowedUsers.filter(
                  (value): value is string => typeof value === 'string',
                )
              : [],
            rateLimitPerMinute: readRateLimitPerMinute(channelConfig),
          }),
        }
      : { skipReason: 'channelAccessToken and channelSecret are required' }
  })
}
