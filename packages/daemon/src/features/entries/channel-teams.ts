import { TeamsChannel } from '../../channels/teams.js'
import type { ChannelFactoryRegistry } from '../../server/runtime/channels.js'
import { readRateLimitPerMinute } from './channel-shared.js'

export function registerChannelFactory(registry: ChannelFactoryRegistry): void {
  registry.register('teams', (channel) => {
    const channelConfig = channel.config as Record<string, unknown> | undefined
    return typeof channelConfig?.appId === 'string' &&
      typeof channelConfig?.appPassword === 'string'
      ? {
          channel: new TeamsChannel({
            appId: channelConfig.appId,
            appPassword: channelConfig.appPassword,
            allowedTenants: Array.isArray(channelConfig?.allowedTenants)
              ? channelConfig.allowedTenants.filter(
                  (value): value is string => typeof value === 'string',
                )
              : [],
            rateLimitPerMinute: readRateLimitPerMinute(channelConfig),
          }),
        }
      : { skipReason: 'appId and appPassword are required' }
  })
}
