import { MattermostChannel } from '../../channels/mattermost.js'
import type { ChannelFactoryRegistry } from '../../server/runtime/channels.js'

export function registerChannelFactory(registry: ChannelFactoryRegistry): void {
  registry.register('mattermost', (channel) => {
    const channelConfig = channel.config as Record<string, unknown> | undefined
    return typeof channelConfig?.serverUrl === 'string' &&
      typeof channelConfig?.botToken === 'string' &&
      typeof channelConfig?.webhookToken === 'string'
      ? {
          channel: new MattermostChannel({
            serverUrl: channelConfig.serverUrl,
            botToken: channelConfig.botToken,
            webhookToken: channelConfig.webhookToken,
            allowedTeams: Array.isArray(channelConfig?.allowedTeams)
              ? channelConfig.allowedTeams.filter(
                  (value): value is string => typeof value === 'string',
                )
              : [],
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
            botUserId: typeof channelConfig?.botUserId === 'string'
              ? channelConfig.botUserId
              : undefined,
            botUsername: typeof channelConfig?.botUsername === 'string'
              ? channelConfig.botUsername
              : undefined,
          }),
        }
      : { skipReason: 'serverUrl, botToken, and webhookToken are required' }
  })
}
