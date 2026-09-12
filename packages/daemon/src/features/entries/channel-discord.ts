import { DiscordChannel } from '../../channels/discord.js'
import type { ChannelFactoryRegistry } from '../../server/runtime/channels.js'

export function registerChannelFactory(registry: ChannelFactoryRegistry): void {
  registry.register('discord', (channel) => {
    const channelConfig = channel.config as Record<string, unknown> | undefined
    return typeof channelConfig?.botToken === 'string' &&
      typeof channelConfig?.applicationId === 'string' &&
      typeof channelConfig?.publicKey === 'string' &&
      /^[0-9a-f]{64}$/i.test(channelConfig.publicKey)
      ? {
          channel: new DiscordChannel({
            botToken: channelConfig.botToken,
            applicationId: channelConfig.applicationId,
            publicKey: channelConfig.publicKey,
            allowedGuilds: Array.isArray(channelConfig?.allowedGuilds)
              ? channelConfig.allowedGuilds.filter(
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
          }),
        }
      : { skipReason: 'botToken, applicationId, and a valid publicKey are required' }
  })
}
