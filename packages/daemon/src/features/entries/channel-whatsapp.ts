import { WhatsAppChannel } from '../../channels/whatsapp.js'
import type { ChannelFactoryRegistry } from '../../server/runtime/channels.js'
import { readRateLimitPerMinute } from './channel-shared.js'

export function registerChannelFactory(registry: ChannelFactoryRegistry): void {
  registry.register('whatsapp', (channel) => {
    const channelConfig = channel.config as Record<string, unknown> | undefined
    return typeof channelConfig?.phoneNumberId === 'string' &&
      typeof channelConfig?.accessToken === 'string' &&
      typeof channelConfig?.verifyToken === 'string' &&
      typeof channelConfig?.appSecret === 'string'
      ? {
          channel: new WhatsAppChannel({
            phoneNumberId: channelConfig.phoneNumberId,
            accessToken: channelConfig.accessToken,
            appSecret: channelConfig.appSecret,
            verifyToken: channelConfig.verifyToken,
            allowedNumbers: Array.isArray(channelConfig?.allowedNumbers)
              ? channelConfig.allowedNumbers.filter(
                  (value): value is string => typeof value === 'string',
                )
              : [],
            rateLimitPerMinute: readRateLimitPerMinute(channelConfig),
          }),
        }
      : { skipReason: 'phoneNumberId, accessToken, verifyToken, and appSecret are required' }
  })
}
