import { TelegramChannel } from '../../channels/telegram.js'
import { synthesizeSpeech, transcribeAudio } from '../../media/speech.js'
import type { ChannelFactoryRegistry } from '../../server/runtime/channels.js'

export function registerChannelFactory(registry: ChannelFactoryRegistry): void {
  registry.register('telegram', (channel, deps) => {
    const channelConfig = channel.config as Record<string, unknown> | undefined
    return typeof channelConfig?.botToken === 'string'
      ? {
          channel: new TelegramChannel({
            botToken: channelConfig.botToken,
            allowedUsers: Array.isArray(channelConfig?.allowedUsers)
              ? channelConfig.allowedUsers.filter(
                  (value): value is string => typeof value === 'string',
                )
              : [],
            pairingRequired:
              typeof channelConfig?.pairingRequired === 'boolean'
                ? channelConfig.pairingRequired
                : undefined,
            pairingCodeTtl:
              typeof channelConfig?.pairingCodeTtl === 'number'
                ? channelConfig.pairingCodeTtl
                : undefined,
            rateLimitPerMinute:
              typeof channelConfig?.rateLimitPerMinute === 'number'
                ? channelConfig.rateLimitPerMinute
                : undefined,
            onPairedUser: deps.onChannelUserPaired
              ? (userId) => deps.onChannelUserPaired?.('telegram', userId)
              : undefined,
            loadPendingPairing: deps.loadTelegramPendingPairing,
            clearPendingPairing: deps.clearTelegramPendingPairing,
            transcribeAudioFile: async (path: string) =>
              (await transcribeAudio({ audioPath: path })).text,
            synthesizeReply: async (text: string) => {
              const r = await synthesizeSpeech({ text })
              return { audioPath: r.audioPath, temporary: r.temporary }
            },
          }),
        }
      : { skipReason: 'botToken is required' }
  })
}
