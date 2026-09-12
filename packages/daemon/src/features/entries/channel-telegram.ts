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
            // Incoming voice notes are transcribed with a local whisper
            // binary (never an external API). If whisper isn't installed
            // this rejects with SPEECH_BINARY_MISSING and the channel
            // tells the user how to enable it.
            transcribeAudioFile: async (path: string) =>
              (await transcribeAudio({ audioPath: path })).text,
            // Reply to a voice note with a voice note. piper-backed,
            // local-only; rejects with SPEECH_BINARY_MISSING if
            // SEPILOTD_PIPER_MODEL isn't set, which the channel logs and
            // skips (the text reply was already delivered).
            synthesizeReply: async (text: string) => {
              const r = await synthesizeSpeech({ text })
              return { audioPath: r.audioPath, temporary: r.temporary }
            },
          }),
        }
      : { skipReason: 'botToken is required' }
  })
}
