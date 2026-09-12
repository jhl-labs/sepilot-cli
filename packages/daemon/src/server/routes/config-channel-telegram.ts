import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import {
  clearTelegramPendingPairing,
  removeTelegramAllowedUser,
  saveTelegramPendingPairing,
} from '../runtime/channel-pairing-persistence.js'
import {
  persistRuntimeConfig,
  reconfigureRuntimeChannelType,
} from '../runtime/config-runtime.js'
import { zodRequestValidation } from './utils.js'
import {
  telegramAllowedUserParamsSchema,
  telegramChannelConfigSchema,
  telegramChannelToggleRequestSchema,
  telegramPairingCodeSchema,
  type TelegramAllowedUserParams,
  type TelegramChannelBody,
  type TelegramChannelToggleBody,
} from './config-schema.js'
import {
  getTelegramMutableRuntimeChannel,
  getTelegramPairingRuntimeChannel,
  readConfiguredTelegramChannel,
  replaceConfiguredTelegramChannel,
  restoreTelegramBotToken,
  summarizeTelegramChannel,
} from './config-channels-internals.js'

export function registerConfigChannelTelegramRoutes(app: FastifyInstance): void {
  const runtime = app.runtime

  app.get('/config/channels/telegram', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    return {
      data: summarizeTelegramChannel(runtime),
    }
  })

  app.post<{ Body: TelegramChannelBody }>('/config/channels/telegram', {
    preValidation: zodRequestValidation({
      body: {
        schema: telegramChannelConfigSchema,
        message: 'Invalid Telegram channel request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    await runtime.configMutationService.apply(
      'config.channels.telegram.upsert',
      async () => {
        replaceConfiguredTelegramChannel(
          runtime.config,
          restoreTelegramBotToken(
            readConfiguredTelegramChannel(runtime.config),
            request.body,
          ),
        )
        await clearTelegramPendingPairing(runtime.dataDir)
        await reconfigureRuntimeChannelType(runtime, 'telegram')
        await persistRuntimeConfig(runtime, new Set(['channels']))
      },
    )
    return { data: { updated: ['channels'] } }
  })

  app.delete('/config/channels/telegram', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const outcome = await runtime.configMutationService.apply(
      'config.channels.telegram.delete',
      async () => {
        const existing = readConfiguredTelegramChannel(runtime.config)
        if (!existing) {
          return 'not_found' as const
        }

        runtime.config.channels = runtime.config.channels.filter(
          (channel) => channel.type !== 'telegram',
        )
        await clearTelegramPendingPairing(runtime.dataDir)
        await reconfigureRuntimeChannelType(runtime, 'telegram')
        await persistRuntimeConfig(runtime, new Set(['channels']))
        return 'ok' as const
      },
    )

    if (outcome === 'not_found') {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'Telegram channel not found',
        },
      })
    }
    return { data: { updated: ['channels'] } }
  })

  app.post<{ Body: TelegramChannelToggleBody }>('/config/channels/telegram/enable', {
    preValidation: zodRequestValidation({
      body: {
        schema: telegramChannelToggleRequestSchema,
        message: 'Invalid Telegram channel toggle request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const body = request.body

    const outcome = await runtime.configMutationService.apply(
      'config.channels.telegram.enable',
      async () => {
        const existing = readConfiguredTelegramChannel(runtime.config)
        if (!existing) {
          return 'not_found' as const
        }

        replaceConfiguredTelegramChannel(runtime.config, {
          ...existing,
          enabled: body.enabled,
        })
        if (!body.enabled) {
          await clearTelegramPendingPairing(runtime.dataDir)
        }
        await reconfigureRuntimeChannelType(runtime, 'telegram')
        await persistRuntimeConfig(runtime, new Set(['channels']))
        return 'ok' as const
      },
    )

    if (outcome === 'not_found') {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'Telegram channel not found',
        },
      })
    }
    return { data: { updated: ['channels'] } }
  })

  app.post('/config/channels/telegram/pairing-code', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const configured = readConfiguredTelegramChannel(runtime.config)
    if (!configured) {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'Telegram channel not found',
        },
      })
    }

    if (!configured.enabled) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: 'Telegram channel is disabled',
        },
      })
    }

    if (!configured.pairingRequired) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: 'Telegram channel does not require pairing',
        },
      })
    }

    const channel = getTelegramPairingRuntimeChannel(runtime)
    if (!channel) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Telegram channel is not active',
        },
      })
    }

    const pairingCode = telegramPairingCodeSchema.parse(channel.generatePairingCode())
    await saveTelegramPendingPairing(runtime.dataDir, {
      code: pairingCode.code,
      expiresAt: Date.parse(pairingCode.expiresAt),
    })

    return {
      data: pairingCode,
    }
  })

  app.delete<{ Params: TelegramAllowedUserParams }>(
    '/config/channels/telegram/allowed-users/:userId',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: telegramAllowedUserParamsSchema,
          message: 'Invalid Telegram allowed user params',
        },
      }),
    },
    async (request, reply) => {
      if (!runtime) {
        return reply.status(503).send({
          error: {
            code: 'SERVICE_UNAVAILABLE',
            message: 'Runtime not initialized',
          },
        })
      }

      const params = request.params

      const outcome = await runtime.configMutationService.apply(
        'config.channels.telegram.allowed-users.delete',
        async () => {
          const configured = readConfiguredTelegramChannel(runtime.config)
          if (!configured) {
            return 'channel_not_found' as const
          }

          const changed = removeTelegramAllowedUser(
            runtime.config,
            params.userId,
          )
          if (!changed) {
            return 'user_not_paired' as const
          }

          runtime.channelAcl?.removeAllowedUser('telegram', params.userId)
          getTelegramMutableRuntimeChannel(runtime)?.revokeAllowedUser(
            params.userId,
          )
          await persistRuntimeConfig(runtime, new Set(['channels']))
          return 'ok' as const
        },
      )

      if (outcome === 'channel_not_found') {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: 'Telegram channel not found',
          },
        })
      }
      if (outcome === 'user_not_paired') {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: `Telegram user not paired: ${params.userId}`,
          },
        })
      }
      return { data: { updated: ['channels'] } }
    },
  )
}
