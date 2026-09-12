import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import {
  persistRuntimeConfig,
  reconfigureRuntimeChannelType,
} from '../runtime/config-runtime.js'
import { zodRequestValidation } from './utils.js'
import {
  channelToggleRequestSchema,
  discordChannelConfigSchema,
  type ChannelToggleBody,
  type DiscordChannelBody,
} from './config-schema.js'
import {
  readConfiguredDiscordChannel,
  replaceConfiguredDiscordChannel,
  restoreDiscordChannelSecrets,
  summarizeDiscordChannel,
} from './config-channels-internals.js'

export function registerConfigChannelDiscordRoutes(app: FastifyInstance): void {
  const runtime = app.runtime

  app.get('/config/channels/discord', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    return {
      data: summarizeDiscordChannel(runtime),
    }
  })

  app.post<{ Body: DiscordChannelBody }>('/config/channels/discord', {
    preValidation: zodRequestValidation({
      body: {
        schema: discordChannelConfigSchema,
        message: 'Invalid Discord channel request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    await runtime.configMutationService.apply(
      'config.channels.discord.upsert',
      async () => {
        replaceConfiguredDiscordChannel(
          runtime.config,
          restoreDiscordChannelSecrets(
            readConfiguredDiscordChannel(runtime.config),
            request.body,
          ),
        )
        await reconfigureRuntimeChannelType(runtime, 'discord')
        await persistRuntimeConfig(runtime, new Set(['channels']))
      },
    )
    return { data: { updated: ['channels'] } }
  })

  app.delete('/config/channels/discord', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const outcome = await runtime.configMutationService.apply(
      'config.channels.discord.delete',
      async () => {
        const existing = readConfiguredDiscordChannel(runtime.config)
        if (!existing) {
          return 'not_found' as const
        }

        runtime.config.channels = runtime.config.channels.filter(
          (channel) => channel.type !== 'discord',
        )
        await reconfigureRuntimeChannelType(runtime, 'discord')
        await persistRuntimeConfig(runtime, new Set(['channels']))
        return 'ok' as const
      },
    )

    if (outcome === 'not_found') {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'Discord channel not found',
        },
      })
    }
    return { data: { updated: ['channels'] } }
  })

  app.post<{ Body: ChannelToggleBody }>('/config/channels/discord/enable', {
    preValidation: zodRequestValidation({
      body: {
        schema: channelToggleRequestSchema,
        message: 'Invalid Discord channel toggle request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const body = request.body

    const outcome = await runtime.configMutationService.apply(
      'config.channels.discord.enable',
      async () => {
        const existing = readConfiguredDiscordChannel(runtime.config)
        if (!existing) {
          return 'not_found' as const
        }

        replaceConfiguredDiscordChannel(runtime.config, {
          ...existing,
          enabled: body.enabled,
        })
        await reconfigureRuntimeChannelType(runtime, 'discord')
        await persistRuntimeConfig(runtime, new Set(['channels']))
        return 'ok' as const
      },
    )

    if (outcome === 'not_found') {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'Discord channel not found',
        },
      })
    }
    return { data: { updated: ['channels'] } }
  })
}
