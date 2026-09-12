import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import {
  persistRuntimeConfig,
  reconfigureRuntimeChannelType,
} from '../runtime/config-runtime.js'
import { zodRequestValidation } from './utils.js'
import {
  channelToggleRequestSchema,
  slackChannelConfigSchema,
  type ChannelToggleBody,
  type SlackChannelBody,
} from './config-schema.js'
import {
  readConfiguredSlackChannel,
  replaceConfiguredSlackChannel,
  restoreSlackChannelSecrets,
  summarizeSlackChannel,
} from './config-channels-internals.js'

export function registerConfigChannelSlackRoutes(app: FastifyInstance): void {
  const runtime = app.runtime

  app.get('/config/channels/slack', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    return {
      data: summarizeSlackChannel(runtime),
    }
  })

  app.post<{ Body: SlackChannelBody }>('/config/channels/slack', {
    preValidation: zodRequestValidation({
      body: {
        schema: slackChannelConfigSchema,
        message: 'Invalid Slack channel request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    await runtime.configMutationService.apply(
      'config.channels.slack.upsert',
      async () => {
        replaceConfiguredSlackChannel(
          runtime.config,
          restoreSlackChannelSecrets(
            readConfiguredSlackChannel(runtime.config),
            request.body,
          ),
        )
        await reconfigureRuntimeChannelType(runtime, 'slack')
        await persistRuntimeConfig(runtime, new Set(['channels']))
      },
    )
    return { data: { updated: ['channels'] } }
  })

  app.delete('/config/channels/slack', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const outcome = await runtime.configMutationService.apply(
      'config.channels.slack.delete',
      async () => {
        const existing = readConfiguredSlackChannel(runtime.config)
        if (!existing) {
          return 'not_found' as const
        }

        runtime.config.channels = runtime.config.channels.filter(
          (channel) => channel.type !== 'slack',
        )
        await reconfigureRuntimeChannelType(runtime, 'slack')
        await persistRuntimeConfig(runtime, new Set(['channels']))
        return 'ok' as const
      },
    )

    if (outcome === 'not_found') {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'Slack channel not found',
        },
      })
    }
    return { data: { updated: ['channels'] } }
  })

  app.post<{ Body: ChannelToggleBody }>('/config/channels/slack/enable', {
    preValidation: zodRequestValidation({
      body: {
        schema: channelToggleRequestSchema,
        message: 'Invalid Slack channel toggle request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const body = request.body

    const outcome = await runtime.configMutationService.apply(
      'config.channels.slack.enable',
      async () => {
        const existing = readConfiguredSlackChannel(runtime.config)
        if (!existing) {
          return 'not_found' as const
        }

        replaceConfiguredSlackChannel(runtime.config, {
          ...existing,
          enabled: body.enabled,
        })
        await reconfigureRuntimeChannelType(runtime, 'slack')
        await persistRuntimeConfig(runtime, new Set(['channels']))
        return 'ok' as const
      },
    )

    if (outcome === 'not_found') {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'Slack channel not found',
        },
      })
    }
    return { data: { updated: ['channels'] } }
  })
}
