import type { FastifyInstance, FastifyReply } from 'fastify'
import '../fastify-types.js'
import { isRuntimeManagedChannel } from '../../config/runtime-channel-env.js'
import {
  persistRuntimeConfig,
  reconfigureRuntimeChannelType,
} from '../runtime/config-runtime.js'
import { zodRequestValidation } from './utils.js'
import {
  channelToggleRequestSchema,
  mattermostChannelConfigSchema,
  type ChannelToggleBody,
  type MattermostChannelBody,
} from './config-schema.js'
import {
  readConfiguredMattermostChannel,
  replaceConfiguredMattermostChannel,
  restoreMattermostChannelSecrets,
  summarizeMattermostChannel,
} from './config-channels-internals.js'

export function registerConfigChannelMattermostRoutes(app: FastifyInstance): void {
  const runtime = app.runtime

  const rejectEnvironmentManagedMutation = (reply: FastifyReply) => {
    if (!runtime || !isRuntimeManagedChannel(runtime.config, 'mattermost')) {
      return null
    }
    return reply.status(409).send({
      error: {
        code: 'CHANNEL_MANAGED_BY_ENVIRONMENT',
        message:
          'Mattermost is managed by the daemon runtime environment. Update the ExternalSecret/runtime environment and restart the daemon instead.',
      },
    })
  }

  app.get('/config/channels/mattermost', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    return {
      data: summarizeMattermostChannel(runtime),
    }
  })

  app.post<{ Body: MattermostChannelBody }>('/config/channels/mattermost', {
    preValidation: zodRequestValidation({
      body: {
        schema: mattermostChannelConfigSchema,
        message: 'Invalid Mattermost channel request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const managed = rejectEnvironmentManagedMutation(reply)
    if (managed) return managed

    await runtime.configMutationService.apply(
      'config.channels.mattermost.upsert',
      async () => {
        replaceConfiguredMattermostChannel(
          runtime.config,
          restoreMattermostChannelSecrets(
            readConfiguredMattermostChannel(runtime.config),
            request.body,
          ),
        )
        await reconfigureRuntimeChannelType(runtime, 'mattermost')
        await persistRuntimeConfig(runtime, new Set(['channels']))
      },
    )
    return { data: { updated: ['channels'] } }
  })

  app.delete('/config/channels/mattermost', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const managed = rejectEnvironmentManagedMutation(reply)
    if (managed) return managed

    const outcome = await runtime.configMutationService.apply(
      'config.channels.mattermost.delete',
      async () => {
        const existing = readConfiguredMattermostChannel(runtime.config)
        if (!existing) {
          return 'not_found' as const
        }

        runtime.config.channels = runtime.config.channels.filter(
          (channel) => channel.type !== 'mattermost',
        )
        await reconfigureRuntimeChannelType(runtime, 'mattermost')
        await persistRuntimeConfig(runtime, new Set(['channels']))
        return 'ok' as const
      },
    )

    if (outcome === 'not_found') {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'Mattermost channel not found',
        },
      })
    }
    return { data: { updated: ['channels'] } }
  })

  app.post<{ Body: ChannelToggleBody }>('/config/channels/mattermost/enable', {
    preValidation: zodRequestValidation({
      body: {
        schema: channelToggleRequestSchema,
        message: 'Invalid Mattermost channel toggle request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const managed = rejectEnvironmentManagedMutation(reply)
    if (managed) return managed

    const body = request.body

    const outcome = await runtime.configMutationService.apply(
      'config.channels.mattermost.enable',
      async () => {
        const existing = readConfiguredMattermostChannel(runtime.config)
        if (!existing) {
          return 'not_found' as const
        }

        replaceConfiguredMattermostChannel(runtime.config, {
          ...existing,
          enabled: body.enabled,
        })
        await reconfigureRuntimeChannelType(runtime, 'mattermost')
        await persistRuntimeConfig(runtime, new Set(['channels']))
        return 'ok' as const
      },
    )

    if (outcome === 'not_found') {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'Mattermost channel not found',
        },
      })
    }
    return { data: { updated: ['channels'] } }
  })
}
