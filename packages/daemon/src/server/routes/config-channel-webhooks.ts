import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import {
  normalizeWebhookEndpointPath,
  webhookEndpointIdFromPath,
} from '../../channels/webhook.js'
import {
  persistRuntimeConfig,
  reconfigureRuntimeChannelType,
} from '../runtime/config-runtime.js'
import { zodRequestValidation } from './utils.js'
import {
  webhookEndpointIdParamsSchema,
  webhookEndpointSchema,
  webhookEndpointToggleRequestSchema,
  type WebhookEndpointBody,
  type WebhookEndpointIdParams,
  type WebhookEndpointToggleBody,
} from './config-schema.js'
import {
  findWebhookEndpointIndexById,
  listConfiguredWebhookEndpoints,
  listWebhookEndpointSummaries,
  replaceConfiguredWebhookEndpoints,
  restoreWebhookEndpointSecret,
} from './config-channels-internals.js'

export function registerConfigChannelWebhooksRoutes(app: FastifyInstance): void {
  const runtime = app.runtime

  app.get('/config/channels/webhook/endpoints', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    return {
      data: listWebhookEndpointSummaries(runtime.config),
    }
  })

  app.post<{ Body: WebhookEndpointBody }>('/config/channels/webhook/endpoints', {
    preValidation: zodRequestValidation({
      body: {
        schema: webhookEndpointSchema,
        message: 'Invalid webhook endpoint request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    await runtime.configMutationService.apply(
      'config.channels.webhook.upsert',
      async () => {
        const endpoint = restoreWebhookEndpointSecret(
          listConfiguredWebhookEndpoints(runtime.config),
          request.body,
        )

        const nextEndpoints = listConfiguredWebhookEndpoints(runtime.config)
        const normalizedEndpoint = {
          ...endpoint,
          path: normalizeWebhookEndpointPath(endpoint.path),
        }
        const existingIndex = nextEndpoints.findIndex(
          (entry) =>
            webhookEndpointIdFromPath(entry.path)
            === webhookEndpointIdFromPath(normalizedEndpoint.path),
        )

        if (existingIndex >= 0) {
          nextEndpoints[existingIndex] = normalizedEndpoint
        } else {
          nextEndpoints.push(normalizedEndpoint)
        }

        replaceConfiguredWebhookEndpoints(runtime.config, nextEndpoints)
        await reconfigureRuntimeChannelType(runtime, 'webhook')
        await persistRuntimeConfig(runtime, new Set(['channels']))
      },
    )
    return { data: { updated: ['channels'] } }
  })

  app.delete<{ Params: WebhookEndpointIdParams }>(
    '/config/channels/webhook/endpoints/:id',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: webhookEndpointIdParamsSchema,
          message: 'Invalid webhook endpoint params',
        },
      }),
    },
    async (request, reply) => {
      if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

      const params = request.params

      const outcome = await runtime.configMutationService.apply(
        'config.channels.webhook.delete',
        async () => {
          const existing = listConfiguredWebhookEndpoints(runtime.config)
          const nextEndpoints = existing.filter(
            (endpoint) => webhookEndpointIdFromPath(endpoint.path) !== params.id,
          )
          if (nextEndpoints.length === existing.length) {
            return 'not_found' as const
          }

          replaceConfiguredWebhookEndpoints(runtime.config, nextEndpoints)
          await reconfigureRuntimeChannelType(runtime, 'webhook')
          await persistRuntimeConfig(runtime, new Set(['channels']))
          return 'ok' as const
        },
      )

      if (outcome === 'not_found') {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: `Webhook endpoint not found: ${params.id}`,
          },
        })
      }
      return { data: { updated: ['channels'] } }
    },
  )

  app.post<{
    Params: WebhookEndpointIdParams
    Body: WebhookEndpointToggleBody
  }>('/config/channels/webhook/endpoints/:id/enable', {
    preValidation: zodRequestValidation({
      params: {
        schema: webhookEndpointIdParamsSchema,
        message: 'Invalid webhook endpoint params',
      },
      body: {
        schema: webhookEndpointToggleRequestSchema,
        message: 'Invalid webhook endpoint toggle request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const params = request.params
    const body = request.body

    const outcome = await runtime.configMutationService.apply(
      'config.channels.webhook.enable',
      async () => {
        const nextEndpoints = listConfiguredWebhookEndpoints(runtime.config)
        const index = findWebhookEndpointIndexById(runtime.config, params.id)
        if (index < 0) {
          return 'not_found' as const
        }

        nextEndpoints[index] = {
          ...nextEndpoints[index]!,
          enabled: body.enabled,
        }
        replaceConfiguredWebhookEndpoints(runtime.config, nextEndpoints)
        await reconfigureRuntimeChannelType(runtime, 'webhook')
        await persistRuntimeConfig(runtime, new Set(['channels']))
        return 'ok' as const
      },
    )

    if (outcome === 'not_found') {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: `Webhook endpoint not found: ${params.id}`,
        },
      })
    }
    return { data: { updated: ['channels'] } }
  })
}
