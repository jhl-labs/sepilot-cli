import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import { outboundWebhookSchema } from '../../config/schema.js'
import {
  OUTBOUND_WEBHOOK_DELIVERY_AUDIT_EVENT,
  deliverOutboundWebhook,
  outboundWebhookIdFromUrl,
} from '../../hooks/outbound-webhook.js'
import { applyAndPersistRuntimeUpdate } from '../runtime/config-runtime.js'
import { restoreOutboundWebhookSecrets } from '../runtime/config-mutations.js'
import { zodRequestValidation } from './utils.js'
import {
  outboundWebhookDeadLetterAckHistoryQuerySchema,
  outboundWebhookDeadLetterAckParamsSchema,
  outboundWebhookDeadLetterSchema,
  outboundWebhookDeadLetterAckRequestSchema,
  outboundWebhookDeadLetterQuerySchema,
  outboundWebhookDeadLetterTimelineQuerySchema,
  outboundWebhookDeliveryQuerySchema,
  outboundWebhookDeliveryReplayParamsSchema,
  outboundWebhookDeliveryReplayRequestSchema,
  outboundWebhookIdParamsSchema,
  outboundWebhookToggleRequestSchema,
  type OutboundWebhookBody,
  type OutboundWebhookDeadLetterAckBody,
  type OutboundWebhookDeadLetterAckHistoryQuery,
  type OutboundWebhookDeadLetterAckParams,
  type OutboundWebhookDeadLetterQuery,
  type OutboundWebhookDeadLetterTimelineQuery,
  type OutboundWebhookDeliveryQuery,
  type OutboundWebhookDeliveryReplayBody,
  type OutboundWebhookDeliveryReplayParams,
  type OutboundWebhookIdParams,
  type OutboundWebhookToggleBody,
} from './config-schema.js'
import {
  OUTBOUND_WEBHOOK_DEAD_LETTER_ACK_AUDIT_EVENT,
} from './config-schema.js'
import {
  findOutboundWebhookDeadLetter,
  findOutboundWebhookDeliveryAuditRecord,
  findOutboundWebhookDeliveryDetail,
  findOutboundWebhookIndexById,
  hasOutboundWebhookReplayChildren,
  listOutboundWebhookDeadLetterAcknowledgements,
  listOutboundWebhookDeadLetterTimeline,
  listOutboundWebhookDeadLetters,
  listOutboundWebhookDeliveries,
  listOutboundWebhookSummaries,
  normalizeOutboundWebhookDeliveries,
  queryOutboundWebhookAuditEvents,
  queryOutboundWebhookDeadLetterAcknowledgementAuditEvents,
} from './config-hooks-internals.js'

export async function registerConfigHooksOutboundRoutes(app: FastifyInstance): Promise<void> {
  const runtime = app.runtime

  app.get('/config/hooks/outbound-webhooks', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    return {
      data: listOutboundWebhookSummaries(runtime.config),
    }
  })

  app.get<{
    Querystring: OutboundWebhookDeliveryQuery
  }>('/config/hooks/outbound-webhooks/deliveries', {
    preValidation: zodRequestValidation({
      query: {
        schema: outboundWebhookDeliveryQuerySchema,
        message: 'Invalid outbound webhook delivery query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const query = request.query

    const auditEvents = runtime.auditLogger
      ? await runtime.auditLogger.query({
          event: OUTBOUND_WEBHOOK_DELIVERY_AUDIT_EVENT,
        })
      : []

    try {
      return listOutboundWebhookDeliveries(auditEvents, query)
    } catch {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: 'Invalid outbound webhook delivery cursor',
        },
      })
    }
  })

  app.get<{
    Params: OutboundWebhookDeliveryReplayParams
  }>('/config/hooks/outbound-webhooks/deliveries/:deliveryId', {
    preValidation: zodRequestValidation({
      params: {
        schema: outboundWebhookDeliveryReplayParamsSchema,
        message: 'Invalid outbound webhook delivery params',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const { deliveryId } = request.params

    const auditEvents = await queryOutboundWebhookAuditEvents(runtime)
    const delivery = findOutboundWebhookDeliveryDetail(
      auditEvents,
      deliveryId,
    )

    if (!delivery) {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: `Outbound webhook delivery not found: ${deliveryId}`,
        },
      })
    }

    return {
      data: delivery,
    }
  })

  app.get<{
    Querystring: OutboundWebhookDeadLetterQuery
  }>('/config/hooks/outbound-webhooks/dead-letters', {
    preValidation: zodRequestValidation({
      query: {
        schema: outboundWebhookDeadLetterQuerySchema,
        message: 'Invalid outbound webhook dead letter query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const query = request.query

    const auditEvents = await queryOutboundWebhookAuditEvents(runtime)

    try {
      return listOutboundWebhookDeadLetters(auditEvents, query)
    } catch {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: 'Invalid outbound webhook dead letter cursor',
        },
      })
    }
  })

  app.get<{
    Querystring: OutboundWebhookDeadLetterAckHistoryQuery
  }>('/config/hooks/outbound-webhooks/dead-letters/acknowledgements', {
    preValidation: zodRequestValidation({
      query: {
        schema: outboundWebhookDeadLetterAckHistoryQuerySchema,
        message: 'Invalid outbound webhook dead letter acknowledgement query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const query = request.query
    const auditEvents =
      await queryOutboundWebhookDeadLetterAcknowledgementAuditEvents(
        runtime,
        query.since,
      )

    try {
      return listOutboundWebhookDeadLetterAcknowledgements(auditEvents, query)
    } catch {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: 'Invalid outbound webhook dead letter acknowledgement cursor',
        },
      })
    }
  })

  app.get<{
    Params: OutboundWebhookDeadLetterAckParams
  }>('/config/hooks/outbound-webhooks/dead-letters/:rootDeliveryId', {
    preValidation: zodRequestValidation({
      params: {
        schema: outboundWebhookDeadLetterAckParamsSchema,
        message: 'Invalid outbound webhook dead letter params',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const params = request.params
    const auditEvents = await queryOutboundWebhookAuditEvents(runtime)
    const deadLetter = findOutboundWebhookDeadLetter(
      auditEvents,
      params.rootDeliveryId,
    )

    if (!deadLetter) {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: `Outbound webhook dead letter not found: ${params.rootDeliveryId}`,
        },
      })
    }

    return {
      data: deadLetter,
    }
  })

  app.get<{
    Params: OutboundWebhookDeadLetterAckParams
    Querystring: OutboundWebhookDeadLetterTimelineQuery
  }>('/config/hooks/outbound-webhooks/dead-letters/:rootDeliveryId/timeline', {
    preValidation: zodRequestValidation({
      params: {
        schema: outboundWebhookDeadLetterAckParamsSchema,
        message: 'Invalid outbound webhook dead letter params',
      },
      query: {
        schema: outboundWebhookDeadLetterTimelineQuerySchema,
        message: 'Invalid outbound webhook dead letter timeline query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const params = request.params
    const query = request.query
    const auditEvents = await queryOutboundWebhookAuditEvents(runtime)
    const deadLetter = findOutboundWebhookDeadLetter(
      auditEvents,
      params.rootDeliveryId,
    )

    if (!deadLetter) {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: `Outbound webhook dead letter not found: ${params.rootDeliveryId}`,
        },
      })
    }

    try {
      return listOutboundWebhookDeadLetterTimeline(
        auditEvents,
        params.rootDeliveryId,
        query,
      )
    } catch {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: 'Invalid outbound webhook dead letter timeline cursor',
        },
      })
    }
  })

  app.post<{
    Params: OutboundWebhookDeadLetterAckParams
    Body: OutboundWebhookDeadLetterAckBody
  }>('/config/hooks/outbound-webhooks/dead-letters/:rootDeliveryId/ack', {
    preValidation: zodRequestValidation({
      params: {
        schema: outboundWebhookDeadLetterAckParamsSchema,
        message: 'Invalid outbound webhook dead letter params',
      },
      body: {
        schema: outboundWebhookDeadLetterAckRequestSchema,
        message: 'Invalid outbound webhook dead letter acknowledgment body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    if (!runtime.auditLogger) {
      return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Audit logger not initialized' } })
    }
    const params = request.params
    const body = request.body

    const auditEvents = await queryOutboundWebhookAuditEvents(runtime)
    const deadLetter = findOutboundWebhookDeadLetter(
      auditEvents,
      params.rootDeliveryId,
    )

    if (!deadLetter) {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: `Outbound webhook dead letter not found: ${params.rootDeliveryId}`,
        },
      })
    }

    const acknowledgedAt = new Date().toISOString()
    await runtime.auditLogger.log({
      timestamp: acknowledgedAt,
      event: OUTBOUND_WEBHOOK_DEAD_LETTER_ACK_AUDIT_EVENT,
      device: runtime.config.device.name,
      rootDeliveryId: deadLetter.rootDeliveryId,
      latestDeliveryId: deadLetter.latestDeliveryId,
      webhookId: deadLetter.webhookId,
      ...(body.note ? { note: body.note } : {}),
    })

    return {
      data: outboundWebhookDeadLetterSchema.parse({
        ...deadLetter,
        state: 'acknowledged',
        acknowledgedAt,
        acknowledgedByDevice: runtime.config.device.name,
        ...(body.note ? { acknowledgmentNote: body.note } : {}),
      }),
    }
  })

  app.post<{
    Params: OutboundWebhookDeliveryReplayParams
    Body: OutboundWebhookDeliveryReplayBody
  }>('/config/hooks/outbound-webhooks/deliveries/:deliveryId/replay', {
    preValidation: zodRequestValidation({
      params: {
        schema: outboundWebhookDeliveryReplayParamsSchema,
        message: 'Invalid outbound webhook delivery params',
      },
      body: {
        schema: outboundWebhookDeliveryReplayRequestSchema,
        message: 'Invalid outbound webhook replay request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const params = request.params
    const body = request.body

    const auditEvents = await queryOutboundWebhookAuditEvents(runtime)
    const normalizedDeliveries = normalizeOutboundWebhookDeliveries(auditEvents)
    const delivery = findOutboundWebhookDeliveryAuditRecord(
      auditEvents,
      params.deliveryId,
    )

    if (!delivery) {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: `Outbound webhook delivery not found: ${params.deliveryId}`,
        },
      })
    }

    if (!delivery.payload) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: `Replay unavailable for delivery ${params.deliveryId}; payload snapshot not recorded`,
        },
      })
    }

    if (delivery.deliveryStatus === 'success' && !body.force) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: `Replay defaults to failed deliveries only: ${params.deliveryId}`,
        },
      })
    }

    if (
      hasOutboundWebhookReplayChildren(normalizedDeliveries, delivery.deliveryId)
      && !body.force
    ) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: `Replay of delivery with existing replay attempts requires force: ${params.deliveryId}`,
        },
      })
    }

    const index = findOutboundWebhookIndexById(runtime.config, delivery.webhookId)
    if (index < 0) {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: `Outbound webhook not found: ${delivery.webhookId}`,
        },
      })
    }

    const webhook = runtime.config.hooks.outboundWebhooks[index]!
    if (webhook.enabled === false) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: `Outbound webhook is disabled: ${delivery.webhookId}`,
        },
      })
    }

    const replay = await deliverOutboundWebhook(
      webhook,
      delivery.payload,
      {
        auditLogger: runtime.auditLogger,
        deviceName: runtime.config.device.name,
      },
      {
        replayedFromDeliveryId: delivery.deliveryId,
      },
    )

    return { data: replay }
  })

  app.post<{ Body: OutboundWebhookBody }>('/config/hooks/outbound-webhooks', {
    preValidation: zodRequestValidation({
      body: {
        schema: outboundWebhookSchema,
        message: 'Invalid outbound webhook request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    await runtime.configMutationService.apply(
      'config.hooks.outboundWebhooks.upsert',
      async () => {
        const [webhook] = restoreOutboundWebhookSecrets(
          runtime.config.hooks.outboundWebhooks,
          [request.body],
        )
        if (!webhook) {
          return
        }

        const existingIndex = findOutboundWebhookIndexById(
          runtime.config,
          outboundWebhookIdFromUrl(webhook.url),
        )

        if (existingIndex >= 0) {
          runtime.config.hooks.outboundWebhooks[existingIndex] = webhook
        } else {
          runtime.config.hooks.outboundWebhooks.push(webhook)
        }

        await applyAndPersistRuntimeUpdate(
          runtime,
          new Set(['hooks.outboundWebhooks']),
        )
      },
    )
    return { data: { updated: ['hooks.outboundWebhooks'] } }
  })








  app.delete<{ Params: OutboundWebhookIdParams }>(
    '/config/hooks/outbound-webhooks/:id',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: outboundWebhookIdParamsSchema,
          message: 'Invalid outbound webhook params',
        },
      }),
    },
    async (request, reply) => {
      if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

      const params = request.params

      const outcome = await runtime.configMutationService.apply(
        'config.hooks.outboundWebhooks.delete',
        async () => {
          const nextWebhooks = runtime.config.hooks.outboundWebhooks.filter(
            (webhook) => outboundWebhookIdFromUrl(webhook.url) !== params.id,
          )
          if (nextWebhooks.length === runtime.config.hooks.outboundWebhooks.length) {
            return 'not_found' as const
          }

          runtime.config.hooks.outboundWebhooks = nextWebhooks
          await applyAndPersistRuntimeUpdate(
            runtime,
            new Set(['hooks.outboundWebhooks']),
          )
          return 'ok' as const
        },
      )

      if (outcome === 'not_found') {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: `Outbound webhook not found: ${params.id}`,
          },
        })
      }
      return { data: { updated: ['hooks.outboundWebhooks'] } }
    },
  )


  app.post<{
    Params: OutboundWebhookIdParams
    Body: OutboundWebhookToggleBody
  }>('/config/hooks/outbound-webhooks/:id/enable', {
    preValidation: zodRequestValidation({
      params: {
        schema: outboundWebhookIdParamsSchema,
        message: 'Invalid outbound webhook params',
      },
      body: {
        schema: outboundWebhookToggleRequestSchema,
        message: 'Invalid outbound webhook toggle request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const params = request.params
    const body = request.body

    const outcome = await runtime.configMutationService.apply(
      'config.hooks.outboundWebhooks.enable',
      async () => {
        const index = findOutboundWebhookIndexById(runtime.config, params.id)
        if (index < 0) {
          return 'not_found' as const
        }

        runtime.config.hooks.outboundWebhooks[index] = {
          ...runtime.config.hooks.outboundWebhooks[index]!,
          enabled: body.enabled,
        }
        await applyAndPersistRuntimeUpdate(
          runtime,
          new Set(['hooks.outboundWebhooks']),
        )
        return 'ok' as const
      },
    )

    if (outcome === 'not_found') {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: `Outbound webhook not found: ${params.id}`,
        },
      })
    }
    return { data: { updated: ['hooks.outboundWebhooks'] } }
  })
}
