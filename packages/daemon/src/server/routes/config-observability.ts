import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import { resolveChannelPipelineHealthConfig } from '../runtime/channel-pipeline-health.js'
import { resolveWebhookSecurityHealthConfig } from '../runtime/webhook-security-health.js'
import { applyConfigUpdate } from '../runtime/config-mutations.js'
import { applyAndPersistRuntimeUpdate } from '../runtime/config-runtime.js'
import { zodRequestValidation } from './utils.js'
import {
  CHANNEL_PIPELINE_HEALTH_CONFIG_AUDIT_EVENT,
  WEBHOOK_SECURITY_HEALTH_CONFIG_AUDIT_EVENT,
  configChannelPipelineHealthHistoryQuerySchema,
  configChannelPipelineHealthUpdateRequestSchema,
  configWebhookSecurityHealthHistoryQuerySchema,
  configWebhookSecurityHealthUpdateRequestSchema,
  type ConfigChannelPipelineHealthBody,
  type ConfigChannelPipelineHealthHistoryQuery,
  type ConfigWebhookSecurityHealthBody,
  type ConfigWebhookSecurityHealthHistoryQuery,
} from './config-schema.js'
import {
  listChannelPipelineHealthConfigAuditEvents,
  listWebhookSecurityHealthConfigAuditEvents,
  recordChannelPipelineHealthConfigAuditEvent,
  recordWebhookSecurityHealthConfigAuditEvent,
} from './config-audit-events.js'

export function registerConfigObservabilityRoutes(app: FastifyInstance): void {
  const runtime = app.runtime

  app.get('/config/observability/channel-pipeline-health', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    return {
      data: resolveChannelPipelineHealthConfig(runtime.config),
    }
  })

  app.get<{
    Querystring: ConfigChannelPipelineHealthHistoryQuery
  }>('/config/observability/channel-pipeline-health/history', {
    preValidation: zodRequestValidation({
      query: {
        schema: configChannelPipelineHealthHistoryQuerySchema,
        message: 'Invalid channel pipeline health history query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const query = request.query
    const auditEvents = runtime.auditLogger
      ? await runtime.auditLogger.query({
          event: CHANNEL_PIPELINE_HEALTH_CONFIG_AUDIT_EVENT,
          since: query.since,
        })
      : []

    try {
      return listChannelPipelineHealthConfigAuditEvents(auditEvents, query)
    } catch {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: 'Invalid channel pipeline health history cursor',
        },
      })
    }
  })

  app.put<{ Body: ConfigChannelPipelineHealthBody }>('/config/observability/channel-pipeline-health', {
    preValidation: zodRequestValidation({
      body: {
        schema: configChannelPipelineHealthUpdateRequestSchema,
        message: 'Invalid channel pipeline health config update request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const current = await runtime.configMutationService.apply(
      'config.observability.channelPipelineHealth.update',
      async () => {
        const previous = resolveChannelPipelineHealthConfig(runtime.config)
        applyConfigUpdate(
          runtime.config,
          'observability.channelPipelineHealth',
          request.body,
        )
        await applyAndPersistRuntimeUpdate(
          runtime,
          new Set(['observability.channelPipelineHealth']),
        )
        const updated = resolveChannelPipelineHealthConfig(runtime.config)
        await recordChannelPipelineHealthConfigAuditEvent(runtime, {
          route: '/api/v1/config/observability/channel-pipeline-health',
          requested: request.body,
          previous,
          current: updated,
        })
        return updated
      },
    )

    return {
      data: current,
    }
  })

  app.get('/config/observability/webhook-security-health', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    return {
      data: resolveWebhookSecurityHealthConfig(runtime.config),
    }
  })

  app.get<{
    Querystring: ConfigWebhookSecurityHealthHistoryQuery
  }>('/config/observability/webhook-security-health/history', {
    preValidation: zodRequestValidation({
      query: {
        schema: configWebhookSecurityHealthHistoryQuerySchema,
        message: 'Invalid webhook security health history query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const query = request.query
    const auditEvents = runtime.auditLogger
      ? await runtime.auditLogger.query({
          event: WEBHOOK_SECURITY_HEALTH_CONFIG_AUDIT_EVENT,
          since: query.since,
        })
      : []

    try {
      return listWebhookSecurityHealthConfigAuditEvents(auditEvents, query)
    } catch {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: 'Invalid webhook security health history cursor',
        },
      })
    }
  })

  app.put<{ Body: ConfigWebhookSecurityHealthBody }>('/config/observability/webhook-security-health', {
    preValidation: zodRequestValidation({
      body: {
        schema: configWebhookSecurityHealthUpdateRequestSchema,
        message: 'Invalid webhook security health config update request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const current = await runtime.configMutationService.apply(
      'config.observability.webhookSecurityHealth.update',
      async () => {
        const previous = resolveWebhookSecurityHealthConfig(runtime.config)
        applyConfigUpdate(
          runtime.config,
          'observability.webhookSecurityHealth',
          request.body,
        )
        await applyAndPersistRuntimeUpdate(
          runtime,
          new Set(['observability.webhookSecurityHealth']),
        )
        const updated = resolveWebhookSecurityHealthConfig(runtime.config)
        await recordWebhookSecurityHealthConfigAuditEvent(runtime, {
          route: '/api/v1/config/observability/webhook-security-health',
          requested: request.body,
          previous,
          current: updated,
        })
        return updated
      },
    )

    return {
      data: current,
    }
  })
}
