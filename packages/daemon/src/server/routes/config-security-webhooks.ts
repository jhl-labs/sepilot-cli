import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import { resolveWebhookSecurityPolicy } from '../runtime/webhook-security-policy.js'
import { applyConfigUpdate } from '../runtime/config-mutations.js'
import { applyAndPersistRuntimeUpdate } from '../runtime/config-runtime.js'
import { zodRequestValidation } from './utils.js'
import {
  WEBHOOK_SECURITY_POLICY_CONFIG_AUDIT_EVENT,
  configWebhookSecurityPolicyHistoryQuerySchema,
  configWebhookSecurityPolicyUpdateRequestSchema,
  type ConfigWebhookSecurityPolicyBody,
  type ConfigWebhookSecurityPolicyHistoryQuery,
} from './config-schema.js'
import {
  listWebhookSecurityPolicyConfigAuditEvents,
  recordWebhookSecurityPolicyConfigAuditEvent,
} from './config-audit-events.js'

export function registerConfigSecurityWebhookRoutes(app: FastifyInstance): void {
  const runtime = app.runtime

  app.get('/config/security/webhooks', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    return {
      data: resolveWebhookSecurityPolicy(runtime.config),
    }
  })

  app.get<{
    Querystring: ConfigWebhookSecurityPolicyHistoryQuery
  }>('/config/security/webhooks/history', {
    preValidation: zodRequestValidation({
      query: {
        schema: configWebhookSecurityPolicyHistoryQuerySchema,
        message: 'Invalid webhook security policy history query',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const query = request.query
    const auditEvents = runtime.auditLogger
      ? await runtime.auditLogger.query({
          event: WEBHOOK_SECURITY_POLICY_CONFIG_AUDIT_EVENT,
          since: query.since,
        })
      : []

    try {
      return listWebhookSecurityPolicyConfigAuditEvents(auditEvents, query)
    } catch {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: 'Invalid webhook security policy history cursor',
        },
      })
    }
  })

  app.put<{ Body: ConfigWebhookSecurityPolicyBody }>('/config/security/webhooks', {
    preValidation: zodRequestValidation({
      body: {
        schema: configWebhookSecurityPolicyUpdateRequestSchema,
        message: 'Invalid webhook security policy update request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const current = await runtime.configMutationService.apply(
      'config.security.webhooks.update',
      async () => {
        const previous = resolveWebhookSecurityPolicy(runtime.config)
        applyConfigUpdate(
          runtime.config,
          'security.webhooks',
          request.body,
        )
        await applyAndPersistRuntimeUpdate(
          runtime,
          new Set(['security.webhooks']),
        )
        const updated = resolveWebhookSecurityPolicy(runtime.config)
        await recordWebhookSecurityPolicyConfigAuditEvent(runtime, {
          route: '/api/v1/config/security/webhooks',
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
