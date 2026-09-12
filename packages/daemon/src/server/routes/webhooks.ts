import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import type { SlackChannel } from '../../channels/slack.js'
import type { DiscordChannel } from '../../channels/discord.js'
import type { LINEChannel } from '../../channels/line.js'
import type { MattermostChannel } from '../../channels/mattermost.js'
import type { TeamsChannel } from '../../channels/teams.js'
import type { WebhookChannel } from '../../channels/webhook.js'
import type { WhatsAppChannel } from '../../channels/whatsapp.js'
import {
  getChannelWebhookDispatcher,
  webhookIdempotencyKey,
} from '../../channels/webhook-dispatch.js'
import { normalizeHeaders, parseRequestInput } from './utils.js'
import {
  resolveWebhookSecurityPolicy,
} from '../runtime/webhook-security-policy.js'
import {
  getLatestWebhookSecurityPolicyChange,
} from '../runtime/webhook-security-policy-audit.js'
import {
  listWebhookEndpoints,
  summarizeWebhookSecurity,
} from '../runtime/webhook-endpoints.js'
import {
  genericWebhookBodySchema,
  genericWebhookWildcardParamsSchema,
  lineWebhookBodySchema,
  teamsActivitySchema,
  webhookSecurityQuerySchema,
  whatsAppVerifyQuerySchema,
  whatsAppWebhookBodySchema,
  type GenericWebhookBody,
  type WhatsAppVerifyQuery,
} from './webhooks-schema.js'

export { webhookOpenApiComponents, webhookOpenApiOverrides } from './webhooks-openapi.js'

import {
  getRawBody,
  getSlackBody,
  getWebhookBody,
  sendWebhookVerificationUnavailable,
} from './webhooks-internals.js'

export async function webhookRoutes(app: FastifyInstance) {
  const runtime = app.runtime
  if (!runtime) return

  app.get('/webhooks', async () => ({
    data: listWebhookEndpoints(runtime),
    meta: {
      latestSecurityPolicyChange:
        await getLatestWebhookSecurityPolicyChange(runtime),
    },
  }))

  app.get('/webhooks/security', async (request, reply) => {
    const query = parseRequestInput(
      reply,
      webhookSecurityQuerySchema,
      request.query,
      'Invalid webhook security query',
    )
    if (!query) return reply

    const summary = summarizeWebhookSecurity(
      runtime,
      {
        channelType: query.channelType,
        verificationReady: query.verificationReady,
        missingRequirement: query.missingRequirement,
      },
      {
        unreadyOffset: query.unreadyOffset,
        unreadyLimit: query.unreadyLimit,
      },
    )

    return {
      data: summary,
      meta: {
        latestSecurityPolicyChange:
          await getLatestWebhookSecurityPolicyChange(runtime),
        filters: {
          channelType: query.channelType ?? null,
          verificationReady: query.verificationReady ?? null,
          missingRequirement: query.missingRequirement ?? null,
        },
        unreadyPage: {
          offset: query.unreadyOffset,
          limit: query.unreadyLimit,
          total: summary.unreadySummary.totalEndpoints,
          returned: summary.unreadyEndpoints.length,
        },
      },
    }
  })

  // Slack Events API
  app.post('/webhooks/slack', async (request, reply) => {
    const slackChannel = runtime.channels.find(c => c.type === 'slack') as SlackChannel | undefined
    const securityPolicy = resolveWebhookSecurityPolicy(runtime.config, 'slack')
    if (!slackChannel) {
      return reply.status(404).send({ error: 'Slack channel not configured' })
    }
    if (!slackChannel.canVerifySignature()) {
      return sendWebhookVerificationUnavailable(
        reply,
        runtime,
        'slack',
        'Slack signingSecret is required for webhook verification',
      )
    }

    const body = getSlackBody(request.body)
    const headers = normalizeHeaders(request.headers)
    const rawBody = getRawBody(request.body, request.rawBody)

    // Handle URL verification challenge
    if (body?.type === 'url_verification') {
      if (!slackChannel.verifySignature(
        rawBody,
        headers,
        securityPolicy.signatureMaxSkewSeconds,
      )) {
        return reply.status(401).send({ error: 'Invalid Slack signature' })
      }
      return { challenge: body.challenge }
    }

    if (!slackChannel.verifySignature(
      rawBody,
      headers,
      securityPolicy.signatureMaxSkewSeconds,
    )) {
      return reply.status(401).send({ error: 'Invalid Slack signature' })
    }

    const verifiedBody = getWebhookBody(request.body)
    getChannelWebhookDispatcher(runtime).enqueue({
      idempotencyKey: webhookIdempotencyKey('slack', verifiedBody),
      channelType: 'slack',
      kind: 'slack.event',
      payload: { body: verifiedBody },
    })
    return { ok: true }
  })

  // Discord Interactions
  app.post('/webhooks/discord', async (request, reply) => {
    const discordChannel = runtime.channels.find(c => c.type === 'discord') as DiscordChannel | undefined
    const securityPolicy = resolveWebhookSecurityPolicy(runtime.config, 'discord')
    if (!discordChannel) {
      return reply.status(404).send({ error: 'Discord channel not configured' })
    }
    if (!discordChannel.canVerifyInteractionSignature()) {
      return sendWebhookVerificationUnavailable(
        reply,
        runtime,
        'discord',
        'Discord publicKey is required for webhook verification',
      )
    }

    const headers = normalizeHeaders(request.headers)
    const signature = headers['x-signature-ed25519']
    const timestamp = headers['x-signature-timestamp']
    if (!discordChannel.verifyInteractionSignature(
      getRawBody(request.body, request.rawBody),
      signature ?? '',
      timestamp ?? '',
      securityPolicy.signatureMaxSkewSeconds,
    )) {
      return reply.status(401).send({ error: 'Invalid Discord signature' })
    }

    const body = getWebhookBody(request.body)
    if (body.type === 1) {
      return discordChannel.handleInteraction(body)
    }
    getChannelWebhookDispatcher(runtime).enqueue({
      idempotencyKey: webhookIdempotencyKey('discord', body),
      channelType: 'discord',
      kind: 'discord.interaction',
      payload: { body },
    })
    return { type: 5 }
  })

  // Mattermost slash command / outgoing webhook callback
  app.post('/webhooks/mattermost', async (request, reply) => {
    const mattermostChannel = runtime.channels.find(c => c.type === 'mattermost') as MattermostChannel | undefined
    if (!mattermostChannel) {
      return reply.status(404).send({ error: 'Mattermost channel not configured' })
    }
    if (!mattermostChannel.canVerifyWebhookToken()) {
      return sendWebhookVerificationUnavailable(
        reply,
        runtime,
        'mattermost',
        'Mattermost webhookToken is required for webhook verification',
      )
    }

    const body = getWebhookBody(request.body)
    if (!mattermostChannel.verifyWebhookToken(body)) {
      return reply.status(401).send({ error: 'Invalid Mattermost token' })
    }

    getChannelWebhookDispatcher(runtime).enqueue({
      idempotencyKey: webhookIdempotencyKey('mattermost', body),
      channelType: 'mattermost',
      kind: 'mattermost.webhook',
      payload: { body },
    })
    return {
      response_type: 'ephemeral',
      text: 'Received. sepilotd will reply in this channel when the run finishes.',
    }
  })

  // WhatsApp Cloud API
  app.get<{ Querystring: WhatsAppVerifyQuery }>('/webhooks/whatsapp', async (request, reply) => {
    const whatsappChannel = runtime.channels.find(c => c.type === 'whatsapp') as WhatsAppChannel | undefined
    if (!whatsappChannel) {
      return reply.status(404).send({ error: 'WhatsApp channel not configured' })
    }
    if (!whatsappChannel.canVerifyWebhookChallenge()) {
      return sendWebhookVerificationUnavailable(
        reply,
        runtime,
        'whatsapp',
        'WhatsApp verifyToken is required for webhook verification',
      )
    }

    const query = parseRequestInput(
      reply,
      whatsAppVerifyQuerySchema,
      request.query,
      'Invalid WhatsApp verification query',
    )
    if (!query) return reply
    const challenge = whatsappChannel.verifyWebhook(
      query['hub.mode'] ?? '',
      query['hub.verify_token'] ?? '',
      query['hub.challenge'] ?? '',
    )

    if (!challenge) {
      return reply.status(403).send({ error: 'WhatsApp verification failed' })
    }

    reply.type('text/plain')
    return challenge
  })

  app.post('/webhooks/whatsapp', async (request, reply) => {
    const whatsappChannel = runtime.channels.find(c => c.type === 'whatsapp') as WhatsAppChannel | undefined
    if (!whatsappChannel) {
      return reply.status(404).send({ error: 'WhatsApp channel not configured' })
    }
    if (!whatsappChannel.canVerifySignature()) {
      return sendWebhookVerificationUnavailable(
        reply,
        runtime,
        'whatsapp',
        'WhatsApp appSecret is required for webhook verification',
      )
    }

    const headers = normalizeHeaders(request.headers)
    if (!whatsappChannel.verifySignature(
      getRawBody(request.body, request.rawBody),
      headers['x-hub-signature-256'],
    )) {
      return reply.status(401).send({ error: 'Invalid WhatsApp signature' })
    }

    const body = parseRequestInput(
      reply,
      whatsAppWebhookBodySchema,
      request.body,
      'Invalid WhatsApp webhook body',
    )
    if (!body) return reply
    getChannelWebhookDispatcher(runtime).enqueue({
      idempotencyKey: webhookIdempotencyKey('whatsapp', body as Record<string, unknown>),
      channelType: 'whatsapp',
      kind: 'whatsapp.webhook',
      payload: { body: body as Record<string, unknown> },
    })
    return { ok: true }
  })

  // Microsoft Teams Bot Framework
  app.post('/webhooks/teams', async (request, reply) => {
    const teamsChannel = runtime.channels.find(c => c.type === 'teams') as TeamsChannel | undefined
    if (!teamsChannel) {
      return reply.status(404).send({ error: 'Teams channel not configured' })
    }
    if (!teamsChannel.canValidateRequest()) {
      return sendWebhookVerificationUnavailable(
        reply,
        runtime,
        'teams',
        'Teams appId and appPassword are required for webhook verification',
      )
    }

    const headers = normalizeHeaders(request.headers)
    const body = parseRequestInput(
      reply,
      teamsActivitySchema,
      request.body,
      'Invalid Teams activity body',
    )
    if (!body) return reply
    // Bind the JWT serviceurl claim to the inbound activity's serviceUrl so a
    // valid token cannot be replayed against an attacker-chosen serviceUrl.
    if (!await teamsChannel.validateRequest(headers, body.serviceUrl)) {
      return reply.status(401).send({ error: 'Invalid Teams authorization' })
    }
    const activity = {
      type: body.type,
      id: body.id,
      text: body.text,
      timestamp: body.timestamp,
      serviceUrl: body.serviceUrl,
      from: body.from
        ? {
            id: body.from.id,
            name: body.from.name,
          }
        : undefined,
      conversation: body.conversation?.id
        ? { id: body.conversation.id }
        : undefined,
      channelData: body.channelData?.tenant
        ? {
            tenant: {
              id: body.channelData.tenant.id,
            },
          }
        : undefined,
    }
    getChannelWebhookDispatcher(runtime).enqueue({
      idempotencyKey: webhookIdempotencyKey('teams', activity),
      channelType: 'teams',
      kind: 'teams.activity',
      payload: { body: activity },
    })
    return { ok: true }
  })

  // LINE Messaging API
  app.post('/webhooks/line', async (request, reply) => {
    const lineChannel = runtime.channels.find(c => c.type === 'line') as LINEChannel | undefined
    if (!lineChannel) {
      return reply.status(404).send({ error: 'LINE channel not configured' })
    }
    if (!lineChannel.canVerifySignature()) {
      return sendWebhookVerificationUnavailable(
        reply,
        runtime,
        'line',
        'LINE channelSecret is required for webhook verification',
      )
    }

    const headers = normalizeHeaders(request.headers)
    const rawBody = getRawBody(request.body, request.rawBody)
    const body = parseRequestInput(
      reply,
      lineWebhookBodySchema,
      request.body,
      'Invalid LINE webhook body',
    )
    if (!body) return reply
    const signature = headers['x-line-signature'] ?? ''
    if (
      typeof lineChannel.verifySignature !== 'function'
      || !lineChannel.verifySignature(rawBody, signature)
    ) {
      return reply.status(401).send({ error: 'Invalid LINE signature' })
    }
    getChannelWebhookDispatcher(runtime).enqueue({
      idempotencyKey: webhookIdempotencyKey('line', body as Record<string, unknown>),
      channelType: 'line',
      kind: 'line.webhook',
      payload: { body: body as Record<string, unknown>, rawBody, signature },
    })
    return { ok: true }
  })

  // Generic Webhooks
  app.post<{ Params: Record<'*', string> }>('/webhooks/*', async (request, reply) => {
    const params = parseRequestInput(
      reply,
      genericWebhookWildcardParamsSchema,
      request.params,
      'Invalid webhook path',
    )
    if (!params) return reply
    const body = parseRequestInput(reply, genericWebhookBodySchema, getWebhookBody(request.body), 'Invalid webhook body')
    if (!body) return reply
    const webhookChannels = runtime.channels.filter(
      (channel) => channel.type === 'webhook',
    ) as WebhookChannel[]
    for (const webhookChannel of webhookChannels) {
      if (webhookChannel.getEndpointVerificationState(`/hook/${params.path}`) === 'verification-unavailable') {
        return sendWebhookVerificationUnavailable(
          reply,
          runtime,
          'webhook',
          'Webhook endpoint secret configuration is required for verification',
        )
      }
      const handled = await webhookChannel.handleWebhook(
        `/hook/${params.path}`,
        body as GenericWebhookBody,
        normalizeHeaders(request.headers),
        request.ip,
        getRawBody(request.body, request.rawBody),
      )
      if (handled) {
        return { ok: true }
      }
    }
    return reply.status(404).send({ error: 'Webhook endpoint not found' })
  })
}
