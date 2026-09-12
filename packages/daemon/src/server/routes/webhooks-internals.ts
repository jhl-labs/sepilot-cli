import type { FastifyReply } from 'fastify'
import type { ConfigMutationCapabilities } from '../runtime/capabilities.js'
import { isRecord } from './utils.js'
import { resolveWebhookSecurityPolicy } from '../runtime/webhook-security-policy.js'
import {
  slackUrlVerificationBodySchema,
  type SlackUrlVerificationBody,
} from './webhooks-schema.js'

export function getSlackBody(body: unknown): SlackUrlVerificationBody {
  const parsed = slackUrlVerificationBodySchema.safeParse(body)
  return parsed.success ? parsed.data : {}
}

export function getWebhookBody(body: unknown): Record<string, unknown> {
  return isRecord(body) ? body : {}
}

export function getRawBody(body: unknown, rawBody?: string): string {
  if (typeof rawBody === 'string') return rawBody
  return JSON.stringify(getWebhookBody(body))
}

export function webhookVerificationUnavailableStatusCode(
  runtime: ConfigMutationCapabilities,
  channelType: string,
): 404 | 503 {
  return resolveWebhookSecurityPolicy(runtime.config, channelType)
    .verificationUnavailableStatus === 'not_found'
    ? 404
    : 503
}

export function sendWebhookVerificationUnavailable(
  reply: FastifyReply,
  runtime: ConfigMutationCapabilities,
  channelType: string,
  error: string,
) {
  return reply.status(
    webhookVerificationUnavailableStatusCode(runtime, channelType),
  ).send({ error })
}
