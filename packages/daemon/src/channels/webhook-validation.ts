import { z } from 'zod'

const RESERVED_GENERIC_WEBHOOK_ROUTE_SEGMENTS = new Set([
  'discord',
  'line',
  'mattermost',
  'security',
  'slack',
  'teams',
  'whatsapp',
])

function publicWebhookRoutePath(input: string): string {
  const withLeadingSlash = input.startsWith('/') ? input : `/${input}`
  if (withLeadingSlash.startsWith('/api/v1/webhooks/')) {
    return withLeadingSlash.slice('/api/v1/webhooks/'.length)
  }
  if (withLeadingSlash.startsWith('/webhooks/')) {
    return withLeadingSlash.slice('/webhooks/'.length)
  }
  if (withLeadingSlash.startsWith('/hook/')) {
    return withLeadingSlash.slice('/hook/'.length)
  }
  return withLeadingSlash.replace(/^\/+/, '')
}

function firstPublicWebhookRouteSegment(input: string): string {
  return publicWebhookRoutePath(input)
    .replace(/^\/+/, '')
    .split('/', 1)[0]!
    .toLowerCase()
}

export function validateWebhookPath(input: string): string {
  const trimmed = input.trim()
  if (trimmed.length === 0) {
    throw new Error('Webhook path cannot be empty')
  }
  if (/(?:^|\/)\.\.(?:\/|$)/.test(trimmed)) {
    throw new Error(`Webhook path contains '..' segment: ${input}`)
  }
  if (/[?#%]/.test(trimmed)) {
    throw new Error(`Webhook path contains URL delimiter or escape char: ${input}`)
  }
  if (/[\s\x00-\x1f]/.test(trimmed)) {
    throw new Error(`Webhook path contains whitespace or control char: ${input}`)
  }
  const firstSegment = firstPublicWebhookRouteSegment(trimmed)
  if (RESERVED_GENERIC_WEBHOOK_ROUTE_SEGMENTS.has(firstSegment)) {
    throw new Error(`Webhook path uses reserved route segment: ${firstSegment}`)
  }
  return trimmed
}

export function isSafeWebhookPath(input: string): boolean {
  try {
    validateWebhookPath(input)
    return true
  } catch {
    return false
  }
}

export const safeWebhookPathSchema = z
  .string()
  .min(1, 'Webhook path cannot be empty')
  .transform((value, ctx) => {
    try {
      return validateWebhookPath(value)
    } catch (err) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: err instanceof Error ? err.message : String(err),
      })
      return z.NEVER
    }
  })

export const webhookEndpointConfigSchema = z.object({
  enabled: z.boolean().default(true),
  path: safeWebhookPathSchema,
  secretHeader: z.string().min(1),
  secretValue: z.string().min(1),
  allowedIps: z.array(z.string()).default([]),
  allowedEvents: z.array(z.string()).default([]),
  requireTimestamp: z.boolean().default(false),
  timestampHeader: z.string().min(1).default('x-webhook-timestamp'),
  timestampMaxSkewSeconds: z.coerce.number().int().positive().default(300),
  mode: z.enum(['chat', 'route']).default('chat'),
  auth: z.enum(['hmac', 'token']).default('hmac'),
  eventTypeHeader: z.string().min(1).optional(),
  deliveryIdHeader: z.string().min(1).optional(),
})
