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

/**
 * Validate a user-supplied webhook path. Mirrors the checks in
 * `packages/cli/src/commands/webhooks.ts` `assertSafeWebhookPath`
 * to provide defense-in-depth against direct (non-CLI) HTTP callers.
 *
 * Returns the trimmed path on success. Throws on:
 * - empty / whitespace-only input
 * - any '..' path segment (path traversal)
 * - any whitespace or control char
 * - URL delimiter / escape chars that would make the public route ambiguous
 * - first path segments reserved for built-in webhook or management routes
 *
 * Note: '..foo' / 'foo..' are accepted because they are not '..' segments -
 * matches CLI behavior verbatim.
 */
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

/**
 * zod schema that validates a webhook path field. Use as a drop-in
 * replacement for `z.string().min(1)` in any schema that accepts a
 * user-supplied webhook path.
 *
 * Returns the trimmed string on success. On failure, zod produces a
 * standard validation error and Fastify renders 400.
 */
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
  /**
   * Opt-in replay protection. Operators set `requireTimestamp: true`
   * when the upstream sender includes a timestamp header (and signs it
   * into the body, e.g. Slack's X-Slack-Request-Timestamp pattern).
   * Default false to preserve existing integrations that only send a
   * raw HMAC over the body.
   */
  requireTimestamp: z.boolean().default(false),
  timestampHeader: z.string().min(1).default('x-webhook-timestamp'),
  /** Max age in seconds for a delivery before it is treated as replayed. */
  timestampMaxSkewSeconds: z.coerce.number().int().positive().default(300),
})
