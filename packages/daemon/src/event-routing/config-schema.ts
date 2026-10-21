import { z } from 'zod'
import { rejectPrivateLiteralUrl } from '../utils/ssrf-guard.js'

export const REDACTED = '***redacted***'
const ID_PATTERN = /^[a-z0-9][a-z0-9-]{0,63}$/
const idSchema = z.string().regex(ID_PATTERN, 'id must match [a-z0-9][a-z0-9-]{0,63}')

const egressUrlSchema = z.string().url().superRefine((value, ctx) => {
  const parsed = new URL(value)
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'url must be http(s)' })
    return
  }
  if (parsed.username || parsed.password) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'url must not embed credentials' })
    return
  }
  if (process.env.SEPILOTD_OUTBOUND_WEBHOOK_ALLOW_PRIVATE === '1') return
  try {
    rejectPrivateLiteralUrl(parsed)
  } catch {
    ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'url must not target a private/loopback/metadata address' })
  }
})

export const messengerPresetSchema = z.enum(['mattermost', 'slack', 'discord'])
export type MessengerPreset = z.infer<typeof messengerPresetSchema>

const enabled = z.boolean().default(true)

export const destinationSchema = z.discriminatedUnion('kind', [
  z.object({ id: idSchema, kind: z.literal('messenger-webhook'), enabled, preset: messengerPresetSchema, url: egressUrlSchema }),
  z.object({
    id: idSchema,
    kind: z.literal('http'),
    enabled,
    url: egressUrlSchema,
    secret: z.string().min(1).optional(),
    headers: z.record(z.string()).default({}),
  }),
  z.object({ id: idSchema, kind: z.literal('channel'), enabled, channelType: z.string().min(1), channelTarget: z.string().min(1) }),
  z.object({ id: idSchema, kind: z.literal('notification'), enabled, topic: z.string().min(1).optional() }),
])
export type EventDestination = z.infer<typeof destinationSchema>

const scalarSchema = z.union([z.string(), z.number(), z.boolean()])
const whereValueSchema = z.union([scalarSchema, z.array(scalarSchema).min(1)])

export const LLM_PRIVILEGES = ['readonly'] as const

export const DEFAULT_RULE_RATE_LIMIT_PER_MINUTE = 10

export const MAX_CONCURRENT_EVENT_LLM_RUNS = 3

export const llmProcessSchema = z.object({
  mode: z.literal('llm'),
  prompt: z.string().trim().min(1).max(8000),
  bot: z.string().trim().min(1).max(64)
    .refine((bot) => bot.toLowerCase() !== 'auto', { message: 'bot must name a role bot; `auto` is not allowed for event routing' })
    .optional(),
  project: z.string().trim().min(1).max(128).optional(),
  network: z.boolean().default(false),
  privilege: z.enum(LLM_PRIVILEGES).default('readonly'),
})
export type LlmProcess = z.infer<typeof llmProcessSchema>

const processSchema = z
  .discriminatedUnion('mode', [z.object({ mode: z.literal('passthrough') }), llmProcessSchema])
  .superRefine((process, ctx) => {
    if (process.mode === 'llm' && process.network && !process.bot) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, path: ['network'], message: 'network requires a bot; a turn without a bot has no tools' })
    }
  })
  .default({ mode: 'passthrough' })

export const ruleSchema = z.object({
  id: idSchema,
  enabled,
  match: z.object({
    source: z.string().min(1),
    types: z.array(z.string().min(1)).default([]),
    where: z.record(whereValueSchema).default({}),
  }),
  process: processSchema,
  rateLimitPerMinute: z.number().int().min(1).max(600).default(DEFAULT_RULE_RATE_LIMIT_PER_MINUTE),
  deliverOnFailure: z.boolean().default(true),
  deliver: z.array(idSchema).min(1),
  template: z.string().min(1).max(8000).optional(),
})
export type EventRule = z.infer<typeof ruleSchema>

export const DEFAULT_EVENT_ROUTING_LIMITS = { maxConcurrentLlmRuns: 2, llmRunTimeoutMs: 300_000 }

const limitsSchema = z
  .object({
    maxConcurrentLlmRuns: z.number().int().min(1).max(MAX_CONCURRENT_EVENT_LLM_RUNS).default(DEFAULT_EVENT_ROUTING_LIMITS.maxConcurrentLlmRuns),
    llmRunTimeoutMs: z.number().int().min(10_000).max(1_800_000).default(DEFAULT_EVENT_ROUTING_LIMITS.llmRunTimeoutMs),
  })
  .default(DEFAULT_EVENT_ROUTING_LIMITS)

export const DEFAULT_EVENT_ROUTING_CONFIG = { enabled: true, limits: DEFAULT_EVENT_ROUTING_LIMITS, destinations: [], rules: [] }

export const eventRoutingConfigSchema = z
  .object({
    enabled,
    limits: limitsSchema,
    destinations: z.array(destinationSchema).default([]),
    rules: z.array(ruleSchema).default([]),
  })
  .superRefine((cfg, ctx) => {
    const destinationIds = new Set<string>()
    cfg.destinations.forEach((destination, index) => {
      if (destinationIds.has(destination.id)) {
        ctx.addIssue({ code: z.ZodIssueCode.custom, path: ['destinations', index, 'id'], message: `duplicate destination id: ${destination.id}` })
      }
      destinationIds.add(destination.id)
    })
    const ruleIds = new Set<string>()
    cfg.rules.forEach((rule, index) => {
      if (ruleIds.has(rule.id)) {
        ctx.addIssue({ code: z.ZodIssueCode.custom, path: ['rules', index, 'id'], message: `duplicate rule id: ${rule.id}` })
      }
      ruleIds.add(rule.id)
      rule.deliver.forEach((destinationId, deliverIndex) => {
        if (!destinationIds.has(destinationId)) {
          ctx.addIssue({ code: z.ZodIssueCode.custom, path: ['rules', index, 'deliver', deliverIndex], message: `unknown destination: ${destinationId}` })
        }
      })
    })
  })
  .default(DEFAULT_EVENT_ROUTING_CONFIG)
export type EventRoutingConfig = z.infer<typeof eventRoutingConfigSchema>

export function redactUrl(url: string): string {
  const parsed = new URL(url)
  if ((parsed.pathname === '/' || parsed.pathname === '') && !parsed.search) return `${parsed.origin}/`
  return `${parsed.origin}/${REDACTED}`
}

export function redactDestination(destination: EventDestination): EventDestination {
  switch (destination.kind) {
    case 'messenger-webhook':
      return { ...destination, url: redactUrl(destination.url) }
    case 'http':
      return {
        ...destination,
        url: redactUrl(destination.url),
        ...(destination.secret ? { secret: REDACTED } : {}),
        headers: Object.fromEntries(Object.keys(destination.headers).map((key) => [key, REDACTED])),
      }
    default:
      return destination
  }
}

export function redactEventRoutingConfig(cfg: EventRoutingConfig): EventRoutingConfig {
  return { ...cfg, destinations: cfg.destinations.map(redactDestination) }
}

export type EventDestinationInput = z.input<typeof destinationSchema>

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function restoreDestinationSecrets<T extends EventDestinationInput>(
  existing: EventDestination | undefined,
  incoming: T,
): T {
  if (!existing || existing.kind !== incoming.kind) return incoming
  if (incoming.kind === 'messenger-webhook' && existing.kind === 'messenger-webhook') {
    return incoming.url === redactUrl(existing.url) ? { ...incoming, url: existing.url } : incoming
  }
  if (incoming.kind === 'http' && existing.kind === 'http') {
    const { headers } = incoming
    return {
      ...incoming,
      url: incoming.url === redactUrl(existing.url) ? existing.url : incoming.url,
      secret: incoming.secret === REDACTED ? existing.secret : incoming.secret,
      ...(isPlainRecord(headers)
        ? {
            headers: Object.fromEntries(
              Object.entries(headers).map(([key, value]) => [
                key,
                value === REDACTED && existing.headers[key] !== undefined ? existing.headers[key]! : value,
              ]),
            ),
          }
        : {}),
    }
  }
  return incoming
}
