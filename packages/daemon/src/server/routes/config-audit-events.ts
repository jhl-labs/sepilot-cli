import type { AuditLogCapabilities } from '../runtime/capabilities.js'
import type { z } from 'zod'
import type { resolveChannelPipelineHealthConfig } from '../runtime/channel-pipeline-health.js'
import type { resolveWebhookSecurityHealthConfig } from '../runtime/webhook-security-health.js'
import type { resolveWebhookSecurityPolicy } from '../runtime/webhook-security-policy.js'
import {
  CHANNEL_PIPELINE_HEALTH_CONFIG_AUDIT_EVENT,
  WEBHOOK_SECURITY_HEALTH_CONFIG_AUDIT_EVENT,
  WEBHOOK_SECURITY_POLICY_CONFIG_AUDIT_EVENT,
  configChannelPipelineHealthHistoryEntrySchema,
  type configChannelPipelineHealthHistoryResponseSchema,
  configChannelPipelineHistoryCursorPayloadSchema,
  configWebhookSecurityHealthHistoryCursorPayloadSchema,
  configWebhookSecurityHealthHistoryEntrySchema,
  type configWebhookSecurityHealthHistoryResponseSchema,
  configWebhookSecurityPolicyHistoryCursorPayloadSchema,
  configWebhookSecurityPolicyHistoryEntrySchema,
  type configWebhookSecurityPolicyHistoryResponseSchema,
  type ConfigChannelPipelineHealthBody,
  type ConfigChannelPipelineHealthHistoryEntry,
  type ConfigChannelPipelineHealthHistoryQuery,
  type ConfigWebhookSecurityHealthBody,
  type ConfigWebhookSecurityHealthHistoryEntry,
  type ConfigWebhookSecurityHealthHistoryQuery,
  type ConfigWebhookSecurityPolicyBody,
  type ConfigWebhookSecurityPolicyHistoryEntry,
  type ConfigWebhookSecurityPolicyHistoryQuery,
} from './config-schema.js'

interface ConfigAuditHistoryEntryBase {
  timestamp: string
  device: string
  route: string
}

interface ConfigAuditHistoryQueryBase {
  limit: number
  cursor?: string
  device?: string
  since?: string
  route?: string
}

interface ConfigAuditHistoryResponse<TEntry> {
  data: TEntry[]
  meta: {
    limit: number
    returned: number
    nextCursor: string | null
  }
}

function channelPipelineHealthConfigsEqual(
  left: ReturnType<typeof resolveChannelPipelineHealthConfig>,
  right: ReturnType<typeof resolveChannelPipelineHealthConfig>,
): boolean {
  return JSON.stringify(left) === JSON.stringify(right)
}

function webhookSecurityHealthConfigsEqual(
  left: ReturnType<typeof resolveWebhookSecurityHealthConfig>,
  right: ReturnType<typeof resolveWebhookSecurityHealthConfig>,
): boolean {
  return JSON.stringify(left) === JSON.stringify(right)
}

function webhookSecurityPolicyConfigsEqual(
  left: ReturnType<typeof resolveWebhookSecurityPolicy>,
  right: ReturnType<typeof resolveWebhookSecurityPolicy>,
): boolean {
  return JSON.stringify(left) === JSON.stringify(right)
}

function listConfigAuditHistory<
  TEntry extends ConfigAuditHistoryEntryBase,
  TQuery extends ConfigAuditHistoryQueryBase,
>(
  auditEvents: unknown[],
  query: TQuery,
  entrySchema: z.ZodTypeAny,
  cursorPayloadSchema: z.ZodTypeAny,
): ConfigAuditHistoryResponse<TEntry> {
  const entries = auditEvents
    .map((event) => entrySchema.safeParse(event))
    .filter(
      (
        result,
      ): result is {
        success: true
        data: TEntry
      } => result.success,
    )
    .map((result) => result.data as TEntry)
    .filter((event) => !query.device || event.device === query.device)
    .filter((event) => !query.since || event.timestamp >= query.since)
    .filter((event) => !query.route || event.route === query.route)
    .sort((left, right) => right.timestamp.localeCompare(left.timestamp))

  let startIndex = 0
  if (query.cursor) {
    let decodedCursor: Pick<TEntry, 'timestamp' | 'route' | 'device'> | null = null
    try {
      const parsed = JSON.parse(
        Buffer.from(query.cursor, 'base64url').toString('utf-8'),
      )
      decodedCursor = cursorPayloadSchema.parse(parsed) as Pick<TEntry, 'timestamp' | 'route' | 'device'>
    } catch {
      throw new Error('INVALID_HISTORY_CURSOR')
    }

    const cursorIndex = entries.findIndex(
      (entry) =>
        entry.timestamp === decodedCursor.timestamp
        && entry.route === decodedCursor.route
        && entry.device === decodedCursor.device,
    )
    if (cursorIndex < 0) {
      throw new Error('INVALID_HISTORY_CURSOR')
    }
    startIndex = cursorIndex + 1
  }

  const data = entries.slice(startIndex, startIndex + query.limit)
  const hasMore = startIndex + data.length < entries.length
  const nextCursor = hasMore && data.length > 0
    ? Buffer.from(
        JSON.stringify({
          timestamp: data[data.length - 1]!.timestamp,
          route: data[data.length - 1]!.route,
          device: data[data.length - 1]!.device,
        } satisfies Pick<TEntry, 'timestamp' | 'route' | 'device'>),
        'utf-8',
      ).toString('base64url')
    : null

  return {
    data,
    meta: {
      limit: query.limit,
      returned: data.length,
      nextCursor,
    },
  }
}

export async function recordChannelPipelineHealthConfigAuditEvent(
  runtime: AuditLogCapabilities | undefined,
  options: {
    route: '/api/v1/config' | '/api/v1/config/observability/channel-pipeline-health'
    requested: ConfigChannelPipelineHealthBody
    previous: ReturnType<typeof resolveChannelPipelineHealthConfig>
    current: ReturnType<typeof resolveChannelPipelineHealthConfig>
  },
): Promise<void> {
  if (!runtime?.auditLogger) {
    return
  }
  if (channelPipelineHealthConfigsEqual(options.previous, options.current)) {
    return
  }

  await runtime.auditLogger.log({
    timestamp: new Date().toISOString(),
    event: CHANNEL_PIPELINE_HEALTH_CONFIG_AUDIT_EVENT,
    device: runtime.config.device.name,
    route: options.route,
    configKey: 'observability.channelPipelineHealth',
    requested: options.requested,
    previous: options.previous,
    current: options.current,
  })
}

export async function recordWebhookSecurityHealthConfigAuditEvent(
  runtime: AuditLogCapabilities | undefined,
  options: {
    route: '/api/v1/config' | '/api/v1/config/observability/webhook-security-health'
    requested: ConfigWebhookSecurityHealthBody
    previous: ReturnType<typeof resolveWebhookSecurityHealthConfig>
    current: ReturnType<typeof resolveWebhookSecurityHealthConfig>
  },
): Promise<void> {
  if (!runtime?.auditLogger) {
    return
  }
  if (webhookSecurityHealthConfigsEqual(options.previous, options.current)) {
    return
  }

  await runtime.auditLogger.log({
    timestamp: new Date().toISOString(),
    event: WEBHOOK_SECURITY_HEALTH_CONFIG_AUDIT_EVENT,
    device: runtime.config.device.name,
    route: options.route,
    configKey: 'observability.webhookSecurityHealth',
    requested: options.requested,
    previous: options.previous,
    current: options.current,
  })
}

export async function recordWebhookSecurityPolicyConfigAuditEvent(
  runtime: AuditLogCapabilities | undefined,
  options: {
    route: '/api/v1/config' | '/api/v1/config/security/webhooks'
    requested: ConfigWebhookSecurityPolicyBody
    previous: ReturnType<typeof resolveWebhookSecurityPolicy>
    current: ReturnType<typeof resolveWebhookSecurityPolicy>
  },
): Promise<void> {
  if (!runtime?.auditLogger) {
    return
  }
  if (webhookSecurityPolicyConfigsEqual(options.previous, options.current)) {
    return
  }

  await runtime.auditLogger.log({
    timestamp: new Date().toISOString(),
    event: WEBHOOK_SECURITY_POLICY_CONFIG_AUDIT_EVENT,
    device: runtime.config.device.name,
    route: options.route,
    configKey: 'security.webhooks',
    requested: options.requested,
    previous: options.previous,
    current: options.current,
  })
}

export function listChannelPipelineHealthConfigAuditEvents(
  auditEvents: unknown[],
  query: ConfigChannelPipelineHealthHistoryQuery,
): z.infer<typeof configChannelPipelineHealthHistoryResponseSchema> {
  return listConfigAuditHistory<ConfigChannelPipelineHealthHistoryEntry, typeof query>(
    auditEvents,
    query,
    configChannelPipelineHealthHistoryEntrySchema,
    configChannelPipelineHistoryCursorPayloadSchema,
  )
}

export function listWebhookSecurityHealthConfigAuditEvents(
  auditEvents: unknown[],
  query: ConfigWebhookSecurityHealthHistoryQuery,
): z.infer<typeof configWebhookSecurityHealthHistoryResponseSchema> {
  return listConfigAuditHistory<ConfigWebhookSecurityHealthHistoryEntry, typeof query>(
    auditEvents,
    query,
    configWebhookSecurityHealthHistoryEntrySchema,
    configWebhookSecurityHealthHistoryCursorPayloadSchema,
  )
}

export function listWebhookSecurityPolicyConfigAuditEvents(
  auditEvents: unknown[],
  query: ConfigWebhookSecurityPolicyHistoryQuery,
): z.infer<typeof configWebhookSecurityPolicyHistoryResponseSchema> {
  return listConfigAuditHistory<ConfigWebhookSecurityPolicyHistoryEntry, typeof query>(
    auditEvents,
    query,
    configWebhookSecurityPolicyHistoryEntrySchema,
    configWebhookSecurityPolicyHistoryCursorPayloadSchema,
  )
}
