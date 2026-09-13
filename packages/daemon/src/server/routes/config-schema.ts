import { z } from 'zod'
import {
  channelPipelineConfigSchema,
  channelPipelineHealthConfigSchema,
  commandHookSchema,
  configSchema,
  agentGraphNodeModelOverridesSchema,
  memoryCustomApiSchema,
  memoryHttpAuthSchema,
  memoryMaintenanceSchema,
  memoryMeilisearchSchema,
  memoryQdrantSchema,
  memoryRagSettingsSchema,
  memorySearchEngineSchema,
  mcpClientConfigSchema,
  mcpServerSchema,
  outboundHookEventSchema,
  outboundWebhookRetrySchema,
  outboundWebhookSchema,
  providerSchema,
  schedulerConfigSchema,
  schedulerSurfaceAccessSchema,
  skillSourceSecuritySchema,
  skillSourceSecurityUpdateSchema,
  webhookSecurityHealthConfigSchema,
  webhookSecurityPolicyConfigSchema,
  webSearchProviderUpdateSchema,
  webSearchTrustedDomainsSchema,
} from '../../config/schema.js'
import { configChannelsUpdateSchema } from '../../config/channel-update-schema.js'
import { webhookEndpointConfigSchema } from '../../channels/webhook-validation.js'
import { OUTBOUND_WEBHOOK_DELIVERY_AUDIT_EVENT } from '../../hooks/outbound-webhook.js'
import { MODEL_PROBE_TIMEOUT_MS } from '../../providers/model-probe.js'

export {
  channelPipelineConfigSchema,
  channelPipelineHealthConfigSchema,
  commandHookSchema,
  configSchema,
  agentGraphNodeModelOverridesSchema,
  memoryCustomApiSchema,
  memoryHttpAuthSchema,
  memoryMaintenanceSchema,
  memoryMeilisearchSchema,
  memoryQdrantSchema,
  memoryRagSettingsSchema,
  memorySearchEngineSchema,
  mcpClientConfigSchema,
  mcpServerSchema,
  outboundHookEventSchema,
  outboundWebhookRetrySchema,
  outboundWebhookSchema,
  providerSchema,
  schedulerConfigSchema,
  schedulerSurfaceAccessSchema,
  skillSourceSecuritySchema,
  skillSourceSecurityUpdateSchema,
  webhookSecurityHealthConfigSchema,
  webhookSecurityPolicyConfigSchema,
  webSearchProviderUpdateSchema,
  webSearchTrustedDomainsSchema,
}

export const CHANNEL_PIPELINE_HEALTH_CONFIG_AUDIT_EVENT = 'config.channel_pipeline_health.updated'
export const WEBHOOK_SECURITY_HEALTH_CONFIG_AUDIT_EVENT = 'config.webhook_security_health.updated'
export const WEBHOOK_SECURITY_POLICY_CONFIG_AUDIT_EVENT = 'config.webhook_security.updated'
export const OUTBOUND_WEBHOOK_DEAD_LETTER_ACK_AUDIT_EVENT = 'hook.outbound_webhook.dead_letter.ack'

export const modelCapabilitiesSchema = z.object({
  vision: z.boolean(),
  toolUse: z.boolean(),
  streaming: z.boolean(),
  embedding: z.boolean(),
  thinking: z.boolean(),
  adaptivePromptReact: z.boolean().optional(),
  promptReactPreferred: z.boolean().optional(),
  deepCoderAnalysis: z.boolean().optional(),
})

export const modelCompatibilitySchema = z.object({
  toolTransport: z.enum(['auto', 'native', 'prompt-react', 'adaptive']).optional(),
  answerProtocol: z.enum(['auto', 'repair-left-truncated-answer-stem']).optional(),
  notes: z.array(z.string()).optional(),
})

export const modelInfoSchema = z.object({
  id: z.string(),
  name: z.string(),
  contextWindow: z.number().int(),
  maxOutputTokens: z.number().int(),
  capabilities: modelCapabilitiesSchema,
  compatibility: modelCompatibilitySchema.optional(),
  inputCostPer1k: z.number().optional(),
  outputCostPer1k: z.number().optional(),
})

export const configProviderHealthSchema = z.object({
  status: z.enum(['ready', 'env_missing', 'unavailable']),
  message: z.string().optional(),
  missingEnvVars: z.array(z.string()),
})

export const configProviderInfoSchema = z.object({
  id: z.string(),
  name: z.string(),
  models: z.array(modelInfoSchema),
  supportsEmbedding: z.boolean(),
  embeddingModelIds: z.array(z.string()),
  configuredModelIds: z.array(z.string()),
  modelCatalogAuthority: z.enum(['configured', 'endpoint']).optional(),
  unavailableConfiguredModelIds: z.array(z.string()).optional(),
  health: configProviderHealthSchema,
})

export const webhookEndpointSchema = webhookEndpointConfigSchema

export const configUpdateRequestSchema = z
  .object({
    'agent.mode': z.string().trim().min(1).optional(),
    'agent.defaultModel': z.string().min(1).optional(),
    'agent.auxModel': z.string().min(1).optional(),
    'agent.defaultProvider': z.string().min(1).optional(),
    'agent.autonomy': z
      .enum(['readonly', 'accept-edits', 'workspace-write', 'supervised', 'autonomous'])
      .optional(),
    'agent.thinkingLevel': z.enum(['auto', 'off', 'low', 'medium', 'high', 'max']).optional(),
    'agent.disabledTools': z.array(z.string().trim().min(1)).max(1000).optional(),
    'agent.graphNodeModelOverrides': agentGraphNodeModelOverridesSchema.optional(),
    'device.name': z.string().min(1).optional(),
    'webSearch.trustedDomains': webSearchTrustedDomainsSchema.optional(),
    'webSearch.provider': webSearchProviderUpdateSchema.optional(),
    channels: configChannelsUpdateSchema.optional(),
    'channels.sharedGroupContext': z.boolean().optional(),
    'channels.maxGlobalRuns': z.number().int().min(1).max(100).optional(),
    'channels.maxGlobalQueuedRuns': z.number().int().min(0).max(100).optional(),
    'channelPipeline.sharedGroupContext': z.boolean().optional(),
    'channelPipeline.maxGlobalRuns': z.number().int().min(1).max(100).optional(),
    'channelPipeline.maxGlobalQueuedRuns': z.number().int().min(0).max(100).optional(),
    'channelPipeline.defaultWorkspaceRoot': z.union([
      z.string().trim().min(1),
      z.null(),
    ]).optional(),
    providers: z.array(providerSchema).optional(),
    'mcp.servers': z.array(mcpServerSchema).optional(),
    'mcp.client': mcpClientConfigSchema.optional(),
    'memory.vectorBackend': z
      .enum([
        'auto',
        'sqlite-vec',
        'sqlite-scan',
        'qdrant',
        'opensearch',
        'elasticsearch',
        'meilisearch',
        'custom-api',
      ])
      .optional(),
    'memory.embeddingProvider': z.union([z.string().trim(), z.null()]).optional(),
    'memory.embeddingModel': z.union([z.string().trim(), z.null()]).optional(),
    'memory.qdrant': memoryQdrantSchema.optional(),
    'memory.opensearch': memorySearchEngineSchema.optional(),
    'memory.elasticsearch': memorySearchEngineSchema.optional(),
    'memory.meilisearch': memoryMeilisearchSchema.optional(),
    'memory.customApi': memoryCustomApiSchema.optional(),
    'memory.rag': memoryRagSettingsSchema.optional(),
    'memory.maintenance': memoryMaintenanceSchema.optional(),
    scheduler: schedulerConfigSchema.optional(),
    'scheduler.timezone': z.union([z.string().trim().min(1), z.null()]).optional(),
    'scheduler.surfaces': schedulerSurfaceAccessSchema.optional(),
    'scheduler.surfaces.cli': z.boolean().optional(),
    'scheduler.surfaces.desktop': z.boolean().optional(),
    'scheduler.surfaces.mobile': z.boolean().optional(),
    'hooks.outboundWebhooks': z.array(outboundWebhookSchema).optional(),
    'hooks.commandHooks': z.array(commandHookSchema).optional(),
    'security.skillSources': skillSourceSecurityUpdateSchema.optional(),
    'security.webhooks': webhookSecurityPolicyConfigSchema.partial().optional(),
    'observability.channelPipelineHealth': channelPipelineHealthConfigSchema.partial().optional(),
    'observability.webhookSecurityHealth': webhookSecurityHealthConfigSchema.partial().optional(),
  })
  .catchall(z.unknown())

export const mcpServerNameParamsSchema = z.object({
  name: z.string().min(1),
})

export const mcpServerToggleRequestSchema = z.object({
  enabled: z.boolean(),
})

export const mcpServerDisabledToolsRequestSchema = z.object({
  disabledTools: z.array(z.string().trim().min(1)),
})

export const outboundWebhookIdParamsSchema = z.object({
  id: z.string().min(1),
})

export const outboundWebhookToggleRequestSchema = z.object({
  enabled: z.boolean(),
})

export const outboundWebhookDeliveryQuerySchema = z.object({
  id: z.string().min(1).optional(),
  status: z.enum(['success', 'error']).optional(),
  cursor: z.string().min(1).optional(),
  limit: z.coerce.number().int().min(1).max(200).optional(),
})

export const outboundWebhookDeadLetterQuerySchema = z.object({
  id: z.string().min(1).optional(),
  state: z.enum(['open', 'acknowledged', 'all']).optional(),
  cursor: z.string().min(1).optional(),
  limit: z.coerce.number().int().min(1).max(200).optional(),
})

export const outboundWebhookDeadLetterAckParamsSchema = z.object({
  rootDeliveryId: z.string().min(1),
})

export const outboundWebhookDeadLetterAckRequestSchema = z
  .object({
    note: z.string().trim().min(1).max(500).optional(),
  })
  .default({})

export const outboundWebhookDeadLetterAckHistoryQuerySchema = z.object({
  id: z.string().min(1).optional(),
  rootDeliveryId: z.string().min(1).optional(),
  latestDeliveryId: z.string().min(1).optional(),
  device: z.string().min(1).optional(),
  since: z.string().datetime().optional(),
  cursor: z.string().min(1).optional(),
  limit: z.coerce.number().int().min(1).max(200).optional().default(20),
})

export const outboundWebhookDeadLetterTimelineQuerySchema = z.object({
  cursor: z.string().min(1).optional(),
  limit: z.coerce.number().int().min(1).max(200).optional().default(100),
})

export const outboundWebhookDeliveryReplayParamsSchema = z.object({
  deliveryId: z.string().min(1),
})

export const outboundWebhookDeliveryReplayRequestSchema = z
  .object({
    force: z.boolean().default(false),
  })
  .default({ force: false })

export const webhookEndpointIdParamsSchema = z.object({
  id: z.string().min(1),
})

export const webhookEndpointToggleRequestSchema = z.object({
  enabled: z.boolean(),
})

export const telegramChannelConfigSchema = z.object({
  enabled: z.boolean().default(true),
  botToken: z.string().trim().min(1),
  allowedUsers: z.array(z.string().trim().min(1)).default([]),
  pairingRequired: z.boolean().default(true),
  pairingCodeTtl: z.number().int().min(30).max(3600).default(300),
  rateLimitPerMinute: z.number().int().min(1).max(1000).default(30),
})

export const slackChannelConfigSchema = z.object({
  enabled: z.boolean().default(true),
  botToken: z.string().trim().min(1),
  signingSecret: z.string().trim().min(1),
  allowedChannels: z.array(z.string().trim().min(1)).default([]),
  allowedUsers: z.array(z.string().trim().min(1)).default([]),
})

export const discordPublicKeySchema = z
  .string()
  .regex(/^[0-9a-f]{64}$/i, 'Discord publicKey must be a 64-character hex string')

export const discordChannelConfigSchema = z.object({
  enabled: z.boolean().default(true),
  botToken: z.string().trim().min(1),
  applicationId: z.string().trim().min(1),
  publicKey: discordPublicKeySchema,
  allowedGuilds: z.array(z.string().trim().min(1)).default([]),
  allowedChannels: z.array(z.string().trim().min(1)).default([]),
  allowedUsers: z.array(z.string().trim().min(1)).default([]),
})

export const mattermostChannelConfigSchema = z.object({
  enabled: z.boolean().default(true),
  serverUrl: z
    .string()
    .trim()
    .url()
    .refine(
      (value) => ['http:', 'https:'].includes(new URL(value).protocol),
      'Mattermost serverUrl must use http or https',
    ),
  botToken: z.string().trim().min(1),
  webhookToken: z.string().trim().min(1),
  allowedTeams: z.array(z.string().trim().min(1)).default([]),
  allowedChannels: z.array(z.string().trim().min(1)).default([]),
  allowedUsers: z.array(z.string().trim().min(1)).default([]),
})

export const telegramChannelToggleRequestSchema = z.object({
  enabled: z.boolean(),
})

export const channelToggleRequestSchema = z.object({
  enabled: z.boolean(),
})

export const telegramAllowedUserParamsSchema = z.object({
  userId: z.string().min(1),
})

export const telegramPairingCodeSchema = z.object({
  code: z.string().length(6),
  expiresAt: z.string().datetime(),
})

export const channelStatusSchema = z.enum(['connected', 'disconnected', 'connecting', 'error'])

export const outboundWebhookSummarySchema = z.object({
  id: z.string(),
  enabled: z.boolean(),
  url: z.string().url(),
  events: z.array(outboundHookEventSchema),
  hasSecret: z.boolean(),
  headerKeys: z.array(z.string()),
  retry: outboundWebhookRetrySchema,
})

export const outboundWebhookDeliverySchema = z.object({
  timestamp: z.string(),
  deliveryId: z.string(),
  webhookId: z.string(),
  url: z.string().url(),
  hookEvent: outboundHookEventSchema,
  deliveryStatus: z.enum(['success', 'error']),
  attemptCount: z.number().int().min(1).default(1),
  statusCode: z.number().int().optional(),
  durationMs: z.number().int(),
  error: z.string().optional(),
  sessionId: z.string().optional(),
  replayedFromDeliveryId: z.string().optional(),
})

export const outboundWebhookDeliveryCursorPayloadSchema = z.object({
  timestamp: z.string().datetime(),
  deliveryId: z.string(),
  webhookId: z.string(),
})

export const outboundWebhookDeliveryPayloadSnapshotSchema = z.object({
  event: outboundHookEventSchema,
  data: z.record(z.string(), z.unknown()),
})

export const outboundWebhookDeliveryReplayChildSchema = z.object({
  deliveryId: z.string(),
  timestamp: z.string().datetime(),
  deliveryStatus: z.enum(['success', 'error']),
  attemptCount: z.number().int().min(1),
  statusCode: z.number().int().optional(),
  error: z.string().optional(),
  deliveryDetailPath: z.string(),
})

export const outboundWebhookDeliveryLineageEntrySchema = z.object({
  deliveryId: z.string(),
  timestamp: z.string().datetime(),
  device: z.string(),
  deliveryStatus: z.enum(['success', 'error']),
  attemptCount: z.number().int().min(1),
  statusCode: z.number().int().optional(),
  error: z.string().optional(),
  sessionId: z.string().optional(),
  replayedFromDeliveryId: z.string().optional(),
  depth: z.number().int().min(1),
  deliveryDetailPath: z.string(),
})

export const outboundWebhookDeliveryDetailSchema = outboundWebhookDeliverySchema.extend({
  selfPath: z.string(),
  device: z.string(),
  rootDeliveryId: z.string(),
  hasPayloadSnapshot: z.boolean(),
  payloadSnapshot: outboundWebhookDeliveryPayloadSnapshotSchema.optional(),
  replayChildren: z.array(outboundWebhookDeliveryReplayChildSchema),
  replayAncestors: z.array(outboundWebhookDeliveryLineageEntrySchema),
  replayDescendants: z.array(outboundWebhookDeliveryLineageEntrySchema),
})

export const outboundWebhookDeadLetterSchema = z.object({
  rootDeliveryId: z.string(),
  latestDeliveryId: z.string(),
  selfPath: z.string(),
  timelinePath: z.string(),
  latestDeliveryDetailPath: z.string(),
  acknowledgementsPath: z.string(),
  webhookId: z.string(),
  url: z.string().url(),
  hookEvent: outboundHookEventSchema,
  firstFailedAt: z.string(),
  lastAttemptAt: z.string(),
  replayCount: z.number().int().min(0),
  attemptCount: z.number().int().min(1),
  statusCode: z.number().int().optional(),
  durationMs: z.number().int(),
  error: z.string().optional(),
  sessionId: z.string().optional(),
  state: z.enum(['open', 'acknowledged']),
  latestAcknowledgement: z
    .object({
      timestamp: z.string().datetime(),
      device: z.string(),
      note: z.string().optional(),
    })
    .optional(),
  acknowledgedAt: z.string().datetime().optional(),
  acknowledgedByDevice: z.string().optional(),
  acknowledgmentNote: z.string().optional(),
})

export const outboundWebhookDeadLetterCursorPayloadSchema = z.object({
  lastAttemptAt: z.string().datetime(),
  rootDeliveryId: z.string(),
  latestDeliveryId: z.string(),
})

export const outboundWebhookDeadLetterAckHistoryCursorPayloadSchema = z.object({
  timestamp: z.string().datetime(),
  rootDeliveryId: z.string(),
  latestDeliveryId: z.string(),
  device: z.string(),
})

export const outboundWebhookDeadLetterTimelineCursorPayloadSchema = z.object({
  timestamp: z.string().datetime(),
  type: z.enum(['delivery', 'acknowledgement']),
  key: z.string().min(1),
})

export const outboundWebhookDeliveryAuditRecordSchema = outboundWebhookDeliverySchema.extend({
  event: z.literal(OUTBOUND_WEBHOOK_DELIVERY_AUDIT_EVENT),
  device: z.string(),
  session: z.string().optional(),
  payload: z
    .object({
      event: outboundHookEventSchema,
      data: z.record(z.string(), z.unknown()),
    })
    .optional(),
})

export const outboundWebhookDeadLetterAckAuditRecordSchema = z.object({
  timestamp: z.string().datetime(),
  event: z.literal(OUTBOUND_WEBHOOK_DEAD_LETTER_ACK_AUDIT_EVENT),
  device: z.string(),
  rootDeliveryId: z.string(),
  latestDeliveryId: z.string(),
  webhookId: z.string(),
  note: z.string().optional(),
})

export const outboundWebhookDeadLetterAckHistoryEntrySchema =
  outboundWebhookDeadLetterAckAuditRecordSchema.extend({
    deadLetterPath: z.string(),
    latestDeliveryDetailPath: z.string(),
  })

export const outboundWebhookDeadLetterTimelineDeliveryEntrySchema = z.object({
  type: z.literal('delivery'),
  timestamp: z.string().datetime(),
  rootDeliveryId: z.string(),
  deliveryId: z.string(),
  deliveryDetailPath: z.string(),
  webhookId: z.string(),
  url: z.string().url(),
  hookEvent: outboundHookEventSchema,
  deliveryStatus: z.enum(['success', 'error']),
  attemptCount: z.number().int().min(1),
  statusCode: z.number().int().optional(),
  durationMs: z.number().int(),
  error: z.string().optional(),
  sessionId: z.string().optional(),
  replayedFromDeliveryId: z.string().optional(),
})

export const outboundWebhookDeadLetterTimelineAcknowledgementEntrySchema =
  outboundWebhookDeadLetterAckHistoryEntrySchema.extend({
    type: z.literal('acknowledgement'),
  })

export const outboundWebhookDeadLetterTimelineEntrySchema = z.discriminatedUnion('type', [
  outboundWebhookDeadLetterTimelineDeliveryEntrySchema,
  outboundWebhookDeadLetterTimelineAcknowledgementEntrySchema,
])

export const webhookEndpointSummarySchema = z.object({
  id: z.string(),
  enabled: z.boolean(),
  path: z.string(),
  publicRoute: z.string(),
  secretHeader: z.string(),
  hasSecretValue: z.boolean(),
  allowedIps: z.array(z.string()),
  allowedEvents: z.array(z.string()),
})

export const telegramChannelSummarySchema = z.object({
  enabled: z.boolean(),
  status: channelStatusSchema,
  hasBotToken: z.boolean(),
  allowedUsers: z.array(z.string()),
  pairingRequired: z.boolean(),
  pairingCodeTtl: z.number().int(),
  rateLimitPerMinute: z.number().int(),
})

export const slackChannelSummarySchema = z.object({
  enabled: z.boolean(),
  status: channelStatusSchema,
  hasBotToken: z.boolean(),
  hasSigningSecret: z.boolean(),
  allowedChannels: z.array(z.string()),
  allowedUsers: z.array(z.string()),
})

export const discordChannelSummarySchema = z.object({
  enabled: z.boolean(),
  status: channelStatusSchema,
  hasBotToken: z.boolean(),
  hasApplicationId: z.boolean(),
  hasPublicKey: z.boolean(),
  allowedGuilds: z.array(z.string()),
  allowedChannels: z.array(z.string()),
  allowedUsers: z.array(z.string()),
})

export const mattermostChannelSummarySchema = z.object({
  enabled: z.boolean(),
  status: channelStatusSchema,
  hasServerUrl: z.boolean(),
  hasBotToken: z.boolean(),
  hasWebhookToken: z.boolean(),
  serverUrl: z.string(),
  allowedTeams: z.array(z.string()),
  allowedChannels: z.array(z.string()),
  allowedUsers: z.array(z.string()),
})

export const configProvidersResponseSchema = z.object({
  data: z.array(configProviderInfoSchema),
})

export const configEnvVarNameSchema = z.string().regex(/^[A-Za-z_][A-Za-z0-9_]*$/)
export const configEnvValueSchema = z.string().nullable()

export function validateEnvUpdateKeys(
  updates: Record<string, string | null>,
  ctx: z.RefinementCtx,
  pathPrefix: Array<string | number>,
): void {
  for (const key of Object.keys(updates)) {
    if (!configEnvVarNameSchema.safeParse(key).success) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: [...pathPrefix, key],
        message: 'Invalid environment variable name.',
      })
    }
  }
}

export const configProviderValidationRequestSchema = z.object({
  provider: providerSchema,
  model: z.string().trim().min(1).optional(),
  timeoutMs: z.number().int().min(1_000).max(30_000).optional().default(MODEL_PROBE_TIMEOUT_MS),
  env: z.record(configEnvVarNameSchema, z.string()).optional(),
})

export const configProviderValidationResponseSchema = z.object({
  data: z.object({
    ok: z.literal(true),
    providerId: z.string(),
    providerType: z.string(),
    model: z.string(),
    latencyMs: z.number().int().min(0),
    message: z.string(),
  }),
})

export const configProviderDiscoverModelsRequestSchema = z.object({
  type: z.string().trim().min(1),
  baseUrl: z.string().trim().url().optional(),
  apiKey: z.string().trim().min(1).optional(),
  headers: z.record(z.string()).optional(),
  timeoutMs: z.number().int().min(1_000).max(30_000).optional().default(10_000),
})

export const configEnvUpdateRequestSchema = z
  .object({
    updates: z.record(configEnvValueSchema),
  })
  .superRefine((value, ctx) => {
    validateEnvUpdateKeys(value.updates, ctx, ['updates'])
  })

export const configEnvUpdateResponseSchema = z.object({
  data: z.object({
    updated: z.array(z.string()),
    removed: z.array(z.string()),
    path: z.string(),
  }),
})

export const configUpdateResponseSchema = z.object({
  data: z.object({
    updated: z.array(z.string()),
  }),
})

export const outboundWebhookListResponseSchema = z.object({
  data: z.array(outboundWebhookSummarySchema),
})

export const outboundWebhookDeliveryListMetaSchema = z.object({
  limit: z.number().int(),
  returned: z.number().int(),
  nextCursor: z.string().nullable(),
})

export const outboundWebhookDeliveryListResponseSchema = z.object({
  data: z.array(outboundWebhookDeliverySchema),
  meta: outboundWebhookDeliveryListMetaSchema,
})

export const outboundWebhookDeliveryDetailResponseSchema = z.object({
  data: outboundWebhookDeliveryDetailSchema,
})

export const outboundWebhookDeadLetterListResponseSchema = z.object({
  data: z.array(outboundWebhookDeadLetterSchema),
  meta: outboundWebhookDeliveryListMetaSchema,
})

export const outboundWebhookDeadLetterAckHistoryResponseSchema = z.object({
  data: z.array(outboundWebhookDeadLetterAckHistoryEntrySchema),
  meta: outboundWebhookDeliveryListMetaSchema,
})

export const outboundWebhookDeadLetterTimelineResponseSchema = z.object({
  data: z.array(outboundWebhookDeadLetterTimelineEntrySchema),
  meta: outboundWebhookDeliveryListMetaSchema,
})

export const outboundWebhookDeliveryResponseSchema = z.object({
  data: outboundWebhookDeliverySchema,
})

export const outboundWebhookDeadLetterResponseSchema = z.object({
  data: outboundWebhookDeadLetterSchema,
})

export const webhookEndpointListResponseSchema = z.object({
  data: z.array(webhookEndpointSummarySchema),
})

export const telegramChannelResponseSchema = z.object({
  data: telegramChannelSummarySchema.nullable(),
})

export const slackChannelResponseSchema = z.object({
  data: slackChannelSummarySchema.nullable(),
})

export const discordChannelResponseSchema = z.object({
  data: discordChannelSummarySchema.nullable(),
})

export const mattermostChannelResponseSchema = z.object({
  data: mattermostChannelSummarySchema.nullable(),
})

export const telegramPairingCodeResponseSchema = z.object({
  data: telegramPairingCodeSchema,
})

export const configChannelPipelineHealthUpdateRequestSchema =
  channelPipelineHealthConfigSchema.partial()

export const configChannelPipelineHealthResponseSchema = z.object({
  data: channelPipelineHealthConfigSchema,
})

export const channelPipelineHealthAuditRouteSchema = z.enum([
  '/api/v1/config',
  '/api/v1/config/observability/channel-pipeline-health',
])

export const configChannelPipelineHealthHistoryCursorSchema = z.string().min(1)

export const configChannelPipelineHealthHistoryQuerySchema = z.object({
  route: channelPipelineHealthAuditRouteSchema.optional(),
  device: z.string().min(1).optional(),
  since: z.string().datetime().optional(),
  cursor: configChannelPipelineHealthHistoryCursorSchema.optional(),
  limit: z.coerce.number().int().min(1).max(200).optional().default(20),
})

export const configChannelPipelineHealthHistoryEntrySchema = z.object({
  timestamp: z.string().datetime(),
  event: z.literal(CHANNEL_PIPELINE_HEALTH_CONFIG_AUDIT_EVENT),
  device: z.string(),
  route: channelPipelineHealthAuditRouteSchema,
  configKey: z.literal('observability.channelPipelineHealth'),
  requested: configChannelPipelineHealthUpdateRequestSchema,
  previous: channelPipelineHealthConfigSchema,
  current: channelPipelineHealthConfigSchema,
})

export const configChannelPipelineHistoryCursorPayloadSchema = z.object({
  timestamp: z.string().datetime(),
  route: channelPipelineHealthAuditRouteSchema,
  device: z.string(),
})

export const configChannelPipelineHealthHistoryMetaSchema = z.object({
  limit: z.number().int(),
  returned: z.number().int(),
  nextCursor: z.string().nullable(),
})

export const configChannelPipelineHealthHistoryResponseSchema = z.object({
  data: z.array(configChannelPipelineHealthHistoryEntrySchema),
  meta: configChannelPipelineHealthHistoryMetaSchema,
})

export const configWebhookSecurityHealthUpdateRequestSchema =
  webhookSecurityHealthConfigSchema.partial()

export const configWebhookSecurityHealthResponseSchema = z.object({
  data: webhookSecurityHealthConfigSchema,
})

export const webhookSecurityHealthAuditRouteSchema = z.enum([
  '/api/v1/config',
  '/api/v1/config/observability/webhook-security-health',
])

export const configWebhookSecurityHealthHistoryCursorSchema = z.string().min(1)

export const configWebhookSecurityHealthHistoryQuerySchema = z.object({
  route: webhookSecurityHealthAuditRouteSchema.optional(),
  device: z.string().min(1).optional(),
  since: z.string().datetime().optional(),
  cursor: configWebhookSecurityHealthHistoryCursorSchema.optional(),
  limit: z.coerce.number().int().min(1).max(200).optional().default(20),
})

export const configWebhookSecurityHealthHistoryEntrySchema = z.object({
  timestamp: z.string().datetime(),
  event: z.literal(WEBHOOK_SECURITY_HEALTH_CONFIG_AUDIT_EVENT),
  device: z.string(),
  route: webhookSecurityHealthAuditRouteSchema,
  configKey: z.literal('observability.webhookSecurityHealth'),
  requested: configWebhookSecurityHealthUpdateRequestSchema,
  previous: webhookSecurityHealthConfigSchema,
  current: webhookSecurityHealthConfigSchema,
})

export const configWebhookSecurityHealthHistoryCursorPayloadSchema = z.object({
  timestamp: z.string().datetime(),
  route: webhookSecurityHealthAuditRouteSchema,
  device: z.string(),
})

export const configWebhookSecurityHealthHistoryMetaSchema = z.object({
  limit: z.number().int(),
  returned: z.number().int(),
  nextCursor: z.string().nullable(),
})

export const configWebhookSecurityHealthHistoryResponseSchema = z.object({
  data: z.array(configWebhookSecurityHealthHistoryEntrySchema),
  meta: configWebhookSecurityHealthHistoryMetaSchema,
})

export const configWebhookSecurityPolicyUpdateRequestSchema =
  webhookSecurityPolicyConfigSchema.partial()

export const configWebhookSecurityPolicyResponseSchema = z.object({
  data: webhookSecurityPolicyConfigSchema,
})

export const webhookSecurityPolicyAuditRouteSchema = z.enum([
  '/api/v1/config',
  '/api/v1/config/security/webhooks',
])

export const configWebhookSecurityPolicyHistoryCursorSchema = z.string().min(1)

export const configWebhookSecurityPolicyHistoryQuerySchema = z.object({
  route: webhookSecurityPolicyAuditRouteSchema.optional(),
  device: z.string().min(1).optional(),
  since: z.string().datetime().optional(),
  cursor: configWebhookSecurityPolicyHistoryCursorSchema.optional(),
  limit: z.coerce.number().int().min(1).max(200).optional().default(20),
})

export const configWebhookSecurityPolicyHistoryEntrySchema = z.object({
  timestamp: z.string().datetime(),
  event: z.literal(WEBHOOK_SECURITY_POLICY_CONFIG_AUDIT_EVENT),
  device: z.string(),
  route: webhookSecurityPolicyAuditRouteSchema,
  configKey: z.literal('security.webhooks'),
  requested: configWebhookSecurityPolicyUpdateRequestSchema,
  previous: webhookSecurityPolicyConfigSchema,
  current: webhookSecurityPolicyConfigSchema,
})

export const configWebhookSecurityPolicyHistoryCursorPayloadSchema = z.object({
  timestamp: z.string().datetime(),
  route: webhookSecurityPolicyAuditRouteSchema,
  device: z.string(),
})

export const configWebhookSecurityPolicyHistoryMetaSchema = z.object({
  limit: z.number().int(),
  returned: z.number().int(),
  nextCursor: z.string().nullable(),
})

export const configWebhookSecurityPolicyHistoryResponseSchema = z.object({
  data: z.array(configWebhookSecurityPolicyHistoryEntrySchema),
  meta: configWebhookSecurityPolicyHistoryMetaSchema,
})

export type ConfigProviderValidationBody = z.infer<typeof configProviderValidationRequestSchema>
export type ConfigProviderDiscoverModelsBody = z.infer<
  typeof configProviderDiscoverModelsRequestSchema
>
export type ConfigEnvUpdateBody = z.infer<typeof configEnvUpdateRequestSchema>
export type ConfigUpdateBody = z.infer<typeof configUpdateRequestSchema>
export type McpServerBody = z.infer<typeof mcpServerSchema>
export type OutboundWebhookBody = z.infer<typeof outboundWebhookSchema>
export type WebhookEndpointBody = z.infer<typeof webhookEndpointSchema>
export type TelegramChannelBody = z.infer<typeof telegramChannelConfigSchema>
export type SlackChannelBody = z.infer<typeof slackChannelConfigSchema>
export type DiscordChannelBody = z.infer<typeof discordChannelConfigSchema>
export type MattermostChannelBody = z.infer<typeof mattermostChannelConfigSchema>
export type OutboundWebhookDeliveryQuery = z.infer<typeof outboundWebhookDeliveryQuerySchema>
export type OutboundWebhookDeadLetterQuery = z.infer<typeof outboundWebhookDeadLetterQuerySchema>
export type OutboundWebhookDeadLetterAckParams = z.infer<
  typeof outboundWebhookDeadLetterAckParamsSchema
>
export type OutboundWebhookDeadLetterAckBody = z.infer<
  typeof outboundWebhookDeadLetterAckRequestSchema
>
export type OutboundWebhookDeadLetterAckHistoryQuery = z.infer<
  typeof outboundWebhookDeadLetterAckHistoryQuerySchema
>
export type OutboundWebhookDeadLetterTimelineQuery = z.infer<
  typeof outboundWebhookDeadLetterTimelineQuerySchema
>
export type OutboundWebhookDeliveryReplayParams = z.infer<
  typeof outboundWebhookDeliveryReplayParamsSchema
>
export type OutboundWebhookDeliveryReplayBody = z.infer<
  typeof outboundWebhookDeliveryReplayRequestSchema
>
export type McpServerNameParams = z.infer<typeof mcpServerNameParamsSchema>
export type OutboundWebhookIdParams = z.infer<typeof outboundWebhookIdParamsSchema>
export type WebhookEndpointIdParams = z.infer<typeof webhookEndpointIdParamsSchema>
export type McpServerToggleBody = z.infer<typeof mcpServerToggleRequestSchema>
export type McpServerDisabledToolsBody = z.infer<typeof mcpServerDisabledToolsRequestSchema>
export type OutboundWebhookToggleBody = z.infer<typeof outboundWebhookToggleRequestSchema>
export type WebhookEndpointToggleBody = z.infer<typeof webhookEndpointToggleRequestSchema>
export type TelegramChannelToggleBody = z.infer<typeof telegramChannelToggleRequestSchema>
export type ChannelToggleBody = z.infer<typeof channelToggleRequestSchema>
export type TelegramAllowedUserParams = z.infer<typeof telegramAllowedUserParamsSchema>
export type OutboundWebhookDeadLetter = z.infer<typeof outboundWebhookDeadLetterSchema>
export type ConfigChannelPipelineHealthBody = z.infer<
  typeof configChannelPipelineHealthUpdateRequestSchema
>
export type ConfigChannelPipelineHealthHistoryQuery = z.infer<
  typeof configChannelPipelineHealthHistoryQuerySchema
>
export type ConfigChannelPipelineHealthHistoryEntry = z.infer<
  typeof configChannelPipelineHealthHistoryEntrySchema
>
export type ConfigWebhookSecurityHealthBody = z.infer<
  typeof configWebhookSecurityHealthUpdateRequestSchema
>
export type ConfigWebhookSecurityHealthHistoryQuery = z.infer<
  typeof configWebhookSecurityHealthHistoryQuerySchema
>
export type ConfigWebhookSecurityHealthHistoryEntry = z.infer<
  typeof configWebhookSecurityHealthHistoryEntrySchema
>
export type ConfigWebhookSecurityPolicyBody = z.infer<
  typeof configWebhookSecurityPolicyUpdateRequestSchema
>
export type ConfigWebhookSecurityPolicyHistoryQuery = z.infer<
  typeof configWebhookSecurityPolicyHistoryQuerySchema
>
export type ConfigWebhookSecurityPolicyHistoryEntry = z.infer<
  typeof configWebhookSecurityPolicyHistoryEntrySchema
>
