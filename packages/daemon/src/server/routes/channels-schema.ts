import { z } from 'zod'
import { latestChannelPipelineHealthChangeSchema } from '../runtime/channel-pipeline-health-audit.js'

export const replaySummarySchema = z.object({
  enabled: z.boolean(),
  totalRecords: z.number().int(),
  processingRecords: z.number().int(),
  processedRecords: z.number().int(),
  staleProcessingRecords: z.number().int(),
})

export const sessionSummarySchema = z.object({
  enabled: z.boolean(),
  totalBindings: z.number().int(),
  staleBindings: z.number().int(),
})

export const pipelineStageStatsSchema = z.object({
  stage: z.string(),
  count: z.number().int(),
  totalDurationMs: z.number().int(),
  avgDurationMs: z.number().int(),
  maxDurationMs: z.number().int(),
  lastDurationMs: z.number().int(),
})

export const pipelineFailureSampleSchema = z.object({
  timestamp: z.string(),
  channelType: z.string(),
  outcome: z.enum(['no_provider', 'error']),
})

export const channelPipelineOutcomeQuerySchema = z.enum([
  'processed',
  'duplicate',
  'blocked',
  'no_provider',
  'error',
])

export const pipelineHealthThresholdSchema = z.object({
  minRecentEvents: z.number().int(),
  degradeFailureRate: z.number(),
  minAgentSamples: z.number().int(),
  degradeAgentAvgLatencyMs: z.number().int(),
})

export const pipelineHealthOverrideSchema = pipelineHealthThresholdSchema.partial()

export const pipelineHealthPolicySchema = pipelineHealthThresholdSchema.extend({
  hotChannelTopN: z.number().int(),
  byChannelType: z.record(pipelineHealthOverrideSchema),
})

export const pipelineRecentChannelStatsSchema = z.object({
  channelType: z.string(),
  totalEvents: z.number().int(),
  processedEvents: z.number().int(),
  duplicateEvents: z.number().int(),
  blockedEvents: z.number().int(),
  noProviderEvents: z.number().int(),
  errorEvents: z.number().int(),
  failureEvents: z.number().int(),
  failureRate: z.number(),
  byStage: z.array(pipelineStageStatsSchema),
})

export const pipelineRecentStatsSchema = z.object({
  windowMs: z.number().int(),
  totalEvents: z.number().int(),
  processedEvents: z.number().int(),
  duplicateEvents: z.number().int(),
  blockedEvents: z.number().int(),
  noProviderEvents: z.number().int(),
  errorEvents: z.number().int(),
  failureEvents: z.number().int(),
  failureRate: z.number(),
  byChannelType: z.array(pipelineRecentChannelStatsSchema),
  byStage: z.array(pipelineStageStatsSchema),
  failureSamples: z.array(pipelineFailureSampleSchema),
})

export const channelPipelineRecentQuerySchema = z.object({
  channelType: z.string().min(1).optional(),
  outcome: channelPipelineOutcomeQuerySchema.optional(),
  failureOnly: z.union([
    z.boolean(),
    z.enum(['true', 'false']),
  ]).transform((value) => value === true || value === 'true').optional().default(false),
  sampleOffset: z.coerce.number().int().min(0).optional().default(0),
  sampleLimit: z.coerce.number().int().min(1).max(100).optional().default(20),
})

export const pipelineChannelStatsSchema = z.object({
  channelType: z.string(),
  inFlight: z.number().int(),
  totalEvents: z.number().int(),
  processedEvents: z.number().int(),
  duplicateEvents: z.number().int(),
  blockedEvents: z.number().int(),
  noProviderEvents: z.number().int(),
  errorEvents: z.number().int(),
})

export const pipelineStatsSchema = z.object({
  inFlight: z.number().int(),
  totalEvents: z.number().int(),
  processedEvents: z.number().int(),
  duplicateEvents: z.number().int(),
  blockedEvents: z.number().int(),
  noProviderEvents: z.number().int(),
  errorEvents: z.number().int(),
  byChannelType: z.array(pipelineChannelStatsSchema),
  byStage: z.array(pipelineStageStatsSchema),
  recent: pipelineRecentStatsSchema,
})

export const pipelineSummarySchema = z.object({
  enabled: z.boolean(),
  inFlight: z.number().int(),
  totalEvents: z.number().int(),
  processedEvents: z.number().int(),
  duplicateEvents: z.number().int(),
  blockedEvents: z.number().int(),
  noProviderEvents: z.number().int(),
  errorEvents: z.number().int(),
  stageStats: z.array(pipelineStageStatsSchema),
  recentSummary: z.object({
    windowMs: z.number().int(),
    totalEvents: z.number().int(),
    processedEvents: z.number().int(),
    duplicateEvents: z.number().int(),
    blockedEvents: z.number().int(),
    noProviderEvents: z.number().int(),
    errorEvents: z.number().int(),
    failureEvents: z.number().int(),
    failureRate: z.number(),
    stageStats: z.array(pipelineStageStatsSchema),
  }),
})

export const channelCatalogEntrySchema = z.object({
  type: z.string(),
  source: z.enum(['builtin', 'plugin']),
  displayName: z.string(),
  configured: z.boolean(),
  configuredCount: z.number().int(),
  enabled: z.boolean(),
  enabledCount: z.number().int(),
  activeCount: z.number().int(),
  status: z.enum([
    'connected',
    'disconnected',
    'connecting',
    'error',
    'not_configured',
  ]),
  ingress: z.enum(['webhook', 'polling', 'local']).nullable(),
  outbound: z.boolean().nullable(),
  replyMode: z.enum(['channel', 'message', 'protocol-specific', 'none']).nullable(),
  authType: z.string().nullable(),
  webhookRequired: z.boolean().nullable(),
  maxMessageLength: z.number().int().nullable(),
  replayProtected: z.boolean(),
  replayRecords: z.number().int(),
  replayProcessingRecords: z.number().int(),
  replayProcessedRecords: z.number().int(),
  replayStaleProcessingRecords: z.number().int(),
  sessionScoped: z.boolean(),
  sessionBindings: z.number().int(),
  staleSessionBindings: z.number().int(),
  pipelineTracked: z.boolean(),
  pipelineInFlight: z.number().int(),
  pipelineEvents: z.number().int(),
  pipelineProcessedEvents: z.number().int(),
  pipelineDuplicateEvents: z.number().int(),
  pipelineBlockedEvents: z.number().int(),
  pipelineNoProviderEvents: z.number().int(),
  pipelineErrorEvents: z.number().int(),
  capabilities: z.array(z.string()),
})

export const channelsResponseSchema = z.object({
  data: z.array(channelCatalogEntrySchema),
  meta: z.object({
    pipelineSummary: pipelineSummarySchema,
    replaySummary: replaySummarySchema,
    sessionSummary: sessionSummarySchema,
    latestHealthPolicyChange: latestChannelPipelineHealthChangeSchema.nullable(),
  }),
})

export const channelPipelineResponseSchema = z.object({
  data: pipelineStatsSchema,
  meta: z.object({
    enabled: z.boolean(),
    healthPolicy: pipelineHealthPolicySchema,
    latestHealthPolicyChange: latestChannelPipelineHealthChangeSchema.nullable(),
  }),
})

export const channelPipelineRecentResponseSchema = z.object({
  data: pipelineRecentStatsSchema,
  meta: z.object({
    enabled: z.boolean(),
    filters: z.object({
      channelType: z.string().nullable(),
      outcome: channelPipelineOutcomeQuerySchema.nullable(),
      failureOnly: z.boolean(),
      sampleOffset: z.number().int(),
      sampleLimit: z.number().int(),
    }),
    samples: z.object({
      totalFailureSamples: z.number().int(),
      returnedFailureSamples: z.number().int(),
    }),
  }),
})
