import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { fastifySchemaFromZod } from './utils.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiParameterRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import {
  createObservabilityRepo,
  type ObservabilityRange,
} from '../../observability/events.js'
import { buildObservabilitySupportBundle } from '../../observability/support-bundle.js'
import { buildHealthExportSnapshot } from '../health-support.js'

const observabilityRangeSchema = z.enum(['24h', '7d', '30d'])
const observabilitySourceSchema = z.enum([
  'daemon',
  'cli',
  'tui',
  'desktop-main',
  'desktop-renderer',
  'web',
  'channel',
])
const observabilitySeveritySchema = z.enum([
  'debug',
  'info',
  'warning',
  'error',
  'fatal',
])
const observabilityPrivacySchema = z.enum([
  'operational',
  'diagnostic',
  'sensitive',
])

const observabilityEventSchema = z.object({
  id: z.string().min(1).optional(),
  schemaVersion: z.number().int().min(1).optional(),
  timestamp: z.string().datetime().optional(),
  source: observabilitySourceSchema,
  surface: z.string().min(1).max(80).optional(),
  eventType: z.string().min(1).max(160),
  severity: observabilitySeveritySchema.optional(),
  privacy: observabilityPrivacySchema.optional(),
  sessionId: z.string().min(1).optional(),
  runId: z.string().min(1).optional(),
  messageId: z.string().min(1).optional(),
  taskId: z.string().min(1).optional(),
  channelIdHash: z.string().min(1).optional(),
  userIdHash: z.string().min(1).optional(),
  provider: z.string().min(1).optional(),
  model: z.string().min(1).optional(),
  attributes: z.record(z.unknown()).optional(),
})

const eventIngestRequestSchema = z.object({
  events: z.array(observabilityEventSchema).min(1).max(100),
})

const eventIngestResponseSchema = z.object({
  data: z.object({
    accepted: z.number().int(),
  }),
})

const feedbackRequestSchema = z.object({
  id: z.string().min(1).optional(),
  timestamp: z.string().datetime().optional(),
  sessionId: z.string().min(1).optional(),
  messageId: z.string().min(1).optional(),
  runId: z.string().min(1).optional(),
  rating: z.enum(['positive', 'neutral', 'negative']),
  reason: z.string().max(160).optional(),
  note: z.string().max(2_000).optional(),
  source: observabilitySourceSchema.optional(),
  surface: z.string().min(1).max(80).optional(),
})

const feedbackRecordSchema = z.object({
  id: z.string(),
  timestamp: z.string(),
  sessionId: z.string().optional(),
  messageId: z.string().optional(),
  runId: z.string().optional(),
  rating: z.enum(['positive', 'neutral', 'negative']),
  reason: z.string().optional(),
  noteRedacted: z.string().optional(),
  source: observabilitySourceSchema,
  surface: z.string().optional(),
})

const feedbackResponseSchema = z.object({
  data: feedbackRecordSchema,
})

const observabilityPrivacySettingsSchema = z.object({
  localCollectionEnabled: z.boolean(),
  diagnosticCollectionEnabled: z.boolean(),
  sensitiveCollectionEnabled: z.boolean(),
  feedbackCollectionEnabled: z.boolean(),
  retentionDays: z.number().int().min(1).max(365),
  feedbackPromptCooldownHours: z.number().int().min(1).max(720),
  updatedAt: z.string(),
})

const observabilityPrivacyUpdateSchema = z.object({
  localCollectionEnabled: z.boolean().optional(),
  diagnosticCollectionEnabled: z.boolean().optional(),
  sensitiveCollectionEnabled: z.boolean().optional(),
  feedbackCollectionEnabled: z.boolean().optional(),
  retentionDays: z.number().int().min(1).max(365).optional(),
  feedbackPromptCooldownHours: z.number().int().min(1).max(720).optional(),
})

const observabilityPrivacyResponseSchema = z.object({
  data: observabilityPrivacySettingsSchema,
})

const feedbackPromptStateQuerySchema = z.object({
  surface: z.string().min(1).max(80).optional(),
  sessionId: z.string().min(1).optional(),
  messageId: z.string().min(1).optional(),
  userIdHash: z.string().min(1).optional(),
})

const feedbackPromptStateSchema = z.object({
  shouldPrompt: z.boolean(),
  reason: z.enum(['eligible', 'disabled', 'cooldown', 'recent-feedback']),
  cooldownHours: z.number().int(),
  promptCount24h: z.number().int(),
  feedbackCount24h: z.number().int(),
  lastPromptAt: z.string().optional(),
  nextPromptAfter: z.string().optional(),
})

const feedbackPromptStateResponseSchema = z.object({
  data: feedbackPromptStateSchema,
})

const eventResponseSchema = observabilityEventSchema.extend({
  id: z.string(),
  schemaVersion: z.number().int(),
  timestamp: z.string(),
  severity: observabilitySeveritySchema,
  privacy: observabilityPrivacySchema,
  attributes: z.record(z.unknown()),
})

const trendPointSchema = z.object({
  label: z.string(),
  from: z.string(),
  to: z.string(),
  totalEvents: z.number().int(),
  errorEvents: z.number().int(),
  crashReports: z.number().int(),
  feedbackPositive: z.number().int(),
  feedbackNeutral: z.number().int(),
  feedbackNegative: z.number().int(),
  explicitSatisfaction: z.number().nullable(),
  tasksStarted: z.number().int(),
  tasksCompleted: z.number().int(),
  completionRate: z.number().nullable(),
  assistedTaskThroughputPerDay: z.number(),
  channelTasksStarted: z.number().int(),
  channelTasksCompleted: z.number().int(),
  channelResolutionRate: z.number().nullable(),
  toolSuccessRate: z.number().nullable(),
})

const comparisonSchema = z.object({
  previousFrom: z.string(),
  previousTo: z.string(),
  errorEventsDelta: z.number().int(),
  crashReportsDelta: z.number().int(),
  satisfactionDelta: z.number().nullable(),
  completionRateDelta: z.number().nullable(),
  throughputPerDayDelta: z.number(),
  channelResolutionRateDelta: z.number().nullable(),
  toolSuccessRateDelta: z.number().nullable(),
})

const segmentSchema = z.object({
  key: z.string(),
  label: z.string(),
  source: observabilitySourceSchema,
  surface: z.string().optional(),
  totalEvents: z.number().int(),
  errorEvents: z.number().int(),
  crashReports: z.number().int(),
  feedbackPositive: z.number().int(),
  feedbackNeutral: z.number().int(),
  feedbackNegative: z.number().int(),
  explicitSatisfaction: z.number().nullable(),
  tasksStarted: z.number().int(),
  tasksCompleted: z.number().int(),
  completionRate: z.number().nullable(),
  channelTasksStarted: z.number().int(),
  channelTasksCompleted: z.number().int(),
  channelResolutionRate: z.number().nullable(),
  toolSuccessRate: z.number().nullable(),
})

const hotspotSchema = z.object({
  key: z.string(),
  label: z.string(),
  count: z.number().int(),
  source: observabilitySourceSchema.optional(),
  surface: z.string().optional(),
  eventType: z.string().optional(),
  severity: observabilitySeveritySchema.optional(),
})

const alertSchema = z.object({
  id: z.string(),
  level: z.enum(['info', 'warning', 'critical']),
  title: z.string(),
  detail: z.string(),
  metric: z.string(),
  value: z.number().nullable(),
  threshold: z.number().optional(),
})

const observabilitySnapshotSchema = z.object({
  generatedAt: z.string(),
  range: z.object({
    key: observabilityRangeSchema,
    from: z.string(),
    to: z.string(),
  }),
  reliability: z.object({
    totalEvents: z.number().int(),
    errorEvents: z.number().int(),
    fatalEvents: z.number().int(),
    crashReports: z.number().int(),
    crashFreeSessions: z.number().nullable(),
    providerFailureRate: z.number().nullable(),
    channelDeliverySuccessRate: z.number().nullable(),
    routeErrorRate: z.number().nullable(),
  }),
  quality: z.object({
    feedbackPositive: z.number().int(),
    feedbackNeutral: z.number().int(),
    feedbackNegative: z.number().int(),
    explicitSatisfaction: z.number().nullable(),
    promptResponseRate: z.number().nullable(),
    implicitAcceptance: z.number().nullable(),
    regenerationRate: z.number().nullable(),
    stopRate: z.number().nullable(),
  }),
  productivity: z.object({
    tasksStarted: z.number().int(),
    tasksCompleted: z.number().int(),
    completionRate: z.number().nullable(),
    assistedTaskThroughputPerDay: z.number(),
    autonomousCompletionRate: z.number().nullable(),
    approvalFrictionMs: z.number().nullable(),
    toolSuccessRate: z.number().nullable(),
    costPerResolvedTask: z.number().nullable(),
    tokensPerResolvedTask: z.number().nullable(),
    recoverySuccessRate: z.number().nullable(),
    channelResolutionRate: z.number().nullable(),
  }),
  comparison: comparisonSchema,
  trend: z.array(trendPointSchema),
  segments: z.array(segmentSchema),
  hotspots: z.object({
    errorEvents: z.array(hotspotSchema),
    feedbackReasons: z.array(hotspotSchema),
  }),
  alerts: z.array(alertSchema),
  recent: z.array(eventResponseSchema),
})

const observabilitySnapshotResponseSchema = z.object({
  data: observabilitySnapshotSchema,
})

const observabilityEventsResponseSchema = z.object({
  data: z.array(eventResponseSchema),
})

const observabilityExportRequestSchema = z.object({
  range: observabilityRangeSchema.optional(),
  limit: z.number().int().min(1).max(1_000).optional(),
  includeEvents: z.boolean().optional(),
  includeCrashes: z.boolean().optional(),
  includeFeedback: z.boolean().optional(),
  persist: z.boolean().optional(),
})

const observabilitySupportBundleRequestSchema = observabilityExportRequestSchema.extend({
  includeHealth: z.boolean().optional(),
})

const observabilityExportBundleSchema = z.object({
  generatedAt: z.string(),
  privacy: observabilityPrivacySettingsSchema,
  snapshot: observabilitySnapshotSchema,
  events: z.array(eventResponseSchema),
  crashes: z.array(eventResponseSchema),
  feedback: z.array(feedbackRecordSchema),
  redaction: z.object({
    attributes: z.literal('sanitized-at-ingest'),
    notes: z.literal('redacted-or-truncated'),
  }),
})

const observabilityExportResponseSchema = z.object({
  data: z.object({
    path: z.string().optional(),
    eventCount: z.number().int(),
    crashCount: z.number().int(),
    feedbackCount: z.number().int(),
    bundle: observabilityExportBundleSchema,
  }),
})

const observabilitySupportBundleFileSchema = z.object({
  path: z.string(),
  mediaType: z.string(),
  bytes: z.number().int().nonnegative(),
})

const observabilitySupportBundleRedactionSchema = z.object({
  level: z.literal('support'),
  appliedAt: z.string(),
  rules: z.array(z.string()),
  pathAliases: z.record(z.string()),
})

const observabilitySupportBundleManifestSchema = z.object({
  schemaVersion: z.literal(1),
  kind: z.literal('sepilotd-observability-support-bundle'),
  generatedAt: z.string(),
  range: observabilityRangeSchema,
  counts: z.object({
    events: z.number().int().nonnegative(),
    crashes: z.number().int().nonnegative(),
    feedback: z.number().int().nonnegative(),
  }),
  healthIncluded: z.boolean(),
  redaction: observabilitySupportBundleRedactionSchema,
  contents: z.array(observabilitySupportBundleFileSchema),
})

const observabilitySupportBundleResponseSchema = z.object({
  data: z.object({
    path: z.string().optional(),
    fileCount: z.number().int().nonnegative(),
    totalBytes: z.number().int().nonnegative(),
    eventCount: z.number().int().nonnegative(),
    crashCount: z.number().int().nonnegative(),
    feedbackCount: z.number().int().nonnegative(),
    files: z.array(observabilitySupportBundleFileSchema),
    manifest: observabilitySupportBundleManifestSchema,
    redaction: observabilitySupportBundleRedactionSchema,
  }),
})

const observabilityPruneResponseSchema = z.object({
  data: z.object({
    cutoff: z.string(),
    deletedEvents: z.number().int(),
    deletedFeedback: z.number().int(),
  }),
})

const observabilitySnapshotQuerySchema = z.object({
  range: observabilityRangeSchema.optional(),
})

const observabilityEventsQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(200).optional(),
  severity: observabilitySeveritySchema.optional(),
  eventType: z.string().min(1).max(160).optional(),
})

type ObservabilitySnapshotQuery = z.infer<typeof observabilitySnapshotQuerySchema>
type ObservabilityEventsQuery = z.infer<typeof observabilityEventsQuerySchema>
type EventIngestRequest = z.infer<typeof eventIngestRequestSchema>
type FeedbackRequest = z.infer<typeof feedbackRequestSchema>
type ObservabilityPrivacyUpdateRequest =
  z.infer<typeof observabilityPrivacyUpdateSchema>
type FeedbackPromptStateQuery = z.infer<typeof feedbackPromptStateQuerySchema>
type ObservabilityExportRequest = z.infer<typeof observabilityExportRequestSchema>
type ObservabilitySupportBundleRequest =
  z.infer<typeof observabilitySupportBundleRequestSchema>

export const observabilityOpenApiComponents: OpenApiComponentOverrides =
  openApiComponentsFromZod({
    schemas: {
      ObservabilityEvent: eventResponseSchema,
      ObservabilityEventIngestRequest: eventIngestRequestSchema,
      ObservabilityEventIngestResponse: eventIngestResponseSchema,
      FeedbackRequest: feedbackRequestSchema,
      FeedbackResponse: feedbackResponseSchema,
      FeedbackPromptStateResponse: feedbackPromptStateResponseSchema,
      ObservabilityPrivacyUpdateRequest: observabilityPrivacyUpdateSchema,
      ObservabilityPrivacyResponse: observabilityPrivacyResponseSchema,
      ObservabilitySnapshot: observabilitySnapshotSchema,
      ObservabilitySnapshotResponse: observabilitySnapshotResponseSchema,
      ObservabilityEventsResponse: observabilityEventsResponseSchema,
      ObservabilityExportRequest: observabilityExportRequestSchema,
      ObservabilityExportResponse: observabilityExportResponseSchema,
      ObservabilitySupportBundleRequest: observabilitySupportBundleRequestSchema,
      ObservabilitySupportBundleResponse: observabilitySupportBundleResponseSchema,
      ObservabilityPruneResponse: observabilityPruneResponseSchema,
    },
    parameters: {
      ObservabilityRangeParam: {
        name: 'range',
        in: 'query',
        schema: observabilityRangeSchema,
      },
      ObservabilityLimitParam: {
        name: 'limit',
        in: 'query',
        schema: z.number().int().min(1).max(200),
      },
      ObservabilitySeverityParam: {
        name: 'severity',
        in: 'query',
        schema: observabilitySeveritySchema,
      },
    },
  })

export const observabilityOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/observability/events': {
    post: {
      summary: 'Ingest local observability events',
      tags: ['Observability'],
      requestBody: {
        required: true,
        content: {
          'application/json': {
            schema: { $ref: '#/components/schemas/ObservabilityEventIngestRequest' },
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ObservabilityEventIngestResponse'),
      },
    },
    get: {
      summary: 'List local observability events',
      tags: ['Observability'],
      parameters: [
        openApiParameterRef('ObservabilityLimitParam'),
        openApiParameterRef('ObservabilitySeverityParam'),
      ],
      responses: {
        200: openApiJsonResponseRef('ObservabilityEventsResponse'),
      },
    },
  },
  '/api/v1/observability/snapshot': {
    get: {
      summary: 'Observability dashboard snapshot',
      tags: ['Observability'],
      parameters: [openApiParameterRef('ObservabilityRangeParam')],
      responses: {
        200: openApiJsonResponseRef('ObservabilitySnapshotResponse'),
      },
    },
  },
  '/api/v1/observability/crashes': {
    get: {
      summary: 'List local crash and fatal events',
      tags: ['Observability'],
      parameters: [openApiParameterRef('ObservabilityLimitParam')],
      responses: {
        200: openApiJsonResponseRef('ObservabilityEventsResponse'),
      },
    },
  },
  '/api/v1/observability/privacy': {
    get: {
      summary: 'Get local observability privacy settings',
      tags: ['Observability'],
      responses: {
        200: openApiJsonResponseRef('ObservabilityPrivacyResponse'),
      },
    },
    patch: {
      summary: 'Update local observability privacy settings',
      tags: ['Observability'],
      requestBody: {
        required: true,
        content: {
          'application/json': {
            schema: { $ref: '#/components/schemas/ObservabilityPrivacyUpdateRequest' },
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ObservabilityPrivacyResponse'),
      },
    },
  },
  '/api/v1/observability/export': {
    post: {
      summary: 'Export sanitized local observability bundle',
      tags: ['Observability'],
      requestBody: {
        required: false,
        content: {
          'application/json': {
            schema: { $ref: '#/components/schemas/ObservabilityExportRequest' },
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ObservabilityExportResponse'),
      },
    },
  },
  '/api/v1/observability/support-bundle': {
    post: {
      summary: 'Create a redacted diagnostic support bundle',
      tags: ['Observability'],
      requestBody: {
        required: false,
        content: {
          'application/json': {
            schema: { $ref: '#/components/schemas/ObservabilitySupportBundleRequest' },
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ObservabilitySupportBundleResponse'),
      },
    },
  },
  '/api/v1/observability/prune': {
    post: {
      summary: 'Prune local observability data past retention',
      tags: ['Observability'],
      responses: {
        200: openApiJsonResponseRef('ObservabilityPruneResponse'),
      },
    },
  },
  '/api/v1/feedback': {
    post: {
      summary: 'Record low-friction user feedback',
      tags: ['Observability'],
      requestBody: {
        required: true,
        content: {
          'application/json': {
            schema: { $ref: '#/components/schemas/FeedbackRequest' },
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('FeedbackResponse'),
      },
    },
  },
  '/api/v1/feedback/prompt-state': {
    get: {
      summary: 'Get non-intrusive feedback prompt eligibility',
      tags: ['Observability'],
      responses: {
        200: openApiJsonResponseRef('FeedbackPromptStateResponse'),
      },
    },
  },
}

export async function observabilityRoutes(app: FastifyInstance) {
  const repo = createObservabilityRepo()

  app.post<{ Body: EventIngestRequest }>('/observability/events', {
    schema: fastifySchemaFromZod({ body: eventIngestRequestSchema }),
  }, async (request) => {
    const body = request.body
    return { data: repo.recordEvents(body.events) }
  })

  app.get<{ Querystring: ObservabilityEventsQuery }>('/observability/events', {
    schema: fastifySchemaFromZod({ querystring: observabilityEventsQuerySchema }),
  }, async (request) => {
    const query = request.query
    return { data: repo.listEvents(query) }
  })

  app.get<{ Querystring: ObservabilitySnapshotQuery }>(
    '/observability/snapshot',
    {
      schema: fastifySchemaFromZod({
        querystring: observabilitySnapshotQuerySchema,
      }),
    },
    async (request) => {
      const query = request.query
      return {
        data: repo.snapshot((query.range ?? '7d') as ObservabilityRange),
      }
    },
  )

  app.get<{ Querystring: Pick<ObservabilityEventsQuery, 'limit'> }>(
    '/observability/crashes',
    {
      schema: fastifySchemaFromZod({
        querystring: observabilityEventsQuerySchema.pick({ limit: true }),
      }),
    },
    async (request) => {
      const query = request.query
      return { data: repo.listCrashes({ limit: query.limit }) }
    },
  )

  app.get('/observability/privacy', async () => ({
    data: repo.getPrivacySettings(),
  }))

  app.patch<{ Body: ObservabilityPrivacyUpdateRequest }>(
    '/observability/privacy',
    {
      schema: fastifySchemaFromZod({ body: observabilityPrivacyUpdateSchema }),
    },
    async (request) => ({
      data: repo.updatePrivacySettings(request.body),
    }),
  )

  app.post<{ Body: ObservabilityExportRequest }>(
    '/observability/export',
    {
      schema: fastifySchemaFromZod({
        body: observabilityExportRequestSchema.optional(),
      }),
    },
    async (request) => ({
      data: repo.exportBundle(request.body ?? {}),
    }),
  )

  app.post<{ Body: ObservabilitySupportBundleRequest }>(
    '/observability/support-bundle',
    {
      schema: fastifySchemaFromZod({
        body: observabilitySupportBundleRequestSchema.optional(),
      }),
    },
    async (request) => {
      const body = request.body ?? {}
      const range = (body.range ?? '7d') as ObservabilityRange
      const exportResult = repo.exportBundle({
        range,
        limit: body.limit,
        includeEvents: body.includeEvents,
        includeCrashes: body.includeCrashes,
        includeFeedback: body.includeFeedback,
        persist: false,
      })
      const health = body.includeHealth === false
        ? undefined
        : await buildHealthExportSnapshot(app)
      return {
        data: buildObservabilitySupportBundle({
          range,
          exportResult,
          health,
          persist: body.persist,
        }),
      }
    },
  )

  app.post('/observability/prune', async () => ({
    data: repo.prune(),
  }))

  app.post<{ Body: FeedbackRequest }>('/feedback', {
    schema: fastifySchemaFromZod({ body: feedbackRequestSchema }),
  }, async (request) => {
    const body = request.body
    return { data: repo.recordFeedback(body) }
  })

  app.get<{ Querystring: FeedbackPromptStateQuery }>(
    '/feedback/prompt-state',
    {
      schema: fastifySchemaFromZod({
        querystring: feedbackPromptStateQuerySchema,
      }),
    },
    async (request) => ({
      data: repo.feedbackPromptState(request.query),
    }),
  )
}
