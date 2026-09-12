import type { ChannelType } from '@sepilotd/core'
import type { FastifyInstance } from 'fastify'
import type { z } from 'zod'
import '../fastify-types.js'
import {
  getLatestChannelPipelineHealthPolicyChange,
} from '../runtime/channel-pipeline-health-audit.js'
import { resolveChannelPipelineHealthConfig } from '../runtime/channel-pipeline-health.js'
import { zodRequestValidation } from './utils.js'
import {
  channelPipelineRecentQuerySchema,
} from './channels-schema.js'
import {
  emptyPipelineRecentStats,
  emptyPipelineStats,
  filterPipelineRecentStats,
  summarizeChannelStatus,
} from './channels-internals.js'

export { channelOpenApiComponents, channelOpenApiOverrides } from './channels-openapi.js'

type CatalogSource = 'builtin' | 'plugin'
type CatalogIngress = 'webhook' | 'polling' | 'local'
type CatalogReplyMode = 'channel' | 'message' | 'protocol-specific' | 'none'
interface ChannelCatalogMetadata {
  displayName: string
  ingress: CatalogIngress
  outbound: boolean
  replyMode: CatalogReplyMode
  authType: string
  webhookRequired: boolean
  maxMessageLength: number | null
  capabilities: string[]
}

const BUILTIN_CHANNEL_CATALOG: Record<ChannelType, ChannelCatalogMetadata> = {
  'github-issue': {
    displayName: 'GitHub Issues',
    ingress: 'polling',
    outbound: true,
    replyMode: 'message',
    authType: 'gateway-service',
    webhookRequired: false,
    maxMessageLength: null,
    capabilities: ['ingress:polling', 'egress:comment', 'ticket:auto-close'],
  },
  telegram: {
    displayName: 'Telegram',
    ingress: 'polling',
    outbound: true,
    replyMode: 'channel',
    authType: 'bot-token',
    webhookRequired: false,
    maxMessageLength: null,
    capabilities: ['ingress:polling', 'egress:send', 'pairing:optional'],
  },
  slack: {
    displayName: 'Slack',
    ingress: 'webhook',
    outbound: true,
    replyMode: 'channel',
    authType: 'bot-token+signing-secret',
    webhookRequired: true,
    maxMessageLength: null,
    capabilities: ['ingress:webhook', 'egress:send', 'dedupe:retry'],
  },
  discord: {
    displayName: 'Discord',
    ingress: 'webhook',
    outbound: true,
    replyMode: 'channel',
    authType: 'bot-token+interaction-signature',
    webhookRequired: true,
    maxMessageLength: null,
    capabilities: ['ingress:webhook', 'egress:send', 'dedupe:retry'],
  },
  mattermost: {
    displayName: 'Mattermost',
    ingress: 'webhook',
    outbound: true,
    replyMode: 'channel',
    authType: 'bot-token+webhook-token',
    webhookRequired: true,
    maxMessageLength: null,
    capabilities: ['ingress:webhook', 'egress:send', 'dedupe:retry'],
  },
  webhook: {
    displayName: 'Generic Webhook',
    ingress: 'webhook',
    outbound: false,
    replyMode: 'none',
    authType: 'shared-secret',
    webhookRequired: true,
    maxMessageLength: null,
    capabilities: ['ingress:webhook', 'verify:hmac', 'dedupe:delivery-id'],
  },
  webchat: {
    displayName: 'Webchat',
    ingress: 'local',
    outbound: true,
    replyMode: 'message',
    authType: 'session',
    webhookRequired: false,
    maxMessageLength: null,
    capabilities: ['ingress:local', 'egress:deferred-response'],
  },
  whatsapp: {
    displayName: 'WhatsApp',
    ingress: 'webhook',
    outbound: true,
    replyMode: 'channel',
    authType: 'access-token+app-secret',
    webhookRequired: true,
    maxMessageLength: null,
    capabilities: ['ingress:webhook', 'verify:meta-signature', 'dedupe:message-id'],
  },
  teams: {
    displayName: 'Microsoft Teams',
    ingress: 'webhook',
    outbound: true,
    replyMode: 'protocol-specific',
    authType: 'app-id+bot-framework-jwt',
    webhookRequired: true,
    maxMessageLength: null,
    capabilities: ['ingress:webhook', 'verify:jwt', 'dedupe:activity-id'],
  },
  line: {
    displayName: 'LINE',
    ingress: 'webhook',
    outbound: true,
    replyMode: 'protocol-specific',
    authType: 'channel-access-token+signature',
    webhookRequired: true,
    maxMessageLength: null,
    capabilities: ['ingress:webhook', 'reply:token', 'dedupe:message-id'],
  },
}

const REPLAY_PROTECTED_CHANNEL_TYPES = new Set<ChannelType>([
  'slack',
  'discord',
  'mattermost',
  'webhook',
  'whatsapp',
  'teams',
  'line',
])

const SESSION_SCOPED_CHANNEL_TYPES = new Set<ChannelType>([
  'github-issue',
  'telegram',
  'slack',
  'discord',
  'mattermost',
  'webchat',
  'whatsapp',
  'teams',
  'line',
])


export async function channelRoutes(app: FastifyInstance) {
  app.get<{
    Querystring: z.infer<typeof channelPipelineRecentQuerySchema>
  }>('/channels/pipeline/recent', {
    preValidation: zodRequestValidation({
      query: {
        schema: channelPipelineRecentQuerySchema,
        message: 'Invalid channel pipeline recent query',
      },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Runtime not initialized',
        },
      })
    }

    const recent = runtime.channelPipelineMonitor?.getStats?.().recent
      ?? emptyPipelineRecentStats()
    const filtered = filterPipelineRecentStats(recent, request.query)

    return {
      data: filtered.data,
      meta: {
        enabled: Boolean(runtime.channelPipelineMonitor),
        ...filtered.meta,
      },
    }
  })

  app.get('/channels/pipeline', async (_request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Runtime not initialized',
        },
      })
    }

    const latestHealthPolicyChange =
      await getLatestChannelPipelineHealthPolicyChange(runtime)

    return {
      data: runtime.channelPipelineMonitor?.getStats?.() ?? emptyPipelineStats(),
      meta: {
        enabled: Boolean(runtime.channelPipelineMonitor),
        healthPolicy: resolveChannelPipelineHealthConfig(runtime.config),
        latestHealthPolicyChange,
      },
    }
  })

  app.get('/channels', async (_request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Runtime not initialized',
        },
      })
    }

    const configuredCount = new Map<string, number>()
    const enabledCount = new Map<string, number>()
    for (const channel of runtime.config.channels ?? []) {
      configuredCount.set(
        channel.type,
        (configuredCount.get(channel.type) ?? 0) + 1,
      )
      if (channel.enabled) {
        enabledCount.set(
          channel.type,
          (enabledCount.get(channel.type) ?? 0) + 1,
        )
      }
    }

    const activeByType = new Map<string, string[]>()
    for (const channel of runtime.channels) {
      const statuses = activeByType.get(channel.type) ?? []
      statuses.push(channel.getStatus())
      activeByType.set(channel.type, statuses)
    }

    const replayStats = await runtime.channelReplayStore?.getStats?.()
    const replayByType = new Map(
      (replayStats?.byChannelType ?? []).map((entry) => [entry.channelType, entry]),
    )
    const sessionStats = await runtime.channelSessionStore?.getStats?.()
    const sessionByType = new Map(
      (sessionStats?.byChannelType ?? []).map((entry) => [entry.channelType, entry]),
    )
    const pipelineStats = runtime.channelPipelineMonitor?.getStats?.() ?? emptyPipelineStats()
    const latestHealthPolicyChange =
      await getLatestChannelPipelineHealthPolicyChange(runtime)
    const pipelineByType = new Map(
      (pipelineStats?.byChannelType ?? []).map((entry) => [entry.channelType, entry]),
    )

    const knownTypes = new Set<string>([
      ...Object.keys(BUILTIN_CHANNEL_CATALOG),
      ...(runtime.channelFactoryRegistry?.listTypes?.() ?? []),
      ...configuredCount.keys(),
      ...activeByType.keys(),
      ...pipelineByType.keys(),
    ])

    return {
      data: Array.from(knownTypes)
        .sort((left, right) => left.localeCompare(right))
        .map((type) => {
          const metadata = BUILTIN_CHANNEL_CATALOG[type as ChannelType]
          const statuses = activeByType.get(type) ?? []
          const replay = replayByType.get(type)
          const session = sessionByType.get(type)
          const pipeline = pipelineByType.get(type)

          return {
            type,
            source: (metadata ? 'builtin' : 'plugin') as CatalogSource,
            displayName: metadata?.displayName ?? type,
            configured: (configuredCount.get(type) ?? 0) > 0,
            configuredCount: configuredCount.get(type) ?? 0,
            enabled: (enabledCount.get(type) ?? 0) > 0,
            enabledCount: enabledCount.get(type) ?? 0,
            activeCount: statuses.length,
            status: summarizeChannelStatus(statuses),
            ingress: metadata?.ingress ?? null,
            outbound: metadata?.outbound ?? null,
            replyMode: metadata?.replyMode ?? null,
            authType: metadata?.authType ?? null,
            webhookRequired: metadata?.webhookRequired ?? null,
            maxMessageLength: metadata?.maxMessageLength ?? null,
            replayProtected: metadata
              ? REPLAY_PROTECTED_CHANNEL_TYPES.has(type as ChannelType)
              : false,
            replayRecords: replay?.totalRecords ?? 0,
            replayProcessingRecords: replay?.processingRecords ?? 0,
            replayProcessedRecords: replay?.processedRecords ?? 0,
            replayStaleProcessingRecords: replay?.staleProcessingRecords ?? 0,
            sessionScoped: metadata
              ? SESSION_SCOPED_CHANNEL_TYPES.has(type as ChannelType)
              : false,
            sessionBindings: session?.totalBindings ?? 0,
            staleSessionBindings: session?.staleBindings ?? 0,
            pipelineTracked: Boolean(runtime.channelPipelineMonitor),
            pipelineInFlight: pipeline?.inFlight ?? 0,
            pipelineEvents: pipeline?.totalEvents ?? 0,
            pipelineProcessedEvents: pipeline?.processedEvents ?? 0,
            pipelineDuplicateEvents: pipeline?.duplicateEvents ?? 0,
            pipelineBlockedEvents: pipeline?.blockedEvents ?? 0,
            pipelineNoProviderEvents: pipeline?.noProviderEvents ?? 0,
            pipelineErrorEvents: pipeline?.errorEvents ?? 0,
            capabilities: metadata?.capabilities ?? [],
          }
        }),
      meta: {
        pipelineSummary: {
          enabled: Boolean(runtime.channelPipelineMonitor),
          inFlight: pipelineStats.inFlight,
          totalEvents: pipelineStats.totalEvents,
          processedEvents: pipelineStats.processedEvents,
          duplicateEvents: pipelineStats.duplicateEvents,
          blockedEvents: pipelineStats.blockedEvents,
          noProviderEvents: pipelineStats.noProviderEvents,
          errorEvents: pipelineStats.errorEvents,
          stageStats: pipelineStats.byStage,
          recentSummary: {
            windowMs: pipelineStats.recent.windowMs,
            totalEvents: pipelineStats.recent.totalEvents,
            processedEvents: pipelineStats.recent.processedEvents,
            duplicateEvents: pipelineStats.recent.duplicateEvents,
            blockedEvents: pipelineStats.recent.blockedEvents,
            noProviderEvents: pipelineStats.recent.noProviderEvents,
            errorEvents: pipelineStats.recent.errorEvents,
            failureEvents: pipelineStats.recent.failureEvents,
            failureRate: pipelineStats.recent.failureRate,
            stageStats: pipelineStats.recent.byStage,
          },
        },
        replaySummary: {
          enabled: Boolean(runtime.channelReplayStore),
          totalRecords: replayStats?.totalRecords ?? 0,
          processingRecords: replayStats?.processingRecords ?? 0,
          processedRecords: replayStats?.processedRecords ?? 0,
          staleProcessingRecords: replayStats?.staleProcessingRecords ?? 0,
        },
        sessionSummary: {
          enabled: Boolean(runtime.channelSessionStore),
          totalBindings: sessionStats?.totalBindings ?? 0,
          staleBindings: sessionStats?.staleBindings ?? 0,
        },
        latestHealthPolicyChange,
      },
    }
  })
}
