import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { createLogger } from '../../logger.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import { getLatestChannelPipelineHealthPolicyChange } from '../runtime/channel-pipeline-health-audit.js'
import { summarizeWebhookSecurity } from '../runtime/webhook-endpoints.js'
import { getLatestWebhookSecurityPolicyChange } from '../runtime/webhook-security-policy-audit.js'

const log = createLogger('metrics-route')

const metricsResponseSchema = z.object({
  data: z.record(z.unknown()),
})

export const metricsOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    MetricsResponse: metricsResponseSchema,
  },
})

export const metricsOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/metrics': {
    get: {
      summary: 'System metrics',
      tags: ['System'],
      responses: { 200: openApiJsonResponseRef('MetricsResponse') },
    },
  },
}

export async function metricsRoutes(app: FastifyInstance) {
  app.get('/metrics', async () => {
    const runtime = app.runtime

    const metrics: Record<string, unknown> = {
      uptime_seconds: process.uptime(),
      memory_rss_mb: Math.round(process.memoryUsage().rss / 1024 / 1024),
      memory_heap_mb: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
    }

    if (runtime) {
      // Usage stats
      try {
        const usage = runtime.usageTracker.getTotalUsage()
        metrics.total_input_tokens = usage.inputTokens
        metrics.total_output_tokens = usage.outputTokens
        metrics.total_cost_usd = usage.costUsd
        metrics.total_requests = usage.requestCount
      } catch (err) {
        log.debug('usage metrics unavailable', {
          error: err instanceof Error ? err.message : String(err),
        })
      }

      // Session count
      try {
        const sessions = await runtime.sessions.list({ perPage: 1 })
        metrics.total_sessions = sessions.totalCount
      } catch (err) {
        log.debug('session count metrics unavailable', {
          error: err instanceof Error ? err.message : String(err),
        })
      }

      // Provider count
      metrics.providers_count = runtime.providerRegistry.list().length

      // Tool count
      metrics.tools_count = runtime.toolRegistry?.list().length ?? 0

      const runLimiterStats = runtime.runLimiter?.getStats()
      if (runLimiterStats) {
        metrics.run_limiter_active = runLimiterStats.active
        metrics.run_limiter_queued = runLimiterStats.queued
        metrics.run_limiter_max_active = runLimiterStats.maxActive
        metrics.run_limiter_max_queued = runLimiterStats.maxQueued
        metrics.run_limiter_accepting = runLimiterStats.accepting ? 1 : 0
      }

      const breakerSummary = runtime.providerCircuitBreaker?.getSummary()
      if (breakerSummary) {
        metrics.provider_circuit_breaker_tracked = breakerSummary.trackedCircuits
        metrics.provider_circuit_breaker_open = breakerSummary.openCircuits
        metrics.provider_circuit_breaker_half_open = breakerSummary.halfOpenCircuits
      }

      // Channel count
      metrics.channels_total = runtime.channels.length
      metrics.channels_connected = runtime.channels.filter(c => c.getStatus() === 'connected').length
      const pipelineStats = runtime.channelPipelineMonitor?.getStats?.()
      if (pipelineStats) {
        const latestHealthPolicyChange =
          await getLatestChannelPipelineHealthPolicyChange(runtime)
        metrics.channel_pipeline_in_flight = pipelineStats.inFlight
        metrics.channel_pipeline_total = pipelineStats.totalEvents
        metrics.channel_pipeline_processed = pipelineStats.processedEvents
        metrics.channel_pipeline_duplicates = pipelineStats.duplicateEvents
        metrics.channel_pipeline_blocked = pipelineStats.blockedEvents
        metrics.channel_pipeline_no_provider = pipelineStats.noProviderEvents
        metrics.channel_pipeline_errors = pipelineStats.errorEvents
        metrics.channel_pipeline_by_type = pipelineStats.byChannelType
        metrics.channel_pipeline_by_stage = pipelineStats.byStage
        metrics.channel_pipeline_recent_window_seconds = Math.round(
          pipelineStats.recent.windowMs / 1000,
        )
        metrics.channel_pipeline_recent_total = pipelineStats.recent.totalEvents
        metrics.channel_pipeline_recent_processed = pipelineStats.recent.processedEvents
        metrics.channel_pipeline_recent_duplicates = pipelineStats.recent.duplicateEvents
        metrics.channel_pipeline_recent_blocked = pipelineStats.recent.blockedEvents
        metrics.channel_pipeline_recent_no_provider = pipelineStats.recent.noProviderEvents
        metrics.channel_pipeline_recent_errors = pipelineStats.recent.errorEvents
        metrics.channel_pipeline_recent_failures = pipelineStats.recent.failureEvents
        metrics.channel_pipeline_recent_failure_rate = pipelineStats.recent.failureRate
        metrics.channel_pipeline_recent_by_type = pipelineStats.recent.byChannelType
        metrics.channel_pipeline_recent_by_stage = pipelineStats.recent.byStage
        if (latestHealthPolicyChange) {
          metrics.channel_pipeline_health_policy_last_changed_at =
            latestHealthPolicyChange.timestamp
          metrics.channel_pipeline_health_policy_last_changed_route =
            latestHealthPolicyChange.route
          metrics.channel_pipeline_health_policy_last_changed_device =
            latestHealthPolicyChange.device
        }
      }
      const latestWebhookSecurityPolicyChange =
        await getLatestWebhookSecurityPolicyChange(runtime)
      const webhookSecuritySummary = summarizeWebhookSecurity(runtime)
      metrics.webhook_security_total_endpoints =
        webhookSecuritySummary.totalEndpoints
      metrics.webhook_security_ready_endpoints =
        webhookSecuritySummary.verificationReadyEndpoints
      metrics.webhook_security_unready_endpoints =
        webhookSecuritySummary.verificationNotReadyEndpoints
      metrics.webhook_security_verification_ready =
        webhookSecuritySummary.verificationNotReadyEndpoints === 0 ? 1 : 0
      metrics.webhook_security_by_type = webhookSecuritySummary.byChannelType
      metrics.webhook_security_unready_by_requirement =
        webhookSecuritySummary.unreadySummary.byMissingRequirement
      if (latestWebhookSecurityPolicyChange) {
        metrics.webhook_security_policy_last_changed_at =
          latestWebhookSecurityPolicyChange.timestamp
        metrics.webhook_security_policy_last_changed_route =
          latestWebhookSecurityPolicyChange.route
        metrics.webhook_security_policy_last_changed_device =
          latestWebhookSecurityPolicyChange.device
      }
      if (runtime.channelReplayStore) {
        const replayStats = await runtime.channelReplayStore.getStats()
        metrics.channel_replays_total = replayStats.totalRecords
        metrics.channel_replays_processing = replayStats.processingRecords
        metrics.channel_replays_processed = replayStats.processedRecords
        metrics.channel_replays_stale_processing = replayStats.staleProcessingRecords
        metrics.channel_replays_by_type = replayStats.byChannelType
      }
      if (runtime.channelSessionStore) {
        const sessionStats = await runtime.channelSessionStore.getStats()
        metrics.channel_sessions_total = sessionStats.totalBindings
        metrics.channel_sessions_stale = sessionStats.staleBindings
        metrics.channel_sessions_by_type = sessionStats.byChannelType
      }

      // Scheduled jobs
      metrics.cron_tasks = runtime.jobStore?.list({ status: ['pending'] }).length ?? 0

      // Telemetry
      if (runtime.telemetry) {
        metrics.telemetry = runtime.telemetry.getMetrics()
      }

      // LLM Cache stats
      if (runtime.llmCache) {
        metrics.llm_cache = runtime.llmCache.getStats()
      }
    }

    return { data: metrics }
  })
}
