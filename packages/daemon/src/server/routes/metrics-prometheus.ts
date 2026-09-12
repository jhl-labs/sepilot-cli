import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import type { OpenApiOverrideMap } from '../openapi.js'
import { getLatestChannelPipelineHealthPolicyChange } from '../runtime/channel-pipeline-health-audit.js'
import { summarizeWebhookSecurity } from '../runtime/webhook-endpoints.js'
import { getLatestWebhookSecurityPolicyChange } from '../runtime/webhook-security-policy-audit.js'

export const prometheusOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/metrics/prometheus': {
    get: {
      summary: 'Prometheus metrics',
      tags: ['System'],
      responses: {
        200: {
          description: 'Prometheus text format',
          content: {
            'text/plain': {
              schema: {
                type: 'string',
              },
            },
          },
        },
      },
    },
  },
}

export async function prometheusRoutes(app: FastifyInstance) {
  app.get('/metrics/prometheus', async (_request, reply) => {
    const runtime = app.runtime
    const lines: string[] = []

    // Process metrics
    lines.push(`# HELP sepilotd_uptime_seconds Daemon uptime in seconds`)
    lines.push(`# TYPE sepilotd_uptime_seconds gauge`)
    lines.push(`sepilotd_uptime_seconds ${process.uptime().toFixed(1)}`)

    lines.push(`# HELP sepilotd_memory_rss_bytes RSS memory in bytes`)
    lines.push(`# TYPE sepilotd_memory_rss_bytes gauge`)
    lines.push(`sepilotd_memory_rss_bytes ${process.memoryUsage().rss}`)

    lines.push(`# HELP sepilotd_memory_heap_bytes Heap memory in bytes`)
    lines.push(`# TYPE sepilotd_memory_heap_bytes gauge`)
    lines.push(`sepilotd_memory_heap_bytes ${process.memoryUsage().heapUsed}`)

    if (runtime) {
      try {
        const usage = runtime.usageTracker.getTotalUsage()
        lines.push(`# HELP sepilotd_tokens_input_total Total input tokens`)
        lines.push(`# TYPE sepilotd_tokens_input_total counter`)
        lines.push(`sepilotd_tokens_input_total ${usage.inputTokens}`)

        lines.push(`# HELP sepilotd_tokens_output_total Total output tokens`)
        lines.push(`# TYPE sepilotd_tokens_output_total counter`)
        lines.push(`sepilotd_tokens_output_total ${usage.outputTokens}`)

        lines.push(`# HELP sepilotd_cost_usd_total Total cost in USD`)
        lines.push(`# TYPE sepilotd_cost_usd_total counter`)
        lines.push(`sepilotd_cost_usd_total ${usage.costUsd}`)

        lines.push(`# HELP sepilotd_requests_total Total LLM requests`)
        lines.push(`# TYPE sepilotd_requests_total counter`)
        lines.push(`sepilotd_requests_total ${usage.requestCount}`)
      } catch { /* usage tracker not available */ }

      lines.push(`# HELP sepilotd_providers_count Number of configured providers`)
      lines.push(`# TYPE sepilotd_providers_count gauge`)
      lines.push(`sepilotd_providers_count ${runtime.providerRegistry.list().length}`)

      lines.push(`# HELP sepilotd_tools_count Number of registered tools`)
      lines.push(`# TYPE sepilotd_tools_count gauge`)
      lines.push(`sepilotd_tools_count ${runtime.toolRegistry.list().length}`)

      const runLimiterStats = runtime.runLimiter?.getStats()
      if (runLimiterStats) {
        lines.push(`# HELP sepilotd_run_limiter_active Active agent runs admitted by the concurrency gate`)
        lines.push(`# TYPE sepilotd_run_limiter_active gauge`)
        lines.push(`sepilotd_run_limiter_active ${runLimiterStats.active}`)

        lines.push(`# HELP sepilotd_run_limiter_queued Agent runs waiting for an execution slot`)
        lines.push(`# TYPE sepilotd_run_limiter_queued gauge`)
        lines.push(`sepilotd_run_limiter_queued ${runLimiterStats.queued}`)

        lines.push(`# HELP sepilotd_run_limiter_max_active Configured maximum active agent runs`)
        lines.push(`# TYPE sepilotd_run_limiter_max_active gauge`)
        lines.push(`sepilotd_run_limiter_max_active ${runLimiterStats.maxActive}`)

        lines.push(`# HELP sepilotd_run_limiter_max_queued Configured maximum queued agent runs`)
        lines.push(`# TYPE sepilotd_run_limiter_max_queued gauge`)
        lines.push(`sepilotd_run_limiter_max_queued ${runLimiterStats.maxQueued}`)

        lines.push(`# HELP sepilotd_run_limiter_accepting Whether the run limiter is accepting new work (1 = accepting)`)
        lines.push(`# TYPE sepilotd_run_limiter_accepting gauge`)
        lines.push(`sepilotd_run_limiter_accepting ${runLimiterStats.accepting ? 1 : 0}`)
      }

      const breakerSummary = runtime.providerCircuitBreaker?.getSummary()
      if (breakerSummary) {
        lines.push(`# HELP sepilotd_provider_circuit_breaker_tracked Provider/model circuits tracked by the breaker`)
        lines.push(`# TYPE sepilotd_provider_circuit_breaker_tracked gauge`)
        lines.push(`sepilotd_provider_circuit_breaker_tracked ${breakerSummary.trackedCircuits}`)

        lines.push(`# HELP sepilotd_provider_circuit_breaker_open Provider/model circuits currently open`)
        lines.push(`# TYPE sepilotd_provider_circuit_breaker_open gauge`)
        lines.push(`sepilotd_provider_circuit_breaker_open ${breakerSummary.openCircuits}`)

        lines.push(`# HELP sepilotd_provider_circuit_breaker_half_open Provider/model circuits currently half-open`)
        lines.push(`# TYPE sepilotd_provider_circuit_breaker_half_open gauge`)
        lines.push(`sepilotd_provider_circuit_breaker_half_open ${breakerSummary.halfOpenCircuits}`)
      }

      lines.push(`# HELP sepilotd_channels_total Total channels`)
      lines.push(`# TYPE sepilotd_channels_total gauge`)
      lines.push(`sepilotd_channels_total ${runtime.channels.length}`)

      const pipelineStats = runtime.channelPipelineMonitor?.getStats?.()
      if (pipelineStats) {
        const latestHealthPolicyChange =
          await getLatestChannelPipelineHealthPolicyChange(runtime)
        lines.push(`# HELP sepilotd_channel_pipeline_in_flight Channel messages currently in the processing pipeline`)
        lines.push(`# TYPE sepilotd_channel_pipeline_in_flight gauge`)
        lines.push(`sepilotd_channel_pipeline_in_flight ${pipelineStats.inFlight}`)

        lines.push(`# HELP sepilotd_channel_pipeline_total Channel pipeline terminal outcomes by outcome`)
        lines.push(`# TYPE sepilotd_channel_pipeline_total gauge`)
        lines.push(`sepilotd_channel_pipeline_total{outcome="processed"} ${pipelineStats.processedEvents}`)
        lines.push(`sepilotd_channel_pipeline_total{outcome="duplicate"} ${pipelineStats.duplicateEvents}`)
        lines.push(`sepilotd_channel_pipeline_total{outcome="blocked"} ${pipelineStats.blockedEvents}`)
        lines.push(`sepilotd_channel_pipeline_total{outcome="no_provider"} ${pipelineStats.noProviderEvents}`)
        lines.push(`sepilotd_channel_pipeline_total{outcome="error"} ${pipelineStats.errorEvents}`)

        lines.push(`# HELP sepilotd_channel_pipeline_by_type Channel pipeline terminal outcomes by channel type and outcome`)
        lines.push(`# TYPE sepilotd_channel_pipeline_by_type gauge`)
        for (const stat of pipelineStats.byChannelType) {
          const channelType = prometheusLabelValue(stat.channelType)
          lines.push(`sepilotd_channel_pipeline_by_type{channel_type="${channelType}",outcome="processed"} ${stat.processedEvents}`)
          lines.push(`sepilotd_channel_pipeline_by_type{channel_type="${channelType}",outcome="duplicate"} ${stat.duplicateEvents}`)
          lines.push(`sepilotd_channel_pipeline_by_type{channel_type="${channelType}",outcome="blocked"} ${stat.blockedEvents}`)
          lines.push(`sepilotd_channel_pipeline_by_type{channel_type="${channelType}",outcome="no_provider"} ${stat.noProviderEvents}`)
          lines.push(`sepilotd_channel_pipeline_by_type{channel_type="${channelType}",outcome="error"} ${stat.errorEvents}`)
        }

        lines.push(`# HELP sepilotd_channel_pipeline_stage_duration_ms Channel pipeline stage duration aggregates in milliseconds`)
        lines.push(`# TYPE sepilotd_channel_pipeline_stage_duration_ms gauge`)
        for (const stat of pipelineStats.byStage) {
          const stage = prometheusLabelValue(stat.stage)
          lines.push(`sepilotd_channel_pipeline_stage_duration_ms{stage="${stage}",stat="avg"} ${stat.avgDurationMs}`)
          lines.push(`sepilotd_channel_pipeline_stage_duration_ms{stage="${stage}",stat="max"} ${stat.maxDurationMs}`)
          lines.push(`sepilotd_channel_pipeline_stage_duration_ms{stage="${stage}",stat="last"} ${stat.lastDurationMs}`)
          lines.push(`sepilotd_channel_pipeline_stage_duration_ms{stage="${stage}",stat="sum"} ${stat.totalDurationMs}`)
          lines.push(`sepilotd_channel_pipeline_stage_duration_ms{stage="${stage}",stat="count"} ${stat.count}`)
        }

        lines.push(`# HELP sepilotd_channel_pipeline_recent_window_seconds Channel pipeline recent observation window in seconds`)
        lines.push(`# TYPE sepilotd_channel_pipeline_recent_window_seconds gauge`)
        lines.push(`sepilotd_channel_pipeline_recent_window_seconds ${(pipelineStats.recent.windowMs / 1000).toFixed(0)}`)

        lines.push(`# HELP sepilotd_channel_pipeline_recent_total Recent channel pipeline outcomes by outcome`)
        lines.push(`# TYPE sepilotd_channel_pipeline_recent_total gauge`)
        lines.push(`sepilotd_channel_pipeline_recent_total{outcome="processed"} ${pipelineStats.recent.processedEvents}`)
        lines.push(`sepilotd_channel_pipeline_recent_total{outcome="duplicate"} ${pipelineStats.recent.duplicateEvents}`)
        lines.push(`sepilotd_channel_pipeline_recent_total{outcome="blocked"} ${pipelineStats.recent.blockedEvents}`)
        lines.push(`sepilotd_channel_pipeline_recent_total{outcome="no_provider"} ${pipelineStats.recent.noProviderEvents}`)
        lines.push(`sepilotd_channel_pipeline_recent_total{outcome="error"} ${pipelineStats.recent.errorEvents}`)

        lines.push(`# HELP sepilotd_channel_pipeline_recent_failure_rate Recent channel pipeline failure rate over the observation window`)
        lines.push(`# TYPE sepilotd_channel_pipeline_recent_failure_rate gauge`)
        lines.push(`sepilotd_channel_pipeline_recent_failure_rate ${pipelineStats.recent.failureRate}`)

        lines.push(`# HELP sepilotd_channel_pipeline_recent_by_type Recent channel pipeline outcomes by channel type and outcome`)
        lines.push(`# TYPE sepilotd_channel_pipeline_recent_by_type gauge`)
        for (const stat of pipelineStats.recent.byChannelType) {
          const channelType = prometheusLabelValue(stat.channelType)
          lines.push(`sepilotd_channel_pipeline_recent_by_type{channel_type="${channelType}",outcome="processed"} ${stat.processedEvents}`)
          lines.push(`sepilotd_channel_pipeline_recent_by_type{channel_type="${channelType}",outcome="duplicate"} ${stat.duplicateEvents}`)
          lines.push(`sepilotd_channel_pipeline_recent_by_type{channel_type="${channelType}",outcome="blocked"} ${stat.blockedEvents}`)
          lines.push(`sepilotd_channel_pipeline_recent_by_type{channel_type="${channelType}",outcome="no_provider"} ${stat.noProviderEvents}`)
          lines.push(`sepilotd_channel_pipeline_recent_by_type{channel_type="${channelType}",outcome="error"} ${stat.errorEvents}`)
        }

        lines.push(`# HELP sepilotd_channel_pipeline_recent_stage_duration_ms Recent channel pipeline stage duration aggregates in milliseconds`)
        lines.push(`# TYPE sepilotd_channel_pipeline_recent_stage_duration_ms gauge`)
        for (const stat of pipelineStats.recent.byStage) {
          const stage = prometheusLabelValue(stat.stage)
          lines.push(`sepilotd_channel_pipeline_recent_stage_duration_ms{stage="${stage}",stat="avg"} ${stat.avgDurationMs}`)
          lines.push(`sepilotd_channel_pipeline_recent_stage_duration_ms{stage="${stage}",stat="max"} ${stat.maxDurationMs}`)
          lines.push(`sepilotd_channel_pipeline_recent_stage_duration_ms{stage="${stage}",stat="last"} ${stat.lastDurationMs}`)
          lines.push(`sepilotd_channel_pipeline_recent_stage_duration_ms{stage="${stage}",stat="sum"} ${stat.totalDurationMs}`)
          lines.push(`sepilotd_channel_pipeline_recent_stage_duration_ms{stage="${stage}",stat="count"} ${stat.count}`)
        }

        if (latestHealthPolicyChange) {
          lines.push(`# HELP sepilotd_channel_pipeline_health_policy_last_changed_timestamp_seconds Unix timestamp of the latest channel pipeline health policy change`)
          lines.push(`# TYPE sepilotd_channel_pipeline_health_policy_last_changed_timestamp_seconds gauge`)
          lines.push(
            `sepilotd_channel_pipeline_health_policy_last_changed_timestamp_seconds ${Math.floor(Date.parse(latestHealthPolicyChange.timestamp) / 1000)}`,
          )
          lines.push(`# HELP sepilotd_channel_pipeline_health_policy_last_change_info Latest channel pipeline health policy change metadata`)
          lines.push(`# TYPE sepilotd_channel_pipeline_health_policy_last_change_info gauge`)
          lines.push(
            `sepilotd_channel_pipeline_health_policy_last_change_info{route="${prometheusLabelValue(latestHealthPolicyChange.route)}",device="${prometheusLabelValue(latestHealthPolicyChange.device)}"} 1`,
          )
        }
      }

      const webhookSecuritySummary = summarizeWebhookSecurity(runtime)
      lines.push(`# HELP sepilotd_webhook_security_endpoints Inbound webhook endpoints by readiness state`)
      lines.push(`# TYPE sepilotd_webhook_security_endpoints gauge`)
      lines.push(
        `sepilotd_webhook_security_endpoints{state="total"} ${webhookSecuritySummary.totalEndpoints}`,
      )
      lines.push(
        `sepilotd_webhook_security_endpoints{state="ready"} ${webhookSecuritySummary.verificationReadyEndpoints}`,
      )
      lines.push(
        `sepilotd_webhook_security_endpoints{state="unready"} ${webhookSecuritySummary.verificationNotReadyEndpoints}`,
      )

      lines.push(`# HELP sepilotd_webhook_security_verification_ready Whether all configured inbound webhook endpoints are verification-ready`)
      lines.push(`# TYPE sepilotd_webhook_security_verification_ready gauge`)
      lines.push(
        `sepilotd_webhook_security_verification_ready ${webhookSecuritySummary.verificationNotReadyEndpoints === 0 ? 1 : 0}`,
      )

      lines.push(`# HELP sepilotd_webhook_security_by_type Inbound webhook endpoint readiness by channel type`)
      lines.push(`# TYPE sepilotd_webhook_security_by_type gauge`)
      for (const stat of webhookSecuritySummary.byChannelType) {
        const channelType = prometheusLabelValue(stat.channelType)
        lines.push(
          `sepilotd_webhook_security_by_type{channel_type="${channelType}",state="total"} ${stat.endpointCount}`,
        )
        lines.push(
          `sepilotd_webhook_security_by_type{channel_type="${channelType}",state="ready"} ${stat.verificationReadyCount}`,
        )
        lines.push(
          `sepilotd_webhook_security_by_type{channel_type="${channelType}",state="unready"} ${stat.verificationNotReadyCount}`,
        )
      }

      lines.push(`# HELP sepilotd_webhook_security_unready_by_requirement Inbound webhook endpoints grouped by missing verification requirement`)
      lines.push(`# TYPE sepilotd_webhook_security_unready_by_requirement gauge`)
      for (const stat of webhookSecuritySummary.unreadySummary.byMissingRequirement) {
        lines.push(
          `sepilotd_webhook_security_unready_by_requirement{requirement="${prometheusLabelValue(stat.requirement)}"} ${stat.endpointCount}`,
        )
      }

      const latestWebhookSecurityPolicyChange =
        await getLatestWebhookSecurityPolicyChange(runtime)
      if (latestWebhookSecurityPolicyChange) {
        lines.push(`# HELP sepilotd_webhook_security_policy_last_changed_timestamp_seconds Unix timestamp of the latest webhook security policy change`)
        lines.push(`# TYPE sepilotd_webhook_security_policy_last_changed_timestamp_seconds gauge`)
        lines.push(
          `sepilotd_webhook_security_policy_last_changed_timestamp_seconds ${Math.floor(Date.parse(latestWebhookSecurityPolicyChange.timestamp) / 1000)}`,
        )
        lines.push(`# HELP sepilotd_webhook_security_policy_last_change_info Latest webhook security policy change metadata`)
        lines.push(`# TYPE sepilotd_webhook_security_policy_last_change_info gauge`)
        lines.push(
          `sepilotd_webhook_security_policy_last_change_info{route="${prometheusLabelValue(latestWebhookSecurityPolicyChange.route)}",device="${prometheusLabelValue(latestWebhookSecurityPolicyChange.device)}"} 1`,
        )
      }

      if (runtime.channelReplayStore) {
        const replayStats = await runtime.channelReplayStore.getStats()

        lines.push(`# HELP sepilotd_channel_replays_total Active replay journal entries by state`)
        lines.push(`# TYPE sepilotd_channel_replays_total gauge`)
        lines.push(`sepilotd_channel_replays_total{state="processing"} ${replayStats.processingRecords}`)
        lines.push(`sepilotd_channel_replays_total{state="processed"} ${replayStats.processedRecords}`)

        lines.push(`# HELP sepilotd_channel_replays_stale_processing Active replay entries stuck in processing beyond the stale threshold`)
        lines.push(`# TYPE sepilotd_channel_replays_stale_processing gauge`)
        lines.push(`sepilotd_channel_replays_stale_processing ${replayStats.staleProcessingRecords}`)

        lines.push(`# HELP sepilotd_channel_replays_by_type Active replay journal entries by channel type and state`)
        lines.push(`# TYPE sepilotd_channel_replays_by_type gauge`)
        for (const stat of replayStats.byChannelType) {
          const channelType = prometheusLabelValue(stat.channelType)
          lines.push(`sepilotd_channel_replays_by_type{channel_type="${channelType}",state="processing"} ${stat.processingRecords}`)
          lines.push(`sepilotd_channel_replays_by_type{channel_type="${channelType}",state="processed"} ${stat.processedRecords}`)
          lines.push(`sepilotd_channel_replays_by_type{channel_type="${channelType}",state="stale_processing"} ${stat.staleProcessingRecords}`)
        }
      }

      if (runtime.channelSessionStore) {
        const sessionStats = await runtime.channelSessionStore.getStats()

        lines.push(`# HELP sepilotd_channel_sessions_total Active channel session bindings`)
        lines.push(`# TYPE sepilotd_channel_sessions_total gauge`)
        lines.push(`sepilotd_channel_sessions_total ${sessionStats.totalBindings}`)

        lines.push(`# HELP sepilotd_channel_sessions_stale Active channel session bindings older than the stale threshold`)
        lines.push(`# TYPE sepilotd_channel_sessions_stale gauge`)
        lines.push(`sepilotd_channel_sessions_stale ${sessionStats.staleBindings}`)

        lines.push(`# HELP sepilotd_channel_sessions_by_type Active channel session bindings by channel type`)
        lines.push(`# TYPE sepilotd_channel_sessions_by_type gauge`)
        for (const stat of sessionStats.byChannelType) {
          const channelType = prometheusLabelValue(stat.channelType)
          lines.push(`sepilotd_channel_sessions_by_type{channel_type="${channelType}",state="active"} ${stat.totalBindings}`)
          lines.push(`sepilotd_channel_sessions_by_type{channel_type="${channelType}",state="stale"} ${stat.staleBindings}`)
        }
      }

      if (runtime.llmCache) {
        const stats = runtime.llmCache.getStats()
        lines.push(`# HELP sepilotd_cache_size LLM cache entries`)
        lines.push(`# TYPE sepilotd_cache_size gauge`)
        lines.push(`sepilotd_cache_size ${stats.size}`)
      }

      if (runtime.semanticIndex && typeof runtime.semanticIndex.getStatus === 'function') {
        const semantic = runtime.semanticIndex.getStatus()

        lines.push(`# HELP sepilotd_memory_semantic_status Semantic memory status by state (1 = current state)`)
        lines.push(`# TYPE sepilotd_memory_semantic_status gauge`)
        for (const status of ['disabled', 'ready', 'backfilling', 'degraded', 'reindex_required'] as const) {
          lines.push(`sepilotd_memory_semantic_status{status="${status}"} ${semantic.status === status ? 1 : 0}`)
        }

        lines.push(`# HELP sepilotd_memory_semantic_pending Pending semantic memory rows`)
        lines.push(`# TYPE sepilotd_memory_semantic_pending gauge`)
        lines.push(`sepilotd_memory_semantic_pending ${semantic.pendingCount}`)

        lines.push(`# HELP sepilotd_memory_semantic_failed Failed semantic memory rows`)
        lines.push(`# TYPE sepilotd_memory_semantic_failed gauge`)
        lines.push(`sepilotd_memory_semantic_failed ${semantic.failedCount}`)

        lines.push(`# HELP sepilotd_memory_semantic_vec_available sqlite-vec availability (1 = available)`)
        lines.push(`# TYPE sepilotd_memory_semantic_vec_available gauge`)
        lines.push(`sepilotd_memory_semantic_vec_available ${semantic.vecAvailable ? 1 : 0}`)

        lines.push(`# HELP sepilotd_memory_semantic_backend_info Semantic memory vector backend metadata`)
        lines.push(`# TYPE sepilotd_memory_semantic_backend_info gauge`)
        lines.push(
          `sepilotd_memory_semantic_backend_info{backend="${prometheusLabelValue(semantic.vectorBackend ?? 'unknown')}",available="${semantic.backendAvailable ? 'true' : 'false'}"} 1`,
        )

        if (semantic.dimensions) {
          lines.push(`# HELP sepilotd_memory_semantic_dimensions Active semantic embedding dimensions`)
          lines.push(`# TYPE sepilotd_memory_semantic_dimensions gauge`)
          lines.push(`sepilotd_memory_semantic_dimensions ${semantic.dimensions}`)
        }

        if (semantic.configuredProviderId && semantic.configuredModel) {
          lines.push(`# HELP sepilotd_memory_semantic_config_info Semantic memory config metadata`)
          lines.push(`# TYPE sepilotd_memory_semantic_config_info gauge`)
          lines.push(
            `sepilotd_memory_semantic_config_info{provider_id="${prometheusLabelValue(semantic.configuredProviderId)}",model="${prometheusLabelValue(semantic.configuredModel)}"} 1`,
          )
        }
      }
    }

    reply.header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
    return lines.join('\n') + '\n'
  })
}

function prometheusLabelValue(value: string | number | boolean | null | undefined): string {
  return String(value ?? 'unknown')
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
}
