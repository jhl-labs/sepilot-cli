import type {
  ChannelPipelineHealthConfig,
  ChannelPipelineHealthOverrideConfig,
  ChannelPipelineHealthThresholdConfig,
} from '../../config/schema.js'
import {
  DEFAULT_CHANNEL_PIPELINE_HEALTH_CONFIG,
} from '../../config/schema.js'

interface RuntimeConfigShape {
  observability?: {
    channelPipelineHealth?: Partial<ChannelPipelineHealthConfig> & {
      byChannelType?: Record<string, ChannelPipelineHealthOverrideConfig | undefined>
    }
  }
}

export function resolveChannelPipelineHealthConfig(
  config?: RuntimeConfigShape,
): ChannelPipelineHealthConfig {
  const raw = config?.observability?.channelPipelineHealth

  return {
    minRecentEvents:
      raw?.minRecentEvents
      ?? DEFAULT_CHANNEL_PIPELINE_HEALTH_CONFIG.minRecentEvents,
    degradeFailureRate:
      raw?.degradeFailureRate
      ?? DEFAULT_CHANNEL_PIPELINE_HEALTH_CONFIG.degradeFailureRate,
    minAgentSamples:
      raw?.minAgentSamples
      ?? DEFAULT_CHANNEL_PIPELINE_HEALTH_CONFIG.minAgentSamples,
    degradeAgentAvgLatencyMs:
      raw?.degradeAgentAvgLatencyMs
      ?? DEFAULT_CHANNEL_PIPELINE_HEALTH_CONFIG.degradeAgentAvgLatencyMs,
    hotChannelTopN:
      raw?.hotChannelTopN
      ?? DEFAULT_CHANNEL_PIPELINE_HEALTH_CONFIG.hotChannelTopN,
    byChannelType: Object.fromEntries(
      Object.entries(raw?.byChannelType ?? {}).map(([channelType, override]) => [
        channelType,
        { ...(override ?? {}) },
      ]),
    ),
  }
}

export function resolveChannelPipelineHealthThresholds(
  config?: RuntimeConfigShape,
  channelType?: string,
): ChannelPipelineHealthThresholdConfig {
  const policy = resolveChannelPipelineHealthConfig(config)
  const override = channelType
    ? policy.byChannelType[channelType]
    : undefined

  return {
    minRecentEvents: override?.minRecentEvents ?? policy.minRecentEvents,
    degradeFailureRate: override?.degradeFailureRate ?? policy.degradeFailureRate,
    minAgentSamples: override?.minAgentSamples ?? policy.minAgentSamples,
    degradeAgentAvgLatencyMs:
      override?.degradeAgentAvgLatencyMs ?? policy.degradeAgentAvgLatencyMs,
  }
}
