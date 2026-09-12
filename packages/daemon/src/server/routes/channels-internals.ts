import type { z } from 'zod'
import type {
  channelPipelineRecentQuerySchema,
  pipelineRecentChannelStatsSchema,
  pipelineRecentStatsSchema,
} from './channels-schema.js'

type CatalogStatus =
  | 'connected'
  | 'disconnected'
  | 'connecting'
  | 'error'
  | 'not_configured'

export function summarizeChannelStatus(statuses: string[]): CatalogStatus {
  if (statuses.length === 0) {
    return 'not_configured'
  }
  if (statuses.includes('error')) {
    return 'error'
  }
  if (statuses.includes('connecting')) {
    return 'connecting'
  }
  if (statuses.includes('connected')) {
    return 'connected'
  }
  return 'disconnected'
}

export function emptyPipelineStats() {
  return {
    inFlight: 0,
    totalEvents: 0,
    processedEvents: 0,
    duplicateEvents: 0,
    blockedEvents: 0,
    noProviderEvents: 0,
    errorEvents: 0,
    byChannelType: [],
    byStage: [],
    recent: {
      windowMs: 0,
      totalEvents: 0,
      processedEvents: 0,
      duplicateEvents: 0,
      blockedEvents: 0,
      noProviderEvents: 0,
      errorEvents: 0,
      failureEvents: 0,
      failureRate: 0,
      byChannelType: [],
      byStage: [],
      failureSamples: [],
    },
  }
}

export function emptyPipelineRecentStats(windowMs = 0) {
  return {
    windowMs,
    totalEvents: 0,
    processedEvents: 0,
    duplicateEvents: 0,
    blockedEvents: 0,
    noProviderEvents: 0,
    errorEvents: 0,
    failureEvents: 0,
    failureRate: 0,
    byChannelType: [],
    byStage: [],
    failureSamples: [],
  }
}

export function aggregatePipelineStageStats(
  entries: Array<{
    byStage: Array<{
      stage: string
      count: number
      totalDurationMs: number
      avgDurationMs: number
      maxDurationMs: number
      lastDurationMs: number
    }>
  }>,
) {
  const byStage = new Map<string, {
    stage: string
    count: number
    totalDurationMs: number
    maxDurationMs: number
    lastDurationMs: number
  }>()

  for (const entry of entries) {
    for (const stage of entry.byStage) {
      const current = byStage.get(stage.stage) ?? {
        stage: stage.stage,
        count: 0,
        totalDurationMs: 0,
        maxDurationMs: 0,
        lastDurationMs: 0,
      }
      current.count += stage.count
      current.totalDurationMs += stage.totalDurationMs
      current.maxDurationMs = Math.max(current.maxDurationMs, stage.maxDurationMs)
      current.lastDurationMs = Math.max(current.lastDurationMs, stage.lastDurationMs)
      byStage.set(stage.stage, current)
    }
  }

  return Array.from(byStage.values())
    .sort((left, right) => left.stage.localeCompare(right.stage))
    .map((entry) => ({
      stage: entry.stage,
      count: entry.count,
      totalDurationMs: entry.totalDurationMs,
      avgDurationMs: entry.count > 0
        ? Math.round(entry.totalDurationMs / entry.count)
        : 0,
      maxDurationMs: entry.maxDurationMs,
      lastDurationMs: entry.lastDurationMs,
    }))
}

export function filterPipelineRecentStats(
  recent: z.infer<typeof pipelineRecentStatsSchema>,
  query: z.infer<typeof channelPipelineRecentQuerySchema>,
) {
  const getOutcomeCount = (
    entry: z.infer<typeof pipelineRecentChannelStatsSchema>,
  ): number => {
    switch (query.outcome) {
      case 'processed':
        return entry.processedEvents
      case 'duplicate':
        return entry.duplicateEvents
      case 'blocked':
        return entry.blockedEvents
      case 'no_provider':
        return entry.noProviderEvents
      case 'error':
        return entry.errorEvents
      default:
        return entry.totalEvents
    }
  }

  const filteredByChannelType = recent.byChannelType.filter((entry) => {
    if (query.channelType && entry.channelType !== query.channelType) {
      return false
    }
    if (query.outcome && getOutcomeCount(entry) <= 0) {
      return false
    }
    if (query.failureOnly && entry.failureEvents <= 0) {
      return false
    }
    return true
  })

  const filteredFailureSamples = recent.failureSamples.filter((sample) => {
    if (query.channelType && sample.channelType !== query.channelType) {
      return false
    }
    if (query.outcome && sample.outcome !== query.outcome) {
      return false
    }
    return true
  })
  const pagedFailureSamples = filteredFailureSamples.slice(
    query.sampleOffset,
    query.sampleOffset + query.sampleLimit,
  )

  const totals = filteredByChannelType.reduce(
    (summary, entry) => {
      summary.totalEvents += entry.totalEvents
      summary.processedEvents += entry.processedEvents
      summary.duplicateEvents += entry.duplicateEvents
      summary.blockedEvents += entry.blockedEvents
      summary.noProviderEvents += entry.noProviderEvents
      summary.errorEvents += entry.errorEvents
      summary.failureEvents += entry.failureEvents
      return summary
    },
    emptyPipelineRecentStats(recent.windowMs),
  )

  return {
    data: {
      ...totals,
      failureRate: totals.totalEvents > 0
        ? Number((totals.failureEvents / totals.totalEvents).toFixed(3))
        : 0,
      byChannelType: filteredByChannelType,
      byStage: aggregatePipelineStageStats(filteredByChannelType),
      failureSamples: pagedFailureSamples,
    },
    meta: {
      filters: {
        channelType: query.channelType ?? null,
        outcome: query.outcome ?? null,
        failureOnly: query.failureOnly,
        sampleOffset: query.sampleOffset,
        sampleLimit: query.sampleLimit,
      },
      samples: {
        totalFailureSamples: filteredFailureSamples.length,
        returnedFailureSamples: pagedFailureSamples.length,
      },
    },
  }
}
