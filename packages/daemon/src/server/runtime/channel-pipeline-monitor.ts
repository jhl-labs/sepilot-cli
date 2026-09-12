import type {
  ChannelPipelineMonitorSnapshot,
  ChannelPipelineMonitorStore,
  ChannelPipelineRecentOutcomeSnapshot,
  ChannelPipelineRecentStageSnapshot,
  ChannelPipelineStageSnapshot,
} from './channel-pipeline-monitor-store.js'

export type ChannelPipelineOutcome =
  | 'processed'
  | 'duplicate'
  | 'blocked'
  | 'no_provider'
  | 'error'

export type ChannelPipelineStage =
  | 'normalize'
  | 'replay_claim'
  | 'access_check'
  | 'session_resolve'
  | 'agent_execute'
  | 'response_dispatch'
  | 'post_process'

export interface ChannelPipelineRunHandle {
  channelType: string
  finished: boolean
}

interface RecentOutcomeSample {
  timestampMs: number
  channelType: string
  outcome: ChannelPipelineOutcome
}

interface RecentStageSample {
  timestampMs: number
  stage: ChannelPipelineStage
  durationMs: number
  channelType?: string
}

export interface ChannelPipelineFailureSample {
  timestamp: string
  channelType: string
  outcome: Extract<ChannelPipelineOutcome, 'no_provider' | 'error'>
}

interface ChannelPipelineStageAccumulator {
  stage: ChannelPipelineStage
  count: number
  totalDurationMs: number
  maxDurationMs: number
  lastDurationMs: number
}

export interface ChannelPipelineRecentStats {
  windowMs: number
  totalEvents: number
  processedEvents: number
  duplicateEvents: number
  blockedEvents: number
  noProviderEvents: number
  errorEvents: number
  failureEvents: number
  failureRate: number
  byChannelType: ChannelPipelineRecentChannelStats[]
  byStage: ChannelPipelineStageStats[]
  failureSamples: ChannelPipelineFailureSample[]
}

export interface ChannelPipelineRecentChannelStats {
  channelType: string
  totalEvents: number
  processedEvents: number
  duplicateEvents: number
  blockedEvents: number
  noProviderEvents: number
  errorEvents: number
  failureEvents: number
  failureRate: number
  byStage: ChannelPipelineStageStats[]
}

export interface ChannelPipelineStageStats {
  stage: ChannelPipelineStage
  count: number
  totalDurationMs: number
  avgDurationMs: number
  maxDurationMs: number
  lastDurationMs: number
}

export interface ChannelPipelineChannelStats {
  channelType: string
  inFlight: number
  totalEvents: number
  processedEvents: number
  duplicateEvents: number
  blockedEvents: number
  noProviderEvents: number
  errorEvents: number
}

export interface ChannelPipelineStats {
  inFlight: number
  totalEvents: number
  processedEvents: number
  duplicateEvents: number
  blockedEvents: number
  noProviderEvents: number
  errorEvents: number
  byChannelType: ChannelPipelineChannelStats[]
  byStage: ChannelPipelineStageStats[]
  recent: ChannelPipelineRecentStats
}

export class ChannelPipelineMonitor {
  private static readonly RECENT_WINDOW_MS = 5 * 60 * 1000
  private static readonly RECENT_SAMPLE_LIMIT = 500
  private static readonly RECENT_FAILURE_SAMPLE_LIMIT = 20
  private inFlight = 0
  private readonly byChannelType = new Map<string, ChannelPipelineChannelStats>()
  private readonly byStage = new Map<ChannelPipelineStage, ChannelPipelineStageAccumulator>()
  private readonly recentOutcomes: RecentOutcomeSample[] = []
  private readonly recentStageSamples: RecentStageSample[] = []
  private persistQueue = Promise.resolve()

  constructor(private readonly store?: ChannelPipelineMonitorStore) {}

  async init(now = new Date()): Promise<void> {
    if (!this.store) {
      return
    }

    const snapshot = await this.store.load()
    if (!snapshot) {
      return
    }

    this.restore(snapshot)
    this.pruneRecent(now.getTime())
    this.schedulePersist()
  }

  start(channelType: string): ChannelPipelineRunHandle {
    this.inFlight++
    const stats = this.getOrCreateChannelStats(channelType)
    stats.inFlight++
    this.schedulePersist()
    return {
      channelType,
      finished: false,
    }
  }

  finish(run?: ChannelPipelineRunHandle): void {
    if (!run || run.finished) {
      return
    }

    run.finished = true
    this.inFlight = Math.max(0, this.inFlight - 1)
    const stats = this.getOrCreateChannelStats(run.channelType)
    stats.inFlight = Math.max(0, stats.inFlight - 1)
    this.schedulePersist()
  }

  record(
    channelType: string,
    outcome: ChannelPipelineOutcome,
    now = new Date(),
  ): void {
    const timestampMs = now.getTime()
    const stats = this.getOrCreateChannelStats(channelType)

    stats.totalEvents++
    switch (outcome) {
      case 'processed':
        stats.processedEvents++
        break
      case 'duplicate':
        stats.duplicateEvents++
        break
      case 'blocked':
        stats.blockedEvents++
        break
      case 'no_provider':
        stats.noProviderEvents++
        break
      case 'error':
        stats.errorEvents++
        break
    }

    this.byChannelType.set(channelType, stats)
    this.recentOutcomes.push({ timestampMs, channelType, outcome })
    this.pruneRecent(timestampMs)
    this.schedulePersist()
  }

  recordStage(
    stage: ChannelPipelineStage,
    durationMs: number,
    now = new Date(),
    channelType?: string,
  ): void {
    const timestampMs = now.getTime()
    const normalizedDuration = Math.max(0, Math.round(durationMs))
    const stats = this.byStage.get(stage) ?? {
      stage,
      count: 0,
      totalDurationMs: 0,
      maxDurationMs: 0,
      lastDurationMs: 0,
    }

    stats.count++
    stats.totalDurationMs += normalizedDuration
    stats.maxDurationMs = Math.max(stats.maxDurationMs, normalizedDuration)
    stats.lastDurationMs = normalizedDuration
    this.byStage.set(stage, stats)
    this.recentStageSamples.push({
      timestampMs,
      stage,
      durationMs: normalizedDuration,
      channelType,
    })
    this.pruneRecent(timestampMs)
    this.schedulePersist()
  }

  getStats(now = new Date()): ChannelPipelineStats {
    const nowMs = now.getTime()
    this.pruneRecent(nowMs)
    const byChannelType = Array.from(this.byChannelType.values()).sort((left, right) =>
      left.channelType.localeCompare(right.channelType),
    )
    const byStage = this.toStageStats(Array.from(this.byStage.values()))
    const recent = this.buildRecentStats()

    return byChannelType.reduce<ChannelPipelineStats>(
      (summary, entry) => {
        summary.totalEvents += entry.totalEvents
        summary.processedEvents += entry.processedEvents
        summary.duplicateEvents += entry.duplicateEvents
        summary.blockedEvents += entry.blockedEvents
        summary.noProviderEvents += entry.noProviderEvents
        summary.errorEvents += entry.errorEvents
        summary.byChannelType.push(entry)
        return summary
      },
      {
        inFlight: this.inFlight,
        totalEvents: 0,
        processedEvents: 0,
        duplicateEvents: 0,
        blockedEvents: 0,
        noProviderEvents: 0,
        errorEvents: 0,
        byChannelType: [],
        byStage,
        recent,
      },
    )
  }

  private getOrCreateChannelStats(channelType: string): ChannelPipelineChannelStats {
    return this.byChannelType.get(channelType) ?? {
      channelType,
      inFlight: 0,
      totalEvents: 0,
      processedEvents: 0,
      duplicateEvents: 0,
      blockedEvents: 0,
      noProviderEvents: 0,
      errorEvents: 0,
    }
  }

  async flush(): Promise<void> {
    await this.persistQueue
  }

  private buildRecentStats(): ChannelPipelineRecentStats {
    const stageAccumulators = new Map<ChannelPipelineStage, ChannelPipelineStageAccumulator>()
    const byChannelType = new Map<
      string,
      {
        channelType: string
        totalEvents: number
        processedEvents: number
        duplicateEvents: number
        blockedEvents: number
        noProviderEvents: number
        errorEvents: number
        stageAccumulators: Map<ChannelPipelineStage, ChannelPipelineStageAccumulator>
      }
    >()
    const failureSamples: ChannelPipelineFailureSample[] = []
    let totalEvents = 0
    let processedEvents = 0
    let duplicateEvents = 0
    let blockedEvents = 0
    let noProviderEvents = 0
    let errorEvents = 0

    for (const sample of this.recentOutcomes) {
      const channelStats = byChannelType.get(sample.channelType) ?? {
        channelType: sample.channelType,
        totalEvents: 0,
        processedEvents: 0,
        duplicateEvents: 0,
        blockedEvents: 0,
        noProviderEvents: 0,
        errorEvents: 0,
        stageAccumulators: new Map<ChannelPipelineStage, ChannelPipelineStageAccumulator>(),
      }

      totalEvents++
      channelStats.totalEvents++
      switch (sample.outcome) {
        case 'processed':
          processedEvents++
          channelStats.processedEvents++
          break
        case 'duplicate':
          duplicateEvents++
          channelStats.duplicateEvents++
          break
        case 'blocked':
          blockedEvents++
          channelStats.blockedEvents++
          break
        case 'no_provider':
          noProviderEvents++
          channelStats.noProviderEvents++
          failureSamples.push({
            timestamp: new Date(sample.timestampMs).toISOString(),
            channelType: sample.channelType,
            outcome: 'no_provider',
          })
          break
        case 'error':
          errorEvents++
          channelStats.errorEvents++
          failureSamples.push({
            timestamp: new Date(sample.timestampMs).toISOString(),
            channelType: sample.channelType,
            outcome: 'error',
          })
          break
      }

      byChannelType.set(sample.channelType, channelStats)
    }

    for (const sample of this.recentStageSamples) {
      const current = stageAccumulators.get(sample.stage) ?? {
        stage: sample.stage,
        count: 0,
        totalDurationMs: 0,
        maxDurationMs: 0,
        lastDurationMs: 0,
      }
      current.count++
      current.totalDurationMs += sample.durationMs
      current.maxDurationMs = Math.max(current.maxDurationMs, sample.durationMs)
      current.lastDurationMs = sample.durationMs
      stageAccumulators.set(sample.stage, current)

      if (sample.channelType) {
        const channelStats = byChannelType.get(sample.channelType) ?? {
          channelType: sample.channelType,
          totalEvents: 0,
          processedEvents: 0,
          duplicateEvents: 0,
          blockedEvents: 0,
          noProviderEvents: 0,
          errorEvents: 0,
          stageAccumulators: new Map<ChannelPipelineStage, ChannelPipelineStageAccumulator>(),
        }
        const stageStats = channelStats.stageAccumulators.get(sample.stage) ?? {
          stage: sample.stage,
          count: 0,
          totalDurationMs: 0,
          maxDurationMs: 0,
          lastDurationMs: 0,
        }
        stageStats.count++
        stageStats.totalDurationMs += sample.durationMs
        stageStats.maxDurationMs = Math.max(stageStats.maxDurationMs, sample.durationMs)
        stageStats.lastDurationMs = sample.durationMs
        channelStats.stageAccumulators.set(sample.stage, stageStats)
        byChannelType.set(sample.channelType, channelStats)
      }
    }

    const failureEvents = errorEvents + noProviderEvents
    return {
      windowMs: ChannelPipelineMonitor.RECENT_WINDOW_MS,
      totalEvents,
      processedEvents,
      duplicateEvents,
      blockedEvents,
      noProviderEvents,
      errorEvents,
      failureEvents,
      failureRate: totalEvents > 0
        ? Number((failureEvents / totalEvents).toFixed(3))
        : 0,
      byChannelType: Array.from(byChannelType.values())
        .sort((left, right) => left.channelType.localeCompare(right.channelType))
        .map((entry) => {
          const channelFailureEvents = entry.errorEvents + entry.noProviderEvents
          return {
            channelType: entry.channelType,
            totalEvents: entry.totalEvents,
            processedEvents: entry.processedEvents,
            duplicateEvents: entry.duplicateEvents,
            blockedEvents: entry.blockedEvents,
            noProviderEvents: entry.noProviderEvents,
            errorEvents: entry.errorEvents,
            failureEvents: channelFailureEvents,
            failureRate: entry.totalEvents > 0
              ? Number((channelFailureEvents / entry.totalEvents).toFixed(3))
              : 0,
            byStage: this.toStageStats(Array.from(entry.stageAccumulators.values())),
          }
        }),
      byStage: this.toStageStats(Array.from(stageAccumulators.values())),
      failureSamples: failureSamples
        .slice(-ChannelPipelineMonitor.RECENT_FAILURE_SAMPLE_LIMIT)
        .reverse(),
    }
  }

  private toStageStats(
    entries: ChannelPipelineStageAccumulator[],
  ): ChannelPipelineStageStats[] {
    return entries
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

  private pruneRecent(nowMs: number): void {
    const minTimestampMs = nowMs - ChannelPipelineMonitor.RECENT_WINDOW_MS
    while (
      this.recentOutcomes.length > 0
      && this.recentOutcomes[0]!.timestampMs < minTimestampMs
    ) {
      this.recentOutcomes.shift()
    }
    while (
      this.recentStageSamples.length > 0
      && this.recentStageSamples[0]!.timestampMs < minTimestampMs
    ) {
      this.recentStageSamples.shift()
    }
    while (this.recentOutcomes.length > ChannelPipelineMonitor.RECENT_SAMPLE_LIMIT) {
      this.recentOutcomes.shift()
    }
    while (this.recentStageSamples.length > ChannelPipelineMonitor.RECENT_SAMPLE_LIMIT) {
      this.recentStageSamples.shift()
    }
  }

  private restore(snapshot: ChannelPipelineMonitorSnapshot): void {
    this.inFlight = 0

    this.byChannelType.clear()
    for (const entry of snapshot.byChannelType ?? []) {
      this.byChannelType.set(entry.channelType, {
        ...entry,
        inFlight: 0,
      })
    }

    this.byStage.clear()
    for (const entry of snapshot.byStage ?? []) {
      this.byStage.set(entry.stage, { ...entry })
    }

    this.recentOutcomes.length = 0
    for (const entry of snapshot.recentOutcomes ?? []) {
      this.recentOutcomes.push({ ...entry })
    }

    this.recentStageSamples.length = 0
    for (const entry of snapshot.recentStageSamples ?? []) {
      this.recentStageSamples.push({ ...entry })
    }
  }

  private schedulePersist(): void {
    if (!this.store) {
      return
    }

    const snapshot = this.snapshot()
    this.persistQueue = this.persistQueue
      .catch(() => undefined)
      .then(() => this.store!.save(snapshot))
  }

  private snapshot(): ChannelPipelineMonitorSnapshot {
    return {
      byChannelType: Array.from(this.byChannelType.values()).map((entry) => ({
        ...entry,
        inFlight: 0,
      })),
      byStage: Array.from(this.byStage.values()).map<ChannelPipelineStageSnapshot>(
        (entry) => ({ ...entry }),
      ),
      recentOutcomes: this.recentOutcomes.map<ChannelPipelineRecentOutcomeSnapshot>(
        (entry) => ({ ...entry }),
      ),
      recentStageSamples: this.recentStageSamples.map<ChannelPipelineRecentStageSnapshot>(
        (entry) => ({ ...entry }),
      ),
    }
  }
}
