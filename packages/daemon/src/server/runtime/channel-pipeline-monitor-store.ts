import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import type {
  ChannelPipelineChannelStats,
  ChannelPipelineOutcome,
  ChannelPipelineStage,
} from './channel-pipeline-monitor.js'

export interface ChannelPipelineStageSnapshot {
  stage: ChannelPipelineStage
  count: number
  totalDurationMs: number
  maxDurationMs: number
  lastDurationMs: number
}

export interface ChannelPipelineRecentOutcomeSnapshot {
  timestampMs: number
  channelType: string
  outcome: ChannelPipelineOutcome
}

export interface ChannelPipelineRecentStageSnapshot {
  timestampMs: number
  stage: ChannelPipelineStage
  durationMs: number
  channelType?: string
}

export interface ChannelPipelineMonitorSnapshot {
  byChannelType: ChannelPipelineChannelStats[]
  byStage: ChannelPipelineStageSnapshot[]
  recentOutcomes: ChannelPipelineRecentOutcomeSnapshot[]
  recentStageSamples: ChannelPipelineRecentStageSnapshot[]
}

export class ChannelPipelineMonitorStore {
  constructor(private readonly filePath: string) {}

  async init(): Promise<void> {
    await mkdir(dirname(this.filePath), { recursive: true })
  }

  async load(): Promise<ChannelPipelineMonitorSnapshot | null> {
    await this.init()
    try {
      const raw = await readFile(this.filePath, 'utf-8')
      return JSON.parse(raw) as ChannelPipelineMonitorSnapshot
    } catch {
      return null
    }
  }

  async save(snapshot: ChannelPipelineMonitorSnapshot): Promise<void> {
    await this.init()
    await writeFile(
      this.filePath,
      JSON.stringify(snapshot, null, 2),
      'utf-8',
    )
  }
}
