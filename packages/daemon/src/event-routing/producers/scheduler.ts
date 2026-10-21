import { createLogger } from '../../logger.js'
import type { EventRoutingConfig } from '../config-schema.js'
import type { RoutedEventInput } from '../types.js'
import { anyRuleMatchesSource, redactForEvent } from './shared.js'

const logger = createLogger('event-routing')

const OUTPUT_MAX = 8_000
const ERROR_MAX = 1_000

export interface SchedulerRunFinishedJob {
  id: string
  name: string
  kind: string
  channelType?: string | null
}

export interface SchedulerRunFinishedRun {
  id: string
  durationMs: number | null
  error: string | null
  outputExcerpt: string | null
  stopReason?: string | null
  agentSessionId?: string | null
}

export interface SchedulerRunFinished {
  job: SchedulerRunFinishedJob
  run: SchedulerRunFinishedRun
  status: 'succeeded' | 'failed' | 'interrupted' | 'cancelled'
  trigger: 'scheduled' | 'manual'
  willRetry?: boolean
}

export function schedulerEventFromRun(f: SchedulerRunFinished): RoutedEventInput {
  const { job, run } = f
  return {
    source: 'scheduler',
    type: f.status,
    subject: job.name,
    dedupeKey: `scheduler:${run.id}`,
    occurredAt: Date.now(),
    payload: {
      jobId: job.id,
      jobName: job.name,
      kind: job.kind,
      runId: run.id,
      trigger: f.trigger,
      status: f.status,
      durationMs: run.durationMs ?? undefined,
      stopReason: run.stopReason ?? undefined,
      willRetry: f.willRetry,
      output: redactForEvent(run.outputExcerpt ?? undefined, OUTPUT_MAX),
      error: redactForEvent(run.error ?? undefined, ERROR_MAX),
      agentSessionId: run.agentSessionId ?? undefined,
      channelType: job.channelType ?? undefined,
    },
  }
}

export interface CreateSchedulerProducerDeps {
  ingest: (event: RoutedEventInput) => unknown
  getConfig: () => EventRoutingConfig
}

export function createSchedulerProducer(
  deps: CreateSchedulerProducerDeps,
): (f: SchedulerRunFinished) => void {
  return (f) => {
    try {
      if (!anyRuleMatchesSource(deps.getConfig(), 'scheduler')) return
      const event = schedulerEventFromRun(f)
      void Promise.resolve(deps.ingest(event)).catch((error) => {
        logger.error('scheduler producer failed to ingest', {
          error: error instanceof Error ? error.message : String(error),
        })
      })
    } catch (error) {
      logger.error('scheduler producer failed to build event', {
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }
}
