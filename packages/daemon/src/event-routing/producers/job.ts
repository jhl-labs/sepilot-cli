import { createLogger } from '../../logger.js'
import type { EventRoutingConfig } from '../config-schema.js'
import type { RoutedEventInput } from '../types.js'
import { anyRuleMatchesSource, redactForEvent } from './shared.js'

const logger = createLogger('event-routing')

const ERROR_MAX = 1_000

const STATUS_TYPE: Record<string, string> = {
  completed: 'completed',
  failed: 'failed',
  canceled: 'cancelled',
}

export interface JobFinishedJob {
  id: string
  kind: string
  status: string
  total: number
  succeeded: number
  failed: number
  error: string | null
}

export interface JobFinished {
  job: JobFinishedJob
}

export function jobEventFromJob(job: JobFinishedJob): RoutedEventInput | null {
  if (job.kind === 'hook') return null
  const type = STATUS_TYPE[job.status] ?? job.status
  return {
    source: 'job',
    type,
    subject: `${job.kind} job`,
    dedupeKey: `job:${job.id}:${job.status}`,
    occurredAt: Date.now(),
    payload: {
      jobId: job.id,
      kind: job.kind,
      status: job.status,
      total: job.total,
      completedItems: job.succeeded,
      failedItems: job.failed,
      error: redactForEvent(job.error ?? undefined, ERROR_MAX),
      sessionId: undefined,
    },
  }
}

export interface JobRunnerLike {
  onJobFinished(listener: (job: JobFinishedJob) => void): { dispose(): void }
}

export interface RegisterJobProducerDeps {
  runner: JobRunnerLike
  ingest: (event: RoutedEventInput) => unknown
  getConfig: () => EventRoutingConfig
}

export function registerJobProducer(deps: RegisterJobProducerDeps): { dispose(): void } {
  return deps.runner.onJobFinished((job) => {
    try {
      if (!anyRuleMatchesSource(deps.getConfig(), 'job')) return
      const event = jobEventFromJob(job)
      if (!event) return
      void Promise.resolve(deps.ingest(event)).catch((error) => {
        logger.error('job producer failed to ingest', {
          error: error instanceof Error ? error.message : String(error),
        })
      })
    } catch (error) {
      logger.error('job producer failed to build event', {
        error: error instanceof Error ? error.message : String(error),
      })
    }
  })
}
