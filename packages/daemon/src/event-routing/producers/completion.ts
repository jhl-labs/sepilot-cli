import type { EventRoutingConfig } from '../config-schema.js'
import { EVENT_ROUTING_SESSION_TAG } from '../origin.js'
import type { RoutedEventInput } from '../types.js'
import { registerAgentRunProducer, type HookRegistryLike } from './agent-run.js'
import { registerJobProducer, type JobRunnerLike } from './job.js'
import { createSchedulerProducer, type SchedulerRunFinished } from './scheduler.js'

export interface CompletionProducersDeps {
  hooks: HookRegistryLike
  jobs: JobRunnerLike
  ingest: (event: RoutedEventInput) => unknown
  getConfig: () => EventRoutingConfig
  isEventRoutingSession: (sessionId: string) => Promise<boolean>
}

export interface CompletionProducers {
  onSchedulerRunFinished: (finished: SchedulerRunFinished) => void
  dispose(): void
}

export function registerCompletionProducers(deps: CompletionProducersDeps): CompletionProducers {
  const { ingest, getConfig } = deps
  const agentRun = registerAgentRunProducer({ hooks: deps.hooks, ingest, getConfig, isEventRoutingSession: deps.isEventRoutingSession })
  const job = registerJobProducer({ runner: deps.jobs, ingest, getConfig })
  const scheduler = createSchedulerProducer({ ingest, getConfig })
  let disposed = false
  return {
    onSchedulerRunFinished: (finished) => {
      if (!disposed) scheduler(finished)
    },
    dispose() {
      if (disposed) return
      disposed = true
      agentRun.dispose()
      job.dispose()
    },
  }
}

export function createEventRoutingSessionCheck(
  getSession: (sessionId: string) => Promise<{ tags?: readonly string[] } | null | undefined>,
): (sessionId: string) => Promise<boolean> {
  return async (sessionId) => {
    try {
      return (await getSession(sessionId))?.tags?.includes(EVENT_ROUTING_SESSION_TAG) ?? false
    } catch {
      return false
    }
  }
}
