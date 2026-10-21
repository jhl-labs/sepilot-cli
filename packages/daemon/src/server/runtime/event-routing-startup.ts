import {
  createEventDeliverTool,
  createEventDestinationsTool,
  EVENT_DELIVER_TOOL,
  EVENT_DESTINATIONS_TOOL,
} from '../../event-routing/agent-tools.js'
import { DEFAULT_EVENT_ROUTING_CONFIG } from '../../event-routing/config-schema.js'
import {
  createEventRoutingSessionCheck,
  registerCompletionProducers,
  type CompletionProducers,
} from '../../event-routing/producers/completion.js'
import type { JobRunnerLike } from '../../event-routing/producers/job.js'
import { createEventRoutingService } from '../../event-routing/service.js'
import { onJobFinished } from '../../jobs/repo.js'
import type { RuntimeServices } from './types.js'

export type EventRoutingStartupRuntime = Pick<
  RuntimeServices,
  | 'config'
  | 'channels'
  | 'hookRegistry'
  | 'sessions'
  | 'toolRegistry'
  | 'eventRouting'
  | 'eventRoutingLlmRunner'
  | 'eventRoutingProducers'
>

export type EventRoutingStopRuntime = Partial<Pick<RuntimeServices, 'eventRouting' | 'eventRoutingProducers' | 'toolRegistry'>>

const AGENT_TOOL_NAMES = [EVENT_DELIVER_TOOL, EVENT_DESTINATIONS_TOOL] as const

export function startEventRouting(
  runtime: EventRoutingStartupRuntime,
  jobs: JobRunnerLike = { onJobFinished },
): CompletionProducers {
  const getConfig = () => runtime.config.eventRouting ?? DEFAULT_EVENT_ROUTING_CONFIG
  const service = createEventRoutingService({
    getConfig,
    channels: () => runtime.channels,
    llmRunner: () => runtime.eventRoutingLlmRunner,
  })
  runtime.eventRouting = service
  service.start()

  runtime.eventRoutingProducers?.dispose()
  const producers = registerCompletionProducers({
    hooks: runtime.hookRegistry,
    jobs,
    ingest: (event) => service.ingest(event),
    getConfig,
    isEventRoutingSession: createEventRoutingSessionCheck((sessionId) => runtime.sessions.get(sessionId)),
  })
  runtime.eventRoutingProducers = producers

  const toolDeps = { getConfig, outbox: service.outbox, scheduleDrain: () => service.scheduleDrain() }
  runtime.toolRegistry.register(createEventDestinationsTool(toolDeps))
  runtime.toolRegistry.register(createEventDeliverTool(toolDeps))
  return producers
}

export async function stopEventRouting(runtime: EventRoutingStopRuntime): Promise<void> {
  for (const name of AGENT_TOOL_NAMES) runtime.toolRegistry?.unregister(name)
  runtime.eventRoutingProducers?.dispose()
  await runtime.eventRouting?.stop()
}
