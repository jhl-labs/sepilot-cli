import type { ISessionStore } from '@sepilotd/core'
import type { SepilotdConfig } from '../config/schema.js'
import type { ToolRegistry } from '../tools/registry.js'
import type { GatewayClient } from '../gateway/client.js'
import type { FileSkillRegistry } from '../skills/registry.js'
import type { TaskDelegator } from '../agent/delegator.js'
import type { SemanticMemoryStore } from '../memory/types.js'
import type { UsageTracker } from '../memory/usage-tracker.js'
import type { DreamingEngine } from '../memory/dreaming.js'
import type { RuntimeServices } from '../server/runtime/types.js'
import type { SchedulerRunFinished } from '../event-routing/producers/scheduler.js'

export interface FeatureToolDeps {
  config: SepilotdConfig
  dataDir?: string
  gatewayClient: GatewayClient
  skillRegistry: FileSkillRegistry
  isManagedLoopbackUrl?: (
    sessionId: string | undefined,
    rawUrl: string,
    workspaceRoot?: string,
  ) => boolean
  managedLoopbackSocketForUrl?: (
    sessionId: string | undefined,
    rawUrl: string,
    workspaceRoot?: string,
  ) => string | null
  delegator?: TaskDelegator
}

export type FeatureToolRegistrar = (registry: ToolRegistry, deps: FeatureToolDeps) => void

export interface FeatureRuntimeToolDeps {
  dataDir: string
  delegator: TaskDelegator
  semanticIndex: SemanticMemoryStore
  sessions: ISessionStore
  deviceName: string
  dreaming?: DreamingEngine
  usageTracker?: UsageTracker
}

export type FeatureRuntimeToolRegistrar = (
  registry: ToolRegistry,
  deps: FeatureRuntimeToolDeps,
) => void

export interface FeatureRouteDeps {
  prefix: string
}

export type FeatureEventRoutingRuntime = Pick<
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

export type FeatureEventRoutingStopRuntime = Partial<Pick<RuntimeServices, 'eventRouting' | 'eventRoutingProducers' | 'toolRegistry'>>

export interface FeatureEventRoutingHooks {
  onSchedulerRunFinished?: (finished: SchedulerRunFinished) => void
}
