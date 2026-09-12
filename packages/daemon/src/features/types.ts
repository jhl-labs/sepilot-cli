import type { ISessionStore } from '@sepilotd/core'
import type { SepilotdConfig } from '../config/schema.js'
import type { ToolRegistry } from '../tools/registry.js'
import type { GatewayClient } from '../gateway/client.js'
import type { FileSkillRegistry } from '../skills/registry.js'
import type { TaskDelegator } from '../agent/delegator.js'
import type { SemanticMemoryStore } from '../memory/types.js'
import type { DreamingEngine } from '../memory/dreaming.js'

export interface FeatureToolDeps {
  config: SepilotdConfig
  dataDir?: string
  gatewayClient: GatewayClient
  skillRegistry: FileSkillRegistry
  /** True only for loopback URLs exposed within this session/workspace capability. */
  isManagedLoopbackUrl?: (
    sessionId: string | undefined,
    rawUrl: string,
    workspaceRoot?: string,
  ) => boolean
  /** Unix bridge socket for an accessible managed-loopback URL, when available. */
  managedLoopbackSocketForUrl?: (
    sessionId: string | undefined,
    rawUrl: string,
    workspaceRoot?: string,
  ) => string | null
  /** delegation feature만 사용; tools.ts가 생성해 주입 */
  delegator?: TaskDelegator
}

export type FeatureToolRegistrar = (registry: ToolRegistry, deps: FeatureToolDeps) => void

/**
 * Deps for the second-pass tool/service registrations done inside
 * `buildRuntime` (after the storage layer exists). These entries re-register
 * feature tools with richer wiring than the `buildToolRegistry` pass — e.g.
 * the delegation tool with session-event append, apps tools with the semantic
 * index, and the a2a/acp runtime services. Kept separate from
 * `FeatureToolDeps` so the earlier `buildToolRegistry` pass need not supply
 * sessions/semanticIndex.
 */
export interface FeatureRuntimeToolDeps {
  dataDir: string
  delegator: TaskDelegator
  semanticIndex: SemanticMemoryStore
  sessions: ISessionStore
  deviceName: string
  /** Only supplied for the late acp registration (after dreaming is built). */
  dreaming?: DreamingEngine
}

export type FeatureRuntimeToolRegistrar = (
  registry: ToolRegistry,
  deps: FeatureRuntimeToolDeps,
) => void

export interface FeatureRouteDeps {
  prefix: string
}
