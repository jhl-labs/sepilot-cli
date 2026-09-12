import type { IChannel, ILLMProvider } from '@sepilotd/core'
import type { GraphAgentInfo, GraphAgentRegistry } from '../agent/graph/registry.js'
import type { SepilotdConfig } from '../config/schema.js'
import type { GatewayClient } from '../gateway/client.js'
import type { HookRegistry } from '../hook/registry.js'
import type { createLogger } from '../logger.js'
import type { FileSkillRegistry } from '../skills/registry.js'
import type { ToolRegistry } from '../tools/registry.js'

export type Logger = ReturnType<typeof createLogger>
export type ProviderConfig = SepilotdConfig['providers'][number]
export type ChannelConfig = SepilotdConfig['channels'][number]

export interface ProviderFactoryResult {
  provider?: ILLMProvider
  skipReason?: string
}

export type ProviderFactory = (
  config: ProviderConfig,
) => ProviderFactoryResult

export interface ChannelFactoryDeps {
  dataDir?: string
  gatewayClient: GatewayClient
  gatewayUrl: string
  onChannelUserPaired?: (
    channelType: string,
    userId: string,
  ) => Promise<void> | void
  loadTelegramPendingPairing?: () => Promise<{ code: string; expiresAt: number } | null>
  clearTelegramPendingPairing?: () => Promise<void>
}

export interface ChannelFactoryResult {
  channel?: IChannel
  skipReason?: string
}

export type ChannelFactory = (
  config: ChannelConfig,
  deps: ChannelFactoryDeps,
) => ChannelFactoryResult

export interface IProviderFactoryRegistry {
  register(type: string, factory: ProviderFactory): void
  get(type: string): ProviderFactory | undefined
  has(type: string): boolean
  listTypes(): string[]
}

export interface IChannelFactoryRegistry {
  register(type: string, factory: ChannelFactory): void
  get(type: string): ChannelFactory | undefined
  has(type: string): boolean
  listTypes(): string[]
}

export interface PluginEntry {
  id: string
  name: string
  version: string
  register(ctx: PluginContext): Promise<void> | void
  shutdown?(): Promise<void> | void
}

export type PluginGraphAgent = Omit<GraphAgentInfo, 'source'> & {
  source?: 'plugin'
}

export type PluginGraphRegistrar = (
  registry: GraphAgentRegistry,
  ctx: PluginContext,
) => Promise<void> | void

export interface PluginContext {
  pluginId: string
  pluginDir: string
  providers: IProviderFactoryRegistry
  channels: IChannelFactoryRegistry
  tools: ToolRegistry
  hooks: HookRegistry
  skills: FileSkillRegistry
  graphs: GraphAgentRegistry
  events: import('./event-bus.js').PluginEventBus
  log: Logger
  config: Record<string, unknown>
}
