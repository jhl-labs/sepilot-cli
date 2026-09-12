import type { IChannel } from '@sepilotd/core'
import type { SepilotdConfig } from '../../config/schema.js'
import type { GatewayClient } from '../../gateway/client.js'
import { createLogger } from '../../logger.js'
import { registerFeatureChannelFactories } from '../../generated/feature-registration.js'
import type {
  ChannelConfig,
  ChannelFactory,
  ChannelFactoryDeps,
  IChannelFactoryRegistry,
} from '../../plugins/contracts.js'

const log = createLogger('runtime.channels')

export class ChannelFactoryRegistry implements IChannelFactoryRegistry {
  private factories = new Map<string, ChannelFactory>()

  register(type: string, factory: ChannelFactory): void {
    this.factories.set(type, factory)
  }

  get(type: string): ChannelFactory | undefined {
    return this.factories.get(type)
  }

  has(type: string): boolean {
    return this.factories.has(type)
  }

  listTypes(): string[] {
    return Array.from(this.factories.keys()).sort()
  }
}

function registerBuiltinChannelFactories(registry: ChannelFactoryRegistry): void {
  registerFeatureChannelFactories(registry)
}

export function createChannelFactoryRegistry(): ChannelFactoryRegistry {
  const registry = new ChannelFactoryRegistry()
  registerBuiltinChannelFactories(registry)
  return registry
}

export function buildChannels(
  config: SepilotdConfig,
  gatewayClient: GatewayClient,
  factoryRegistry: IChannelFactoryRegistry = createChannelFactoryRegistry(),
  extraDeps: Partial<ChannelFactoryDeps> = {},
): IChannel[] {
  const channels: IChannel[] = []
  const deps: ChannelFactoryDeps = {
    gatewayClient,
    gatewayUrl: config.gateway.url,
    ...extraDeps,
  }

  for (const channel of config.channels) {
    if (!channel.enabled) continue

    const factory = factoryRegistry.get(channel.type)
    if (!factory) {
      log.warn(`Skipping channel ${channel.type}`, {
        reason: 'no factory registered for channel type',
      })
      continue
    }

    try {
      const result = factory(channel as ChannelConfig, deps)
      if (!result.channel) {
        log.warn(`Skipping channel ${channel.type}`, {
          reason: result.skipReason ?? 'factory returned no channel',
        })
        continue
      }

      channels.push(result.channel)
    } catch (error) {
      log.error(`Failed to create channel ${channel.type}`, {
        error: String(error),
      })
    }
  }

  return channels
}
