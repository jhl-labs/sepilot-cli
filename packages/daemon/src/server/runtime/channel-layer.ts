import type { SepilotdConfig } from '../../config/schema.js'
import type { GatewayClient } from '../../gateway/client.js'
import { buildChannels, type createChannelFactoryRegistry } from './channels.js'
import {
  clearTelegramPendingPairing,
  createPersistingChannelUserPairHandler,
  readTelegramPendingPairing,
} from './channel-pairing-persistence.js'
import type { buildChannelAcl } from './security.js'

type ChannelFactoryRegistry = ReturnType<typeof createChannelFactoryRegistry>
type ChannelAclInstance = ReturnType<typeof buildChannelAcl>
type ApplyConfigMutation = <T>(
  description: string,
  fn: () => Promise<T>,
) => Promise<T>

export interface ChannelLayer {
  channels: ReturnType<typeof buildChannels>
}

/**
 * Build the channel adapter list. Needs the factory registry
 * (already populated with plugin-contributed factories by the
 * time this runs), the gateway client, and the ACL so newly
 * paired users persist to config.
 */
export function assembleChannelLayer(args: {
  config: SepilotdConfig
  getConfig?: () => SepilotdConfig
  dataDir: string
  gatewayClient: GatewayClient
  channelFactoryRegistry: ChannelFactoryRegistry
  channelAcl: ChannelAclInstance
  getChannelAcl?: () => ChannelAclInstance
  applyConfigMutation?: ApplyConfigMutation
  persistConfig?: (config: SepilotdConfig) => Promise<void>
}): ChannelLayer {
  const {
    config,
    getConfig,
    dataDir,
    gatewayClient,
    channelFactoryRegistry,
    channelAcl,
    getChannelAcl,
    applyConfigMutation,
    persistConfig,
  } = args

  const onChannelUserPaired = createPersistingChannelUserPairHandler({
    config: getConfig ? undefined : config,
    getConfig,
    dataDir,
    addAllowedUser: (channelType, userId) => {
      const acl = getChannelAcl?.() ?? channelAcl
      acl.addAllowedUser(channelType, userId)
    },
    applyConfigMutation,
    persistConfig,
  })

  const channels = buildChannels(
    config,
    gatewayClient,
    channelFactoryRegistry,
    {
      dataDir,
      onChannelUserPaired,
      loadTelegramPendingPairing: () => readTelegramPendingPairing(dataDir),
      clearTelegramPendingPairing: () => clearTelegramPendingPairing(dataDir),
    },
  )

  return { channels }
}
