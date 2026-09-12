import { WebchatChannel } from '../../channels/webchat.js'
import type { ChannelFactoryRegistry } from '../../server/runtime/channels.js'

export function registerChannelFactory(registry: ChannelFactoryRegistry): void {
  registry.register('webchat', () => ({
    channel: new WebchatChannel(),
  }))
}
