import type { SepilotdConfig } from '../../config/schema.js'
import {
  buildDiscovery,
  buildTelemetry,
  buildUpdater,
} from './services.js'

export interface IntegrationLayer {
  telemetry: Awaited<ReturnType<typeof buildTelemetry>>
  mdns: ReturnType<typeof buildDiscovery>
  updater: ReturnType<typeof buildUpdater>
}

export async function assembleIntegrationLayer(args: {
  config: SepilotdConfig
}): Promise<IntegrationLayer> {
  const telemetry = await buildTelemetry(args.config)
  const mdns = buildDiscovery(args.config)
  const updater = buildUpdater()

  return { telemetry, mdns, updater }
}
