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

/**
 * Passive integrations: OpenTelemetry exporter, mDNS discovery,
 * auto-updater. Construction is pure; all three are activated
 * later by startRuntime().
 *
 * The SQLite-backed SchedulerStack is built in startRuntime() instead
 * (lifecycle.ts), where all deps — channels, ChannelPipelineCapabilities,
 * providerRegistry, etc. — are fully assembled.
 */
export async function assembleIntegrationLayer(args: {
  config: SepilotdConfig
}): Promise<IntegrationLayer> {
  const telemetry = await buildTelemetry(args.config)
  const mdns = buildDiscovery(args.config)
  const updater = buildUpdater()

  return { telemetry, mdns, updater }
}
