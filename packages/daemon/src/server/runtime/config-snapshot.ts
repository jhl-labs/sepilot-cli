import {
  copyNetworkConfigRuntimeMetadata,
  hasLegacyUnsafeProxyMigration,
  type SepilotdConfig,
} from '../../config/schema.js'
import { copyRuntimeChannelEnvironmentMetadata } from '../../config/runtime-channel-env.js'

export function cloneRuntimeConfigSnapshot(config: SepilotdConfig): SepilotdConfig {
  const snapshot = structuredClone(config) as SepilotdConfig
  copyNetworkConfigRuntimeMetadata(config.network, snapshot.network)
  copyRuntimeChannelEnvironmentMetadata(config, snapshot)
  return snapshot
}

export function isSameRuntimeNetworkConfigSnapshot(
  current: SepilotdConfig['network'] | undefined,
  next: SepilotdConfig['network'] | undefined,
): boolean {
  return (
    hasLegacyUnsafeProxyMigration(current)
      === hasLegacyUnsafeProxyMigration(next)
    && JSON.stringify(current) === JSON.stringify(next)
  )
}

function comparableRuntimeConfig(config: SepilotdConfig): string {
  const snapshot = cloneRuntimeConfigSnapshot(config)
  delete snapshot.configRevision
  return JSON.stringify(snapshot)
}

export function isSameRuntimeConfigSnapshot(
  current: SepilotdConfig | undefined,
  next: SepilotdConfig,
): boolean {
  if (!current) return false
  if (!isSameRuntimeNetworkConfigSnapshot(current.network, next.network)) {
    return false
  }
  return JSON.stringify(current) === JSON.stringify(next)
    || comparableRuntimeConfig(current) === comparableRuntimeConfig(next)
}
