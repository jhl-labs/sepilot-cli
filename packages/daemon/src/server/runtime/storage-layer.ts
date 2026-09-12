import type { SepilotdConfig } from '../../config/schema.js'
import { basename, join } from 'node:path'
import { createLogger } from '../../logger.js'
import { SqliteSemanticIndex } from '../../memory/semantic-index.js'
import { recoverSqliteDatabaseIfCorrupt } from '../../memory/sqlite-recovery.js'
import type { SemanticMemoryStore } from '../../memory/types.js'
import { rebindableSemanticStore } from '../../memory/rebindable-store.js'
import { UsageTracker } from '../../memory/usage-tracker.js'
import {
  buildChannelPipelineMonitor,
  buildChannelOriginStore,
  buildChannelReplayStore,
  buildChannelSessionStore,
  buildDevicePairingRegistry,
  buildFileMemory,
  buildSemanticIndex,
  buildSessionStore,
  buildUsageTracker,
} from './storage.js'
import type { buildEncryption } from './security.js'
import type { buildProviderRegistry } from './providers.js'
import { SessionWatchBroker, WatchedSessionStore } from './session-watch.js'

type EncryptionInstance = Awaited<ReturnType<typeof buildEncryption>>
type ProviderRegistryInstance = ReturnType<typeof buildProviderRegistry>

const log = createLogger('storage-layer')

export interface StorageDegradedStore {
  store: 'semanticIndex' | 'usageTracker'
  error: string
}

export interface StorageLayer {
  sessions: WatchedSessionStore
  sessionWatchBroker: SessionWatchBroker
  fileMemory: Awaited<ReturnType<typeof buildFileMemory>>
  semanticIndex: Awaited<ReturnType<typeof buildSemanticIndex>>
  usageTracker: Awaited<ReturnType<typeof buildUsageTracker>>
  channelPipelineMonitor: Awaited<ReturnType<typeof buildChannelPipelineMonitor>>
  channelOriginStore: Awaited<ReturnType<typeof buildChannelOriginStore>>
  devicePairingRegistry: Awaited<ReturnType<typeof buildDevicePairingRegistry>>
  channelReplayStore: Awaited<ReturnType<typeof buildChannelReplayStore>>
  channelSessionStore: Awaited<ReturnType<typeof buildChannelSessionStore>>
  degradedStores: StorageDegradedStore[]
}

/**
 * Build the per-user session / memory / channel replay / usage
 * stores that sit on the local filesystem under dataDir. Needs
 * encryption (for session-store at-rest crypto) and
 * providerRegistry (for semantic-index embedding lookups) from the
 * security + agent layers.
 */
export async function assembleStorageLayer(args: {
  config: SepilotdConfig
  dataDir: string
  encryption: EncryptionInstance
  providerRegistry: ProviderRegistryInstance
}): Promise<StorageLayer> {
  const { config, dataDir, encryption, providerRegistry } = args
  const degradedStores: StorageDegradedStore[] = []
  const semanticDbPath = join(dataDir, 'memory', 'local.db')
  const usageDbPath = join(dataDir, 'memory', 'usage.db')

  const semanticQuarantine = recoverSqliteDatabaseIfCorrupt(semanticDbPath)
  if (semanticQuarantine.length > 0) {
    degradedStores.push({
      store: 'semanticIndex',
      error: `corrupt database quarantined: ${semanticQuarantine.map((path) => basename(path)).join(', ')}`,
    })
  }

  const usageQuarantine = recoverSqliteDatabaseIfCorrupt(usageDbPath)
  if (usageQuarantine.length > 0) {
    degradedStores.push({
      store: 'usageTracker',
      error: `corrupt database quarantined: ${usageQuarantine.map((path) => basename(path)).join(', ')}`,
    })
  }

  const buildOptional = async <T>(
    store: StorageDegradedStore['store'],
    build: () => Promise<T>,
    fallback: () => Promise<T>,
  ): Promise<T> => {
    try {
      return await build()
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      degradedStores.push({ store, error: message })
      log.error('Non-essential storage store failed; using fallback', {
        store,
        error: message,
      })
      return fallback()
    }
  }

  const [
    rawSessions,
    fileMemory,
    semanticIndex,
    usageTracker,
    channelPipelineMonitor,
    channelOriginStore,
    devicePairingRegistry,
    channelReplayStore,
    channelSessionStore,
  ] = await Promise.all([
    buildSessionStore(config, dataDir, encryption),
    buildFileMemory(dataDir),
    buildOptional(
      'semanticIndex',
      () => buildSemanticIndex(config, dataDir, providerRegistry),
      () => SqliteSemanticIndex.create(':memory:') as Promise<SemanticMemoryStore>,
    ),
    buildOptional(
      'usageTracker',
      () => buildUsageTracker(dataDir),
      () => UsageTracker.create(':memory:'),
    ),
    buildChannelPipelineMonitor(dataDir),
    buildChannelOriginStore(dataDir),
    buildDevicePairingRegistry(dataDir),
    buildChannelReplayStore(dataDir),
    buildChannelSessionStore(dataDir),
  ])

  const sessionWatchBroker = new SessionWatchBroker()
  const sessions = new WatchedSessionStore(rawSessions, sessionWatchBroker)

  return {
    sessions,
    sessionWatchBroker,
    fileMemory,
    semanticIndex: rebindableSemanticStore(semanticIndex),
    usageTracker,
    channelPipelineMonitor,
    channelOriginStore,
    devicePairingRegistry,
    channelReplayStore,
    channelSessionStore,
    degradedStores,
  }
}
