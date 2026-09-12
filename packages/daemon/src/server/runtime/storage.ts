import type { ISessionStore } from '@sepilotd/core'
import type { SepilotdConfig } from '../../config/schema.js'
import { EncryptedSessionStore } from '../../memory/encrypted-session-store.js'
import { FileMemory } from '../../memory/file-memory.js'
import { JsonlSessionStore } from '../../memory/session-store.js'
import { SqliteSemanticIndex } from '../../memory/semantic-index.js'
import type { SemanticMemoryStore } from '../../memory/types.js'
import { UsageTracker } from '../../memory/usage-tracker.js'
import type { ProviderRegistry } from '../../providers/registry.js'
import { JsonlAuditLogger } from '../../security/audit-logger.js'
import type { EncryptionManager } from '../../security/encryption.js'
import { seedBuiltinSkills } from '../../skills/builtin.js'
import { defaultCompatibleSkillRoots } from '../../skills/compat.js'
import { FileSkillRegistry } from '../../skills/registry.js'
import { SkillStore } from '../../skills/store.js'
import { MarketplaceCatalog } from '../../skills/marketplace-catalog.js'
import { ChannelPipelineMonitor } from './channel-pipeline-monitor.js'
import { ChannelPipelineMonitorStore } from './channel-pipeline-monitor-store.js'
import { DevicePairingRegistry } from './device-pairing.js'
import { ChannelReplayStore } from './channel-replays.js'
import { ChannelSessionStore } from './channel-sessions.js'
import { ChannelOriginStore } from '../../channels/channel-origin-store.js'

export async function buildSessionStore(
  config: SepilotdConfig,
  dataDir: string,
  encryption: EncryptionManager,
): Promise<ISessionStore> {
  const sessions: ISessionStore =
    config.memory.encryption && encryption.isEnabled()
      ? new EncryptedSessionStore(`${dataDir}/sessions`, encryption)
      : new JsonlSessionStore(`${dataDir}/sessions`)

  await sessions.init?.()
  return sessions
}

export async function buildAuditLogger(
  dataDir: string,
): Promise<JsonlAuditLogger> {
  const auditLogger = new JsonlAuditLogger(`${dataDir}/security/audit.log`)
  await auditLogger.init()
  return auditLogger
}

export async function buildSkillRegistry(
  dataDir: string,
  options: { trustProjectSkills?: boolean } = {},
): Promise<FileSkillRegistry> {
  const skillRegistry = new FileSkillRegistry(`${dataDir}/skills`, {
    additionalRoots: await defaultCompatibleSkillRoots(),
    trustProjectSkills: options.trustProjectSkills ?? false,
  })
  await skillRegistry.init()
  await seedBuiltinSkills(skillRegistry)
  return skillRegistry
}

export async function buildFileMemory(
  dataDir: string,
): Promise<FileMemory> {
  const fileMemory = new FileMemory(`${dataDir}/memory`)
  await fileMemory.init()
  return fileMemory
}

export async function buildSemanticIndex(
  config: SepilotdConfig,
  dataDir: string,
  providerRegistry: ProviderRegistry,
): Promise<SemanticMemoryStore> {
  const embeddingProvider = config.memory.embeddingProvider
  const embeddingModel = config.memory.embeddingModel
  const provider = embeddingProvider ? providerRegistry.get(embeddingProvider) : undefined
  const semanticIndex = await SqliteSemanticIndex.create(
    `${dataDir}/memory/local.db`,
    {
      embeddingModel,
      configuredProviderId: embeddingProvider,
      vectorBackend: config.memory.vectorBackend,
      qdrant: config.memory.qdrant,
      opensearch: config.memory.opensearch,
      elasticsearch: config.memory.elasticsearch,
      meilisearch: config.memory.meilisearch,
      customApi: config.memory.customApi,
      embedder: provider?.embed
        ? {
          providerId: embeddingProvider!,
          embed: (texts, model) => provider.embed!(texts, model),
        }
        : undefined,
    },
  )

  semanticIndex.startBackgroundBackfill()

  return semanticIndex
}

export function buildUsageTracker(dataDir: string): Promise<UsageTracker> {
  return UsageTracker.create(`${dataDir}/memory/usage.db`)
}

export function buildSkillStore(dataDir: string): Promise<SkillStore> {
  return SkillStore.create(`${dataDir}/skills/store.db`)
}

export async function buildChannelPipelineMonitor(
  dataDir: string,
): Promise<ChannelPipelineMonitor> {
  const store = new ChannelPipelineMonitorStore(
    `${dataDir}/channel-pipeline/monitor.json`,
  )
  const monitor = new ChannelPipelineMonitor(store)
  await monitor.init()
  return monitor
}

export async function buildChannelReplayStore(
  dataDir: string,
): Promise<ChannelReplayStore> {
  const replayStore = new ChannelReplayStore(`${dataDir}/channel-replays`)
  await replayStore.init()
  await replayStore.pruneExpired()
  return replayStore
}

export async function buildChannelSessionStore(
  dataDir: string,
): Promise<ChannelSessionStore> {
  const sessionStore = new ChannelSessionStore(`${dataDir}/channel-sessions`)
  await sessionStore.init()
  await sessionStore.pruneStale()
  return sessionStore
}

export async function buildChannelOriginStore(
  dataDir: string,
): Promise<ChannelOriginStore> {
  const originStore = new ChannelOriginStore(`${dataDir}/channel-origins`)
  await originStore.init()
  await originStore.pruneOlderThan(24 * 60 * 60 * 1000)
  return originStore
}

export async function buildDevicePairingRegistry(
  dataDir: string,
): Promise<DevicePairingRegistry> {
  const registry = new DevicePairingRegistry(`${dataDir}/security/paired-devices.json`)
  await registry.init()
  return registry
}

export async function buildMarketplaceCatalog(dataDir: string): Promise<MarketplaceCatalog> {
  const catalog = new MarketplaceCatalog(`${dataDir}/skills`)
  await catalog.init()
  return catalog
}
