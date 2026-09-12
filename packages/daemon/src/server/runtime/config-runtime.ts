import { readFile, rename, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import YAML from 'yaml'
import type { SepilotdConfig } from '../../config/schema.js'
import {
  channelsForPersistence,
  configForPersistence,
} from '../../config/runtime-channel-env.js'
import { replaceOutboundWebhooks } from '../../hooks/outbound-webhook.js'
import { replaceCommandHooks } from '../../hooks/command-hook.js'
import { buildChannels } from './channels.js'
import { buildChannelAcl } from './security.js'
import {
  clearTelegramPendingPairing,
  createPersistingChannelUserPairHandler,
  readTelegramPendingPairing,
} from './channel-pairing-persistence.js'
import {
  applyConfigUpdate,
  getConfigUpdateValue,
  type ConfigUpdateKey,
  type ConfigUpdateValues,
} from './config-mutations.js'
import { buildProviderRegistry, resolveAutonomy, syncDynamicProviderModels } from './providers.js'
import { buildModelRouter, syncDreamingProvider } from './services.js'
import { buildSemanticIndex } from './storage.js'
import { replaceSemanticStore } from '../../memory/rebindable-store.js'
import {
  configureProviderHttpTimeout,
  configureProviderNetworkBlocked,
  CUSTOM_CA_UNAVAILABLE_DEGRADED_REASON,
  getProviderDispatcherPlan,
  NETWORK_PROXY_UNAVAILABLE_DEGRADED_REASON,
  ProviderNetworkConfigurationError,
} from '../../providers/http-timeout.js'
import type {
  ChannelRebindCapabilities,
  ConfigAutonomyCapabilities,
  ConfigPersistenceCapabilities,
  ConfigRebindCapabilities,
} from './capabilities.js'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export class ConfigRevisionConflictError extends Error {
  readonly code = 'CONFIG_REVISION_CONFLICT' as const
  constructor(
    message: string,
    readonly expectedRevision: number,
    readonly diskRevision: number,
  ) {
    super(message)
    this.name = 'ConfigRevisionConflictError'
  }
}

/**
 * Thrown from inside a config-mutation transaction when the post-mutation
 * config fails cross-field validation (validateConfig). Throwing lets the
 * mutation service roll back the in-memory config and restore extensions,
 * so an incoherent config is never persisted and never bricks the next boot.
 */
export class ConfigValidationError extends Error {
  readonly code = 'CONFIG_VALIDATION_FAILED' as const
  constructor(
    message: string,
    readonly errors: string[],
  ) {
    super(message)
    this.name = 'ConfigValidationError'
  }
}

export async function persistRuntimeConfig(
  runtime: ConfigPersistenceCapabilities | undefined,
  updatedKeys: Set<ConfigUpdateKey>,
  updatedValues: ConfigUpdateValues = {},
): Promise<void> {
  if (!runtime || typeof runtime.dataDir !== 'string') {
    return
  }

  const currentRevision = runtime.config.configRevision ?? 0
  const nextRevision = currentRevision + 1

  const configPath = join(runtime.dataDir, 'config.yaml')
  const tempPath = `${configPath}.tmp`
  let persistedConfig: SepilotdConfig = runtime.config

  try {
    const rawConfig = YAML.parse(await readFile(configPath, 'utf-8'))
    if (isRecord(rawConfig)) {
      persistedConfig = rawConfig as SepilotdConfig

      // CAS: verify on-disk revision matches what we expect.
      // If an external writer (editor, another daemon, hot-reload) changed
      // config.yaml after we last read it, the revision will differ and
      // we reject the write to avoid clobbering.
      const diskRevision = persistedConfig.configRevision ?? 0
      if (diskRevision !== currentRevision) {
        throw new ConfigRevisionConflictError(
          `Config revision conflict: expected ${currentRevision} on disk, found ${diskRevision}`,
          currentRevision,
          diskRevision,
        )
      }

      for (const key of updatedKeys) {
        const value = Object.hasOwn(updatedValues, key)
          ? updatedValues[key]
          : getConfigUpdateValue(runtime.config, key)
        if (key === 'channels' && Array.isArray(value)) {
          // The raw environment-owned entry may intentionally contain
          // `${VAR}` placeholders that are not a valid URL until startup
          // substitution. It has already been accepted on canonical load,
          // while every non-managed mutation was validated by its route.
          // Assign this merged persistence form directly instead of parsing
          // the unresolved raw entry as a fresh API payload.
          persistedConfig.channels = channelsForPersistence(
            runtime.config,
            persistedConfig.channels,
            value as SepilotdConfig['channels'],
          )
          continue
        }
        if (value !== undefined) {
          applyConfigUpdate(persistedConfig, key, structuredClone(value))
        }
      }
    }
  } catch (err) {
    if (err instanceof ConfigRevisionConflictError) throw err
    persistedConfig = configForPersistence(runtime.config)
  }

  // The recovery path owns a detached, secret-filtered snapshot. Keep its
  // persisted revision in lockstep with the in-memory CAS revision just as in
  // the ordinary read-modify-write path.
  persistedConfig.configRevision = nextRevision

  // Bump in-memory revision only after CAS passes or file was unreadable.
  // If CAS failed, we already threw above — the in-memory revision is
  // untouched so the caller can retry.
  runtime.config.configRevision = nextRevision

  const yaml = YAML.stringify(persistedConfig)
  await writeFile(tempPath, yaml, { mode: 0o600 })
  await rename(tempPath, configPath)
}

export async function reconfigureRuntimeProviders(
  runtime: ConfigRebindCapabilities | undefined,
): Promise<void> {
  if (!runtime) {
    return
  }

  runtime.providerRegistry = buildProviderRegistry(
    runtime.config,
    runtime.providerFactoryRegistry,
  )
  await syncDynamicProviderModels(runtime.providerRegistry, runtime.config)
  runtime.modelRouter = buildModelRouter(runtime.providerRegistry)
  syncDreamingProvider(runtime.dreaming, runtime.providerRegistry, runtime.config.agent)
}

const SEMANTIC_INDEX_CONFIG_KEYS = new Set<ConfigUpdateKey>([
  'providers',
  'memory.vectorBackend',
  'memory.embeddingProvider',
  'memory.embeddingModel',
  'memory.qdrant',
  'memory.opensearch',
  'memory.elasticsearch',
  'memory.meilisearch',
  'memory.customApi',
])

async function reconfigureRuntimeSemanticIndex(
  runtime: ConfigRebindCapabilities | undefined,
): Promise<void> {
  if (!runtime) {
    return
  }
  const previous = runtime.semanticIndex
  const replacement = await buildSemanticIndex(
    runtime.config,
    runtime.dataDir,
    runtime.providerRegistry,
  )
  runtime.semanticIndex = await replaceSemanticStore(previous, replacement)
}

export function reconfigureRuntimeAutonomy(
  runtime: ConfigAutonomyCapabilities | undefined,
): void {
  if (!runtime) {
    return
  }

  runtime.autonomy = resolveAutonomy(runtime.config)
  runtime.channelAcl = buildChannelAcl(runtime.config, runtime.autonomy)
}

/**
 * Restore the previous network policy during a config transaction rollback.
 * A stale CA in the old snapshot must fail closed without preventing the
 * provider/MCP/channel portions of the rollback from continuing.
 */
export function restoreRuntimeNetworkPolicy(
  runtime: ConfigRebindCapabilities | undefined,
): string | null {
  if (!runtime) return null
  try {
    configureProviderHttpTimeout(runtime.config.network)
    return getProviderDispatcherPlan()?.degradedReason ?? null
  } catch (error) {
    if (!(error instanceof ProviderNetworkConfigurationError)) {
      throw error
    }
    const degradedReason = error.code === 'NETWORK_CA_INVALID'
      ? CUSTOM_CA_UNAVAILABLE_DEGRADED_REASON
      : NETWORK_PROXY_UNAVAILABLE_DEGRADED_REASON
    configureProviderNetworkBlocked(
      runtime.config.network,
      degradedReason,
    )
    return degradedReason
  }
}

export function reconfigureRuntimeSkillSources(
  runtime: ConfigRebindCapabilities | undefined,
): void {
  if (!runtime) {
    return
  }

  runtime.skillSourceUrlPolicy.update(runtime.config.security.skillSources)
}

export async function reconfigureRuntimeExtensions(
  runtime: ConfigRebindCapabilities | undefined,
  updatedKeys: Set<ConfigUpdateKey>,
  options: { deferMcpManifestPersistence?: boolean } = {},
): Promise<void> {
  if (!runtime) {
    return
  }

  if (updatedKeys.has('agent.disabledTools')) {
    runtime.toolRegistry.setDisabledTools(runtime.config.agent.disabledTools)
  }

  if (updatedKeys.has('network')) {
    configureProviderHttpTimeout(runtime.config.network)
  }

  if (
    updatedKeys.has('providers')
    || updatedKeys.has('agent.defaultProvider')
    || updatedKeys.has('agent.defaultModel')
    || updatedKeys.has('agent.auxModel')
  ) {
    await reconfigureRuntimeProviders(runtime)
  }

  if ([...updatedKeys].some((key) => SEMANTIC_INDEX_CONFIG_KEYS.has(key))) {
    await reconfigureRuntimeSemanticIndex(runtime)
  }

  if (updatedKeys.has('hooks.outboundWebhooks')) {
    replaceOutboundWebhooks(
      runtime.hookRegistry,
      runtime.config.hooks?.outboundWebhooks ?? [],
      {
        auditLogger: runtime.auditLogger,
        deviceName: runtime.config.device.name,
      },
    )
  }

  if (updatedKeys.has('hooks.commandHooks')) {
    replaceCommandHooks(runtime.hookRegistry, runtime.config.hooks?.commandHooks ?? [])
  }

  if (updatedKeys.has('mcp.servers') || updatedKeys.has('mcp.client')) {
    await runtime.mcpManager.configureServers(
      runtime.config.mcp?.servers,
      runtime.toolRegistry,
      runtime.mcpPromptsRegistry,
      runtime.config.mcp?.client,
      {
        deferInitialToolManifestPersistence:
          options.deferMcpManifestPersistence === true,
      },
    )
  }

  if (updatedKeys.has('security.skillSources')) {
    reconfigureRuntimeSkillSources(runtime)
  }
}

export async function applyAndPersistRuntimeUpdate(
  runtime: (
    ConfigAutonomyCapabilities
    & ConfigRebindCapabilities
    & ConfigPersistenceCapabilities
  ) | undefined,
  updatedKeys: Set<ConfigUpdateKey>,
  updatedValues: ConfigUpdateValues = {},
): Promise<void> {
  if (!runtime) {
    return
  }

  if (updatedKeys.has('agent.autonomy')) {
    reconfigureRuntimeAutonomy(runtime)
  }
  await reconfigureRuntimeExtensions(runtime, updatedKeys, {
    deferMcpManifestPersistence: updatedKeys.has('mcp.servers'),
  })
  if (updatedKeys.size > 0) {
    await persistRuntimeConfig(runtime, updatedKeys, updatedValues)
  }
}

export async function reconfigureRuntimeChannelType(
  runtime: ChannelRebindCapabilities | undefined,
  channelType: string,
): Promise<void> {
  if (!runtime) {
    return
  }

  const previousChannels = runtime.channels.filter(
    (channel) => channel.type === channelType,
  )
  const nextChannels = buildChannels(
    {
      ...runtime.config,
      channels: runtime.config.channels.filter(
        (channel) => channel.type === channelType,
      ),
    },
    runtime.gatewayClient,
    runtime.channelFactoryRegistry,
    {
      onChannelUserPaired: createPersistingChannelUserPairHandler({
        getConfig: () => runtime.config,
        dataDir: runtime.dataDir,
        addAllowedUser: (pairedChannelType, userId) => {
          runtime.channelAcl.addAllowedUser(pairedChannelType, userId)
        },
        applyConfigMutation: (description, fn) =>
          runtime.configMutationService.apply(description, fn),
        persistConfig: async () => {
          await persistRuntimeConfig(runtime, new Set(['channels']))
        },
      }),
      loadTelegramPendingPairing: () =>
        readTelegramPendingPairing(runtime.dataDir),
      clearTelegramPendingPairing: () =>
        clearTelegramPendingPairing(runtime.dataDir),
    },
  ).filter((channel) => channel.type === channelType)

  for (const channel of previousChannels) {
    runtime.channelRouter.unwireChannel(channel)
  }
  for (const channel of previousChannels) {
    await channel.stop().catch(() => undefined)
  }

  runtime.channels = runtime.channels.filter(
    (channel) => channel.type !== channelType,
  )

  for (const channel of nextChannels) {
    runtime.channelRouter.wireChannel(channel)
    runtime.channels.push(channel)
    await channel.start().catch(() => undefined)
  }
}
