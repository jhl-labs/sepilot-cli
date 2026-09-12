import type { FastifyInstance } from 'fastify'
import { bindCapability } from '../capabilities/bind.js'
import {
  configSchema,
  hasLegacyUnsafeProxyMigration,
  type SepilotdConfig,
} from '../../config/schema.js'
import { validateConfig } from '../../config/validator.js'
import {
  configForPersistence,
  isRuntimeManagedChannel,
  preserveRuntimeManagedChannels,
} from '../../config/runtime-channel-env.js'
import { readConfigYaml, writeConfigYamlAtomic } from '../../config/yaml-store.js'
import { applyLoggingConfig } from '../../observability/apply-logging-config.js'
import {
  LEGACY_PROXY_UNSAFE_DEGRADED_REASON,
  planProviderDispatcher,
  ProviderNetworkConfigurationError,
} from '../../providers/http-timeout.js'
import {
  ConfigRevisionConflictError,
  reconfigureRuntimeAutonomy,
  reconfigureRuntimeChannelType,
  reconfigureRuntimeExtensions,
} from '../runtime/config-runtime.js'
import {
  cloneRuntimeConfigSnapshot,
  isSameRuntimeNetworkConfigSnapshot,
} from '../runtime/config-snapshot.js'
import { openJsonWatch } from './watch-sse.js'

type SettingsJsonWatchPayload =
  | { type: 'snapshot'; document: Record<string, unknown> }
  | { type: 'heartbeat'; timestamp: string }

function replaceConfig(target: SepilotdConfig, next: SepilotdConfig): void {
  for (const key of Object.keys(target) as Array<keyof SepilotdConfig>) {
    delete target[key]
  }
  Object.assign(target, next)
}

function buildUpdatedKeys(
  previous: SepilotdConfig,
  next: SepilotdConfig,
): Set<
  | 'network'
  | 'providers'
  | 'agent.defaultProvider'
  | 'agent.defaultModel'
  | 'agent.disabledTools'
  | 'hooks.outboundWebhooks'
  | 'mcp.servers'
  | 'mcp.client'
> {
  const updatedKeys = new Set<
    | 'network'
    | 'providers'
    | 'agent.defaultProvider'
    | 'agent.defaultModel'
    | 'agent.disabledTools'
    | 'hooks.outboundWebhooks'
    | 'mcp.servers'
    | 'mcp.client'
  >()

  if (!isSameRuntimeNetworkConfigSnapshot(previous.network, next.network)) {
    updatedKeys.add('network')
  }
  if (JSON.stringify(previous.providers) !== JSON.stringify(next.providers)) {
    updatedKeys.add('providers')
  }
  if (previous.agent.defaultProvider !== next.agent.defaultProvider) {
    updatedKeys.add('agent.defaultProvider')
  }
  if (previous.agent.defaultModel !== next.agent.defaultModel) {
    updatedKeys.add('agent.defaultModel')
  }
  if (JSON.stringify(previous.agent.disabledTools) !== JSON.stringify(next.agent.disabledTools)) {
    updatedKeys.add('agent.disabledTools')
  }
  if (
    JSON.stringify(previous.hooks?.outboundWebhooks ?? []) !==
    JSON.stringify(next.hooks?.outboundWebhooks ?? [])
  ) {
    updatedKeys.add('hooks.outboundWebhooks')
  }
  if (JSON.stringify(previous.mcp?.servers ?? []) !== JSON.stringify(next.mcp?.servers ?? [])) {
    updatedKeys.add('mcp.servers')
  }
  if (JSON.stringify(previous.mcp?.client ?? {}) !== JSON.stringify(next.mcp?.client ?? {})) {
    updatedKeys.add('mcp.client')
  }

  return updatedKeys
}

function collectChannelTypesForRebind(
  previous: SepilotdConfig,
  next: SepilotdConfig,
  runtimeChannels: Array<{ type: string }> = [],
): string[] {
  if (JSON.stringify(previous.channels ?? []) === JSON.stringify(next.channels ?? [])) {
    return []
  }

  const channelTypes = new Set<string>()
  for (const channel of previous.channels ?? []) {
    channelTypes.add(channel.type)
  }
  for (const channel of next.channels ?? []) {
    channelTypes.add(channel.type)
  }
  for (const channel of runtimeChannels) {
    channelTypes.add(channel.type)
  }

  return Array.from(channelTypes).sort()
}

export async function registerSettingsJsonRoutes(app: FastifyInstance): Promise<void> {
  const watchSubscribers = new Set<(payload: SettingsJsonWatchPayload) => void>()

  function readSettingsJsonDocument(): Record<string, unknown> {
    return app.runtime
      ? (configForPersistence(app.runtime.config) as Record<string, unknown>)
      : readConfigYaml()
  }

  function buildWatchSnapshot(): SettingsJsonWatchPayload {
    return {
      type: 'snapshot',
      document: readSettingsJsonDocument(),
    }
  }

  function publishWatchSnapshot(): void {
    if (watchSubscribers.size === 0) return
    const payload = buildWatchSnapshot()
    for (const subscriber of watchSubscribers) subscriber(payload)
  }

  await bindCapability(
    app,
    {
      name: 'settings-json',
      version: '1',
      methods: [
        { method: 'GET', path: '/settings/json' },
        { method: 'GET', path: '/settings/json/watch' },
        { method: 'PUT', path: '/settings/json' },
      ],
    },
    async (a) => {
      a.get('/settings/json', async () => readSettingsJsonDocument())
      a.get('/settings/json/watch', async (req, reply) => {
        openJsonWatch(req, reply, {
          eventName: 'settings-json',
          subscribers: watchSubscribers,
          buildSnapshot: buildWatchSnapshot,
          buildHeartbeat: () => ({ type: 'heartbeat' as const, timestamp: new Date().toISOString() }),
        })
      })
      a.put('/settings/json', async (req, reply) => {
        const runtime = app.runtime
        if (runtime?.configLoadFailed) {
          return reply.status(409).send({
            code: 'CONFIG_LOAD_FAILED',
            message: 'Config was loaded in degraded read-only mode; refusing to overwrite config.yaml until the startup config error is fixed.',
            details: runtime.configLoadError,
            retriable: false,
          })
        }

        const parsed = configSchema.safeParse(req.body)
        if (!parsed.success) {
          return reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
        }

        if (
          runtime
          && isRuntimeManagedChannel(runtime.config, 'mattermost')
          && parsed.data.channels.some((channel) => channel.type === 'mattermost')
        ) {
          return reply.status(409).send({
            code: 'CHANNEL_MANAGED_BY_ENVIRONMENT',
            message:
              'Mattermost is managed by the daemon runtime environment. Update the ExternalSecret/runtime environment and restart the daemon instead.',
            retriable: false,
          })
        }

        if (hasLegacyUnsafeProxyMigration(parsed.data.network)) {
          return reply.status(400).send({
            code: LEGACY_PROXY_UNSAFE_DEGRADED_REASON,
            message: 'Choose and save an explicit environment, direct, or manual proxy mode.',
            retriable: false,
          })
        }

        const validation = validateConfig(parsed.data)
        if (!validation.valid) {
          return reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: validation.errors.join(' '),
            retriable: false,
          })
        }

        try {
          // Validate paths and environment-derived network inputs before any
          // in-memory mutation or config.yaml write. This keeps settings JSON
          // imports consistent with the dedicated Network endpoint.
          planProviderDispatcher(parsed.data.network)
        } catch (error) {
          if (error instanceof ProviderNetworkConfigurationError) {
            return reply.status(400).send({
              code: error.code,
              message: error.message,
              retriable: false,
            })
          }
          throw error
        }

        if (!runtime) {
          await writeConfigYamlAtomic(parsed.data as Record<string, unknown>)
          publishWatchSnapshot()
          return parsed.data
        }

        let savedDocument: SepilotdConfig = parsed.data
        await runtime.configMutationService.apply('settings-json.replace', async () => {
          const previousConfig = cloneRuntimeConfigSnapshot(runtime.config)
          const currentRevision = previousConfig.configRevision ?? 0
          const incomingRevision = parsed.data.configRevision

          if (typeof incomingRevision === 'number' && incomingRevision !== currentRevision) {
            throw new ConfigRevisionConflictError(
              `Config revision conflict: expected ${currentRevision}, received ${incomingRevision}`,
              currentRevision,
              incomingRevision,
            )
          }

          const nextConfig = preserveRuntimeManagedChannels(
            previousConfig,
            cloneRuntimeConfigSnapshot(parsed.data),
          )
          nextConfig.configRevision = currentRevision + 1

          const updatedKeys = buildUpdatedKeys(previousConfig, nextConfig)
          const channelTypes = collectChannelTypesForRebind(
            previousConfig,
            nextConfig,
            runtime.channels,
          )

          const loggingChanged =
            JSON.stringify(previousConfig.logging) !== JSON.stringify(nextConfig.logging)

          replaceConfig(runtime.config, nextConfig)
          runtime.configMutationService.configure({
            deviceName: runtime.config.device.name,
          })
          reconfigureRuntimeAutonomy(runtime)
          if (loggingChanged) {
            applyLoggingConfig(runtime.config.logging, {
              envLogLevel: process.env.SEPILOTD_LOG_LEVEL,
            })
          }
          if (updatedKeys.size > 0) {
            await reconfigureRuntimeExtensions(runtime, updatedKeys)
          }
          for (const channelType of channelTypes) {
            await reconfigureRuntimeChannelType(runtime, channelType)
          }

          const rawDocument = readConfigYaml()
          const rawChannels = Array.isArray(rawDocument.channels)
            ? rawDocument.channels
            : undefined
          const persistedConfig = configForPersistence(
            runtime.config,
            rawChannels,
          )
          await writeConfigYamlAtomic(persistedConfig as Record<string, unknown>)
          savedDocument = persistedConfig
        })

        publishWatchSnapshot()
        return savedDocument
      })
    },
  )
}
