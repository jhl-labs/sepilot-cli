import {
  DAEMON_CONFIG_UPDATE_KEYS,
  isDaemonConfigUpdateKey,
  type DaemonConfigUpdateKey,
} from '@sepilotd/api-client'
import {
  channelPipelineConfigSchema,
  channelPipelineHealthConfigSchema,
  commandHookSchema,
  agentGraphNodeModelOverridesSchema,
  memoryMaintenanceSchema,
  mcpClientConfigSchema,
  networkConfigSchema,
  schedulerConfigSchema,
  schedulerSurfaceAccessSchema,
  skillSourceSecuritySchema,
  type SepilotdConfig,
  webhookSecurityHealthConfigSchema,
  webhookSecurityPolicyConfigSchema,
  webSearchProviderUpdateSchema,
  webSearchTrustedDomainsSchema,
} from '../../config/schema.js'
import { configChannelsUpdateSchema } from '../../config/channel-update-schema.js'
import { restoreChannelConfigSecrets } from '../routes/config-channels-internals.js'
import { outboundWebhookIdFromUrl } from '../../hooks/outbound-webhook.js'

export type ConfigUpdateKey = DaemonConfigUpdateKey

export type ConfigUpdateValues = Partial<Record<ConfigUpdateKey, unknown>>

export { DAEMON_CONFIG_UPDATE_KEYS }

export function isConfigUpdateKey(value: string): value is ConfigUpdateKey {
  return isDaemonConfigUpdateKey(value)
}

export function restoreProviderApiKey(
  existingProviders: SepilotdConfig['providers'],
  provider: SepilotdConfig['providers'][number],
): SepilotdConfig['providers'][number] {
  const existingProvider = existingProviders.find((candidate) => candidate.id === provider.id)
  const credentialTargetUnchanged =
    existingProvider?.type === provider.type &&
    (existingProvider.baseUrl ?? '') === (provider.baseUrl ?? '')

  const apiKey =
    provider.apiKey !== '***redacted***'
      ? provider.apiKey
      : credentialTargetUnchanged
        ? existingProvider?.apiKey
        : undefined

  const nextHeaderEntries = Object.entries(provider.headers ?? {}).flatMap(([key, value]) => {
    if (value !== '***redacted***') {
      return [[key, value] as const]
    }

    const restoredValue = credentialTargetUnchanged ? existingProvider?.headers?.[key] : undefined
    return typeof restoredValue === 'string' ? [[key, restoredValue] as const] : []
  })

  if (
    provider.apiKey !== '***redacted***' &&
    nextHeaderEntries.length === Object.keys(provider.headers ?? {}).length
  ) {
    return provider
  }

  return {
    ...provider,
    apiKey,
    headers: Object.fromEntries(nextHeaderEntries),
  }
}

export function restoreProviderSecrets(
  existingProviders: SepilotdConfig['providers'],
  providers: SepilotdConfig['providers'],
): SepilotdConfig['providers'] {
  return providers.map((provider) => restoreProviderApiKey(existingProviders, provider))
}

export function restoreMcpServerEnv(
  existingServers: SepilotdConfig['mcp']['servers'],
  server: SepilotdConfig['mcp']['servers'][number],
): SepilotdConfig['mcp']['servers'][number] {
  const existingServer = existingServers.find((candidate) => candidate.name === server.name)

  const restoreRecord = (
    next: Record<string, string> | undefined,
    existing: Record<string, string> | undefined,
  ): Record<string, string> =>
    Object.fromEntries(
      Object.entries(next ?? {}).flatMap(([key, value]) => {
        if (value !== '***redacted***') {
          return [[key, value] as const]
        }
        const restoredValue = existing?.[key]
        return typeof restoredValue === 'string' ? [[key, restoredValue] as const] : []
      }),
    )

  if (server.transport === 'stdio') {
    const existingEnv =
      existingServer &&
      'env' in existingServer &&
      existingServer.env &&
      typeof existingServer.env === 'object' &&
      !Array.isArray(existingServer.env)
        ? existingServer.env
        : undefined

    return {
      ...server,
      env: restoreRecord(server.env, existingEnv),
    }
  }

  const existingHeaders =
    existingServer &&
    'headers' in existingServer &&
    existingServer.headers &&
    typeof existingServer.headers === 'object' &&
    !Array.isArray(existingServer.headers)
      ? existingServer.headers
      : undefined

  return {
    ...server,
    headers: restoreRecord(server.headers, existingHeaders),
  }
}

export function restoreMcpServerSecrets(
  existingServers: SepilotdConfig['mcp']['servers'],
  servers: SepilotdConfig['mcp']['servers'],
): SepilotdConfig['mcp']['servers'] {
  return servers.map((server) => restoreMcpServerEnv(existingServers, server))
}

function outboundWebhookIdentity(url: string): string {
  try {
    return outboundWebhookIdFromUrl(url)
  } catch {
    return url
  }
}

export function restoreOutboundWebhookSecrets(
  existingWebhooks: SepilotdConfig['hooks']['outboundWebhooks'],
  webhooks: SepilotdConfig['hooks']['outboundWebhooks'],
): SepilotdConfig['hooks']['outboundWebhooks'] {
  return webhooks.map((webhook) => {
    const webhookId = outboundWebhookIdentity(webhook.url)
    const existingWebhook = existingWebhooks.find(
      (candidate) => outboundWebhookIdentity(candidate.url) === webhookId,
    )
    const nextSecret =
      webhook.secret !== '***redacted***' ? webhook.secret : existingWebhook?.secret
    const nextHeaders = Object.fromEntries(
      Object.entries(webhook.headers ?? {}).flatMap(([key, value]) => {
        if (value !== '***redacted***') {
          return [[key, value] as const]
        }
        const restoredValue = existingWebhook?.headers?.[key]
        return typeof restoredValue === 'string' ? [[key, restoredValue] as const] : []
      }),
    )

    return {
      ...webhook,
      ...(nextSecret ? { secret: nextSecret } : { secret: undefined }),
      headers: nextHeaders,
    }
  })
}

function restoreRedactedString(
  next: string | undefined,
  existing: string | undefined,
): string | undefined {
  return next === '***redacted***' ? existing : next
}

function restoreRecordSecrets(
  next: Record<string, string> | undefined,
  existing: Record<string, string> | undefined,
): Record<string, string> | undefined {
  if (!next) return next
  return Object.fromEntries(
    Object.entries(next).map(([key, value]) => [
      key,
      restoreRedactedString(value, existing?.[key]) ?? value,
    ]),
  )
}

function restoreMemoryAuthSecrets<T extends { auth?: { password?: string } }>(
  next: T,
  existing: T | undefined,
): T {
  if (!next.auth) return next
  return {
    ...next,
    auth: {
      ...next.auth,
      password: restoreRedactedString(next.auth.password, existing?.auth?.password),
    },
  }
}

function restoreMemorySearchEngineSecrets<T extends SepilotdConfig['memory']['opensearch']>(
  next: T,
  existing: T,
): T {
  if (!next) return next
  return {
    ...next,
    apiKey: restoreRedactedString(next.apiKey, existing?.apiKey),
    password: restoreRedactedString(next.password, existing?.password),
  } as T
}

function restoreMemoryConfigSecrets(
  existingMemory: SepilotdConfig['memory'],
  key: ConfigUpdateKey,
  value: unknown,
): unknown {
  if (key === 'memory.qdrant' && value && typeof value === 'object' && !Array.isArray(value)) {
    const next = value as SepilotdConfig['memory']['qdrant']
    return {
      ...next,
      apiKey: restoreRedactedString(next?.apiKey, existingMemory.qdrant?.apiKey),
    }
  }
  if (key === 'memory.opensearch' && value && typeof value === 'object' && !Array.isArray(value)) {
    return restoreMemorySearchEngineSecrets(
      value as SepilotdConfig['memory']['opensearch'],
      existingMemory.opensearch,
    )
  }
  if (
    key === 'memory.elasticsearch' &&
    value &&
    typeof value === 'object' &&
    !Array.isArray(value)
  ) {
    return restoreMemorySearchEngineSecrets(
      value as SepilotdConfig['memory']['elasticsearch'],
      existingMemory.elasticsearch,
    )
  }
  if (key === 'memory.meilisearch' && value && typeof value === 'object' && !Array.isArray(value)) {
    const next = value as SepilotdConfig['memory']['meilisearch']
    return {
      ...next,
      apiKey: restoreRedactedString(next?.apiKey, existingMemory.meilisearch?.apiKey),
    }
  }
  if (key === 'memory.customApi' && value && typeof value === 'object' && !Array.isArray(value)) {
    // Cast strips the `| undefined` from the schema type because we
    // already narrowed `value` non-null in the if-guard above.
    // Without the strip, restoreMemoryAuthSecrets sees its T as a
    // possibly-undefined object and rejects the spread.
    const next = value as NonNullable<SepilotdConfig['memory']['customApi']>
    return {
      ...restoreMemoryAuthSecrets(next, existingMemory.customApi),
      apiKey: restoreRedactedString(next.apiKey, existingMemory.customApi?.apiKey),
      headers: restoreRecordSecrets(next.headers, existingMemory.customApi?.headers),
    }
  }
  if (key === 'memory.rag' && value && typeof value === 'object' && !Array.isArray(value)) {
    const next = value as SepilotdConfig['memory']['rag']
    return {
      ...next,
      rerank: next?.rerank
        ? {
            ...restoreMemoryAuthSecrets(next.rerank, existingMemory.rag?.rerank),
            apiKey: restoreRedactedString(next.rerank.apiKey, existingMemory.rag?.rerank?.apiKey),
            headers: restoreRecordSecrets(next.rerank.headers, existingMemory.rag?.rerank?.headers),
          }
        : next?.rerank,
    }
  }
  return value
}

export function applyConfigUpdate(
  config: SepilotdConfig,
  key: ConfigUpdateKey,
  value: unknown,
): boolean {
  switch (key) {
    case 'agent.mode':
      if (typeof value === 'string' && value.trim().length > 0) {
        config.agent.mode = value.trim()
        return true
      }
      return false
    case 'agent.defaultModel':
      if (typeof value === 'string' && value.length > 0) {
        config.agent.defaultModel = value
        return true
      }
      return false
    case 'agent.defaultProvider':
      if (typeof value === 'string' && value.length > 0) {
        config.agent.defaultProvider = value
        return true
      }
      return false
    case 'agent.auxModel':
      if (typeof value === 'string' && value.trim().length > 0) {
        config.agent.auxModel = value.trim()
        return true
      }
      if (value === null || value === undefined || value === '') {
        delete config.agent.auxModel
        return true
      }
      return false
    case 'agent.autonomy':
      if (
        value === 'readonly' ||
        value === 'accept-edits' ||
        value === 'workspace-write' ||
        value === 'supervised' ||
        value === 'autonomous'
      ) {
        config.agent.autonomy = value
        return true
      }
      return false
    case 'agent.thinkingLevel':
      if (
        value === 'off' ||
        value === 'low' ||
        value === 'medium' ||
        value === 'high' ||
        value === 'max'
      ) {
        config.agent.thinkingLevel = value
        return true
      }
      return false
    case 'agent.disabledTools':
      if (
        Array.isArray(value)
        && value.every((name) => typeof name === 'string' && name.trim().length > 0)
      ) {
        config.agent.disabledTools = [...new Set(value.map((name) => name.trim()))].sort()
        return true
      }
      return false
    case 'agent.graphNodeModelOverrides': {
      const parsed = agentGraphNodeModelOverridesSchema.safeParse(value)
      if (!parsed.success) {
        return false
      }
      config.agent.graphNodeModelOverrides = parsed.data
      return true
    }
    case 'daemon.resumeArtifactRetentionDays':
      if (typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= 3650) {
        config.daemon.resumeArtifactRetentionDays = value
        return true
      }
      return false
    case 'device.name':
      if (typeof value === 'string' && value.length > 0) {
        config.device.name = value
        return true
      }
      return false
    case 'network': {
      const parsed = networkConfigSchema.safeParse(value)
      if (!parsed.success) {
        return false
      }
      config.network = parsed.data
      return true
    }
    case 'webSearch.trustedDomains': {
      const parsed = webSearchTrustedDomainsSchema.safeParse(value)
      if (!parsed.success) {
        return false
      }
      // Persisting a granular update can replay it onto a legacy raw YAML
      // snapshot that predates this top-level section and has not gone through
      // configSchema defaults. Materialize the section before assignment.
      config.webSearch = {
        ...(config.webSearch ?? { trustedDomains: [] }),
        trustedDomains: parsed.data,
      }
      return true
    }
    case 'webSearch.provider': {
      const parsed = webSearchProviderUpdateSchema.safeParse(value)
      if (!parsed.success) {
        return false
      }
      const existing = config.webSearch ?? { trustedDomains: [], provider: 'auto' as const }
      // `***redacted***` is what a settings form gets back from GET /config;
      // treating it as a literal key would overwrite the stored credential
      // with a placeholder the moment anyone saved the form.
      const apiKey = restoreRedactedString(parsed.data.apiKey, existing.apiKey) || undefined
      config.webSearch = {
        ...existing,
        provider: parsed.data.provider,
        apiKey,
        endpoint: parsed.data.endpoint || undefined,
      }
      return true
    }
    case 'channels':
      if (Array.isArray(value)) {
        const parsed = configChannelsUpdateSchema.safeParse(value)
        if (!parsed.success) {
          return false
        }
        // Preserve existing secrets when the update payload contains the
        // `***redacted***` placeholder that the desktop UI inherits from
        // GET /config responses. Without this, saving any channel field
        // would clobber every other secret in that channel.
        config.channels = restoreChannelConfigSecrets(config.channels, parsed.data)
        return true
      }
      return false
    case 'channels.sharedGroupContext':
    case 'channelPipeline.sharedGroupContext':
      if (typeof value === 'boolean') {
        config.channelPipeline = {
          ...(config.channelPipeline ?? channelPipelineConfigSchema.parse({})),
          sharedGroupContext: value,
        }
        return true
      }
      return false
    case 'channels.maxGlobalRuns':
    case 'channelPipeline.maxGlobalRuns':
      if (typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= 100) {
        config.channelPipeline = {
          ...(config.channelPipeline ?? channelPipelineConfigSchema.parse({})),
          maxGlobalRuns: value,
        }
        return true
      }
      return false
    case 'channels.maxGlobalQueuedRuns':
    case 'channelPipeline.maxGlobalQueuedRuns':
      if (typeof value === 'number' && Number.isInteger(value) && value >= 0 && value <= 100) {
        config.channelPipeline = {
          ...(config.channelPipeline ?? channelPipelineConfigSchema.parse({})),
          maxGlobalQueuedRuns: value,
        }
        return true
      }
      return false
    case 'channelPipeline.defaultWorkspaceRoot':
      if (value === null) {
        if (config.channelPipeline) delete config.channelPipeline.defaultWorkspaceRoot
        return true
      }
      if (typeof value === 'string' && value.trim()) {
        config.channelPipeline = {
          ...(config.channelPipeline ?? channelPipelineConfigSchema.parse({})),
          defaultWorkspaceRoot: value.trim(),
        }
        return true
      }
      return false
    case 'providers':
      if (Array.isArray(value)) {
        config.providers = restoreProviderSecrets(
          config.providers,
          value as SepilotdConfig['providers'],
        )
        return true
      }
      return false
    case 'mcp.servers':
      if (Array.isArray(value)) {
        config.mcp.servers = restoreMcpServerSecrets(
          config.mcp.servers,
          value as SepilotdConfig['mcp']['servers'],
        )
        return true
      }
      return false
    case 'mcp.client': {
      const parsed = mcpClientConfigSchema.safeParse(value)
      if (!parsed.success) {
        return false
      }
      config.mcp.client = parsed.data
      return true
    }
    case 'memory.vectorBackend':
      if (
        value === 'auto' ||
        value === 'sqlite-vec' ||
        value === 'sqlite-scan' ||
        value === 'qdrant' ||
        value === 'opensearch' ||
        value === 'elasticsearch' ||
        value === 'meilisearch' ||
        value === 'custom-api'
      ) {
        config.memory.vectorBackend = value
        return true
      }
      return false
    case 'memory.embeddingProvider':
      if (typeof value === 'string' && value.trim().length > 0) {
        config.memory.embeddingProvider = value.trim()
        return true
      }
      if (value === null || value === '') {
        delete config.memory.embeddingProvider
        return true
      }
      return false
    case 'memory.embeddingModel':
      if (typeof value === 'string' && value.trim().length > 0) {
        config.memory.embeddingModel = value.trim()
        return true
      }
      if (value === null || value === '') {
        delete config.memory.embeddingModel
        return true
      }
      return false
    case 'memory.qdrant':
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        config.memory.qdrant = restoreMemoryConfigSecrets(
          config.memory,
          key,
          value,
        ) as SepilotdConfig['memory']['qdrant']
        return true
      }
      return false
    case 'memory.opensearch':
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        config.memory.opensearch = restoreMemoryConfigSecrets(
          config.memory,
          key,
          value,
        ) as SepilotdConfig['memory']['opensearch']
        return true
      }
      return false
    case 'memory.elasticsearch':
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        config.memory.elasticsearch = restoreMemoryConfigSecrets(
          config.memory,
          key,
          value,
        ) as SepilotdConfig['memory']['elasticsearch']
        return true
      }
      return false
    case 'memory.meilisearch':
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        config.memory.meilisearch = restoreMemoryConfigSecrets(
          config.memory,
          key,
          value,
        ) as SepilotdConfig['memory']['meilisearch']
        return true
      }
      return false
    case 'memory.customApi':
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        config.memory.customApi = restoreMemoryConfigSecrets(
          config.memory,
          key,
          value,
        ) as SepilotdConfig['memory']['customApi']
        return true
      }
      return false
    case 'memory.rag':
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        config.memory.rag = restoreMemoryConfigSecrets(
          config.memory,
          key,
          value,
        ) as SepilotdConfig['memory']['rag']
        return true
      }
      return false
    case 'memory.maintenance': {
      const parsed = memoryMaintenanceSchema.safeParse(value)
      if (!parsed.success) {
        return false
      }
      config.memory.maintenance = parsed.data
      return true
    }
    case 'scheduler': {
      const parsed = schedulerConfigSchema.safeParse(value)
      if (!parsed.success) {
        return false
      }
      config.scheduler = parsed.data
      return true
    }
    case 'scheduler.enabled':
      if (typeof value === 'boolean') {
        config.scheduler = {
          ...(config.scheduler ?? schedulerConfigSchema.parse({})),
          enabled: value,
        }
        return true
      }
      return false
    case 'scheduler.timezone':
      if (typeof value === 'string' && value.trim().length > 0) {
        config.scheduler = {
          ...(config.scheduler ?? schedulerConfigSchema.parse({})),
          timezone: value.trim(),
        }
        return true
      }
      if (value === null || value === '') {
        config.scheduler = {
          ...(config.scheduler ?? schedulerConfigSchema.parse({})),
          timezone: undefined,
        }
        return true
      }
      return false
    case 'scheduler.dailyTokenBudget':
      if (value === null || value === undefined || value === '') {
        const current = config.scheduler ?? schedulerConfigSchema.parse({})
        const next = { ...current }
        delete next.dailyTokenBudget
        config.scheduler = next
        return true
      }
      if (typeof value === 'number' && Number.isInteger(value) && value > 0) {
        config.scheduler = {
          ...(config.scheduler ?? schedulerConfigSchema.parse({})),
          dailyTokenBudget: value,
        }
        return true
      }
      return false
    case 'scheduler.maxConsecutiveFailures':
      if (typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= 100) {
        config.scheduler = {
          ...(config.scheduler ?? schedulerConfigSchema.parse({})),
          maxConsecutiveFailures: value,
        }
        return true
      }
      return false
    case 'scheduler.surfaces': {
      const parsed = schedulerSurfaceAccessSchema.safeParse(value)
      if (!parsed.success) {
        return false
      }
      config.scheduler = {
        ...(config.scheduler ?? schedulerConfigSchema.parse({})),
        surfaces: parsed.data,
      }
      return true
    }
    case 'scheduler.surfaces.cli':
      if (typeof value === 'boolean') {
        const current = config.scheduler ?? schedulerConfigSchema.parse({})
        config.scheduler = {
          ...current,
          surfaces: {
            ...schedulerSurfaceAccessSchema.parse(current.surfaces),
            cli: value,
          },
        }
        return true
      }
      return false
    case 'scheduler.surfaces.desktop':
      if (typeof value === 'boolean') {
        const current = config.scheduler ?? schedulerConfigSchema.parse({})
        config.scheduler = {
          ...current,
          surfaces: {
            ...schedulerSurfaceAccessSchema.parse(current.surfaces),
            desktop: value,
          },
        }
        return true
      }
      return false
    case 'scheduler.surfaces.mobile':
      if (typeof value === 'boolean') {
        const current = config.scheduler ?? schedulerConfigSchema.parse({})
        config.scheduler = {
          ...current,
          surfaces: {
            ...schedulerSurfaceAccessSchema.parse(current.surfaces),
            mobile: value,
          },
        }
        return true
      }
      return false
    case 'hooks.outboundWebhooks':
      if (Array.isArray(value)) {
        config.hooks.outboundWebhooks = restoreOutboundWebhookSecrets(
          config.hooks.outboundWebhooks,
          value as SepilotdConfig['hooks']['outboundWebhooks'],
        )
        return true
      }
      return false
    case 'hooks.commandHooks':
      if (Array.isArray(value)) {
        // Validate every command hook before persisting. An unvalidated shell
        // hook (bad event, empty command, out-of-range timeout) would survive
        // the API write and then throw on the next startup config parse,
        // bricking boot. Reject the whole mutation if any item is invalid.
        const validated: SepilotdConfig['hooks']['commandHooks'] = []
        for (const item of value) {
          const parsed = commandHookSchema.safeParse(item)
          if (!parsed.success) {
            return false
          }
          validated.push(parsed.data)
        }
        config.hooks.commandHooks = validated
        return true
      }
      return false
    case 'security.webhooks':
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        const current = config.security.webhooks ?? webhookSecurityPolicyConfigSchema.parse({})
        const nextValue = value as Partial<SepilotdConfig['security']['webhooks']>
        config.security.webhooks = {
          ...current,
          ...nextValue,
          byChannelType: nextValue.byChannelType
            ? { ...nextValue.byChannelType }
            : current.byChannelType,
        }
        return true
      }
      return false
    case 'security.skillSources':
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        const parsed = skillSourceSecuritySchema.safeParse({
          ...(config.security.skillSources ?? {}),
          ...(value as Partial<SepilotdConfig['security']['skillSources']>),
        })
        if (!parsed.success) {
          return false
        }
        config.security.skillSources = parsed.data
        return true
      }
      return false
    case 'observability.channelPipelineHealth':
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        const current =
          config.observability.channelPipelineHealth ?? channelPipelineHealthConfigSchema.parse({})
        const nextValue = value as Partial<SepilotdConfig['observability']['channelPipelineHealth']>
        config.observability.channelPipelineHealth = {
          ...current,
          ...nextValue,
          byChannelType: nextValue.byChannelType
            ? { ...nextValue.byChannelType }
            : current.byChannelType,
        }
        return true
      }
      return false
    case 'observability.webhookSecurityHealth':
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        const current =
          config.observability.webhookSecurityHealth ?? webhookSecurityHealthConfigSchema.parse({})
        const nextValue = value as Partial<SepilotdConfig['observability']['webhookSecurityHealth']>
        config.observability.webhookSecurityHealth = {
          ...current,
          ...nextValue,
        }
        return true
      }
      return false
  }

  return false
}

export function getConfigUpdateValue(config: SepilotdConfig, key: ConfigUpdateKey): unknown {
  switch (key) {
    case 'agent.mode':
      return config.agent.mode
    case 'agent.defaultModel':
      return config.agent.defaultModel
    case 'agent.defaultProvider':
      return config.agent.defaultProvider
    case 'agent.auxModel':
      return config.agent.auxModel
    case 'agent.autonomy':
      return config.agent.autonomy
    case 'agent.thinkingLevel':
      return config.agent.thinkingLevel
    case 'agent.disabledTools':
      return config.agent.disabledTools
    case 'agent.graphNodeModelOverrides':
      return config.agent.graphNodeModelOverrides
    case 'daemon.resumeArtifactRetentionDays':
      return config.daemon.resumeArtifactRetentionDays
    case 'device.name':
      return config.device.name
    case 'network':
      return config.network
    case 'webSearch.trustedDomains':
      return config.webSearch?.trustedDomains ?? []
    case 'webSearch.provider':
      return {
        provider: config.webSearch?.provider ?? 'auto',
        endpoint: config.webSearch?.endpoint,
        // Never echo the credential back through an audit/diff path.
        apiKey: config.webSearch?.apiKey ? '***redacted***' : undefined,
      }
    case 'channels':
      return config.channels
    case 'channels.sharedGroupContext':
    case 'channelPipeline.sharedGroupContext':
      return config.channelPipeline?.sharedGroupContext
    case 'channels.maxGlobalRuns':
    case 'channelPipeline.maxGlobalRuns':
      return config.channelPipeline?.maxGlobalRuns
    case 'channels.maxGlobalQueuedRuns':
    case 'channelPipeline.maxGlobalQueuedRuns':
      return config.channelPipeline?.maxGlobalQueuedRuns
    case 'channelPipeline.defaultWorkspaceRoot':
      return config.channelPipeline?.defaultWorkspaceRoot
    case 'providers':
      return config.providers
    case 'mcp.servers':
      return config.mcp.servers
    case 'mcp.client':
      return config.mcp.client
    case 'hooks.outboundWebhooks':
      return config.hooks.outboundWebhooks
    case 'hooks.commandHooks':
      return config.hooks.commandHooks
    case 'security.webhooks':
      return config.security.webhooks
    case 'security.skillSources':
      return config.security.skillSources
    case 'observability.channelPipelineHealth':
      return config.observability.channelPipelineHealth
    case 'observability.webhookSecurityHealth':
      return config.observability.webhookSecurityHealth
    case 'memory.vectorBackend':
      return config.memory.vectorBackend
    case 'memory.embeddingProvider':
      return config.memory.embeddingProvider
    case 'memory.embeddingModel':
      return config.memory.embeddingModel
    case 'memory.qdrant':
      return config.memory.qdrant
    case 'memory.opensearch':
      return config.memory.opensearch
    case 'memory.elasticsearch':
      return config.memory.elasticsearch
    case 'memory.meilisearch':
      return config.memory.meilisearch
    case 'memory.customApi':
      return config.memory.customApi
    case 'memory.rag':
      return config.memory.rag
    case 'memory.maintenance':
      return config.memory.maintenance
    case 'scheduler':
      return config.scheduler
    case 'scheduler.enabled':
      return config.scheduler?.enabled
    case 'scheduler.timezone':
      return config.scheduler?.timezone
    case 'scheduler.dailyTokenBudget':
      return config.scheduler?.dailyTokenBudget
    case 'scheduler.maxConsecutiveFailures':
      return config.scheduler?.maxConsecutiveFailures
    case 'scheduler.surfaces':
      return config.scheduler?.surfaces
    case 'scheduler.surfaces.cli':
      return config.scheduler?.surfaces?.cli
    case 'scheduler.surfaces.desktop':
      return config.scheduler?.surfaces?.desktop
    case 'scheduler.surfaces.mobile':
      return config.scheduler?.surfaces?.mobile
  }
}
