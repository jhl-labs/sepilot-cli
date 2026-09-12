import type { SepilotdConfig } from './schema.js'
import { configChannelsUpdateSchema } from './channel-update-schema.js'

export interface ConfigValidationResult {
  valid: boolean
  errors: string[]
  warnings: string[]
}

const EMBEDDING_PROVIDER_TYPES = new Set(['openai', 'ollama', 'groq', 'together', 'deepseek'])

const API_KEY_PROVIDER_TYPES = new Set([
  'openai',
  'anthropic',
  'gemini',
  'groq',
  'together',
  'deepseek',
  'openrouter',
])

export function validateConfig(config: SepilotdConfig): ConfigValidationResult {
  const errors: string[] = []
  const warnings: string[] = []
  const providerIds = new Set<string>()
  const mcpServerNames = new Set<string>()
  const outboundWebhookTargets = new Set<string>()

  // Check device
  if (!config.device.id || config.device.id === 'auto-generated-on-init') {
    warnings.push('Device ID is placeholder. Run "sepilot init" to generate a proper ID.')
  }
  if (!config.device.name || config.device.name === 'default') {
    warnings.push('Device name is "default". Set a meaningful name in config.yaml.')
  }

  // Check providers
  if (config.providers.length === 0) {
    warnings.push('No LLM providers configured. Add at least one provider to config.yaml.')
  }
  for (const p of config.providers) {
    if (providerIds.has(p.id)) {
      errors.push(`Provider ID "${p.id}" is duplicated. Provider IDs must be unique.`)
    } else {
      providerIds.add(p.id)
    }
    if (API_KEY_PROVIDER_TYPES.has(p.type) && !p.apiKey) {
      errors.push(`Provider "${p.id}" requires an API key. Use \${ENV_VAR} syntax.`)
    }
    if (p.models.length === 0) {
      warnings.push(`Provider "${p.id}" has no models configured and is not ready for chat.`)
    }
  }

  const explicitDefaultProvider = config.agent.defaultProvider
    ? config.providers.find((provider) => provider.id === config.agent.defaultProvider)
    : undefined
  if (config.agent.defaultProvider && !explicitDefaultProvider) {
    errors.push(
      `agent.defaultProvider "${config.agent.defaultProvider}" does not match any configured provider ID.`,
    )
  }
  if (explicitDefaultProvider?.models.length === 0) {
    errors.push(
      `Default provider "${explicitDefaultProvider.id}" must have at least one configured model.`,
    )
  }
  for (const provider of config.providers) {
    if (provider.default && provider.models.length === 0 && provider !== explicitDefaultProvider) {
      errors.push(`Default provider "${provider.id}" must have at least one configured model.`)
    }
  }

  const effectiveDefaultProvider =
    explicitDefaultProvider ??
    config.providers.find((provider) => provider.default) ??
    config.providers[0]
  if (config.agent.defaultModel) {
    if (!effectiveDefaultProvider) {
      errors.push(
        `agent.defaultModel "${config.agent.defaultModel}" requires a configured provider.`,
      )
    } else if (!effectiveDefaultProvider.models.includes(config.agent.defaultModel)) {
      errors.push(
        `agent.defaultModel "${config.agent.defaultModel}" is not configured for provider "${effectiveDefaultProvider.id}".`,
      )
    }
  }

  const intentProviderId = config.agent.intentRouter?.provider
  const intentProvider = intentProviderId
    ? config.providers.find((provider) => provider.id === intentProviderId)
    : effectiveDefaultProvider
  if (intentProviderId && !intentProvider) {
    errors.push(
      `agent.intentRouter.provider "${intentProviderId}" does not match any configured provider ID.`,
    )
  }
  if (
    config.agent.intentRouter?.model &&
    intentProvider &&
    !intentProvider.models.includes(config.agent.intentRouter.model)
  ) {
    errors.push(
      `agent.intentRouter.model "${config.agent.intentRouter.model}" is not configured for provider "${intentProvider.id}".`,
    )
  }

  if (config.memory.embeddingProvider || config.memory.embeddingModel) {
    if (!config.memory.embeddingProvider || !config.memory.embeddingModel) {
      errors.push('memory.embeddingProvider and memory.embeddingModel must be configured together.')
    } else {
      const embeddingProvider = config.providers.find(
        (provider) => provider.id === config.memory.embeddingProvider,
      )
      if (!embeddingProvider) {
        errors.push(
          `memory.embeddingProvider "${config.memory.embeddingProvider}" does not match any configured provider ID.`,
        )
      } else if (
        API_KEY_PROVIDER_TYPES.has(embeddingProvider.type) ||
        embeddingProvider.type === 'ollama'
      ) {
        if (!EMBEDDING_PROVIDER_TYPES.has(embeddingProvider.type)) {
          errors.push(
            `Provider "${embeddingProvider.id}" of type "${embeddingProvider.type}" does not support embeddings.`,
          )
        }
      } else if (embeddingProvider.type.trim().length === 0) {
        errors.push(
          `Provider "${embeddingProvider.id}" of type "${embeddingProvider.type}" does not support embeddings.`,
        )
      }
    }
  }

  const vectorBackend = config.memory.vectorBackend ?? 'auto'
  if (
    ![
      'auto',
      'sqlite-vec',
      'sqlite-scan',
      'qdrant',
      'opensearch',
      'elasticsearch',
      'meilisearch',
      'custom-api',
    ].includes(vectorBackend)
  ) {
    errors.push(`memory.vectorBackend "${vectorBackend}" is not supported.`)
  }
  if (vectorBackend === 'qdrant' && !config.memory.qdrant?.url) {
    errors.push('memory.qdrant.url is required when memory.vectorBackend is "qdrant".')
  }
  if (vectorBackend === 'opensearch' && !config.memory.opensearch?.url) {
    errors.push('memory.opensearch.url is required when memory.vectorBackend is "opensearch".')
  }
  if (vectorBackend === 'elasticsearch' && !config.memory.elasticsearch?.url) {
    errors.push(
      'memory.elasticsearch.url is required when memory.vectorBackend is "elasticsearch".',
    )
  }
  if (vectorBackend === 'meilisearch' && !config.memory.meilisearch?.url) {
    errors.push('memory.meilisearch.url is required when memory.vectorBackend is "meilisearch".')
  }
  if (vectorBackend === 'custom-api' && !config.memory.customApi?.url) {
    errors.push('memory.customApi.url is required when memory.vectorBackend is "custom-api".')
  }
  if (
    vectorBackend === 'custom-api' &&
    config.memory.customApi?.auth?.type === 'basic' &&
    (!config.memory.customApi.auth.username || !config.memory.customApi.auth.password)
  ) {
    errors.push(
      'memory.customApi.auth.username and password are required when auth.type is "basic".',
    )
  }
  if (
    config.memory.rag?.rerank?.enabled &&
    config.memory.rag.rerank.provider === 'custom-api' &&
    !config.memory.rag.rerank.endpoint
  ) {
    errors.push('memory.rag.rerank.endpoint is required when custom rerank is enabled.')
  }
  if (
    config.memory.rag?.rerank?.provider === 'custom-api' &&
    config.memory.rag.rerank.auth?.type === 'basic' &&
    (!config.memory.rag.rerank.auth.username || !config.memory.rag.rerank.auth.password)
  ) {
    errors.push(
      'memory.rag.rerank.auth.username and password are required when auth.type is "basic".',
    )
  }

  // Check ports
  if (config.daemon.port === Number(config.gateway?.url?.match(/:(\d+)/)?.[1])) {
    errors.push('Daemon and Gateway ports cannot be the same.')
  }

  // Check security
  if (!config.security.auditLog) {
    warnings.push('Audit logging is disabled. Enable for security compliance.')
  }

  // Surface the unsandboxed default. sandbox 'local' runs tool commands with no
  // isolation; operators should know they are relying on the tool policy alone.
  const sandbox = config.security.sandbox ?? 'local'
  if (sandbox === 'local') {
    warnings.push(
      'security.sandbox is "local" (no isolation). Tool commands run directly on the host; set docker or bubblewrap for isolation.',
    )
  } else if (sandbox === 'docker' && !config.security.sandboxDocker) {
    warnings.push(
      'security.sandbox is "docker" but security.sandboxDocker is not configured; falling back to defaults.',
    )
  } else if (sandbox === 'bubblewrap' && !config.security.sandboxBubblewrap) {
    warnings.push(
      'security.sandbox is "bubblewrap" but security.sandboxBubblewrap is not configured; falling back to defaults.',
    )
  }
  if (
    sandbox !== 'local'
    && config.security.sandboxHostFileTools === 'allow'
  ) {
    warnings.push(
      `security.sandboxHostFileTools is "allow"; built-in host file tools bypass the ${sandbox} filesystem boundary.`,
    )
  }

  // Check channels
  const channelValidation = configChannelsUpdateSchema.safeParse(config.channels)
  if (!channelValidation.success) {
    for (const issue of channelValidation.error.issues) {
      const path = issue.path.length > 0 ? ` at channels.${issue.path.join('.')}` : ''
      errors.push(`Invalid channel config${path}: ${issue.message}`)
    }
  }

  // Check MCP servers
  for (const server of config.mcp.servers) {
    if (mcpServerNames.has(server.name)) {
      errors.push(`MCP server "${server.name}" is duplicated. MCP server names must be unique.`)
    } else {
      mcpServerNames.add(server.name)
    }
  }

  for (const webhook of config.hooks?.outboundWebhooks ?? []) {
    const key = `${webhook.url}::${[...webhook.events].sort().join(',')}`
    if (outboundWebhookTargets.has(key)) {
      warnings.push(
        `Outbound webhook "${webhook.url}" is registered more than once for the same event set.`,
      )
      continue
    }
    outboundWebhookTargets.add(key)
  }

  return { valid: errors.length === 0, errors, warnings }
}
