import type { ILLMProvider, ModelInfo } from '@sepilotd/core'
import { AutonomyLevel } from '@sepilotd/core'
import type { SepilotdConfig } from '../../config/schema.js'
import { createLogger } from '../../logger.js'
import type {
  IProviderFactoryRegistry,
  ProviderConfig,
  ProviderFactory,
} from '../../plugins/contracts.js'
import { AnthropicProvider } from '../../providers/anthropic.js'
import { CodexProvider } from '../../providers/codex.js'
import { GeminiProvider } from '../../providers/gemini.js'
import { OllamaProvider } from '../../providers/ollama.js'
import { OpencodeProvider } from '../../providers/opencode.js'
import { OpenAICompatibleProvider } from '../../providers/openai-compatible.js'
import { OpenAIProvider } from '../../providers/openai.js'
import { wrapProviderForRecordReplay } from '../../providers/record-replay-provider.js'
import { ProviderRegistry } from '../../providers/registry.js'

const log = createLogger('runtime.providers')

export type ProviderFactorySource = 'builtin' | 'plugin'

export class ProviderFactoryRegistry implements IProviderFactoryRegistry {
  private factories = new Map<string, ProviderFactory>()
  private sources = new Map<string, ProviderFactorySource>()
  private pluginScope: { pluginId: string; types: Set<string> } | null = null

  register(type: string, factory: ProviderFactory, options?: { source?: ProviderFactorySource }): void {
    const source = options?.source ?? (this.pluginScope ? 'plugin' : 'builtin')
    const existing = this.sources.get(type)
    // A plugin may never overwrite a built-in provider factory (e.g. 'anthropic')
    // nor another plugin's factory — that is a silent credential/impl hijack.
    if (source === 'plugin' && existing) {
      throw new Error(
        `plugin cannot register provider factory '${type}': already registered by ${existing}`,
      )
    }
    this.factories.set(type, factory)
    this.sources.set(type, source)
    if (source === 'plugin') this.pluginScope?.types.add(type)
  }

  unregister(type: string): boolean {
    const removed = this.factories.delete(type)
    if (removed) this.sources.delete(type)
    return removed
  }

  beginPluginScope(pluginId: string): void {
    this.pluginScope = { pluginId, types: new Set() }
  }

  endPluginScope(): string[] {
    const types = this.pluginScope ? [...this.pluginScope.types] : []
    this.pluginScope = null
    return types
  }

  get(type: string): ProviderFactory | undefined {
    return this.factories.get(type)
  }

  has(type: string): boolean {
    return this.factories.has(type)
  }

  listTypes(): string[] {
    return Array.from(this.factories.keys()).sort()
  }
}

function registerBuiltinProviderFactories(
  registry: ProviderFactoryRegistry,
): void {
  registry.register('ollama', (provider) => ({
    provider: new OllamaProvider({
      baseUrl: provider.baseUrl ?? 'http://localhost:11434',
      headers: provider.headers,
      models: provider.models,
    }),
  }))

  registry.register('opencode', (provider) => ({
    provider: new OpencodeProvider({
      models: provider.models,
      defaultContextWindow: provider.defaultContextWindow,
      defaultMaxOutputTokens: provider.defaultMaxOutputTokens,
    }),
  }))

  registry.register('codex', (provider) => ({
    provider: new CodexProvider({
      models: provider.models,
      defaultContextWindow: provider.defaultContextWindow,
      defaultMaxOutputTokens: provider.defaultMaxOutputTokens,
    }),
  }))

  registry.register('openai', (provider) => provider.apiKey
    ? {
        provider: new OpenAIProvider({
          apiKey: provider.apiKey,
          models: provider.models,
          baseUrl: provider.baseUrl,
          headers: provider.headers,
        }),
      }
    : { skipReason: 'apiKey is required' })

  registry.register('anthropic', (provider) => provider.apiKey
    ? {
        provider: new AnthropicProvider({
          apiKey: provider.apiKey,
          models: provider.models,
          baseUrl: provider.baseUrl,
          headers: provider.headers,
        }),
      }
    : { skipReason: 'apiKey is required' })

  registry.register('gemini', (provider) => provider.apiKey
    ? {
        provider: new GeminiProvider({
          apiKey: provider.apiKey,
          models: provider.models,
          baseUrl: provider.baseUrl,
          headers: provider.headers,
        }),
      }
    : { skipReason: 'apiKey is required' })

  registry.register('groq', (provider) => provider.apiKey
    ? {
        provider: OpenAICompatibleProvider.groq(
          provider.apiKey,
          provider.models,
          { baseUrl: provider.baseUrl, headers: provider.headers },
        ),
      }
    : { skipReason: 'apiKey is required' })

  registry.register('together', (provider) => provider.apiKey
    ? {
        provider: OpenAICompatibleProvider.together(
          provider.apiKey,
          provider.models,
          { baseUrl: provider.baseUrl, headers: provider.headers },
        ),
      }
    : { skipReason: 'apiKey is required' })

  registry.register('deepseek', (provider) => provider.apiKey
    ? {
        provider: OpenAICompatibleProvider.deepseek(
          provider.apiKey,
          provider.models,
          { baseUrl: provider.baseUrl, headers: provider.headers },
        ),
      }
    : { skipReason: 'apiKey is required' })

  registry.register('openrouter', (provider) => provider.apiKey
    ? {
        provider: OpenAICompatibleProvider.openrouter(
          provider.apiKey,
          provider.models,
          {
            baseUrl: provider.baseUrl,
            headers: provider.headers,
            defaultContextWindow: provider.defaultContextWindow,
            defaultMaxOutputTokens: provider.defaultMaxOutputTokens,
            capabilities: provider.capabilities,
          },
        ),
      }
    : { skipReason: 'apiKey is required' })

  const openAICompatibleFactory: ProviderFactory = (provider) => provider.baseUrl
    ? {
        provider: new OpenAICompatibleProvider({
          id: provider.id,
          name: provider.id,
          apiKey: provider.apiKey || 'local',
          baseUrl: provider.baseUrl,
          headers: provider.headers,
          models: provider.models,
          defaultContextWindow: provider.defaultContextWindow,
          defaultMaxOutputTokens: provider.defaultMaxOutputTokens,
          capabilities: provider.capabilities,
        }),
      }
    : { skipReason: 'baseUrl is required' }
  registry.register('custom', openAICompatibleFactory)
  registry.register('openai-compat', openAICompatibleFactory)
  registry.register('openai-compatible', openAICompatibleFactory)
}

export function applyProviderModelOverrides(
  models: ModelInfo[],
  provider: ProviderConfig,
): void {
  const configuredOverrides = provider.modelOverrides ?? []
  const exactModelOverrides = new Map(
    configuredOverrides.map((override) => [override.id, override]),
  )

  const modelIdParts = (value: string): string[] => value
    .trim()
    .toLowerCase()
    .split(/[:/_-]+/)
    .filter(Boolean)
  const partsStartWith = (candidate: string[], prefix: string[]): boolean => (
    prefix.length >= 2
    && prefix.length <= candidate.length
    && prefix.every((part, index) => candidate[index] === part)
  )
  const resolveOverride = (
    modelId: string,
  ): NonNullable<ProviderConfig['modelOverrides']>[number] | undefined => {
    const exact = exactModelOverrides.get(modelId)
    if (exact) return exact

    // Endpoint discovery can replace a configured alias with a separator-only
    // or family-suffixed exact id. Inherit only the single most-specific
    // explicit override; equal candidates are ambiguous and intentionally
    // leave the model at provider defaults.
    const targetParts = modelIdParts(modelId)
    const prefixMatches = configuredOverrides
      .map((override) => ({ override, parts: modelIdParts(override.id) }))
      .filter(({ parts }) => parts.length <= targetParts.length && partsStartWith(targetParts, parts))
    if (prefixMatches.length === 0) return undefined

    const longestPrefix = Math.max(...prefixMatches.map(({ parts }) => parts.length))
    const mostSpecific = prefixMatches.filter(({ parts }) => parts.length === longestPrefix)
    return mostSpecific.length === 1 ? mostSpecific[0]?.override : undefined
  }

  for (const model of models) {
    if (provider.defaultContextWindow !== undefined) {
      model.contextWindow = provider.defaultContextWindow
    }
    if (provider.defaultMaxOutputTokens !== undefined) {
      model.maxOutputTokens = provider.defaultMaxOutputTokens
    }
    if (provider.capabilities) {
      model.capabilities = {
        ...model.capabilities,
        ...provider.capabilities,
      }
    }

    const override = resolveOverride(model.id)
    if (!override) {
      continue
    }

    if (override.contextWindow !== undefined) {
      model.contextWindow = override.contextWindow
    }
    if (override.maxOutputTokens !== undefined) {
      model.maxOutputTokens = override.maxOutputTokens
    }
    if (override.capabilities) {
      model.capabilities = {
        ...model.capabilities,
        ...override.capabilities,
      }
    }
    if (override.compatibility) {
      model.compatibility = {
        ...model.compatibility,
        ...override.compatibility,
        notes: override.compatibility.notes
          ? [...override.compatibility.notes]
          : model.compatibility?.notes,
      }
    }
  }
}

export function resolveConfigEnvReference(
  value: string | undefined,
  env: Record<string, string | undefined> = process.env,
): string | undefined {
  if (typeof value !== 'string' || value.length === 0) {
    return value
  }

  return value.replace(/\$\{([^}]+)\}/g, (match, variableName: string) => (
    env[variableName] ?? match
  ))
}

export function extractMissingConfigEnvReferences(
  value: string | undefined,
  env: Record<string, string | undefined> = process.env,
): string[] {
  if (typeof value !== 'string' || value.length === 0) {
    return []
  }

  const missing = new Set<string>()
  for (const match of value.matchAll(/\$\{([^}]+)\}/g)) {
    const variableName = match[1]
    if (variableName && env[variableName] == null) {
      missing.add(variableName)
    }
  }

  return Array.from(missing)
}

export function resolveConfigHeaderReferences(
  headers: Record<string, string> | undefined,
  env: Record<string, string | undefined> = process.env,
): Record<string, string> {
  return Object.fromEntries(
    Object.entries(headers ?? {}).map(([key, value]) => [
      key,
      resolveConfigEnvReference(value, env) ?? value,
    ]),
  )
}

export function extractMissingConfigHeaderEnvReferences(
  headers: Record<string, string> | undefined,
  env: Record<string, string | undefined> = process.env,
): string[] {
  return Array.from(new Set(
    Object.values(headers ?? {}).flatMap((value) => (
      extractMissingConfigEnvReferences(value, env)
    )),
  )).sort()
}

export interface BuildProviderInstanceResult {
  provider?: ILLMProvider
  resolvedProvider: ProviderConfig
  skipReason?: string
}

export function buildProviderInstance(
  provider: ProviderConfig,
  factoryRegistry: IProviderFactoryRegistry,
  env: Record<string, string | undefined> = process.env,
): BuildProviderInstanceResult {
  const factory = factoryRegistry.get(provider.type)
  if (!factory) {
    return {
      resolvedProvider: provider,
      skipReason: 'no factory registered for provider type',
    }
  }

  const resolvedProvider = {
    ...provider,
    apiKey: resolveConfigEnvReference(provider.apiKey, env),
    baseUrl: resolveConfigEnvReference(provider.baseUrl, env),
    headers: resolveConfigHeaderReferences(provider.headers, env),
  }

  const result = factory(resolvedProvider)
  if (!result.provider) {
    return {
      resolvedProvider,
      skipReason: result.skipReason ?? 'factory returned no provider',
    }
  }

  applyProviderModelOverrides(result.provider.models, resolvedProvider)

  return {
    provider: result.provider,
    resolvedProvider,
  }
}

export function createProviderFactoryRegistry(): ProviderFactoryRegistry {
  const registry = new ProviderFactoryRegistry()
  registerBuiltinProviderFactories(registry)
  return registry
}

export async function syncDynamicProviderModels(
  registry: ProviderRegistry,
  config: SepilotdConfig,
): Promise<void> {
  for (const providerCfg of config.providers) {
    const instance = registry.get(providerCfg.id)
    if (!instance?.refreshModelCatalog) continue

    try {
      const result = await instance.refreshModelCatalog()
      applyProviderModelOverrides(instance.models, providerCfg)
      if (result.added.length > 0 || result.removed.length > 0) {
        log.info(`Provider ${providerCfg.id} reconciled its endpoint model catalog`, {
          added: result.added,
          removed: result.removed,
          availableCount: result.modelIds.length,
        })
      }
    } catch (error) {
      log.warn(`Failed to sync provider ${providerCfg.id} models`, {
        reason: error instanceof Error ? error.message : String(error),
      })
    }
  }
}

export function buildProviderRegistry(
  config: SepilotdConfig,
  factoryRegistry: IProviderFactoryRegistry,
): ProviderRegistry {
  const providerRegistry = new ProviderRegistry()
  const markedDefaultId = config.providers.find(provider => provider.default)?.id
  const configuredDefaultId = config.agent.defaultProvider
  const defaultProviderId = configuredDefaultId ?? markedDefaultId ?? config.providers[0]?.id

  for (const provider of config.providers) {
    try {
      const result = buildProviderInstance(provider, factoryRegistry)
      if (!result.provider) {
        log.warn(`Skipping provider ${provider.id}`, {
          type: provider.type,
          reason: result.skipReason ?? 'factory returned no provider',
        })
        continue
      }

      providerRegistry.register(
        provider.id,
        wrapProviderForRecordReplay(provider.id, result.provider),
        { default: provider.id === defaultProviderId },
      )
    } catch (error) {
      log.warn(`Skipping provider ${provider.id}`, {
        type: provider.type,
        reason: error instanceof Error ? error.message : String(error),
      })
    }
  }

  return providerRegistry
}

export function resolveAutonomy(config: SepilotdConfig): AutonomyLevel {
  const autonomyMap: Record<string, AutonomyLevel> = {
    readonly: AutonomyLevel.ReadOnly,
    'accept-edits': AutonomyLevel.AcceptEdits,
    'workspace-write': AutonomyLevel.WorkspaceWrite,
    supervised: AutonomyLevel.Supervised,
    autonomous: AutonomyLevel.Autonomous,
  }

  return autonomyMap[config.agent.autonomy] ?? AutonomyLevel.Supervised
}
