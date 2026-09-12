import type { ILLMProvider, ModelInfo } from '@sepilotd/core'
import type { SepilotdConfig } from '../../config/schema.js'
import { createLogger } from '../../logger.js'
import {
  MODEL_PROBE_TIMEOUT_MS,
  probeProviderModel,
  type ModelProbeResult,
} from '../../providers/model-probe.js'
import type { ConfigUpdateKey } from './config-mutations.js'
import { applyAndPersistRuntimeUpdate } from './config-runtime.js'
import { resolveConfigEnvReference } from './providers.js'
import type { RuntimeServices } from './types.js'

const log = createLogger('model-control')

export type ModelControlRuntime = Pick<RuntimeServices,
  | 'config'
  | 'providerRegistry'
  | 'configMutationService'
  | 'providerFactoryRegistry'
  | 'dataDir'
  | 'dreaming'
  | 'modelRouter'
  | 'mcpManager'
  | 'mcpPromptsRegistry'
  | 'toolRegistry'
  | 'skillSourceUrlPolicy'
  | 'semanticIndex'
  | 'hookRegistry'
  | 'auditLogger'
  | 'autonomy'
  | 'channelAcl'
>

export interface ModelTarget {
  providerId: string
  modelId: string
}

export interface AvailableProviderModels {
  providerId: string
  providerName: string
  providerType?: string
  models: ModelInfo[]
  configuredModelIds: string[]
  defaultModelId?: string
  ready: boolean
}

export type ModelTargetResolution =
  | { status: 'matched'; target: ModelTarget }
  | { status: 'not_found'; query: string }
  | { status: 'ambiguous'; query: string; matches: ModelTarget[] }

export type ModelSelfTestResult = ModelProbeResult

export type ModelSwitchResult =
  | {
      ok: true
      status: 'switched' | 'no_change'
      providerId: string
      model: string
      previousProviderId?: string
      previousModel?: string
      latencyMs?: number
      message: string
    }
  | {
      ok: false
      status: 'self_test_failed_rolled_back' | 'self_test_failed_no_previous' | 'rollback_failed'
      providerId: string
      model: string
      previousProviderId?: string
      previousModel?: string
      latencyMs?: number
      reason: string
      rollbackError?: string
      message: string
    }

export interface ModelPullResult {
  ok: true
  providerId: string
  model: string
  alreadyAvailable: boolean
  alreadyConfigured: boolean
  latencyMs: number
  message: string
}

function configuredModelIds(
  provider: { models?: unknown[] },
): string[] {
  if (!Array.isArray(provider.models)) return []
  return provider.models
    .map((model) => {
      if (typeof model === 'string') return model.trim()
      if (model && typeof model === 'object' && 'id' in model) {
        return String(model.id).trim()
      }
      return ''
    })
    .filter((model) => model.length > 0)
}

function modelIsChatCapable(model: ModelInfo): boolean {
  const capabilities = model.capabilities
  return capabilities?.embedding !== true || capabilities.toolUse === true
}

function firstChatModel(provider: ILLMProvider | undefined): string | undefined {
  if (!provider) return undefined
  return provider.models.find(modelIsChatCapable)?.id ?? provider.models[0]?.id
}

export function normalizeModelLookupId(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[\s_-]+/g, '')
}

export function cleanModelTargetText(value: string): string {
  let candidate = value.trim()
  if (
    (candidate.startsWith('"') && candidate.endsWith('"'))
    || (candidate.startsWith("'") && candidate.endsWith("'"))
    || (candidate.startsWith('`') && candidate.endsWith('`'))
  ) {
    candidate = candidate.slice(1, -1).trim()
  }
  return candidate
    .replace(/^[\s:：=]+/u, '')
    .replace(/[),.;!?。！？]+$/u, '')
    .replace(/(?:\s+model|모델)$/iu, '')
    .replace(/(?:으로|로|에서|에|를|을|은|는|이|가)$/u, '')
    .trim()
}

function configuredProvider(runtime: ModelControlRuntime, providerId: string) {
  return runtime.config.providers.find((provider) => provider.id === providerId)
}

export function currentModelTarget(runtime: ModelControlRuntime): ModelTarget | null {
  const defaultProvider = runtime.config.agent?.defaultProvider
  const defaultModel = runtime.config.agent?.defaultModel
  if (defaultProvider && defaultModel) {
    return { providerId: defaultProvider, modelId: defaultModel }
  }

  const provider = runtime.providerRegistry.getDefault()
  const model = defaultModel ?? firstChatModel(provider)
  if (!provider || !model) return null
  return { providerId: provider.id, modelId: model }
}

export function availableProviderModels(runtime: ModelControlRuntime): AvailableProviderModels[] {
  const configured = runtime.config.providers ?? []
  const seen = new Set<string>()
  const orderedIds: string[] = []

  for (const provider of configured) {
    if (!seen.has(provider.id)) {
      orderedIds.push(provider.id)
      seen.add(provider.id)
    }
  }

  const registry = runtime.providerRegistry as unknown as {
    list?: () => ILLMProvider[]
  }
  for (const provider of registry.list?.() ?? []) {
    if (!seen.has(provider.id)) {
      orderedIds.push(provider.id)
      seen.add(provider.id)
    }
  }

  return orderedIds.map((providerId) => {
    const configProvider = configuredProvider(runtime, providerId)
    const runtimeProvider = runtime.providerRegistry.get(providerId)
    return {
      providerId,
      providerName: runtimeProvider?.name ?? providerId,
      providerType: configProvider?.type,
      models: runtimeProvider?.models ?? [],
      configuredModelIds: configProvider ? configuredModelIds(configProvider) : [],
      defaultModelId: firstChatModel(runtimeProvider),
      ready: Boolean(runtimeProvider),
    }
  })
}

export function formatAvailableModels(runtime: ModelControlRuntime): string[] {
  return availableProviderModels(runtime).map((provider) => {
    if (!provider.ready) return `• ${provider.providerId}: unavailable`
    const models = provider.models.map((model) => model.id).join(', ')
    return `• ${provider.providerId}: ${models || '(no models)'}`
  })
}

function matchModelInProvider(
  provider: ILLMProvider | undefined,
  query: string,
): string | null {
  if (!provider) return null
  const cleaned = cleanModelTargetText(query)
  if (!cleaned) return null
  const normalizedQuery = normalizeModelLookupId(cleaned)
  const exact = provider.models.find((model) => model.id === cleaned)
    ?? provider.models.find((model) => normalizeModelLookupId(model.id) === normalizedQuery)
  if (exact) return exact.id

  const prefixMatches = provider.models.filter((model) =>
    normalizeModelLookupId(model.id).startsWith(normalizedQuery),
  )
  return prefixMatches.length === 1 ? prefixMatches[0]!.id : null
}

function providerModelFromDelimitedArg(
  runtime: ModelControlRuntime,
  arg: string,
): ModelTarget | null {
  const cleaned = cleanModelTargetText(arg)
  const slash = cleaned.indexOf('/')
  if (slash > 0) {
    const providerId = cleaned.slice(0, slash)
    const modelQuery = cleaned.slice(slash + 1)
    const provider = runtime.providerRegistry.get(providerId)
    const modelId = matchModelInProvider(provider, modelQuery)
    if (modelId) return { providerId, modelId }
  }

  const colon = cleaned.indexOf(':')
  if (colon > 0) {
    const providerId = cleaned.slice(0, colon)
    if (runtime.providerRegistry.get(providerId)) {
      const modelId = matchModelInProvider(
        runtime.providerRegistry.get(providerId),
        cleaned.slice(colon + 1),
      )
      if (modelId) return { providerId, modelId }
    }
  }

  return null
}

function bareModelMatches(
  runtime: ModelControlRuntime,
  query: string,
): ModelTarget[] {
  const cleaned = cleanModelTargetText(query)
  const normalizedQuery = normalizeModelLookupId(cleaned)
  const providers = availableProviderModels(runtime)
    .map((provider) => ({
      providerId: provider.providerId,
      provider: runtime.providerRegistry.get(provider.providerId),
    }))
    .filter((entry): entry is { providerId: string; provider: ILLMProvider } =>
      Boolean(entry.provider),
    )

  const exact: ModelTarget[] = []
  for (const { providerId, provider } of providers) {
    for (const model of provider.models) {
      if (model.id === cleaned || normalizeModelLookupId(model.id) === normalizedQuery) {
        exact.push({ providerId, modelId: model.id })
      }
    }
  }
  if (exact.length > 0) return exact

  const prefix: ModelTarget[] = []
  for (const { providerId, provider } of providers) {
    for (const model of provider.models) {
      if (normalizeModelLookupId(model.id).startsWith(normalizedQuery)) {
        prefix.push({ providerId, modelId: model.id })
      }
    }
  }
  return prefix
}

const MIN_FUZZY_QUERY_LENGTH = 3

function isSubsequenceMatch(query: string, candidate: string): boolean {
  if (!query) return false
  let queryIndex = 0
  for (let i = 0; i < candidate.length && queryIndex < query.length; i++) {
    if (candidate[i] === query[queryIndex]) queryIndex++
  }
  return queryIndex === query.length
}

function fuzzyModelMatches(
  runtime: ModelControlRuntime,
  query: string,
): ModelTarget[] {
  const cleaned = cleanModelTargetText(query).toLowerCase()
  if (cleaned.length < MIN_FUZZY_QUERY_LENGTH) return []
  const providers = availableProviderModels(runtime)
    .map((provider) => ({
      providerId: provider.providerId,
      provider: runtime.providerRegistry.get(provider.providerId),
    }))
    .filter((entry): entry is { providerId: string; provider: ILLMProvider } =>
      Boolean(entry.provider),
    )

  const matches: ModelTarget[] = []
  for (const { providerId, provider } of providers) {
    for (const model of provider.models) {
      const combined = `${providerId}/${model.id}`.toLowerCase()
      if (isSubsequenceMatch(cleaned, combined)) {
        matches.push({ providerId, modelId: model.id })
      }
    }
  }
  return matches
}

function preferCurrentProvider(
  runtime: ModelControlRuntime,
  matches: ModelTarget[],
): ModelTarget[] {
  const currentProvider = currentModelTarget(runtime)?.providerId
  if (!currentProvider) return matches
  const preferred = matches.filter((match) => match.providerId === currentProvider)
  return preferred.length > 0 ? preferred : matches
}

export function resolveModelTarget(
  runtime: ModelControlRuntime,
  rawArgs: string,
): ModelTargetResolution {
  const query = cleanModelTargetText(rawArgs)
  if (!query) return { status: 'not_found', query }

  const delimited = providerModelFromDelimitedArg(runtime, query)
  if (delimited) return { status: 'matched', target: delimited }

  const tokens = query.split(/\s+/).filter(Boolean)
  if (tokens.length >= 2) {
    const providerId = tokens[0]!
    const provider = runtime.providerRegistry.get(providerId)
    const modelId = matchModelInProvider(provider, tokens.slice(1).join(' '))
    if (modelId) return { status: 'matched', target: { providerId, modelId } }
  }

  const directProvider = runtime.providerRegistry.get(query)
  const directProviderModel = firstChatModel(directProvider)
  if (directProvider && directProviderModel) {
    return { status: 'matched', target: { providerId: query, modelId: directProviderModel } }
  }

  const matches = preferCurrentProvider(runtime, bareModelMatches(runtime, query))
  if (matches.length === 1) return { status: 'matched', target: matches[0]! }
  if (matches.length > 1) return { status: 'ambiguous', query, matches }

  const fuzzyMatches = preferCurrentProvider(runtime, fuzzyModelMatches(runtime, query))
  if (fuzzyMatches.length === 1) return { status: 'matched', target: fuzzyMatches[0]! }
  if (fuzzyMatches.length > 1) return { status: 'ambiguous', query, matches: fuzzyMatches }

  return { status: 'not_found', query }
}

export async function selfTestProviderModel(
  runtime: ModelControlRuntime,
  target: ModelTarget,
  timeoutMs = MODEL_PROBE_TIMEOUT_MS,
): Promise<ModelSelfTestResult> {
  const provider = runtime.providerRegistry.get(target.providerId)
  if (!provider) {
    return {
      ok: false,
      providerId: target.providerId,
      modelId: target.modelId,
      latencyMs: 0,
      reason: 'provider is not initialized',
    }
  }

  return probeProviderModel(provider, target, timeoutMs)
}

export async function applyDefaultModel(
  runtime: ModelControlRuntime,
  target: ModelTarget,
  reason: string,
): Promise<void> {
  await runtime.configMutationService.apply(reason, async () => {
    const configured = runtime.config.providers.find(
      (provider) => provider.id === target.providerId,
    )
    if (!configured || !configuredModelIds(configured).includes(target.modelId)) {
      throw new Error(
        `Model ${target.providerId}/${target.modelId} is no longer present in the provider configuration.`,
      )
    }
    runtime.config.agent.defaultProvider = target.providerId
    runtime.config.agent.defaultModel = target.modelId
    await applyAndPersistRuntimeUpdate(
      runtime,
      new Set<ConfigUpdateKey>(['agent.defaultProvider', 'agent.defaultModel']),
      {
        'agent.defaultProvider': target.providerId,
        'agent.defaultModel': target.modelId,
      },
    )
  })
}

export async function switchDefaultModelWithSelfTest(
  runtime: ModelControlRuntime,
  target: ModelTarget,
  options: { timeoutMs?: number; reason?: string } = {},
): Promise<ModelSwitchResult> {
  const previous = currentModelTarget(runtime)
  if (previous?.providerId === target.providerId && previous.modelId === target.modelId) {
    return {
      ok: true,
      status: 'no_change',
      providerId: target.providerId,
      model: target.modelId,
      previousProviderId: previous.providerId,
      previousModel: previous.modelId,
      message: `Already using ${target.providerId} / ${target.modelId}.`,
    }
  }

  const test = await selfTestProviderModel(
    runtime,
    target,
    options.timeoutMs ?? MODEL_PROBE_TIMEOUT_MS,
  )
  if (!test.ok) {
    return {
      ok: false,
      status: previous ? 'self_test_failed_rolled_back' : 'self_test_failed_no_previous',
      providerId: target.providerId,
      model: target.modelId,
      previousProviderId: previous?.providerId,
      previousModel: previous?.modelId,
      latencyMs: test.latencyMs,
      reason: test.reason ?? 'self-test failed',
      message: previous
        ? `Self-test failed for ${target.providerId} / ${target.modelId} (${test.reason ?? 'unknown error'}); the default remains ${previous.providerId} / ${previous.modelId}.`
        : `Self-test failed for ${target.providerId} / ${target.modelId} (${test.reason ?? 'unknown error'}); no default model was saved.`,
    }
  }

  await applyDefaultModel(runtime, target, options.reason ?? 'model.switch')
  return {
    ok: true,
    status: 'switched',
    providerId: target.providerId,
    model: target.modelId,
    previousProviderId: previous?.providerId,
    previousModel: previous?.modelId,
    latencyMs: test.latencyMs,
    message: `Switched to ${target.providerId} / ${target.modelId} (self-test ${test.latencyMs}ms).`,
  }
}

function resolveHeaderReferences(
  headers: Record<string, string> | undefined,
): Record<string, string> {
  return Object.fromEntries(
    Object.entries(headers ?? {}).map(([key, value]) => [
      key,
      resolveConfigEnvReference(value) ?? value,
    ]),
  )
}

function timeoutSignal(timeoutMs: number): { signal: AbortSignal; clear: () => void } {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  return {
    signal: controller.signal,
    clear: () => clearTimeout(timer),
  }
}

function modelConfigured(provider: SepilotdConfig['providers'][number], modelId: string): boolean {
  return configuredModelIds(provider).some((configured) => configured === modelId)
}

function addConfiguredModel(
  provider: SepilotdConfig['providers'][number],
  modelId: string,
): SepilotdConfig['providers'][number] {
  if (modelConfigured(provider, modelId)) return provider
  return {
    ...provider,
    models: [...configuredModelIds(provider), modelId],
  }
}

export function resolveOllamaPullTarget(
  runtime: ModelControlRuntime,
  rawModel: string,
  explicitProviderId?: string,
): ModelTarget {
  const query = cleanModelTargetText(rawModel)
  const delimited = providerModelFromDelimitedArg(runtime, query)
  if (delimited) return delimited

  const tokens = query.split(/\s+/).filter(Boolean)
  if (tokens.length >= 2 && configuredProvider(runtime, tokens[0]!)) {
    return { providerId: tokens[0]!, modelId: cleanModelTargetText(tokens.slice(1).join(' ')) }
  }

  const currentProviderId = currentModelTarget(runtime)?.providerId
  const providerId = explicitProviderId
    ?? (currentProviderId && configuredProvider(runtime, currentProviderId)?.type === 'ollama'
      ? currentProviderId
      : undefined)
    ?? runtime.config.providers.find((provider) => provider.type === 'ollama')?.id

  if (!providerId) {
    throw new Error('No Ollama provider is configured.')
  }
  return { providerId, modelId: query }
}

export async function pullOllamaModel(
  runtime: ModelControlRuntime,
  rawModel: string,
  options: { providerId?: string; timeoutMs?: number } = {},
): Promise<ModelPullResult> {
  const target = resolveOllamaPullTarget(runtime, rawModel, options.providerId)
  if (!target.modelId) {
    throw new Error('Model pull requires a model id.')
  }

  const providerConfig = configuredProvider(runtime, target.providerId)
  if (!providerConfig) {
    throw new Error(`Provider ${target.providerId} is not configured.`)
  }
  if (providerConfig.type !== 'ollama') {
    throw new Error(`Provider ${target.providerId} is ${providerConfig.type}, not ollama.`)
  }

  const alreadyAvailable = Boolean(
    matchModelInProvider(runtime.providerRegistry.get(target.providerId), target.modelId),
  )
  const alreadyConfigured = modelConfigured(providerConfig, target.modelId)
  const startedAt = Date.now()

  if (!alreadyAvailable) {
    const baseUrl = resolveConfigEnvReference(providerConfig.baseUrl) ?? 'http://localhost:11434'
    const url = new URL('/api/pull', baseUrl).toString()
    const timeout = timeoutSignal(options.timeoutMs ?? 120_000)
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          ...resolveHeaderReferences(providerConfig.headers),
          'content-type': 'application/json',
        },
        body: JSON.stringify({ name: target.modelId, stream: false }),
        signal: timeout.signal,
      })
      if (!response.ok) {
        const body = await response.text().catch(() => '')
        const detail = body ? ` — ${body.slice(0, 240)}` : ''
        throw new Error(`Ollama /api/pull responded ${response.status}${detail}`)
      }
    } finally {
      timeout.clear()
    }
  }

  if (!alreadyConfigured) {
    await runtime.configMutationService.apply('model.pull', async () => {
      const providers = runtime.config.providers.map((provider) =>
        provider.id === target.providerId ? addConfiguredModel(provider, target.modelId) : provider,
      )
      runtime.config.providers = providers
      await applyAndPersistRuntimeUpdate(
        runtime,
        new Set<ConfigUpdateKey>(['providers']),
        { providers },
      )
    })
  } else if (!alreadyAvailable) {
    try {
      await runtime.configMutationService.apply('model.pull.refresh', async () => {
        await applyAndPersistRuntimeUpdate(
          runtime,
          new Set<ConfigUpdateKey>(['providers']),
          { providers: runtime.config.providers },
        )
      })
    } catch (error) {
      log.warn('Pulled model but failed to refresh provider registry', {
        providerId: target.providerId,
        model: target.modelId,
        error: error instanceof Error ? error.message : String(error),
      })
      throw error
    }
  }

  const latencyMs = Date.now() - startedAt
  return {
    ok: true,
    providerId: target.providerId,
    model: target.modelId,
    alreadyAvailable,
    alreadyConfigured,
    latencyMs,
    message: alreadyAvailable
      ? `${target.providerId} / ${target.modelId} is already available.`
      : `Pulled ${target.providerId} / ${target.modelId} (${latencyMs}ms).`,
  }
}
