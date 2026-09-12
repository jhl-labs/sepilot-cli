import type {
  DaemonConfig,
  DaemonConfigEnvUpdateInput,
  DaemonConfigProvider,
  DaemonConfigUpdateInput,
} from '@sepilotd/api-client'
import {
  extractEnvVarReference,
  findProviderWizardPreset,
  type ProviderWizardPreset,
} from '../../utils/provider-presets.js'

export type ConfiguredProviderRecord =
  DaemonConfigProvider
  & Record<string, unknown>

type SupportedProviderType = DaemonConfigProvider['type']

const SUPPORTED_PROVIDER_TYPES = new Set<SupportedProviderType>([
  'openai',
  'anthropic',
  'ollama',
  'opencode',
  'codex',
  'gemini',
  'groq',
  'together',
  'deepseek',
  'custom',
])

export interface ProviderSetupDraft {
  sourceProviderId: string | null
  preset: ProviderWizardPreset
  providerId: string
  baseUrl: string
  apiKeyEnvVar: string
  headers?: Record<string, string>
  model: string
  models?: string[]
  /** Previous configured id when endpoint discovery uniquely reconciled an alias. */
  modelAliasSource?: string
}

export interface ProviderConfigSnapshot {
  providers: DaemonConfig['providers']
  agent?: Pick<DaemonConfig['agent'], 'defaultProvider' | 'defaultModel'>
}

export function isValidProviderEnvVarName(value: string): boolean {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(value.trim())
}

/**
 * Choose the daemon-managed environment variable used when the operator
 * enters an API key directly. The key value itself is deliberately not
 * interpreted or validated; only this generated storage reference must obey
 * environment-variable naming rules.
 */
export function managedProviderApiKeyEnvVar(options: {
  providerId: string
  preset: ProviderWizardPreset
  currentEnvVar?: string | null
}): string {
  const currentEnvVar = options.currentEnvVar?.trim()
  if (currentEnvVar && isValidProviderEnvVarName(currentEnvVar)) return currentEnvVar
  if (options.preset.apiKeyEnvVar) return options.preset.apiKeyEnvVar

  const providerId = options.providerId.trim()
  const readableId = providerId
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 48)
  const encodedId = Array.from(providerId)
    .map((character) => character.codePointAt(0)?.toString(16).toUpperCase() ?? '')
    .filter(Boolean)
    .join('_')
    .slice(0, 48)

  return `SEPILOT_${readableId || encodedId || 'CUSTOM'}_API_KEY`
}

export function findBuiltinProviderPreset(
  query: string | null | undefined,
): ProviderWizardPreset | null {
  return findProviderWizardPreset(query)
}

function appendOpenAiVersionPath(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, '')
  if (!trimmed) return ''
  return trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`
}

function defaultOpenAiCompatibleBaseUrl(
  preset: ProviderWizardPreset,
  env: Record<string, string | undefined>,
): string {
  return (
    env.OPENAI_COMPATIBLE_BASE_URL?.trim()
    || env.OPENAI_BASE_URL?.trim()
    || appendOpenAiVersionPath(env.OLLAMA_HOST ?? '')
    || preset.defaultBaseUrl
    || ''
  )
}

function normalizeModelList(models: unknown): string[] {
  return Array.isArray(models)
    ? models
      .map((value) => typeof value === 'string' ? value.trim() : '')
      .filter(Boolean)
    : []
}

function normalizedProviderModels(provider: ConfiguredProviderRecord | null | undefined): string[] {
  return normalizeModelList(provider?.models)
}

export function buildProviderSetupDraft(options: {
  provider?: ConfiguredProviderRecord | null
  preset?: ProviderWizardPreset | null
  currentProvider?: string
  currentModel?: string
  env?: Record<string, string | undefined>
}): ProviderSetupDraft {
  const env = options.env ?? process.env
  const preset = options.preset
    ?? findBuiltinProviderPreset(options.provider?.type)
    ?? findBuiltinProviderPreset(options.currentProvider)
    ?? findBuiltinProviderPreset('custom')

  if (!preset) {
    throw new Error('No supported provider preset is available.')
  }

  const provider = options.provider
  const currentModel = options.currentModel?.trim() ?? ''
  const currentModelForPreset = (
    options.currentProvider
    && (options.currentProvider === provider?.id || options.currentProvider === preset.type)
  )
    ? currentModel
    : ''
  const providerModels = normalizedProviderModels(provider)
  const providerModel = providerModels[0] ?? ''
  const defaultModel = (
    currentModelForPreset
    || providerModel
    || preset.suggestedModels?.[0]
    || (preset.type === 'ollama' ? 'llama3.3' : '')
  )
  const providerBaseUrl = typeof provider?.baseUrl === 'string' ? provider.baseUrl.trim() : ''

  return {
    sourceProviderId: provider?.id ? String(provider.id) : null,
    preset,
    providerId: provider?.id
      ? String(provider.id)
      : (preset.defaultProviderId ?? preset.type),
    baseUrl: preset.type === 'custom'
      ? (providerBaseUrl || defaultOpenAiCompatibleBaseUrl(preset, env))
      : preset.type === 'ollama'
      ? (
          providerBaseUrl
            ? providerBaseUrl
            : (env.OLLAMA_HOST ?? '')
        )
      : '',
    apiKeyEnvVar: preset.type === 'ollama' || preset.type === 'opencode' || preset.type === 'codex'
      ? ''
      : preset.type === 'custom'
      ? (
          extractEnvVarReference(
            typeof provider?.apiKey === 'string' ? provider.apiKey : undefined,
          )
          ?? ''
        )
      : (
          extractEnvVarReference(
            typeof provider?.apiKey === 'string' ? provider.apiKey : undefined,
          )
          ?? preset.apiKeyEnvVar
          ?? 'API_KEY'
        ),
    headers: provider?.headers && typeof provider.headers === 'object'
      ? { ...provider.headers }
      : {},
    model: defaultModel,
    models: providerModels,
  }
}

export function buildProviderModelSuggestions(
  draft: ProviderSetupDraft,
  sourceModels: string[] = [],
): string[] {
  const presetModels = draft.preset.suggestedModels ?? []

  return Array.from(new Set([
    draft.model,
    ...(draft.models ?? []),
    ...sourceModels,
    ...presetModels,
  ])).filter(Boolean)
}

export interface ReconciledProviderModelDiscovery {
  model: string
  modelSuggestions: string[]
  modelSuggestionIndex: number
  modelAliasSource?: string
}

function modelAliasParts(value: string): string[] {
  return value
    .trim()
    .toLowerCase()
    .split(/[:/_-]+/)
    .filter(Boolean)
}

function partsStartWith(candidate: string[], prefix: string[]): boolean {
  return prefix.length > 0
    && prefix.length <= candidate.length
    && prefix.every((part, index) => candidate[index] === part)
}

/**
 * Reconcile a model selected for one endpoint with the exact ids advertised by
 * another endpoint. Exact ids win. A separator-only alias or a unique prefix
 * may be upgraded to the advertised id, but ambiguous and unrelated matches
 * deliberately preserve the user's value for explicit correction.
 */
export function reconcileProviderModelDiscovery(options: {
  currentModel: string
  discoveredModels: string[]
  fallbackModels?: string[]
  defaultModel?: string
}): ReconciledProviderModelDiscovery {
  const currentModel = options.currentModel.trim()
  const discoveredModels = normalizeModelList(options.discoveredModels)
  const uniqueDiscoveredModels = Array.from(new Set(discoveredModels))

  if (uniqueDiscoveredModels.length > 0) {
    let model = uniqueDiscoveredModels.find((candidate) => candidate === currentModel) ?? ''
    const currentParts = modelAliasParts(currentModel)

    if (!model && currentParts.length > 0) {
      const canonicalMatches = uniqueDiscoveredModels.filter((candidate) => {
        const candidateParts = modelAliasParts(candidate)
        return candidateParts.length === currentParts.length
          && partsStartWith(candidateParts, currentParts)
      })
      if (canonicalMatches.length === 1) {
        model = canonicalMatches[0]!
      }
    }

    if (!model && currentParts.length > 0) {
      const prefixMatches = uniqueDiscoveredModels.filter((candidate) => (
        partsStartWith(modelAliasParts(candidate), currentParts)
      ))
      if (prefixMatches.length === 1) {
        model = prefixMatches[0]!
      }
    }

    model ||= currentModel || uniqueDiscoveredModels[0]!
    const selectedIndex = uniqueDiscoveredModels.indexOf(model)
    return {
      model,
      modelSuggestions: uniqueDiscoveredModels,
      modelSuggestionIndex: selectedIndex >= 0 ? selectedIndex : 0,
      ...(currentModel && model !== currentModel
        ? { modelAliasSource: currentModel }
        : {}),
    }
  }

  const fallbackModels = normalizeModelList(options.fallbackModels)
  const model = currentModel || fallbackModels[0] || options.defaultModel?.trim() || ''
  const modelSuggestions = Array.from(new Set([model, ...fallbackModels])).filter(Boolean)
  const selectedIndex = modelSuggestions.indexOf(model)
  return {
    model,
    modelSuggestions,
    modelSuggestionIndex: selectedIndex >= 0 ? selectedIndex : 0,
  }
}

export function buildProviderSecretEnvUpdate(options: {
  preset: ProviderWizardPreset
  apiKeyEnvVar: string
  apiKeyValue?: string | null
}): DaemonConfigEnvUpdateInput | null {
  if (
    options.preset.type === 'ollama'
    || options.preset.type === 'opencode'
    || options.preset.type === 'codex'
  ) {
    return null
  }

  const envVar = options.apiKeyEnvVar.trim()
  const secretValue = options.apiKeyValue ?? ''
  if (!envVar || !secretValue) {
    return null
  }

  return {
    updates: {
      [envVar]: secretValue,
    },
  }
}

function cleanConfiguredProviders(
  providers: DaemonConfig['providers'],
): ConfiguredProviderRecord[] {
  return Array.isArray(providers)
    ? providers.filter(
      (provider): provider is ConfiguredProviderRecord => (
        typeof provider === 'object'
        && provider !== null
        && !Array.isArray(provider)
      ),
    )
    : []
}

function resolveProviderDefaultModel(
  provider: ConfiguredProviderRecord,
  candidates: Array<string | null | undefined>,
): string {
  const models = normalizedProviderModels(provider)
  const selected = candidates.find((candidate) => (
    typeof candidate === 'string' && models.includes(candidate)
  ))
  if (selected) {
    return selected
  }
  if (models[0]) {
    return models[0]
  }
  throw new Error(`Provider ${provider.id} has no configured models.`)
}

function resolveSupportedProviderType(type: string): SupportedProviderType {
  if (SUPPORTED_PROVIDER_TYPES.has(type as SupportedProviderType)) {
    return type as SupportedProviderType
  }

  throw new Error(`Unsupported provider type: ${type}`)
}

export function buildProviderConfigRecord(options: {
  config: Pick<DaemonConfig, 'providers'>
  draft: ProviderSetupDraft
}): ConfiguredProviderRecord {
  const providers = cleanConfiguredProviders(options.config.providers)
  const removalId = options.draft.sourceProviderId ?? options.draft.providerId
  const providerType = resolveSupportedProviderType(options.draft.preset.type)
  const existingProvider = providers.find((provider) => provider.id === removalId) ?? null
  const preserveExistingFields = (
    existingProvider
    && existingProvider.type === providerType
  )
    ? existingProvider
    : null

  const nextProvider: ConfiguredProviderRecord = preserveExistingFields
    ? { ...preserveExistingFields, default: true }
    : {
        id: options.draft.providerId,
        type: providerType,
        models: [],
        default: true,
      }

  nextProvider.id = options.draft.providerId
  nextProvider.type = providerType
  const preserveExistingModelList = Boolean(
    preserveExistingFields
    && preserveExistingFields.id === options.draft.providerId,
  )
  const existingModels = preserveExistingModelList
    ? normalizedProviderModels(preserveExistingFields)
    : []
  nextProvider.models = Array.from(new Set(
    [
      options.draft.model,
      ...(options.draft.models ?? []),
      ...existingModels,
    ],
  )).filter((value): value is string => (
    typeof value === 'string' && value.trim().length > 0
  ))
  nextProvider.default = true

  if (options.draft.headers !== undefined) {
    nextProvider.headers = { ...options.draft.headers }
  }

  const aliasSource = options.draft.modelAliasSource?.trim()
  if (
    preserveExistingFields
    && aliasSource
    && aliasSource !== options.draft.model
  ) {
    const modelOverrides = Array.isArray(preserveExistingFields.modelOverrides)
      ? preserveExistingFields.modelOverrides
      : []
    const sourceOverride = modelOverrides.find((override) => override.id === aliasSource)
    const targetOverride = modelOverrides.find((override) => override.id === options.draft.model)
    if (sourceOverride && !targetOverride) {
      nextProvider.modelOverrides = [
        ...modelOverrides,
        { ...sourceOverride, id: options.draft.model },
      ]
    }
  }

  if (providerType === 'ollama') {
    nextProvider.baseUrl = options.draft.baseUrl
    delete nextProvider.apiKey
  } else if (providerType === 'custom') {
    nextProvider.baseUrl = options.draft.baseUrl
    nextProvider.apiKey = options.draft.apiKeyEnvVar
      ? `\${${options.draft.apiKeyEnvVar}}`
      : preserveExistingFields?.apiKey === '***redacted***'
      ? '***redacted***'
      : 'local'
  } else {
    nextProvider.apiKey = `\${${options.draft.apiKeyEnvVar}}`
    delete nextProvider.baseUrl
  }

  return nextProvider
}

export function buildProviderConfigUpdate(options: {
  config: Pick<DaemonConfig, 'providers'>
  draft: ProviderSetupDraft
}): DaemonConfigUpdateInput {
  const providers = cleanConfiguredProviders(options.config.providers)
  const removalId = options.draft.sourceProviderId ?? options.draft.providerId
  const nextProvider = buildProviderConfigRecord(options)

  const nextProviders: ConfiguredProviderRecord[] = providers
    .filter((provider) => provider.id !== removalId)
    .map((provider) => ({ ...provider, default: false }))
  nextProviders.push(nextProvider)

  const update: DaemonConfigUpdateInput = {
    providers: nextProviders,
    'agent.defaultProvider': options.draft.providerId,
    'agent.defaultModel': options.draft.model,
  }
  if (providers.length === 0) {
    update['agent.mode'] = 'react'
  }
  return update
}

export function buildProviderDefaultUpdate(options: {
  config: ProviderConfigSnapshot
  providerId: string
  preferredModel?: string | null
}): DaemonConfigUpdateInput {
  const providers = cleanConfiguredProviders(options.config.providers)
  const providerId = options.providerId.trim()
  const targetProvider = providers.find((provider) => provider.id === providerId)
  if (!targetProvider) {
    throw new Error(`Provider ${providerId} is not configured.`)
  }

  const nextModel = resolveProviderDefaultModel(targetProvider, [
    options.preferredModel,
    options.config.agent?.defaultProvider === providerId
      ? options.config.agent?.defaultModel
      : null,
  ])

  return {
    providers: providers.map((provider) => ({
      ...provider,
      default: provider.id === providerId,
    })),
    'agent.defaultProvider': providerId,
    'agent.defaultModel': nextModel,
  }
}

export function buildProviderDeleteUpdate(options: {
  config: ProviderConfigSnapshot
  providerId: string
  preferredFallbackProviderId?: string | null
  preferredFallbackModel?: string | null
}): DaemonConfigUpdateInput {
  const providers = cleanConfiguredProviders(options.config.providers)
  const providerId = options.providerId.trim()
  const remainingProviders = providers.filter((provider) => provider.id !== providerId)

  if (remainingProviders.length === providers.length) {
    throw new Error(`Provider ${providerId} is not configured.`)
  }
  if (remainingProviders.length === 0) {
    throw new Error('Cannot delete the last configured provider.')
  }

  const preferredFallback = options.preferredFallbackProviderId
    ? remainingProviders.find((provider) => provider.id === options.preferredFallbackProviderId)
    : null
  const existingDefault = (
    options.config.agent?.defaultProvider
    && options.config.agent.defaultProvider !== providerId
  )
    ? remainingProviders.find((provider) => provider.id === options.config.agent?.defaultProvider)
    : null
  const nextProvider = preferredFallback ?? existingDefault ?? remainingProviders[0]!
  const nextModel = resolveProviderDefaultModel(nextProvider, [
    nextProvider.id === options.preferredFallbackProviderId
      ? options.preferredFallbackModel
      : null,
    nextProvider.id === options.config.agent?.defaultProvider
      ? options.config.agent?.defaultModel
      : null,
  ])

  return {
    providers: remainingProviders.map((provider) => ({
      ...provider,
      default: provider.id === nextProvider.id,
    })),
    'agent.defaultProvider': nextProvider.id,
    'agent.defaultModel': nextModel,
  }
}

/**
 * Build the SYSTEM_MESSAGE body for a successful provider save. The
 * message has three independent axes:
 *
 *   - mode: 'edit' (Updated provider X) vs 'new' (Added provider X)
 *   - secretUpdate: whether the daemon stored a fresh API key in
 *     the managed .env file
 *   - applyToCurrentSession: whether the operator opted to switch
 *     this session onto the new provider/model immediately, in
 *     which case the message also reports whether the session
 *     was already on the same provider/model pair.
 *
 * Pulled out of saveProviderSetup so each branch is unit-testable
 * without mocking out the entire wizard's HTTP/state choreography.
 */
export function buildProviderSetupSuccessMessage(opts: {
  mode: 'edit' | 'new'
  draft: Pick<ProviderSetupDraft, 'providerId' | 'apiKeyEnvVar' | 'model'>
  validationLatencyMs: number
  secretUpdated: boolean
  defaultModeChangedTo?: string | null
  applyToCurrentSession: boolean
  sessionAlreadyMatches: boolean
  currentSessionProvider: string
  currentSessionModel: string
}): string {
  const verb = opts.mode === 'edit' ? 'Updated' : 'Added'
  const head =
    `${verb} provider ${opts.draft.providerId} and validated ${opts.draft.model} in ${opts.validationLatencyMs}ms.`
  const secretLine = opts.secretUpdated
    ? `Stored ${opts.draft.apiKeyEnvVar} in the daemon-managed .env file.`
    : null
  const modeLine = opts.defaultModeChangedTo
    ? `Default mode set to ${opts.defaultModeChangedTo}. Use /mode auto when you want graph routing.`
    : null

  if (opts.applyToCurrentSession) {
    const tail = opts.sessionAlreadyMatches
      ? `Daemon default and current session already use ${opts.draft.providerId}/${opts.draft.model}.`
      : `Daemon default saved and current session switched to ${opts.draft.providerId}/${opts.draft.model}.`
    return [
      head,
      secretLine,
      modeLine,
      tail,
      'Use Ctrl+T to review models or /provider current for details.',
    ].filter((line): line is string => line !== null).join('\n')
  }

  return [
    head,
    secretLine,
    modeLine,
    `Daemon default now uses ${opts.draft.providerId}/${opts.draft.model}. Current session stays on ${opts.currentSessionProvider}/${opts.currentSessionModel}.`,
    'Run /model default apply or use Ctrl+T if you want to switch this session too.',
  ].filter((line): line is string => line !== null).join('\n')
}

/**
 * Decide which provider/model the operator's current session should
 * be promoted to as the daemon default after `deletedProviderId` is
 * removed. The rule is "only override the daemon's automatic
 * fallback when the user is deleting the daemon default *and* their
 * current session has already moved off it" — preserving the user's
 * implicit choice instead of letting the deletion silently re-route
 * the daemon to a provider they walked away from. Pulled out so the
 * delete-prepare and delete-confirm paths in App.tsx don't drift.
 */
export function derivePreferredProviderFallback(opts: {
  deletedProviderId: string
  daemonDefaultProviderId: string | null
  sessionProvider: string
  sessionModel: string
}): { providerId: string | null; model: string | null } {
  const isDeletingDaemonDefault = opts.daemonDefaultProviderId === opts.deletedProviderId
  const sessionOffDeleted = opts.sessionProvider !== opts.deletedProviderId
  if (isDeletingDaemonDefault && sessionOffDeleted) {
    return { providerId: opts.sessionProvider, model: opts.sessionModel }
  }
  return { providerId: null, model: null }
}
