import type { DaemonProviderInfo } from '@sepilotd/api-client'
import {
  PROVIDER_WIZARD_PRESETS,
  type ProviderWizardPreset,
} from '../../utils/provider-presets.js'

export interface ProviderModelOption {
  kind: 'model'
  providerId: string
  providerName: string
  modelId: string
  modelName: string
  provider: DaemonProviderInfo | null
  model: DaemonProviderInfo['models'][number] | null
  isCurrent: boolean
  isDefault: boolean
  healthStatus: DaemonProviderInfo['health']['status']
  healthMessage?: string
  missingEnvVars?: string[]
  shortcut?: number
  synthetic?: boolean
}

interface ProviderPickerActionOption {
  kind: 'action'
  action: 'setup' | 'setup-preset' | 'edit-provider' | 'set-default-provider' | 'delete-provider' | 'sync-session-default'
  label: string
  detail: string
  providerId?: string
  modelId?: string
  presetType?: ProviderWizardPreset['type']
  isCurrentProvider?: boolean
  isDefaultProvider?: boolean
  healthStatus?: DaemonProviderInfo['health']['status']
  healthMessage?: string
  missingEnvVars?: string[]
  shortcut?: number
}

export type ProviderModelPickerItem =
  | ProviderModelOption
  | ProviderPickerActionOption

interface ProviderMatchResult {
  match: DaemonProviderInfo | null
  ambiguousMatches: DaemonProviderInfo[]
}

interface ModelMatchResult {
  match: ProviderModelOption | null
  ambiguousMatches: ProviderModelOption[]
}

const HEALTH_RANK: Record<string, number> = {
  ready: 0,
  env_missing: 1,
  unavailable: 2,
}

function healthRank(status: string | undefined): number {
  return status !== undefined && status in HEALTH_RANK ? HEALTH_RANK[status]! : 3
}

const NOT_IN_MRU_RANK = Number.MAX_SAFE_INTEGER

function mruRank(option: ProviderModelOption, mru: string[]): number {
  if (mru.length === 0) return NOT_IN_MRU_RANK
  const target = `${option.providerId}/${option.modelId}`.toLowerCase()
  const index = mru.findIndex((entry) => entry.toLowerCase() === target)
  return index === -1 ? NOT_IN_MRU_RANK : index
}

function compareOption(
  left: ProviderModelOption,
  right: ProviderModelOption,
  mru: string[] = [],
): number {
  const mruCompare = mruRank(left, mru) - mruRank(right, mru)
  if (mruCompare !== 0) {
    return mruCompare
  }

  if (left.isCurrent !== right.isCurrent) {
    return left.isCurrent ? -1 : 1
  }

  const healthCompare = healthRank(left.healthStatus) - healthRank(right.healthStatus)
  if (healthCompare !== 0) {
    return healthCompare
  }

  const providerCompare = left.providerId.localeCompare(right.providerId)
  if (providerCompare !== 0) {
    return providerCompare
  }

  return left.modelId.localeCompare(right.modelId)
}

function flattenProviderModels(
  providers: DaemonProviderInfo[],
  currentProviderId: string,
  currentModelId: string,
  defaultProviderId: string,
  defaultModelId: string,
  mru: string[] = [],
): ProviderModelOption[] {
  const options: ProviderModelOption[] = providers.flatMap((provider) => (
    provider.models.map((model) => ({
      kind: 'model' as const,
      providerId: provider.id,
      providerName: provider.name,
      modelId: model.id,
      modelName: model.name,
      provider,
      model,
      isCurrent: provider.id === currentProviderId && model.id === currentModelId,
      isDefault: provider.id === defaultProviderId && model.id === defaultModelId,
      healthStatus: provider.health.status,
      healthMessage: provider.health.message,
      missingEnvVars: provider.health.missingEnvVars,
    }))
  ))

  const hasCurrent = options.some((option) => option.isCurrent)
  if (
    !hasCurrent
    && currentProviderId
    && currentModelId
    && currentProviderId !== 'default'
    && currentModelId !== 'default'
  ) {
    options.unshift({
      kind: 'model',
      providerId: currentProviderId,
      providerName: currentProviderId,
      modelId: currentModelId,
      modelName: currentModelId,
      provider: null,
      model: null,
      isCurrent: true,
      isDefault: currentProviderId === defaultProviderId && currentModelId === defaultModelId,
      healthStatus: 'unavailable',
      healthMessage: 'Current session model is not configured on the daemon.',
      missingEnvVars: [],
      synthetic: true,
    })
  }

  return options.sort((left, right) => compareOption(left, right, mru))
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

function matchesNormalizedQuery(values: string[], normalizedQuery: string): boolean {
  if (!normalizedQuery) return true
  return values.some((value) => value.toLowerCase().includes(normalizedQuery))
}

function matchesQuery(option: ProviderModelOption, query: string): boolean {
  const normalizedQuery = query.trim().toLowerCase()
  if (!normalizedQuery) return true
  if (matchesNormalizedQuery([
    option.providerId,
    option.providerName,
    option.modelId,
    option.modelName,
    `${option.providerId}/${option.modelId}`,
  ], normalizedQuery)) {
    return true
  }
  if (normalizedQuery.length >= MIN_FUZZY_QUERY_LENGTH) {
    return isSubsequenceMatch(normalizedQuery, `${option.providerId}/${option.modelId}`.toLowerCase())
  }
  return false
}

function buildPresetActionDetail(preset: ProviderWizardPreset): string {
  if (preset.type === 'ollama') {
    return 'Local Ollama provider'
  }
  if (preset.type === 'custom') {
    return `${preset.defaultBaseUrl ?? 'OpenAI-compatible /v1 endpoint'} | local or gateway`
  }

  const suggestions = preset.suggestedModels?.slice(0, 2).join(', ')
  return `${preset.apiKeyEnvVar ?? 'API_KEY'}${suggestions ? ` | ${suggestions}` : ''}`
}

function buildProviderManagementActions(
  providers: DaemonProviderInfo[],
  currentProviderId: string,
  currentModelId: string,
  defaultProviderId: string,
  defaultModelId: string,
  query: string,
  matchedModelCount: number,
): ProviderPickerActionOption[] {
  const normalizedQuery = query.trim().toLowerCase()
  const currentProvider = providers.find((provider) => provider.id === currentProviderId) ?? null
  const defaultProvider = providers.find((provider) => provider.id === defaultProviderId) ?? null
  const actions: ProviderPickerActionOption[] = []
  const seen = new Set<string>()
  const currentProviderHealthIssue = currentProvider?.health.status && currentProvider.health.status !== 'ready'
  const defaultProviderHealthIssue = defaultProvider?.health.status && defaultProvider.health.status !== 'ready'

  const pushAction = (action: ProviderPickerActionOption) => {
    const key = `${action.action}:${action.providerId ?? action.presetType ?? ''}`
    if (seen.has(key)) {
      return
    }
    seen.add(key)
    actions.push(action)
  }

  const shouldShowConfigure = (
    !normalizedQuery
    || matchedModelCount === 0
    || matchesNormalizedQuery(
      ['configure providers', 'provider setup', 'manage providers', 'add provider'],
      normalizedQuery,
    )
  )

  if (shouldShowConfigure) {
    pushAction({
      kind: 'action',
      action: 'setup',
      label: 'Configure providers...',
      detail: 'Add, edit, or replace daemon providers.',
    })
  }

  if (
    defaultProvider
    && defaultModelId
    && defaultProvider.health.status === 'ready'
    && defaultProvider.models.some((model) => model.id === defaultModelId)
    && (currentProviderId !== defaultProviderId || currentModelId !== defaultModelId)
    && (
      !normalizedQuery
      || matchesNormalizedQuery(
        [
          'apply daemon default',
          'sync default',
          'switch current session to daemon default',
          `${defaultProviderId}/${defaultModelId}`,
        ],
        normalizedQuery,
      )
    )
  ) {
    pushAction({
      kind: 'action',
      action: 'sync-session-default',
      providerId: defaultProviderId,
      modelId: defaultModelId,
      label: 'Switch current session to daemon default',
      detail: `${defaultProviderId}/${defaultModelId}`,
      isDefaultProvider: true,
      healthStatus: defaultProvider.health.status,
      healthMessage: defaultProvider.health.message,
      missingEnvVars: defaultProvider.health.missingEnvVars,
    })
  }

  if (
    currentProvider
    && (
      !normalizedQuery
      || matchesNormalizedQuery(
        [
          currentProvider.id,
          currentProvider.name,
          'edit current provider',
          `edit ${currentProvider.id}`,
        ],
        normalizedQuery,
      )
    )
  ) {
    pushAction({
      kind: 'action',
      action: 'edit-provider',
      providerId: currentProvider.id,
      label: currentProviderHealthIssue ? 'Repair current provider' : 'Edit current provider',
      detail: currentProviderHealthIssue
        ? (currentProvider.health.message ?? `${currentProvider.name} (${currentProvider.id})`)
        : `${currentProvider.name} (${currentProvider.id})`,
      isCurrentProvider: true,
      isDefaultProvider: currentProvider.id === defaultProviderId,
      healthStatus: currentProvider.health.status,
      healthMessage: currentProvider.health.message,
      missingEnvVars: currentProvider.health.missingEnvVars,
    })
  }

  if (currentProvider && currentProvider.id !== defaultProviderId) {
    const nextModel = currentProvider.models.find((model) => model.id === currentModelId)?.id
      ?? currentProvider.models[0]?.id
    if (nextModel) {
      pushAction({
        kind: 'action',
        action: 'set-default-provider',
        providerId: currentProvider.id,
        modelId: nextModel,
        label: 'Set current provider as daemon default',
        detail: `${currentProvider.id}/${nextModel}`,
        isCurrentProvider: true,
        healthStatus: currentProvider.health.status,
        healthMessage: currentProvider.health.message,
        missingEnvVars: currentProvider.health.missingEnvVars,
      })
    }
  }

  if (currentProvider && providers.length > 1) {
    pushAction({
      kind: 'action',
      action: 'delete-provider',
      providerId: currentProvider.id,
      label: 'Delete current provider',
      detail: `Remove ${currentProvider.id} from daemon config`,
      isCurrentProvider: true,
      isDefaultProvider: currentProvider.id === defaultProviderId,
      healthStatus: currentProvider.health.status,
      healthMessage: currentProvider.health.message,
      missingEnvVars: currentProvider.health.missingEnvVars,
    })
  }

  if (!normalizedQuery && matchedModelCount === 0) {
    PROVIDER_WIZARD_PRESETS.forEach((preset) => {
      pushAction({
        kind: 'action',
        action: 'setup-preset',
        presetType: preset.type,
        label: `Add ${preset.label} provider`,
        detail: buildPresetActionDetail(preset),
      })
    })
    return actions
  }

  if (normalizedQuery) {
    providers.forEach((provider) => {
      if (!matchesNormalizedQuery([
        provider.id,
        provider.name,
        `edit ${provider.id}`,
        `edit ${provider.name}`,
      ], normalizedQuery)) {
        return
      }
      pushAction({
        kind: 'action',
        action: 'edit-provider',
        providerId: provider.id,
        label: provider.health.status === 'ready'
          ? `Edit ${provider.name} provider`
          : `Repair ${provider.name} provider`,
        detail: provider.health.status === 'ready'
          ? (
              provider.id === provider.name
                ? provider.id
                : `${provider.name} (${provider.id})`
            )
          : (provider.health.message ?? provider.id),
        isCurrentProvider: provider.id === currentProviderId,
        isDefaultProvider: provider.id === defaultProviderId,
        healthStatus: provider.health.status,
        healthMessage: provider.health.message,
        missingEnvVars: provider.health.missingEnvVars,
      })

      if (
        provider.id === defaultProviderId
        && defaultProviderHealthIssue
        && provider.id !== currentProviderId
      ) {
        pushAction({
          kind: 'action',
          action: 'edit-provider',
          providerId: provider.id,
          label: 'Repair default provider',
          detail: provider.health.message ?? provider.id,
          isDefaultProvider: true,
          healthStatus: provider.health.status,
          healthMessage: provider.health.message,
          missingEnvVars: provider.health.missingEnvVars,
        })
      }

      const defaultModel = provider.id === currentProviderId
        ? provider.models.find((model) => model.id === currentModelId)?.id
          ?? provider.models[0]?.id
        : provider.id === defaultProviderId
          ? provider.models.find((model) => model.id === defaultModelId)?.id
            ?? provider.models[0]?.id
          : provider.models[0]?.id
      if (provider.id !== defaultProviderId && defaultModel) {
        pushAction({
          kind: 'action',
          action: 'set-default-provider',
          providerId: provider.id,
          modelId: defaultModel,
          label: `Set ${provider.name} as daemon default`,
            detail: `${provider.id}/${defaultModel}`,
            isCurrentProvider: provider.id === currentProviderId,
            healthStatus: provider.health.status,
            healthMessage: provider.health.message,
            missingEnvVars: provider.health.missingEnvVars,
          })
      }
      if (providers.length > 1) {
        pushAction({
          kind: 'action',
          action: 'delete-provider',
          providerId: provider.id,
          label: `Delete ${provider.name} provider`,
          detail: `Remove ${provider.id} from daemon config`,
          isCurrentProvider: provider.id === currentProviderId,
          isDefaultProvider: provider.id === defaultProviderId,
          healthStatus: provider.health.status,
          healthMessage: provider.health.message,
          missingEnvVars: provider.health.missingEnvVars,
        })
      }
    })

    PROVIDER_WIZARD_PRESETS.forEach((preset) => {
      if (!matchesNormalizedQuery([
        preset.type,
        preset.label,
        preset.defaultProviderId ?? '',
        preset.type === 'custom' ? 'compatible' : '',
        preset.type === 'custom' ? 'setup compatible' : '',
        `add ${preset.label} provider`,
        `setup ${preset.label}`,
        `setup ${preset.type}`,
        preset.defaultProviderId ? `setup ${preset.defaultProviderId}` : '',
      ], normalizedQuery)) {
        return
      }
      pushAction({
        kind: 'action',
        action: 'setup-preset',
        presetType: preset.type,
        label: `Add ${preset.label} provider`,
        detail: buildPresetActionDetail(preset),
      })
    })
  }

  return actions
}

export function buildProviderModelPickerList(
  providers: DaemonProviderInfo[],
  currentProviderId: string,
  currentModelId: string,
  query: string,
  options: {
    defaultProviderId?: string | null
    defaultModelId?: string | null
    mru?: string[]
  } = {},
): { ordered: ProviderModelPickerItem[] } {
  const trimmedQuery = query.trim().toLowerCase()
  const defaultProviderId = options.defaultProviderId?.trim() ?? ''
  const defaultModelId = options.defaultModelId?.trim() ?? ''
  const mru = options.mru ?? []
  const models = flattenProviderModels(
    providers,
    currentProviderId,
    currentModelId,
    defaultProviderId,
    defaultModelId,
    mru,
  )
    .filter((option) => matchesQuery(option, trimmedQuery))
  const actions = buildProviderManagementActions(
    providers,
    currentProviderId,
    currentModelId,
    defaultProviderId,
    defaultModelId,
    query,
    models.length,
  )
  const orderedItems = models.length > 0
    ? [...models, ...actions]
    : actions

  return {
    ordered: orderedItems.map((item, index) => ({
      ...item,
      shortcut: !trimmedQuery && index < 9 ? index + 1 : undefined,
    })),
  }
}

export function findProviderMatch(
  providers: DaemonProviderInfo[],
  query: string,
): ProviderMatchResult {
  const normalizedQuery = query.trim().toLowerCase()
  if (!normalizedQuery) {
    return { match: null, ambiguousMatches: [] }
  }

  const exactMatch = providers.find((provider) => (
    provider.id.toLowerCase() === normalizedQuery
    || provider.name.toLowerCase() === normalizedQuery
  ))
  if (exactMatch) {
    return { match: exactMatch, ambiguousMatches: [] }
  }

  const partialMatches = providers.filter((provider) => (
    provider.id.toLowerCase().includes(normalizedQuery)
    || provider.name.toLowerCase().includes(normalizedQuery)
  ))

  return partialMatches.length === 1
    ? { match: partialMatches[0] ?? null, ambiguousMatches: [] }
    : { match: null, ambiguousMatches: partialMatches }
}

export function findModelMatch(
  providers: DaemonProviderInfo[],
  currentProviderId: string,
  currentModelId: string,
  query: string,
  options: {
    defaultProviderId?: string | null
    defaultModelId?: string | null
  } = {},
): ModelMatchResult {
  const normalizedQuery = query.trim().toLowerCase()
  if (!normalizedQuery) {
    return { match: null, ambiguousMatches: [] }
  }

  const defaultProviderId = options.defaultProviderId?.trim() ?? ''
  const defaultModelId = options.defaultModelId?.trim() ?? ''
  const modelOptions = flattenProviderModels(
    providers,
    currentProviderId,
    currentModelId,
    defaultProviderId,
    defaultModelId,
  )
    .filter((option) => !option.synthetic)

  const exactFullMatch = modelOptions.find((option) => (
    `${option.providerId}/${option.modelId}`.toLowerCase() === normalizedQuery
  ))
  if (exactFullMatch) {
    return { match: exactFullMatch, ambiguousMatches: [] }
  }

  const exactCurrentProviderMatch = modelOptions.find((option) => (
    option.providerId === currentProviderId
    && (
      option.modelId.toLowerCase() === normalizedQuery
      || option.modelName.toLowerCase() === normalizedQuery
    )
  ))
  if (exactCurrentProviderMatch) {
    return { match: exactCurrentProviderMatch, ambiguousMatches: [] }
  }

  const exactMatches = modelOptions.filter((option) => (
    option.modelId.toLowerCase() === normalizedQuery
    || option.modelName.toLowerCase() === normalizedQuery
  ))
  if (exactMatches.length === 1) {
    return { match: exactMatches[0] ?? null, ambiguousMatches: [] }
  }
  if (exactMatches.length > 1) {
    return { match: null, ambiguousMatches: exactMatches }
  }

  const partialCurrentProviderMatches = modelOptions.filter((option) => (
    option.providerId === currentProviderId
    && matchesQuery(option, normalizedQuery)
  ))
  if (partialCurrentProviderMatches.length === 1) {
    return { match: partialCurrentProviderMatches[0] ?? null, ambiguousMatches: [] }
  }

  const partialMatches = modelOptions.filter((option) => matchesQuery(option, normalizedQuery))
  return partialMatches.length === 1
    ? { match: partialMatches[0] ?? null, ambiguousMatches: [] }
    : { match: null, ambiguousMatches: partialMatches }
}
