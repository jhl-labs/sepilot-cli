import type { DaemonConfigProvider } from '@sepilotd/api-client'

/**
 * Per-model capability flags that operators may toggle from the TUI.
 * Mirrors the daemon's providerModelCapabilitiesOverrideSchema subset that is
 * meant as an operator judgement (weak-model support), not a hardware fact.
 */
export const TOGGLEABLE_MODEL_CAPABILITIES = [
  'adaptivePromptReact',
  'promptReactPreferred',
  'deepCoderAnalysis',
] as const

export type ToggleableModelCapability = (typeof TOGGLEABLE_MODEL_CAPABILITIES)[number]

const CAPABILITY_ALIASES: Record<string, ToggleableModelCapability> = {
  adaptivepromptreact: 'adaptivePromptReact',
  'adaptive-prompt-react': 'adaptivePromptReact',
  adaptive: 'adaptivePromptReact',
  promptreactpreferred: 'promptReactPreferred',
  'prompt-react-preferred': 'promptReactPreferred',
  'prompt-react': 'promptReactPreferred',
  'portable-tools': 'promptReactPreferred',
  deepcoderanalysis: 'deepCoderAnalysis',
  'deep-coder-analysis': 'deepCoderAnalysis',
  deep: 'deepCoderAnalysis',
  'deep-analysis': 'deepCoderAnalysis',
}

export function resolveCapabilityName(input: string): ToggleableModelCapability | null {
  return CAPABILITY_ALIASES[input.trim().toLowerCase()] ?? null
}

export function parseCapabilityToggle(input: string): boolean | null {
  const normalized = input.trim().toLowerCase()
  if (['on', 'true', '1', 'enable', 'enabled'].includes(normalized)) return true
  if (['off', 'false', '0', 'disable', 'disabled'].includes(normalized)) return false
  return null
}

interface ProviderModelOverrideRecord {
  id: string
  capabilities?: Record<string, unknown>
  [key: string]: unknown
}

function readModelOverrides(provider: DaemonConfigProvider): ProviderModelOverrideRecord[] {
  const overrides = provider.modelOverrides
  return Array.isArray(overrides) ? (overrides as ProviderModelOverrideRecord[]) : []
}

export interface ModelCapabilityView {
  capability: ToggleableModelCapability
  /** Effective value: model override, else provider capabilities, else off. */
  enabled: boolean
  source: 'model-override' | 'provider' | 'default'
}

export function describeModelCapabilities(
  providers: DaemonConfigProvider[],
  providerId: string,
  modelId: string,
): ModelCapabilityView[] | null {
  const provider = providers.find((entry) => entry.id === providerId)
  if (!provider) return null
  const override = readModelOverrides(provider).find((entry) => entry.id === modelId)
  const providerCapabilities = (provider.capabilities ?? {}) as Record<string, unknown>
  return TOGGLEABLE_MODEL_CAPABILITIES.map((capability) => {
    const fromOverride = override?.capabilities?.[capability]
    if (typeof fromOverride === 'boolean') {
      return { capability, enabled: fromOverride, source: 'model-override' }
    }
    const fromProvider = providerCapabilities[capability]
    if (typeof fromProvider === 'boolean') {
      return { capability, enabled: fromProvider, source: 'provider' }
    }
    return { capability, enabled: false, source: 'default' }
  })
}

export interface ApplyCapabilityResult {
  providers: DaemonConfigProvider[]
  changed: boolean
}

/**
 * Returns a new providers array with the capability set on the model's
 * modelOverrides entry (created if missing). Does not mutate the input.
 */
export function applyModelCapabilityOverride(
  providers: DaemonConfigProvider[],
  providerId: string,
  modelId: string,
  capability: ToggleableModelCapability,
  enabled: boolean,
): ApplyCapabilityResult | null {
  const providerIndex = providers.findIndex((entry) => entry.id === providerId)
  if (providerIndex === -1) return null

  const provider = providers[providerIndex]
  const overrides = readModelOverrides(provider)
  const overrideIndex = overrides.findIndex((entry) => entry.id === modelId)
  const existing = overrideIndex === -1 ? undefined : overrides[overrideIndex]

  if (existing?.capabilities?.[capability] === enabled) {
    return { providers, changed: false }
  }

  const nextOverride: ProviderModelOverrideRecord = {
    ...(existing ?? { id: modelId }),
    capabilities: {
      ...(existing?.capabilities ?? {}),
      [capability]: enabled,
    },
  }
  const nextOverrides =
    overrideIndex === -1
      ? [...overrides, nextOverride]
      : overrides.map((entry, index) => (index === overrideIndex ? nextOverride : entry))

  const nextProviders = providers.map((entry, index) =>
    index === providerIndex ? { ...entry, modelOverrides: nextOverrides } : entry,
  )
  return { providers: nextProviders, changed: true }
}
