import type {
  DaemonConfigProvider,
  DaemonProviderDiscoverModelsInput,
  DaemonProviderDiscoverModelsResult,
  DaemonProviderValidationInput,
  DaemonProviderValidationResult,
} from '@sepilotd/api-client'
import type { ProviderWizardPreset } from './provider-presets.js'

export interface ProviderSetupIO {
  log(message: string): void
}

export interface ProviderSetupClient {
  validateProvider(input: DaemonProviderValidationInput): Promise<DaemonProviderValidationResult>
  discoverProviderModels(
    input: DaemonProviderDiscoverModelsInput,
  ): Promise<DaemonProviderDiscoverModelsResult>
}

export interface ProviderSetupInput {
  id?: string
  type: DaemonConfigProvider['type']
  apiKey?: string
  baseUrl?: string
  headers?: Record<string, string>
  model?: string
  timeoutMs?: number
  env?: Record<string, string>
}

export interface ProviderSetupSuccess {
  ok: true
  provider: DaemonConfigProvider
  models: string[]
  defaultTarget?: string
  discoveryWarning?: string
}

export interface ProviderSetupFailure {
  ok: false
  reason: string
}

export type ProviderSetupResult = ProviderSetupSuccess | ProviderSetupFailure

function extractErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message
  }
  return String(error)
}

function buildProvider(
  preset: ProviderWizardPreset,
  input: ProviderSetupInput,
  models: string[],
  capabilities?: DaemonConfigProvider['capabilities'],
): DaemonConfigProvider {
  return {
    id: input.id?.trim() || preset.defaultProviderId || preset.type,
    type: input.type,
    apiKey: input.apiKey,
    baseUrl: input.baseUrl,
    headers: input.headers,
    models,
    ...(capabilities ? { capabilities } : {}),
  }
}

/**
 * Shared provider setup flow: preset/custom input -> discover-models -> validate.
 *
 * Discovery runs first because compatible endpoints can advertise transport-level
 * capabilities that must be applied to the validation request itself. On validation
 * failure, nothing is saved and the caller gets `{ ok: false, reason }`. If discovery
 * fails but validation succeeds, the provider is returned with an empty model list and
 * a `discoveryWarning` so the caller can prompt for a manual model name.
 */
export async function runProviderSetupFlow(
  io: ProviderSetupIO,
  client: ProviderSetupClient,
  preset: ProviderWizardPreset,
  input: ProviderSetupInput,
): Promise<ProviderSetupResult> {
  let discoveredModels: string[] = []
  let discoveredCapabilities: DaemonConfigProvider['capabilities'] | undefined
  let discoveryWarning: string | undefined
  try {
    const discovered = await client.discoverProviderModels({
      type: input.type,
      baseUrl: input.baseUrl,
      apiKey: input.apiKey,
      headers: input.headers,
      timeoutMs: input.timeoutMs,
    })
    discoveredModels = discovered.models
    discoveredCapabilities = discovered.compatibility?.capabilities
  } catch (error) {
    const message = extractErrorMessage(error)
    discoveryWarning = `Model discovery failed: ${message}. Enter a model name manually.`
    io.log(discoveryWarning)
  }

  const initialModels = input.model ? [input.model] : (preset.suggestedModels ?? [])
  const validationModels = discoveredModels.length > 0 ? discoveredModels : initialModels
  const candidateProvider = buildProvider(
    preset,
    input,
    validationModels,
    discoveredCapabilities,
  )
  try {
    await client.validateProvider({
      provider: candidateProvider,
      model: input.model,
      timeoutMs: input.timeoutMs,
      env: input.env,
    })
  } catch (error) {
    return { ok: false, reason: extractErrorMessage(error) }
  }

  const models = discoveredModels
  const provider = buildProvider(preset, input, models, discoveredCapabilities)

  return {
    ok: true,
    provider,
    models,
    defaultTarget: input.model ?? models[0],
    discoveryWarning,
  }
}
