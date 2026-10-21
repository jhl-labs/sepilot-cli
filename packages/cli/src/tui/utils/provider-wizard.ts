import { DEFAULT_OPENAI_COMPATIBLE_BASE_URL } from '@sepilotd/api-client'
import type { ProviderWizardPreset } from '../../utils/provider-presets.js'
import { parseProviderHeadersInput } from '../../utils/provider-http-options.js'
import { isValidProviderEnvVarName, managedProviderApiKeyEnvVar } from './provider-setup.js'

export type ProviderSetupStep = 'preset' | 'connection' | 'model' | 'confirm'

export const PROVIDER_SETUP_STEP_ORDER: ProviderSetupStep[] = [
  'preset',
  'connection',
  'model',
  'confirm',
]

export const PROVIDER_SETUP_STEP_LABELS: Record<ProviderSetupStep, string> = {
  preset: 'Provider',
  connection: 'Connection',
  model: 'Model',
  confirm: 'Review',
}

export type ProviderConnectionFieldKey =
  | 'providerId'
  | 'baseUrl'
  | 'apiKeyValue'
  | 'apiKeyEnvVar'
  | 'headers'

export interface ProviderConnectionField {
  key: ProviderConnectionFieldKey
  label: string
  optional: boolean
  secret: boolean
  placeholder: string
  hint: string
}

export interface ProviderConnectionValues {
  providerId: string
  baseUrl: string
  apiKeyEnvVar: string
  apiKeyValue: string
  headersText: string
}

type PresetAuthKind = 'none' | 'optional' | 'required'

function presetAuthKind(preset: ProviderWizardPreset): PresetAuthKind {
  switch (preset.type) {
    case 'ollama':
    case 'opencode':
    case 'codex':
      return 'none'
    case 'custom':
      return 'optional'
    default:
      return 'required'
  }
}

export function presetUsesBaseUrl(preset: ProviderWizardPreset | null | undefined): boolean {
  return preset?.type === 'ollama' || preset?.type === 'custom'
}

export function presetUsesApiKey(preset: ProviderWizardPreset | null | undefined): boolean {
  return preset ? presetAuthKind(preset) !== 'none' : false
}

export function describeProviderPreset(preset: ProviderWizardPreset): string {
  if (preset.type === 'ollama') return 'Local models served by Ollama'
  if (preset.type === 'custom') return 'Any OpenAI-compatible /v1 endpoint (LM Studio, vLLM, gateways)'
  if (preset.type === 'opencode' || preset.type === 'codex') {
    return `Uses your existing ${preset.label} login`
  }
  const models = preset.suggestedModels?.slice(0, 2).join(', ')
  return [preset.apiKeyEnvVar ?? 'API key', models].filter(Boolean).join(' · ')
}

export function connectionFieldsForPreset(
  preset: ProviderWizardPreset | null | undefined,
): ProviderConnectionField[] {
  if (!preset) return []
  const fields: ProviderConnectionField[] = [
    {
      key: 'providerId',
      label: 'Provider id',
      optional: false,
      secret: false,
      placeholder: preset.defaultProviderId ?? preset.type,
      hint: 'Short name used by /provider, /model, and session state.',
    },
  ]

  if (presetUsesBaseUrl(preset)) {
    fields.push({
      key: 'baseUrl',
      label: 'Base URL',
      optional: false,
      secret: false,
      placeholder: preset.type === 'custom'
        ? DEFAULT_OPENAI_COMPATIBLE_BASE_URL
        : 'http://127.0.0.1:11434',
      hint: preset.type === 'custom'
        ? 'Endpoint root; /v1 is appended when missing.'
        : 'Daemon-side Ollama endpoint. OLLAMA_HOST is used when set.',
    })
  }

  const auth = presetAuthKind(preset)
  if (auth !== 'none') {
    fields.push({
      key: 'apiKeyValue',
      label: 'API key',
      optional: true,
      secret: true,
      placeholder: auth === 'required'
        ? 'Paste a key, or leave blank to use the env var below'
        : 'Leave blank for a local server without auth',
      hint: 'Saved to ~/.sepilotd/.env after the connection test passes.',
    })
    fields.push({
      key: 'apiKeyEnvVar',
      label: 'Key env var',
      optional: auth === 'optional',
      secret: false,
      placeholder: preset.apiKeyEnvVar ?? 'API_KEY',
      hint: 'Environment variable the daemon reads the key from.',
    })
  }

  if (preset.type === 'custom') {
    fields.push({
      key: 'headers',
      label: 'Headers',
      optional: true,
      secret: false,
      placeholder: '{"X-API-Key":"${MY_API_KEY}"}',
      hint: 'Optional JSON object sent with every request. Use ${ENV_VAR} for secrets.',
    })
  }

  return fields
}

export function initialConnectionField(
  preset: ProviderWizardPreset | null | undefined,
): ProviderConnectionFieldKey {
  if (!preset) return 'providerId'
  if (presetUsesBaseUrl(preset)) return 'baseUrl'
  if (presetAuthKind(preset) === 'required') return 'apiKeyValue'
  return 'providerId'
}

export type ProviderModelDiscoveryRequest =
  | { kind: 'ollama'; baseUrl: string }
  | {
      kind: 'endpoint'
      type: ProviderWizardPreset['type']
      baseUrl: string
      apiKey?: string
      headers: Record<string, string>
    }

export type ProviderConnectionValidation =
  | { ok: false; field: ProviderConnectionFieldKey; error: string }
  | {
      ok: true
      values: ProviderConnectionValues
      headers: Record<string, string>
      discovery: ProviderModelDiscoveryRequest | null
    }

function normalizeCustomBaseUrl(raw: string): string {
  const trimmed = raw.trim().replace(/\/+$/, '')
  if (!trimmed) return ''
  return trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`
}

function isHttpUrl(value: string): boolean {
  try {
    const url = new URL(value)
    return url.protocol === 'http:' || url.protocol === 'https:'
  } catch {
    return false
  }
}

export function validateProviderConnection(
  values: ProviderConnectionValues,
  preset: ProviderWizardPreset,
): ProviderConnectionValidation {
  const providerId = values.providerId.trim()
  if (!providerId) {
    return { ok: false, field: 'providerId', error: 'Provider id is required.' }
  }
  if (/\s/.test(providerId)) {
    return { ok: false, field: 'providerId', error: 'Provider id must not contain spaces.' }
  }

  let baseUrl = ''
  if (presetUsesBaseUrl(preset)) {
    baseUrl = preset.type === 'custom'
      ? normalizeCustomBaseUrl(values.baseUrl)
      : values.baseUrl.trim().replace(/\/+$/, '')
    if (!baseUrl) {
      return { ok: false, field: 'baseUrl', error: 'Base URL is required.' }
    }
    if (!isHttpUrl(baseUrl)) {
      return { ok: false, field: 'baseUrl', error: 'Base URL must start with http:// or https://.' }
    }
  }

  const auth = presetAuthKind(preset)
  let apiKeyEnvVar = ''
  let apiKeyValue = ''
  if (auth !== 'none') {
    apiKeyValue = values.apiKeyValue
    const enteredEnvVar = values.apiKeyEnvVar.trim()
    if (enteredEnvVar && !isValidProviderEnvVarName(enteredEnvVar)) {
      return {
        ok: false,
        field: 'apiKeyEnvVar',
        error: 'Key env var must match [A-Za-z_][A-Za-z0-9_]*.',
      }
    }
    if (apiKeyValue.length > 0) {
      apiKeyEnvVar = managedProviderApiKeyEnvVar({
        providerId,
        preset,
        currentEnvVar: enteredEnvVar,
      })
    } else if (enteredEnvVar) {
      apiKeyEnvVar = enteredEnvVar
    } else if (auth === 'required') {
      return {
        ok: false,
        field: 'apiKeyValue',
        error: 'Enter an API key, or the name of an environment variable that holds one.',
      }
    }
  }

  let headers: Record<string, string> = {}
  let headersText = ''
  if (preset.type === 'custom') {
    try {
      headers = parseProviderHeadersInput(values.headersText)
    } catch (error) {
      return {
        ok: false,
        field: 'headers',
        error: error instanceof Error ? error.message : 'Invalid custom headers.',
      }
    }
    headersText = values.headersText.trim()
  }

  const discovery: ProviderModelDiscoveryRequest | null = preset.type === 'ollama'
    ? { kind: 'ollama', baseUrl }
    : preset.type === 'custom'
      ? {
          kind: 'endpoint',
          type: preset.type,
          baseUrl,
          apiKey: apiKeyValue.length > 0
            ? apiKeyValue
            : (apiKeyEnvVar ? `\${${apiKeyEnvVar}}` : undefined),
          headers,
        }
      : null

  return {
    ok: true,
    values: { providerId, baseUrl, apiKeyEnvVar, apiKeyValue, headersText },
    headers,
    discovery,
  }
}

export function firstProviderSetupStep(allowPresetSelection: boolean): ProviderSetupStep {
  return allowPresetSelection ? 'preset' : 'connection'
}

export function isFirstProviderSetupStep(
  step: ProviderSetupStep,
  allowPresetSelection: boolean,
): boolean {
  return step === firstProviderSetupStep(allowPresetSelection)
}

export function previousProviderSetupStep(
  step: ProviderSetupStep,
  allowPresetSelection: boolean,
): ProviderSetupStep | null {
  switch (step) {
    case 'confirm':
      return 'model'
    case 'model':
      return 'connection'
    case 'connection':
      return allowPresetSelection ? 'preset' : null
    case 'preset':
      return null
  }
}

export function filterProviderPresets(
  presets: ProviderWizardPreset[],
  query: string,
): ProviderWizardPreset[] {
  const normalized = query.trim().toLowerCase()
  if (!normalized) return presets
  return presets.filter((preset) => (
    preset.type.toLowerCase().includes(normalized)
    || preset.label.toLowerCase().includes(normalized)
    || (preset.defaultProviderId ?? '').toLowerCase().includes(normalized)
    || describeProviderPreset(preset).toLowerCase().includes(normalized)
  ))
}

export function filterModelSuggestions(suggestions: string[], query: string): string[] {
  const normalized = query.trim().toLowerCase()
  if (!normalized) return suggestions
  return suggestions.filter((model) => model.toLowerCase().includes(normalized))
}

export function describeApiKeySource(options: {
  preset: ProviderWizardPreset | null
  apiKeyEnvVar: string
  apiKeyValue: string
}): string | null {
  if (!options.preset || !presetUsesApiKey(options.preset)) return null
  if (options.apiKeyValue.length > 0) {
    return `entered key → saved as ${options.apiKeyEnvVar || 'managed env var'} in ~/.sepilotd/.env`
  }
  if (options.apiKeyEnvVar) {
    return `read ${options.apiKeyEnvVar} from the daemon environment`
  }
  return 'none (endpoint without authentication)'
}
