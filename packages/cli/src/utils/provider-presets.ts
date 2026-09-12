import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import {
  DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
  type DaemonConfigProvider,
} from '@sepilotd/api-client'

export { discoverOpenAiCompatibleModels } from '@sepilotd/api-client'

export interface ProviderWizardPreset {
  type: DaemonConfigProvider['type']
  label: string
  defaultProviderId?: string
  defaultBaseUrl?: string
  apiKeyEnvVar?: string
  suggestedModels?: string[]
}

const execFileAsync = promisify(execFile)

const LEGACY_PROVIDER_WIZARD_PRESETS: ProviderWizardPreset[] = [
  { type: 'ollama', label: 'Ollama' },
]

export const PROVIDER_WIZARD_PRESETS: ProviderWizardPreset[] = [
  {
    type: 'custom',
    label: 'OpenAI-compatible',
    defaultProviderId: 'openai-compatible',
    defaultBaseUrl: DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
  },
  {
    type: 'opencode',
    label: 'Opencode',
    defaultProviderId: 'opencode',
    suggestedModels: ['default'],
  },
  {
    type: 'codex',
    label: 'Codex',
    defaultProviderId: 'codex',
    suggestedModels: ['default'],
  },
  {
    type: 'openai',
    label: 'OpenAI',
    apiKeyEnvVar: 'OPENAI_API_KEY',
    suggestedModels: ['gpt-5.4', 'gpt-5.4-mini', 'gpt-5.4-nano'],
  },
  {
    type: 'anthropic',
    label: 'Anthropic',
    apiKeyEnvVar: 'ANTHROPIC_API_KEY',
    suggestedModels: ['claude-sonnet-4-6', 'claude-haiku-4-5', 'claude-opus-4-6'],
  },
  {
    type: 'gemini',
    label: 'Gemini',
    apiKeyEnvVar: 'GEMINI_API_KEY',
    suggestedModels: ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite'],
  },
  {
    type: 'groq',
    label: 'Groq',
    apiKeyEnvVar: 'GROQ_API_KEY',
    suggestedModels: ['openai/gpt-oss-120b', 'llama-3.3-70b-versatile', 'llama-3.1-8b-instant'],
  },
  {
    type: 'together',
    label: 'Together',
    apiKeyEnvVar: 'TOGETHER_API_KEY',
    suggestedModels: ['moonshotai/Kimi-K2.5', 'zai-org/GLM-5', 'openai/gpt-oss-120b'],
  },
  {
    type: 'deepseek',
    label: 'DeepSeek',
    apiKeyEnvVar: 'DEEPSEEK_API_KEY',
    suggestedModels: ['deepseek-chat', 'deepseek-reasoner'],
  },
]

export function findProviderWizardPreset(
  query: string | null | undefined,
): ProviderWizardPreset | null {
  const normalized = query?.trim().toLowerCase()
  if (!normalized) {
    return null
  }

  const presets = [...PROVIDER_WIZARD_PRESETS, ...LEGACY_PROVIDER_WIZARD_PRESETS]
  const exact = presets.find((preset) => {
    const label = preset.label.toLowerCase()
    const defaultProviderId = preset.defaultProviderId?.toLowerCase() ?? ''
    return (
      preset.type === normalized
      || label === normalized
      || defaultProviderId === normalized
    )
  })
  if (exact) return exact

  return presets.find((preset) => {
    const label = preset.label.toLowerCase()
    const defaultProviderId = preset.defaultProviderId?.toLowerCase() ?? ''
    return label.includes(normalized) || defaultProviderId.includes(normalized)
  }) ?? null
}

export function extractEnvVarReference(value: string | undefined): string | null {
  if (!value) return null
  const match = value.match(/^\$\{([^}]+)\}$/)
  return match?.[1] ?? null
}

function parseOllamaListOutput(stdout: string): string[] {
  return stdout
    .split('\n')
    .slice(1)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.split(/\s{2,}/)[0]?.trim() ?? '')
    .filter(Boolean)
}

export async function detectOllamaModels(_baseUrl: string): Promise<string[]> {
  try {
    const { stdout } = await execFileAsync('ollama', ['list'])
    return parseOllamaListOutput(stdout)
  } catch {
    return []
  }
}
