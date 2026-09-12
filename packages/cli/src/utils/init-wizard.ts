import { readFile, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { createInterface, type Interface } from 'node:readline/promises'
import chalk from 'chalk'
import YAML from 'yaml'
import {
  daemonEnvPath,
  mergeDaemonEnv,
  readDaemonEnvFile,
  updateDaemonEnvFile,
} from './daemon-env.js'
import {
  detectOllamaModels,
  discoverOpenAiCompatibleModels,
  extractEnvVarReference as extractEnvVarReferenceFromPreset,
  PROVIDER_WIZARD_PRESETS,
  type ProviderWizardPreset,
} from './provider-presets.js'
import {
  formatProviderHeadersInput,
  parseProviderHeadersInput,
  resolveProviderHeaderEnvReferences,
} from './provider-http-options.js'

export { PROVIDER_WIZARD_PRESETS } from './provider-presets.js'

type LooseConfig = Record<string, any>
type LooseProviderConfig = Record<string, any>

export interface ProviderWizardChoice {
  preset: ProviderWizardPreset | null
  value: string
  label: string
  description: string
  recommended: boolean
}

interface WizardSelection {
  provider: LooseProviderConfig
  defaultModel: string
  managedEnvVarName?: string
  managedEnvValue?: string | null
  managedEnvMode?: 'saved' | 'kept' | 'external'
}

interface ProviderWizardEnvironment {
  [key: string]: string | undefined
}

function configPath(homeDir = homedir()): string {
  return join(homeDir, '.sepilotd', 'config.yaml')
}

function isValidEnvVarName(value: string): boolean {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(value.trim())
}

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function normalizeConfig(raw: unknown): LooseConfig {
  return isRecord(raw) ? raw : {}
}

function currentDefaultProviderId(config: LooseConfig): string | null {
  if (typeof config.agent?.defaultProvider === 'string' && config.agent.defaultProvider.trim()) {
    return config.agent.defaultProvider.trim()
  }

  const providers = Array.isArray(config.providers) ? config.providers : []
  const markedDefault = providers.find((provider) => provider?.default)
  if (markedDefault?.id) {
    return String(markedDefault.id)
  }

  return providers[0]?.id ? String(providers[0].id) : null
}

function currentProviderById(config: LooseConfig, providerId: string | null): LooseProviderConfig | null {
  if (!providerId) return null
  const providers = Array.isArray(config.providers) ? config.providers : []
  const provider = providers.find((item) => item?.id === providerId)
  return isRecord(provider) ? provider : null
}

function currentProviderByType(config: LooseConfig, type: string): LooseProviderConfig | null {
  const providers = Array.isArray(config.providers) ? config.providers : []
  const provider = providers.find((item) => item?.type === type)
  return isRecord(provider) ? provider : null
}

function currentDefaultModel(config: LooseConfig, provider: LooseProviderConfig | null): string {
  if (typeof config.agent?.defaultModel === 'string' && config.agent.defaultModel.trim()) {
    return config.agent.defaultModel.trim()
  }

  const firstModel = Array.isArray(provider?.models) ? provider.models[0] : undefined
  return typeof firstModel === 'string' ? firstModel : ''
}

function appendOpenAiVersionPath(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, '')
  if (!trimmed) return ''
  return trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`
}

function defaultOpenAiCompatibleBaseUrl(
  preset: ProviderWizardPreset,
  env: ProviderWizardEnvironment,
  currentBaseUrl = '',
): string {
  return (
    currentBaseUrl.trim()
    || env.OPENAI_COMPATIBLE_BASE_URL?.trim()
    || env.OPENAI_BASE_URL?.trim()
    || appendOpenAiVersionPath(env.OLLAMA_HOST ?? '')
    || preset.defaultBaseUrl
    || ''
  )
}

export function extractEnvVarReference(value: string | undefined): string | null {
  return extractEnvVarReferenceFromPreset(value)
}

export function buildProviderWizardChoices(options: {
  config: LooseConfig
  ollamaModels: string[]
  env: ProviderWizardEnvironment
}): ProviderWizardChoice[] {
  const currentProviderId = currentDefaultProviderId(options.config)
  const currentProvider = currentProviderById(options.config, currentProviderId)

  const choices = PROVIDER_WIZARD_PRESETS.map((preset) => {
    const details: string[] = []
    const isCurrent = currentProvider?.type === preset.type || currentProvider?.id === preset.type

    if (preset.type === 'custom') {
      if (options.ollamaModels.length > 0) {
        details.push(`${options.ollamaModels.length} local models detected`)
      } else {
        details.push('no local models detected')
      }
      if (options.env.OPENAI_COMPATIBLE_BASE_URL || options.env.OPENAI_BASE_URL || options.env.OLLAMA_HOST) {
        details.push('base URL available')
      }
    } else if (preset.type === 'ollama') {
      if (options.ollamaModels.length > 0) {
        details.push(`${options.ollamaModels.length} local models detected`)
      } else {
        details.push('no local models detected')
      }
    } else if (preset.apiKeyEnvVar) {
      details.push(options.env[preset.apiKeyEnvVar] ? `${preset.apiKeyEnvVar} is set` : `${preset.apiKeyEnvVar} not set`)
    }

    if (isCurrent) {
      details.push('current default')
    }

    return {
      preset,
      value: preset.type,
      label: preset.label,
      description: details.join(', '),
      recommended: preset.type === 'ollama'
        ? options.ollamaModels.length > 0
        : preset.type === 'custom'
          ? (
              options.ollamaModels.length > 0
              || Boolean(options.env.OPENAI_COMPATIBLE_BASE_URL || options.env.OPENAI_BASE_URL || options.env.OLLAMA_HOST)
            )
          : Boolean(preset.apiKeyEnvVar && options.env[preset.apiKeyEnvVar]),
    }
  })

  return [
    ...choices,
    {
      preset: null,
      value: 'skip',
      label: 'Skip for now',
      description: 'keep the current config as-is',
      recommended: false,
    },
  ]
}

export function applyProviderWizardSelection(
  config: LooseConfig,
  selection: WizardSelection,
): LooseConfig {
  const nextConfig = normalizeConfig(config)
  const providers = Array.isArray(nextConfig.providers)
    ? nextConfig.providers.filter((provider): provider is LooseProviderConfig => isRecord(provider))
    : []

  const nextProviders = providers
    .filter((provider) => provider.id !== selection.provider.id)
    .map((provider) => ({ ...provider, default: false }))

  nextProviders.push({
    ...selection.provider,
    default: true,
  })

  nextConfig.providers = nextProviders
  nextConfig.agent = isRecord(nextConfig.agent) ? nextConfig.agent : {}
  nextConfig.agent.defaultProvider = selection.provider.id
  nextConfig.agent.defaultModel = selection.defaultModel
  if (providers.length === 0) {
    nextConfig.agent.mode = 'react'
  }

  return nextConfig
}

async function promptYesNo(
  rl: Interface,
  message: string,
  defaultValue = true,
): Promise<boolean> {
  const suffix = defaultValue ? ' [Y/n] ' : ' [y/N] '
  const answer = (await rl.question(`${message}${suffix}`)).trim().toLowerCase()
  if (!answer) return defaultValue
  return answer === 'y' || answer === 'yes'
}

async function promptText(
  rl: Interface,
  message: string,
  defaultValue = '',
  validate?: (value: string) => string | null,
): Promise<string> {
  while (true) {
    const renderedDefault = defaultValue ? ` [${defaultValue}]` : ''
    const answer = (await rl.question(`${message}${renderedDefault}: `)).trim()
    const value = answer || defaultValue
    const validationError = validate?.(value) ?? null
    if (!validationError) {
      return value
    }
    console.log(chalk.red(validationError))
  }
}

async function promptSecret(
  rl: Interface,
  message: string,
  options: {
    allowEmpty?: boolean
    validate?: (value: string) => string | null
  } = {},
): Promise<string> {
  const writable = rl as Interface & {
    _writeToOutput?: (value: string) => void
  }
  const originalWrite = writable._writeToOutput?.bind(rl)

  while (true) {
    process.stdout.write(`${message}: `)
    writable._writeToOutput = () => {}
    const answer = await rl.question('')
    writable._writeToOutput = originalWrite
    process.stdout.write('\n')

    const value = answer.trim()
    if (!value && options.allowEmpty) {
      return ''
    }

    const validationError = options.validate?.(value)
      ?? (!value ? 'A value is required.' : null)
    if (!validationError) {
      return value
    }
    console.log(chalk.red(validationError))
  }
}

async function promptChoice(
  rl: Interface,
  message: string,
  choices: ProviderWizardChoice[],
): Promise<ProviderWizardChoice> {
  const recommendedIndex = choices.findIndex((choice) => choice.recommended)
  const defaultIndex = recommendedIndex >= 0 ? recommendedIndex : 0

  while (true) {
    console.log(`\n${message}`)
    for (const [index, choice] of choices.entries()) {
      const markers = [
        choice.recommended ? 'recommended' : null,
      ].filter(Boolean)
      const suffix = markers.length > 0 ? ` (${markers.join(', ')})` : ''
      console.log(`  ${index + 1}. ${choice.label}${suffix}`)
      console.log(`     ${choice.description}`)
    }

    const answer = (await rl.question(`Choose an option [${defaultIndex + 1}]: `)).trim()
    const numeric = answer ? Number.parseInt(answer, 10) : defaultIndex + 1
    if (Number.isInteger(numeric) && numeric >= 1 && numeric <= choices.length) {
      return choices[numeric - 1]
    }
    console.log(chalk.red('Enter one of the listed numbers.'))
  }
}

async function chooseOllamaModel(
  rl: Interface,
  models: string[],
  defaultModel: string,
): Promise<string> {
  if (models.length === 0) {
    return promptText(
      rl,
      'Enter the Ollama model name',
      defaultModel || 'llama3.3',
      (value) => value ? null : 'Model name is required.',
    )
  }

  const choices = models.map((model) => ({
    preset: null,
    value: model,
    label: model,
    description: 'local model',
    recommended: model === defaultModel,
  }))
  choices.push({
    preset: null,
    value: '__manual__',
    label: 'Enter a different model',
    description: 'type a model name manually',
    recommended: false,
  })

  const selected = await promptChoice(rl, 'Select the default Ollama model', choices)
  if (selected.value !== '__manual__') {
    return selected.value
  }

  return promptText(
    rl,
    'Enter the Ollama model name',
    defaultModel || models[0] || 'llama3.3',
    (value) => value ? null : 'Model name is required.',
  )
}

export function buildSuggestedModelChoices(
  models: string[],
  currentModel: string,
): ProviderWizardChoice[] {
  const mergedModels = currentModel && !models.includes(currentModel)
    ? [currentModel, ...models]
    : models

  return [
    ...mergedModels.map((model, index) => ({
      preset: null,
      value: model,
      label: model,
      description: index === 0 ? 'recommended' : 'suggested',
      recommended: model === currentModel || (!currentModel && index === 0),
    })),
    {
      preset: null,
      value: '__manual__',
      label: 'Enter a different model',
      description: 'type a model name manually',
      recommended: false,
    },
  ]
}

async function chooseSuggestedModel(
  rl: Interface,
  providerLabel: string,
  models: string[],
  currentModel: string,
): Promise<string> {
  const selected = await promptChoice(
    rl,
    `Select the default ${providerLabel} model`,
    buildSuggestedModelChoices(models, currentModel),
  )

  if (selected.value !== '__manual__') {
    return selected.value
  }

  return promptText(
    rl,
    `Enter the ${providerLabel} model name`,
    currentModel || models[0] || '',
    (value) => value ? null : 'Model name is required.',
  )
}

async function configureProviderSelection(
  rl: Interface,
  config: LooseConfig,
  preset: ProviderWizardPreset,
  ollamaModels: string[],
  options: {
    shellEnv: ProviderWizardEnvironment
    managedEnv: ProviderWizardEnvironment
    effectiveEnv: ProviderWizardEnvironment
    configFilePath?: string
  },
): Promise<WizardSelection> {
  const currentByType = currentProviderByType(config, preset.type)
  const legacyOllamaProvider = preset.type === 'custom'
    ? currentProviderByType(config, 'ollama')
    : null
  const currentModel = currentDefaultModel(config, currentByType)

  if (preset.type === 'custom') {
    const currentBaseUrl =
      typeof currentByType?.baseUrl === 'string'
        ? currentByType.baseUrl
        : typeof legacyOllamaProvider?.baseUrl === 'string'
          ? appendOpenAiVersionPath(legacyOllamaProvider.baseUrl)
          : ''
    const baseUrl = await promptText(
      rl,
      'OpenAI-compatible base URL',
      defaultOpenAiCompatibleBaseUrl(preset, options.effectiveEnv, currentBaseUrl),
      (value) => value ? null : 'Base URL is required.',
    )
    const currentApiKeyEnvVar = extractEnvVarReference(
      typeof currentByType?.apiKey === 'string' ? currentByType.apiKey : undefined,
    ) ?? ''
    const apiKeyEnvVar = await promptText(
      rl,
      'OpenAI-compatible API key env var (optional)',
      currentApiKeyEnvVar,
      (value) => value && !isValidEnvVarName(value)
        ? 'Env var name must match [A-Za-z_][A-Za-z0-9_]*.'
        : null,
    )

    let managedEnvValue: string | null = null
    let managedEnvMode: WizardSelection['managedEnvMode'] = 'external'
    if (apiKeyEnvVar) {
      const shellValue = options.shellEnv[apiKeyEnvVar]?.trim() ?? ''
      const existingManagedValue = options.managedEnv[apiKeyEnvVar]?.trim() ?? ''
      const shouldStoreInManagedEnv = await promptYesNo(
        rl,
        existingManagedValue
          ? `Store ${apiKeyEnvVar} in ${daemonEnvPath({ configFilePath: options.configFilePath })}?`
          : `Store ${apiKeyEnvVar} in ${daemonEnvPath({ configFilePath: options.configFilePath })} for sepilotd?`,
        !shellValue || Boolean(existingManagedValue),
      )
      if (shouldStoreInManagedEnv) {
        const secretValue = await promptSecret(
          rl,
          existingManagedValue
            ? `Enter ${apiKeyEnvVar} value (press Enter to keep existing managed value)`
            : `Enter ${apiKeyEnvVar} value`,
          {
            allowEmpty: Boolean(existingManagedValue),
            validate: (value) => value || existingManagedValue
              ? null
              : 'A secret value is required to write the managed env file.',
          },
        )
        managedEnvValue = secretValue || existingManagedValue
        managedEnvMode = secretValue ? 'saved' : 'kept'
      }
    }

    const currentHeaders = isRecord(currentByType?.headers)
      ? currentByType.headers as Record<string, string>
      : {}
    const headersText = await promptText(
      rl,
      'Custom HTTP headers JSON (optional)',
      formatProviderHeadersInput(currentHeaders),
      (value) => {
        try {
          parseProviderHeadersInput(value)
          return null
        } catch (error) {
          return error instanceof Error ? error.message : 'Invalid custom headers.'
        }
      },
    )
    const headers = parseProviderHeadersInput(headersText)
    const discoveryEnv = {
      ...options.effectiveEnv,
      ...(apiKeyEnvVar && managedEnvValue ? { [apiKeyEnvVar]: managedEnvValue } : {}),
    }
    const discoveredModels = await discoverOpenAiCompatibleModels(baseUrl, {
      apiKey: apiKeyEnvVar ? discoveryEnv[apiKeyEnvVar] : undefined,
      headers: resolveProviderHeaderEnvReferences(headers, discoveryEnv),
    })
    const modelChoices = discoveredModels.length > 0 ? discoveredModels : ollamaModels
    const defaultModel = modelChoices.length > 0
      ? await chooseSuggestedModel(
          rl,
          preset.label,
          modelChoices,
          currentModel,
        )
      : await promptText(
          rl,
          `${preset.label} default model`,
          currentModel,
          (value) => value ? null : 'Model name is required.',
        )
    const configuredModels = Array.from(new Set([defaultModel, ...modelChoices]))

    return {
      provider: {
        ...currentByType,
        id: currentByType?.id ?? preset.defaultProviderId ?? preset.type,
        type: 'custom',
        baseUrl,
        apiKey: apiKeyEnvVar ? `\${${apiKeyEnvVar}}` : 'local',
        headers,
        models: configuredModels.length > 0 ? configuredModels : [defaultModel],
      },
      defaultModel,
      ...(apiKeyEnvVar
        ? {
            managedEnvVarName: apiKeyEnvVar,
            managedEnvValue,
            managedEnvMode,
          }
        : {}),
    }
  }

  if (preset.type === 'ollama') {
    const baseUrl = await promptText(
      rl,
      'Ollama base URL',
      typeof currentByType?.baseUrl === 'string' ? currentByType.baseUrl : (options.effectiveEnv.OLLAMA_HOST ?? ''),
      (value) => value ? null : 'Base URL is required.',
    )
    const availableModels = ollamaModels.length > 0 ? ollamaModels : await detectOllamaModels(baseUrl)
    const defaultModel = await chooseOllamaModel(rl, availableModels, currentModel)
    const configuredModels = Array.from(new Set([defaultModel, ...availableModels]))

    return {
      provider: {
        ...currentByType,
        id: 'ollama',
        type: 'ollama',
        baseUrl,
        models: configuredModels.length > 0 ? configuredModels : [defaultModel],
      },
      defaultModel,
    }
  }

  const defaultEnvVar = extractEnvVarReference(currentByType?.apiKey) ?? preset.apiKeyEnvVar ?? 'API_KEY'
  const envVarName = await promptText(
    rl,
    `${preset.label} API key env var`,
    defaultEnvVar,
    (value) => {
      if (!value) return 'Env var name is required.'
      return isValidEnvVarName(value) ? null : 'Env var name must match [A-Za-z_][A-Za-z0-9_]*.'
    },
  )
  const defaultModel = preset.suggestedModels && preset.suggestedModels.length > 0
    ? await chooseSuggestedModel(
        rl,
        preset.label,
        preset.suggestedModels,
        currentModel,
      )
    : await promptText(
        rl,
        `${preset.label} default model`,
        currentModel,
        (value) => value ? null : 'Model name is required.',
      )

  const shellValue = options.shellEnv[envVarName]?.trim() ?? ''
  const managedValue = options.managedEnv[envVarName]?.trim() ?? ''
  const effectiveValue = options.effectiveEnv[envVarName]?.trim() ?? ''
  const defaultStoreInManagedEnv = !shellValue || Boolean(managedValue)
  const shouldStoreInManagedEnv = await promptYesNo(
    rl,
    managedValue
      ? `Store ${envVarName} in ${daemonEnvPath({ configFilePath: options.configFilePath })}?`
      : `Store ${envVarName} in ${daemonEnvPath({ configFilePath: options.configFilePath })} for sepilotd?`,
    defaultStoreInManagedEnv,
  )

  let managedEnvValue: string | null = null
  let managedEnvMode: WizardSelection['managedEnvMode'] = 'external'
  if (shouldStoreInManagedEnv) {
    const secretValue = await promptSecret(
      rl,
      managedValue
        ? `Enter ${envVarName} value (press Enter to keep existing managed value)`
        : `Enter ${envVarName} value`,
      {
        allowEmpty: Boolean(managedValue),
        validate: (value) => {
          if (value || managedValue) {
            return null
          }
          return 'A secret value is required to write the managed env file.'
        },
      },
    )
    managedEnvValue = secretValue || managedValue
    managedEnvMode = secretValue ? 'saved' : 'kept'
  }

  if (!effectiveValue && !managedEnvValue) {
    console.log(
      chalk.yellow(
        `  ${envVarName} is not set in this shell. sepilotd will need that env var when it starts.`,
      ),
    )
  }

  return {
    provider: {
      ...currentByType,
      id: preset.type,
      type: preset.type,
      apiKey: `\${${envVarName}}`,
      models: [defaultModel],
    },
    defaultModel,
    managedEnvVarName: envVarName,
    managedEnvValue,
    managedEnvMode,
  }
}

export async function maybeRunInitWizard(options: {
  enabled: boolean
  configFilePath?: string
  env?: ProviderWizardEnvironment
}): Promise<boolean> {
  if (!options.enabled || !process.stdin.isTTY || !process.stdout.isTTY) {
    return false
  }

  const filePath = options.configFilePath ?? configPath()
  let rawConfig: string
  try {
    rawConfig = await readFile(filePath, 'utf-8')
  } catch {
    return false
  }

  const config = normalizeConfig(YAML.parse(rawConfig))
  const shellEnv = options.env ?? process.env
  const managedEnv = await readDaemonEnvFile({ configFilePath: filePath })
  const env = mergeDaemonEnv(shellEnv, managedEnv)
  const currentProvider = currentProviderById(config, currentDefaultProviderId(config))
  const currentModel = currentDefaultModel(config, currentProvider)
  const ollamaBaseUrl = typeof currentProvider?.baseUrl === 'string'
    ? currentProvider.baseUrl
    : (env.OLLAMA_HOST ?? '')
  const ollamaModels = await detectOllamaModels(ollamaBaseUrl)

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  })

  try {
    console.log('\nProvider setup wizard')
    if (currentProvider) {
      console.log(
        chalk.gray(
          `Current default: ${currentProvider.id}/${currentModel || '(unset model)'}`,
        ),
      )
    }

    const shouldConfigure = await promptYesNo(
      rl,
      'Configure the default provider and model now?',
      true,
    )
    if (!shouldConfigure) {
      return false
    }

    const choice = await promptChoice(
      rl,
      'Select a provider',
      buildProviderWizardChoices({ config, ollamaModels, env }),
    )
    if (!choice.preset) {
      return false
    }

    const selection = await configureProviderSelection(
      rl,
      config,
      choice.preset,
      ollamaModels,
      {
        shellEnv,
        managedEnv,
        effectiveEnv: env,
        configFilePath: filePath,
      },
    )
    const updatedConfig = applyProviderWizardSelection(config, selection)

    await writeFile(filePath, YAML.stringify(updatedConfig), 'utf-8')
    if (
      selection.managedEnvVarName
      && selection.managedEnvValue
      && selection.managedEnvMode !== 'external'
    ) {
      await updateDaemonEnvFile(
        {
          [selection.managedEnvVarName]: selection.managedEnvValue,
        },
        { configFilePath: filePath },
      )
    }

    console.log(
      chalk.green(
        `Saved default provider ${selection.provider.id} with model ${selection.defaultModel}.`,
      ),
    )
    if (selection.managedEnvVarName) {
      if (selection.managedEnvMode === 'saved') {
        console.log(
          chalk.green(
            `Stored ${selection.managedEnvVarName} in ${daemonEnvPath({ configFilePath: filePath })}.`,
          ),
        )
      } else if (selection.managedEnvMode === 'kept') {
        console.log(
          chalk.gray(
            `Kept the existing managed value for ${selection.managedEnvVarName} in ${daemonEnvPath({ configFilePath: filePath })}.`,
          ),
        )
      } else if (!env[selection.managedEnvVarName]) {
        console.log(
          chalk.yellow(
            `sepilotd will still need ${selection.managedEnvVarName} from the shell or service environment when it starts.`,
          ),
        )
      }
    }
    console.log(chalk.gray('Restart daemon to apply changes: sepilot restart'))
    return true
  } finally {
    rl.close()
  }
}
