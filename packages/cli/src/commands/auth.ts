import { createInterface } from 'node:readline/promises'
import chalk from 'chalk'
import type { DaemonConfigProvider } from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { friendlyErrorMessage, printApiError } from '../utils/error-message.js'
import {
  PROVIDER_WIZARD_PRESETS,
  extractEnvVarReference,
  type ProviderWizardPreset,
} from '../utils/provider-presets.js'
import { runProviderSetupFlow, type ProviderSetupInput } from '../utils/provider-setup-flow.js'
import { parseProviderHeadersInput } from '../utils/provider-http-options.js'

export interface AuthCommandOptions {
  url?: string
}

export interface AuthCommandIO {
  log(message: string): void
  question(prompt: string): Promise<string>
}

function createReadlineIo(): AuthCommandIO & { close(): void } {
  const rl = createInterface({ input: process.stdin, output: process.stdout })
  return {
    log: (message: string) => console.log(message),
    question: (prompt: string) => rl.question(prompt),
    close: () => rl.close(),
  }
}

const OLLAMA_PRESET: ProviderWizardPreset = { type: 'ollama', label: 'Ollama' }

function loginPresetChoices(): ProviderWizardPreset[] {
  return [...PROVIDER_WIZARD_PRESETS, OLLAMA_PRESET]
}

/** Mask a raw provider apiKey for display — never echo the plaintext value. */
function describeKeySource(apiKey: string | undefined): string | null {
  if (!apiKey) return null
  const envVar = extractEnvVarReference(apiKey)
  if (envVar) return `env:${envVar}`
  return 'config(***redacted***)'
}

export async function authListCommand(options: AuthCommandOptions): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    const [providers, config] = await Promise.all([client.providers(), client.config()])
    const rows = providers.map((provider) => {
      const configEntry = config.providers?.find((entry) => entry.id === provider.id)
      const keySource = describeKeySource(configEntry?.apiKey)
      return { id: provider.id, status: provider.health.status, keySource }
    })
    output(rows, (data) =>
      data
        .map((row) => {
          const keyLabel = row.keySource ? ` (${row.keySource})` : ''
          return `${row.id}: ${row.status}${keyLabel}`
        })
        .join('\n'),
    )
  } catch (err) {
    if (printApiError(err)) {
      process.exit(1)
    }
    console.error(chalk.red(`Failed to list providers: ${friendlyErrorMessage(err)}`))
    process.exit(1)
  }
}

export async function authLogoutCommand(
  providerId: string,
  options: AuthCommandOptions,
): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    const config = await client.config()
    const existing = config.providers ?? []
    const target = existing.find((entry) => entry.id === providerId)
    if (!target) {
      console.error(chalk.red(`Unknown provider: ${providerId}`))
      process.exit(1)
      return
    }

    const nextProviders = existing.map((entry) =>
      entry.id === providerId ? { ...entry, apiKey: undefined, headers: undefined } : entry,
    )

    await client.updateConfig({ providers: nextProviders })
    output({ ok: true, provider: providerId }, () =>
      chalk.green(`Logged out '${providerId}' (provider kept, auth cleared).`),
    )
  } catch (err) {
    if (printApiError(err)) {
      process.exit(1)
    }
    console.error(chalk.red(`Failed to log out '${providerId}': ${friendlyErrorMessage(err)}`))
    process.exit(1)
  }
}

async function promptPreset(io: AuthCommandIO): Promise<ProviderWizardPreset> {
  const choices = loginPresetChoices()
  io.log('Select a provider:')
  choices.forEach((choice, index) => {
    io.log(`  ${index + 1}. ${choice.label}`)
  })
  const answer = (await io.question(`Choose an option [1]: `)).trim()
  const numeric = answer ? Number.parseInt(answer, 10) : 1
  if (Number.isInteger(numeric) && numeric >= 1 && numeric <= choices.length) {
    return choices[numeric - 1]!
  }
  return choices[0]!
}

async function buildSetupInput(
  io: AuthCommandIO,
  preset: ProviderWizardPreset,
): Promise<ProviderSetupInput> {
  if (preset.type === 'ollama') {
    return { type: 'ollama', baseUrl: preset.defaultBaseUrl }
  }

  const input: ProviderSetupInput = { type: preset.type, baseUrl: preset.defaultBaseUrl }

  if (preset.apiKeyEnvVar) {
    input.apiKey = `\${${preset.apiKeyEnvVar}}`
  } else {
    const baseUrl = (
      await io.question(`Base URL${preset.defaultBaseUrl ? ` [${preset.defaultBaseUrl}]` : ''}: `)
    ).trim()
    input.baseUrl = baseUrl || preset.defaultBaseUrl
    const apiKey = (await io.question('API key (leave blank if none): ')).trim()
    if (apiKey) input.apiKey = apiKey
    const headers = parseProviderHeadersInput(
      await io.question('Custom HTTP headers JSON (leave blank if none): '),
    )
    if (Object.keys(headers).length > 0) input.headers = headers
  }

  return input
}

export async function authLoginCommand(
  options: AuthCommandOptions,
  io: AuthCommandIO = createReadlineIo(),
): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    const preset = await promptPreset(io)
    const input = await buildSetupInput(io, preset)

    const result = await runProviderSetupFlow(io, client, preset, input)
    if (!result.ok) {
      console.error(chalk.red(`Login failed: ${result.reason}`))
      process.exit(1)
      return
    }

    const config = await client.config()
    const existing = config.providers ?? []
    const isFirstProvider = existing.length === 0
    let defaultModel = result.defaultTarget ?? result.models[0]

    if (isFirstProvider && result.models.length > 0) {
      io.log('This is your first configured provider.')
      result.models.forEach((model, index) => {
        io.log(`  ${index + 1}. ${model}`)
      })
      const answer = (await io.question(`Choose a default model [1]: `)).trim()
      const numeric = answer ? Number.parseInt(answer, 10) : 1
      if (Number.isInteger(numeric) && numeric >= 1 && numeric <= result.models.length) {
        defaultModel = result.models[numeric - 1]
      }
    }

    const provider: DaemonConfigProvider = { ...result.provider, default: isFirstProvider }
    const nextProviders = [...existing.filter((entry) => entry.id !== provider.id), provider]

    await client.updateConfig({
      providers: nextProviders,
      ...(isFirstProvider
        ? {
            'agent.defaultProvider': provider.id,
            ...(defaultModel ? { 'agent.defaultModel': defaultModel } : {}),
          }
        : {}),
    })

    io.log(
      chalk.green(
        `Provider '${provider.id}' activated, ${result.models.length} model(s) discovered.`,
      ),
    )
    if (result.discoveryWarning) {
      io.log(chalk.yellow(result.discoveryWarning))
    }
  } catch (err) {
    if (printApiError(err)) {
      process.exit(1)
      return
    }
    console.error(chalk.red(`Login failed: ${friendlyErrorMessage(err)}`))
    process.exit(1)
  } finally {
    const maybeClose = io as AuthCommandIO & { close?: () => void }
    maybeClose.close?.()
  }
}
