import { readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import chalk from 'chalk'
import YAML from 'yaml'
import { isDaemonConfigUpdateKey, type DaemonConfigUpdateInput } from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { resolveDaemonDataDir } from '../client/token.js'
import { friendlyErrorMessage as errorMessage } from '../utils/error-message.js'
import { detectCliLocale } from '../utils/locale.js'

const CONFIG_SET_COPY = {
  en: {
    appliedViaDaemon: 'Applied via daemon API (no restart needed).',
    daemonUnreachable: (msg: string) => `Daemon unreachable (${msg}). Falling back to direct YAML edit.`,
    unset: '(unset)',
    restartHint: 'Restart daemon to apply: sepilot restart',
    failedPrefix: (msg: string) => `Failed: ${msg}`,
    initHint: 'Run "sepilot init" first.',
  },
  ko: {
    appliedViaDaemon: 'daemon API를 통해 적용됨 (재시작 불필요).',
    daemonUnreachable: (msg: string) => `daemon에 연결 불가 (${msg}). 직접 YAML 편집으로 폴백합니다.`,
    unset: '(미설정)',
    restartHint: '적용하려면 daemon을 다시 시작하세요: sepilot restart',
    failedPrefix: (msg: string) => `실패: ${msg}`,
    initHint: '먼저 "sepilot init"을 실행하세요.',
  },
} as const

function configSetCopy() {
  return CONFIG_SET_COPY[detectCliLocale()] ?? CONFIG_SET_COPY.en
}

// Scalar keys that can be safely set through daemon API via config-set CLI.
// Structural keys (providers, mcp.servers, etc.) also pass isDaemonConfigUpdateKey
// but are excluded here because setting them to a single scalar value would be
// malformed — those require dedicated CLI commands.
const SCALAR_DAEMON_KEYS = new Set<string>([
  'agent.mode',
  'agent.defaultProvider',
  'agent.defaultModel',
  'agent.autonomy',
  'agent.thinkingLevel',
  'device.name',
  'scheduler.timezone',
  'scheduler.surfaces.cli',
  'scheduler.surfaces.desktop',
])

function shouldUseDaemonApi(key: string): boolean {
  return SCALAR_DAEMON_KEYS.has(key) && isDaemonConfigUpdateKey(key)
}

type ConfigSetValue = string | number | boolean | null | unknown[] | Record<string, unknown>

function coerceTypedValue(value: string, allowStructured: boolean): ConfigSetValue {
  if (value === 'true') return true
  if (value === 'false') return false
  if (/^\d+$/.test(value)) return parseInt(value)
  if (/^\d+\.\d+$/.test(value)) return parseFloat(value)
  if (allowStructured && (value.startsWith('[') || value.startsWith('{'))) {
    try {
      const parsed: unknown = JSON.parse(value)
      if (Array.isArray(parsed) || (parsed !== null && typeof parsed === 'object')) {
        return parsed as unknown[] | Record<string, unknown>
      }
    } catch {
      // Preserve a non-JSON value as a string; YAML serialization remains safe.
    }
  }
  return value
}

export async function configSetCommand(
  key: string,
  value: string,
  options: { url?: string } = {},
) {
  const copy = configSetCopy()
  const useDaemonApi = shouldUseDaemonApi(key)
  const typedValue = coerceTypedValue(value, !useDaemonApi)

  if (useDaemonApi) {
    try {
      const client = new DaemonClient(options.url)
      await client.updateConfig({ [key]: typedValue } as DaemonConfigUpdateInput)
      console.log(chalk.green(`${key} → ${JSON.stringify(typedValue)}`))
      console.log(chalk.gray(copy.appliedViaDaemon))
      return
    } catch (err) {
      console.warn(
        chalk.yellow(copy.daemonUnreachable(errorMessage(err))),
      )
    }
  }

  await configSetViaYaml(key, typedValue)
}

async function configSetViaYaml(
  key: string,
  typedValue: ConfigSetValue,
): Promise<void> {
  const copy = configSetCopy()
  const configPath = join(resolveDaemonDataDir(), 'config.yaml')

  try {
    const content = await readFile(configPath, 'utf-8')
    const config = YAML.parse(content)

    const parts = key.split('.')
    let obj = config
    for (let i = 0; i < parts.length - 1; i++) {
      if (obj[parts[i]] === undefined) {
        obj[parts[i]] = {}
      }
      obj = obj[parts[i]]
    }

    const lastKey = parts[parts.length - 1]
    const oldValue = obj[lastKey]
    obj[lastKey] = typedValue

    await writeFile(configPath, YAML.stringify(config), 'utf-8')

    const before = oldValue === undefined ? chalk.gray(copy.unset) : JSON.stringify(oldValue)
    console.log(chalk.green(`${key}: ${before} → ${JSON.stringify(typedValue)}`))
    console.log(chalk.gray(copy.restartHint))
  } catch (err) {
    console.error(chalk.red(copy.failedPrefix(errorMessage(err))))
    console.error(chalk.gray(copy.initHint))
    process.exit(1)
  }
}
