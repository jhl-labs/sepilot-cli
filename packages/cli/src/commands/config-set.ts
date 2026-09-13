import { readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import chalk from 'chalk'
import YAML from 'yaml'
import { ApiHttpError, isDaemonConfigUpdateKey, type DaemonConfigUpdateInput } from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { resolveDaemonDataDir, resolveDaemonEndpointScope } from '../client/token.js'
import { friendlyErrorMessage as errorMessage } from '../utils/error-message.js'
import { detectCliLocale } from '../utils/locale.js'

const CONFIG_SET_COPY = {
  en: {
    appliedViaDaemon: 'Applied via daemon API (no restart needed).',
    daemonUnreachable: (msg: string) => `Daemon unreachable (${msg}). Falling back to direct YAML edit.`,
    unset: '(unset)',
    updated: 'updated (value hidden)',
    restartHint: 'Restart daemon to apply: sepilot restart',
    failedPrefix: (msg: string) => `Failed: ${msg}`,
    initHint: 'Run "sepilot init" first.',
  },
  ko: {
    appliedViaDaemon: 'daemon API를 통해 적용됨 (재시작 불필요).',
    daemonUnreachable: (msg: string) => `daemon에 연결 불가 (${msg}). 직접 YAML 편집으로 폴백합니다.`,
    unset: '(미설정)',
    updated: '변경됨 (값 숨김)',
    restartHint: '적용하려면 daemon을 다시 시작하세요: sepilot restart',
    failedPrefix: (msg: string) => `실패: ${msg}`,
    initHint: '먼저 "sepilot init"을 실행하세요.',
  },
} as const

function configSetCopy() {
  return CONFIG_SET_COPY[detectCliLocale()] ?? CONFIG_SET_COPY.en
}

const UNSAFE_CONFIG_PATH_SEGMENTS = new Set(['__proto__', 'prototype', 'constructor'])

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
  const useDaemonApi = isDaemonConfigUpdateKey(key)
  const typedValue = coerceTypedValue(value, true)

  if (useDaemonApi) {
    try {
      const client = new DaemonClient(options.url)
      await client.updateConfig({ [key]: typedValue } as DaemonConfigUpdateInput)
      console.log(chalk.green(`${key}: ${copy.updated}`))
      console.log(chalk.gray(copy.appliedViaDaemon))
      return
    } catch (err) {
      // A server response is authoritative, not an offline condition. In
      // particular, validation and permission denials must never become local
      // YAML writes. Remote failure must not mutate an unrelated local profile.
      if (err instanceof ApiHttpError || resolveDaemonEndpointScope(options.url) !== 'loopback') {
        throw err
      }
      console.warn(
        chalk.yellow(copy.daemonUnreachable(errorMessage(err))),
      )
    }
  }

  if (resolveDaemonEndpointScope(options.url) !== 'loopback') {
    throw new Error(`Config key is not supported by the remote daemon API: ${key}`)
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
    if (
      parts.some((part) => part.length === 0)
      || parts.some((part) => UNSAFE_CONFIG_PATH_SEGMENTS.has(part))
    ) {
      throw new Error(`unsafe config path: ${key}`)
    }
    if (!config || typeof config !== 'object' || Array.isArray(config)) {
      throw new Error('config root must be a mapping')
    }
    let obj = config as Record<string, unknown>
    for (let i = 0; i < parts.length - 1; i++) {
      const part = parts[i]!
      if (obj[part] === undefined) {
        obj[part] = {}
      }
      const child = obj[part]
      if (!child || typeof child !== 'object' || Array.isArray(child)) {
        throw new Error(`config path is not a mapping: ${parts.slice(0, i + 1).join('.')}`)
      }
      obj = child as Record<string, unknown>
    }

    const lastKey = parts[parts.length - 1]!
    obj[lastKey] = typedValue

    await writeFile(configPath, YAML.stringify(config), 'utf-8')

    // Arbitrary config can contain credentials or executable command payloads.
    // Report the applied key, never the old or new value.
    console.log(chalk.green(`${key}: ${copy.updated}`))
    console.log(chalk.gray(copy.restartHint))
  } catch (err) {
    console.error(chalk.red(copy.failedPrefix(errorMessage(err))))
    console.error(chalk.gray(copy.initHint))
    process.exit(1)
  }
}
