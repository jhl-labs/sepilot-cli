import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import YAML from 'yaml'
import { createLogger } from '../../logger.js'
import { isNodeFsError } from '../../utils/fs-error.js'
import type { SepilotdConfig } from '../../config/schema.js'

const log = createLogger('channel-pairing-persistence')

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export interface PersistedTelegramPairing {
  code: string
  expiresAt: number
}

function telegramPairingPath(dataDir: string): string {
  return join(dataDir, 'channels', 'telegram', 'pairing-code.json')
}

export function appendTelegramAllowedUser(
  config: SepilotdConfig,
  userId: string,
): boolean {
  for (const channel of config.channels ?? []) {
    if (channel.type !== 'telegram') {
      continue
    }

    const channelConfig = isRecord(channel.config) ? channel.config : {}
    const rawAllowedUsers = channelConfig.allowedUsers
    const allowedUsers = Array.isArray(rawAllowedUsers)
      ? rawAllowedUsers.filter((value): value is string => typeof value === 'string')
      : []

    if (allowedUsers.includes(userId)) {
      return false
    }

    channel.config = {
      ...channelConfig,
      allowedUsers: [...allowedUsers, userId],
    }
    return true
  }

  return false
}

export function removeTelegramAllowedUser(
  config: SepilotdConfig,
  userId: string,
): boolean {
  for (const channel of config.channels ?? []) {
    if (channel.type !== 'telegram') {
      continue
    }

    const channelConfig = isRecord(channel.config) ? channel.config : {}
    const rawAllowedUsers = channelConfig.allowedUsers
    const allowedUsers = Array.isArray(rawAllowedUsers)
      ? rawAllowedUsers.filter((value): value is string => typeof value === 'string')
      : []

    if (!allowedUsers.includes(userId)) {
      return false
    }

    channel.config = {
      ...channelConfig,
      allowedUsers: allowedUsers.filter((value) => value !== userId),
    }
    return true
  }

  return false
}

export async function persistChannelsConfig(
  dataDir: string,
  runtimeConfig: SepilotdConfig,
): Promise<void> {
  const configPath = join(dataDir, 'config.yaml')
  const tempPath = `${configPath}.tmp`
  let persistedConfig: SepilotdConfig = runtimeConfig

  // Read the on-disk config so we only rewrite the `channels`
  // section and leave the operator's other edits (memory settings,
  // providers, agent flags, ...) untouched. The original code
  // swallowed every read or YAML.parse failure and wrote
  // `runtimeConfig` whole, silently clobbering every other field
  // when the file was momentarily unreadable or had a transient
  // YAML error.
  let raw: string | null = null
  try {
    raw = await readFile(configPath, 'utf-8')
  } catch (err) {
    if (!isNodeFsError(err, 'ENOENT')) {
      log.error('failed to read config.yaml during channels persist', {
        path: configPath,
        error: err instanceof Error ? err.message : String(err),
      })
      throw err
    }
    // ENOENT — first run, no config file yet. Writing the runtime
    // config wholesale is the right thing.
  }

  if (raw !== null) {
    try {
      const rawConfig = YAML.parse(raw)
      if (isRecord(rawConfig)) {
        persistedConfig = rawConfig as SepilotdConfig
        persistedConfig.channels = structuredClone(runtimeConfig.channels)
      } else {
        log.warn('config.yaml did not parse to an object; channels-only persist refused', {
          path: configPath,
        })
        throw new Error(
          `Refusing to clobber config.yaml at ${configPath}: top-level value is not a YAML mapping.`,
        )
      }
    } catch (err) {
      // YAML.parse threw — the file exists but is malformed.
      // Refuse rather than wipe the operator's other edits.
      log.error('config.yaml is unparseable; channels-only persist refused', {
        path: configPath,
        error: err instanceof Error ? err.message : String(err),
      })
      throw err instanceof Error
        ? err
        : new Error(`Failed to parse config.yaml at ${configPath}: ${String(err)}`)
    }
  }

  const yaml = YAML.stringify(persistedConfig)
  await writeFile(tempPath, yaml, { mode: 0o600 })
  await rename(tempPath, configPath)
}

export async function saveTelegramPendingPairing(
  dataDir: string,
  pairing: PersistedTelegramPairing,
): Promise<void> {
  const filePath = telegramPairingPath(dataDir)
  const tempPath = `${filePath}.tmp`
  await mkdir(dirname(filePath), { recursive: true, mode: 0o700 })
  await writeFile(tempPath, `${JSON.stringify(pairing, null, 2)}\n`, { mode: 0o600 })
  await rename(tempPath, filePath)
}

export async function clearTelegramPendingPairing(
  dataDir: string,
): Promise<void> {
  await rm(telegramPairingPath(dataDir), { force: true })
}

export async function readTelegramPendingPairing(
  dataDir: string,
): Promise<PersistedTelegramPairing | null> {
  let parsed: unknown

  try {
    parsed = JSON.parse(await readFile(telegramPairingPath(dataDir), 'utf-8'))
  } catch {
    return null
  }

  if (!isRecord(parsed)) {
    return null
  }

  const pairing = {
    code: typeof parsed.code === 'string' ? parsed.code : '',
    expiresAt: typeof parsed.expiresAt === 'number' ? parsed.expiresAt : 0,
  }

  if (!/^\d{6}$/.test(pairing.code) || !Number.isFinite(pairing.expiresAt)) {
    return null
  }

  if (Date.now() > pairing.expiresAt) {
    await clearTelegramPendingPairing(dataDir)
    return null
  }

  return pairing
}

type ApplyConfigMutation = <T>(
  description: string,
  fn: () => Promise<T>,
) => Promise<T>

export function createPersistingChannelUserPairHandler(options: {
  config?: SepilotdConfig
  getConfig?: () => SepilotdConfig
  dataDir: string
  addAllowedUser?: (channelType: string, userId: string) => void
  applyConfigMutation?: ApplyConfigMutation
  persistConfig?: (config: SepilotdConfig) => Promise<void>
}): (channelType: string, userId: string) => Promise<void> {
  return async (channelType: string, userId: string) => {
    if (channelType !== 'telegram') {
      return
    }

    const persistPairing = async (): Promise<void> => {
      const config = options.getConfig?.() ?? options.config
      if (!config) {
        throw new Error('Runtime config is not available')
      }

      const changed = appendTelegramAllowedUser(config, userId)
      if (!changed) {
        return
      }

      options.addAllowedUser?.(channelType, userId)
      if (options.persistConfig) {
        await options.persistConfig(config)
      } else {
        await persistChannelsConfig(options.dataDir, config)
      }
    }

    if (options.applyConfigMutation) {
      await options.applyConfigMutation(
        'config.channels.telegram.pair-user',
        persistPairing,
      )
      return
    }

    await persistPairing()
  }
}
