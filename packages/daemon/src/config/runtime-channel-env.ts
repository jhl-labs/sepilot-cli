import type { SepilotdConfig } from './schema.js'

export const MATTERMOST_RUNTIME_ENV_FLAG = 'SEPILOTD_MATTERMOST_FROM_ENV'

const MATTERMOST_REQUIRED_ENV = [
  'MATTERMOST_SERVER_URL',
  'MATTERMOST_BOT_TOKEN',
  'MATTERMOST_WEBHOOK_TOKEN',
] as const

const MATTERMOST_ALLOWLIST_ENV = [
  'SEPILOTD_MATTERMOST_ALLOWED_TEAMS',
  'SEPILOTD_MATTERMOST_ALLOWED_CHANNELS',
  'SEPILOTD_MATTERMOST_ALLOWED_USERS',
] as const

const MAX_ALLOWLIST_ENTRIES = 100
const MAX_ALLOWLIST_ENTRY_LENGTH = 160
const runtimeManagedChannelTypes = Symbol('sepilotd.runtimeManagedChannelTypes')

type RuntimeManagedConfig = SepilotdConfig & {
  [runtimeManagedChannelTypes]?: ReadonlySet<string>
}

export class RuntimeChannelEnvironmentError extends Error {
  readonly code = 'RUNTIME_CHANNEL_ENV_INVALID' as const

  constructor(message: string) {
    super(message)
    this.name = 'RuntimeChannelEnvironmentError'
  }
}

function trimmed(
  env: Record<string, string | undefined>,
  name: string,
): string | undefined {
  const value = env[name]?.trim()
  return value ? value : undefined
}

function enabledFlag(
  env: Record<string, string | undefined>,
  name: string,
): boolean {
  const value = env[name]?.trim().toLowerCase()
  if (!value || value === '0' || value === 'false') return false
  if (value === '1' || value === 'true') return true
  throw new RuntimeChannelEnvironmentError(
    `${name} must be one of 1, true, 0, or false.`,
  )
}

function parseAllowlist(
  env: Record<string, string | undefined>,
  name: string,
): string[] {
  const value = trimmed(env, name)
  if (!value) return []

  const entries = [...new Set(
    value
      .split(/[\n,]/)
      .map((entry) => entry.trim())
      .filter(Boolean),
  )]
  if (entries.length > MAX_ALLOWLIST_ENTRIES) {
    throw new RuntimeChannelEnvironmentError(
      `${name} may contain at most ${MAX_ALLOWLIST_ENTRIES} entries.`,
    )
  }
  if (entries.some((entry) => entry.length > MAX_ALLOWLIST_ENTRY_LENGTH)) {
    throw new RuntimeChannelEnvironmentError(
      `${name} entries may contain at most ${MAX_ALLOWLIST_ENTRY_LENGTH} characters.`,
    )
  }
  return entries
}

function setRuntimeManagedChannelTypes(
  config: SepilotdConfig,
  types: ReadonlySet<string>,
): void {
  Object.defineProperty(config, runtimeManagedChannelTypes, {
    configurable: true,
    enumerable: false,
    value: new Set(types),
    writable: false,
  })
}

export function copyRuntimeChannelEnvironmentMetadata(
  source: SepilotdConfig,
  target: SepilotdConfig,
): void {
  const types = (source as RuntimeManagedConfig)[runtimeManagedChannelTypes]
  if (types) setRuntimeManagedChannelTypes(target, types)
}

export function isRuntimeManagedChannel(
  config: SepilotdConfig,
  channelType: string,
): boolean {
  return Boolean(
    (config as RuntimeManagedConfig)[runtimeManagedChannelTypes]?.has(channelType),
  )
}

/**
 * Apply an operator-controlled Mattermost channel without persisting its
 * credentials. The opt-in is explicit so ordinary desktop/CLI processes that
 * happen to inherit a MATTERMOST_* variable do not silently grow a channel.
 */
export function applyRuntimeChannelEnvironment(
  config: SepilotdConfig,
  env: Record<string, string | undefined> = process.env,
): SepilotdConfig {
  if (!enabledFlag(env, MATTERMOST_RUNTIME_ENV_FLAG)) return config

  const missing = MATTERMOST_REQUIRED_ENV.filter((name) => !trimmed(env, name))
  if (missing.length > 0) {
    throw new RuntimeChannelEnvironmentError(
      `${MATTERMOST_RUNTIME_ENV_FLAG} is enabled but required variables are missing: ${missing.join(', ')}.`,
    )
  }

  const allowedTeams = parseAllowlist(env, MATTERMOST_ALLOWLIST_ENV[0])
  const allowedChannels = parseAllowlist(env, MATTERMOST_ALLOWLIST_ENV[1])
  const allowedUsers = parseAllowlist(env, MATTERMOST_ALLOWLIST_ENV[2])
  const allowUnscoped = enabledFlag(env, 'SEPILOTD_MATTERMOST_ALLOW_UNSCOPED')
  if (
    !allowUnscoped
    && allowedTeams.length === 0
    && allowedChannels.length === 0
    && allowedUsers.length === 0
  ) {
    throw new RuntimeChannelEnvironmentError(
      `Environment-managed Mattermost requires at least one of ${MATTERMOST_ALLOWLIST_ENV.join(', ')}. Set SEPILOTD_MATTERMOST_ALLOW_UNSCOPED=1 only for an intentional unrestricted development channel.`,
    )
  }

  const next: SepilotdConfig = {
    ...config,
    channels: [
      ...(config.channels ?? []).filter((channel) => channel.type !== 'mattermost'),
      {
        type: 'mattermost',
        enabled: true,
        config: {
          serverUrl: trimmed(env, 'MATTERMOST_SERVER_URL')!,
          botToken: trimmed(env, 'MATTERMOST_BOT_TOKEN')!,
          webhookToken: trimmed(env, 'MATTERMOST_WEBHOOK_TOKEN')!,
          allowedTeams,
          allowedChannels,
          allowedUsers,
          ...(trimmed(env, 'SEPILOTD_MATTERMOST_BOT_USER_ID')
            ? { botUserId: trimmed(env, 'SEPILOTD_MATTERMOST_BOT_USER_ID')! }
            : {}),
          ...(trimmed(env, 'SEPILOTD_MATTERMOST_BOT_USERNAME')
            ? { botUsername: trimmed(env, 'SEPILOTD_MATTERMOST_BOT_USERNAME')! }
            : {}),
        },
      },
    ],
  }
  setRuntimeManagedChannelTypes(next, new Set(['mattermost']))
  return next
}

/**
 * Merge a runtime channel update into the raw on-disk list. Environment-owned
 * types are always kept from the raw YAML (or omitted if absent), never copied
 * from the effective runtime config where credentials have been resolved.
 */
export function channelsForPersistence(
  config: SepilotdConfig,
  rawChannels: unknown,
  nextChannels: SepilotdConfig['channels'],
): SepilotdConfig['channels'] {
  const managed = (config as RuntimeManagedConfig)[runtimeManagedChannelTypes]
  if (!managed?.size) return nextChannels

  const managedRawChannels = Array.isArray(rawChannels)
    ? rawChannels.filter(
      (channel): channel is SepilotdConfig['channels'][number] =>
        typeof channel === 'object'
        && channel !== null
        && 'type' in channel
        && typeof channel.type === 'string'
        && managed.has(channel.type),
    )
    : []

  return [
    ...nextChannels.filter((channel) => !managed.has(channel.type)),
    ...managedRawChannels,
  ]
}

/** Keep operator-owned runtime channels effective across a full-document
 * settings update while accepting all non-managed channels from that update. */
export function preserveRuntimeManagedChannels(
  source: SepilotdConfig,
  target: SepilotdConfig,
): SepilotdConfig {
  const managed = (source as RuntimeManagedConfig)[runtimeManagedChannelTypes]
  if (!managed?.size) return target

  target.channels = [
    ...(target.channels ?? []).filter((channel) => !managed.has(channel.type)),
    ...(source.channels ?? []).filter((channel) => managed.has(channel.type)),
  ]
  setRuntimeManagedChannelTypes(target, managed)
  return target
}

export function configForPersistence(
  config: SepilotdConfig,
  rawChannels?: unknown,
): SepilotdConfig {
  const persisted = structuredClone(config) as SepilotdConfig
  persisted.channels = channelsForPersistence(config, rawChannels, config.channels)
  return persisted
}
