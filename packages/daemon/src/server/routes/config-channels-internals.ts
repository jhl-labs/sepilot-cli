import type { ChannelSummaryCapabilities } from '../runtime/capabilities.js'
import type { IChannel } from '@sepilotd/core'
import type { z } from 'zod'
import type { SepilotdConfig } from '../../config/schema.js'
import {
  normalizeWebhookEndpointPath,
  webhookEndpointIdFromPath,
  webhookPublicRouteFromPath,
  type WebhookEndpointConfig,
} from '../../channels/webhook.js'
import {
  discordChannelConfigSchema,
  mattermostChannelConfigSchema,
  slackChannelConfigSchema,
  type discordChannelResponseSchema,
  type mattermostChannelResponseSchema,
  type slackChannelResponseSchema,
  telegramChannelConfigSchema,
  type telegramChannelResponseSchema,
  type telegramPairingCodeSchema,
  type webhookEndpointListResponseSchema,
  webhookEndpointSchema,
  type DiscordChannelBody,
  type MattermostChannelBody,
  type SlackChannelBody,
  type TelegramChannelBody,
} from './config-schema.js'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function isSensitiveChannelConfigKey(key: string): boolean {
  const normalized = key.toLowerCase()
  if (normalized === 'secretheader') {
    return false
  }
  return normalized === 'apikey'
    || normalized === 'secretvalue'
    || normalized.endsWith('token')
    || normalized.endsWith('secret')
    || normalized.endsWith('password')
}

export function redactChannelConfigValue(value: unknown, key?: string): unknown {
  if (typeof value === 'string' && key && isSensitiveChannelConfigKey(key)) {
    return '***redacted***'
  }
  if (Array.isArray(value)) {
    return value.map((entry) => redactChannelConfigValue(entry))
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([entryKey, entryValue]) => [
        entryKey,
        redactChannelConfigValue(entryValue, entryKey),
      ]),
    )
  }
  return value
}

/**
 * When the desktop UI sends a channel update with `***redacted***` literals
 * (because the input was prefilled from a redacted GET response), substitute
 * the existing stored value back so the secret is not clobbered.
 *
 * Matches by channel `type` + (when present) `id`. If no matching existing
 * channel is found the redacted literal is dropped (treated as unset).
 */
function restoreSensitiveStrings(
  existingValue: unknown,
  nextValue: unknown,
  key?: string,
): unknown {
  if (typeof nextValue === 'string') {
    if (
      key
      && isSensitiveChannelConfigKey(key)
      && nextValue === '***redacted***'
    ) {
      return typeof existingValue === 'string' ? existingValue : ''
    }
    return nextValue
  }
  if (Array.isArray(nextValue)) {
    if (Array.isArray(existingValue)) {
      return nextValue.map((entry, index) =>
        restoreSensitiveStrings(existingValue[index], entry),
      )
    }
    return nextValue.map((entry) => restoreSensitiveStrings(undefined, entry))
  }
  if (nextValue && typeof nextValue === 'object') {
    const existingObj = isRecord(existingValue) ? existingValue : {}
    return Object.fromEntries(
      Object.entries(nextValue).map(([entryKey, entryValue]) => [
        entryKey,
        restoreSensitiveStrings(existingObj[entryKey], entryValue, entryKey),
      ]),
    )
  }
  return nextValue
}

export function restoreChannelConfigSecrets<
  T extends { type?: string; config?: unknown; id?: string | number },
>(existing: readonly T[] | undefined, next: readonly T[]): T[] {
  const existingArr = existing ?? []
  return next.map((channel, index) => {
    // Prefer id-based match when both sides agree on an id. The
    // current persisted schema has no id field but downstream code
    // may grow one; matching by id stays correct under reorder.
    let match: T | undefined
    if (channel.id != null) {
      match = existingArr.find(
        (candidate) =>
          candidate.type === channel.type && candidate.id === channel.id,
      )
    }
    // Fall back to position-based match (same index, same type).
    // This is the realistic default because the desktop UI submits
    // the whole channel array verbatim. Matching purely by `type`
    // would alias multiple same-type channels (e.g. two telegram
    // bots) onto the first existing entry, leaking that bot's
    // secrets into a sibling channel's update.
    if (!match) {
      const positional = existingArr[index]
      if (positional && positional.type === channel.type) {
        match = positional
      }
    }
    return {
      ...channel,
      // Walk regardless of match: when there is a match, redacted literals
      // are restored; when there is no match (e.g. a brand-new channel that
      // somehow contains a `***redacted***` placeholder), the literal is
      // cleared rather than being persisted as the secret value.
      config: restoreSensitiveStrings(match?.config, channel.config),
    }
  })
}

export function listConfiguredWebhookEndpoints(
  config: SepilotdConfig,
): WebhookEndpointConfig[] {
  const endpoints: WebhookEndpointConfig[] = []
  for (const channel of config.channels ?? []) {
    if (channel.type !== 'webhook') {
      continue
    }
    const channelConfig = channel.config as Record<string, unknown> | undefined
    const rawEndpoints = channelConfig?.endpoints
    if (!Array.isArray(rawEndpoints)) {
      continue
    }
    for (const rawEndpoint of rawEndpoints) {
      const parsed = webhookEndpointSchema.safeParse(rawEndpoint)
      if (!parsed.success) {
        continue
      }
      endpoints.push({
        ...parsed.data,
        path: normalizeWebhookEndpointPath(parsed.data.path),
      })
    }
  }
  return endpoints
}

export function currentWebhookChannelRateLimit(
  config: SepilotdConfig,
): number | undefined {
  for (const channel of config.channels ?? []) {
    if (channel.type !== 'webhook') {
      continue
    }
    const rateLimit = (channel.config as Record<string, unknown> | undefined)?.rateLimitPerMinute
    if (typeof rateLimit === 'number') {
      return rateLimit
    }
  }
  return undefined
}

export function replaceConfiguredWebhookEndpoints(
  config: SepilotdConfig,
  endpoints: readonly WebhookEndpointConfig[],
): void {
  const nextChannels = (config.channels ?? []).filter(
    (channel) => channel.type !== 'webhook',
  )
  if (endpoints.length > 0) {
    const rateLimitPerMinute = currentWebhookChannelRateLimit(config)
    nextChannels.push({
      type: 'webhook',
      enabled: endpoints.some((endpoint) => endpoint.enabled !== false),
      config: {
        ...(rateLimitPerMinute != null ? { rateLimitPerMinute } : {}),
        endpoints: endpoints.map((endpoint) => ({
          ...endpoint,
          path: normalizeWebhookEndpointPath(endpoint.path),
        })),
      },
    })
  }
  config.channels = nextChannels
}

export function listWebhookEndpointSummaries(
  config: SepilotdConfig,
): z.infer<typeof webhookEndpointListResponseSchema>['data'] {
  return listConfiguredWebhookEndpoints(config).map((endpoint) => ({
    id: webhookEndpointIdFromPath(endpoint.path),
    enabled: endpoint.enabled !== false,
    path: normalizeWebhookEndpointPath(endpoint.path),
    publicRoute: webhookPublicRouteFromPath(endpoint.path),
    secretHeader: endpoint.secretHeader,
    hasSecretValue: endpoint.secretValue.length > 0,
    allowedIps: [...(endpoint.allowedIps ?? [])],
    allowedEvents: [...(endpoint.allowedEvents ?? [])],
  }))
}

export function findWebhookEndpointIndexById(
  config: SepilotdConfig,
  id: string,
): number {
  return listConfiguredWebhookEndpoints(config).findIndex(
    (endpoint) => webhookEndpointIdFromPath(endpoint.path) === id,
  )
}

export function readConfiguredSlackChannel(
  config: SepilotdConfig,
): SlackChannelBody | null {
  const channel = (config.channels ?? []).find((entry) => entry.type === 'slack')
  if (!channel) {
    return null
  }

  const channelConfig = isRecord(channel.config) ? channel.config : {}
  const parsed = slackChannelConfigSchema.safeParse({
    enabled: channel.enabled !== false,
    ...channelConfig,
  })
  return parsed.success ? parsed.data : null
}

export function replaceConfiguredSlackChannel(
  config: SepilotdConfig,
  channel: SlackChannelBody,
): void {
  const nextChannels = (config.channels ?? []).filter(
    (entry) => entry.type !== 'slack',
  )

  nextChannels.push({
    type: 'slack',
    enabled: channel.enabled,
    config: {
      botToken: channel.botToken,
      signingSecret: channel.signingSecret,
      allowedChannels: [...channel.allowedChannels],
      allowedUsers: [...channel.allowedUsers],
    },
  })

  config.channels = nextChannels
}

export function summarizeSlackChannel(
  runtime: ChannelSummaryCapabilities | undefined,
): z.infer<typeof slackChannelResponseSchema>['data'] {
  if (!runtime) {
    return null
  }

  const config = readConfiguredSlackChannel(runtime.config)
  if (!config) {
    return null
  }

  const activeChannel = Array.isArray(runtime.channels)
    ? runtime.channels.find((entry) => entry.type === 'slack')
    : undefined

  return {
    enabled: config.enabled,
    status: activeChannel?.getStatus() ?? 'disconnected',
    hasBotToken: config.botToken.length > 0,
    hasSigningSecret: config.signingSecret.length > 0,
    allowedChannels: [...config.allowedChannels],
    allowedUsers: [...config.allowedUsers],
  }
}

export function readConfiguredDiscordChannel(
  config: SepilotdConfig,
): DiscordChannelBody | null {
  const channel = (config.channels ?? []).find((entry) => entry.type === 'discord')
  if (!channel) {
    return null
  }

  const channelConfig = isRecord(channel.config) ? channel.config : {}
  const parsed = discordChannelConfigSchema.safeParse({
    enabled: channel.enabled !== false,
    ...channelConfig,
  })
  return parsed.success ? parsed.data : null
}

export function replaceConfiguredDiscordChannel(
  config: SepilotdConfig,
  channel: DiscordChannelBody,
): void {
  const nextChannels = (config.channels ?? []).filter(
    (entry) => entry.type !== 'discord',
  )

  nextChannels.push({
    type: 'discord',
    enabled: channel.enabled,
    config: {
      botToken: channel.botToken,
      applicationId: channel.applicationId,
      publicKey: channel.publicKey,
      allowedGuilds: [...channel.allowedGuilds],
      allowedChannels: [...channel.allowedChannels],
      allowedUsers: [...channel.allowedUsers],
    },
  })

  config.channels = nextChannels
}

export function summarizeDiscordChannel(
  runtime: ChannelSummaryCapabilities | undefined,
): z.infer<typeof discordChannelResponseSchema>['data'] {
  if (!runtime) {
    return null
  }

  const config = readConfiguredDiscordChannel(runtime.config)
  if (!config) {
    return null
  }

  const activeChannel = Array.isArray(runtime.channels)
    ? runtime.channels.find((entry) => entry.type === 'discord')
    : undefined

  return {
    enabled: config.enabled,
    status: activeChannel?.getStatus() ?? 'disconnected',
    hasBotToken: config.botToken.length > 0,
    hasApplicationId: config.applicationId.length > 0,
    hasPublicKey: config.publicKey.length > 0,
    allowedGuilds: [...config.allowedGuilds],
    allowedChannels: [...config.allowedChannels],
    allowedUsers: [...config.allowedUsers],
  }
}

export function readConfiguredMattermostChannel(
  config: SepilotdConfig,
): MattermostChannelBody | null {
  const channel = (config.channels ?? []).find((entry) => entry.type === 'mattermost')
  if (!channel) {
    return null
  }

  const channelConfig = isRecord(channel.config) ? channel.config : {}
  const parsed = mattermostChannelConfigSchema.safeParse({
    enabled: channel.enabled !== false,
    ...channelConfig,
  })
  return parsed.success ? parsed.data : null
}

export function replaceConfiguredMattermostChannel(
  config: SepilotdConfig,
  channel: MattermostChannelBody,
): void {
  const nextChannels = (config.channels ?? []).filter(
    (entry) => entry.type !== 'mattermost',
  )

  nextChannels.push({
    type: 'mattermost',
    enabled: channel.enabled,
    config: {
      serverUrl: channel.serverUrl,
      botToken: channel.botToken,
      webhookToken: channel.webhookToken,
      allowedTeams: [...channel.allowedTeams],
      allowedChannels: [...channel.allowedChannels],
      allowedUsers: [...channel.allowedUsers],
    },
  })

  config.channels = nextChannels
}

export function summarizeMattermostChannel(
  runtime: ChannelSummaryCapabilities | undefined,
): z.infer<typeof mattermostChannelResponseSchema>['data'] {
  if (!runtime) {
    return null
  }

  const config = readConfiguredMattermostChannel(runtime.config)
  if (!config) {
    return null
  }

  const activeChannel = Array.isArray(runtime.channels)
    ? runtime.channels.find((entry) => entry.type === 'mattermost')
    : undefined

  return {
    enabled: config.enabled,
    status: activeChannel?.getStatus() ?? 'disconnected',
    hasServerUrl: config.serverUrl.length > 0,
    hasBotToken: config.botToken.length > 0,
    hasWebhookToken: config.webhookToken.length > 0,
    serverUrl: config.serverUrl,
    allowedTeams: [...config.allowedTeams],
    allowedChannels: [...config.allowedChannels],
    allowedUsers: [...config.allowedUsers],
  }
}

export function readConfiguredTelegramChannel(
  config: SepilotdConfig,
): TelegramChannelBody | null {
  const channel = (config.channels ?? []).find((entry) => entry.type === 'telegram')
  if (!channel) {
    return null
  }

  const channelConfig = isRecord(channel.config) ? channel.config : {}
  const parsed = telegramChannelConfigSchema.safeParse({
    enabled: channel.enabled !== false,
    ...channelConfig,
  })
  return parsed.success ? parsed.data : null
}

export function replaceConfiguredTelegramChannel(
  config: SepilotdConfig,
  channel: TelegramChannelBody,
): void {
  const nextChannels = (config.channels ?? []).filter(
    (entry) => entry.type !== 'telegram',
  )

  nextChannels.push({
    type: 'telegram',
    enabled: channel.enabled,
    config: {
      botToken: channel.botToken,
      allowedUsers: [...channel.allowedUsers],
      pairingRequired: channel.pairingRequired,
      pairingCodeTtl: channel.pairingCodeTtl,
      rateLimitPerMinute: channel.rateLimitPerMinute,
    },
  })

  config.channels = nextChannels
}

export function summarizeTelegramChannel(
  runtime: ChannelSummaryCapabilities | undefined,
): z.infer<typeof telegramChannelResponseSchema>['data'] {
  if (!runtime) {
    return null
  }

  const config = readConfiguredTelegramChannel(runtime.config)
  if (!config) {
    return null
  }

  const activeChannel = Array.isArray(runtime.channels)
    ? runtime.channels.find((entry) => entry.type === 'telegram')
    : undefined

  return {
    enabled: config.enabled,
    status: activeChannel?.getStatus() ?? 'disconnected',
    hasBotToken: config.botToken.length > 0,
    allowedUsers: [...config.allowedUsers],
    pairingRequired: config.pairingRequired,
    pairingCodeTtl: config.pairingCodeTtl,
    rateLimitPerMinute: config.rateLimitPerMinute,
  }
}

export function getTelegramPairingRuntimeChannel(
  runtime: ChannelSummaryCapabilities | undefined,
): { generatePairingCode: () => z.infer<typeof telegramPairingCodeSchema> } | null {
  if (!runtime || !Array.isArray(runtime.channels)) {
    return null
  }

  const channel = runtime.channels.find(
    (entry): entry is IChannel & { generatePairingCode: () => z.infer<typeof telegramPairingCodeSchema> } =>
      entry.type === 'telegram'
      && typeof (entry as IChannel & { generatePairingCode?: unknown }).generatePairingCode === 'function',
  )

  if (!channel) {
    return null
  }

  return channel
}

export function getTelegramMutableRuntimeChannel(
  runtime: ChannelSummaryCapabilities | undefined,
): { revokeAllowedUser: (userId: string) => boolean } | null {
  if (!runtime || !Array.isArray(runtime.channels)) {
    return null
  }

  const channel = runtime.channels.find(
    (entry): entry is IChannel & { revokeAllowedUser: (userId: string) => boolean } =>
      entry.type === 'telegram'
      && typeof (entry as IChannel & { revokeAllowedUser?: unknown }).revokeAllowedUser === 'function',
  )

  if (!channel) {
    return null
  }

  return channel
}


export function redactProviderApiKey(value: string | undefined): string | undefined {
  if (!value) {
    return undefined
  }

  return /\$\{[^}]+\}/.test(value)
    ? value
    : '***redacted***'
}

export function restoreWebhookEndpointSecret(
  existingEndpoints: readonly WebhookEndpointConfig[],
  endpoint: WebhookEndpointConfig,
): WebhookEndpointConfig {
  if (endpoint.secretValue !== '***redacted***') {
    return endpoint
  }

  const existingEndpoint = existingEndpoints.find(
    (candidate) =>
      webhookEndpointIdFromPath(candidate.path)
      === webhookEndpointIdFromPath(normalizeWebhookEndpointPath(endpoint.path)),
  )
  if (!existingEndpoint?.secretValue) {
    return endpoint
  }

  return {
    ...endpoint,
    secretValue: existingEndpoint.secretValue,
  }
}

export function restoreSlackChannelSecrets(
  existingChannel: SlackChannelBody | null,
  channel: SlackChannelBody,
): SlackChannelBody {
  let nextChannel = channel

  if (nextChannel.botToken === '***redacted***' && existingChannel?.botToken) {
    nextChannel = {
      ...nextChannel,
      botToken: existingChannel.botToken,
    }
  }

  if (
    nextChannel.signingSecret === '***redacted***'
    && existingChannel?.signingSecret
  ) {
    nextChannel = {
      ...nextChannel,
      signingSecret: existingChannel.signingSecret,
    }
  }

  return nextChannel
}

export function restoreDiscordChannelSecrets(
  existingChannel: DiscordChannelBody | null,
  channel: DiscordChannelBody,
): DiscordChannelBody {
  let nextChannel = channel

  if (nextChannel.botToken === '***redacted***' && existingChannel?.botToken) {
    nextChannel = {
      ...nextChannel,
      botToken: existingChannel.botToken,
    }
  }

  return nextChannel
}

export function restoreMattermostChannelSecrets(
  existingChannel: MattermostChannelBody | null,
  channel: MattermostChannelBody,
): MattermostChannelBody {
  let nextChannel = channel

  if (nextChannel.botToken === '***redacted***' && existingChannel?.botToken) {
    nextChannel = {
      ...nextChannel,
      botToken: existingChannel.botToken,
    }
  }

  if (
    nextChannel.webhookToken === '***redacted***'
    && existingChannel?.webhookToken
  ) {
    nextChannel = {
      ...nextChannel,
      webhookToken: existingChannel.webhookToken,
    }
  }

  return nextChannel
}

export function restoreTelegramBotToken(
  existingChannel: TelegramChannelBody | null,
  channel: TelegramChannelBody,
): TelegramChannelBody {
  if (channel.botToken !== '***redacted***') {
    return channel
  }
  if (!existingChannel?.botToken) {
    return channel
  }

  return {
    ...channel,
    botToken: existingChannel.botToken,
  }
}
