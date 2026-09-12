import chalk from 'chalk'
import type {
  DaemonMattermostChannelSummary,
  DaemonTelegramChannelSummary,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { resolveDaemonEndpointScope } from '../client/token.js'
import { output } from '../output/formatter.js'
import { detectCliLocale } from '../utils/locale.js'

const CHANNEL_COPY = {
  en: {
    unsupportedType: (type: string) =>
      `Unsupported channel type: ${type}. CLI bootstrap supports Telegram and Mattermost.`,
    telegramOnly: (operation: string) => `${operation} is only supported for Telegram.`,
    localMattermostOnly:
      'Mattermost secret bootstrap is limited to a local daemon. Use OpenBao/ExternalSecret for a remote or cluster daemon.',
    mattermostEnvRequired:
      'Mattermost credentials must be read from environment variables. Use --from-env with MATTERMOST_BOT_TOKEN and MATTERMOST_WEBHOOK_TOKEN.',
    mattermostServerRequired:
      '`--server-url <url>` or MATTERMOST_SERVER_URL is required for Mattermost.',
    invalidFlagValue: (flag: string, value: string) => `Invalid ${flag} value: ${value}`,
    enabled: 'enabled',
    disabled: 'disabled',
    pairingRequired: 'required',
    pairingOpen: 'open',
    noChannels: 'No channels configured.\nUse `sepilot channel add telegram --bot-token <token>` to add one.',
    botTokenRequired: '`--bot-token <token>` is required for Telegram.',
    channelUpdated: 'Telegram channel updated.',
    channelUpdatedWith: (summary: string) => `Telegram channel updated.\n${summary}`,
    mattermostUpdated: 'Mattermost channel updated.',
    mattermostUpdatedWith: (summary: string) => `Mattermost channel updated.\n${summary}`,
    channelRemoved: 'Telegram channel removed.',
    mattermostRemoved: 'Mattermost channel removed.',
    channelEnabled: 'Telegram channel enabled.',
    channelEnabledWith: (summary: string) => `Telegram channel enabled.\n${summary}`,
    mattermostEnabled: 'Mattermost channel enabled.',
    mattermostEnabledWith: (summary: string) => `Mattermost channel enabled.\n${summary}`,
    channelDisabled: 'Telegram channel disabled.',
    channelDisabledWith: (summary: string) => `Telegram channel disabled.\n${summary}`,
    mattermostDisabled: 'Mattermost channel disabled.',
    mattermostDisabledWith: (summary: string) => `Mattermost channel disabled.\n${summary}`,
    notConfigured: 'Telegram channel is not configured.\nUse `sepilot channel add telegram --bot-token <token>` first.',
    channelDisabledHint: 'Telegram channel is disabled.\nUse `sepilot channel enable telegram` first.',
    pairingDisabled: 'Telegram channel pairing is disabled.\nRe-add it without `--no-pairing-required` to generate pairing codes.',
    pairingEndpointMissing: 'Telegram pairing endpoint is not available on this daemon.',
    pairingEndpointHint: 'Make sure the daemon was built with channel support.',
    pairingCode: (code: string) => `Telegram pairing code: ${code}`,
    expires: (at: string) => `Expires: ${at}`,
    sendPairHint: (code: string) => `Send \`/pair ${code}\` to your Telegram bot.`,
    noPairedUsers: 'No paired Telegram users.',
    pairedUsersHeader: 'Paired Telegram users:',
    userNotPaired: (id: string) => `Telegram user not paired: ${id}`,
    userUnpaired: (id: string) => `Telegram user unpaired: ${id}`,
  },
  ko: {
    unsupportedType: (type: string) =>
      `지원되지 않는 채널 유형: ${type}. CLI 부트스트랩은 Telegram과 Mattermost를 지원합니다.`,
    telegramOnly: (operation: string) => `${operation} 작업은 Telegram에서만 지원됩니다.`,
    localMattermostOnly:
      'Mattermost 비밀 부트스트랩은 로컬 daemon에서만 허용됩니다. 원격/클러스터 daemon은 OpenBao/ExternalSecret을 사용하세요.',
    mattermostEnvRequired:
      'Mattermost 자격증명은 환경변수에서 읽어야 합니다. MATTERMOST_BOT_TOKEN과 MATTERMOST_WEBHOOK_TOKEN을 설정하고 --from-env를 사용하세요.',
    mattermostServerRequired:
      'Mattermost에는 `--server-url <url>` 또는 MATTERMOST_SERVER_URL이 필요합니다.',
    invalidFlagValue: (flag: string, value: string) => `잘못된 ${flag} 값: ${value}`,
    enabled: '활성화됨',
    disabled: '비활성화됨',
    pairingRequired: '필수',
    pairingOpen: '공개',
    noChannels: '구성된 채널이 없습니다.\n`sepilot channel add telegram --bot-token <토큰>`으로 추가하세요.',
    botTokenRequired: 'Telegram에는 `--bot-token <토큰>`이 필요합니다.',
    channelUpdated: 'Telegram 채널이 업데이트되었습니다.',
    channelUpdatedWith: (summary: string) => `Telegram 채널이 업데이트되었습니다.\n${summary}`,
    mattermostUpdated: 'Mattermost 채널이 업데이트되었습니다.',
    mattermostUpdatedWith: (summary: string) => `Mattermost 채널이 업데이트되었습니다.\n${summary}`,
    channelRemoved: 'Telegram 채널이 제거되었습니다.',
    mattermostRemoved: 'Mattermost 채널이 제거되었습니다.',
    channelEnabled: 'Telegram 채널이 활성화되었습니다.',
    channelEnabledWith: (summary: string) => `Telegram 채널이 활성화되었습니다.\n${summary}`,
    mattermostEnabled: 'Mattermost 채널이 활성화되었습니다.',
    mattermostEnabledWith: (summary: string) => `Mattermost 채널이 활성화되었습니다.\n${summary}`,
    channelDisabled: 'Telegram 채널이 비활성화되었습니다.',
    channelDisabledWith: (summary: string) => `Telegram 채널이 비활성화되었습니다.\n${summary}`,
    mattermostDisabled: 'Mattermost 채널이 비활성화되었습니다.',
    mattermostDisabledWith: (summary: string) => `Mattermost 채널이 비활성화되었습니다.\n${summary}`,
    notConfigured: 'Telegram 채널이 구성되지 않았습니다.\n먼저 `sepilot channel add telegram --bot-token <토큰>`을 사용하세요.',
    channelDisabledHint: 'Telegram 채널이 비활성화되어 있습니다.\n먼저 `sepilot channel enable telegram`을 사용하세요.',
    pairingDisabled: 'Telegram 채널 페어링이 비활성화되어 있습니다.\n페어링 코드를 생성하려면 `--no-pairing-required` 없이 다시 추가하세요.',
    pairingEndpointMissing: '이 daemon에서는 Telegram 페어링 엔드포인트를 사용할 수 없습니다.',
    pairingEndpointHint: 'daemon이 채널 지원과 함께 빌드되었는지 확인하세요.',
    pairingCode: (code: string) => `Telegram 페어링 코드: ${code}`,
    expires: (at: string) => `만료: ${at}`,
    sendPairHint: (code: string) => `Telegram 봇에게 \`/pair ${code}\`를 보내세요.`,
    noPairedUsers: '페어링된 Telegram 사용자가 없습니다.',
    pairedUsersHeader: '페어링된 Telegram 사용자:',
    userNotPaired: (id: string) => `Telegram 사용자가 페어링되지 않음: ${id}`,
    userUnpaired: (id: string) => `Telegram 사용자 페어링 해제됨: ${id}`,
  },
} as const

type ChannelCopy = (typeof CHANNEL_COPY)[keyof typeof CHANNEL_COPY]

function channelCopy(): ChannelCopy {
  return CHANNEL_COPY[detectCliLocale()] ?? CHANNEL_COPY.en
}

export interface ChannelMutationOptions {
  url?: string
}

export interface TelegramChannelAddOptions extends ChannelMutationOptions {
  botToken?: string
  user?: string[]
  pairingRequired?: boolean
  pairingCodeTtl?: string
  rateLimitPerMinute?: string
  disabled?: boolean
  fromEnv?: boolean
  serverUrl?: string
  team?: string[]
  channel?: string[]
}

function assertSupportedChannelType(channelType: string): asserts channelType is 'telegram' | 'mattermost' {
  if (channelType === 'telegram' || channelType === 'mattermost') {
    return
  }

  throw new Error(channelCopy().unsupportedType(channelType))
}

function assertTelegramOperation(channelType: string, operation: string): void {
  if (channelType === 'telegram') return
  if (channelType === 'mattermost') throw new Error(channelCopy().telegramOnly(operation))
  throw new Error(channelCopy().unsupportedType(channelType))
}

function assertLocalMattermostDaemon(explicitUrl: string | undefined): void {
  try {
    if (resolveDaemonEndpointScope(explicitUrl) === 'loopback') return
  } catch {
    const resolved = explicitUrl?.trim() || process.env.SEPILOTD_URL?.trim() || '<default>'
    throw new Error(`Invalid daemon URL: ${resolved}`)
  }
  throw new Error(channelCopy().localMattermostOnly)
}

function parseOptionalInt(value: string | undefined, flagName: string): number | undefined {
  if (!value) {
    return undefined
  }

  const parsed = Number.parseInt(value, 10)
  if (!Number.isFinite(parsed)) {
    throw new Error(channelCopy().invalidFlagValue(flagName, value))
  }
  return parsed
}

function formatTelegramChannelSummary(
  summary: DaemonTelegramChannelSummary,
): string {
  const copy = channelCopy()
  const enabled = summary.enabled
    ? chalk.green(copy.enabled)
    : chalk.gray(copy.disabled)
  const status = summary.status === 'connected'
    ? chalk.green(summary.status)
    : summary.status === 'error'
      ? chalk.red(summary.status)
      : summary.status === 'connecting'
        ? chalk.yellow(summary.status)
        : chalk.gray(summary.status)
  const extras = [
    `pairing=${summary.pairingRequired ? copy.pairingRequired : copy.pairingOpen}`,
    `users=${summary.allowedUsers.length}`,
    `ttl=${summary.pairingCodeTtl}s`,
    `rate=${summary.rateLimitPerMinute}/min`,
  ]

  // Two stat columns labelled — `admin=enabled` (operator toggle) vs
  // `link=connected` (transport health) — were ambiguous side-by-side.
  return `  ${chalk.bold('telegram'.padEnd(10))}  admin=${enabled}  link=${status}  ${chalk.gray(`[${extras.join(' ')}]`)}`
}

function formatMattermostChannelSummary(
  summary: DaemonMattermostChannelSummary,
): string {
  const copy = channelCopy()
  const enabled = summary.enabled
    ? chalk.green(copy.enabled)
    : chalk.gray(copy.disabled)
  const status = summary.status === 'connected'
    ? chalk.green(summary.status)
    : summary.status === 'error'
      ? chalk.red(summary.status)
      : summary.status === 'connecting'
        ? chalk.yellow(summary.status)
        : chalk.gray(summary.status)
  const ready = summary.hasServerUrl && summary.hasBotToken && summary.hasWebhookToken
  const extras = [
    `credentials=${ready ? 'ready' : 'incomplete'}`,
    `teams=${summary.allowedTeams.length}`,
    `channels=${summary.allowedChannels.length}`,
    `users=${summary.allowedUsers.length}`,
  ]
  return `  ${chalk.bold('mattermost'.padEnd(10))}  admin=${enabled}  link=${status}  ${chalk.gray(`[${extras.join(' ')}]`)}`
}

function formatGenericChannelLine(channel: { type: string; enabled?: boolean }): string {
  const copy = channelCopy()
  const enabled = channel.enabled !== false
    ? chalk.green(copy.enabled)
    : chalk.gray(copy.disabled)
  return `  ${channel.type.padEnd(15)} ${enabled}`
}

export async function channelListCommand(options: ChannelMutationOptions) {
  const copy = channelCopy()
  const client = new DaemonClient(options.url)
  const config = await client.config()
  const hasTelegram = config.channels.some((channel) => channel.type === 'telegram')
  const hasMattermost = config.channels.some((channel) => channel.type === 'mattermost')
  const [telegram, mattermost] = await Promise.all([
    hasTelegram ? client.telegramChannel() : Promise.resolve(null),
    hasMattermost ? client.mattermostChannel() : Promise.resolve(null),
  ])

  output(config.channels ?? [], (channels) => {
    if (!channels.length) {
      return copy.noChannels
    }

    return channels.map((channel) => {
      if (channel.type === 'telegram' && telegram) {
        return formatTelegramChannelSummary(telegram)
      }
      if (channel.type === 'mattermost' && mattermost) {
        return formatMattermostChannelSummary(mattermost)
      }
      return formatGenericChannelLine(channel)
    }).join('\n')
  })
}

export async function channelAddCommand(
  channelType: string,
  options: TelegramChannelAddOptions,
) {
  const copy = channelCopy()
  assertSupportedChannelType(channelType)

  const client = new DaemonClient(options.url)
  if (channelType === 'mattermost') {
    if (!options.fromEnv) throw new Error(copy.mattermostEnvRequired)
    assertLocalMattermostDaemon(options.url)
    const serverUrl = options.serverUrl?.trim() || process.env.MATTERMOST_SERVER_URL?.trim()
    const botToken = process.env.MATTERMOST_BOT_TOKEN?.trim()
    const webhookToken = process.env.MATTERMOST_WEBHOOK_TOKEN?.trim()
    if (!serverUrl) throw new Error(copy.mattermostServerRequired)
    if (!botToken || !webhookToken) throw new Error(copy.mattermostEnvRequired)

    await client.upsertMattermostChannel({
      enabled: options.disabled ? false : true,
      serverUrl,
      botToken,
      webhookToken,
      allowedTeams: options.team ?? [],
      allowedChannels: options.channel ?? [],
      allowedUsers: options.user ?? [],
    })
    const summary = await client.mattermostChannel()
    output(
      { type: channelType, action: 'upserted', summary },
      (result) => result.summary
        ? copy.mattermostUpdatedWith(formatMattermostChannelSummary(result.summary))
        : copy.mattermostUpdated,
    )
    return
  }

  if (!options.botToken?.trim()) throw new Error(copy.botTokenRequired)
  await client.upsertTelegramChannel({
    enabled: options.disabled ? false : true,
    botToken: options.botToken.trim(),
    allowedUsers: options.user ?? [],
    pairingRequired: options.pairingRequired,
    pairingCodeTtl: parseOptionalInt(options.pairingCodeTtl, '--pairing-code-ttl'),
    rateLimitPerMinute: parseOptionalInt(
      options.rateLimitPerMinute,
      '--rate-limit-per-minute',
    ),
  })

  const summary = await client.telegramChannel()
  output(
    {
      type: channelType,
      action: 'upserted',
      summary,
    },
    (result) => {
      if (!result.summary) {
        return copy.channelUpdated
      }
      return copy.channelUpdatedWith(formatTelegramChannelSummary(result.summary))
    },
  )
}

export async function channelRemoveCommand(
  channelType: string,
  options: ChannelMutationOptions,
) {
  const copy = channelCopy()
  assertSupportedChannelType(channelType)
  const client = new DaemonClient(options.url)
  if (channelType === 'mattermost') await client.deleteMattermostChannel()
  else await client.deleteTelegramChannel()
  output(
    {
      type: channelType,
      action: 'removed',
    },
    () => channelType === 'mattermost' ? copy.mattermostRemoved : copy.channelRemoved,
  )
}

export async function channelEnableCommand(
  channelType: string,
  options: ChannelMutationOptions,
) {
  const copy = channelCopy()
  assertSupportedChannelType(channelType)
  const client = new DaemonClient(options.url)
  if (channelType === 'mattermost') await client.setMattermostChannelEnabled(true)
  else await client.setTelegramChannelEnabled(true)
  const summary = channelType === 'mattermost'
    ? await client.mattermostChannel()
    : await client.telegramChannel()
  output(
    {
      type: channelType,
      action: 'enabled',
      summary,
    },
    (result) => {
      if (!result.summary) return channelType === 'mattermost'
        ? copy.mattermostEnabled
        : copy.channelEnabled
      return channelType === 'mattermost'
        ? copy.mattermostEnabledWith(formatMattermostChannelSummary(result.summary as DaemonMattermostChannelSummary))
        : copy.channelEnabledWith(formatTelegramChannelSummary(result.summary as DaemonTelegramChannelSummary))
    },
  )
}

export async function channelDisableCommand(
  channelType: string,
  options: ChannelMutationOptions,
) {
  const copy = channelCopy()
  assertSupportedChannelType(channelType)
  const client = new DaemonClient(options.url)
  if (channelType === 'mattermost') await client.setMattermostChannelEnabled(false)
  else await client.setTelegramChannelEnabled(false)
  const summary = channelType === 'mattermost'
    ? await client.mattermostChannel()
    : await client.telegramChannel()
  output(
    {
      type: channelType,
      action: 'disabled',
      summary,
    },
    (result) => {
      if (!result.summary) return channelType === 'mattermost'
        ? copy.mattermostDisabled
        : copy.channelDisabled
      return channelType === 'mattermost'
        ? copy.mattermostDisabledWith(formatMattermostChannelSummary(result.summary as DaemonMattermostChannelSummary))
        : copy.channelDisabledWith(formatTelegramChannelSummary(result.summary as DaemonTelegramChannelSummary))
    },
  )
}

export async function channelPairCommand(
  channelType: string,
  options: ChannelMutationOptions,
) {
  const copy = channelCopy()
  assertTelegramOperation(channelType, 'Pairing')

  const client = new DaemonClient(options.url)
  const summary = await client.telegramChannel()
  if (!summary) {
    output(
      { type: channelType, configured: false },
      () => copy.notConfigured,
    )
    return
  }

  if (!summary.enabled) {
    output(
      { type: channelType, enabled: false },
      () => copy.channelDisabledHint,
    )
    return
  }

  if (!summary.pairingRequired) {
    output(
      { type: channelType, pairingRequired: false },
      () => copy.pairingDisabled,
    )
    return
  }

  let pairing
  try {
    pairing = await client.telegramPairingCode()
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    if (/\b404\b|not_found/i.test(msg)) {
      console.error(chalk.red(copy.pairingEndpointMissing))
      console.error(chalk.gray(copy.pairingEndpointHint))
      process.exit(1)
    }
    throw err
  }
  output(
    {
      type: channelType,
      pairing,
    },
    (result) => [
      copy.pairingCode(chalk.yellow(result.pairing.code)),
      copy.expires(result.pairing.expiresAt),
      copy.sendPairHint(result.pairing.code),
    ].join('\n'),
  )
}

export async function channelUsersCommand(
  channelType: string,
  options: ChannelMutationOptions,
) {
  const copy = channelCopy()
  assertTelegramOperation(channelType, 'Listing paired users')

  const client = new DaemonClient(options.url)
  const summary = await client.telegramChannel()
  if (!summary) {
    output(
      { type: channelType, configured: false },
      () => copy.notConfigured,
    )
    return
  }

  output(
    {
      type: channelType,
      users: summary.allowedUsers,
    },
    (result) => {
      if (!result.users.length) {
        return copy.noPairedUsers
      }

      return [
        copy.pairedUsersHeader,
        ...result.users.map((userId: string) => `- ${userId}`),
      ].join('\n')
    },
  )
}

export async function channelUnpairCommand(
  channelType: string,
  userId: string,
  options: ChannelMutationOptions,
) {
  const copy = channelCopy()
  assertTelegramOperation(channelType, 'Unpairing users')

  const client = new DaemonClient(options.url)
  try {
    await client.revokeTelegramAllowedUser(userId)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    if (/\b404\b|not_found/i.test(msg)) {
      console.error(chalk.red(copy.userNotPaired(userId)))
      process.exit(1)
    }
    throw err
  }
  const summary = await client.telegramChannel()

  output(
    {
      type: channelType,
      action: 'unpaired',
      userId,
      summary,
    },
    (result) => {
      const lines = [copy.userUnpaired(result.userId)]
      if (result.summary) {
        lines.push(formatTelegramChannelSummary(result.summary))
      }
      return lines.join('\n')
    },
  )
}
