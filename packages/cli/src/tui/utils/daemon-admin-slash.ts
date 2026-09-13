import { mkdir, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type {
  ApiRequestInit,
  DaemonAgentDescriptor,
  DaemonBackupItem,
  DaemonConfig,
  DaemonConfigEnvUpdateInput,
  DaemonConfigUpdateInput,
  DaemonDevice,
  DaemonExtensionTokenScope,
  DaemonGitHubChatSyncConfig,
  DaemonGitHubChatSyncConfigUpdate,
  DaemonGitHubChatSyncResult,
  DaemonGitHubSyncRepo,
  DaemonHealth,
  DaemonIssueExtensionTokenInput,
  DaemonMessageSubscriptionItem,
  DaemonNetworkConfig,
  DaemonNotificationChannelSetting,
  DaemonNotificationItem,
  DaemonObservabilityRange,
  DaemonObservabilitySeverity,
  DaemonPersona,
  DaemonPersonalDoc,
  DaemonPromptTemplate,
  DaemonQuickInputSettings,
  DaemonSchedulerJob,
  DaemonSchedulerJobInput,
  DaemonSettingsJsonDocument,
  DaemonTeamDocsConfig,
  DaemonTeamDocsDocument,
  DaemonTelegramChannelSummary,
  DaemonWebhookEndpointSummary,
  JobsClient,
  MigrationClient,
} from '@sepilotd/api-client'
import { hasPendingScheduledRun } from '../../utils/scheduler-state.js'
import { runTasksCommand, type TasksClient } from '../../commands/tasks.js'
import { maskSecretValue, truncateJsonOutput } from './admin-output.js'

export interface DaemonAdminSlashClient {
  readonly jobs?: JobsClient
  readonly migration?: MigrationClient
  imageGenFile?(id: string): Promise<Blob>
  imageGenJob?(id: string): Promise<ImageGenJob>
  health(): Promise<DaemonHealth>
  systemShutdown(): Promise<{ status: string }>
  systemClients(): Promise<{
    count: number
    byKind: { ws: number; sse: number }
    idleMs: number
    clients: Array<{ id: string; kind: string; label: string; openedAt: number }>
  }>
  request<T = unknown>(path: string, init?: ApiRequestInit): Promise<T>
  config(): Promise<DaemonConfig>
  updateConfig(input: DaemonConfigUpdateInput): Promise<unknown>
  updateConfigEnv(input: DaemonConfigEnvUpdateInput): Promise<unknown>
  policy(): Promise<unknown>
  devices(): Promise<DaemonDevice[]>
  personas(): Promise<DaemonPersona[]>
  agents(): Promise<DaemonAgentDescriptor[]>
  createUserAgent(input: {
    id: string
    name?: string
    description: string
    base?: string
    model?: string
    temperature?: number
    maxIterations?: number
    systemPrompt: string
  }): Promise<DaemonAgentDescriptor>
  deleteUserAgent(id: string): Promise<void>
  userCommands(): Promise<
    Array<{
      id: string
      name: string
      description: string
      args: 'none' | 'optional' | 'required'
      agent?: string
      model?: string
    }>
  >
  createUserCommand(input: {
    id: string
    name?: string
    description: string
    args?: 'none' | 'optional' | 'required'
    agent?: string
    model?: string
    body: string
  }): Promise<{ id: string }>
  deleteUserCommand(id: string): Promise<void>
  providers(): Promise<
    Array<{ id: string; name: string; health: { status: string }; models: Array<{ id: string }> }>
  >
  skills(): Promise<Array<{ id: string; name: string; description?: string }>>
  listMarketplaces(): Promise<Array<{ name: string; url: string }>>
  addMarketplace(name: string, url: string): Promise<unknown>
  removeMarketplace(name: string): Promise<{ removed: boolean }>
  extensionTokens(): Promise<
    Array<{
      id: string
      label: string
      scopes: string[]
      active: boolean
      expiresAt?: string
    }>
  >
  issueExtensionToken(
    input: DaemonIssueExtensionTokenInput,
  ): Promise<{ id: string; label: string; token: string }>
  revokeExtensionToken(
    id: string,
  ): Promise<{ id: string; label: string; scopes: string[]; active: boolean }>
  listSecrets(): Promise<{ keys: string[] }>
  setSecret(key: string, value: string): Promise<{ ok: boolean }>
  removeSecret(key: string): Promise<{ removed: boolean }>
  webhookEndpoints(): Promise<DaemonWebhookEndpointSummary[]>
  upsertWebhookEndpoint(input: {
    path: string
    secretHeader: string
    secretValue: string
    allowedEvents?: string[]
    allowedIps?: string[]
    enabled?: boolean
  }): Promise<unknown>
  deleteWebhookEndpoint(id: string): Promise<unknown>
  setWebhookEndpointEnabled(id: string, enabled: boolean): Promise<unknown>
  telegramChannel(): Promise<DaemonTelegramChannelSummary | null>
  upsertTelegramChannel(input: {
    enabled?: boolean
    botToken: string
    allowedUsers?: string[]
    pairingRequired?: boolean
    pairingCodeTtl?: number
    rateLimitPerMinute?: number
  }): Promise<unknown>
  deleteTelegramChannel(): Promise<unknown>
  setTelegramChannelEnabled(enabled: boolean): Promise<unknown>
  telegramPairingCode(): Promise<{ code: string; expiresAt: string }>
  revokeTelegramAllowedUser(userId: string): Promise<unknown>
  backups(): Promise<DaemonBackupItem[]>
  createBackup(): Promise<DaemonBackupItem>
  deleteBackup(id: string): Promise<void>
  networkConfig(): Promise<DaemonNetworkConfig>
  updateNetworkConfig(input: DaemonNetworkConfig): Promise<DaemonNetworkConfig>
  probeNetwork(
    url: string,
  ): Promise<{ ok: boolean; status?: number; latencyMs?: number; reason?: string }>
  quickInputSettings(): Promise<DaemonQuickInputSettings>
  updateQuickInputSettings(input: DaemonQuickInputSettings): Promise<DaemonQuickInputSettings>
  publishQuickInput(text: string): Promise<{ ok: boolean }>
  settingsJson(): Promise<DaemonSettingsJsonDocument>
  updateSettingsJson(input: DaemonSettingsJsonDocument): Promise<DaemonSettingsJsonDocument>
  notifications(): Promise<DaemonNotificationItem[]>
  publishNotification(input: {
    id?: string
    title: string
    body?: string
    url?: string | null
  }): Promise<DaemonNotificationItem>
  markNotificationRead(id: string): Promise<void>
  notificationSettings(): Promise<{ channels: DaemonNotificationChannelSetting[] }>
  updateNotificationSettings(input: {
    channels: DaemonNotificationChannelSetting[]
  }): Promise<{ channels: DaemonNotificationChannelSetting[] }>
  schedulerJobs(): Promise<DaemonSchedulerJob[]>
  upsertSchedulerJob(input: DaemonSchedulerJobInput): Promise<DaemonSchedulerJob>
  pauseSchedulerJob?(id: string): Promise<DaemonSchedulerJob>
  resumeSchedulerJob?(id: string): Promise<DaemonSchedulerJob>
  deleteSchedulerJob(id: string): Promise<void>
  startGitHubOAuth(): Promise<{ url: string; warning?: string }>
  githubOAuthStatus(): Promise<{ connected: boolean; login: string | null }>
  githubSyncRepos(): Promise<DaemonGitHubSyncRepo[]>
  updateGitHubSyncRepos(repos: DaemonGitHubSyncRepo[]): Promise<DaemonGitHubSyncRepo[]>
  githubSyncPolicy(): Promise<{
    intervalMin: number
    pullRequests: boolean
    issues: boolean
    releases: boolean
  }>
  updateGitHubSyncPolicy(input: {
    intervalMin: number
    pullRequests: boolean
    issues: boolean
    releases: boolean
  }): Promise<unknown>
  githubChatSyncConfig(): Promise<DaemonGitHubChatSyncConfig>
  updateGitHubChatSyncConfig(
    input: DaemonGitHubChatSyncConfigUpdate,
  ): Promise<DaemonGitHubChatSyncConfig>
  runGitHubChatSync(): Promise<DaemonGitHubChatSyncResult>
  messageSubscriptionOverview(): Promise<{
    config: { enabled: boolean; connectionType: string; autoProcess: boolean }
    queueStatus: {
      pending: number
      processing: number
      completed: number
      failed: number
      totalProcessed: number
    }
    status: { isConnected: boolean; lastError: string | null }
    recentMessages: DaemonMessageSubscriptionItem[]
  }>
  startMessageSubscription(): Promise<unknown>
  stopMessageSubscription(): Promise<unknown>
  refreshMessageSubscription(): Promise<{
    success: boolean
    count: number
    processed?: number
    error?: string
  }>
  processPendingMessageSubscription(): Promise<{ success: boolean; processed: number }>
  messageSubscriptionMessages(
    status?: DaemonMessageSubscriptionItem['status'],
    limit?: number,
  ): Promise<DaemonMessageSubscriptionItem[]>
  reprocessMessageSubscriptionMessage(hash: string): Promise<DaemonMessageSubscriptionItem | null>
  deleteMessageSubscriptionMessage(hash: string): Promise<void>
  teamDocs(): Promise<DaemonTeamDocsConfig[]>
  upsertTeamDocsConfig(input: {
    id?: string
    name: string
    description?: string
    serverType?: 'github.com' | 'ghes'
    ghesUrl?: string
    token: string
    owner: string
    repo: string
    branch?: string
    docsPath?: string
    enabled?: boolean
    autoSync?: boolean
    syncInterval?: number
  }): Promise<DaemonTeamDocsConfig>
  deleteTeamDocsConfig(id: string): Promise<void>
  teamDocsDocuments(id: string): Promise<DaemonTeamDocsDocument[]>
  teamDocsDocument(id: string, path: string): Promise<DaemonTeamDocsDocument & { content: string }>
  testTeamDocsConnection(id: string): Promise<{ success: boolean; message: string }>
  syncTeamDocsConfig(id: string): Promise<{ success: boolean; message: string }>
  syncAllTeamDocs(): Promise<{ success: boolean; total: number; succeeded: number; failed: number }>
  personalDocs(): Promise<DaemonPersonalDoc[]>
  personalDoc(id: string): Promise<DaemonPersonalDoc & { content: string }>
  upsertPersonalDoc(input: {
    id?: string
    path: string
    content: string
  }): Promise<DaemonPersonalDoc>
  deletePersonalDoc(id: string): Promise<void>
  promptTemplates(): Promise<DaemonPromptTemplate[]>
  upsertPromptTemplate(input: {
    id?: string
    title: string
    body: string
  }): Promise<DaemonPromptTemplate>
  deletePromptTemplate(id: string): Promise<void>
  usageSnapshot(): Promise<unknown>
  observabilitySnapshot(range?: DaemonObservabilityRange): Promise<unknown>
  observabilityEvents(input: {
    limit?: number
    severity?: DaemonObservabilitySeverity
    eventType?: string
  }): Promise<unknown[]>
  observabilityCrashes(limit?: number): Promise<unknown[]>
  observabilityPrivacy(): Promise<unknown>
  pruneObservability(): Promise<{ deletedEvents: number; deletedFeedback: number }>
  dispatchSubagent(input: {
    prompt: string
    system?: string
    model?: string
    tools?: string[]
    maxIterations?: number
  }): Promise<{
    sessionId: string
    status: string
    output: string
    iterations: number
    truncated: boolean
  }>
}

export interface DaemonAdminSlashCommandSpec {
  name: string
  description: string
  args?: string
  destructiveActions?: string[]
}

export const DAEMON_ADMIN_SLASH_COMMANDS: DaemonAdminSlashCommandSpec[] = [
  {
    name: '/status',
    description: 'Show daemon health, boot info, clients, or capabilities',
    args: '[boot|clients|capabilities|shutdown confirm]',
  },
  {
    name: '/config',
    description: 'Show config, policy, or set daemon config keys',
    args: '[policy|set <key> <value>|env set <KEY> <value>]',
  },
  {
    name: '/settings-json',
    description: 'Show, replace, or shallow-patch daemon settings JSON',
    args: '[show|set <json>|patch <json>]',
  },
  {
    name: '/settings',
    description: 'Inspect or update daemon general/LLM settings',
    args: '[general|llm]',
  },
  {
    name: '/llm',
    description: 'Inspect LLM providers or test daemon LLM connectivity',
    args: '[providers|test <provider> [model]]',
  },
  {
    name: '/network',
    description: 'Inspect network settings or probe/update connectivity',
    args: '[probe <url>|set proxy|timeout|concurrency]',
  },
  {
    name: '/quick-input',
    description: 'Inspect/update quick input settings or publish text',
    args: '[publish <text>|set prefix|hotkey]',
  },
  {
    name: '/notifications',
    description: 'List, publish, read, or inspect notifications',
    args: '[list|publish|read|settings]',
  },
  {
    name: '/scheduler',
    description: 'List, add, enable, disable, or remove scheduler jobs',
    args: '[add|enable|disable|remove]',
    destructiveActions: ['remove'],
  },
  {
    name: '/github',
    description: 'Manage GitHub OAuth and sync settings',
    args: '[status|login|repos|policy|chat-sync]',
  },
  {
    name: '/messages',
    description: 'Manage message subscription queue',
    args: '[overview|list|start|stop|refresh|process]',
    destructiveActions: ['delete'],
  },
  {
    name: '/team-docs',
    description: 'Manage team documentation sources',
    args: '[list|add|docs|show|test|sync|sync-all|remove]',
    destructiveActions: ['remove'],
  },
  {
    name: '/personal-docs',
    description: 'Manage personal docs',
    args: '[list|show|set|delete]',
    destructiveActions: ['delete'],
  },
  {
    name: '/prompts',
    description: 'Manage prompt templates',
    args: '[list|show|set|delete]',
    destructiveActions: ['delete'],
  },
  {
    name: '/skills-admin',
    description: 'List, enable, or disable daemon skills',
    args: '[list|enable|disable]',
  },
  {
    name: '/rag-admin',
    description: 'Manage local RAG folders, documents, and diagnostics',
    args: '[folders|docs|show|set|delete|sync|test-vector|test-rerank]',
    destructiveActions: ['remove-folder', 'delete-doc'],
  },
  {
    name: '/snippets',
    description: 'Manage reusable snippets',
    args: '[list|set|delete]',
    destructiveActions: ['delete'],
  },
  {
    name: '/wiki',
    description: 'Manage daemon wiki nodes',
    args: '[tree|search|set|delete|move]',
    destructiveActions: ['delete'],
  },
  { name: '/queue', description: 'Show daemon message queue snapshot' },
  { name: '/usage-snapshot', description: 'Show raw daemon usage counters' },
  {
    name: '/files-admin',
    description: 'Inspect, index, search, or delete uploaded files',
    args: '[search|info|content-part|index|delete]',
    destructiveActions: ['delete'],
  },
  {
    name: '/backups',
    description: 'List, create, or delete daemon backups',
    args: '[list|create|delete]',
    destructiveActions: ['delete'],
  },
  {
    name: '/tokens',
    description: 'List, issue, or revoke daemon extension tokens',
    args: '[list|issue|revoke]',
    destructiveActions: ['revoke'],
  },
  {
    name: '/secrets',
    description: 'Set, list, or remove encrypted daemon secrets',
    args: '[list|set|remove]',
    destructiveActions: ['remove'],
  },
  {
    name: '/webhooks',
    description: 'Manage generic inbound webhook endpoints',
    args: '[list|add|enable|disable|remove]',
    destructiveActions: ['remove'],
  },
  {
    name: '/channels',
    description: 'Manage configured channels and channel security diagnostics',
    args: '[list|telegram|slack|pipeline|security ...]',
    destructiveActions: ['remove', 'unpair'],
  },
  {
    name: '/devices',
    description: 'List or manage paired daemon devices',
    args: '[list|challenge|complete|revoke]',
    destructiveActions: ['revoke'],
  },
  { name: '/personas', description: 'List daemon personas' },
  { name: '/plugins', description: 'List daemon plugin loader state' },
  {
    name: '/agents',
    description: 'List, create, or delete user agents',
    args: '[list|create|delete]',
    destructiveActions: ['delete'],
  },
  {
    name: '/commands',
    description: 'List, create, or delete custom slash commands',
    args: '[list|create|delete]',
    destructiveActions: ['delete'],
  },
  {
    name: '/cron',
    description: 'Manage legacy daemon cron tasks',
    args: '[list|add|remove]',
    destructiveActions: ['remove'],
  },
  {
    name: '/extensions',
    description: 'List, install, enable, disable, or remove extensions',
    args: '[list|install|enable|disable|remove]',
    destructiveActions: ['remove'],
  },
  {
    name: '/image-gen',
    description: 'List providers/jobs, create, or cancel image jobs',
    args: '[providers|jobs|create|cancel]',
    destructiveActions: ['cancel'],
  },
  {
    name: '/jobs',
    description: 'List, inspect, collect, or cancel background jobs',
    args: '[list|status|items|cancel]',
    destructiveActions: ['cancel'],
  },
  {
    name: '/tasks',
    description: 'Background jobs: list, inspect, collect, or cancel',
    args: '[list|status|items|cancel]',
    destructiveActions: ['cancel'],
  },
  {
    name: '/migration',
    description: 'Run, inspect, report, or cancel migrations',
    args: '[run|status|report|cancel]',
    destructiveActions: ['cancel'],
  },
  {
    name: '/observability',
    description: 'Inspect observability snapshots, events, crashes, or privacy',
    args: '[snapshot|events|crashes|privacy|prune]',
    destructiveActions: ['prune'],
  },
  { name: '/metrics', description: 'Show daemon metrics snapshot' },
  {
    name: '/marketplaces',
    description: 'Manage skill marketplaces',
    args: '[list|add|remove]',
    destructiveActions: ['remove'],
  },
  { name: '/subagent', description: 'Dispatch an isolated subagent', args: 'dispatch <prompt>' },
]

export type DestructiveGateResult =
  | { gated: false; effectiveArgs: string[] }
  | { gated: true; message: string }

export function resolveDestructiveGate(input: {
  spec: { name: string; destructiveActions?: string[] }
  args: string[]
}): DestructiveGateResult {
  const effectiveArgs = input.args.filter((token) => token !== '--yes')
  const destructive = input.spec.destructiveActions
  if (!destructive || destructive.length === 0) return { gated: false, effectiveArgs }
  if (input.args.includes('--yes')) return { gated: false, effectiveArgs }

  const action = [input.args[0]?.toLowerCase(), input.args[1]?.toLowerCase()].find(
    (value): value is string => Boolean(value && destructive.includes(value)),
  )
  if (!action) return { gated: false, effectiveArgs }

  return {
    gated: true,
    message: [
      `'${input.spec.name} ${action}' is destructive and needs confirmation.`,
      `Re-run with: ${input.spec.name} ${input.args.join(' ')} --yes`,
    ].join('\n'),
  }
}

interface ParsedArgs {
  positional: string[]
  flags: Map<string, string[]>
  booleans: Set<string>
}

interface ImageGenJob {
  id: string
  providerId?: string
  status: string
  prompt: string
  progress?: number
  outputs?: Array<{ id: string; mime: string; path?: string }>
  error?: string | null
}

export function splitSlashCommandInput(input: string): string[] {
  const tokens: string[] = []
  let current = ''
  let quote: '"' | "'" | null = null
  let escaping = false

  for (const char of input.trim()) {
    if (escaping) {
      current += char
      escaping = false
      continue
    }
    if (char === '\\') {
      escaping = true
      continue
    }
    if (quote) {
      if (char === quote) {
        quote = null
      } else {
        current += char
      }
      continue
    }
    if (char === '"' || char === "'") {
      quote = char
      continue
    }
    if (/\s/.test(char)) {
      if (current) {
        tokens.push(current)
        current = ''
      }
      continue
    }
    current += char
  }

  if (escaping) current += '\\'
  if (current) tokens.push(current)
  return tokens
}

function parseArgs(args: string[]): ParsedArgs {
  const positional: string[] = []
  const flags = new Map<string, string[]>()
  const booleans = new Set<string>()

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index]!
    if (!arg.startsWith('--')) {
      positional.push(arg)
      continue
    }

    const raw = arg.slice(2)
    const equalIndex = raw.indexOf('=')
    const key = equalIndex >= 0 ? raw.slice(0, equalIndex) : raw
    const inline = equalIndex >= 0 ? raw.slice(equalIndex + 1) : undefined
    if (!key) continue

    const next = args[index + 1]
    if (inline !== undefined) {
      addFlag(flags, key, inline)
    } else if (next && !next.startsWith('--')) {
      addFlag(flags, key, next)
      index += 1
    } else {
      booleans.add(key)
    }
  }

  return { positional, flags, booleans }
}

function addFlag(flags: Map<string, string[]>, key: string, value: string): void {
  flags.set(key, [...(flags.get(key) ?? []), value])
}

function flag(parsed: ParsedArgs, name: string): string | undefined {
  return parsed.flags.get(name)?.at(-1)
}

function flagList(parsed: ParsedArgs, name: string): string[] {
  return parsed.flags.get(name) ?? []
}

function hasFlag(parsed: ParsedArgs, name: string): boolean {
  return parsed.booleans.has(name)
}

function requireArg(value: string | undefined, usage: string): string {
  if (!value) throw new Error(usage)
  return value
}

function parseInteger(value: string | undefined, label: string): number | undefined {
  if (value == null || value === '') return undefined
  const parsed = Number.parseInt(value, 10)
  if (!Number.isFinite(parsed)) throw new Error(`Invalid ${label}: ${value}`)
  return parsed
}

function parsePositiveInteger(value: string | undefined, label: string): number | undefined {
  const parsed = parseInteger(value, label)
  if (parsed !== undefined && parsed <= 0) throw new Error(`Invalid ${label}: ${value}`)
  return parsed
}

function parseNumberInRange(
  value: string | undefined,
  label: string,
  min: number,
  max: number,
): number | undefined {
  if (value == null || value === '') return undefined
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < min || parsed > max) {
    throw new Error(`Invalid ${label}: ${value}`)
  }
  return parsed
}

function parseBool(value: string | undefined, fallback: boolean): boolean {
  if (value == null) return fallback
  if (['1', 'true', 'yes', 'on', 'enabled'].includes(value.toLowerCase())) return true
  if (['0', 'false', 'no', 'off', 'disabled'].includes(value.toLowerCase())) return false
  return fallback
}

function parseObservabilityRange(value: string | undefined): DaemonObservabilityRange | undefined {
  return value === '24h' || value === '7d' || value === '30d' ? value : undefined
}

function parseObservabilitySeverity(
  value: string | undefined,
): DaemonObservabilitySeverity | undefined {
  return value === 'debug' ||
    value === 'info' ||
    value === 'warning' ||
    value === 'error' ||
    value === 'fatal'
    ? value
    : undefined
}

function parseJsonValue(value: string): unknown {
  try {
    return JSON.parse(value) as unknown
  } catch {
    if (/^-?\d+(\.\d+)?$/.test(value)) return Number(value)
    if (['true', 'false'].includes(value)) return value === 'true'
    if (value === 'null') return null
    return value
  }
}

function parseJsonObject(value: string, label: string): Record<string, unknown> {
  const parsed = parseJsonValue(value)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object.`)
  }
  return parsed as Record<string, unknown>
}

function json(value: unknown): string {
  return truncateJsonOutput(JSON.stringify(value, null, 2))
}

function compactJson(value: unknown): string {
  return truncateJsonOutput(JSON.stringify(value))
}

function unwrapData<T>(value: T | { data: T }): T {
  if (value && typeof value === 'object' && 'data' in value) {
    return (value as { data: T }).data
  }
  return value as T
}

function formatTime(value: number | string | null | undefined): string {
  if (value == null) return 'never'
  if (typeof value === 'string') return value
  if (!Number.isFinite(value)) return 'unknown'
  return new Date(value).toISOString()
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`
  return `${(value / (1024 * 1024)).toFixed(1)} MiB`
}

function truncate(value: string, max = 160): string {
  const normalized = value.replace(/\s+/g, ' ').trim()
  return normalized.length > max ? `${normalized.slice(0, max - 1)}...` : normalized
}

function formatContentPartSummary(value: unknown): string {
  if (!value || typeof value !== 'object') {
    return 'File content part: non-object payload\nContent parts may contain large or binary payloads; use the HTTP API directly to consume the full value.'
  }
  const part = value as Record<string, unknown>
  const fields = [
    typeof part.type === 'string' ? `type=${part.type}` : null,
    typeof part.mediaType === 'string' ? `mediaType=${part.mediaType}` : null,
    typeof part.mimeType === 'string' ? `mimeType=${part.mimeType}` : null,
    typeof part.text === 'string' ? `textLength=${part.text.length}` : null,
    typeof part.data === 'string' ? `dataLength=${part.data.length}` : null,
  ].filter(Boolean)
  return [
    `File content part: ${fields.join(' ') || 'payload present'}`,
    'Content parts may contain large or binary payloads; use the HTTP API directly to consume the full value.',
  ].join('\n')
}

function nullishId(value: string | undefined): string | null | undefined {
  if (value == null) return undefined
  return ['none', 'null', 'root'].includes(value.toLowerCase()) ? null : value
}

function usage(command: string): string {
  switch (command) {
    case '/status':
      return 'Usage: /status [boot|clients|capabilities|shutdown confirm]'
    case '/config':
      return 'Usage: /config [policy|set <key> <value>|env set <KEY> <value>]'
    case '/settings-json':
      return 'Usage: /settings-json [show|set <json>|patch <json>]'
    case '/settings':
      return 'Usage: /settings [general|general set --theme light|dark|system --locale ko|en --working-dir <path|none>|llm|llm set --provider <id> [--model <id>]]'
    case '/llm':
      return 'Usage: /llm [providers|test <providerId> [modelId]]'
    case '/network':
      return 'Usage: /network [current|probe <url>|set proxy <url|off>|set timeout <ms>|set concurrency <n>]'
    case '/quick-input':
      return 'Usage: /quick-input [current|publish <text>|set prefix <text>|set hotkey <keys>]'
    case '/notifications':
      return 'Usage: /notifications [list|publish <title> [body...] [--url <url>]|read <id>|settings|settings enable|disable <id>]'
    case '/scheduler':
      return 'Usage: /scheduler [list|add <name> --cron <expr> --command <cmd> [--id <id>] [--disabled]|enable <id>|disable <id>|remove <id>]'
    case '/github':
      return 'Usage: /github [status|login|repos|add <owner/repo>|enable <owner/repo>|disable <owner/repo>|policy|policy set --interval <min> [--prs true|false] [--issues true|false] [--releases true|false]|chat-sync status|chat-sync set <owner/repo> [--branch main]|chat-sync enable|chat-sync disable|chat-sync run]'
    case '/messages':
      return 'Usage: /messages [overview|list [status] [--limit n]|start|stop|refresh|process|reprocess <hash>|delete <hash>]'
    case '/team-docs':
      return 'Usage: /team-docs [list|add <name> --token <token> --owner <owner> --repo <repo> [--branch main] [--path docs]|docs <id>|show <id> <path>|test <id>|sync <id>|sync-all|remove <id>]'
    case '/personal-docs':
      return 'Usage: /personal-docs [list|show <id>|set <path> <content...> [--id id]|delete <id>]'
    case '/prompts':
      return 'Usage: /prompts [list|show <id>|set <title> <body...> [--id id]|delete <id>]'
    case '/skills-admin':
      return 'Usage: /skills-admin [list|enable <name>|disable <name>]'
    case '/rag-admin':
      return 'Usage: /rag-admin [folders|add-folder <name> [--id id] [--path path] [--git]|remove-folder <id>|docs <folderId>|show <docId>|set-doc <folderId> <title> <body...> [--id id]|delete-doc <id>|search <query> [--limit n]|sync|vector-db|test-vector|test-rerank]'
    case '/snippets':
      return 'Usage: /snippets [list [query]|set <title> --language <lang> --body <text> [--id id] [--tag tag]|delete <id>]'
    case '/wiki':
      return 'Usage: /wiki [tree|search <query>|set <title> <body...> [--id id] [--parent id|null]|delete <id>|move <id> --parent <id|null> --order <n>]'
    case '/queue':
      return 'Usage: /queue'
    case '/usage-snapshot':
      return 'Usage: /usage-snapshot'
    case '/files-admin':
      return 'Usage: /files-admin [search <cwd> [query] [--limit n]|info <id>|content-part <id>|index <id> [--title title] [--path path] [--tag tag] [--ocr]|delete <id>]'
    case '/backups':
      return 'Usage: /backups [list|create|delete <id>]'
    case '/tokens':
      return 'Usage: /tokens [list|issue <label> --scope <scope>[,<scope>] [--expires-at ts]|revoke <id>]'
    case '/secrets':
      return 'Usage: /secrets [list|set <key> <value>|remove <key>]'
    case '/webhooks':
      return 'Usage: /webhooks [list|add <path> --header <name> --secret <value> [--event e] [--ip cidr] [--disabled]|enable <id>|disable <id>|remove <id>]'
    case '/channels':
      return 'Usage: /channels [list|telegram add --bot-token <token>|telegram pair|telegram users|telegram enable|telegram disable|telegram remove|telegram unpair <userId>|slack show|add|enable|disable|remove|pipeline [summary|health|health set <json>|history]|security [summary|policy|policy set <json>|policy history|health|health set <json>|health history]]'
    case '/devices':
      return 'Usage: /devices [list|challenge [--ttl seconds]|complete --challenge-id id --device-id id --name name --role desktop|server|edge --public-key pem --signature sig|revoke <id>]'
    case '/agents':
      return 'Usage: /agents [list|create <id> --description <text> --prompt <text> [--base id] [--model id]|delete <id>]'
    case '/commands':
      return 'Usage: /commands [list|create <id> --description <text> --body <prompt> [--args none|optional|required]|delete <id>]'
    case '/extensions':
      return 'Usage: /extensions [list|install <manifest-json>|enable <id>|disable <id>|remove <id>]'
    case '/plugins':
      return 'Usage: /plugins'
    case '/cron':
      return 'Usage: /cron [list|add <name> --schedule <cron> --instruction <text> [--disabled]|remove <id>]'
    case '/image-gen':
      return 'Usage: /image-gen [providers|jobs [--limit n]|create <providerId> <prompt...> [--params json] [--wait [ms]] [--output dir] [--download dir]|cancel <id>]'
    case '/tasks':
    case '/jobs':
      return 'Usage: /jobs [list [--status state] [--kind kind] [--limit n] [--offset n]|status <id>|items <id> [--since n]|cancel <id>|submit <items-json-array>]'
    case '/migration':
      return 'Usage: /migration [run <sourcePath> [--step name] [--dry-run]|status <id>|report <id>|cancel <id>]'
    case '/observability':
      return 'Usage: /observability [snapshot [range]|events [--limit n]|crashes [--limit n]|privacy|prune]'
    case '/marketplaces':
      return 'Usage: /marketplaces [list|add <name> <url>|remove <name>]'
    case '/subagent':
      return 'Usage: /subagent dispatch <prompt> [--system <text>] [--model <model>] [--tools a,b] [--max-iterations n]'
    default:
      return `Usage: ${command}`
  }
}

export async function runDaemonAdminSlashCommand(
  client: DaemonAdminSlashClient,
  command: string,
  args: string[],
): Promise<string> {
  const spec = DAEMON_ADMIN_SLASH_COMMANDS.find((entry) => entry.name === command)
  if (spec) {
    const gate = resolveDestructiveGate({ spec, args })
    if (gate.gated) return gate.message
    args = gate.effectiveArgs
  }

  switch (command) {
    case '/status':
      return runStatusCommand(client, args)
    case '/config':
      return runConfigCommand(client, args)
    case '/settings-json':
      return runSettingsJsonCommand(client, args)
    case '/settings':
      return runSettingsCommand(client, args)
    case '/llm':
      return runLlmCommand(client, args)
    case '/network':
      return runNetworkCommand(client, args)
    case '/quick-input':
      return runQuickInputCommand(client, args)
    case '/notifications':
      return runNotificationsCommand(client, args)
    case '/scheduler':
      return runSchedulerCommand(client, args)
    case '/github':
      return runGithubCommand(client, args)
    case '/messages':
      return runMessagesCommand(client, args)
    case '/team-docs':
      return runTeamDocsCommand(client, args)
    case '/personal-docs':
      return runPersonalDocsCommand(client, args)
    case '/prompts':
      return runPromptsCommand(client, args)
    case '/skills-admin':
      return runSkillsAdminCommand(client, args)
    case '/rag-admin':
      return runRagAdminCommand(client, args)
    case '/snippets':
      return runSnippetsCommand(client, args)
    case '/wiki':
      return runWikiCommand(client, args)
    case '/queue':
      return runQueueCommand(client)
    case '/usage-snapshot':
      return runUsageSnapshotCommand(client)
    case '/files-admin':
      return runFilesAdminCommand(client, args)
    case '/backups':
      return runBackupsCommand(client, args)
    case '/tokens':
      return runTokensCommand(client, args)
    case '/secrets':
      return runSecretsCommand(client, args)
    case '/webhooks':
      return runWebhooksCommand(client, args)
    case '/channels':
      return runChannelsCommand(client, args)
    case '/devices':
      return runDevicesCommand(client, args)
    case '/personas':
      return runPersonasCommand(client)
    case '/plugins':
      return runPluginsCommand(client)
    case '/agents':
      return runAgentsCommand(client, args)
    case '/commands':
      return runCommandsCommand(client, args)
    case '/cron':
      return runCronCommand(client, args)
    case '/extensions':
      return runExtensionsCommand(client, args)
    case '/image-gen':
      return runImageGenCommand(client, args)
    case '/tasks':
      return json(await runTasksCommand(client as unknown as TasksClient, args))
    case '/jobs':
      return runJobsCommand(client, args)
    case '/migration':
      return runMigrationCommand(client, args)
    case '/observability':
      return runObservabilityCommand(client, args)
    case '/metrics':
      return runMetricsCommand(client)
    case '/marketplaces':
      return runMarketplacesCommand(client, args)
    case '/subagent':
      return runSubagentCommand(client, args)
    default:
      throw new Error(`Unknown daemon admin command: ${command}`)
  }
}

async function runStatusCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'health'
  if (action === 'health' || action === 'current') {
    const health = await client.health()
    return [
      `Daemon: ${health.status} (${health.version})`,
      health.uptime != null ? `Uptime: ${Math.round(health.uptime)}s` : null,
      health.memory
        ? `Memory: rss ${formatBytes(health.memory.rss)} / heap ${formatBytes(health.memory.heap)}`
        : null,
      health.components
        ? Object.entries(health.components)
            .map(
              ([name, component]) =>
                `  ${name}: ${component.status}${component.details ? ` - ${component.details}` : ''}`,
            )
            .join('\n')
        : null,
    ]
      .filter(Boolean)
      .join('\n')
  }
  if (action === 'clients') {
    const clients = await client.systemClients()
    return [
      `Clients: ${clients.count} (ws ${clients.byKind.ws}, sse ${clients.byKind.sse})`,
      `Idle: ${clients.idleMs}ms`,
      ...clients.clients
        .slice(0, 12)
        .map(
          (entry) =>
            `  ${entry.kind} ${entry.label || entry.id} opened ${formatTime(entry.openedAt)}`,
        ),
    ].join('\n')
  }
  if (action === 'capabilities') {
    const result = await client.request<{
      capabilities: Array<{
        name: string
        version: string
        methods: Array<{ method: string; path: string }>
      }>
    }>('/capabilities')
    return result.capabilities
      .map(
        (capability) =>
          `${capability.name}@${capability.version}\n${capability.methods.map((m) => `  ${m.method} ${m.path}`).join('\n')}`,
      )
      .join('\n')
  }
  if (action === 'boot' || action === 'boot-info' || action === 'system') {
    const info = await client.request<{
      daemon: {
        version: string
        nodeVersion: string
        pid: number
        platform: string
        uptimeSeconds: number
        startedAt: string
      }
      paths: { home: string; entries: Array<{ label: string; path: string }> }
      env: Array<{
        name: string
        effective: string | null
        fromEnv: boolean
        default: string | null
        purpose: string
      }>
    }>('/system/boot-info')
    return [
      `Boot: ${info.daemon.version} pid=${info.daemon.pid} node=${info.daemon.nodeVersion} platform=${info.daemon.platform}`,
      `Started: ${info.daemon.startedAt} uptime=${info.daemon.uptimeSeconds}s`,
      `Home: ${info.paths.home}`,
      'Paths:',
      ...info.paths.entries.map((entry) => `  ${entry.label}: ${entry.path}`),
      'Env:',
      ...info.env.map(
        (entry) =>
          `  ${entry.name}=${entry.effective ?? '(unset)'}${entry.fromEnv ? ' [env]' : entry.default != null ? ' [default]' : ''}`,
      ),
    ].join('\n')
  }
  if (action === 'shutdown' && args[1] === 'confirm') {
    const result = await client.systemShutdown()
    return `Daemon shutdown requested: ${result.status}`
  }
  return usage('/status')
}

async function runConfigCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'show'
  if (action === 'show' || action === 'current') {
    const config = await client.config()
    return [
      `Device: ${config.device.name} (${config.device.role})`,
      `Daemon: ${config.daemon.host}:${config.daemon.port}`,
      `Agent: provider=${config.agent.defaultProvider ?? '?'} model=${config.agent.defaultModel ?? '?'} autonomy=${config.agent.autonomy} thinking=${config.agent.thinkingLevel}`,
      `Providers: ${config.providers.map((provider) => provider.id).join(', ') || 'none'}`,
      `Channels: ${config.channels.map((channel) => `${channel.type}${channel.enabled === false ? ':disabled' : ''}`).join(', ') || 'none'}`,
    ].join('\n')
  }
  if (action === 'policy') {
    return json(await client.policy())
  }
  if (action === 'set') {
    const key = requireArg(args[1], usage('/config'))
    const value = requireArg(args.slice(2).join(' '), usage('/config'))
    await client.updateConfig({ [key]: parseJsonValue(value) } as DaemonConfigUpdateInput)
    return `Config updated: ${key}`
  }
  if (action === 'env' && args[1] === 'set') {
    const key = requireArg(args[2], usage('/config'))
    const value = requireArg(args.slice(3).join(' '), usage('/config'))
    await client.updateConfigEnv({ updates: { [key]: value } })
    return `Daemon env updated: ${key}`
  }
  return usage('/config')
}

async function runSettingsJsonCommand(
  client: DaemonAdminSlashClient,
  args: string[],
): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'show'
  if (action === 'show' || action === 'current') return json(await client.settingsJson())
  if (action === 'set') {
    const next = parseJsonObject(
      requireArg(args.slice(1).join(' '), usage('/settings-json')),
      'settings-json',
    )
    await client.updateSettingsJson(next)
    return 'Settings JSON replaced.'
  }
  if (action === 'patch') {
    const patch = parseJsonObject(
      requireArg(args.slice(1).join(' '), usage('/settings-json')),
      'settings-json patch',
    )
    const current = await client.settingsJson()
    await client.updateSettingsJson({ ...current, ...patch })
    return `Settings JSON patched: ${Object.keys(patch).join(', ')}`
  }
  return usage('/settings-json')
}

async function runSettingsCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const target = args[0]?.toLowerCase() ?? 'general'
  if (target === 'general') {
    if (!args[1]) {
      const settings = await client.request<{
        theme: string
        locale: string
        workingDir: string | null
      }>('/settings/general')
      return `General settings: theme=${settings.theme} locale=${settings.locale} workingDir=${settings.workingDir ?? 'default'}`
    }
    if (args[1] !== 'set') return usage('/settings')

    const parsed = parseArgs(args.slice(2))
    const theme = flag(parsed, 'theme')
    const locale = flag(parsed, 'locale')
    const workingDir = flag(parsed, 'working-dir')
    const next: { theme?: string; locale?: string; workingDir?: string | null } = {}

    if (theme) {
      if (!['light', 'dark', 'system'].includes(theme))
        throw new Error('Theme must be light, dark, or system.')
      next.theme = theme
    }
    if (locale) {
      if (!['ko', 'en'].includes(locale)) throw new Error('Locale must be ko or en.')
      next.locale = locale
    }
    if (workingDir !== undefined) next.workingDir = nullishId(workingDir) ?? null
    if (Object.keys(next).length === 0) return usage('/settings')

    await client.request('/settings/general', { method: 'PUT', body: next })
    return 'General settings updated.'
  }

  if (target === 'llm') {
    if (!args[1]) {
      const settings = await client.request<{ providerId: string | null; modelId: string | null }>(
        '/settings/llm',
      )
      return `LLM settings: provider=${settings.providerId ?? 'none'} model=${settings.modelId ?? 'none'}`
    }
    if (args[1] !== 'set') return usage('/settings')

    const parsed = parseArgs(args.slice(2))
    const providerId = requireArg(
      flag(parsed, 'provider') ?? parsed.positional[0],
      usage('/settings'),
    )
    const modelId = flag(parsed, 'model') ?? null
    await client.request('/settings/llm', {
      method: 'PUT',
      body: { providerId, modelId },
    })
    return `LLM settings updated: provider=${providerId}${modelId ? ` model=${modelId}` : ''}`
  }

  return usage('/settings')
}

async function runLlmCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'providers'
  if (action === 'providers' || action === 'list') {
    const providers =
      await client.request<
        Array<{ id: string; label: string; models: Array<{ id: string; label: string }> }>
      >('/llm/providers')
    return providers.length
      ? providers
          .map(
            (provider) =>
              `${provider.id} ${provider.label}\n  models=${provider.models.map((model) => model.id).join(', ') || 'none'}`,
          )
          .join('\n')
      : 'No LLM providers registered.'
  }
  if (action === 'test') {
    const providerId = requireArg(args[1], usage('/llm'))
    const modelId = args[2]
    const result = await client.request<{ ok: boolean; latencyMs?: number; reason?: string }>(
      '/llm/test-connection',
      {
        method: 'POST',
        body: { providerId, modelId },
      },
    )
    return result.ok
      ? `LLM connection ok: ${providerId}${modelId ? `/${modelId}` : ''} latency=${result.latencyMs ?? '?'}ms`
      : `LLM connection failed: ${providerId}${modelId ? `/${modelId}` : ''} ${result.reason ?? 'unknown'}`
  }
  return usage('/llm')
}

async function runSnippetsCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list' || action === 'search') {
    const parsed = parseArgs(args.slice(1))
    const query = flag(parsed, 'query') ?? parsed.positional.join(' ')
    const path = query ? `/snippets?q=${encodeURIComponent(query)}` : '/snippets'
    const snippets =
      await client.request<
        Array<{ id: string; title: string; language: string; body: string; tags: string[] }>
      >(path)
    return snippets.length ? snippets.map(formatSnippet).join('\n') : 'No snippets.'
  }

  if (action === 'set' || action === 'add') {
    const parsed = parseArgs(args.slice(1))
    const title = requireArg(parsed.positional[0], usage('/snippets'))
    const body = requireArg(
      flag(parsed, 'body') ?? parsed.positional.slice(1).join(' '),
      usage('/snippets'),
    )
    const language = requireArg(
      flag(parsed, 'language') ?? flag(parsed, 'lang'),
      usage('/snippets'),
    )
    const tags = [
      ...flagList(parsed, 'tag'),
      ...flagList(parsed, 'tags').flatMap((value) => value.split(',')),
    ]
      .map((value) => value.trim())
      .filter(Boolean)
    const snippet = await client.request<{
      id: string
      title: string
      language: string
      body: string
      tags: string[]
    }>('/snippets', {
      method: 'POST',
      body: {
        id: flag(parsed, 'id'),
        title,
        language,
        body,
        tags,
      },
    })
    return `Snippet saved: ${formatSnippet(snippet)}`
  }

  if (action === 'delete' || action === 'remove') {
    const id = requireArg(args[1], usage('/snippets'))
    await client.request(`/snippets/${encodeURIComponent(id)}`, { method: 'DELETE' })
    return `Snippet deleted: ${id}`
  }

  return usage('/snippets')
}

function formatSnippet(snippet: {
  id: string
  title: string
  language: string
  body: string
  tags: string[]
}): string {
  const tagSummary = snippet.tags.length ? ` tags=${snippet.tags.join(',')}` : ''
  return `${snippet.id} ${snippet.title} [${snippet.language}]${tagSummary}\n  ${truncate(snippet.body)}`
}

async function runWikiCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'tree'
  if (action === 'tree' || action === 'list') {
    const nodes = await client.request<DaemonWikiNode[]>('/wiki/tree')
    return nodes.length ? nodes.map(formatWikiNode).join('\n') : 'No wiki nodes.'
  }

  if (action === 'search') {
    const parsed = parseArgs(args.slice(1))
    const query = requireArg(flag(parsed, 'query') ?? parsed.positional.join(' '), usage('/wiki'))
    const limit = parseInteger(flag(parsed, 'limit'), 'limit') ?? 50
    const nodes = await client.request<Array<DaemonWikiNode & { snippet?: string }>>(
      `/wiki/search?q=${encodeURIComponent(query)}&limit=${encodeURIComponent(String(limit))}`,
    )
    return nodes.length ? nodes.map(formatWikiNode).join('\n') : 'No wiki search results.'
  }

  if (action === 'set' || action === 'add') {
    const parsed = parseArgs(args.slice(1))
    const title = requireArg(parsed.positional[0], usage('/wiki'))
    const body = flag(parsed, 'body') ?? parsed.positional.slice(1).join(' ')
    const node = await client.request<DaemonWikiNode>('/wiki/nodes', {
      method: 'POST',
      body: {
        id: flag(parsed, 'id'),
        parentId: nullishId(flag(parsed, 'parent')),
        title,
        icon: nullishId(flag(parsed, 'icon')),
        group: nullishId(flag(parsed, 'group')),
        body,
      },
    })
    return `Wiki node saved: ${formatWikiNode(node)}`
  }

  if (action === 'delete' || action === 'remove') {
    const id = requireArg(args[1], usage('/wiki'))
    await client.request(`/wiki/nodes/${encodeURIComponent(id)}`, { method: 'DELETE' })
    return `Wiki node deleted: ${id}`
  }

  if (action === 'move') {
    const id = requireArg(args[1], usage('/wiki'))
    const parsed = parseArgs(args.slice(2))
    const order = parseInteger(requireArg(flag(parsed, 'order'), usage('/wiki')), 'order')
    if (order == null || order < 0)
      throw new Error('Wiki node order must be a non-negative integer.')
    const parentId = nullishId(requireArg(flag(parsed, 'parent'), usage('/wiki')))
    const node = await client.request<DaemonWikiNode>(
      `/wiki/nodes/${encodeURIComponent(id)}/move`,
      {
        method: 'POST',
        body: { parentId: parentId ?? null, order },
      },
    )
    return `Wiki node moved: ${formatWikiNode(node)}`
  }

  return usage('/wiki')
}

interface DaemonWikiNode {
  id: string
  parentId: string | null
  title: string
  icon: string | null
  group: string | null
  order: number
  body: string
  updatedAt: number
  snippet?: string
}

function formatWikiNode(node: DaemonWikiNode): string {
  const meta = [
    node.parentId ? `parent=${node.parentId}` : 'root',
    `order=${node.order}`,
    node.group ? `group=${node.group}` : null,
    node.icon ? `icon=${node.icon}` : null,
    `updated=${formatTime(node.updatedAt)}`,
  ]
    .filter(Boolean)
    .join(' ')
  return `${node.id} ${node.title} (${meta})\n  ${truncate(node.snippet ?? node.body)}`
}

async function runQueueCommand(client: DaemonAdminSlashClient): Promise<string> {
  const snapshot =
    await client.request<Array<{ topic: string; pending: number; inFlight: number }>>(
      '/message-queue/snapshot',
    )
  return snapshot.length
    ? snapshot
        .map((entry) => `${entry.topic} pending=${entry.pending} inFlight=${entry.inFlight}`)
        .join('\n')
    : 'Message queue empty.'
}

async function runUsageSnapshotCommand(client: DaemonAdminSlashClient): Promise<string> {
  const snapshot = await client.request<{
    totalSessions: number
    totalMessages: number
    totalToolCalls: number
  }>('/usage/snapshot')
  return `Usage snapshot: sessions=${snapshot.totalSessions} messages=${snapshot.totalMessages} toolCalls=${snapshot.totalToolCalls}`
}

async function runFilesAdminCommand(
  client: DaemonAdminSlashClient,
  args: string[],
): Promise<string> {
  const action = args[0]?.toLowerCase()
  if (action === 'search') {
    const parsed = parseArgs(args.slice(1))
    const cwd = requireArg(parsed.positional[0], usage('/files-admin'))
    const query = flag(parsed, 'query') ?? parsed.positional.slice(1).join(' ')
    const params = new URLSearchParams({ cwd })
    if (query) params.set('q', query)
    const limit = parseInteger(flag(parsed, 'limit'), 'limit')
    if (limit) params.set('limit', String(limit))
    const result = unwrapData(
      await client.request<
        | { data: { matches: Array<{ path: string; basename: string; isDirectory: boolean }> } }
        | { matches: Array<{ path: string; basename: string; isDirectory: boolean }> }
      >(`/api/v1/files/search?${params.toString()}`),
    )
    return result.matches.length
      ? result.matches
          .map((entry) => `${entry.isDirectory ? 'dir ' : 'file'} ${entry.path}`)
          .join('\n')
      : 'No file search matches.'
  }
  if (action === 'info') {
    const id = requireArg(args[1], usage('/files-admin'))
    const file = unwrapData(
      await client.request<
        | {
            data: {
              id: string
              filename: string
              mimeType: string
              size: number
              uploadedAt: string
            }
          }
        | { id: string; filename: string; mimeType: string; size: number; uploadedAt: string }
      >(`/api/v1/files/${encodeURIComponent(id)}`),
    )
    return `${file.id} ${file.filename} ${file.mimeType} ${formatBytes(file.size)} uploaded=${file.uploadedAt}`
  }
  if (action === 'content-part') {
    const id = requireArg(args[1], usage('/files-admin'))
    return formatContentPartSummary(
      unwrapData(await client.request(`/api/v1/files/${encodeURIComponent(id)}/content-part`)),
    )
  }
  if (action === 'index') {
    const id = requireArg(args[1], usage('/files-admin'))
    const parsed = parseArgs(args.slice(2))
    const result = unwrapData(
      await client.request(`/api/v1/files/${encodeURIComponent(id)}/index`, {
        method: 'POST',
        body: {
          title: flag(parsed, 'title'),
          path: flag(parsed, 'path'),
          tags: flagList(parsed, 'tag'),
          ocr: hasFlag(parsed, 'ocr') ? true : undefined,
          ocrLanguages: flagList(parsed, 'ocr-language'),
          ocrMaxPages: parseInteger(flag(parsed, 'ocr-max-pages'), 'ocr-max-pages'),
        },
      }),
    )
    return `File indexed: ${compactJson(result)}`
  }
  if (action === 'delete' || action === 'remove') {
    const id = requireArg(args[1], usage('/files-admin'))
    await client.request(`/api/v1/files/${encodeURIComponent(id)}`, { method: 'DELETE' })
    return `File deleted: ${id}`
  }
  return usage('/files-admin')
}

async function runNetworkCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'current'
  if (action === 'current' || action === 'show') {
    const config = await client.networkConfig()
    return `Network: proxy=${config.proxyUrl ?? 'off'} timeout=${config.timeoutMs}ms concurrency=${config.maxConcurrency}`
  }
  if (action === 'probe') {
    const url = requireArg(args[1], usage('/network'))
    const result = await client.probeNetwork(url)
    return result.ok
      ? `Network probe ok: ${url} status=${result.status ?? 'n/a'} latency=${result.latencyMs ?? '?'}ms`
      : `Network probe failed: ${url} ${result.reason ?? 'unknown'}`
  }
  if (action === 'set') {
    const field = args[1]?.toLowerCase()
    const value = requireArg(args[2], usage('/network'))
    const current = await client.networkConfig()
    if (field === 'proxy') {
      current.proxyUrl = ['off', 'none', 'null'].includes(value.toLowerCase()) ? null : value
    } else if (field === 'timeout') {
      current.timeoutMs = parseInteger(value, 'timeout') ?? current.timeoutMs
    } else if (field === 'concurrency') {
      current.maxConcurrency = parseInteger(value, 'concurrency') ?? current.maxConcurrency
    } else {
      return usage('/network')
    }
    const updated = await client.updateNetworkConfig(current)
    return `Network updated: proxy=${updated.proxyUrl ?? 'off'} timeout=${updated.timeoutMs}ms concurrency=${updated.maxConcurrency}`
  }
  return usage('/network')
}

async function runQuickInputCommand(
  client: DaemonAdminSlashClient,
  args: string[],
): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'current'
  if (action === 'current' || action === 'show' || action === 'settings') {
    const settings = await client.quickInputSettings()
    return [
      `Quick input: hotkey=${settings.hotkey} prefix=${settings.prefix || '(empty)'}`,
      ...(settings.quickQuestions ?? []).map(
        (q) => `  ${q.enabled ? 'on ' : 'off'} ${q.shortcut} ${q.name}: ${truncate(q.prompt, 80)}`,
      ),
    ].join('\n')
  }
  if (action === 'publish') {
    const text = requireArg(args.slice(1).join(' '), usage('/quick-input'))
    await client.publishQuickInput(text)
    return 'Quick input published.'
  }
  if (action === 'set') {
    const field = args[1]?.toLowerCase()
    const value = requireArg(args.slice(2).join(' '), usage('/quick-input'))
    const current = await client.quickInputSettings()
    if (field === 'prefix') current.prefix = value
    else if (field === 'hotkey') current.hotkey = value
    else return usage('/quick-input')
    const updated = await client.updateQuickInputSettings(current)
    return `Quick input updated: hotkey=${updated.hotkey} prefix=${updated.prefix || '(empty)'}`
  }
  return usage('/quick-input')
}

async function runNotificationsCommand(
  client: DaemonAdminSlashClient,
  args: string[],
): Promise<string> {
  const parsed = parseArgs(args.slice(1))
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list' || action === 'current') {
    const items = await client.notifications()
    return items.length ? items.map(formatNotification).join('\n') : 'No notifications.'
  }
  if (action === 'publish') {
    const title = requireArg(parsed.positional[0], usage('/notifications'))
    const body = parsed.positional.slice(1).join(' ')
    const item = await client.publishNotification({
      title,
      body,
      url: flag(parsed, 'url') ?? null,
    })
    return `Notification published: ${item.id} ${item.title}`
  }
  if (action === 'read') {
    const id = requireArg(args[1], usage('/notifications'))
    await client.markNotificationRead(id)
    return `Notification marked read: ${id}`
  }
  if (action === 'settings') {
    const sub = args[1]?.toLowerCase()
    if (!sub) return json(await client.notificationSettings())
    const id = requireArg(args[2], usage('/notifications'))
    const settings = await client.notificationSettings()
    const enabled = sub === 'enable'
    if (sub !== 'enable' && sub !== 'disable') return usage('/notifications')
    const next = {
      channels: [...settings.channels.filter((entry) => entry.id !== id), { id, enabled }],
    }
    await client.updateNotificationSettings(next)
    return `Notification channel ${enabled ? 'enabled' : 'disabled'}: ${id}`
  }
  return usage('/notifications')
}

function formatNotification(item: DaemonNotificationItem): string {
  return `${item.readAt ? 'read' : 'new '} ${item.id} ${formatTime(item.createdAt)}\n  ${item.title}${item.body ? ` - ${truncate(item.body)}` : ''}${item.url ? `\n  ${item.url}` : ''}`
}

async function runSchedulerCommand(
  client: DaemonAdminSlashClient,
  args: string[],
): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list' || action === 'current') {
    const jobs = await client.schedulerJobs()
    return jobs.length ? jobs.map(formatSchedulerJob).join('\n') : 'No scheduler jobs.'
  }
  if (action === 'add') {
    const parsed = parseArgs(args.slice(1))
    const name = requireArg(parsed.positional[0], usage('/scheduler'))
    const cron = requireArg(flag(parsed, 'cron'), usage('/scheduler'))
    const instruction = requireArg(
      flag(parsed, 'instruction') ?? flag(parsed, 'command'),
      usage('/scheduler'),
    )
    const job = await client.upsertSchedulerJob({
      id: flag(parsed, 'id'),
      name,
      cron,
      instruction,
      enabled: !hasFlag(parsed, 'disabled'),
    })
    return `Scheduler job saved: ${formatSchedulerJob(job)}`
  }
  if (action === 'remove' || action === 'delete') {
    const id = requireArg(args[1], usage('/scheduler'))
    await client.deleteSchedulerJob(id)
    return `Scheduler job removed: ${id}`
  }
  if (action === 'enable' || action === 'disable') {
    const id = requireArg(args[1], usage('/scheduler'))
    const jobs = await client.schedulerJobs()
    const job = jobs.find((entry) => entry.id === id)
    if (!job) throw new Error(`Scheduler job not found: ${id}`)
    const updated =
      action === 'enable'
        ? client.resumeSchedulerJob
          ? await client.resumeSchedulerJob(id)
          : await client.upsertSchedulerJob({
              id,
              name: job.name,
              cron: job.cron ?? '',
              instruction: job.instruction,
              enabled: true,
            })
        : client.pauseSchedulerJob
          ? await client.pauseSchedulerJob(id)
          : await client.upsertSchedulerJob({
              id,
              name: job.name,
              cron: job.cron ?? '',
              instruction: job.instruction,
              enabled: false,
            })
    return `Scheduler job ${updated.enabled ? 'enabled' : 'disabled'}: ${updated.id}`
  }
  return usage('/scheduler')
}

function formatSchedulerJob(job: DaemonSchedulerJob): string {
  const schedule =
    job.kind === 'recurring' ? `cron=${job.cron ?? ''}` : `runAt=${formatTime(job.runAt)}`
  const next = hasPendingScheduledRun(job) ? formatTime(job.nextRunAt) : '—'
  return `${job.enabled ? 'enabled ' : 'disabled'} ${job.id} ${job.name}\n  ${schedule} next=${next} last=${formatTime(job.lastRunAt)}\n  ${job.instruction}`
}

async function runGithubCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'status'
  if (action === 'chat-sync' || action === 'chat' || action === 'sessions') {
    return runGithubChatSyncCommand(client, args.slice(1))
  }
  if (action === 'status') {
    const status = await client.githubOAuthStatus()
    const chatSync = await client.githubChatSyncConfig().catch(() => null)
    return [
      `GitHub: ${status.connected ? `connected as ${status.login ?? 'unknown'}` : 'not connected'}`,
      chatSync ? `Encrypted chat sync: ${formatGithubChatSyncConfig(chatSync)}` : null,
    ]
      .filter(Boolean)
      .join('\n')
  }
  if (action === 'login' || action === 'oauth' || action === 'start') {
    const result = await client.startGitHubOAuth()
    return [`Open GitHub OAuth URL: ${result.url}`, result.warning].filter(Boolean).join('\n')
  }
  if (action === 'repos' || action === 'list') {
    const repos = await client.githubSyncRepos()
    return repos.length
      ? repos.map(formatGithubRepo).join('\n')
      : 'No GitHub sync repos configured.'
  }
  if (action === 'add' || action === 'enable' || action === 'disable') {
    const fullName = requireArg(args[1], usage('/github'))
    const repos = await client.githubSyncRepos()
    const enabled = action !== 'disable'
    const next = [
      ...repos.filter((repo) => repo.fullName !== fullName),
      {
        fullName,
        enabled,
        lastSyncedAt: repos.find((repo) => repo.fullName === fullName)?.lastSyncedAt ?? null,
      },
    ]
    await client.updateGitHubSyncRepos(next)
    return `GitHub sync repo ${enabled ? 'enabled' : 'disabled'}: ${fullName}`
  }
  if (action === 'policy') {
    if (args[1] !== 'set') return json(await client.githubSyncPolicy())
    const parsed = parseArgs(args.slice(2))
    const current = await client.githubSyncPolicy()
    const next = {
      intervalMin: parseInteger(flag(parsed, 'interval'), 'interval') ?? current.intervalMin,
      pullRequests: parseBool(flag(parsed, 'prs'), current.pullRequests),
      issues: parseBool(flag(parsed, 'issues'), current.issues),
      releases: parseBool(flag(parsed, 'releases'), current.releases),
    }
    await client.updateGitHubSyncPolicy(next)
    return `GitHub sync policy updated: interval=${next.intervalMin}m prs=${next.pullRequests} issues=${next.issues} releases=${next.releases}`
  }
  return usage('/github')
}

function formatGithubRepo(repo: DaemonGitHubSyncRepo): string {
  return `${repo.enabled ? 'enabled ' : 'disabled'} ${repo.fullName} last=${formatTime(repo.lastSyncedAt)}`
}

async function runGithubChatSyncCommand(
  client: DaemonAdminSlashClient,
  args: string[],
): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'status'
  if (action === 'status' || action === 'current') {
    return `Encrypted chat sync: ${formatGithubChatSyncConfig(await client.githubChatSyncConfig())}`
  }
  if (action === 'set' || action === 'config') {
    const fullName = requireArg(args[1], usage('/github'))
    const parsed = parseArgs(args.slice(2))
    const branch = flag(parsed, 'branch')
    const next = await client.updateGitHubChatSyncConfig({
      enabled: true,
      repoFullName: fullName,
      ...(branch ? { branch } : {}),
    })
    return `Encrypted chat sync configured: ${formatGithubChatSyncConfig(next)}`
  }
  if (action === 'enable' || action === 'disable') {
    const next = await client.updateGitHubChatSyncConfig({ enabled: action === 'enable' })
    return `Encrypted chat sync ${next.enabled ? 'enabled' : 'disabled'}: ${formatGithubChatSyncConfig(next)}`
  }
  if (action === 'run' || action === 'sync') {
    const result = await client.runGitHubChatSync()
    return formatGithubChatSyncResult(result)
  }
  return usage('/github')
}

function formatGithubChatSyncConfig(config: DaemonGitHubChatSyncConfig): string {
  return [
    config.enabled ? 'enabled' : 'disabled',
    `repo=${config.repoFullName ?? 'not configured'}`,
    `branch=${config.branch}`,
    `last=${formatTime(config.lastSyncedAt)}`,
    `status=${config.lastSyncStatus ?? 'none'}`,
    `sessions=${config.syncedSessions}`,
    config.lastSyncError ? `error=${config.lastSyncError}` : null,
  ]
    .filter(Boolean)
    .join(' ')
}

function formatGithubChatSyncResult(result: DaemonGitHubChatSyncResult): string {
  return [
    `Encrypted chat sync ${result.pushed ? 'pushed' : 'up to date'}: ${result.repoFullName}@${result.branch}`,
    `sessions=${result.syncedSessions}`,
    `changedFiles=${result.changedFiles}`,
    result.commitSha ? `commit=${result.commitSha.slice(0, 12)}` : 'commit=none',
  ].join(' ')
}

async function runMessagesCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'overview'
  if (action === 'overview' || action === 'current') {
    const overview = await client.messageSubscriptionOverview()
    return [
      `Message subscription: enabled=${overview.config.enabled} type=${overview.config.connectionType} connected=${overview.status.isConnected} auto=${overview.config.autoProcess}`,
      `Queue: pending=${overview.queueStatus.pending} processing=${overview.queueStatus.processing} completed=${overview.queueStatus.completed} failed=${overview.queueStatus.failed} total=${overview.queueStatus.totalProcessed}`,
      overview.status.lastError ? `Last error: ${overview.status.lastError}` : null,
      ...overview.recentMessages.slice(0, 5).map(formatMessageSubscriptionItem),
    ]
      .filter(Boolean)
      .join('\n')
  }
  if (action === 'start') {
    await client.startMessageSubscription()
    return 'Message subscription started.'
  }
  if (action === 'stop') {
    await client.stopMessageSubscription()
    return 'Message subscription stopped.'
  }
  if (action === 'refresh') {
    const result = await client.refreshMessageSubscription()
    return `Message subscription refresh: success=${result.success} count=${result.count} processed=${result.processed ?? 0}${result.error ? ` error=${result.error}` : ''}`
  }
  if (action === 'process') {
    const result = await client.processPendingMessageSubscription()
    return `Message subscription process: success=${result.success} processed=${result.processed}`
  }
  if (action === 'list') {
    const parsed = parseArgs(args.slice(1))
    const status = parsed.positional[0] as DaemonMessageSubscriptionItem['status'] | undefined
    const limit = parseInteger(flag(parsed, 'limit'), 'limit') ?? 20
    const messages = await client.messageSubscriptionMessages(status, limit)
    return messages.length
      ? messages.map(formatMessageSubscriptionItem).join('\n')
      : 'No message subscription items.'
  }
  if (action === 'reprocess') {
    const hash = requireArg(args[1], usage('/messages'))
    const item = await client.reprocessMessageSubscriptionMessage(hash)
    return item
      ? `Message reprocessed: ${formatMessageSubscriptionItem(item)}`
      : `Message not found: ${hash}`
  }
  if (action === 'delete') {
    const hash = requireArg(args[1], usage('/messages'))
    await client.deleteMessageSubscriptionMessage(hash)
    return `Message deleted: ${hash}`
  }
  return usage('/messages')
}

function formatMessageSubscriptionItem(item: DaemonMessageSubscriptionItem): string {
  return `${item.status.padEnd(10)} ${item.hash} ${item.type} ${item.source}\n  ${truncate(item.title || item.content || item.body)}${item.error ? `\n  error=${item.error}` : ''}`
}

async function runTeamDocsCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list' || action === 'current') {
    const configs = await client.teamDocs()
    return configs.length ? configs.map(formatTeamDocsConfig).join('\n') : 'No team docs sources.'
  }
  if (action === 'add') {
    const parsed = parseArgs(args.slice(1))
    const name = requireArg(parsed.positional[0], usage('/team-docs'))
    const config = await client.upsertTeamDocsConfig({
      id: flag(parsed, 'id'),
      name,
      description: flag(parsed, 'description'),
      token: requireArg(flag(parsed, 'token'), usage('/team-docs')),
      owner: requireArg(flag(parsed, 'owner'), usage('/team-docs')),
      repo: requireArg(flag(parsed, 'repo'), usage('/team-docs')),
      branch: flag(parsed, 'branch'),
      docsPath: flag(parsed, 'path'),
      enabled: !hasFlag(parsed, 'disabled'),
      autoSync: hasFlag(parsed, 'auto-sync') ? true : undefined,
      syncInterval: parseInteger(flag(parsed, 'sync-interval'), 'sync-interval'),
    })
    return `Team docs saved: ${formatTeamDocsConfig(config)}`
  }
  if (action === 'docs') {
    const id = requireArg(args[1], usage('/team-docs'))
    const docs = await client.teamDocsDocuments(id)
    return docs.length ? docs.map(formatTeamDoc).join('\n') : `No synced docs for ${id}.`
  }
  if (action === 'show') {
    const id = requireArg(args[1], usage('/team-docs'))
    const path = requireArg(args[2], usage('/team-docs'))
    const doc = await client.teamDocsDocument(id, path)
    return `# ${doc.path}\n${doc.content}`
  }
  if (action === 'test') {
    const id = requireArg(args[1], usage('/team-docs'))
    const result = await client.testTeamDocsConnection(id)
    return `Team docs test ${result.success ? 'ok' : 'failed'}: ${result.message}`
  }
  if (action === 'sync') {
    const id = requireArg(args[1], usage('/team-docs'))
    const result = await client.syncTeamDocsConfig(id)
    return `Team docs sync ${result.success ? 'ok' : 'failed'}: ${result.message}`
  }
  if (action === 'sync-all') {
    const result = await client.syncAllTeamDocs()
    return `Team docs sync-all ${result.success ? 'ok' : 'failed'}: ${result.succeeded}/${result.total} succeeded, ${result.failed} failed`
  }
  if (action === 'remove' || action === 'delete') {
    const id = requireArg(args[1], usage('/team-docs'))
    await client.deleteTeamDocsConfig(id)
    return `Team docs removed: ${id}`
  }
  return usage('/team-docs')
}

function formatTeamDocsConfig(config: DaemonTeamDocsConfig): string {
  return `${config.enabled ? 'enabled ' : 'disabled'} ${config.id} ${config.name} ${config.owner}/${config.repo}@${config.branch}:${config.docsPath}\n  docs=${config.syncedDocuments} lastSync=${formatTime(config.lastSyncAt)} status=${config.lastSyncStatus ?? 'never'}${config.lastSyncError ? ` error=${config.lastSyncError}` : ''}`
}

function formatTeamDoc(doc: DaemonTeamDocsDocument): string {
  return `${doc.path} ${formatBytes(doc.size)} synced=${formatTime(doc.syncedAt)}`
}

async function runPersonalDocsCommand(
  client: DaemonAdminSlashClient,
  args: string[],
): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list') {
    const docs = await client.personalDocs()
    return docs.length
      ? docs.map((doc) => `${doc.id} ${doc.path} updated=${formatTime(doc.updatedAt)}`).join('\n')
      : 'No personal docs.'
  }
  if (action === 'show') {
    const id = requireArg(args[1], usage('/personal-docs'))
    const doc = await client.personalDoc(id)
    return `# ${doc.path}\n${doc.content}`
  }
  if (action === 'set') {
    const parsed = parseArgs(args.slice(1))
    const path = requireArg(parsed.positional[0], usage('/personal-docs'))
    const content = requireArg(parsed.positional.slice(1).join(' '), usage('/personal-docs'))
    const doc = await client.upsertPersonalDoc({ id: flag(parsed, 'id'), path, content })
    return `Personal doc saved: ${doc.id} ${doc.path}`
  }
  if (action === 'delete' || action === 'remove') {
    const id = requireArg(args[1], usage('/personal-docs'))
    await client.deletePersonalDoc(id)
    return `Personal doc deleted: ${id}`
  }
  return usage('/personal-docs')
}

async function runPromptsCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list') {
    const prompts = await client.promptTemplates()
    return prompts.length
      ? prompts
          .map((prompt) => `${prompt.id} ${prompt.title}\n  ${truncate(prompt.body)}`)
          .join('\n')
      : 'No prompt templates.'
  }
  if (action === 'show') {
    const id = requireArg(args[1], usage('/prompts'))
    const prompt = (await client.promptTemplates()).find((entry) => entry.id === id)
    if (!prompt) throw new Error(`Prompt template not found: ${id}`)
    return `# ${prompt.title}\n${prompt.body}`
  }
  if (action === 'set') {
    const parsed = parseArgs(args.slice(1))
    const title = requireArg(parsed.positional[0], usage('/prompts'))
    const body = requireArg(parsed.positional.slice(1).join(' '), usage('/prompts'))
    const prompt = await client.upsertPromptTemplate({ id: flag(parsed, 'id'), title, body })
    return `Prompt template saved: ${prompt.id} ${prompt.title}`
  }
  if (action === 'delete' || action === 'remove') {
    const id = requireArg(args[1], usage('/prompts'))
    await client.deletePromptTemplate(id)
    return `Prompt template deleted: ${id}`
  }
  return usage('/prompts')
}

async function runSkillsAdminCommand(
  client: DaemonAdminSlashClient,
  args: string[],
): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list') {
    const skills =
      await client.request<
        Array<{ name: string; path: string; enabled: boolean; description: string | null }>
      >('/skills')
    return skills.length
      ? skills
          .map((skill) =>
            `${skill.enabled ? 'enabled ' : 'disabled'} ${skill.name}\n  ${skill.description ?? ''}\n  ${skill.path}`.trimEnd(),
          )
          .join('\n')
      : 'No daemon skills installed.'
  }
  if (action === 'enable' || action === 'disable') {
    const name = requireArg(args[1], usage('/skills-admin'))
    await client.request(`/skills/${encodeURIComponent(name)}/${action}`, { method: 'POST' })
    return `Skill ${action}d: ${name}`
  }
  return usage('/skills-admin')
}

interface RagFolder {
  id: string
  name: string
  sourceType?: string
  path?: string | null
  include?: string[]
  exclude?: string[]
}

interface RagDocument {
  id: string
  folderId: string
  title: string
  path?: string | null
  body?: string
  size?: number
  updatedAt?: number | string
  sourceFileId?: string | null
}

interface RagSearchHit {
  documentId: string
  folderId: string
  title: string
  score?: number
  snippet?: string
  path?: string | null
}

async function runRagAdminCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'folders'
  if (action === 'folders' || action === 'list') {
    const folders = await client.request<RagFolder[]>('/rag/folders')
    return folders.length ? folders.map(formatRagFolder).join('\n') : 'No RAG folders.'
  }
  if (action === 'add-folder' || action === 'add') {
    const parsed = parseArgs(args.slice(1))
    const name = requireArg(parsed.positional[0], usage('/rag-admin'))
    const folder = await client.request<RagFolder>('/rag/folders', {
      method: 'POST',
      body: {
        id: flag(parsed, 'id'),
        name,
        sourceType: hasFlag(parsed, 'git') ? 'git' : flag(parsed, 'source-type'),
        path: flag(parsed, 'path'),
        include: flagList(parsed, 'include'),
        exclude: flagList(parsed, 'exclude'),
      },
    })
    return `RAG folder saved: ${formatRagFolder(folder)}`
  }
  if (action === 'remove-folder' || action === 'delete-folder') {
    const id = requireArg(args[1], usage('/rag-admin'))
    await client.request(`/rag/folders/${encodeURIComponent(id)}`, { method: 'DELETE' })
    return `RAG folder removed: ${id}`
  }
  if (action === 'docs' || action === 'documents') {
    const folderId = requireArg(args[1], usage('/rag-admin'))
    const docs = await client.request<RagDocument[]>(
      `/rag/documents?folder=${encodeURIComponent(folderId)}`,
    )
    return docs.length ? docs.map(formatRagDocument).join('\n') : `No RAG documents in ${folderId}.`
  }
  if (action === 'show') {
    const id = requireArg(args[1], usage('/rag-admin'))
    const doc = await client.request<RagDocument>(`/rag/documents/${encodeURIComponent(id)}`)
    return `# ${doc.title}\n${doc.body ?? ''}`
  }
  if (action === 'set-doc' || action === 'add-doc') {
    const parsed = parseArgs(args.slice(1))
    const folderId = requireArg(parsed.positional[0], usage('/rag-admin'))
    const title = requireArg(parsed.positional[1], usage('/rag-admin'))
    const body = flag(parsed, 'body') ?? parsed.positional.slice(2).join(' ')
    const doc = await client.request<RagDocument>('/rag/documents', {
      method: 'POST',
      body: {
        id: flag(parsed, 'id'),
        folderId,
        title,
        body,
        path: flag(parsed, 'path'),
        sourceFileId: flag(parsed, 'source-file-id'),
        size: parseInteger(flag(parsed, 'size'), 'size'),
      },
    })
    return `RAG document saved: ${formatRagDocument(doc)}`
  }
  if (action === 'delete-doc' || action === 'remove-doc') {
    const id = requireArg(args[1], usage('/rag-admin'))
    await client.request(`/rag/documents/${encodeURIComponent(id)}`, { method: 'DELETE' })
    return `RAG document deleted: ${id}`
  }
  if (action === 'search') {
    const parsed = parseArgs(args.slice(1))
    const query = requireArg(parsed.positional.join(' '), usage('/rag-admin'))
    const limit = parseInteger(flag(parsed, 'limit'), 'limit')
    const suffix = limit ? `&limit=${encodeURIComponent(String(limit))}` : ''
    const hits = await client.request<RagSearchHit[]>(
      `/rag/search?q=${encodeURIComponent(query)}${suffix}`,
    )
    return hits.length ? hits.map(formatRagSearchHit).join('\n') : 'No RAG search results.'
  }
  if (action === 'sync')
    return json(await client.request('/rag/sync', { method: 'POST', body: {} }))
  if (action === 'vector-db') return json(await client.request('/rag/vector-db'))
  if (action === 'test-vector')
    return json(await client.request('/rag/vector-db/test', { method: 'POST', body: {} }))
  if (action === 'test-rerank')
    return json(await client.request('/rag/rerank/test', { method: 'POST', body: {} }))
  return usage('/rag-admin')
}

function formatRagFolder(folder: RagFolder): string {
  return `${folder.id} ${folder.name}${folder.sourceType ? ` type=${folder.sourceType}` : ''}${folder.path ? ` path=${folder.path}` : ''}`
}

function formatRagDocument(doc: RagDocument): string {
  return `${doc.id} ${doc.title} folder=${doc.folderId}${doc.path ? ` path=${doc.path}` : ''}${doc.size != null ? ` size=${formatBytes(doc.size)}` : ''}${doc.updatedAt ? ` updated=${formatTime(doc.updatedAt)}` : ''}`
}

function formatRagSearchHit(hit: RagSearchHit): string {
  return `${hit.documentId} ${hit.title} folder=${hit.folderId}${hit.score != null ? ` score=${hit.score.toFixed(3)}` : ''}${hit.path ? ` path=${hit.path}` : ''}\n  ${truncate(hit.snippet ?? '')}`
}

async function runBackupsCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list') {
    const backups = await client.backups()
    return backups.length ? backups.map(formatBackup).join('\n') : 'No daemon backups.'
  }
  if (action === 'create') {
    return `Backup created: ${formatBackup(await client.createBackup())}`
  }
  if (action === 'delete' || action === 'remove') {
    const id = requireArg(args[1], usage('/backups'))
    await client.deleteBackup(id)
    return `Backup deleted: ${id}`
  }
  return usage('/backups')
}

function formatBackup(backup: DaemonBackupItem): string {
  return `${backup.id} ${formatTime(backup.createdAt)} ${formatBytes(backup.sizeBytes)}\n  ${backup.path}\n  sha256=${backup.sha256}`
}

async function runTokensCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list') {
    const tokens = await client.extensionTokens()
    return tokens.length
      ? tokens
          .map(
            (token) =>
              `${token.active ? 'active ' : 'revoked'} ${token.id} ${token.label} scopes=${token.scopes.join(',')} expires=${token.expiresAt ?? 'never'}`,
          )
          .join('\n')
      : 'No daemon-issued extension tokens.'
  }
  if (action === 'issue') {
    const parsed = parseArgs(args.slice(1))
    const label = requireArg(parsed.positional[0], usage('/tokens'))
    const scopes = flagList(parsed, 'scope')
      .flatMap((value) => value.split(','))
      .map((value) => value.trim())
      .filter(Boolean) as DaemonExtensionTokenScope[]
    if (scopes.length === 0) return usage('/tokens')
    const issued = await client.issueExtensionToken({
      label,
      scopes,
      expiresAt: flag(parsed, 'expires-at'),
    })
    return [
      `Extension token issued: ${issued.label} (${issued.id})`,
      hasFlag(parsed, 'reveal')
        ? `Token: ${issued.token}`
        : `Token: ${maskSecretValue(issued.token)} - re-run with --reveal to print the full value once.`,
    ].join('\n')
  }
  if (action === 'revoke') {
    const id = requireArg(args[1], usage('/tokens'))
    await client.revokeExtensionToken(id)
    return `Extension token revoked: ${id}`
  }
  return usage('/tokens')
}

async function runSecretsCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list') {
    const result = await client.listSecrets()
    return result.keys.length ? result.keys.join('\n') : 'No secrets stored.'
  }
  if (action === 'set') {
    const key = requireArg(args[1], usage('/secrets'))
    const value = requireArg(args.slice(2).join(' '), usage('/secrets'))
    await client.setSecret(key, value)
    return `Secret set: ${key}`
  }
  if (action === 'remove' || action === 'delete') {
    const key = requireArg(args[1], usage('/secrets'))
    await client.removeSecret(key)
    return `Secret removed: ${key}`
  }
  return usage('/secrets')
}

async function runWebhooksCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list') {
    const endpoints = await client.webhookEndpoints()
    return endpoints.length
      ? endpoints.map(formatWebhookEndpoint).join('\n')
      : 'No inbound webhook endpoints.'
  }
  if (action === 'add') {
    const parsed = parseArgs(args.slice(1))
    const path = requireArg(parsed.positional[0], usage('/webhooks'))
    await client.upsertWebhookEndpoint({
      path,
      secretHeader: requireArg(flag(parsed, 'header'), usage('/webhooks')),
      secretValue: requireArg(flag(parsed, 'secret'), usage('/webhooks')),
      allowedEvents: flagList(parsed, 'event'),
      allowedIps: flagList(parsed, 'ip'),
      enabled: !hasFlag(parsed, 'disabled'),
    })
    const endpoint = (await client.webhookEndpoints()).find((entry) => entry.path === path)
    return endpoint
      ? `Webhook endpoint saved:\n${formatWebhookEndpoint(endpoint)}`
      : `Webhook endpoint saved: ${path}`
  }
  if (action === 'enable' || action === 'disable') {
    const id = requireArg(args[1], usage('/webhooks'))
    await client.setWebhookEndpointEnabled(id, action === 'enable')
    return `Webhook endpoint ${action}d: ${id}`
  }
  if (action === 'remove' || action === 'delete') {
    const id = requireArg(args[1], usage('/webhooks'))
    await client.deleteWebhookEndpoint(id)
    return `Webhook endpoint removed: ${id}`
  }
  return usage('/webhooks')
}

function formatWebhookEndpoint(endpoint: DaemonWebhookEndpointSummary): string {
  return `${endpoint.enabled ? 'enabled ' : 'disabled'} ${endpoint.id} ${endpoint.publicRoute}\n  path=${endpoint.path} header=${endpoint.secretHeader} events=${endpoint.allowedEvents.join(',') || '*'} ips=${endpoint.allowedIps.join(',') || '*'}`
}

async function runChannelsCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const [target, action = 'list', ...rest] = args
  if (!target || target === 'list') {
    const config = await client.config()
    const lines = config.channels.map(
      (channel) => `${channel.enabled === false ? 'disabled' : 'enabled '} ${channel.type}`,
    )
    const telegram = await client.telegramChannel().catch(() => null)
    if (telegram) lines.push(formatTelegram(telegram))
    const slack = await client
      .request('/api/v1/config/channels/slack')
      .then(unwrapData)
      .catch(() => null)
    if (slack) lines.push(`slack ${compactJson(slack)}`)
    return lines.length ? lines.join('\n') : 'No channels configured.'
  }
  if (target === 'slack') return runSlackChannelCommand(client, action, rest)
  if (target === 'pipeline') return runChannelPipelineCommand(client, action, rest)
  if (target === 'security' || target === 'webhook-security')
    return runChannelSecurityCommand(client, action, rest)
  if (target !== 'telegram') return usage('/channels')
  if (action === 'add') {
    const parsed = parseArgs(rest)
    const botToken = requireArg(flag(parsed, 'bot-token'), usage('/channels'))
    await client.upsertTelegramChannel({
      botToken,
      enabled: !hasFlag(parsed, 'disabled'),
      allowedUsers: flagList(parsed, 'user'),
      pairingRequired: !hasFlag(parsed, 'no-pairing-required'),
      pairingCodeTtl: parseInteger(flag(parsed, 'pairing-code-ttl'), 'pairing-code-ttl'),
      rateLimitPerMinute: parseInteger(
        flag(parsed, 'rate-limit-per-minute'),
        'rate-limit-per-minute',
      ),
    })
    const summary = await client.telegramChannel()
    return summary
      ? `Telegram channel saved:\n${formatTelegram(summary)}`
      : 'Telegram channel saved.'
  }
  if (action === 'pair') {
    const pairing = await client.telegramPairingCode()
    return `Telegram pairing code: ${pairing.code}\nExpires: ${pairing.expiresAt}`
  }
  if (action === 'users') {
    const summary = await client.telegramChannel()
    return summary?.allowedUsers.length
      ? summary.allowedUsers.join('\n')
      : 'No paired Telegram users.'
  }
  if (action === 'enable' || action === 'disable') {
    await client.setTelegramChannelEnabled(action === 'enable')
    const summary = await client.telegramChannel()
    return summary ? formatTelegram(summary) : `Telegram ${action}d.`
  }
  if (action === 'remove' || action === 'delete') {
    await client.deleteTelegramChannel()
    return 'Telegram channel removed.'
  }
  if (action === 'unpair') {
    const userId = requireArg(rest[0], usage('/channels'))
    await client.revokeTelegramAllowedUser(userId)
    return `Telegram user unpaired: ${userId}`
  }
  return usage('/channels')
}

function formatTelegram(summary: DaemonTelegramChannelSummary): string {
  return `telegram admin=${summary.enabled ? 'enabled' : 'disabled'} link=${summary.status} pairing=${summary.pairingRequired ? 'required' : 'open'} users=${summary.allowedUsers.length}`
}

async function runSlackChannelCommand(
  client: DaemonAdminSlashClient,
  action: string,
  rest: string[],
): Promise<string> {
  if (action === 'list' || action === 'show' || action === 'current') {
    return json(unwrapData(await client.request('/api/v1/config/channels/slack')))
  }
  if (action === 'add' || action === 'set') {
    const parsed = parseArgs(rest)
    await client.request('/api/v1/config/channels/slack', {
      method: 'POST',
      body: {
        enabled: !hasFlag(parsed, 'disabled'),
        botToken: requireArg(flag(parsed, 'bot-token'), usage('/channels')),
        signingSecret: requireArg(flag(parsed, 'signing-secret'), usage('/channels')),
        allowedChannels: flagList(parsed, 'channel'),
        allowedUsers: flagList(parsed, 'user'),
      },
    })
    return 'Slack channel saved.'
  }
  if (action === 'enable' || action === 'disable') {
    await client.request('/api/v1/config/channels/slack/enable', {
      method: 'POST',
      body: { enabled: action === 'enable' },
    })
    return `Slack channel ${action}d.`
  }
  if (action === 'remove' || action === 'delete') {
    await client.request('/api/v1/config/channels/slack', { method: 'DELETE' })
    return 'Slack channel removed.'
  }
  return usage('/channels')
}

function firstFlag(parsed: ParsedArgs, names: string[]): string | undefined {
  for (const name of names) {
    const value = flag(parsed, name)
    if (value != null) return value
  }
  return undefined
}

function queryFromParsed(parsed: ParsedArgs, names: Array<string | [string, ...string[]]>): string {
  const params = new URLSearchParams()
  for (const entry of names) {
    const name = Array.isArray(entry) ? entry[0] : entry
    const aliases = Array.isArray(entry) ? entry : [entry]
    const value = firstFlag(parsed, aliases)
    if (value != null) params.set(name, value)
  }
  return params.size ? `?${params.toString()}` : ''
}

async function runChannelPipelineCommand(
  client: DaemonAdminSlashClient,
  action: string,
  rest: string[],
): Promise<string> {
  if (action === 'list' || action === 'summary' || action === 'current') {
    return json(await client.request('/api/v1/channels/pipeline'))
  }
  if (action === 'health') {
    const sub = rest[0]?.toLowerCase()
    if (!sub)
      return json(
        unwrapData(await client.request('/api/v1/config/observability/channel-pipeline-health')),
      )
    if (sub === 'set') {
      const body = parseJsonObject(
        requireArg(rest.slice(1).join(' '), usage('/channels')),
        'channel pipeline health',
      )
      return json(
        unwrapData(
          await client.request('/api/v1/config/observability/channel-pipeline-health', {
            method: 'PUT',
            body,
          }),
        ),
      )
    }
    if (sub === 'history') {
      const parsed = parseArgs(rest.slice(1))
      return json(
        await client.request(
          `/api/v1/config/observability/channel-pipeline-health/history${queryFromParsed(parsed, ['route', 'device', 'since', 'cursor', 'limit'])}`,
        ),
      )
    }
  }
  if (action === 'history') {
    const parsed = parseArgs(rest)
    return json(
      await client.request(
        `/api/v1/config/observability/channel-pipeline-health/history${queryFromParsed(parsed, ['route', 'device', 'since', 'cursor', 'limit'])}`,
      ),
    )
  }
  return usage('/channels')
}

async function runChannelSecurityCommand(
  client: DaemonAdminSlashClient,
  action: string,
  rest: string[],
): Promise<string> {
  if (action === 'list' || action === 'summary' || action === 'current') {
    const parsed = parseArgs(rest)
    return json(
      await client.request(
        `/api/v1/webhooks/security${queryFromParsed(parsed, [
          ['channelType', 'channel-type', 'channelType'],
          ['verificationReady', 'verification-ready', 'verificationReady'],
          ['missingRequirement', 'missing-requirement', 'missingRequirement'],
          ['unreadyOffset', 'unready-offset', 'unreadyOffset'],
          ['unreadyLimit', 'unready-limit', 'unreadyLimit'],
        ])}`,
      ),
    )
  }
  if (action === 'policy') {
    const sub = rest[0]?.toLowerCase()
    if (!sub) return json(unwrapData(await client.request('/api/v1/config/security/webhooks')))
    if (sub === 'set') {
      const body = parseJsonObject(
        requireArg(rest.slice(1).join(' '), usage('/channels')),
        'webhook security policy',
      )
      return json(
        unwrapData(
          await client.request('/api/v1/config/security/webhooks', {
            method: 'PUT',
            body,
          }),
        ),
      )
    }
    if (sub === 'history') {
      const parsed = parseArgs(rest.slice(1))
      return json(
        await client.request(
          `/api/v1/config/security/webhooks/history${queryFromParsed(parsed, ['route', 'device', 'since', 'cursor', 'limit'])}`,
        ),
      )
    }
  }
  if (action === 'health') {
    const sub = rest[0]?.toLowerCase()
    if (!sub)
      return json(
        unwrapData(await client.request('/api/v1/config/observability/webhook-security-health')),
      )
    if (sub === 'set') {
      const body = parseJsonObject(
        requireArg(rest.slice(1).join(' '), usage('/channels')),
        'webhook security health',
      )
      return json(
        unwrapData(
          await client.request('/api/v1/config/observability/webhook-security-health', {
            method: 'PUT',
            body,
          }),
        ),
      )
    }
    if (sub === 'history') {
      const parsed = parseArgs(rest.slice(1))
      return json(
        await client.request(
          `/api/v1/config/observability/webhook-security-health/history${queryFromParsed(parsed, ['route', 'device', 'since', 'cursor', 'limit'])}`,
        ),
      )
    }
  }
  return usage('/channels')
}

async function runDevicesCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'challenge') {
    const parsed = parseArgs(args.slice(1))
    const result = unwrapData(
      await client.request<{
        data: { challengeId: string; challenge: string; payloadToSign: string; expiresAt: string }
      }>('/api/v1/devices/pairing/challenges', {
        method: 'POST',
        body: { ttlSeconds: parseInteger(flag(parsed, 'ttl'), 'ttl') },
      }),
    )
    return `Device pairing challenge: ${result.challengeId}\nPayload: ${result.payloadToSign}\nChallenge: ${result.challenge}\nExpires: ${result.expiresAt}`
  }
  if (action === 'complete') {
    const parsed = parseArgs(args.slice(1))
    const result = unwrapData(
      await client.request<{ data: { id: string } }>('/api/v1/devices/pairing/complete', {
        method: 'POST',
        body: {
          challengeId: requireArg(flag(parsed, 'challenge-id'), usage('/devices')),
          device: {
            id: requireArg(flag(parsed, 'device-id'), usage('/devices')),
            name: requireArg(flag(parsed, 'name'), usage('/devices')),
            role: requireArg(flag(parsed, 'role'), usage('/devices')),
          },
          publicKey: requireArg(flag(parsed, 'public-key'), usage('/devices')),
          signature: requireArg(flag(parsed, 'signature'), usage('/devices')),
        },
      }),
    )
    return `Device paired: ${result.id}`
  }
  if (action === 'revoke' || action === 'remove') {
    const id = requireArg(args[1], usage('/devices'))
    const result = unwrapData(
      await client.request<{ data: { id: string; revoked?: boolean } }>(
        `/api/v1/devices/${encodeURIComponent(id)}/pairing`,
        {
          method: 'DELETE',
        },
      ),
    )
    return `Device pairing revoked: ${result.id} revoked=${result.revoked ?? true}`
  }
  if (action !== 'list' && action !== 'current') return usage('/devices')
  const devices = await client.devices()
  return devices.length
    ? devices
        .map(
          (device) =>
            `${device.id} ${device.name} ${device.role} status=${device.status} lastSeen=${formatTime(device.lastSeen)}`,
        )
        .join('\n')
    : 'No devices registered.'
}

async function runPersonasCommand(client: DaemonAdminSlashClient): Promise<string> {
  const personas = await client.personas()
  return personas.length
    ? personas
        .map((persona) => `${persona.id} ${persona.name}\n  ${persona.description}`)
        .join('\n')
    : 'No personas registered.'
}

async function runPluginsCommand(client: DaemonAdminSlashClient): Promise<string> {
  const result = await client.request<{
    data: Array<{
      name: string
      version: string
      description: string
      status: string
      error?: string
    }>
    meta: {
      providerFactoryTypes: string[]
      channelFactoryTypes: string[]
      hookHandlers: Array<{ event: string; id: string; priority: number }>
    }
  }>('/api/v1/plugins')
  return [
    result.data.length
      ? result.data
          .map(
            (plugin) =>
              `${plugin.status.padEnd(10)} ${plugin.name}@${plugin.version}\n  ${plugin.description}${plugin.error ? `\n  error=${plugin.error}` : ''}`,
          )
          .join('\n')
      : 'No plugins loaded.',
    `Provider factories: ${result.meta.providerFactoryTypes.join(', ') || 'none'}`,
    `Channel factories: ${result.meta.channelFactoryTypes.join(', ') || 'none'}`,
    `Hook handlers: ${result.meta.hookHandlers.length}`,
  ].join('\n')
}

async function runAgentsCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list') {
    const agents = await client.agents()
    return agents.length
      ? agents
          .map(
            (agent) =>
              `${agent.id}${agent.source ? ` [${agent.source}]` : ''}\n  ${agent.description}`,
          )
          .join('\n')
      : 'No agents registered.'
  }
  if (action === 'create') {
    const parsed = parseArgs(args.slice(1))
    const id = requireArg(parsed.positional[0], usage('/agents'))
    if (!/^[a-z0-9][a-z0-9-_]*$/i.test(id)) {
      throw new Error(`Invalid agent id: ${id}`)
    }
    const description = requireArg(flag(parsed, 'description'), usage('/agents')).trim()
    const systemPrompt = requireArg(flag(parsed, 'prompt'), usage('/agents'))
    if (!description) throw new Error('Agent description must not be empty.')
    if (!systemPrompt.trim()) throw new Error('Agent system prompt must not be empty.')
    const result = await client.createUserAgent({
      id,
      name: flag(parsed, 'name'),
      description,
      base: flag(parsed, 'base'),
      model: flag(parsed, 'model'),
      temperature: parseNumberInRange(flag(parsed, 'temperature'), 'temperature', 0, 2),
      maxIterations: parsePositiveInteger(flag(parsed, 'max-iterations'), 'max-iterations'),
      systemPrompt,
    })
    return [
      `Agent saved: ${result.id}`,
      `Try: /subagent dispatch review this change --agent ${result.id}`,
    ].join('\n')
  }
  if (action === 'delete' || action === 'remove') {
    const id = requireArg(args[1], usage('/agents'))
    await client.deleteUserAgent(id)
    return `Agent deleted: ${id}`
  }
  return usage('/agents')
}

async function runCommandsCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list') {
    const commands = await client.userCommands()
    return commands.length
      ? commands
          .map(
            (cmd) =>
              `/${cmd.id} args=${cmd.args}${cmd.agent ? ` agent=${cmd.agent}` : ''}${cmd.model ? ` model=${cmd.model}` : ''}\n  ${cmd.description}`,
          )
          .join('\n')
      : 'No custom commands defined.'
  }
  if (action === 'create') {
    const parsed = parseArgs(args.slice(1))
    const id = requireArg(parsed.positional[0], usage('/commands'))
    const result = await client.createUserCommand({
      id,
      name: flag(parsed, 'name'),
      description: requireArg(flag(parsed, 'description'), usage('/commands')),
      args: flag(parsed, 'args') as 'none' | 'optional' | 'required' | undefined,
      agent: flag(parsed, 'agent'),
      model: flag(parsed, 'model'),
      body: requireArg(flag(parsed, 'body'), usage('/commands')),
    })
    return `Command saved: /${result.id}`
  }
  if (action === 'delete' || action === 'remove') {
    const id = requireArg(args[1], usage('/commands'))
    await client.deleteUserCommand(id)
    return `Command deleted: /${id}`
  }
  return usage('/commands')
}

async function runCronCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list') {
    const result = unwrapData(
      await client.request<{
        data: Array<{
          id: string
          name: string
          schedule: string
          instruction: string
          enabled: boolean
          lastRun?: string
          nextRun?: string
        }>
      }>('/api/v1/cron'),
    )
    return result.length
      ? result
          .map(
            (task) =>
              `${task.enabled ? 'enabled ' : 'disabled'} ${task.id} ${task.name}\n  schedule=${task.schedule} next=${task.nextRun ?? 'unknown'} last=${task.lastRun ?? 'never'}\n  ${task.instruction}`,
          )
          .join('\n')
      : 'No cron tasks.'
  }
  if (action === 'add') {
    const parsed = parseArgs(args.slice(1))
    const name = requireArg(parsed.positional[0], usage('/cron'))
    const result = unwrapData(
      await client.request<{
        data: { id: string; name: string; schedule: string; instruction: string; enabled: boolean }
      }>('/api/v1/cron', {
        method: 'POST',
        body: {
          name,
          schedule: requireArg(flag(parsed, 'schedule'), usage('/cron')),
          instruction: requireArg(
            flag(parsed, 'instruction') ?? parsed.positional.slice(1).join(' '),
            usage('/cron'),
          ),
          enabled: !hasFlag(parsed, 'disabled'),
        },
      }),
    )
    return `Cron task saved: ${result.id} ${result.name}`
  }
  if (action === 'remove' || action === 'delete') {
    const id = requireArg(args[1], usage('/cron'))
    await client.request(`/api/v1/cron/${encodeURIComponent(id)}`, { method: 'DELETE' })
    return `Cron task removed: ${id}`
  }
  return usage('/cron')
}

async function runExtensionsCommand(
  client: DaemonAdminSlashClient,
  args: string[],
): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list') {
    const items =
      await client.request<
        Array<{ id: string; name?: string; enabled?: boolean; version?: string }>
      >('/extensions')
    return items.length
      ? items
          .map(
            (item) =>
              `${item.enabled === false ? 'disabled' : 'enabled '} ${item.id}${item.name ? ` ${item.name}` : ''}${item.version ? `@${item.version}` : ''}`,
          )
          .join('\n')
      : 'No extensions installed.'
  }
  if (action === 'install') {
    const manifest = parseJsonObject(
      requireArg(args.slice(1).join(' '), usage('/extensions')),
      'extension manifest',
    )
    const result = await client.request<{ id: string }>('/extensions/install', {
      method: 'POST',
      body: { manifest },
    })
    return `Extension installed: ${result.id}`
  }
  if (action === 'enable' || action === 'disable') {
    const id = requireArg(args[1], usage('/extensions'))
    await client.request(`/extensions/${encodeURIComponent(id)}/${action}`, { method: 'POST' })
    return `Extension ${action}d: ${id}`
  }
  if (action === 'remove' || action === 'delete') {
    const id = requireArg(args[1], usage('/extensions'))
    await client.request(`/extensions/${encodeURIComponent(id)}`, { method: 'DELETE' })
    return `Extension removed: ${id}`
  }
  return usage('/extensions')
}

function formatImageGenJob(job: ImageGenJob): string {
  const progress = typeof job.progress === 'number' ? ` ${Math.round(job.progress * 100)}%` : ''
  const outputs = job.outputs?.length
    ? `\n  outputs=${job.outputs
        .map((output) => {
          const path = output.path ? ` path=${output.path}` : ''
          return `${output.id} (${output.mime})${path}`
        })
        .join(', ')}`
    : ''
  return `${job.status.padEnd(10)} ${job.id} ${job.providerId ?? 'unknown-provider'}${progress}\n  ${truncate(job.prompt)}${outputs}${job.error ? `\n  error=${job.error}` : ''}`
}

async function getImageJob(client: DaemonAdminSlashClient, id: string): Promise<ImageGenJob> {
  if (client.imageGenJob) return client.imageGenJob(id)
  return client.request<ImageGenJob>(`/image-gen/jobs/${encodeURIComponent(id)}`)
}

async function waitForImageJob(
  client: DaemonAdminSlashClient,
  id: string,
  timeoutMs: number,
): Promise<ImageGenJob> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const job = await getImageJob(client, id)
    if (['succeeded', 'failed', 'cancelled'].includes(job.status)) return job
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  throw new Error(`Timed out waiting for image job ${id}`)
}

function sanitizeImageFilename(id: string): string {
  return `${id.replace(/[^a-z0-9._-]+/gi, '-')}.png`
}

async function downloadImageOutputs(
  client: DaemonAdminSlashClient,
  job: ImageGenJob,
  outputDir: string,
): Promise<string[]> {
  if (!client.imageGenFile) {
    return ['Image download unavailable in this CLI client.']
  }
  await mkdir(outputDir, { recursive: true })
  const lines: string[] = []
  for (const output of job.outputs ?? []) {
    const blob = await client.imageGenFile(output.id)
    const bytes = Buffer.from(await blob.arrayBuffer())
    const target = join(outputDir, sanitizeImageFilename(output.id))
    await writeFile(target, bytes)
    lines.push(`Downloaded ${output.id} -> ${target}`)
  }
  return lines.length > 0 ? lines : ['No image outputs to download.']
}

async function runImageGenCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'jobs'
  if (action === 'providers') {
    const providers =
      await client.request<Array<{ id: string; label?: string; name?: string; models?: string[] }>>(
        '/image-gen/providers',
      )
    return providers.length
      ? providers
          .map((provider) => {
            const label = provider.name ?? provider.label ?? provider.id
            return `${provider.id} ${label}${provider.models?.length ? ` models=${provider.models.join(',')}` : ''}`
          })
          .join('\n')
      : 'No image generation providers.'
  }
  if (action === 'jobs') {
    const parsed = parseArgs(args.slice(1))
    const limit = flag(parsed, 'limit') ?? '20'
    const jobs = await client.request<ImageGenJob[]>(
      `/image-gen/jobs?limit=${encodeURIComponent(limit)}`,
    )
    return jobs.length
      ? jobs.map((job) => formatImageGenJob(job)).join('\n')
      : 'No image generation jobs.'
  }
  if (action === 'create') {
    const parsed = parseArgs(args.slice(1))
    const providerId = requireArg(parsed.positional[0], usage('/image-gen'))
    const prompt = requireArg(parsed.positional.slice(1).join(' '), usage('/image-gen'))
    const params = flag(parsed, 'params')
      ? parseJsonObject(flag(parsed, 'params')!, 'image params')
      : {}
    const outputDir = flag(parsed, 'output')
    if (outputDir) params.outputDir = outputDir
    const job = await client.request<ImageGenJob>('/image-gen/jobs', {
      method: 'POST',
      body: { providerId, prompt, params: Object.keys(params).length > 0 ? params : undefined },
    })
    if (!hasFlag(parsed, 'wait') && !flag(parsed, 'wait') && !flag(parsed, 'download')) {
      return `Image job queued: ${job.id} (${job.status})`
    }
    const waitMs = parseInteger(flag(parsed, 'wait'), 'wait') ?? 10 * 60_000
    const finished = await waitForImageJob(client, job.id, waitMs)
    const lines = [
      `Image job finished: ${finished.id} (${finished.status})`,
      formatImageGenJob(finished),
    ]
    const downloadDir = flag(parsed, 'download')
    if (downloadDir) {
      lines.push(...(await downloadImageOutputs(client, finished, downloadDir)))
    }
    return lines.join('\n')
  }
  if (action === 'cancel') {
    const id = requireArg(args[1], usage('/image-gen'))
    await client.request(`/image-gen/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' })
    return `Image job cancelled: ${id}`
  }
  return usage('/image-gen')
}

async function runJobsCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (!client.jobs) return 'Jobs client is not available in this TUI build.'
  if (action === 'list') {
    const parsed = parseArgs(args.slice(1))
    const page = await client.jobs.list({ status: flag(parsed, 'status'), kind: flag(parsed, 'kind'), limit: parseInteger(flag(parsed, 'limit'), 'limit'), offset: parseInteger(flag(parsed, 'offset'), 'offset') })
    return page.jobs.length ? [
      ...page.jobs.map((job) => `${job.id}  ${job.kind ?? 'job'}  ${job.status}  ${job.succeeded}/${job.total}${job.activity?.some((item) => item.status === 'running' && item.approvalRequestId) ? '  needs approval' : ''}`),
      'Next: /tasks status <id> · /tasks items <id> · /tasks cancel <id>',
      ...(page.nextOffset === null ? [] : [`More: /tasks list --offset ${page.nextOffset}`]),
    ].join('\n') : 'No background jobs.'
  }
  if (action === 'status') {
    const id = requireArg(args[1], usage('/jobs'))
    return json(await client.jobs.get(id))
  }
  if (action === 'items') {
    const parsed = parseArgs(args.slice(2))
    const id = requireArg(args[1], usage('/jobs'))
    return json(await client.jobs.getItems(id, parseInteger(flag(parsed, 'since'), 'since') ?? 0))
  }
  if (action === 'cancel') {
    const id = requireArg(args[1], usage('/jobs'))
    await client.jobs.cancel(id)
    return `Job cancelled: ${id}`
  }
  if (action === 'submit') {
    const parsed = parseArgs(args.slice(1))
    const raw = requireArg(parsed.positional[0], usage('/jobs'))
    const items = parseJsonValue(raw)
    if (!Array.isArray(items)) throw new Error('Job items must be a JSON array.')
    const result = await client.jobs.submitBatch({
      items,
      concurrency: parseInteger(flag(parsed, 'concurrency'), 'concurrency') ?? 1,
      failureMode: flag(parsed, 'failure-mode') === 'abort' ? 'abort' : 'continue',
      preserveOrder: !hasFlag(parsed, 'unordered'),
    })
    return `Job submitted: ${result.jobId} total=${result.total} status=${result.status}`
  }
  return usage('/jobs')
}

async function runMigrationCommand(
  client: DaemonAdminSlashClient,
  args: string[],
): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'status'
  if (!client.migration) return 'Migration client is not available in this TUI build.'
  if (action === 'run') {
    const parsed = parseArgs(args.slice(1))
    const sourcePath = requireArg(parsed.positional[0], usage('/migration'))
    const result = await client.migration.run({
      sourcePath,
      steps: flagList(parsed, 'step'),
      dryRun: hasFlag(parsed, 'dry-run'),
      conflict: flag(parsed, 'conflict') === 'overwrite' ? 'overwrite' : 'skip',
    })
    return `Migration started: ${result.migrationId} status=${result.status} dryRun=${result.dryRun}`
  }
  if (action === 'status') {
    const id = requireArg(args[1], usage('/migration'))
    return json(await client.migration.get(id))
  }
  if (action === 'report') {
    const id = requireArg(args[1], usage('/migration'))
    return json(await client.migration.getReport(id))
  }
  if (action === 'cancel') {
    const id = requireArg(args[1], usage('/migration'))
    await client.migration.cancel(id)
    return `Migration cancelled: ${id}`
  }
  return usage('/migration')
}

async function runObservabilityCommand(
  client: DaemonAdminSlashClient,
  args: string[],
): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'snapshot'
  if (action === 'snapshot')
    return json(await client.observabilitySnapshot(parseObservabilityRange(args[1])))
  if (action === 'events') {
    const parsed = parseArgs(args.slice(1))
    const events = await client.observabilityEvents({
      limit: parseInteger(flag(parsed, 'limit'), 'limit') ?? 20,
      eventType: flag(parsed, 'event-type'),
      severity: parseObservabilitySeverity(flag(parsed, 'severity')),
    })
    return events.length
      ? events.map((event) => compactJson(event)).join('\n')
      : 'No observability events.'
  }
  if (action === 'crashes') {
    const parsed = parseArgs(args.slice(1))
    const crashes = await client.observabilityCrashes(
      parseInteger(flag(parsed, 'limit'), 'limit') ?? 5,
    )
    return crashes.length
      ? crashes.map((event) => compactJson(event)).join('\n')
      : 'No crash events.'
  }
  if (action === 'privacy') return json(await client.observabilityPrivacy())
  if (action === 'prune') {
    const result = await client.pruneObservability()
    return `Observability pruned: events=${result.deletedEvents} feedback=${result.deletedFeedback}`
  }
  return usage('/observability')
}

async function runMetricsCommand(client: DaemonAdminSlashClient): Promise<string> {
  const result = await client.request<{ data?: unknown }>('/api/v1/metrics')
  return json('data' in result ? result.data : result)
}

async function runMarketplacesCommand(
  client: DaemonAdminSlashClient,
  args: string[],
): Promise<string> {
  const action = args[0]?.toLowerCase() ?? 'list'
  if (action === 'list') {
    const marketplaces = await client.listMarketplaces()
    return marketplaces.length
      ? marketplaces.map((entry) => `${entry.name} ${entry.url}`).join('\n')
      : 'No skill marketplaces registered.'
  }
  if (action === 'add') {
    const name = requireArg(args[1], usage('/marketplaces'))
    const url = requireArg(args[2], usage('/marketplaces'))
    await client.addMarketplace(name, url)
    return `Marketplace added: ${name}`
  }
  if (action === 'remove' || action === 'delete') {
    const name = requireArg(args[1], usage('/marketplaces'))
    const result = await client.removeMarketplace(name)
    return result.removed ? `Marketplace removed: ${name}` : `Marketplace not found: ${name}`
  }
  return usage('/marketplaces')
}

async function runSubagentCommand(client: DaemonAdminSlashClient, args: string[]): Promise<string> {
  const action = args[0]?.toLowerCase()
  if (action !== 'dispatch') return usage('/subagent')
  const parsed = parseArgs(args.slice(1))
  const prompt = requireArg(parsed.positional.join(' '), usage('/subagent'))
  const result = await client.dispatchSubagent({
    prompt,
    system: flag(parsed, 'system'),
    model: flag(parsed, 'model'),
    tools: flag(parsed, 'tools')
      ?.split(',')
      .map((tool) => tool.trim())
      .filter(Boolean),
    maxIterations: parseInteger(flag(parsed, 'max-iterations'), 'max-iterations'),
  })
  return [
    `Subagent ${result.sessionId}: status=${result.status} iterations=${result.iterations}${result.truncated ? ' truncated=true' : ''}`,
    result.output,
  ].join('\n')
}
