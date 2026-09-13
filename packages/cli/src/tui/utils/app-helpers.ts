import type {
  DaemonOutboundWebhookDeadLetter,
  DaemonOutboundWebhookDelivery,
  DaemonOutboundWebhookEvent,
  DaemonOutboundWebhookSummary,
  DaemonProviderInfo,
} from '@sepilotd/api-client'
import type {
  CliFileMemorySnapshot,
  CliMemoryAuditEntry,
  CliMemoryLifecycleStatus,
  CliMemoryMaintenanceResult,
} from '../../client/http.js'
import { buildTuiMemoryBacklogSnapshot } from './memory.js'

const OUTBOUND_WEBHOOK_EVENTS: DaemonOutboundWebhookEvent[] = [
  'post:agent:run',
  'post:tool:execute',
  'post:process:start',
  'post:process:exit',
  'post:llm:call',
  'post:channel:msg',
]

export function parseTuiHookValueFlag(
  args: string[],
  index: number,
  flag: string,
): { value: string | null; nextIndex: number } {
  const current = args[index]
  if (!current) {
    return { value: null, nextIndex: index + 1 }
  }
  if (current.startsWith(`${flag}=`)) {
    return { value: current.slice(flag.length + 1), nextIndex: index + 1 }
  }
  const next = args[index + 1]
  if (!next || next.startsWith('--')) {
    return { value: null, nextIndex: index + 1 }
  }
  return { value: next, nextIndex: index + 2 }
}

export function parseTuiHookHeaders(entries: string[]): Record<string, string> {
  const headers: Record<string, string> = {}
  for (const entry of entries) {
    const separator = entry.indexOf('=')
    if (separator <= 0) {
      throw new Error(`Invalid --header entry: ${entry}. Use KEY=VALUE.`)
    }
    headers[entry.slice(0, separator)] = entry.slice(separator + 1)
  }
  return headers
}

export function parseTuiHookEvents(events: string[]): DaemonOutboundWebhookEvent[] {
  if (events.length === 0) {
    throw new Error(`At least one --event is required. Available: ${OUTBOUND_WEBHOOK_EVENTS.join(', ')}`)
  }
  const invalidEvents = events.filter(
    (event) => !OUTBOUND_WEBHOOK_EVENTS.includes(event as DaemonOutboundWebhookEvent),
  )
  if (invalidEvents.length > 0) {
    throw new Error(`Unsupported hook event: ${invalidEvents.join(', ')}. Available: ${OUTBOUND_WEBHOOK_EVENTS.join(', ')}`)
  }
  return [...new Set(events)] as DaemonOutboundWebhookEvent[]
}

export function parseTuiHookLimit(value: string | undefined): number | undefined {
  if (!value) {
    return undefined
  }
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined
}

export function formatTuiHookSummary(webhook: DaemonOutboundWebhookSummary): string {
  const extras = [
    webhook.hasSecret ? 'secret' : null,
    webhook.headerKeys.length > 0 ? `headers=${webhook.headerKeys.join(',')}` : null,
    `retry=${webhook.retry.maxAttempts}x/${webhook.retry.backoffMs}ms`,
  ].filter(Boolean)
  return [
    `${webhook.id}  ${webhook.enabled ? 'enabled' : 'disabled'}  ${webhook.events.join(', ')}`,
    `  ${webhook.url}  [${extras.join(' ')}]`,
  ].join('\n')
}

export function formatTuiHookDelivery(delivery: DaemonOutboundWebhookDelivery): string {
  const extras = [
    `event=${delivery.hookEvent}`,
    `attempts=${delivery.attemptCount}`,
    `ms=${delivery.durationMs}`,
    typeof delivery.statusCode === 'number' ? `http=${delivery.statusCode}` : null,
    delivery.sessionId ? `session=${delivery.sessionId}` : null,
    delivery.replayedFromDeliveryId ? `replayOf=${delivery.replayedFromDeliveryId}` : null,
  ].filter(Boolean)
  return `${delivery.webhookId}  ${delivery.deliveryStatus}  ${delivery.timestamp}  ${delivery.deliveryId}  [${extras.join(' ')}]${delivery.error ? `\n  ${delivery.error}` : ''}`
}

export function formatTuiHookDeadLetter(deadLetter: DaemonOutboundWebhookDeadLetter): string {
  const extras = [
    `root=${deadLetter.rootDeliveryId}`,
    `latest=${deadLetter.latestDeliveryId}`,
    `replays=${deadLetter.replayCount}`,
    `attempts=${deadLetter.attemptCount}`,
    typeof deadLetter.statusCode === 'number' ? `http=${deadLetter.statusCode}` : null,
    deadLetter.sessionId ? `session=${deadLetter.sessionId}` : null,
    deadLetter.acknowledgedAt ? `ackedAt=${deadLetter.acknowledgedAt}` : null,
  ].filter(Boolean)
  return `${deadLetter.webhookId}  ${deadLetter.state}  ${deadLetter.lastAttemptAt}  [${extras.join(' ')}]${deadLetter.error ? `\n  ${deadLetter.error}` : ''}${deadLetter.acknowledgmentNote ? `\n  note=${deadLetter.acknowledgmentNote}` : ''}`
}

export function buildTuiHooksUsage(): string {
  return [
    'Usage: /hooks [list|commands]',
    '       /hooks add <url> --event <event> [--event <event>] [--header KEY=VALUE] [--secret value] [--retry-attempts n] [--retry-backoff-ms ms] [--disabled]',
    '       /hooks remove <id> | enable <id> | disable <id>',
    '       /hooks deliveries [id] [--status success|error] [--limit n]',
    '       /hooks dead-letters [id] [--state open|acknowledged|all] [--limit n]',
    '       /hooks replay <deliveryId> [--force] | ack <rootDeliveryId> [note...]',
    '       /hooks replay-failed [id] [--limit n] [--force]',
    `Events: ${OUTBOUND_WEBHOOK_EVENTS.join(', ')}`,
  ].join('\n')
}

export const TUI_FILE_MEMORY_USAGE = [
  'Usage: /memory file [list]',
  '       /memory file show <section>',
  '       /memory file today | yesterday',
  '       /memory file set <section> -- <content>',
  '       /memory file delete <section>',
  '       /memory backlog [list]',
  '       /memory backlog add <content>',
  '       /memory backlog done <query>',
  '       /memory lifecycle',
  '       /memory audit [memoryId]',
  '       /memory maintenance [--apply]',
].join('\n')

export function formatTuiFileMemorySummary(snapshot: CliFileMemorySnapshot): string {
  const sections = snapshot.sections.length > 0
    ? snapshot.sections
        .map((section) => `  - ${section.title} (${lineCountLabel(section.content)})`)
        .join('\n')
    : '  none'
  return [
    'Markdown memory',
    `File: ${snapshot.memoryPath}`,
    `Today: ${snapshot.todayNotePath} (${lineCountLabel(snapshot.todayNote)})`,
    `Yesterday: ${snapshot.yesterdayNotePath} (${lineCountLabel(snapshot.yesterdayNote)})`,
    'Sections:',
    sections,
  ].join('\n')
}

export function formatTuiDailyMemory(label: 'today' | 'yesterday', path: string, note: string): string {
  return [
    `${label} daily note`,
    path,
    note.trim() || '(empty)',
  ].join('\n')
}

export function formatTuiBacklogSummary(snapshot: CliFileMemorySnapshot): string {
  const backlog = buildTuiMemoryBacklogSnapshot(snapshot)
  const openLoops = backlog.openLoopQueue.length > 0
    ? backlog.openLoopQueue.map((item) => `  - ${item.replace(/^-\s*/, '')}`).join('\n')
    : '  none'
  const todayBacklog = backlog.todayBacklog.length > 0
    ? backlog.todayBacklog.map((item) => `  - ${item.replace(/^-\s*/, '')}`).join('\n')
    : '  none'
  const reflections = backlog.todayReflections.length > 0
    ? backlog.todayReflections.map((item) => `  - ${item.replace(/^-\s*/, '')}`).join('\n')
    : '  none'

  return [
    'Memory backlog',
    `Open loop queue: ${backlog.openLoopQueue.length}`,
    openLoops,
    `Today backlog: ${backlog.todayBacklog.length}`,
    todayBacklog,
    `Reflection ledger: ${backlog.todayReflections.length}`,
    reflections,
  ].join('\n')
}

export function formatTuiMemoryLifecycleSummary(snapshot: CliMemoryLifecycleStatus): string {
  return [
    'Memory lifecycle',
    `Total memories: ${snapshot.totalMemories}`,
    `Cleanup candidates: ${snapshot.pruneCandidateMemories}`,
    `Embeddings: ${snapshot.pendingEmbeddings} pending, ${snapshot.failedEmbeddings} failed`,
    [
      `conversation=${snapshot.conversationMemories}`,
      `documents=${snapshot.documentMemories}`,
      `skills=${snapshot.skillMemories}`,
      `user=${snapshot.userMemories}`,
    ].join('  '),
    `Cleanup signals: ${snapshot.staleConversationMemories} stale, ${snapshot.lowImportanceConversationMemories} low-importance`,
    `Last audit: ${snapshot.lastAuditAt ?? 'never'}`,
  ].join('\n')
}

export function formatTuiMemoryAuditSummary(entries: CliMemoryAuditEntry[]): string {
  if (entries.length === 0) {
    return 'Memory audit trail\n  none'
  }

  return [
    'Memory audit trail',
    ...entries.map((entry) => {
      const parts = [
        `${entry.action} ${entry.memoryId}`,
        `actor=${entry.actor}`,
        `at=${entry.createdAt}`,
        entry.reason ? `reason=${entry.reason}` : null,
      ].filter(Boolean)
      return `  - ${parts.join('  ')}`
    }),
  ].join('\n')
}

export function formatTuiMemoryMaintenanceSummary(result: CliMemoryMaintenanceResult): string {
  return [
    result.dryRun ? 'Memory maintenance preview' : 'Memory maintenance completed',
    `Would prune: ${result.wouldPrune}`,
    `Pruned: ${result.pruned}`,
    `Importance updated: ${result.importanceUpdated}`,
    '',
    formatTuiMemoryLifecycleSummary(result.status),
    result.dryRun
      ? 'Use /memory maintenance --apply to prune matching entries.'
      : 'Maintenance audit entry recorded.',
  ].join('\n')
}

export function lineCountLabel(content: string): string {
  const trimmed = content.trim()
  if (!trimmed) return 'empty'
  const count = trimmed.split('\n').length
  return `${count} line${count === 1 ? '' : 's'}`
}

export function splitSectionAndContent(args: string[]): { section: string; content: string } | null {
  const divider = args.indexOf('--')
  if (divider <= 0 || divider === args.length - 1) {
    return null
  }
  return {
    section: args.slice(0, divider).join(' ').trim(),
    content: args.slice(divider + 1).join(' ').trim(),
  }
}

export function formatProviderHealthSummary(
  provider: Pick<DaemonProviderInfo, 'health'>,
): string {
  if (provider.health.status === 'env_missing') {
    return `env missing${provider.health.missingEnvVars.length > 0 ? `: ${provider.health.missingEnvVars.join(', ')}` : ''}`
  }

  if (provider.health.status === 'unavailable') {
    return provider.health.message
      ? `unavailable: ${provider.health.message}`
      : 'unavailable'
  }

  return 'ready'
}
