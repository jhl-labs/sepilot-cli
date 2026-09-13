import chalk from 'chalk'
import type {
  DaemonOutboundWebhookDeadLetter,
  DaemonOutboundWebhookDelivery,
  DaemonOutboundWebhookEvent,
  DaemonOutboundWebhookSummary,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { detectCliLocale } from '../utils/locale.js'

export async function hooksCommandsCommand(options: { url?: string }): Promise<void> {
  const hooks = (await new DaemonClient(options.url).config()).hooks.commandHooks ?? []
  output(hooks, (items) => items.length ? items.map((hook, index) => `${index + 1}. ${hook.event}  ${hook.enabled ? 'enabled' : 'disabled'}  ${hook.async ? 'background job' : 'blocking'}  timeout=${hook.timeoutMs}ms${hook.toolMatcher ? `  matcher=${hook.toolMatcher}` : ''}`).join('\n') : 'No command hooks configured. Configure hooks.commandHooks; inspect background runs with sepilot jobs list --kind hook.')
}

const OUTBOUND_WEBHOOK_EVENTS = [
  'post:agent:run',
  'post:tool:execute',
  'post:process:start',
  'post:process:exit',
  'post:llm:call',
  'post:channel:msg',
] as const satisfies readonly DaemonOutboundWebhookEvent[]

const HOOKS_COPY = {
  en: {
    invalidEntry: (flagName: string, entry: string) =>
      `Invalid ${flagName} entry: ${entry}. Expected KEY=VALUE.`,
    eventRequired: 'At least one --event is required.',
    unsupportedEvent: (event: string, expected: string) =>
      `Unsupported webhook event: ${event}. Expected one of ${expected}.`,
    enabled: 'enabled',
    disabled: 'disabled',
    secret: 'secret',
    headers: (keys: string) => `headers=${keys}`,
    retry: (maxAttempts: number, backoffMs: number) => `retry=${maxAttempts}x/${backoffMs}ms`,
    success: 'success',
    error: 'error',
    acknowledged: 'acknowledged',
    deadLetter: 'dead-letter',
    note: (note: string) => `note=${note}`,
    noWebhooks:
      'No outbound webhooks configured.\nUse `sepilot hooks add <url> --event post:agent:run` to add one.',
    noDeliveries: 'No outbound webhook deliveries recorded yet.',
    noDeadLetters: 'No unresolved outbound webhook dead letters.',
    nextCursor: (cursor: string) => `Next cursor: ${cursor}`,
    acknowledgedDeadLetter: (rootDeliveryId: string, line: string) =>
      `Acknowledged outbound webhook dead letter: ${rootDeliveryId}\n${line}`,
    webhookUpdated: (url: string) => `Outbound webhook updated: ${url}`,
    webhookUpdatedWith: (url: string, summary: string) =>
      `Outbound webhook updated: ${url}\n${summary}`,
    webhookReplayed: (sourceId: string, deliveryId: string, line: string) =>
      `Outbound webhook replayed: ${sourceId} -> ${deliveryId}\n${line}`,
    noDeadLettersToReplay: 'No unresolved outbound webhook dead letters to replay.',
    replayedDeadLetters: (count: number) =>
      `Replayed ${count} ${count === 1 ? 'dead-letter delivery' : 'dead-letter deliveries'}.`,
    webhookRemoved: (id: string) => `Outbound webhook removed: ${id}`,
    webhookEnabled: (id: string) => `Outbound webhook enabled: ${id}`,
    webhookEnabledWith: (id: string, summary: string) =>
      `Outbound webhook enabled: ${id}\n${summary}`,
    webhookDisabled: (id: string) => `Outbound webhook disabled: ${id}`,
    webhookDisabledWith: (id: string, summary: string) =>
      `Outbound webhook disabled: ${id}\n${summary}`,
  },
  ko: {
    invalidEntry: (flagName: string, entry: string) =>
      `잘못된 ${flagName} 항목: ${entry}. KEY=VALUE 형식이어야 합니다.`,
    eventRequired: '최소 하나의 --event가 필요합니다.',
    unsupportedEvent: (event: string, expected: string) =>
      `지원되지 않는 웹훅 이벤트: ${event}. 다음 중 하나여야 합니다: ${expected}.`,
    enabled: '활성화됨',
    disabled: '비활성화됨',
    secret: 'secret 있음',
    headers: (keys: string) => `headers=${keys}`,
    retry: (maxAttempts: number, backoffMs: number) => `retry=${maxAttempts}회/${backoffMs}ms`,
    success: '성공',
    error: '오류',
    acknowledged: '확인됨',
    deadLetter: '배달 실패',
    note: (note: string) => `메모=${note}`,
    noWebhooks:
      '구성된 아웃바운드 웹훅이 없습니다.\n`sepilot hooks add <url> --event post:agent:run`로 추가하세요.',
    noDeliveries: '아직 기록된 아웃바운드 웹훅 배달이 없습니다.',
    noDeadLetters: '해결되지 않은 아웃바운드 웹훅 배달 실패가 없습니다.',
    nextCursor: (cursor: string) => `다음 cursor: ${cursor}`,
    acknowledgedDeadLetter: (rootDeliveryId: string, line: string) =>
      `아웃바운드 웹훅 배달 실패를 확인 처리했습니다: ${rootDeliveryId}\n${line}`,
    webhookUpdated: (url: string) => `아웃바운드 웹훅 업데이트됨: ${url}`,
    webhookUpdatedWith: (url: string, summary: string) =>
      `아웃바운드 웹훅 업데이트됨: ${url}\n${summary}`,
    webhookReplayed: (sourceId: string, deliveryId: string, line: string) =>
      `아웃바운드 웹훅을 재실행했습니다: ${sourceId} -> ${deliveryId}\n${line}`,
    noDeadLettersToReplay: '재실행할 해결되지 않은 아웃바운드 웹훅 배달 실패가 없습니다.',
    replayedDeadLetters: (count: number) => `배달 실패 ${count}건을 재실행했습니다.`,
    webhookRemoved: (id: string) => `아웃바운드 웹훅 제거됨: ${id}`,
    webhookEnabled: (id: string) => `아웃바운드 웹훅 활성화됨: ${id}`,
    webhookEnabledWith: (id: string, summary: string) =>
      `아웃바운드 웹훅 활성화됨: ${id}\n${summary}`,
    webhookDisabled: (id: string) => `아웃바운드 웹훅 비활성화됨: ${id}`,
    webhookDisabledWith: (id: string, summary: string) =>
      `아웃바운드 웹훅 비활성화됨: ${id}\n${summary}`,
  },
} as const

type HooksCopy = (typeof HOOKS_COPY)[keyof typeof HOOKS_COPY]

function hooksCopy(): HooksCopy {
  return HOOKS_COPY[detectCliLocale()] ?? HOOKS_COPY.en
}

export interface HooksMutationOptions {
  url?: string
}

export interface HooksAddOptions extends HooksMutationOptions {
  event: string[]
  header?: string[]
  secret?: string
  disabled?: boolean
  retryAttempts?: string
  retryBackoffMs?: string
}

export interface HooksDeliveriesOptions extends HooksMutationOptions {
  limit?: string
  status?: 'success' | 'error'
  cursor?: string
}

export interface HooksDeadLettersOptions extends HooksMutationOptions {
  limit?: string
  state?: 'open' | 'acknowledged' | 'all'
  cursor?: string
}

export interface HooksReplayOptions extends HooksMutationOptions {
  force?: boolean
}

export interface HooksReplayFailedOptions extends HooksDeadLettersOptions {
  force?: boolean
}

export interface HooksAckOptions extends HooksMutationOptions {
  note?: string
}

function parseKeyValueEntries(
  entries: string[] | undefined,
  flagName: string,
  copy: HooksCopy = HOOKS_COPY.en,
): Record<string, string> {
  const values: Record<string, string> = {}

  for (const entry of entries ?? []) {
    const separator = entry.indexOf('=')
    if (separator <= 0) {
      throw new Error(copy.invalidEntry(flagName, entry))
    }

    const key = entry.slice(0, separator)
    const value = entry.slice(separator + 1)
    values[key] = value
  }

  return values
}

function parseWebhookEvents(
  events: string[],
  copy: HooksCopy = HOOKS_COPY.en,
): DaemonOutboundWebhookEvent[] {
  if (events.length === 0) {
    throw new Error(copy.eventRequired)
  }

  for (const event of events) {
    if ((OUTBOUND_WEBHOOK_EVENTS as readonly string[]).includes(event)) {
      continue
    }
    throw new Error(copy.unsupportedEvent(event, OUTBOUND_WEBHOOK_EVENTS.join(', ')))
  }

  return [...new Set(events)] as DaemonOutboundWebhookEvent[]
}

function formatWebhookSummary(
  webhook: DaemonOutboundWebhookSummary,
  copy: HooksCopy = HOOKS_COPY.en,
): string {
  const status = webhook.enabled ? chalk.green(copy.enabled) : chalk.gray(copy.disabled)
  const extras: string[] = []
  if (webhook.hasSecret) extras.push(copy.secret)
  if (webhook.headerKeys.length > 0) {
    extras.push(copy.headers(webhook.headerKeys.join(',')))
  }
  extras.push(copy.retry(webhook.retry.maxAttempts, webhook.retry.backoffMs))

  // Two-line layout so url + events stay readable without horizontal
  // overflow even when the id is long.
  const head = `  ${chalk.bold(webhook.id)}  ${status}  ${chalk.gray(webhook.events.join(', '))}`
  const tail = `      ${webhook.url}  ${chalk.gray(`[${extras.join(' ')}]`)}`
  return `${head}\n${tail}`
}

function formatWebhookDelivery(
  delivery: DaemonOutboundWebhookDelivery,
  copy: HooksCopy = HOOKS_COPY.en,
): string {
  const status =
    delivery.deliveryStatus === 'success' ? chalk.green(copy.success) : chalk.red(copy.error)
  const extras = [
    `event=${delivery.hookEvent}`,
    `attempts=${delivery.attemptCount}`,
    `ms=${delivery.durationMs}`,
  ]
  if (typeof delivery.statusCode === 'number') {
    extras.push(`http=${delivery.statusCode}`)
  }
  if (delivery.sessionId) {
    extras.push(`session=${delivery.sessionId}`)
  }
  if (delivery.replayedFromDeliveryId) {
    extras.push(`replayOf=${delivery.replayedFromDeliveryId}`)
  }

  const suffix = delivery.error ? ` ${chalk.red(delivery.error)}` : ''
  return `  ${delivery.webhookId.padEnd(32)} ${status.padEnd(14)} ${delivery.timestamp} [${extras.join(' ')}]${suffix}`
}

function formatWebhookDeadLetter(
  deadLetter: DaemonOutboundWebhookDeadLetter,
  copy: HooksCopy = HOOKS_COPY.en,
): string {
  const state =
    deadLetter.state === 'acknowledged'
      ? chalk.yellow(copy.acknowledged)
      : chalk.red(copy.deadLetter)
  const extras = [
    `root=${deadLetter.rootDeliveryId}`,
    `latest=${deadLetter.latestDeliveryId}`,
    `replays=${deadLetter.replayCount}`,
    `attempts=${deadLetter.attemptCount}`,
    `ms=${deadLetter.durationMs}`,
  ]
  if (typeof deadLetter.statusCode === 'number') {
    extras.push(`http=${deadLetter.statusCode}`)
  }
  if (deadLetter.sessionId) {
    extras.push(`session=${deadLetter.sessionId}`)
  }
  if (deadLetter.acknowledgedAt) {
    extras.push(`ackedAt=${deadLetter.acknowledgedAt}`)
  }
  if (deadLetter.acknowledgedByDevice) {
    extras.push(`ackedBy=${deadLetter.acknowledgedByDevice}`)
  }

  const suffix = deadLetter.error ? ` ${chalk.red(deadLetter.error)}` : ''
  return `  ${deadLetter.webhookId.padEnd(32)} ${state.padEnd(14)} ${deadLetter.lastAttemptAt} [${extras.join(' ')}]${suffix}${deadLetter.acknowledgmentNote ? ` ${copy.note(deadLetter.acknowledgmentNote)}` : ''}`
}

export async function hooksListCommand(options: HooksMutationOptions) {
  const copy = hooksCopy()
  const client = new DaemonClient(options.url)
  const data = await client.outboundWebhooks()
  output(data, (webhooks) => {
    if (!webhooks.length) {
      return copy.noWebhooks
    }
    return webhooks.map((webhook) => formatWebhookSummary(webhook, copy)).join('\n')
  })
}

export async function hooksDeliveriesCommand(
  id: string | undefined,
  options: HooksDeliveriesOptions,
) {
  const copy = hooksCopy()
  const client = new DaemonClient(options.url)
  const limit = options.limit ? Number.parseInt(options.limit, 10) : undefined
  const page = await client.outboundWebhookDeliveriesPage({
    id,
    status: options.status,
    ...(options.cursor ? { cursor: options.cursor } : {}),
    limit: Number.isFinite(limit) ? limit : undefined,
  })
  output(page, (result) => {
    if (!result.data.length) {
      return copy.noDeliveries
    }
    const lines = result.data.map((delivery) => formatWebhookDelivery(delivery, copy))
    if (result.meta.nextCursor) {
      lines.push(copy.nextCursor(result.meta.nextCursor))
    }
    return lines.join('\n')
  })
}

export async function hooksDeadLettersCommand(
  id: string | undefined,
  options: HooksDeadLettersOptions,
) {
  const copy = hooksCopy()
  const client = new DaemonClient(options.url)
  const limit = options.limit ? Number.parseInt(options.limit, 10) : undefined
  const page = await client.outboundWebhookDeadLettersPage({
    id,
    state: options.state,
    ...(options.cursor ? { cursor: options.cursor } : {}),
    limit: Number.isFinite(limit) ? limit : undefined,
  })
  output(page, (result) => {
    if (!result.data.length) {
      return copy.noDeadLetters
    }
    const lines = result.data.map((deadLetter) => formatWebhookDeadLetter(deadLetter, copy))
    if (result.meta.nextCursor) {
      lines.push(copy.nextCursor(result.meta.nextCursor))
    }
    return lines.join('\n')
  })
}

export async function hooksAckCommand(rootDeliveryId: string, options: HooksAckOptions) {
  const copy = hooksCopy()
  const client = new DaemonClient(options.url)
  const deadLetter = await client.acknowledgeOutboundWebhookDeadLetter(
    rootDeliveryId,
    options.note ? { note: options.note } : undefined,
  )
  output(
    {
      rootDeliveryId,
      deadLetter,
    },
    (result) =>
      copy.acknowledgedDeadLetter(
        result.rootDeliveryId,
        formatWebhookDeadLetter(result.deadLetter, copy),
      ),
  )
}

export async function hooksAddCommand(url: string, options: HooksAddOptions) {
  const copy = hooksCopy()
  const client = new DaemonClient(options.url)
  const retryAttempts = options.retryAttempts
    ? Number.parseInt(options.retryAttempts, 10)
    : undefined
  const retryBackoffMs = options.retryBackoffMs
    ? Number.parseInt(options.retryBackoffMs, 10)
    : undefined
  await client.upsertOutboundWebhook({
    enabled: options.disabled ? false : true,
    url,
    events: parseWebhookEvents(options.event, copy),
    secret: options.secret,
    headers: parseKeyValueEntries(options.header, '--header', copy),
    retry:
      Number.isFinite(retryAttempts) || Number.isFinite(retryBackoffMs)
        ? {
            ...(Number.isFinite(retryAttempts) ? { maxAttempts: retryAttempts } : {}),
            ...(Number.isFinite(retryBackoffMs) ? { backoffMs: retryBackoffMs } : {}),
          }
        : undefined,
  })

  const webhook = (await client.outboundWebhooks()).find((entry) => entry.url === url)
  output(
    {
      ok: true,
      url,
      action: 'upserted',
      webhook,
    },
    (result) => {
      if (!result.webhook) {
        return copy.webhookUpdated(result.url)
      }
      return copy.webhookUpdatedWith(result.url, formatWebhookSummary(result.webhook, copy))
    },
  )
}

export async function hooksReplayCommand(deliveryId: string, options: HooksReplayOptions) {
  const copy = hooksCopy()
  const client = new DaemonClient(options.url)
  const delivery = await client.replayOutboundWebhookDelivery(deliveryId, {
    force: options.force,
  })
  output(
    {
      deliveryId,
      delivery,
    },
    (result) =>
      copy.webhookReplayed(
        result.deliveryId,
        result.delivery.deliveryId,
        formatWebhookDelivery(result.delivery, copy),
      ),
  )
}

export async function hooksReplayFailedCommand(
  id: string | undefined,
  options: HooksReplayFailedOptions,
) {
  const copy = hooksCopy()
  const client = new DaemonClient(options.url)
  const limit = options.limit ? Number.parseInt(options.limit, 10) : undefined
  const deadLetters = await client.outboundWebhookDeadLetters({
    id,
    ...(options.cursor ? { cursor: options.cursor } : {}),
    limit: Number.isFinite(limit) ? limit : undefined,
  })

  if (deadLetters.length === 0) {
    output([], () => copy.noDeadLettersToReplay)
    return
  }

  const deliveries: DaemonOutboundWebhookDelivery[] = []
  for (const deadLetter of deadLetters) {
    deliveries.push(
      await client.replayOutboundWebhookDelivery(
        deadLetter.latestDeliveryId,
        options.force ? { force: true } : undefined,
      ),
    )
  }

  output(
    {
      deadLetters,
      deliveries,
    },
    (result) => {
      const lines = [copy.replayedDeadLetters(result.deliveries.length)]
      for (const delivery of result.deliveries) {
        lines.push(formatWebhookDelivery(delivery, copy))
      }
      return lines.join('\n')
    },
  )
}

export async function hooksRemoveCommand(id: string, options: HooksMutationOptions) {
  const copy = hooksCopy()
  const client = new DaemonClient(options.url)
  await client.deleteOutboundWebhook(id)
  output(
    {
      ok: true,
      id,
      action: 'removed',
    },
    (result) => copy.webhookRemoved(result.id),
  )
}

export async function hooksEnableCommand(id: string, options: HooksMutationOptions) {
  const copy = hooksCopy()
  const client = new DaemonClient(options.url)
  await client.setOutboundWebhookEnabled(id, true)
  const webhook = (await client.outboundWebhooks()).find((entry) => entry.id === id)
  output(
    {
      ok: true,
      id,
      action: 'enabled',
      webhook,
    },
    (result) => {
      if (!result.webhook) {
        return copy.webhookEnabled(result.id)
      }
      return copy.webhookEnabledWith(result.id, formatWebhookSummary(result.webhook, copy))
    },
  )
}

export async function hooksDisableCommand(id: string, options: HooksMutationOptions) {
  const copy = hooksCopy()
  const client = new DaemonClient(options.url)
  await client.setOutboundWebhookEnabled(id, false)
  const webhook = (await client.outboundWebhooks()).find((entry) => entry.id === id)
  output(
    {
      ok: true,
      id,
      action: 'disabled',
      webhook,
    },
    (result) => {
      if (!result.webhook) {
        return copy.webhookDisabled(result.id)
      }
      return copy.webhookDisabledWith(result.id, formatWebhookSummary(result.webhook, copy))
    },
  )
}

export const __testables = {
  formatWebhookDeadLetter,
  formatWebhookDelivery,
  formatWebhookSummary,
  parseKeyValueEntries,
  parseWebhookEvents,
}
