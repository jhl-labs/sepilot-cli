import chalk from 'chalk'
import {
  daemonNotifyRelayEvidence,
  type DaemonNotificationItem,
  type DaemonNotifyRelayEvidencePresentation,
  type DaemonNotifyRelayProviderDeliveryReceipt,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { ensureDaemon } from '../client/ensure-daemon.js'
import { parsePositiveIntegerCliOption } from '../chat-options.js'
import { output, outputError } from '../output/formatter.js'
import { friendlyErrorMessage } from '../utils/error-message.js'
import { detectCliLocale } from '../utils/locale.js'

const DEFAULT_NOTIFICATION_LIMIT = 20
const MAX_NOTIFICATION_LIMIT = 200

const COPY = {
  en: {
    empty: 'No matching notifications.',
    header: (shown: number, total: number) => `Notifications (${shown}/${total} shown)`,
    headerUnknown: (shown: number) => `Notifications (${shown} shown; exact total unavailable)`,
    read: (id: string) => `Notification marked read: ${id}`,
    readAll: (count: number) => `Marked ${count} notification(s) read.`,
    notFound: (id: string) => `Notification not found: ${id}`,
    failed: (message: string) => `Notification command failed: ${message}`,
  },
  ko: {
    empty: '조건에 맞는 알림이 없습니다.',
    header: (shown: number, total: number) => `알림 (${total}개 중 ${shown}개 표시)`,
    headerUnknown: (shown: number) => `알림 (${shown}개 표시, 정확한 전체 개수 확인 불가)`,
    read: (id: string) => `알림을 읽음 처리했습니다: ${id}`,
    readAll: (count: number) => `알림 ${count}개를 읽음 처리했습니다.`,
    notFound: (id: string) => `알림을 찾을 수 없습니다: ${id}`,
    failed: (message: string) => `알림 명령 실패: ${message}`,
  },
} as const

interface NotificationSummary {
  id: string
  title: string
  topic: string | null
  createdAt: number
  read: boolean
  relayAcceptance: {
    status: string
    messageId: string | null
    httpStatus: number | null
    errorCode: string | null
  } | null
  relayProviderDelivery: DaemonNotifyRelayProviderDeliveryReceipt | null
  relayEvidence: DaemonNotifyRelayEvidencePresentation | null
}

function notificationSummary(item: DaemonNotificationItem): NotificationSummary {
  const relayEvidence = daemonNotifyRelayEvidence(item)
  return {
    id: item.id,
    title: item.title,
    topic: item.topic ?? null,
    createdAt: item.createdAt,
    read: item.readAt != null,
    relayAcceptance: item.relayDelivery
      ? {
          status: item.relayDelivery.status,
          messageId: item.relayDelivery.messageId ?? null,
          httpStatus: item.relayDelivery.httpStatus,
          errorCode: item.relayDelivery.errorCode ?? null,
        }
      : null,
    relayProviderDelivery: item.relayProviderDelivery ?? null,
    relayEvidence,
  }
}

function parseNotificationLimit(value?: string | number): number {
  const parsed = parsePositiveIntegerCliOption(value, '--limit') ?? DEFAULT_NOTIFICATION_LIMIT
  if (parsed > MAX_NOTIFICATION_LIMIT) {
    throw new Error(`--limit must be at most ${MAX_NOTIFICATION_LIMIT} (got: ${parsed})`)
  }
  return parsed
}

function formatNotificationSummaries(input: {
  items: NotificationSummary[]
  matching: number | null
}): string {
  const copy = COPY[detectCliLocale()] ?? COPY.en
  if (input.items.length === 0) return copy.empty
  return [
    input.matching == null
      ? copy.headerUnknown(input.items.length)
      : copy.header(input.items.length, input.matching),
    ...input.items.flatMap((item) => {
      const marker = item.read ? '○' : '●'
      const metadata = [
        item.id,
        new Date(item.createdAt).toISOString(),
        item.topic ? `topic=${item.topic}` : null,
        item.relayAcceptance
          ? `relay=${item.relayAcceptance.status}${item.relayAcceptance.errorCode ? `(${item.relayAcceptance.errorCode})` : ''}`
          : null,
        item.relayEvidence ? `delivery=${item.relayEvidence.state}` : null,
      ].filter(Boolean).join(' · ')
      return [`${marker} ${item.title}`, chalk.gray(`  ${metadata}`)]
    }),
  ].join('\n')
}

function formatNotification(item: DaemonNotificationItem): string {
  const relayEvidence = daemonNotifyRelayEvidence(item)
  const lines = [
    `${item.readAt == null ? '●' : '○'} ${item.title}`,
    `id: ${item.id}`,
    `created: ${new Date(item.createdAt).toISOString()}`,
    `read: ${item.readAt == null ? 'no' : new Date(item.readAt).toISOString()}`,
  ]
  if (item.topic) lines.push(`topic: ${item.topic}`)
  if (item.url) lines.push(`url: ${item.url}`)
  if (item.relayDelivery) {
    lines.push(`relay acceptance: ${item.relayDelivery.status}`)
    if (item.relayDelivery.httpStatus != null) {
      lines.push(`relay HTTP status: ${item.relayDelivery.httpStatus}`)
    }
    if (item.relayDelivery.errorCode) {
      lines.push(`relay rejection code: ${item.relayDelivery.errorCode}`)
    }
    if (item.relayDelivery.messageId) {
      lines.push(`relay message: ${item.relayDelivery.messageId}`)
    }
  }
  if (relayEvidence) lines.push(`relay delivery: ${relayEvidence.label}`)
  if (item.relayProviderDelivery) {
    lines.push(`relay provider lookup: ${item.relayProviderDelivery.lookupStatus}`)
    if (item.relayProviderDelivery.errorCode) {
      lines.push(`relay provider rejection code: ${item.relayProviderDelivery.errorCode}`)
    }
    lines.push(`relay provider attempt: ${item.relayProviderDelivery.attempt}`)
    lines.push(
      `relay provider checked: ${new Date(item.relayProviderDelivery.checkedAt).toISOString()}`,
    )
  }
  if (item.body) lines.push('', item.body)
  return lines.join('\n')
}

async function withNotificationClient<T>(
  url: string | undefined,
  operation: (client: DaemonClient) => Promise<T>,
): Promise<T> {
  const client = new DaemonClient(url)
  await ensureDaemon(client, { url, quiet: true })
  return operation(client)
}

function failNotificationCommand(error: unknown): never {
  const copy = COPY[detectCliLocale()] ?? COPY.en
  outputError(
    { ok: false, error: 'assistant-notification-command-failed' },
    () => chalk.red(copy.failed(friendlyErrorMessage(error))),
  )
  process.exit(1)
}

export async function assistantNotificationsCommand(options: {
  url?: string
  unread?: boolean
  limit?: string | number
}): Promise<void> {
  try {
    const limit = parseNotificationLimit(options.limit)
    const { items, inventory } = await withNotificationClient(
      options.url,
      async (client) => {
        const [items, inventory] = await Promise.all([
          client.notifications({ unread: options.unread === true, limit }),
          client.notificationInventory().catch(() => null),
        ])
        return { items, inventory }
      },
    )
    const visibleItems = inventory
      ? items
      : items
          .filter((item) => !options.unread || item.readAt == null)
          .slice(0, limit)
    const matching = inventory
      ? options.unread ? inventory.unread : inventory.total
      : null
    const result = {
      items: visibleItems.map(notificationSummary),
      matching,
      matchingExact: inventory !== null,
      unreadOnly: options.unread === true,
      limit,
    }
    output(result, formatNotificationSummaries)
  } catch (error) {
    failNotificationCommand(error)
  }
}

export async function assistantNotificationShowCommand(
  id: string,
  options: { url?: string },
): Promise<void> {
  try {
    const item = await withNotificationClient(options.url, (client) => client.notification(id))
    output(item, formatNotification)
  } catch (error) {
    failNotificationCommand(error)
  }
}

export async function assistantNotificationReadCommand(
  id: string,
  options: { url?: string },
): Promise<void> {
  try {
    await withNotificationClient(options.url, (client) => client.markNotificationRead(id))
    const result = { ok: true, id, read: true }
    const copy = COPY[detectCliLocale()] ?? COPY.en
    output(result, () => copy.read(id))
  } catch (error) {
    failNotificationCommand(error)
  }
}

export async function assistantNotificationsReadAllCommand(options: {
  url?: string
}): Promise<void> {
  try {
    const marked = await withNotificationClient(
      options.url,
      (client) => client.markAllNotificationsRead(),
    )
    const result = { ok: true, marked }
    const copy = COPY[detectCliLocale()] ?? COPY.en
    output(result, () => copy.readAll(marked))
  } catch (error) {
    failNotificationCommand(error)
  }
}

export const __testables = {
  notificationSummary,
  parseNotificationLimit,
}
