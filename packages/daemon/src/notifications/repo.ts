import type { SqliteDatabase } from '../db/sqlite.js'
import { openDomainDb } from '../storage/domain-db.js'
import {
  normalizeNotificationAudience,
  notificationVisibleToSurface,
  type NotificationAudience,
} from './audience.js'
import {
  enqueueNotificationRelayDelivery,
  ensureNotificationRelayOutboxSchema,
} from './relay-outbox.js'
import {
  normalizeNotifyRelayErrorCode,
  type NotifyRelayDeliveryStatus,
  type NotifyRelayMessageStatus,
  type NotifyRelayStatusLookupOutcome,
  type NotifyRelayStatusLookupResult,
} from './push-notify-relay.js'

const NOTIFICATION_RELAY_PROVIDER_CHECK_DELAY_MS = 30_000
const NOTIFICATION_RELAY_PROVIDER_CHECK_STALE_MS = 5 * 60_000

const TERMINAL_RELAY_PROVIDER_STATUSES = new Set<NotifyRelayMessageStatus>([
  'delivered',
  'rejected',
  'denied',
  'dead_letter',
  'expired',
])

const FAILED_RELAY_PROVIDER_STATUSES = new Set<NotifyRelayMessageStatus>([
  'rejected',
  'denied',
  'dead_letter',
  'expired',
])

export function notificationRelayProviderStatusIsTerminal(
  status: NotifyRelayMessageStatus,
): boolean {
  return TERMINAL_RELAY_PROVIDER_STATUSES.has(status)
}

export interface NotificationRelayDeliveryReceipt {
  status: NotifyRelayDeliveryStatus
  messageId: string | null
  httpStatus: number | null
  errorCode?: string | null
  attemptedAt: number
  completedAt: number
}

export interface LatestNotificationRelayDelivery extends NotificationRelayDeliveryReceipt {
  notificationId: string
  topic: string | null
  providerDelivery: NotificationRelayProviderDeliveryReceipt | null
}

export interface NotificationRelayProviderDeliveryReceipt {
  lookupStatus: NotifyRelayStatusLookupOutcome
  status: NotifyRelayMessageStatus | null
  httpStatus: number | null
  errorCode?: string | null
  attempt: number
  checkedAt: number
  completedAt: number | null
}

export interface NotificationRelayProviderCheck {
  notificationId: string
  messageId: string
  attempt: number
}

export interface NotificationRelayProviderSummary {
  total: number
  pending: number
  checking: number
  retrying: number
  delivered: number
  failed: number
  unconfirmed: number
  nextCheckAt: number | null
  lastCheckedAt: number | null
}

export interface SchedulerNotificationCorrelation {
  kind: 'scheduler'
  jobId: string
  runId: string | null
}

export type NotificationCorrelation = SchedulerNotificationCorrelation

export interface NotificationItem {
  id: string
  title: string
  body: string
  url: string | null
  topic: string | null
  audience: NotificationAudience
  createdAt: number
  readAt: number | null
  correlation: NotificationCorrelation | null
  relayDelivery: NotificationRelayDeliveryReceipt | null
  relayProviderDelivery: NotificationRelayProviderDeliveryReceipt | null
}

export interface ChannelSetting {
  id: string
  enabled: boolean
}

export interface NotificationInventory {
  total: number
  unread: number
}

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL DEFAULT '',
    url TEXT, created_at INTEGER NOT NULL, read_at INTEGER
  )`,
  ).run()
  const cols = new Set(
    (db.prepare('PRAGMA table_info(notifications)').all() as Array<{ name: string }>)
      .map((column) => column.name),
  )
  if (!cols.has('topic')) {
    db.prepare('ALTER TABLE notifications ADD COLUMN topic TEXT').run()
  }
  if (!cols.has('audience_json')) {
    db.prepare('ALTER TABLE notifications ADD COLUMN audience_json TEXT').run()
  }
  if (!cols.has('correlation_json')) {
    db.prepare('ALTER TABLE notifications ADD COLUMN correlation_json TEXT').run()
  }
  const relayColumns = [
    ['relay_status', 'TEXT'],
    ['relay_message_id', 'TEXT'],
    ['relay_http_status', 'INTEGER'],
    ['relay_error_code', 'TEXT'],
    ['relay_attempted_at', 'INTEGER'],
    ['relay_completed_at', 'INTEGER'],
    ['relay_provider_status', 'TEXT'],
    ['relay_provider_lookup_status', 'TEXT'],
    ['relay_provider_http_status', 'INTEGER'],
    ['relay_provider_error_code', 'TEXT'],
    ['relay_provider_check_attempt', 'INTEGER NOT NULL DEFAULT 0'],
    ['relay_provider_checked_at', 'INTEGER'],
    ['relay_provider_completed_at', 'INTEGER'],
    ['relay_provider_next_check_at', 'INTEGER'],
    ['relay_provider_checking_at', 'INTEGER'],
  ] as const
  for (const [name, type] of relayColumns) {
    if (!cols.has(name)) {
      db.prepare(`ALTER TABLE notifications ADD COLUMN ${name} ${type}`).run()
    }
  }
  // Upgrade accepted receipts from older daemons into the durable status-check
  // lifecycle. Using the persisted acceptance timestamp makes this idempotent
  // and immediately due after an old daemon has been offline for the delay.
  db.prepare(
    `UPDATE notifications
     SET relay_provider_next_check_at = relay_completed_at + ?
     WHERE relay_status IN ('accepted', 'pending_review')
       AND relay_message_id IS NOT NULL
       AND relay_completed_at IS NOT NULL
       AND relay_provider_completed_at IS NULL
       AND relay_provider_next_check_at IS NULL
       AND relay_provider_checking_at IS NULL`,
  ).run(NOTIFICATION_RELAY_PROVIDER_CHECK_DELAY_MS)
  db.prepare(
    `CREATE TABLE IF NOT EXISTS notification_channels (id TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 1)`,
  ).run()
  ensureNotificationRelayOutboxSchema(db)
}

function parseAudience(raw: string | null | undefined): NotificationAudience {
  if (!raw) return null
  try {
    return normalizeNotificationAudience(JSON.parse(raw))
  } catch {
    return null
  }
}

function stringifyAudience(audience: NotificationAudience | undefined): string | null {
  const normalized = normalizeNotificationAudience(audience)
  return normalized == null ? null : JSON.stringify(normalized)
}

function parseCorrelation(raw: string | null | undefined): NotificationCorrelation | null {
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>
    if (
      parsed.kind === 'scheduler'
      && typeof parsed.jobId === 'string'
      && parsed.jobId.trim().length > 0
      && (parsed.runId === null || typeof parsed.runId === 'string')
    ) {
      return {
        kind: 'scheduler',
        jobId: parsed.jobId,
        runId: typeof parsed.runId === 'string' && parsed.runId.trim().length > 0
          ? parsed.runId
          : null,
      }
    }
  } catch {
    // Fail closed: malformed local metadata is not operational evidence.
  }
  return null
}

function stringifyCorrelation(
  correlation: NotificationCorrelation | null | undefined,
): string | null {
  if (!correlation) return null
  return JSON.stringify(correlation)
}

export interface NotificationsRepo {
  list(options?: {
    surface?: string | null
    unreadOnly?: boolean
    limit?: number
  }): NotificationItem[]
  get(id: string, options?: { surface?: string | null }): NotificationItem | null
  inventory(options?: { surface?: string | null }): NotificationInventory
  upsert(input: {
    id?: string
    title: string
    body: string
    url: string | null
    topic?: string | null
    audience?: string[] | null
    correlation?: NotificationCorrelation | null
    createdAt?: number
    /** Queue external Relay delivery atomically with the notification row. */
    relayDeliveryPending?: boolean
  }): NotificationItem
  markRead(id: string, options?: { surface?: string | null }): boolean
  recordRelayDelivery(
    id: string,
    receipt: NotificationRelayDeliveryReceipt,
  ): NotificationItem | null
  claimRelayProviderChecks(now: number, limit?: number): NotificationRelayProviderCheck[]
  recordRelayProviderStatus(
    id: string,
    result: NotifyRelayStatusLookupResult,
    options: { terminal: boolean; nextCheckAt: number | null },
  ): NotificationItem | null
  relayProviderSummary(): NotificationRelayProviderSummary
  latestRelayDelivery(options?: { surface?: string | null }): LatestNotificationRelayDelivery | null
  prune(options?: { maxAgeDays?: number; keepLatest?: number; now?: number }): { deleted: number }
  markAllRead(options?: { surface?: string | null }): number
  channelSettings(): ChannelSetting[]
  setChannel(id: string, enabled: boolean): void
}

const DEFAULT_NOTIFICATION_MAX_AGE_DAYS = 30
const DEFAULT_NOTIFICATION_KEEP_LATEST = 500
const NOTIFICATION_LIST_LIMIT = 200
const DAY_MS = 24 * 60 * 60 * 1000

type NotificationRow = {
  id: string
  title: string
  body: string
  url: string | null
  topic: string | null
  audience_json: string | null
  correlation_json: string | null
  created_at: number
  read_at: number | null
  relay_status: NotifyRelayDeliveryStatus | null
  relay_message_id: string | null
  relay_http_status: number | null
  relay_error_code: string | null
  relay_attempted_at: number | null
  relay_completed_at: number | null
  relay_provider_status: NotifyRelayMessageStatus | null
  relay_provider_lookup_status: NotifyRelayStatusLookupOutcome | null
  relay_provider_http_status: number | null
  relay_provider_error_code: string | null
  relay_provider_check_attempt: number
  relay_provider_checked_at: number | null
  relay_provider_completed_at: number | null
  relay_provider_next_check_at: number | null
  relay_provider_checking_at: number | null
}

type NotificationVisibilityRow = Pick<
  NotificationRow,
  'id' | 'audience_json' | 'read_at'
>

const NOTIFICATION_SELECT = `SELECT id, title, body, url, topic, audience_json, correlation_json,
  created_at, read_at, relay_status, relay_message_id, relay_http_status, relay_error_code,
  relay_attempted_at, relay_completed_at, relay_provider_status,
  relay_provider_lookup_status, relay_provider_http_status, relay_provider_error_code,
  relay_provider_check_attempt, relay_provider_checked_at,
  relay_provider_completed_at, relay_provider_next_check_at,
  relay_provider_checking_at FROM notifications`

function relayDeliveryFromRow(
  row: NotificationRow,
): NotificationRelayDeliveryReceipt | null {
  if (
    !row.relay_status
    || row.relay_attempted_at == null
    || row.relay_completed_at == null
  ) {
    return null
  }
  return {
    status: row.relay_status,
    messageId: row.relay_message_id,
    httpStatus: row.relay_http_status,
    errorCode: row.relay_error_code,
    attemptedAt: row.relay_attempted_at,
    completedAt: row.relay_completed_at,
  }
}

function relayProviderDeliveryFromRow(
  row: NotificationRow,
): NotificationRelayProviderDeliveryReceipt | null {
  if (!row.relay_provider_lookup_status || row.relay_provider_checked_at == null) {
    return null
  }
  return {
    lookupStatus: row.relay_provider_lookup_status,
    status: row.relay_provider_status,
    httpStatus: row.relay_provider_http_status,
    errorCode: row.relay_provider_error_code,
    attempt: row.relay_provider_check_attempt,
    checkedAt: row.relay_provider_checked_at,
    completedAt: row.relay_provider_completed_at,
  }
}

function notificationFromRow(row: NotificationRow): NotificationItem {
  return {
    id: row.id,
    title: row.title,
    body: row.body,
    url: row.url,
    topic: row.topic,
    audience: parseAudience(row.audience_json),
    createdAt: row.created_at,
    readAt: row.read_at,
    correlation: parseCorrelation(row.correlation_json),
    relayDelivery: relayDeliveryFromRow(row),
    relayProviderDelivery: relayProviderDeliveryFromRow(row),
  }
}

function visibleNotificationRows(
  db: SqliteDatabase,
  options?: { surface?: string | null; unreadOnly?: boolean },
): NotificationVisibilityRow[] {
  const where = options?.unreadOnly ? ' WHERE read_at IS NULL' : ''
  const rows = db
    .prepare(`SELECT id, audience_json, read_at FROM notifications${where}`)
    .all() as NotificationVisibilityRow[]
  return rows.filter((row) => (
    notificationVisibleToSurface(parseAudience(row.audience_json), options?.surface)
  ))
}

function listVisibleNotifications(
  db: SqliteDatabase,
  surface: string | null | undefined,
  options?: { unreadOnly?: boolean; limit?: number },
): NotificationItem[] {
  const limit = Math.min(
    NOTIFICATION_LIST_LIMIT,
    Math.max(1, Math.floor(options?.limit ?? NOTIFICATION_LIST_LIMIT)),
  )
  const where = options?.unreadOnly ? ' WHERE read_at IS NULL' : ''
  const statement = db.prepare(
    `${NOTIFICATION_SELECT}${where} ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?`,
  )
  const visible: NotificationItem[] = []
  let offset = 0
  while (visible.length < limit) {
    const rows = statement.all(NOTIFICATION_LIST_LIMIT, offset) as NotificationRow[]
    if (rows.length === 0) break
    offset += rows.length
    for (const row of rows) {
      const item = notificationFromRow(row)
      if (!notificationVisibleToSurface(item.audience, surface)) continue
      visible.push(item)
      if (visible.length === limit) break
    }
    if (rows.length < NOTIFICATION_LIST_LIMIT) break
  }
  return visible
}

export function createNotificationsRepo(): NotificationsRepo {
  const db = openDomainDb({ name: 'notifications' })
  ensureSchema(db)
  return {
    list(options) {
      return listVisibleNotifications(db, options?.surface, options)
    },
    get(id, options) {
      const row = db.prepare(`${NOTIFICATION_SELECT} WHERE id=?`).get(id) as
        | NotificationRow
        | undefined
      if (!row) return null
      const item = notificationFromRow(row)
      return notificationVisibleToSurface(item.audience, options?.surface)
        ? item
        : null
    },
    inventory(options) {
      const visible = visibleNotificationRows(db, options)
      return {
        total: visible.length,
        unread: visible.filter((row) => row.read_at == null).length,
      }
    },
    upsert(input) {
      const id = input.id || crypto.randomUUID()
      const now = input.createdAt ?? Date.now()
      const audience = normalizeNotificationAudience(input.audience)
      return db.transaction(() => {
        db.prepare(
          `INSERT INTO notifications (id, title, body, url, topic, audience_json, correlation_json, created_at, read_at)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
          ON CONFLICT(id) DO UPDATE SET
            title=excluded.title,
            body=excluded.body,
            url=excluded.url,
            topic=excluded.topic,
            audience_json=excluded.audience_json,
            correlation_json=excluded.correlation_json`,
        ).run(
          id,
          input.title,
          input.body,
          input.url,
          input.topic ?? null,
          stringifyAudience(audience),
          stringifyCorrelation(input.correlation),
          now,
        )
        if (input.relayDeliveryPending) {
          enqueueNotificationRelayDelivery(db, id, now)
        }
        const row = db.prepare(`${NOTIFICATION_SELECT} WHERE id=?`).get(id) as
          | NotificationRow
          | undefined
        if (!row) throw new Error(`Notification was not persisted: ${id}`)
        return notificationFromRow(row)
      })()
    },
    markRead(id, options) {
      const row = db.prepare('SELECT audience_json FROM notifications WHERE id=?').get(id) as
        | Pick<NotificationRow, 'audience_json'>
        | undefined
      if (
        !row
        || !notificationVisibleToSurface(parseAudience(row.audience_json), options?.surface)
      ) {
        return false
      }
      db.prepare('UPDATE notifications SET read_at=? WHERE id=?').run(
        Date.now(),
        id,
      )
      return true
    },
    recordRelayDelivery(id, receipt) {
      const shouldTrackProvider = (
        receipt.status === 'accepted' || receipt.status === 'pending_review'
      ) && receipt.messageId !== null
      const firstProviderCheckAt = receipt.completedAt
        + NOTIFICATION_RELAY_PROVIDER_CHECK_DELAY_MS
      const result = db.prepare(
        `UPDATE notifications SET relay_status=?, relay_message_id=?,
          relay_http_status=?, relay_error_code=?, relay_attempted_at=?, relay_completed_at=?,
          relay_provider_next_check_at=CASE
            WHEN ? = 1 AND relay_provider_completed_at IS NULL
              THEN COALESCE(relay_provider_next_check_at, ?)
            ELSE relay_provider_next_check_at
          END
          WHERE id=?`,
      ).run(
        receipt.status,
        receipt.messageId,
        receipt.httpStatus,
        normalizeNotifyRelayErrorCode(receipt.errorCode),
        receipt.attemptedAt,
        receipt.completedAt,
        shouldTrackProvider ? 1 : 0,
        firstProviderCheckAt,
        id,
      )
      if (Number(result.changes ?? 0) === 0) return null
      const row = db.prepare(`${NOTIFICATION_SELECT} WHERE id=?`).get(id) as
        | NotificationRow
        | undefined
      return row ? notificationFromRow(row) : null
    },
    claimRelayProviderChecks(now, limit = 20) {
      const boundedLimit = Math.max(1, Math.min(100, Math.floor(limit)))
      const staleCutoff = now - NOTIFICATION_RELAY_PROVIDER_CHECK_STALE_MS
      return db.transaction(() => {
        const rows = db.prepare(
          `SELECT id, relay_message_id, relay_provider_check_attempt
           FROM notifications
           WHERE relay_message_id IS NOT NULL
             AND relay_provider_completed_at IS NULL
             AND (
               relay_provider_next_check_at <= ?
               OR (
                 relay_provider_checking_at IS NOT NULL
                 AND relay_provider_checking_at <= ?
               )
             )
           ORDER BY COALESCE(relay_provider_next_check_at, relay_provider_checking_at), id
           LIMIT ?`,
        ).all(now, staleCutoff, boundedLimit) as Array<{
          id: string
          relay_message_id: string
          relay_provider_check_attempt: number
        }>
        const claim = db.prepare(
          `UPDATE notifications
           SET relay_provider_check_attempt = relay_provider_check_attempt + 1,
             relay_provider_checking_at = ?, relay_provider_next_check_at = NULL
           WHERE id = ? AND relay_provider_completed_at IS NULL`,
        )
        const claimed: NotificationRelayProviderCheck[] = []
        for (const row of rows) {
          const result = claim.run(now, row.id)
          if (Number(result.changes ?? 0) === 0) continue
          claimed.push({
            notificationId: row.id,
            messageId: row.relay_message_id,
            attempt: row.relay_provider_check_attempt + 1,
          })
        }
        return claimed
      })()
    },
    recordRelayProviderStatus(id, result, options) {
      const update = db.prepare(
        `UPDATE notifications SET
          relay_provider_status=?, relay_provider_lookup_status=?,
          relay_provider_http_status=?, relay_provider_error_code=?, relay_provider_checked_at=?,
          relay_provider_completed_at=?, relay_provider_next_check_at=?,
          relay_provider_checking_at=NULL
         WHERE id=? AND relay_message_id=? AND relay_provider_completed_at IS NULL`,
      ).run(
        result.messageStatus,
        result.outcome,
        result.httpStatus,
        normalizeNotifyRelayErrorCode(result.errorCode),
        result.completedAt,
        options.terminal ? result.completedAt : null,
        options.terminal ? null : options.nextCheckAt,
        id,
        result.messageId,
      )
      if (Number(update.changes ?? 0) === 0) return null
      const row = db.prepare(`${NOTIFICATION_SELECT} WHERE id=?`).get(id) as
        | NotificationRow
        | undefined
      return row ? notificationFromRow(row) : null
    },
    relayProviderSummary() {
      const failedStatuses = [...FAILED_RELAY_PROVIDER_STATUSES]
      const failedPlaceholders = failedStatuses.map(() => '?').join(',')
      const row = db.prepare(
        `SELECT
          COUNT(*) AS total,
          COALESCE(SUM(CASE WHEN relay_provider_completed_at IS NULL
            AND relay_provider_checking_at IS NULL
            AND (relay_provider_lookup_status IS NULL
              OR relay_provider_lookup_status = 'observed') THEN 1 ELSE 0 END), 0) AS pending,
          COALESCE(SUM(CASE WHEN relay_provider_completed_at IS NULL
            AND relay_provider_checking_at IS NOT NULL THEN 1 ELSE 0 END), 0) AS checking,
          COALESCE(SUM(CASE WHEN relay_provider_completed_at IS NULL
            AND relay_provider_checking_at IS NULL
            AND relay_provider_lookup_status IS NOT NULL
            AND relay_provider_lookup_status != 'observed' THEN 1 ELSE 0 END), 0) AS retrying,
          COALESCE(SUM(CASE WHEN relay_provider_completed_at IS NOT NULL
            AND relay_provider_status = 'delivered' THEN 1 ELSE 0 END), 0) AS delivered,
          COALESCE(SUM(CASE WHEN relay_provider_completed_at IS NOT NULL
            AND relay_provider_status IN (${failedPlaceholders}) THEN 1 ELSE 0 END), 0) AS failed,
          COALESCE(SUM(CASE WHEN relay_provider_completed_at IS NOT NULL
            AND relay_provider_status IS NULL THEN 1 ELSE 0 END), 0) AS unconfirmed,
          MIN(CASE WHEN relay_provider_completed_at IS NULL
            THEN relay_provider_next_check_at END) AS next_check_at,
          MAX(relay_provider_checked_at) AS last_checked_at
         FROM notifications
         WHERE relay_message_id IS NOT NULL
           AND relay_status IN ('accepted', 'pending_review')`,
      ).get(...failedStatuses) as {
        total: number
        pending: number
        checking: number
        retrying: number
        delivered: number
        failed: number
        unconfirmed: number
        next_check_at: number | null
        last_checked_at: number | null
      }
      return {
        total: row.total,
        pending: row.pending,
        checking: row.checking,
        retrying: row.retrying,
        delivered: row.delivered,
        failed: row.failed,
        unconfirmed: row.unconfirmed,
        nextCheckAt: row.next_check_at,
        lastCheckedAt: row.last_checked_at,
      }
    },
    latestRelayDelivery(options) {
      const statement = db.prepare(
        `${NOTIFICATION_SELECT} WHERE relay_completed_at IS NOT NULL
         ORDER BY relay_completed_at DESC, id DESC LIMIT ? OFFSET ?`,
      )
      let offset = 0
      while (true) {
        const rows = statement.all(NOTIFICATION_LIST_LIMIT, offset) as NotificationRow[]
        if (rows.length === 0) return null
        offset += rows.length
        const row = options === undefined
          ? rows[0]
          : rows.find((candidate) => notificationVisibleToSurface(
              parseAudience(candidate.audience_json),
              options.surface,
            ))
        if (row) {
          const receipt = relayDeliveryFromRow(row)
          return receipt
            ? {
                notificationId: row.id,
                topic: row.topic,
                ...receipt,
                providerDelivery: relayProviderDeliveryFromRow(row),
              }
            : null
        }
        if (rows.length < NOTIFICATION_LIST_LIMIT) return null
      }
    },
    prune(options) {
      // Two-tier sweep so notifications cannot grow unbounded (there was no
      // delete path before): (1) drop read notifications older than the age
      // ceiling, (2) cap the table to the newest-K by deleting the oldest
      // beyond it. Unread notifications survive the age sweep so a pending
      // action is never silently dropped.
      const now = options?.now ?? Date.now()
      const maxAgeDays = options?.maxAgeDays ?? DEFAULT_NOTIFICATION_MAX_AGE_DAYS
      const keepLatest = options?.keepLatest ?? DEFAULT_NOTIFICATION_KEEP_LATEST
      const cutoff = now - maxAgeDays * DAY_MS
      return db.transaction(() => {
        const agedOut = db.prepare(
          'DELETE FROM notifications WHERE read_at IS NOT NULL AND created_at < ?',
        ).run(cutoff)
        const overflow = db.prepare(
          `DELETE FROM notifications WHERE id NOT IN (
            SELECT id FROM notifications ORDER BY created_at DESC LIMIT ?
          )`,
        ).run(keepLatest)
        // Relay outbox rows intentionally store only notification identity and
        // retry state. Remove terminal or pending orphans in the same sweep so
        // delivery metadata remains bounded by the notification retention cap.
        db.prepare(
          `DELETE FROM notification_relay_outbox
           WHERE notification_id NOT IN (SELECT id FROM notifications)`,
        ).run()
        return { deleted: agedOut.changes + overflow.changes }
      })()
    },
    markAllRead(options) {
      // A presentation list is intentionally capped at 200, but "all" is a
      // state transition over every retained notification visible to the
      // requesting surface. Never reuse the bounded list as mutation scope.
      const unread = visibleNotificationRows(db, { ...options, unreadOnly: true })
      if (unread.length === 0) return 0
      const timestamp = Date.now()
      const update = db.prepare('UPDATE notifications SET read_at=? WHERE id=? AND read_at IS NULL')
      const tx = db.transaction((ids: string[]) => {
        let marked = 0
        for (const id of ids) {
          const result = update.run(timestamp, id)
          marked += Number(result.changes ?? 0)
        }
        return marked
      })
      return tx(unread.map((item) => item.id)) as number
    },
    channelSettings() {
      return (
        db
          .prepare('SELECT id, enabled FROM notification_channels')
          .all() as { id: string; enabled: number }[]
      ).map((r) => ({ id: r.id, enabled: !!r.enabled }))
    },
    setChannel(id, enabled) {
      db.prepare(
        `INSERT INTO notification_channels (id, enabled) VALUES (?, ?)
        ON CONFLICT(id) DO UPDATE SET enabled=excluded.enabled`,
      ).run(id, enabled ? 1 : 0)
    },
  }
}
