import type { SqliteDatabase } from '../db/sqlite.js'
import { openDomainDb } from '../storage/domain-db.js'

const DEFAULT_CLAIM_STALE_MS = 5 * 60_000

export type NotificationRelayOutboxStatus =
  | 'pending'
  | 'delivering'
  | 'failed'
  | 'delivered'
  | 'dead'

export interface NotificationRelayOutboxRecord {
  notificationId: string
  status: NotificationRelayOutboxStatus
  attempt: number
  nextAttemptAt: number
  lastError: string | null
  createdAt: number
  updatedAt: number
  deliveredAt: number | null
}

export interface NotificationRelayOutboxSummary {
  total: number
  pending: number
  delivering: number
  failed: number
  delivered: number
  dead: number
  /** Earliest due time across pending and failed rows. */
  nextAttemptAt: number | null
  /** Earliest retry time across failed rows only. */
  nextRetryAt: number | null
}

export interface NotificationRelayOutbox {
  enqueue(notificationId: string, now?: number): NotificationRelayOutboxRecord
  get(notificationId: string): NotificationRelayOutboxRecord | null
  list(filter?: { status?: NotificationRelayOutboxStatus[] }): NotificationRelayOutboxRecord[]
  summary(): NotificationRelayOutboxSummary
  claimDue(now: number, limit?: number): NotificationRelayOutboxRecord[]
  markDelivered(notificationId: string, deliveredAt?: number): void
  markFailed(notificationId: string, error: string, nextAttemptAt: number): void
  markDead(notificationId: string, error: string): void
}

type RelayOutboxRow = {
  notification_id: string
  status: NotificationRelayOutboxStatus
  attempt: number
  next_attempt_at: number
  last_error: string | null
  created_at: number
  updated_at: number
  delivered_at: number | null
}

type RelayOutboxSummaryRow = {
  total: number
  pending: number
  delivering: number
  failed: number
  delivered: number
  dead: number
  next_attempt_at: number | null
  next_retry_at: number | null
}

const SELECT_COLUMNS = [
  'notification_id',
  'status',
  'attempt',
  'next_attempt_at',
  'last_error',
  'created_at',
  'updated_at',
  'delivered_at',
].join(', ')

function toRecord(row: RelayOutboxRow): NotificationRelayOutboxRecord {
  return {
    notificationId: row.notification_id,
    status: row.status,
    attempt: row.attempt,
    nextAttemptAt: row.next_attempt_at,
    lastError: row.last_error,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    deliveredAt: row.delivered_at,
  }
}

export function ensureNotificationRelayOutboxSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS notification_relay_outbox (
      notification_id TEXT PRIMARY KEY,
      status TEXT NOT NULL DEFAULT 'pending',
      attempt INTEGER NOT NULL DEFAULT 0,
      next_attempt_at INTEGER NOT NULL,
      last_error TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      delivered_at INTEGER
    )`,
  ).run()
  db.prepare(
    `CREATE INDEX IF NOT EXISTS idx_notification_relay_due
     ON notification_relay_outbox(status, next_attempt_at)`,
  ).run()
}

export function enqueueNotificationRelayDelivery(
  db: SqliteDatabase,
  notificationId: string,
  now: number,
): void {
  db.prepare(
    `INSERT INTO notification_relay_outbox (
      notification_id, status, attempt, next_attempt_at, last_error,
      created_at, updated_at, delivered_at
    ) VALUES (?, 'pending', 0, ?, NULL, ?, ?, NULL)
    ON CONFLICT(notification_id) DO NOTHING`,
  ).run(notificationId, now, now, now)
}

export function createNotificationRelayOutbox(): NotificationRelayOutbox {
  const db = openDomainDb({ name: 'notifications' })
  ensureNotificationRelayOutboxSchema(db)

  const get = (notificationId: string): NotificationRelayOutboxRecord | null => {
    const row = db.prepare(
      `SELECT ${SELECT_COLUMNS} FROM notification_relay_outbox WHERE notification_id = ?`,
    ).get(notificationId) as RelayOutboxRow | undefined
    return row ? toRecord(row) : null
  }

  return {
    enqueue(notificationId, now = Date.now()) {
      enqueueNotificationRelayDelivery(db, notificationId, now)
      const record = get(notificationId)
      if (!record) throw new Error(`Relay delivery was not queued: ${notificationId}`)
      return record
    },
    get,
    list(filter) {
      const statuses = filter?.status ?? []
      const where = statuses.length > 0
        ? `WHERE status IN (${statuses.map(() => '?').join(',')})`
        : ''
      return (
        db.prepare(
          `SELECT ${SELECT_COLUMNS} FROM notification_relay_outbox ${where}
           ORDER BY created_at, notification_id`,
        ).all(...statuses) as RelayOutboxRow[]
      ).map(toRecord)
    },
    summary() {
      const row = db.prepare(
        `SELECT
          COUNT(*) AS total,
          COALESCE(SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END), 0) AS pending,
          COALESCE(SUM(CASE WHEN status = 'delivering' THEN 1 ELSE 0 END), 0) AS delivering,
          COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0) AS failed,
          COALESCE(SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END), 0) AS delivered,
          COALESCE(SUM(CASE WHEN status = 'dead' THEN 1 ELSE 0 END), 0) AS dead,
          MIN(CASE WHEN status IN ('pending', 'failed') THEN next_attempt_at END) AS next_attempt_at,
          MIN(CASE WHEN status = 'failed' THEN next_attempt_at END) AS next_retry_at
        FROM notification_relay_outbox`,
      ).get() as RelayOutboxSummaryRow
      return {
        total: row.total,
        pending: row.pending,
        delivering: row.delivering,
        failed: row.failed,
        delivered: row.delivered,
        dead: row.dead,
        nextAttemptAt: row.next_attempt_at,
        nextRetryAt: row.next_retry_at,
      }
    },
    claimDue(now, limit = 20) {
      const boundedLimit = Math.max(1, Math.min(100, Math.floor(limit)))
      const staleCutoff = now - DEFAULT_CLAIM_STALE_MS
      return db.transaction(() => {
        const rows = db.prepare(
          `SELECT ${SELECT_COLUMNS} FROM notification_relay_outbox
           WHERE (
             status IN ('pending', 'failed') AND next_attempt_at <= ?
           ) OR (
             status = 'delivering' AND updated_at <= ?
           )
           ORDER BY next_attempt_at, created_at
           LIMIT ?`,
        ).all(now, staleCutoff, boundedLimit) as RelayOutboxRow[]
        const claim = db.prepare(
          `UPDATE notification_relay_outbox
           SET status = 'delivering', attempt = attempt + 1, updated_at = ?
           WHERE notification_id = ?`,
        )
        for (const row of rows) {
          claim.run(now, row.notification_id)
          row.status = 'delivering'
          row.attempt += 1
          row.updated_at = now
        }
        return rows.map(toRecord)
      })()
    },
    markDelivered(notificationId, deliveredAt = Date.now()) {
      db.prepare(
        `UPDATE notification_relay_outbox
         SET status = 'delivered', last_error = NULL, delivered_at = ?, updated_at = ?
         WHERE notification_id = ?`,
      ).run(deliveredAt, deliveredAt, notificationId)
    },
    markFailed(notificationId, error, nextAttemptAt) {
      db.prepare(
        `UPDATE notification_relay_outbox
         SET status = 'failed', last_error = ?, next_attempt_at = ?, updated_at = ?
         WHERE notification_id = ?`,
      ).run(error.slice(0, 240), nextAttemptAt, Date.now(), notificationId)
    },
    markDead(notificationId, error) {
      db.prepare(
        `UPDATE notification_relay_outbox
         SET status = 'dead', last_error = ?, updated_at = ?
         WHERE notification_id = ?`,
      ).run(error.slice(0, 240), Date.now(), notificationId)
    },
  }
}
