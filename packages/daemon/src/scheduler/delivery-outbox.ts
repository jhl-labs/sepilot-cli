import { randomUUID } from 'node:crypto'
import type { ChannelMessage, IChannel } from '@sepilotd/core'
import type { SqliteDatabase } from '../db/sqlite.js'
import { createLogger } from '../logger.js'
import { openDomainDb } from '../storage/domain-db.js'
import { safeSchedulerDeliveryError } from './delivery-error.js'

const log = createLogger('scheduler:delivery-outbox')
const DEFAULT_RETRY_BACKOFF_MS = 30_000
const DEFAULT_MAX_BACKOFF_MS = 60 * 60_000
const DEFAULT_CLAIM_STALE_MS = 5 * 60_000

export type SchedulerDeliveryStatus = 'pending' | 'delivering' | 'delivered' | 'failed'

export interface SchedulerDeliveryInput {
  id?: string
  jobId: string
  runId: string | null
  idempotencyKey: string
  channelType: string
  channelId: string
  channelTarget: string
  replyToMessageId: string | null
  text: string
  format?: NonNullable<ChannelMessage['format']>
  commitJobMetadata?: Record<string, unknown> | null
}

export interface SchedulerDeliveryRecord {
  id: string
  jobId: string
  runId: string | null
  idempotencyKey: string
  channelType: string
  channelId: string
  channelTarget: string
  replyToMessageId: string | null
  text: string
  format: NonNullable<ChannelMessage['format']>
  status: SchedulerDeliveryStatus
  attempt: number
  nextAttemptAt: number
  lastError: string | null
  createdAt: number
  updatedAt: number
  deliveredAt: number | null
}

export interface SchedulerDeliveryOutboxSummary {
  total: number
  pending: number
  delivering: number
  failed: number
  delivered: number
  /** Earliest due time across pending and failed rows. */
  nextAttemptAt: number | null
  /** Earliest retry time across failed rows only. */
  nextRetryAt: number | null
}

export interface SchedulerDeliveryOutbox {
  enqueue(input: SchedulerDeliveryInput): SchedulerDeliveryRecord
  get(id: string): SchedulerDeliveryRecord | null
  getByIdempotencyKey(key: string): SchedulerDeliveryRecord | null
  list(filter?: { status?: SchedulerDeliveryStatus[] }): SchedulerDeliveryRecord[]
  listRecent(limit?: number): SchedulerDeliveryRecord[]
  summary(): SchedulerDeliveryOutboxSummary
  claimDue(now: number, limit: number): SchedulerDeliveryRecord[]
  markDelivered(id: string, deliveredAt?: number): void
  markFailed(id: string, error: string, nextAttemptAt: number): void
  drainDue(deliver: (record: SchedulerDeliveryRecord) => Promise<void>, limit?: number): Promise<number>
}

export interface SchedulerDeliveryChannelLookup {
  get(channelType: string): IChannel | undefined
}

export interface SchedulerDeliveryWorker {
  start(): void
  stop(): void
  drainOnce(): Promise<number>
}

interface DeliveryRow {
  id: string
  job_id: string
  run_id: string | null
  idempotency_key: string
  channel_type: string
  channel_id: string
  channel_target: string
  reply_to_message_id: string | null
  text: string
  format: NonNullable<ChannelMessage['format']>
  status: SchedulerDeliveryStatus
  attempt: number
  next_attempt_at: number
  last_error: string | null
  created_at: number
  updated_at: number
  delivered_at: number | null
}

interface DeliverySummaryRow {
  total: number
  pending: number
  delivering: number
  failed: number
  delivered: number
  next_attempt_at: number | null
  next_retry_at: number | null
}

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS scheduler_delivery_outbox (
      id TEXT PRIMARY KEY,
      job_id TEXT NOT NULL,
      run_id TEXT,
      idempotency_key TEXT NOT NULL UNIQUE,
      channel_type TEXT NOT NULL,
      channel_id TEXT NOT NULL,
      channel_target TEXT NOT NULL,
      reply_to_message_id TEXT,
      text TEXT NOT NULL,
      format TEXT NOT NULL DEFAULT 'markdown',
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
    `CREATE INDEX IF NOT EXISTS idx_scheduler_delivery_due
     ON scheduler_delivery_outbox(status, next_attempt_at)`,
  ).run()
  db.prepare(
    `CREATE INDEX IF NOT EXISTS idx_scheduler_delivery_job
     ON scheduler_delivery_outbox(job_id, created_at DESC)`,
  ).run()
}

const toDeliveryRecord = (row: DeliveryRow): SchedulerDeliveryRecord => ({
  id: row.id,
  jobId: row.job_id,
  runId: row.run_id,
  idempotencyKey: row.idempotency_key,
  channelType: row.channel_type,
  channelId: row.channel_id,
  channelTarget: row.channel_target,
  replyToMessageId: row.reply_to_message_id,
  text: row.text,
  format: row.format,
  status: row.status,
  attempt: row.attempt,
  nextAttemptAt: row.next_attempt_at,
  lastError: row.last_error,
  createdAt: row.created_at,
  updatedAt: row.updated_at,
  deliveredAt: row.delivered_at,
})

const SELECT_COLS = [
  'id',
  'job_id',
  'run_id',
  'idempotency_key',
  'channel_type',
  'channel_id',
  'channel_target',
  'reply_to_message_id',
  'text',
  'format',
  'status',
  'attempt',
  'next_attempt_at',
  'last_error',
  'created_at',
  'updated_at',
  'delivered_at',
].join(', ')

function stringifyMetadata(value: Record<string, unknown> | null | undefined): string | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  return JSON.stringify(value)
}

function nextBackoffMs(attempt: number): number {
  const exp = Math.max(0, attempt - 1)
  return Math.min(DEFAULT_MAX_BACKOFF_MS, DEFAULT_RETRY_BACKOFF_MS * (2 ** exp))
}

export function createSchedulerDeliveryOutbox(): SchedulerDeliveryOutbox {
  const db = openDomainDb({ name: 'scheduler' })
  ensureSchema(db)

  const getRow = (id: string): SchedulerDeliveryRecord | null => {
    const row = db.prepare(
      `SELECT ${SELECT_COLS} FROM scheduler_delivery_outbox WHERE id = ?`,
    ).get(id) as DeliveryRow | undefined
    return row ? toDeliveryRecord(row) : null
  }

  const getByKey = (key: string): SchedulerDeliveryRecord | null => {
    const row = db.prepare(
      `SELECT ${SELECT_COLS} FROM scheduler_delivery_outbox WHERE idempotency_key = ?`,
    ).get(key) as DeliveryRow | undefined
    return row ? toDeliveryRecord(row) : null
  }

  return {
    enqueue(input) {
      const existing = getByKey(input.idempotencyKey)
      if (existing) return existing

      const now = Date.now()
      const id = input.id ?? randomUUID()
      const tx = db.transaction(() => {
        db.prepare(
          `INSERT INTO scheduler_delivery_outbox (
             id, job_id, run_id, idempotency_key, channel_type, channel_id, channel_target,
             reply_to_message_id, text, format, status, attempt, next_attempt_at,
             last_error, created_at, updated_at, delivered_at
           )
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, NULL, ?, ?, NULL)`,
        ).run(
          id,
          input.jobId,
          input.runId,
          input.idempotencyKey,
          input.channelType,
          input.channelId,
          input.channelTarget,
          input.replyToMessageId,
          input.text,
          input.format ?? 'markdown',
          now,
          now,
          now,
        )
        if (input.commitJobMetadata !== undefined) {
          db.prepare(
            `UPDATE scheduler_jobs SET metadata_json = ?, updated_at = ? WHERE id = ?`,
          ).run(stringifyMetadata(input.commitJobMetadata), now, input.jobId)
        }
      })
      tx()
      const record = getRow(id)
      if (!record) throw new Error(`queued scheduler delivery disappeared: ${id}`)
      return record
    },
    get: getRow,
    getByIdempotencyKey: getByKey,
    list(filter) {
      const params: unknown[] = []
      const where = filter?.status?.length
        ? `WHERE status IN (${filter.status.map(() => '?').join(',')})`
        : ''
      if (filter?.status?.length) params.push(...filter.status)
      return (
        db.prepare(
          `SELECT ${SELECT_COLS} FROM scheduler_delivery_outbox ${where} ORDER BY created_at`,
        ).all(...params) as DeliveryRow[]
      ).map(toDeliveryRecord)
    },
    listRecent(limit) {
      const lim = Math.max(1, Math.min(limit ?? 50, 200))
      return (
        db.prepare(
          `SELECT ${SELECT_COLS} FROM scheduler_delivery_outbox
           ORDER BY created_at DESC LIMIT ?`,
        ).all(lim) as DeliveryRow[]
      ).map(toDeliveryRecord)
    },
    summary() {
      const row = db.prepare(
        `SELECT
          COUNT(*) AS total,
          COALESCE(SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END), 0) AS pending,
          COALESCE(SUM(CASE WHEN status = 'delivering' THEN 1 ELSE 0 END), 0) AS delivering,
          COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0) AS failed,
          COALESCE(SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END), 0) AS delivered,
          MIN(CASE WHEN status IN ('pending', 'failed') THEN next_attempt_at END) AS next_attempt_at,
          MIN(CASE WHEN status = 'failed' THEN next_attempt_at END) AS next_retry_at
        FROM scheduler_delivery_outbox`,
      ).get() as DeliverySummaryRow
      return {
        total: row.total,
        pending: row.pending,
        delivering: row.delivering,
        failed: row.failed,
        delivered: row.delivered,
        nextAttemptAt: row.next_attempt_at,
        nextRetryAt: row.next_retry_at,
      }
    },
    claimDue(now, limit) {
      const lim = Math.max(1, Math.min(limit, 100))
      const staleCutoff = now - DEFAULT_CLAIM_STALE_MS
      const tx = db.transaction(() => {
        const rows = db.prepare(
          `SELECT ${SELECT_COLS} FROM scheduler_delivery_outbox
           WHERE (
             status IN ('pending', 'failed') AND next_attempt_at <= ?
           ) OR (
             status = 'delivering' AND updated_at <= ?
           )
           ORDER BY next_attempt_at, created_at
           LIMIT ?`,
        ).all(now, staleCutoff, lim) as DeliveryRow[]
        const update = db.prepare(
          `UPDATE scheduler_delivery_outbox
           SET status = 'delivering', attempt = attempt + 1, updated_at = ?
           WHERE id = ?`,
        )
        for (const row of rows) {
          update.run(now, row.id)
          row.status = 'delivering'
          row.attempt += 1
          row.updated_at = now
        }
        return rows.map(toDeliveryRecord)
      })
      return tx()
    },
    markDelivered(id, deliveredAt) {
      const now = deliveredAt ?? Date.now()
      db.prepare(
        `UPDATE scheduler_delivery_outbox
         SET status = 'delivered', delivered_at = ?, updated_at = ?, last_error = NULL
         WHERE id = ?`,
      ).run(now, now, id)
    },
    markFailed(id, error, nextAttemptAt) {
      db.prepare(
        `UPDATE scheduler_delivery_outbox
         SET status = 'failed', last_error = ?, next_attempt_at = ?, updated_at = ?
         WHERE id = ?`,
      ).run(error, nextAttemptAt, Date.now(), id)
    },
    async drainDue(deliver, limit = 20) {
      const claimed = this.claimDue(Date.now(), limit)
      let delivered = 0
      for (const record of claimed) {
        try {
          await deliver(record)
          this.markDelivered(record.id)
          delivered++
        } catch (err) {
          const message = safeSchedulerDeliveryError(err)
          const nextAttemptAt = Date.now() + nextBackoffMs(record.attempt)
          this.markFailed(record.id, message, nextAttemptAt)
          log.warn('scheduled delivery retry failed', {
            deliveryId: record.id,
            jobId: record.jobId,
            attempt: record.attempt,
            nextAttemptAt,
            error: message,
          })
        }
      }
      return delivered
    },
  }
}

export function createSchedulerDeliveryWorker(
  outbox: SchedulerDeliveryOutbox,
  channels: SchedulerDeliveryChannelLookup,
  opts: { intervalMs?: number; batchSize?: number } = {},
): SchedulerDeliveryWorker {
  const intervalMs = opts.intervalMs ?? 30_000
  const batchSize = opts.batchSize ?? 20
  let timer: NodeJS.Timeout | null = null

  const deliver = async (record: SchedulerDeliveryRecord): Promise<void> => {
    const channel = channels.get(record.channelType)
    if (!channel) throw new Error(`channel ${record.channelType} unavailable`)
    await channel.sendMessage(
      { id: record.channelId, type: 'channel' },
      {
        text: record.text,
        format: record.format,
        ...(record.replyToMessageId ? { replyTo: record.replyToMessageId } : {}),
      },
    )
  }

  let activeDrain: Promise<number> | null = null
  const drainOnce = (): Promise<number> => {
    // A slow channel must not be reclaimed as stale by the next timer tick.
    if (activeDrain) return activeDrain
    activeDrain = Promise.resolve().then(() => outbox.drainDue(deliver, batchSize))
      .finally(() => { activeDrain = null })
    return activeDrain
  }
  const tick = () => {
    void drainOnce().catch(error => {
      log.warn('scheduled delivery worker failed', { error: safeSchedulerDeliveryError(error) })
    })
  }

  return {
    start() {
      if (timer) return
      tick()
      timer = setInterval(tick, intervalMs)
      timer.unref?.()
    },
    stop() {
      if (timer) clearInterval(timer)
      timer = null
    },
    drainOnce,
  }
}
