import { randomUUID } from 'node:crypto'
import type { SqliteDatabase } from '../db/sqlite.js'
import type { EventSummary } from './types.js'

export const DELIVERY_MAX_ATTEMPTS = 8
const BASE_BACKOFF_MS = 30_000
const MAX_BACKOFF_MS = 60 * 60_000
const CLAIM_STALE_MS = 5 * 60_000

export function nextDeliveryBackoffMs(attempt: number): number {
  return Math.min(MAX_BACKOFF_MS, BASE_BACKOFF_MS * 2 ** Math.max(0, attempt - 1))
}

export type DeliveryStatus = 'pending' | 'delivering' | 'delivered' | 'dead'

export interface DeliveryInput {
  eventId: string
  ruleId: string
  destinationId: string
  title: string
  text: string
  event: EventSummary
  idempotencyKey?: string
  replayedFrom?: string
}

export interface DeliveryRecord {
  id: string
  eventId: string
  ruleId: string
  destinationId: string
  idempotencyKey: string
  title: string
  text: string
  event: EventSummary
  status: DeliveryStatus
  attempt: number
  nextAttemptAt: number
  lastError: string | null
  replayedFrom: string | null
  createdAt: number
  updatedAt: number
  deliveredAt: number | null
  ackedAt: number | null
}

export interface EventDeliveryOutbox {
  enqueue(input: DeliveryInput, now?: number): DeliveryRecord
  enqueueDetailed(input: DeliveryInput, now?: number): { record: DeliveryRecord; duplicate: boolean }
  get(id: string): DeliveryRecord | null
  list(filter?: { status?: DeliveryStatus[]; eventId?: string; limit?: number }): DeliveryRecord[]
  claimDue(now: number, limit: number): DeliveryRecord[]
  markDelivered(id: string, now?: number): void
  markRetry(id: string, error: string, nextAttemptAt: number, now?: number): void
  markDead(id: string, error: string, now?: number): void
  ack(id: string, now?: number): boolean
  replay(id: string, now?: number): DeliveryRecord | null
  prune(olderThan: number, deadOlderThan: number): number
}

interface DeliveryRow {
  id: string
  event_id: string
  rule_id: string
  destination_id: string
  idempotency_key: string
  title: string
  text: string
  event_json: string
  status: DeliveryStatus
  attempt: number
  next_attempt_at: number
  last_error: string | null
  replayed_from: string | null
  created_at: number
  updated_at: number
  delivered_at: number | null
  acked_at: number | null
}

const COLS = 'id, event_id, rule_id, destination_id, idempotency_key, title, text, event_json, status, attempt, next_attempt_at, last_error, replayed_from, created_at, updated_at, delivered_at, acked_at'

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS event_delivery_outbox (
      id TEXT PRIMARY KEY,
      event_id TEXT NOT NULL,
      rule_id TEXT NOT NULL,
      destination_id TEXT NOT NULL,
      idempotency_key TEXT NOT NULL UNIQUE,
      title TEXT NOT NULL,
      text TEXT NOT NULL,
      event_json TEXT NOT NULL,
      status TEXT NOT NULL,
      attempt INTEGER NOT NULL DEFAULT 0,
      next_attempt_at INTEGER NOT NULL,
      last_error TEXT,
      replayed_from TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      delivered_at INTEGER,
      acked_at INTEGER
    )`,
  ).run()
  db.prepare('CREATE INDEX IF NOT EXISTS idx_event_delivery_due ON event_delivery_outbox(status, next_attempt_at)').run()
  db.prepare('CREATE INDEX IF NOT EXISTS idx_event_delivery_event ON event_delivery_outbox(event_id)').run()
}

function toRecord(row: DeliveryRow): DeliveryRecord {
  return {
    id: row.id,
    eventId: row.event_id,
    ruleId: row.rule_id,
    destinationId: row.destination_id,
    idempotencyKey: row.idempotency_key,
    title: row.title,
    text: row.text,
    event: JSON.parse(row.event_json) as EventSummary,
    status: row.status,
    attempt: row.attempt,
    nextAttemptAt: row.next_attempt_at,
    lastError: row.last_error,
    replayedFrom: row.replayed_from,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    deliveredAt: row.delivered_at,
    ackedAt: row.acked_at,
  }
}

export function createDeliveryOutbox(db: SqliteDatabase): EventDeliveryOutbox {
  ensureSchema(db)
  const get = (id: string): DeliveryRecord | null => {
    const row = db.prepare(`SELECT ${COLS} FROM event_delivery_outbox WHERE id = ?`).get(id) as DeliveryRow | undefined
    return row ? toRecord(row) : null
  }
  const getByKey = (key: string): DeliveryRecord | null => {
    const row = db.prepare(`SELECT ${COLS} FROM event_delivery_outbox WHERE idempotency_key = ?`).get(key) as DeliveryRow | undefined
    return row ? toRecord(row) : null
  }
  const enqueueDetailed = (input: DeliveryInput, now = Date.now()): { record: DeliveryRecord; duplicate: boolean } => {
    const key = input.idempotencyKey ?? `${input.eventId}:${input.ruleId}:${input.destinationId}`
    const existing = getByKey(key)
    if (existing) return { record: existing, duplicate: true }
    const id = randomUUID()
    db.prepare(
      `INSERT INTO event_delivery_outbox (${COLS})
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, NULL, ?, ?, ?, NULL, NULL)`,
    ).run(id, input.eventId, input.ruleId, input.destinationId, key, input.title, input.text, JSON.stringify(input.event), now, input.replayedFrom ?? null, now, now)
    const record = get(id)
    if (!record) throw new Error(`enqueued delivery disappeared: ${id}`)
    return { record, duplicate: false }
  }
  const enqueue = (input: DeliveryInput, now = Date.now()): DeliveryRecord => enqueueDetailed(input, now).record

  return {
    enqueue,
    enqueueDetailed,
    get,
    list(filter = {}) {
      const clauses: string[] = []
      const params: unknown[] = []
      if (filter.status?.length) {
        clauses.push(`status IN (${filter.status.map(() => '?').join(',')})`)
        params.push(...filter.status)
      }
      if (filter.eventId) {
        clauses.push('event_id = ?')
        params.push(filter.eventId)
      }
      const where = clauses.length ? `WHERE ${clauses.join(' AND ')}` : ''
      const limit = Math.max(1, Math.min(filter.limit ?? 100, 500))
      return (db.prepare(`SELECT ${COLS} FROM event_delivery_outbox ${where} ORDER BY created_at DESC, rowid DESC LIMIT ?`)
        .all(...params, limit) as DeliveryRow[]).map(toRecord)
    },
    claimDue(now, limit) {
      const tx = db.transaction(() => {
        const rows = db.prepare(
          `SELECT ${COLS} FROM event_delivery_outbox
           WHERE (status = 'pending' AND next_attempt_at <= ?)
              OR (status = 'delivering' AND updated_at <= ?)
           ORDER BY next_attempt_at, created_at LIMIT ?`,
        ).all(now, now - CLAIM_STALE_MS, Math.max(1, Math.min(limit, 100))) as DeliveryRow[]
        const update = db.prepare(`UPDATE event_delivery_outbox SET status = 'delivering', attempt = attempt + 1, updated_at = ? WHERE id = ?`)
        for (const row of rows) {
          update.run(now, row.id)
          row.status = 'delivering'
          row.attempt += 1
          row.updated_at = now
        }
        return rows.map(toRecord)
      })
      return tx()
    },
    markDelivered(id, now = Date.now()) {
      db.prepare(`UPDATE event_delivery_outbox SET status = 'delivered', delivered_at = ?, last_error = NULL, updated_at = ? WHERE id = ?`).run(now, now, id)
    },
    markRetry(id, error, nextAttemptAt, now = Date.now()) {
      db.prepare(`UPDATE event_delivery_outbox SET status = 'pending', last_error = ?, next_attempt_at = ?, updated_at = ? WHERE id = ?`)
        .run(error.slice(0, 1000), nextAttemptAt, now, id)
    },
    markDead(id, error, now = Date.now()) {
      db.prepare(`UPDATE event_delivery_outbox SET status = 'dead', last_error = ?, updated_at = ? WHERE id = ?`).run(error.slice(0, 1000), now, id)
    },
    ack(id, now = Date.now()) {
      return db.prepare(`UPDATE event_delivery_outbox SET acked_at = ?, updated_at = ? WHERE id = ? AND status = 'dead'`).run(now, now, id).changes > 0
    },
    replay(id, now = Date.now()) {
      const original = get(id)
      if (!original) return null
      return enqueue({
        eventId: original.eventId,
        ruleId: original.ruleId,
        destinationId: original.destinationId,
        title: original.title,
        text: original.text,
        event: original.event,
        idempotencyKey: `${original.idempotencyKey}:replay:${randomUUID()}`,
        replayedFrom: original.id,
      }, now)
    },
    prune(olderThan, deadOlderThan) {
      return db.prepare(
        `DELETE FROM event_delivery_outbox WHERE
          (updated_at < ? AND (status = 'delivered' OR (status = 'dead' AND acked_at IS NOT NULL)))
          OR (updated_at < ? AND status = 'dead' AND acked_at IS NULL)`,
      ).run(olderThan, deadOlderThan).changes
    },
  }
}
