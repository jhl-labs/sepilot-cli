import { randomUUID } from 'node:crypto'
import type { SqliteDatabase } from '../db/sqlite.js'
import { openDomainDb } from '../storage/domain-db.js'
import type { RoutedEvent, RoutedEventInput } from './types.js'

export const EVENT_MAX_ATTEMPTS = 5
export const EVENT_CLAIM_STALE_MS = 120_000

export type EventLogStatus = 'received' | 'processing' | 'processed' | 'unmatched' | 'failed'

export interface EventLogRecord extends RoutedEvent {
  status: EventLogStatus
  matchedRules: string[]
  attempt: number
  nextAttemptAt: number | null
  lastError: string | null
  createdAt: number
  updatedAt: number
}

export interface EventStore {
  append(input: RoutedEventInput, now?: number): { record: EventLogRecord; duplicate: boolean }
  get(id: string): EventLogRecord | null
  list(filter?: { source?: string; status?: EventLogStatus[]; limit?: number }): EventLogRecord[]
  claimDue(now: number, limit: number): EventLogRecord[]
  markRouted(id: string, matchedRules: string[], now?: number): void
  markFailed(id: string, error: string, nextAttemptAt: number | null, now?: number): void
  prune(olderThan: number, maxRows: number): number
}

interface EventRow {
  id: string
  source: string
  type: string
  subject: string | null
  occurred_at: number
  dedupe_key: string
  payload_json: string
  status: EventLogStatus
  matched_rules_json: string
  attempt: number
  next_attempt_at: number | null
  last_error: string | null
  created_at: number
  updated_at: number
}

const COLS = 'id, source, type, subject, occurred_at, dedupe_key, payload_json, status, matched_rules_json, attempt, next_attempt_at, last_error, created_at, updated_at'

export function openEventRoutingDb(): SqliteDatabase {
  return openDomainDb({ name: 'event-routing' })
}

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS event_log (
      id TEXT PRIMARY KEY,
      source TEXT NOT NULL,
      type TEXT NOT NULL,
      subject TEXT,
      occurred_at INTEGER NOT NULL,
      dedupe_key TEXT NOT NULL UNIQUE,
      payload_json TEXT NOT NULL,
      status TEXT NOT NULL,
      matched_rules_json TEXT NOT NULL DEFAULT '[]',
      attempt INTEGER NOT NULL DEFAULT 0,
      next_attempt_at INTEGER,
      last_error TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    )`,
  ).run()
  db.prepare('CREATE INDEX IF NOT EXISTS idx_event_log_due ON event_log(status, next_attempt_at)').run()
  db.prepare('CREATE INDEX IF NOT EXISTS idx_event_log_created ON event_log(created_at)').run()
}

function toRecord(row: EventRow): EventLogRecord {
  return {
    id: row.id,
    source: row.source,
    type: row.type,
    subject: row.subject,
    occurredAt: row.occurred_at,
    dedupeKey: row.dedupe_key,
    payload: JSON.parse(row.payload_json) as Record<string, unknown>,
    status: row.status,
    matchedRules: JSON.parse(row.matched_rules_json) as string[],
    attempt: row.attempt,
    nextAttemptAt: row.next_attempt_at,
    lastError: row.last_error,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  }
}

export function createEventStore(db: SqliteDatabase): EventStore {
  ensureSchema(db)
  const get = (id: string): EventLogRecord | null => {
    const row = db.prepare(`SELECT ${COLS} FROM event_log WHERE id = ?`).get(id) as EventRow | undefined
    return row ? toRecord(row) : null
  }
  const getByDedupeKey = (key: string): EventLogRecord | null => {
    const row = db.prepare(`SELECT ${COLS} FROM event_log WHERE dedupe_key = ?`).get(key) as EventRow | undefined
    return row ? toRecord(row) : null
  }

  return {
    append(input, now = Date.now()) {
      const existing = getByDedupeKey(input.dedupeKey)
      if (existing) return { record: existing, duplicate: true }
      const id = input.id ?? randomUUID()
      db.prepare(
        `INSERT INTO event_log (${COLS})
         VALUES (?, ?, ?, ?, ?, ?, ?, 'received', '[]', 0, ?, NULL, ?, ?)`,
      ).run(id, input.source, input.type, input.subject, input.occurredAt, input.dedupeKey, JSON.stringify(input.payload), now, now, now)
      const record = get(id)
      if (!record) throw new Error(`appended event disappeared: ${id}`)
      return { record, duplicate: false }
    },
    get,
    list(filter = {}) {
      const clauses: string[] = []
      const params: unknown[] = []
      if (filter.source) {
        clauses.push('source = ?')
        params.push(filter.source)
      }
      if (filter.status?.length) {
        clauses.push(`status IN (${filter.status.map(() => '?').join(',')})`)
        params.push(...filter.status)
      }
      const where = clauses.length ? `WHERE ${clauses.join(' AND ')}` : ''
      const limit = Math.max(1, Math.min(filter.limit ?? 100, 500))
      return (db.prepare(`SELECT ${COLS} FROM event_log ${where} ORDER BY created_at DESC, rowid DESC LIMIT ?`)
        .all(...params, limit) as EventRow[]).map(toRecord)
    },
    claimDue(now, limit) {
      const tx = db.transaction(() => {
        const rows = db.prepare(
          `SELECT ${COLS} FROM event_log
           WHERE (status = 'received' AND next_attempt_at <= ?)
              OR (status = 'processing' AND updated_at <= ?)
           ORDER BY next_attempt_at, created_at LIMIT ?`,
        ).all(now, now - EVENT_CLAIM_STALE_MS, Math.max(1, Math.min(limit, 100))) as EventRow[]
        const update = db.prepare(`UPDATE event_log SET status = 'processing', attempt = attempt + 1, updated_at = ? WHERE id = ?`)
        for (const row of rows) {
          update.run(now, row.id)
          row.status = 'processing'
          row.attempt += 1
          row.updated_at = now
        }
        return rows.map(toRecord)
      })
      return tx()
    },
    markRouted(id, matchedRules, now = Date.now()) {
      db.prepare(
        `UPDATE event_log SET status = ?, matched_rules_json = ?, next_attempt_at = NULL, last_error = NULL, updated_at = ? WHERE id = ?`,
      ).run(matchedRules.length ? 'processed' : 'unmatched', JSON.stringify(matchedRules), now, id)
    },
    markFailed(id, error, nextAttemptAt, now = Date.now()) {
      db.prepare(`UPDATE event_log SET status = ?, last_error = ?, next_attempt_at = ?, updated_at = ? WHERE id = ?`)
        .run(nextAttemptAt === null ? 'failed' : 'received', error.slice(0, 1000), nextAttemptAt, now, id)
    },
    prune(olderThan, maxRows) {
      const tx = db.transaction(() => {
        const byAge = db.prepare(`DELETE FROM event_log WHERE created_at < ? AND status NOT IN ('received', 'processing')`).run(olderThan).changes
        const byCap = db.prepare(
          `DELETE FROM event_log WHERE status NOT IN ('received', 'processing') AND id NOT IN (
             SELECT id FROM event_log ORDER BY created_at DESC, rowid DESC LIMIT ?
           )`,
        ).run(Math.max(1, maxRows)).changes
        return byAge + byCap
      })
      return tx()
    },
  }
}
