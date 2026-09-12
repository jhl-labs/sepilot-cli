import { randomUUID } from 'node:crypto'
import type { SqliteDatabase } from '../db/sqlite.js'
import { createLogger } from '../logger.js'
import { openDomainDb } from '../storage/domain-db.js'

const log = createLogger('channel:ingress-queue')
const DEFAULT_RETRY_BACKOFF_MS = 5_000
const DEFAULT_MAX_BACKOFF_MS = 5 * 60_000
const DEFAULT_CLAIM_STALE_MS = 2 * 60_000

export type ChannelIngressStatus = 'pending' | 'processing' | 'done' | 'failed'

export type ChannelIngressKind =
  | 'slack.event'
  | 'discord.interaction'
  | 'line.webhook'
  | 'teams.activity'
  | 'whatsapp.webhook'
  | 'mattermost.webhook'

export interface ChannelIngressInput {
  id?: string
  idempotencyKey: string
  channelType: string
  kind: ChannelIngressKind
  payload: Record<string, unknown>
}

export interface ChannelIngressRecord extends ChannelIngressInput {
  id: string
  status: ChannelIngressStatus
  attempt: number
  nextAttemptAt: number
  lastError: string | null
  createdAt: number
  updatedAt: number
  doneAt: number | null
}

export interface ChannelIngressQueue {
  enqueue(input: ChannelIngressInput): ChannelIngressRecord
  get(id: string): ChannelIngressRecord | null
  getByIdempotencyKey(key: string): ChannelIngressRecord | null
  list(filter?: { status?: ChannelIngressStatus[] }): ChannelIngressRecord[]
  claimDue(now: number, limit: number): ChannelIngressRecord[]
  markDone(id: string, doneAt?: number): void
  markFailed(id: string, error: string, nextAttemptAt: number): void
  drainDue(process: (record: ChannelIngressRecord) => Promise<void>, limit?: number): Promise<number>
  /** False once the backing domain database has been closed (e.g. shutdown). */
  isOpen(): boolean
}

interface IngressRow {
  id: string
  idempotency_key: string
  channel_type: string
  kind: ChannelIngressKind
  payload_json: string
  status: ChannelIngressStatus
  attempt: number
  next_attempt_at: number
  last_error: string | null
  created_at: number
  updated_at: number
  done_at: number | null
}

const SELECT_COLS = [
  'id',
  'idempotency_key',
  'channel_type',
  'kind',
  'payload_json',
  'status',
  'attempt',
  'next_attempt_at',
  'last_error',
  'created_at',
  'updated_at',
  'done_at',
].join(', ')

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS channel_ingress_queue (
      id TEXT PRIMARY KEY,
      idempotency_key TEXT NOT NULL UNIQUE,
      channel_type TEXT NOT NULL,
      kind TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'pending',
      attempt INTEGER NOT NULL DEFAULT 0,
      next_attempt_at INTEGER NOT NULL,
      last_error TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      done_at INTEGER
    )`,
  ).run()
  db.prepare(
    `CREATE INDEX IF NOT EXISTS idx_channel_ingress_due
     ON channel_ingress_queue(status, next_attempt_at)`,
  ).run()
}

function parsePayload(value: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(value) as unknown
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : {}
  } catch {
    return {}
  }
}

function toRecord(row: IngressRow): ChannelIngressRecord {
  return {
    id: row.id,
    idempotencyKey: row.idempotency_key,
    channelType: row.channel_type,
    kind: row.kind,
    payload: parsePayload(row.payload_json),
    status: row.status,
    attempt: row.attempt,
    nextAttemptAt: row.next_attempt_at,
    lastError: row.last_error,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    doneAt: row.done_at,
  }
}

function nextBackoffMs(attempt: number): number {
  return Math.min(DEFAULT_MAX_BACKOFF_MS, DEFAULT_RETRY_BACKOFF_MS * (2 ** Math.max(0, attempt - 1)))
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

export function createChannelIngressQueue(): ChannelIngressQueue {
  const db = openDomainDb({ name: 'channel-ingress' })
  ensureSchema(db)

  const get = (id: string): ChannelIngressRecord | null => {
    const row = db.prepare(
      `SELECT ${SELECT_COLS} FROM channel_ingress_queue WHERE id = ?`,
    ).get(id) as IngressRow | undefined
    return row ? toRecord(row) : null
  }

  const getByKey = (key: string): ChannelIngressRecord | null => {
    const row = db.prepare(
      `SELECT ${SELECT_COLS} FROM channel_ingress_queue WHERE idempotency_key = ?`,
    ).get(key) as IngressRow | undefined
    return row ? toRecord(row) : null
  }

  return {
    isOpen() {
      return db.open
    },
    enqueue(input) {
      const existing = getByKey(input.idempotencyKey)
      if (existing) return existing

      const now = Date.now()
      const id = input.id ?? randomUUID()
      db.prepare(
        `INSERT INTO channel_ingress_queue (
           id, idempotency_key, channel_type, kind, payload_json, status, attempt,
           next_attempt_at, last_error, created_at, updated_at, done_at
         )
         VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, NULL, ?, ?, NULL)`,
      ).run(
        id,
        input.idempotencyKey,
        input.channelType,
        input.kind,
        JSON.stringify(input.payload),
        now,
        now,
        now,
      )
      const record = get(id)
      if (!record) throw new Error(`queued channel ingress disappeared: ${id}`)
      return record
    },
    get,
    getByIdempotencyKey: getByKey,
    list(filter) {
      const params: unknown[] = []
      const where = filter?.status?.length
        ? `WHERE status IN (${filter.status.map(() => '?').join(',')})`
        : ''
      if (filter?.status?.length) params.push(...filter.status)
      return (
        db.prepare(
          `SELECT ${SELECT_COLS} FROM channel_ingress_queue ${where} ORDER BY created_at`,
        ).all(...params) as IngressRow[]
      ).map(toRecord)
    },
    claimDue(now, limit) {
      const lim = Math.max(1, Math.min(limit, 100))
      const staleCutoff = now - DEFAULT_CLAIM_STALE_MS
      const tx = db.transaction(() => {
        const rows = db.prepare(
          `SELECT ${SELECT_COLS} FROM channel_ingress_queue
           WHERE (
             status IN ('pending', 'failed') AND next_attempt_at <= ?
           ) OR (
             status = 'processing' AND updated_at <= ?
           )
           ORDER BY next_attempt_at, created_at
           LIMIT ?`,
        ).all(now, staleCutoff, lim) as IngressRow[]
        const update = db.prepare(
          `UPDATE channel_ingress_queue
           SET status = 'processing', attempt = attempt + 1, updated_at = ?
           WHERE id = ?`,
        )
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
    markDone(id, doneAt) {
      const now = doneAt ?? Date.now()
      db.prepare(
        `UPDATE channel_ingress_queue
         SET status = 'done', done_at = ?, updated_at = ?, last_error = NULL
         WHERE id = ?`,
      ).run(now, now, id)
    },
    markFailed(id, error, nextAttemptAt) {
      db.prepare(
        `UPDATE channel_ingress_queue
         SET status = 'failed', last_error = ?, next_attempt_at = ?, updated_at = ?
         WHERE id = ?`,
      ).run(error, nextAttemptAt, Date.now(), id)
    },
    async drainDue(process, limit = 20) {
      const claimed = this.claimDue(Date.now(), limit)
      let done = 0
      for (const record of claimed) {
        try {
          await process(record)
          this.markDone(record.id)
          done++
        } catch (err) {
          const message = errorMessage(err)
          const nextAttemptAt = Date.now() + nextBackoffMs(record.attempt)
          this.markFailed(record.id, message, nextAttemptAt)
          log.warn('channel ingress processing failed', {
            ingressId: record.id,
            kind: record.kind,
            attempt: record.attempt,
            nextAttemptAt,
            error: message,
          })
        }
      }
      return done
    },
  }
}
