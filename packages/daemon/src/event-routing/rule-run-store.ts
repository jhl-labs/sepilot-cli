import { randomUUID } from 'node:crypto'
import type { SqliteDatabase } from '../db/sqlite.js'

export const RULE_RUN_MAX_ATTEMPTS = 3
export const RULE_RESULT_EXCERPT_MAX = 2000
const ERROR_MAX = 1000

export type RuleRunStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'throttled' | 'skipped'

export interface RuleRunRecord {
  id: string
  eventId: string
  ruleId: string
  status: RuleRunStatus
  attempt: number
  sessionId: string | null
  resultExcerpt: string | null
  error: string | null
  nextAttemptAt: number
  startedAt: number | null
  finishedAt: number | null
  createdAt: number
  updatedAt: number
}

export interface RuleRunStore {
  create(input: { eventId: string; ruleId: string }, now?: number): RuleRunRecord
  get(id: string): RuleRunRecord | null
  listByEvent(eventId: string): RuleRunRecord[]
  claimDue(now: number, limit: number, staleBefore: number, excludeIds?: readonly string[]): RuleRunRecord[]
  countStartedSince(ruleId: string, since: number, excludeId: string): number
  markSucceeded(id: string, result: string, now?: number): void
  markRetry(id: string, error: string, nextAttemptAt: number, now?: number): void
  markFailed(id: string, error: string, now?: number): void
  markThrottled(id: string, now?: number): void
  markSkipped(id: string, reason: string, now?: number): void
  requeue(id: string, now?: number): void
  prune(olderThan: number, maxRows: number): number
}

interface RuleRunRow {
  id: string
  event_id: string
  rule_id: string
  status: RuleRunStatus
  attempt: number
  session_id: string | null
  result_excerpt: string | null
  error: string | null
  next_attempt_at: number
  started_at: number | null
  finished_at: number | null
  created_at: number
  updated_at: number
}

const TERMINAL_STATUSES = "('succeeded', 'failed', 'throttled', 'skipped')"

const COLS = 'id, event_id, rule_id, status, attempt, session_id, result_excerpt, error, next_attempt_at, started_at, finished_at, created_at, updated_at'

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS event_rule_runs (
      id TEXT PRIMARY KEY,
      event_id TEXT NOT NULL,
      rule_id TEXT NOT NULL,
      status TEXT NOT NULL,
      attempt INTEGER NOT NULL DEFAULT 0,
      session_id TEXT,
      result_excerpt TEXT,
      error TEXT,
      next_attempt_at INTEGER NOT NULL,
      started_at INTEGER,
      finished_at INTEGER,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      UNIQUE (event_id, rule_id)
    )`,
  ).run()
  db.prepare('CREATE INDEX IF NOT EXISTS idx_event_rule_runs_due ON event_rule_runs(status, next_attempt_at)').run()
  db.prepare('CREATE INDEX IF NOT EXISTS idx_event_rule_runs_rule ON event_rule_runs(rule_id, started_at)').run()
}

function toRecord(row: RuleRunRow): RuleRunRecord {
  return {
    id: row.id,
    eventId: row.event_id,
    ruleId: row.rule_id,
    status: row.status,
    attempt: row.attempt,
    sessionId: row.session_id,
    resultExcerpt: row.result_excerpt,
    error: row.error,
    nextAttemptAt: row.next_attempt_at,
    startedAt: row.started_at,
    finishedAt: row.finished_at,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  }
}

export function createRuleRunStore(db: SqliteDatabase): RuleRunStore {
  ensureSchema(db)
  const get = (id: string): RuleRunRecord | null => {
    const row = db.prepare(`SELECT ${COLS} FROM event_rule_runs WHERE id = ?`).get(id) as RuleRunRow | undefined
    return row ? toRecord(row) : null
  }
  const finish = (id: string, status: RuleRunStatus, fields: { result?: string | null; error?: string | null }, now: number): void => {
    db.prepare(
      `UPDATE event_rule_runs SET status = ?, result_excerpt = COALESCE(?, result_excerpt), error = ?, finished_at = ?, updated_at = ? WHERE id = ?`,
    ).run(status, fields.result ?? null, fields.error ?? null, now, now, id)
  }

  return {
    create(input, now = Date.now()) {
      const existing = db.prepare(`SELECT ${COLS} FROM event_rule_runs WHERE event_id = ? AND rule_id = ?`)
        .get(input.eventId, input.ruleId) as RuleRunRow | undefined
      if (existing) return toRecord(existing)
      const id = randomUUID()
      db.prepare(
        `INSERT INTO event_rule_runs (${COLS}) VALUES (?, ?, ?, 'pending', 0, NULL, NULL, NULL, ?, NULL, NULL, ?, ?)`,
      ).run(id, input.eventId, input.ruleId, now, now, now)
      const record = get(id)
      if (!record) throw new Error(`created rule run disappeared: ${id}`)
      return record
    },
    get,
    listByEvent(eventId) {
      return (db.prepare(`SELECT ${COLS} FROM event_rule_runs WHERE event_id = ? ORDER BY created_at, rowid`).all(eventId) as RuleRunRow[]).map(toRecord)
    },
    claimDue(now, limit, staleBefore, excludeIds = []) {
      const tx = db.transaction(() => {
        const exclusion = excludeIds.length > 0 ? `AND id NOT IN (${excludeIds.map(() => '?').join(', ')})` : ''
        const rows = db.prepare(
          `SELECT ${COLS} FROM event_rule_runs
           WHERE ((status = 'pending' AND next_attempt_at <= ?)
              OR (status = 'running' AND started_at <= ?))
              ${exclusion}
           ORDER BY next_attempt_at, created_at, rowid LIMIT ?`,
        ).all(now, staleBefore, ...excludeIds, Math.max(1, Math.min(limit, 50))) as RuleRunRow[]
        const update = db.prepare(
          `UPDATE event_rule_runs SET status = 'running', attempt = attempt + 1, session_id = ?, started_at = ?, updated_at = ? WHERE id = ?`,
        )
        for (const row of rows) {
          const sessionId = randomUUID()
          update.run(sessionId, now, now, row.id)
          row.status = 'running'
          row.attempt += 1
          row.session_id = sessionId
          row.started_at = now
          row.updated_at = now
        }
        return rows.map(toRecord)
      })
      return tx()
    },
    countStartedSince(ruleId, since, excludeId) {
      const row = db.prepare(
        `SELECT COUNT(*) AS n FROM event_rule_runs
         WHERE rule_id = ? AND id != ? AND started_at >= ? AND status IN ('pending', 'running', 'succeeded', 'failed')`,
      ).get(ruleId, excludeId, since) as { n: number }
      return row.n
    },
    markSucceeded(id, result, now = Date.now()) {
      finish(id, 'succeeded', { result: result.slice(0, RULE_RESULT_EXCERPT_MAX), error: null }, now)
    },
    markRetry(id, error, nextAttemptAt, now = Date.now()) {
      db.prepare(`UPDATE event_rule_runs SET status = 'pending', error = ?, next_attempt_at = ?, updated_at = ? WHERE id = ?`)
        .run(error.slice(0, ERROR_MAX), nextAttemptAt, now, id)
    },
    markFailed(id, error, now = Date.now()) {
      finish(id, 'failed', { error: error.slice(0, ERROR_MAX) }, now)
    },
    markThrottled(id, now = Date.now()) {
      finish(id, 'throttled', { error: null }, now)
    },
    markSkipped(id, reason, now = Date.now()) {
      finish(id, 'skipped', { error: reason.slice(0, ERROR_MAX) }, now)
    },
    requeue(id, now = Date.now()) {
      db.prepare(
        `UPDATE event_rule_runs SET status = 'pending', attempt = MAX(attempt - 1, 0), session_id = NULL, started_at = NULL, next_attempt_at = ?, updated_at = ? WHERE id = ?`,
      ).run(now, now, id)
    },
    prune(olderThan, maxRows) {
      const tx = db.transaction(() => {
        const byAge = db.prepare(
          `DELETE FROM event_rule_runs WHERE updated_at < ? AND status IN ${TERMINAL_STATUSES}`,
        ).run(olderThan).changes
        const byCap = db.prepare(
          `DELETE FROM event_rule_runs WHERE status IN ${TERMINAL_STATUSES} AND id NOT IN (
             SELECT id FROM event_rule_runs WHERE status IN ${TERMINAL_STATUSES}
             ORDER BY updated_at DESC, rowid DESC LIMIT ?
           )`,
        ).run(Math.max(1, maxRows)).changes
        return byAge + byCap
      })
      return tx()
    },
  }
}
