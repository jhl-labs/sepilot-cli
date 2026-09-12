import { randomUUID } from 'node:crypto'
import type { SqliteDatabase } from '../db/sqlite.js'
import { openDomainDb } from '../storage/domain-db.js'
import type {
  ItemStatus,
  Job,
  JobItem,
  JobKind,
  JobStatus,
  PartitionedQueueStatus,
} from './types.js'

interface JobRow {
  id: string
  kind: JobKind
  status: JobStatus
  total: number
  succeeded: number
  failed: number
  canceled: number
  concurrency: number
  created_at: number
  started_at: number | null
  finished_at: number | null
  error: string | null
}

interface ItemRow {
  job_id: string
  idx: number
  status: ItemStatus
  request_json: string
  result_json: string | null
  error: string | null
  session_id: string | null
  attempts: number
  max_attempts: number
  created_at: number
  started_at: number | null
  finished_at: number | null
}

interface ItemWithJobRow extends ItemRow {
  job_created_at: number
}

const toJob = (r: JobRow): Job => ({
  id: r.id,
  kind: r.kind,
  status: r.status,
  total: r.total,
  succeeded: r.succeeded,
  failed: r.failed,
  canceled: r.canceled,
  concurrency: r.concurrency,
  createdAt: r.created_at,
  startedAt: r.started_at,
  finishedAt: r.finished_at,
  error: r.error,
})

const toItem = (r: ItemRow): JobItem => ({
  jobId: r.job_id,
  idx: r.idx,
  status: r.status,
  request: JSON.parse(r.request_json),
  result: r.result_json ? JSON.parse(r.result_json) : null,
  error: r.error,
  sessionId: r.session_id,
  attempts: r.attempts,
  maxAttempts: r.max_attempts,
  createdAt: r.created_at,
  startedAt: r.started_at,
  finishedAt: r.finished_at,
})

function ensureColumn(db: SqliteDatabase, table: string, column: string, definition: string): void {
  const columns = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>
  if (columns.some((entry) => entry.name === column)) return
  db.prepare(`ALTER TABLE ${table} ADD COLUMN ${definition}`).run()
}

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS jobs (
      id TEXT PRIMARY KEY,
      kind TEXT NOT NULL,
      status TEXT NOT NULL,
      total INTEGER NOT NULL,
      succeeded INTEGER NOT NULL DEFAULT 0,
      failed INTEGER NOT NULL DEFAULT 0,
      canceled INTEGER NOT NULL DEFAULT 0,
      concurrency INTEGER NOT NULL,
      created_at INTEGER NOT NULL,
      started_at INTEGER,
      finished_at INTEGER,
      error TEXT
    )`,
  ).run()
  db.prepare(
    `CREATE TABLE IF NOT EXISTS job_items (
      job_id TEXT NOT NULL,
      idx INTEGER NOT NULL,
      status TEXT NOT NULL,
      request_json TEXT NOT NULL,
      result_json TEXT,
      error TEXT,
      session_id TEXT,
      attempts INTEGER NOT NULL DEFAULT 0,
      max_attempts INTEGER NOT NULL DEFAULT 1,
      created_at INTEGER NOT NULL DEFAULT 0,
      started_at INTEGER,
      finished_at INTEGER,
      PRIMARY KEY (job_id, idx),
      FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
    )`,
  ).run()
  ensureColumn(db, 'job_items', 'session_id', 'session_id TEXT')
  ensureColumn(db, 'job_items', 'attempts', 'attempts INTEGER NOT NULL DEFAULT 0')
  ensureColumn(db, 'job_items', 'max_attempts', 'max_attempts INTEGER NOT NULL DEFAULT 1')
  ensureColumn(db, 'job_items', 'created_at', 'created_at INTEGER NOT NULL DEFAULT 0')
  db.prepare(`UPDATE job_items SET created_at=? WHERE created_at=0`).run(Date.now())
  db.prepare(
    `CREATE INDEX IF NOT EXISTS job_items_finished
      ON job_items(job_id, finished_at)`,
  ).run()
  db.prepare(
    `CREATE INDEX IF NOT EXISTS job_items_session_queue
      ON job_items(job_id, session_id, status, idx)`,
  ).run()
  db.prepare(
    `CREATE INDEX IF NOT EXISTS jobs_kind_status
      ON jobs(kind, status, created_at)`,
  ).run()
}

export interface JobsRepo {
  create(input: { kind: JobKind; total: number; concurrency: number }): Job
  getOrCreateOpenJob(input: { kind: JobKind; concurrency: number }): Job
  get(id: string): Job | null
  insertItems(
    jobId: string,
    items: Array<{
      idx: number
      request: unknown
      sessionId?: string | null
      maxAttempts?: number
    }>,
  ): void
  appendItem(
    jobId: string,
    item: { request: unknown; sessionId?: string | null; maxAttempts?: number },
  ): JobItem
  listItems(jobId: string, sinceIdx: number): JobItem[]
  listCompletedItems(jobId: string, sinceIdx: number): JobItem[]
  startItem(jobId: string, idx: number): void
  claimNextPartitioned(jobId: string): JobItem | null
  completeItem(
    jobId: string,
    idx: number,
    payload: { status: 'succeeded'; result: unknown } | { status: 'failed'; error: string },
  ): void
  requeueItem(jobId: string, idx: number, error?: string | null): void
  finalizePartitionedJob(jobId: string): Job | null
  requeueInterruptedPartitioned(kind: JobKind): number
  positionOfItem(kind: JobKind, sessionId: string, jobId: string, idx: number): number | null
  partitionedQueueStatus(kind: JobKind, sessionId: string): PartitionedQueueStatus
  updateStatus(id: string, status: JobStatus, error?: string | null): void
  markInProgressFailed(): number
  listInProgress(): Job[]
}

export function createJobsRepo(): JobsRepo {
  const db = openDomainDb({ name: 'jobs' })
  ensureSchema(db)
  const createJob = ({
    kind,
    total,
    concurrency,
  }: {
    kind: JobKind
    total: number
    concurrency: number
  }): Job => {
    const id = randomUUID()
    const now = Date.now()
    db.prepare(
      `INSERT INTO jobs (id, kind, status, total, concurrency, created_at)
       VALUES (?, ?, 'pending', ?, ?, ?)`,
    ).run(id, kind, total, concurrency, now)
    return toJob(db.prepare('SELECT * FROM jobs WHERE id=?').get(id) as JobRow)
  }
  return {
    create({ kind, total, concurrency }) {
      return createJob({ kind, total, concurrency })
    },
    getOrCreateOpenJob({ kind, concurrency }) {
      const existing = db
        .prepare(
          `SELECT * FROM jobs
           WHERE kind=? AND status IN ('pending','running')
           ORDER BY created_at ASC
           LIMIT 1`,
        )
        .get(kind) as JobRow | undefined
      if (existing) return toJob(existing)
      return createJob({ kind, total: 0, concurrency })
    },
    get(id) {
      const r = db.prepare('SELECT * FROM jobs WHERE id=?').get(id) as JobRow | undefined
      return r ? toJob(r) : null
    },
    insertItems(jobId, items) {
      const stmt = db.prepare(
        `INSERT INTO job_items (
          job_id, idx, status, request_json, session_id, max_attempts, created_at
        )
         VALUES (?, ?, 'queued', ?, ?, ?, ?)`,
      )
      const tx = db.transaction((rows: typeof items) => {
        const now = Date.now()
        for (const it of rows) {
          stmt.run(
            jobId,
            it.idx,
            JSON.stringify(it.request),
            it.sessionId ?? null,
            Math.max(1, Math.floor(it.maxAttempts ?? 1)),
            now,
          )
        }
      })
      tx(items)
    },
    appendItem(jobId, item) {
      const append = db.transaction(() => {
        const maxIdx = db
          .prepare(`SELECT COALESCE(MAX(idx) + 1, 0) AS idx FROM job_items WHERE job_id=?`)
          .get(jobId) as { idx: number }
        const idx = maxIdx.idx
        const now = Date.now()
        db.prepare(
          `INSERT INTO job_items (
            job_id, idx, status, request_json, session_id, max_attempts, created_at
          )
           VALUES (?, ?, 'queued', ?, ?, ?, ?)`,
        ).run(
          jobId,
          idx,
          JSON.stringify(item.request),
          item.sessionId ?? null,
          Math.max(1, Math.floor(item.maxAttempts ?? 1)),
          now,
        )
        db.prepare(`UPDATE jobs SET total=total+1 WHERE id=?`).run(jobId)
        return db.prepare('SELECT * FROM job_items WHERE job_id=? AND idx=?').get(jobId, idx) as
          | ItemRow
          | undefined
      })
      const row = append()
      if (!row) throw new Error(`job item not found after append: ${jobId}`)
      return toItem(row)
    },
    listItems(jobId, sinceIdx) {
      return (
        db
          .prepare(`SELECT * FROM job_items WHERE job_id=? AND idx>=? ORDER BY idx`)
          .all(jobId, sinceIdx) as ItemRow[]
      ).map(toItem)
    },
    listCompletedItems(jobId, sinceIdx) {
      return (
        db
          .prepare(
            `SELECT * FROM job_items WHERE job_id=? AND idx>=?
             AND status IN ('succeeded','failed') ORDER BY idx`,
          )
          .all(jobId, sinceIdx) as ItemRow[]
      ).map(toItem)
    },
    startItem(jobId, idx) {
      db.prepare(
        `UPDATE job_items
         SET status='running', attempts=attempts+1, error=NULL, started_at=?, finished_at=NULL
         WHERE job_id=? AND idx=?`,
      ).run(Date.now(), jobId, idx)
    },
    claimNextPartitioned(jobId) {
      const claim = db.transaction(() => {
        const row = db
          .prepare(
            `SELECT i.* FROM job_items i
             WHERE i.job_id=? AND i.status='queued'
             AND (
               i.session_id IS NULL
               OR NOT EXISTS (
                 SELECT 1 FROM job_items r
                 WHERE r.job_id=i.job_id
                   AND r.session_id=i.session_id
                   AND r.status='running'
               )
             )
             ORDER BY i.idx ASC
             LIMIT 1`,
          )
          .get(jobId) as ItemRow | undefined
        if (!row) return null
        db.prepare(
          `UPDATE job_items
           SET status='running', attempts=attempts+1, error=NULL, started_at=?, finished_at=NULL
           WHERE job_id=? AND idx=? AND status='queued'`,
        ).run(Date.now(), row.job_id, row.idx)
        return db
          .prepare('SELECT * FROM job_items WHERE job_id=? AND idx=?')
          .get(row.job_id, row.idx) as ItemRow | undefined
      })
      const row = claim()
      return row ? toItem(row) : null
    },
    completeItem(jobId, idx, payload) {
      const now = Date.now()
      if (payload.status === 'succeeded') {
        db.prepare(
          `UPDATE job_items
           SET status='succeeded', result_json=?, error=NULL, finished_at=?
           WHERE job_id=? AND idx=?`,
        ).run(JSON.stringify(payload.result), now, jobId, idx)
        db.prepare(`UPDATE jobs SET succeeded=succeeded+1 WHERE id=?`).run(jobId)
      } else {
        db.prepare(
          `UPDATE job_items SET status='failed', error=?, finished_at=?
           WHERE job_id=? AND idx=?`,
        ).run(payload.error, now, jobId, idx)
        db.prepare(`UPDATE jobs SET failed=failed+1 WHERE id=?`).run(jobId)
      }
    },
    requeueItem(jobId, idx, error = null) {
      db.prepare(
        `UPDATE job_items
         SET status='queued', error=?, started_at=NULL, finished_at=NULL
         WHERE job_id=? AND idx=?`,
      ).run(error, jobId, idx)
    },
    finalizePartitionedJob(jobId) {
      const remaining = db
        .prepare(
          `SELECT COUNT(*) AS count FROM job_items
           WHERE job_id=? AND status IN ('queued','running')`,
        )
        .get(jobId) as { count: number }
      const current = db.prepare('SELECT * FROM jobs WHERE id=?').get(jobId) as JobRow | undefined
      if (remaining.count > 0) return current ? toJob(current) : null
      const failed = db
        .prepare(`SELECT COUNT(*) AS count FROM job_items WHERE job_id=? AND status='failed'`)
        .get(jobId) as { count: number }
      const status: JobStatus = failed.count > 0 ? 'failed' : 'completed'
      const now = Date.now()
      db.prepare(`UPDATE jobs SET status=?, finished_at=?, error=NULL WHERE id=?`).run(
        status,
        now,
        jobId,
      )
      const row = db.prepare('SELECT * FROM jobs WHERE id=?').get(jobId) as JobRow | undefined
      return row ? toJob(row) : null
    },
    requeueInterruptedPartitioned(kind) {
      const tx = db.transaction(() => {
        const running = db
          .prepare(
            `UPDATE job_items
             SET status='queued', started_at=NULL, finished_at=NULL
             WHERE status='running'
               AND job_id IN (SELECT id FROM jobs WHERE kind=?)`,
          )
          .run(kind)
        db.prepare(
          `UPDATE jobs
           SET status='pending', finished_at=NULL, error=NULL
           WHERE kind=? AND status IN ('pending','running')`,
        ).run(kind)
        return running.changes
      })
      return tx()
    },
    positionOfItem(kind, sessionId, jobId, idx) {
      const row = db
        .prepare(
          `SELECT COUNT(*) AS position
           FROM job_items i
           JOIN jobs j ON j.id=i.job_id
           JOIN jobs target_job ON target_job.id=?
           WHERE j.kind=?
             AND i.session_id=?
             AND i.status IN ('queued','running')
             AND (
               j.created_at < target_job.created_at
               OR (i.job_id=? AND i.idx <= ?)
             )`,
        )
        .get(jobId, kind, sessionId, jobId, idx) as { position: number } | undefined
      return row && row.position > 0 ? row.position : null
    },
    partitionedQueueStatus(kind, sessionId) {
      const rows = db
        .prepare(
          `SELECT i.*, j.created_at AS job_created_at
             FROM job_items i
             JOIN jobs j ON j.id=i.job_id
             WHERE j.kind=? AND i.session_id=?
             ORDER BY j.created_at ASC, i.idx ASC`,
        )
        .all(kind, sessionId) as ItemWithJobRow[]
      const counts = rows.reduce(
        (acc, row) => {
          acc[row.status] += 1
          return acc
        },
        { queued: 0, running: 0, succeeded: 0, failed: 0 } as Record<ItemStatus, number>,
      )
      const firstActive = rows.findIndex(
        (row) => row.status === 'queued' || row.status === 'running',
      )
      return {
        sessionId,
        queued: counts.queued,
        running: counts.running,
        succeeded: counts.succeeded,
        failed: counts.failed,
        total: rows.length,
        position: firstActive >= 0 ? firstActive + 1 : null,
        items: rows.map((row) => ({
          itemId: `${row.job_id}:${row.idx}`,
          jobId: row.job_id,
          idx: row.idx,
          status: row.status,
          attempts: row.attempts,
          maxAttempts: row.max_attempts,
          error: row.error,
          createdAt: row.created_at || row.job_created_at,
          startedAt: row.started_at,
          finishedAt: row.finished_at,
        })),
      }
    },
    updateStatus(id, status, error = null) {
      const now = Date.now()
      const isStart = status === 'running'
      const isEnd = status === 'completed' || status === 'failed' || status === 'canceled'
      db.prepare(
        `UPDATE jobs SET status=?,
          started_at=COALESCE(started_at, CASE WHEN ? THEN ? END),
          finished_at=CASE WHEN ? THEN ? ELSE finished_at END,
          error=? WHERE id=?`,
      ).run(status, isStart ? 1 : 0, now, isEnd ? 1 : 0, now, error, id)
    },
    markInProgressFailed() {
      const r = db
        .prepare(
          `UPDATE jobs SET status='failed', error='daemon restart', finished_at=?
         WHERE status IN ('pending','running') AND kind <> 'meeting_voice'`,
        )
        .run(Date.now())
      return r.changes
    },
    listInProgress() {
      return (
        db.prepare(`SELECT * FROM jobs WHERE status IN ('pending','running')`).all() as JobRow[]
      ).map(toJob)
    },
  }
}
