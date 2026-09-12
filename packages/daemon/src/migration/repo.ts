import { randomUUID } from 'node:crypto'
import type { SqliteDatabase } from '../db/sqlite.js'
import { openDomainDb } from '../storage/domain-db.js'
import type {
  MigrationRun,
  MigrationStepRow,
  MigrationStatus,
  StepStatus,
} from './types.js'

interface RunRow {
  id: string
  source_path: string
  status: MigrationStatus
  dry_run: number
  started_at: number | null
  finished_at: number | null
  error: string | null
  created_at: number
}

interface StepRow {
  migration_id: string
  name: string
  status: StepStatus
  copied: number
  skipped: number
  errors_json: string
  started_at: number | null
  finished_at: number | null
}

const toRun = (r: RunRow): MigrationRun => ({
  id: r.id,
  sourcePath: r.source_path,
  status: r.status,
  dryRun: !!r.dry_run,
  startedAt: r.started_at,
  finishedAt: r.finished_at,
  error: r.error,
  createdAt: r.created_at,
})

const toStep = (r: StepRow): MigrationStepRow => ({
  migrationId: r.migration_id,
  name: r.name,
  status: r.status,
  copied: r.copied,
  skipped: r.skipped,
  errors: JSON.parse(r.errors_json) as { path: string; error: string }[],
  startedAt: r.started_at,
  finishedAt: r.finished_at,
})

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS migration_runs (
      id TEXT PRIMARY KEY,
      source_path TEXT NOT NULL,
      status TEXT NOT NULL,
      dry_run INTEGER NOT NULL,
      started_at INTEGER,
      finished_at INTEGER,
      error TEXT,
      created_at INTEGER NOT NULL
    )`,
  ).run()
  db.prepare(
    `CREATE TABLE IF NOT EXISTS migration_steps (
      migration_id TEXT NOT NULL,
      name TEXT NOT NULL,
      status TEXT NOT NULL,
      copied INTEGER NOT NULL DEFAULT 0,
      skipped INTEGER NOT NULL DEFAULT 0,
      errors_json TEXT NOT NULL DEFAULT '[]',
      started_at INTEGER,
      finished_at INTEGER,
      PRIMARY KEY (migration_id, name),
      FOREIGN KEY (migration_id) REFERENCES migration_runs(id) ON DELETE CASCADE
    )`,
  ).run()
}

export interface MigrationRepo {
  create(input: { sourcePath: string; dryRun: boolean }): MigrationRun
  get(id: string): MigrationRun | null
  insertSteps(runId: string, names: string[]): void
  listSteps(runId: string): MigrationStepRow[]
  startStep(runId: string, name: string): void
  completeStep(
    runId: string,
    name: string,
    payload: {
      status: StepStatus
      copied: number
      skipped: number
      errors: { path: string; error: string }[]
    },
  ): void
  updateRunStatus(
    id: string,
    status: MigrationStatus,
    error?: string | null,
  ): void
  markInProgressFailed(): number
}

export function createMigrationRepo(): MigrationRepo {
  // Share the jobs.db file (Phase 1 jobs tables already live here).
  const db = openDomainDb({ name: 'jobs' })
  ensureSchema(db)
  return {
    create({ sourcePath, dryRun }) {
      const id = randomUUID()
      const now = Date.now()
      db.prepare(
        `INSERT INTO migration_runs (id, source_path, status, dry_run, created_at)
         VALUES (?, ?, 'pending', ?, ?)`,
      ).run(id, sourcePath, dryRun ? 1 : 0, now)
      return toRun(
        db
          .prepare('SELECT * FROM migration_runs WHERE id=?')
          .get(id) as RunRow,
      )
    },
    get(id) {
      const r = db
        .prepare('SELECT * FROM migration_runs WHERE id=?')
        .get(id) as RunRow | undefined
      return r ? toRun(r) : null
    },
    insertSteps(runId, names) {
      const stmt = db.prepare(
        `INSERT INTO migration_steps (migration_id, name, status)
         VALUES (?, ?, 'queued')`,
      )
      const tx = db.transaction((rows: string[]) => {
        for (const n of rows) stmt.run(runId, n)
      })
      tx(names)
    },
    listSteps(runId) {
      return (
        db
          .prepare('SELECT * FROM migration_steps WHERE migration_id=?')
          .all(runId) as StepRow[]
      ).map(toStep)
    },
    startStep(runId, name) {
      db.prepare(
        `UPDATE migration_steps SET status='running', started_at=?
         WHERE migration_id=? AND name=?`,
      ).run(Date.now(), runId, name)
    },
    completeStep(runId, name, payload) {
      db.prepare(
        `UPDATE migration_steps
         SET status=?, copied=?, skipped=?, errors_json=?, finished_at=?
         WHERE migration_id=? AND name=?`,
      ).run(
        payload.status,
        payload.copied,
        payload.skipped,
        JSON.stringify(payload.errors),
        Date.now(),
        runId,
        name,
      )
    },
    updateRunStatus(id, status, error = null) {
      const now = Date.now()
      const isStart = status === 'running'
      const isEnd =
        status === 'completed' || status === 'failed' || status === 'canceled'
      db.prepare(
        `UPDATE migration_runs SET status=?,
         started_at=COALESCE(started_at, CASE WHEN ? THEN ? END),
         finished_at=CASE WHEN ? THEN ? ELSE finished_at END,
         error=? WHERE id=?`,
      ).run(status, isStart ? 1 : 0, now, isEnd ? 1 : 0, now, error, id)
    },
    markInProgressFailed() {
      const r = db
        .prepare(
          `UPDATE migration_runs
           SET status='failed', error='daemon restart', finished_at=?
           WHERE status IN ('pending','running')`,
        )
        .run(Date.now())
      return r.changes
    },
  }
}
