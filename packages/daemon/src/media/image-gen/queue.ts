import type { SqliteDatabase } from '../../db/sqlite.js'
import { openDomainDb } from '../../storage/domain-db.js'
import { deleteJobOutputs, writeOutput } from './files.js'
import { publishJob, type JobEvent } from './events.js'
import type { MediaOutputKind, Provider } from './adapter.js'
import type { ImagePromptPreparer } from './prompt-translation.js'

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'

export interface JobRow {
  id: string
  provider_id: string
  prompt: string
  params: string
  status: JobStatus
  progress: number
  outputs: string
  error: string | null
  created_at: number
  updated_at: number
}

export interface Job {
  id: string
  providerId: string
  prompt: string
  status: JobStatus
  progress: number
  outputs: {
    id: string
    mime: string
    path?: string
    kind?: MediaOutputKind
  }[]
  error: string | null
  createdAt: number
}

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, provider_id TEXT NOT NULL, prompt TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0, outputs TEXT NOT NULL DEFAULT '[]',
    error TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
  )`,
  ).run()
}

function rowToJob(r: JobRow): Job {
  let outputs: {
    id: string
    mime: string
    path?: string
    kind?: MediaOutputKind
  }[] = []
  try {
    outputs = JSON.parse(r.outputs) as {
      id: string
      mime: string
      path?: string
      kind?: MediaOutputKind
    }[]
  } catch {
    outputs = []
  }
  return {
    id: r.id,
    providerId: r.provider_id,
    prompt: r.prompt,
    status: r.status,
    progress: r.progress,
    outputs,
    error: r.error,
    createdAt: r.created_at,
  }
}

export interface ImageGenQueue {
  enqueue(input: { providerId: string; prompt: string; params?: Record<string, unknown> }): Job
  list(limit: number): Job[]
  get(id: string): Job | null
  cancel(id: string): void
  remove(id: string): boolean
  setPromptPreparer(preparer: ImagePromptPreparer | null): void
  start(): void
  stop(): void
}

export function createQueue(providers: Map<string, Provider>): ImageGenQueue {
  const db = openDomainDb({ name: 'image-gen', filename: 'jobs.db' })
  ensureSchema(db)
  let timer: ReturnType<typeof setInterval> | null = null
  let running = false
  let promptPreparer: ImagePromptPreparer | null = null
  const activeControllers = new Map<string, AbortController>()

  function emit(job: Job): void {
    publishJob({ ...job } as JobEvent)
  }

  async function tick(): Promise<void> {
    if (running) return
    // In tests, the underlying database may be closed while an unref'd
    // interval is still pending. Bail out cleanly instead of throwing.
    if (!db.open) return
    const next = db
      .prepare(`SELECT * FROM jobs WHERE status='queued' ORDER BY created_at ASC LIMIT 1`)
      .get() as JobRow | undefined
    if (!next) return
    running = true
    const provider = providers.get(next.provider_id)
    if (!provider) {
      const now = Date.now()
      db.prepare(`UPDATE jobs SET status='failed', error=?, updated_at=? WHERE id=?`).run(
        `provider not found: ${next.provider_id}`,
        now,
        next.id,
      )
      emit(
        rowToJob({
          ...next,
          status: 'failed',
          error: `provider not found: ${next.provider_id}`,
          updated_at: now,
        }),
      )
      running = false
      return
    }
    db.prepare(`UPDATE jobs SET status='running', progress=0, updated_at=? WHERE id=?`).run(
      Date.now(),
      next.id,
    )
    emit(rowToJob({ ...next, status: 'running', progress: 0 }))
    const controller = new AbortController()
    activeControllers.set(next.id, controller)
    try {
      const params = JSON.parse(next.params) as Record<string, unknown>
      const prepared = promptPreparer
        ? await promptPreparer({
            providerId: next.provider_id,
            prompt: next.prompt,
            params,
            signal: controller.signal,
          })
        : { prompt: next.prompt, params }
      const out = await provider.run({
        jobId: next.id,
        prompt: prepared.prompt,
        params: prepared.params,
        signal: controller.signal,
        onProgress: (p) => {
          if (controller.signal.aborted) return
          db.prepare(`UPDATE jobs SET progress=?, updated_at=? WHERE id=?`).run(
            p,
            Date.now(),
            next.id,
          )
          emit(rowToJob({ ...next, status: 'running', progress: p }))
        },
      })
      // The generate() await can span a shutdown: the entry guard only covers
      // tick start, so every resumption point must re-check before touching
      // the connection or the settle path throws as an unhandled rejection.
      if (!db.open) return
      const current = db.prepare('SELECT * FROM jobs WHERE id=?').get(next.id) as JobRow | undefined
      if (!current) return
      if (current.status === 'cancelled') {
        emit(rowToJob(current))
        return
      }
      const outputs = out.outputs.map((o) => ({
        id: `${next.id}:${o.id}`,
        mime: o.mime,
        ...(o.path ? { path: o.path } : {}),
        ...(o.kind ? { kind: o.kind } : {}),
      }))
      for (const o of out.outputs) {
        writeOutput(next.id, o.id, o.bytes, o.mime)
      }
      const now = Date.now()
      db.prepare(
        `UPDATE jobs SET status='succeeded', progress=1, outputs=?, updated_at=? WHERE id=?`,
      ).run(JSON.stringify(outputs), now, next.id)
      emit(
        rowToJob({
          ...next,
          status: 'succeeded',
          progress: 1,
          outputs: JSON.stringify(outputs),
          updated_at: now,
        }),
      )
    } catch (err) {
      if (!db.open) return
      const current = db.prepare('SELECT * FROM jobs WHERE id=?').get(next.id) as JobRow | undefined
      if (!current) return
      if (current.status === 'cancelled') {
        emit(rowToJob(current))
        return
      }
      const now = Date.now()
      db.prepare(`UPDATE jobs SET status='failed', error=?, updated_at=? WHERE id=?`).run(
        (err as Error).message,
        now,
        next.id,
      )
      emit(
        rowToJob({
          ...next,
          status: 'failed',
          error: (err as Error).message,
          updated_at: now,
        }),
      )
    } finally {
      activeControllers.delete(next.id)
      running = false
    }
  }

  return {
    enqueue(input) {
      const id = crypto.randomUUID()
      const now = Date.now()
      db.prepare(
        `INSERT INTO jobs (id, provider_id, prompt, params, status, progress, outputs, error, created_at, updated_at) VALUES (?, ?, ?, ?, 'queued', 0, '[]', NULL, ?, ?)`,
      ).run(id, input.providerId, input.prompt, JSON.stringify(input.params ?? {}), now, now)
      const job = rowToJob(db.prepare('SELECT * FROM jobs WHERE id=?').get(id) as JobRow)
      emit(job)
      return job
    },
    list(limit) {
      return (
        db.prepare('SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?').all(limit) as JobRow[]
      ).map(rowToJob)
    },
    get(id) {
      const row = db.prepare('SELECT * FROM jobs WHERE id=?').get(id) as JobRow | undefined
      return row ? rowToJob(row) : null
    },
    cancel(id) {
      activeControllers.get(id)?.abort()
      db.prepare(
        `UPDATE jobs SET status='cancelled', updated_at=? WHERE id=? AND status IN ('queued','running')`,
      ).run(Date.now(), id)
      const j = db.prepare('SELECT * FROM jobs WHERE id=?').get(id) as JobRow | undefined
      if (j) emit(rowToJob(j))
    },
    remove(id) {
      const row = db.prepare('SELECT * FROM jobs WHERE id=?').get(id) as JobRow | undefined
      if (!row) return false
      activeControllers.get(id)?.abort()
      activeControllers.delete(id)
      db.prepare('DELETE FROM jobs WHERE id=?').run(id)
      deleteJobOutputs(id)
      return true
    },
    setPromptPreparer(preparer) {
      promptPreparer = preparer
    },
    start() {
      if (timer) return
      timer = setInterval(() => {
        void tick()
      }, 50)
      // Don't block event loop; tests and clean shutdowns don't need to
      // wait for the queue interval to exit.
      if (typeof (timer as unknown as { unref?: () => void }).unref === 'function') {
        ;(timer as unknown as { unref: () => void }).unref()
      }
    },
    stop() {
      if (timer) clearInterval(timer)
      timer = null
    },
  }
}
