import { createRequire } from 'node:module'
import type {
  OpenDatabaseOptions,
  SqliteDatabase,
  SqliteRunResult,
  SqliteStatement,
} from './sqlite.js'

/**
 * `bun:sqlite` backend for {@link openDatabase}. Used by the Bun runtime (and a
 * `bun build --compile` single-file binary).
 *
 * `bun:sqlite` is a Bun builtin module — it does not exist under Node. This file
 * is statically imported by `./sqlite.js`, which Vitest loads under Node, so the
 * builtin is pulled in *lazily* (`createRequire(...)('bun:sqlite')` inside the
 * opener) rather than via a top-level `import`. That mirrors how `sqlite-node.ts`
 * lazily requires the `better-sqlite3` native addon: a Bun `--compile` build can
 * include this module in its graph yet, since the dispatcher routes Node away
 * from here, `openNodeDatabase` never runs there — and `openBunDatabase` never
 * runs under Node.
 *
 * The wrapper emulates the slice of `better-sqlite3` the daemon actually uses
 * (audited): `.prepare/.exec/.pragma/.transaction/.close`, statement
 * `.get/.all/.run`, `RunResult.changes`. `bun:sqlite` has no `.pluck()` and no
 * array-of-rows `.pragma()`, so those are emulated here to satisfy the
 * {@link SqliteDatabase} / {@link SqliteStatement} contract.
 */

const requireBuiltin = createRequire(import.meta.url)
const DEFAULT_BUN_BUSY_TIMEOUT_MS = 5000

interface BunChanges {
  changes: number
  lastInsertRowid: number | bigint
}

interface BunStatement {
  get(...params: unknown[]): unknown
  all(...params: unknown[]): unknown[]
  run(...params: unknown[]): BunChanges
}

interface BunDatabaseCtorOptions {
  strict: true
  readonly?: boolean
  create?: boolean
  readwrite?: boolean
}

interface BunDatabase {
  prepare(sql: string): BunStatement
  exec(sql: string): void
  // `bun:sqlite`'s `transaction` has the same shape as `better-sqlite3`'s, but
  // its generic signature differs enough that TS can't reconcile them — keep the
  // concrete type loose here and re-narrow at the wrapper boundary.
  transaction(fn: (...args: unknown[]) => unknown): (...args: unknown[]) => unknown
  loadExtension(extension: string, entryPoint?: string): void
  close(): void
}

interface BunDatabaseCtor {
  new (filename: string, options?: BunDatabaseCtorOptions): BunDatabase
}

let DatabaseImpl: BunDatabaseCtor | undefined

function loadDatabaseImpl(): BunDatabaseCtor {
  if (!DatabaseImpl) {
    const mod = requireBuiltin('bun:sqlite') as { Database: BunDatabaseCtor }
    DatabaseImpl = mod.Database
  }
  return DatabaseImpl
}

function wrapStatement(raw: BunStatement): SqliteStatement {
  let pluck = false
  const project = (row: unknown): unknown =>
    pluck && row !== null && typeof row === 'object'
      ? (Object.values(row as Record<string, unknown>)[0] as unknown)
      : row
  const stmt: SqliteStatement = {
    get: (...params) => project(raw.get(...params)),
    all: (...params) => raw.all(...params).map(project),
    run: (...params): SqliteRunResult => {
      const result = raw.run(...params)
      return { changes: result.changes, lastInsertRowid: result.lastInsertRowid }
    },
    pluck(toggle = true) {
      pluck = toggle
      return stmt
    },
  }
  return stmt
}

/**
 * Match better-sqlite3's bare named bindings and missing-parameter errors.
 * Without strict mode Bun silently binds `{ limit: 10 }` to NULL for `@limit`.
 * Keep access flags explicit: passing ctor options disables Bun's implicit
 * read-write/create defaults on supported older runtimes.
 */
function toBunCtorOptions(options: OpenDatabaseOptions): BunDatabaseCtorOptions {
  if (options.readonly) return { strict: true, readonly: true }
  // Bun has no fileMustExist option; disable creation while retaining writes.
  return { strict: true, create: !options.fileMustExist, readwrite: true }
}

export function openBunDatabase(path: string, options: OpenDatabaseOptions = {}): SqliteDatabase {
  const Database = loadDatabaseImpl()
  const db = new Database(path, toBunCtorOptions(options))
  // `bun:sqlite` has no `timeout` ctor option; emulate via PRAGMA busy_timeout.
  db.exec(`PRAGMA busy_timeout = ${Math.trunc(options.timeout ?? DEFAULT_BUN_BUSY_TIMEOUT_MS)}`)
  let closed = false
  return {
    prepare: (sql) => wrapStatement(db.prepare(sql)),
    exec: (sql) => {
      db.exec(sql)
    },
    pragma: (source, opts) => {
      const rows = db.prepare(`PRAGMA ${source}`).all() as Record<string, unknown>[]
      if (opts?.simple) return rows.length > 0 ? (Object.values(rows[0])[0] as unknown) : undefined
      return rows
    },
    transaction: <A extends unknown[], R>(fn: (...args: A) => R): ((...args: A) => R) =>
      db.transaction(fn as (...args: unknown[]) => unknown) as (...args: A) => R,
    loadExtension: (extensionPath) => {
      db.loadExtension(extensionPath)
    },
    close: () => {
      db.close()
      closed = true
    },
    get open() {
      return !closed
    },
  }
}
