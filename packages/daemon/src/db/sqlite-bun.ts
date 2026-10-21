import { createRequire } from 'node:module'
import type {
  OpenDatabaseOptions,
  SqliteDatabase,
  SqliteRunResult,
  SqliteStatement,
} from './sqlite.js'

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

function toBunCtorOptions(options: OpenDatabaseOptions): BunDatabaseCtorOptions {
  if (options.readonly) return { strict: true, readonly: true }
  return { strict: true, create: !options.fileMustExist, readwrite: true }
}

export function openBunDatabase(path: string, options: OpenDatabaseOptions = {}): SqliteDatabase {
  const Database = loadDatabaseImpl()
  const db = new Database(path, toBunCtorOptions(options))
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
