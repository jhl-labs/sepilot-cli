import { openBunDatabase } from './sqlite-bun.js'
import { openNodeDatabase } from './sqlite-node.js'

/**
 * Minimal SQLite abstraction for the daemon.
 *
 * Why this exists: the daemon uses `better-sqlite3` (a native addon) everywhere
 * it touches `~/.sepilotd/*.db`. To ship the daemon as a Bun single-file binary
 * (`bun build --compile`) the native addon must be swapped for `bun:sqlite`.
 * `openDatabase()` picks the backend at runtime; Node keeps `better-sqlite3`,
 * the Bun binary uses `bun:sqlite`.
 *
 * Bundling note: both backend modules are statically imported here, but neither
 * has a top-level native dependency — `sqlite-node.ts` requires `better-sqlite3`
 * lazily (only when `openNodeDatabase` actually runs) and `sqlite-bun.ts` will
 * `require('bun:sqlite')` lazily too. So a `bun build --compile` that marks
 * `better-sqlite3` external never resolves the addon (the Bun runtime always
 * takes the `sqlite-bun.ts` branch), and a Node build never touches `bun:sqlite`.
 *
 * The surface here is intentionally the *subset* of `better-sqlite3` that the
 * daemon actually uses (audited across `packages/daemon/src`). Adding methods
 * "just in case" is discouraged — `better-sqlite3`'s real types are a structural
 * superset of this interface, so widening it later is cheap.
 */

export interface SqliteRunResult {
  changes: number
  lastInsertRowid: number | bigint
}

export interface SqliteStatement {
  get(...params: unknown[]): unknown
  all(...params: unknown[]): unknown[]
  run(...params: unknown[]): SqliteRunResult
  /** `better-sqlite3`'s `.pluck()` — return the first column's scalar instead of a row object. */
  pluck(toggle?: boolean): this
}

export interface SqliteDatabase {
  prepare(sql: string): SqliteStatement
  /** Run one or more SQL statements with no bound parameters and no result. */
  exec(sql: string): void
  /**
   * Run a `PRAGMA`. With `{ simple: true }` returns the single scalar value;
   * otherwise returns the array-of-rows form (e.g. `[{ user_version: 7 }]`).
   */
  pragma(source: string, options?: { simple?: boolean }): unknown
  /** Wrap `fn` so every call runs inside a transaction (nested calls become savepoints). */
  transaction<A extends unknown[], R>(fn: (...args: A) => R): (...args: A) => R
  /** Load a SQLite loadable extension (`.dll`/`.dylib`/`.so`) by file path — used for `sqlite-vec`. */
  loadExtension(path: string): void
  close(): void
  /** `false` once `close()` has been called. */
  readonly open: boolean
}

export interface OpenDatabaseOptions {
  readonly?: boolean
  fileMustExist?: boolean
  timeout?: number
}

export const DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 5000

declare const Bun: unknown

/**
 * Open a SQLite database using the runtime-appropriate backend
 * (`better-sqlite3` under Node, `bun:sqlite` inside a compiled Bun binary).
 */
export function openDatabase(path: string, options: OpenDatabaseOptions = {}): SqliteDatabase {
  const resolvedOptions = {
    ...options,
    timeout: options.timeout ?? DEFAULT_SQLITE_BUSY_TIMEOUT_MS,
  }
  if (typeof Bun !== 'undefined') return openBunDatabase(path, resolvedOptions)
  return openNodeDatabase(path, resolvedOptions)
}
