import { openBunDatabase } from './sqlite-bun.js'
import { openNodeDatabase } from './sqlite-node.js'

export interface SqliteRunResult {
  changes: number
  lastInsertRowid: number | bigint
}

export interface SqliteStatement {
  get(...params: unknown[]): unknown
  all(...params: unknown[]): unknown[]
  run(...params: unknown[]): SqliteRunResult
  pluck(toggle?: boolean): this
}

export interface SqliteDatabase {
  prepare(sql: string): SqliteStatement
  exec(sql: string): void
  pragma(source: string, options?: { simple?: boolean }): unknown
  transaction<A extends unknown[], R>(fn: (...args: A) => R): (...args: A) => R
  loadExtension(path: string): void
  close(): void
  readonly open: boolean
}

export interface OpenDatabaseOptions {
  readonly?: boolean
  fileMustExist?: boolean
  timeout?: number
}

export const DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 5000

declare const Bun: unknown

export function openDatabase(path: string, options: OpenDatabaseOptions = {}): SqliteDatabase {
  const resolvedOptions = {
    ...options,
    timeout: options.timeout ?? DEFAULT_SQLITE_BUSY_TIMEOUT_MS,
  }
  if (typeof Bun !== 'undefined') return openBunDatabase(path, resolvedOptions)
  return openNodeDatabase(path, resolvedOptions)
}
