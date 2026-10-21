import { createRequire } from 'node:module'
import type DatabaseConstructor from 'better-sqlite3'
import type { OpenDatabaseOptions, SqliteDatabase } from './sqlite.js'

const requireFromHere = createRequire(import.meta.url)
let DatabaseImpl: typeof DatabaseConstructor | undefined

function loadDatabaseImpl(): typeof DatabaseConstructor {
  if (!DatabaseImpl) {
    DatabaseImpl = requireFromHere('better-sqlite3') as typeof DatabaseConstructor
  }
  return DatabaseImpl
}

export function openNodeDatabase(path: string, options: OpenDatabaseOptions = {}): SqliteDatabase {
  const Database = loadDatabaseImpl()
  const ctorOptions: { readonly?: boolean; fileMustExist?: boolean; timeout?: number } = {}
  if (options.readonly !== undefined) ctorOptions.readonly = options.readonly
  if (options.fileMustExist !== undefined) ctorOptions.fileMustExist = options.fileMustExist
  if (options.timeout !== undefined) ctorOptions.timeout = options.timeout
  const db = new Database(path, ctorOptions)
  try {
    db.pragma('foreign_keys = ON')
  } catch {
  }
  return db as unknown as SqliteDatabase
}
