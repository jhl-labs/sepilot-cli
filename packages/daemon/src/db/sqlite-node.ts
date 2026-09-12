import { createRequire } from 'node:module'
import type DatabaseConstructor from 'better-sqlite3'
import type { OpenDatabaseOptions, SqliteDatabase } from './sqlite.js'

/**
 * `better-sqlite3` backend for {@link openDatabase}. Used by the Node runtime.
 *
 * The `better-sqlite3` native addon is pulled in lazily (a `createRequire` call
 * inside the opener), not via a top-level `import`. That keeps `sqlite-node.ts`
 * itself a pure-JS module: a Bun `--compile` build can include it in the graph
 * (it is statically imported by `./sqlite.js`) yet — because the dispatcher in
 * `./sqlite.js` routes Bun to the `bun:sqlite` backend — `openNodeDatabase`
 * never runs there, so the native addon is never resolved.
 *
 * `better-sqlite3`'s own types are a structural superset of {@link SqliteDatabase},
 * but TypeScript can't see that without the cast — this `as unknown as` is the
 * established adapter pattern, not a smell.
 */
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
  // `better-sqlite3` rejects `undefined` for these options, so only pass the ones that are set.
  const ctorOptions: { readonly?: boolean; fileMustExist?: boolean; timeout?: number } = {}
  if (options.readonly !== undefined) ctorOptions.readonly = options.readonly
  if (options.fileMustExist !== undefined) ctorOptions.fileMustExist = options.fileMustExist
  if (options.timeout !== undefined) ctorOptions.timeout = options.timeout
  const db = new Database(path, ctorOptions)
  // Enforce foreign-key constraints (off by default in SQLite). Without this,
  // `ON DELETE CASCADE` (e.g. memory_graph_edges → nodes) is a no-op and bad
  // from/to ids are silently accepted, leaving orphaned edges.
  try {
    db.pragma('foreign_keys = ON')
  } catch {
    // Read-only or restricted handles may reject pragmas; safe to ignore.
  }
  return db as unknown as SqliteDatabase
}
