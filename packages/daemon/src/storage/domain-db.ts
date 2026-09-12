import { mkdirSync } from 'node:fs'
import { join } from 'node:path'
import {
  DEFAULT_SQLITE_BUSY_TIMEOUT_MS,
  openDatabase,
  type SqliteDatabase,
} from '../db/sqlite.js'
import { sepilotdHome } from './home.js'
import { secureDir, secureFile } from '../utils/secure-file.js'

const cache = new Map<string, SqliteDatabase>()

export interface OpenDomainDbInput {
  name: string
  filename?: string
}

export function openDomainDb(input: OpenDomainDbInput): SqliteDatabase {
  const dir = join(sepilotdHome(), input.name)
  const file = join(dir, input.filename ?? `${input.name}.db`)
  const cached = cache.get(file)
  if (cached) return cached
  mkdirSync(dir, { recursive: true, mode: 0o700 })
  secureDir(dir)
  const db = openDatabase(file, { timeout: DEFAULT_SQLITE_BUSY_TIMEOUT_MS })
  // Tighten perms on the db file (and the WAL/SHM sidecars that
  // better-sqlite3 may have created during the open above) so a
  // co-tenant on a shared host cannot read team-docs PATs / memory
  // contents / personal-docs markdown that the daemon owns.
  secureFile(file)
  secureFile(`${file}-wal`)
  secureFile(`${file}-shm`)
  db.pragma('journal_mode = WAL')
  db.pragma('foreign_keys = ON')
  cache.set(file, db)
  return db
}

export function closeAllDomainDbs(): void {
  for (const [, db] of cache) {
    try {
      db.close()
    } catch {
      /* ignore */
    }
  }
  cache.clear()
}

/** @deprecated Use closeAllDomainDbs; retained for existing focused tests. */
export const closeAllDomainDbsForTests = closeAllDomainDbs
