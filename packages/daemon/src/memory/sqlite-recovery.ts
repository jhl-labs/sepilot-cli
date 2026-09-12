import { existsSync, renameSync } from 'node:fs'
import { openDatabase } from '../db/sqlite.js'

const SQLITE_SIDECARS = ['', '-wal', '-shm'] as const

function corruptSuffix(): string {
  return `.corrupt-${new Date().toISOString().replace(/[:.]/g, '-')}-${process.pid}`
}

export function recoverSqliteDatabaseIfCorrupt(dbPath: string): string[] {
  if (!existsSync(dbPath)) return []

  let db: ReturnType<typeof openDatabase> | undefined
  try {
    db = openDatabase(dbPath, { readonly: true, fileMustExist: true })
    const result = db.pragma('integrity_check', { simple: true })
    if (result === 'ok') return []
    throw new Error(`SQLite integrity_check failed: ${String(result)}`)
  } catch {
    try {
      db?.close()
      db = undefined
    } catch {
      // best effort; proceed with quarantine below
    }
    const suffix = corruptSuffix()
    const renamed: string[] = []
    for (const sidecar of SQLITE_SIDECARS) {
      const source = `${dbPath}${sidecar}`
      if (!existsSync(source)) continue
      const target = `${source}${suffix}`
      renameSync(source, target)
      renamed.push(target)
    }
    return renamed
  } finally {
    try {
      db?.close()
    } catch {
      // best effort; the file is about to be quarantined or was already closed
    }
  }
}
