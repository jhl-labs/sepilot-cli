import type { SqliteDatabase } from '../db/sqlite.js'
import {
  existsSync,
  mkdirSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { openDomainDb } from '../storage/domain-db.js'
import { sepilotdHome } from '../storage/home.js'
import { assertSafeId } from '../utils/safe-id.js'
import { secureDir, secureFile } from '../utils/secure-file.js'
import type {
  PersonalDoc,
  PersonalDocContent,
  PersonalDocInput,
} from './schema.js'

function fileFor(id: string): string {
  assertSafeId(id, 'personal-doc id')
  const root = resolve(join(sepilotdHome(), 'personal-docs'))
  const file = resolve(join(root, `${id}.md`))
  if (!file.startsWith(`${root}/`)) {
    throw new Error('invalid personal-doc path')
  }
  return file
}

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS personal_docs (
    id TEXT PRIMARY KEY, path TEXT NOT NULL, updated_at INTEGER NOT NULL
  )`,
  ).run()
}

export interface PersonalDocsRepo {
  list(): PersonalDoc[]
  get(id: string): PersonalDocContent | null
  upsert(input: PersonalDocInput): PersonalDoc
  remove(id: string): void
}

export function createPersonalDocsRepo(): PersonalDocsRepo {
  const db = openDomainDb({ name: 'personal-docs' })
  ensureSchema(db)
  return {
    list() {
      return (
        db
          .prepare(
            'SELECT id, path, updated_at FROM personal_docs ORDER BY updated_at DESC',
          )
          .all() as { id: string; path: string; updated_at: number }[]
      ).map((r) => ({ id: r.id, path: r.path, updatedAt: r.updated_at }))
    },
    get(id) {
      const row = db
        .prepare('SELECT id, path, updated_at FROM personal_docs WHERE id=?')
        .get(id) as
          | { id: string; path: string; updated_at: number }
          | undefined
      if (!row) return null
      const file = fileFor(id)
      return {
        id: row.id,
        path: row.path,
        updatedAt: row.updated_at,
        content: existsSync(file) ? readFileSync(file, 'utf-8') : '',
      }
    },
    upsert(input) {
      const id = input.id ?? crypto.randomUUID()
      const file = fileFor(id)
      const dir = dirname(file)
      mkdirSync(dir, { recursive: true, mode: 0o700 })
      secureDir(dir)
      writeFileSync(file, input.content, 'utf-8')
      secureFile(file)
      const now = Date.now()
      db.prepare(
        `INSERT INTO personal_docs (id, path, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET path=excluded.path, updated_at=excluded.updated_at`,
      ).run(id, input.path, now)
      return { id, path: input.path, updatedAt: now }
    },
    remove(id) {
      const file = fileFor(id)
      if (existsSync(file)) unlinkSync(file)
      db.prepare('DELETE FROM personal_docs WHERE id=?').run(id)
    },
  }
}
