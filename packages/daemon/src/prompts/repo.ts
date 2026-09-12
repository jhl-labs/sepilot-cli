import type { SqliteDatabase } from '../db/sqlite.js'
import { openDomainDb } from '../storage/domain-db.js'
import type { PromptTemplate, PromptTemplateInput } from './schema.js'

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS prompt_templates (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
  )`,
  ).run()
}

export interface PromptTemplatesRepo {
  list(): PromptTemplate[]
  upsert(input: PromptTemplateInput): PromptTemplate
  remove(id: string): void
}

export function createPromptTemplatesRepo(): PromptTemplatesRepo {
  const db = openDomainDb({ name: 'prompt-templates' })
  ensureSchema(db)
  return {
    list() {
      return db
        .prepare(
          'SELECT id, title, body FROM prompt_templates ORDER BY title',
        )
        .all() as PromptTemplate[]
    },
    upsert(input) {
      const id = input.id ?? crypto.randomUUID()
      const now = Date.now()
      db.prepare(
        `INSERT INTO prompt_templates (id, title, body, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET title=excluded.title, body=excluded.body, updated_at=excluded.updated_at`,
      ).run(id, input.title, input.body, now, now)
      return db
        .prepare('SELECT id, title, body FROM prompt_templates WHERE id=?')
        .get(id) as PromptTemplate
    },
    remove(id) {
      db.prepare('DELETE FROM prompt_templates WHERE id=?').run(id)
    },
  }
}
