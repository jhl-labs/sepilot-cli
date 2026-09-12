import type { SqliteDatabase } from '../db/sqlite.js'
import { openDomainDb } from '../storage/domain-db.js'
import type { Persona, PersonaInput } from './schema.js'

export interface PersonaRepo {
  list(): Persona[]
  upsert(input: PersonaInput): Persona
  remove(id: string): void
  activate(id: string): void
}

interface Row {
  id: string
  name: string
  system_prompt: string
  memory_scope: string
  is_active: number
}

function rowToPersona(r: Row): Persona {
  return {
    id: r.id,
    name: r.name,
    systemPrompt: r.system_prompt,
    isActive: !!r.is_active,
    memoryScope: r.memory_scope === 'isolated' ? 'isolated' : 'shared',
  }
}

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS persona (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    system_prompt TEXT NOT NULL DEFAULT '',
    is_active INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
  )`,
  ).run()
  const columns = db.prepare('PRAGMA table_info(persona)').all() as Array<{ name: string }>
  if (!columns.some(column => column.name === 'memory_scope')) {
    db.prepare("ALTER TABLE persona ADD COLUMN memory_scope TEXT NOT NULL DEFAULT 'shared'").run()
  }
}

export function createPersonaRepo(): PersonaRepo {
  const db = openDomainDb({ name: 'persona' })
  ensureSchema(db)
  return {
    list() {
      return (
        db
          .prepare(
            'SELECT id, name, system_prompt, is_active, memory_scope FROM persona ORDER BY name',
          )
          .all() as Row[]
      ).map(rowToPersona)
    },
    upsert(input) {
      const now = Date.now()
      const id = input.id ?? crypto.randomUUID()
      const existing = db.prepare('SELECT memory_scope FROM persona WHERE id=?').get(id) as { memory_scope: string } | undefined
      // Isolation is a lifetime identity boundary: editing tone must not move history.
      if (existing?.memory_scope === 'isolated' && input.memoryScope === 'shared') throw new Error('Create a new persona to leave an isolated memory space')
      db.prepare(
        `INSERT INTO persona (id, name, system_prompt, is_active, created_at, updated_at, memory_scope)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET name=excluded.name, system_prompt=excluded.system_prompt, updated_at=excluded.updated_at, memory_scope=excluded.memory_scope`,
      ).run(
        id,
        input.name,
        input.systemPrompt,
        input.isActive ? 1 : 0,
        now,
        now,
        input.memoryScope ?? existing?.memory_scope ?? 'shared',
      )
      return rowToPersona(
        db
          .prepare(
            'SELECT id, name, system_prompt, is_active, memory_scope FROM persona WHERE id=?',
          )
          .get(id) as Row,
      )
    },
    remove(id) {
      db.prepare('DELETE FROM persona WHERE id=?').run(id)
    },
    activate(id) {
      const tx = db.transaction((target: string) => {
        db.prepare('UPDATE persona SET is_active=0').run()
        db.prepare(
          'UPDATE persona SET is_active=1, updated_at=? WHERE id=?',
        ).run(Date.now(), target)
      })
      tx(id)
    },
  }
}
