import { mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import { randomUUID } from 'node:crypto'
import type { SkillMetadata } from '@sepilotd/core'
import { openDatabase, type SqliteDatabase } from '../db/sqlite.js'

export interface StoredSkill {
  metadata: SkillMetadata
  content: string
  downloads: number
  rating: number
  publishedBy: string
  publishedAt: string
}

interface SkillRow {
  id: string
  name: string
  version: string
  description: string
  author: string
  tags: string
  tools: string
  autonomy_required: string
  content: string
  downloads: number
  rating: number
  rating_count: number
  published_by: string | null
  published_at: string
}

interface SkillCountRow {
  count: number
}

interface SkillRatingRow {
  rating: number | null
  rating_count: number | null
}

/**
 * SQLite-backed skill marketplace store.
 * Implements the server side of the SkillStoreClient API.
 */
export class SkillStore {
  private db: SqliteDatabase

  constructor(dbPath: string) {
    this.db = openDatabase(dbPath)
    this.db.pragma('journal_mode = WAL')
    this.initSchema()
  }

  static async create(dbPath: string): Promise<SkillStore> {
    await mkdir(dirname(dbPath), { recursive: true })
    return new SkillStore(dbPath)
  }

  private initSchema(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS skills (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        version TEXT NOT NULL DEFAULT '1.0.0',
        description TEXT NOT NULL,
        author TEXT,
        tags TEXT,
        tools TEXT,
        autonomy_required TEXT,
        content TEXT NOT NULL,
        downloads INTEGER NOT NULL DEFAULT 0,
        rating REAL NOT NULL DEFAULT 0,
        rating_count INTEGER NOT NULL DEFAULT 0,
        published_by TEXT,
        published_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
      )
    `)

    this.db.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
        name, description, tags,
        content=skills, content_rowid=rowid
      )
    `)

    this.db.exec(`
      CREATE TRIGGER IF NOT EXISTS skills_ai AFTER INSERT ON skills BEGIN
        INSERT INTO skills_fts(rowid, name, description, tags) VALUES (new.rowid, new.name, new.description, new.tags);
      END
    `)

    this.db.exec(`
      CREATE TRIGGER IF NOT EXISTS skills_ad AFTER DELETE ON skills BEGIN
        INSERT INTO skills_fts(skills_fts, rowid, name, description, tags) VALUES('delete', old.rowid, old.name, old.description, old.tags);
      END
    `)

    this.db.exec(`
      CREATE TRIGGER IF NOT EXISTS skills_au AFTER UPDATE ON skills BEGIN
        INSERT INTO skills_fts(skills_fts, rowid, name, description, tags) VALUES('delete', old.rowid, old.name, old.description, old.tags);
        INSERT INTO skills_fts(rowid, name, description, tags) VALUES (new.rowid, new.name, new.description, new.tags);
      END
    `)
  }

  publish(metadata: SkillMetadata, content: string, publishedBy?: string): string {
    const id = metadata.id || randomUUID()
    const tags = JSON.stringify(metadata.tags ?? [])
    const tools = JSON.stringify(metadata.tools ?? [])

    this.db.prepare(`
      INSERT OR REPLACE INTO skills (id, name, version, description, author, tags, tools, autonomy_required, content, published_by, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    `).run(
      id,
      metadata.name,
      metadata.version,
      metadata.description,
      metadata.author ?? null,
      tags,
      tools,
      metadata.autonomy_required ?? null,
      content,
      publishedBy ?? null,
    )

    return id
  }

  get(id: string): StoredSkill | null {
    const row = this.db.prepare(`
      SELECT * FROM skills WHERE id = ?
    `).get(id) as SkillRow | undefined

    if (!row) return null

    // Increment download count
    this.db.prepare('UPDATE skills SET downloads = downloads + 1 WHERE id = ?').run(id)

    return this.rowToSkill(row)
  }

  search(query: string, limit: number = 20): SkillMetadata[] {
    if (!query.trim()) return []

    try {
      const escaped = query
        .split(/\s+/)
        .filter(Boolean)
        .map(w => `"${w.replace(/"/g, '""')}"`)
        .join(' ')

      const rows = this.db.prepare(`
        SELECT s.* FROM skills_fts f JOIN skills s ON s.rowid = f.rowid
        WHERE skills_fts MATCH ? ORDER BY rank LIMIT ?
      `).all(escaped, limit) as SkillRow[]

      return rows.map(r => this.rowToMetadata(r))
    } catch {
      // FTS5 fallback
      const rows = this.db.prepare(`
        SELECT * FROM skills WHERE name LIKE ? OR description LIKE ? LIMIT ?
      `).all(`%${query}%`, `%${query}%`, limit) as SkillRow[]

      return rows.map(r => this.rowToMetadata(r))
    }
  }

  list(page: number = 1, pageSize: number = 20): { skills: SkillMetadata[]; total: number } {
    const offset = (page - 1) * pageSize
    const total = (this.db.prepare('SELECT COUNT(*) as count FROM skills').get() as SkillCountRow).count

    const rows = this.db.prepare(`
      SELECT * FROM skills ORDER BY downloads DESC, updated_at DESC LIMIT ? OFFSET ?
    `).all(pageSize, offset) as SkillRow[]

    return {
      skills: rows.map(r => this.rowToMetadata(r)),
      total,
    }
  }

  rate(id: string, score: number): void {
    if (score < 1 || score > 5) return

    const row = this.db.prepare('SELECT rating, rating_count FROM skills WHERE id = ?').get(id) as SkillRatingRow | undefined
    if (!row) return

    const newCount = (row.rating_count || 0) + 1
    const newRating = ((row.rating || 0) * (row.rating_count || 0) + score) / newCount

    this.db.prepare('UPDATE skills SET rating = ?, rating_count = ? WHERE id = ?').run(newRating, newCount, id)
  }

  delete(id: string): boolean {
    const result = this.db.prepare('DELETE FROM skills WHERE id = ?').run(id)
    return result.changes > 0
  }

  private rowToSkill(row: SkillRow): StoredSkill {
    return {
      metadata: this.rowToMetadata(row),
      content: row.content,
      downloads: row.downloads,
      rating: row.rating,
      publishedBy: row.published_by ?? '',
      publishedAt: row.published_at,
    }
  }

  private rowToMetadata(row: SkillRow): SkillMetadata {
    return {
      id: row.id,
      name: row.name,
      version: row.version,
      description: row.description,
      author: row.author,
      tags: JSON.parse(row.tags || '[]'),
      tools: JSON.parse(row.tools || '[]'),
      autonomy_required: row.autonomy_required as SkillMetadata['autonomy_required'],
    }
  }

  close(): void {
    this.db.close()
  }
}
