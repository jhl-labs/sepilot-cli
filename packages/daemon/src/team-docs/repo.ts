import type { SqliteDatabase } from '../db/sqlite.js'
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { openDomainDb } from '../storage/domain-db.js'
import { sepilotdHome } from '../storage/home.js'
import { assertSafeId } from '../utils/safe-id.js'
import type {
  TeamDocsConfig,
  TeamDocsConfigInput,
  TeamDocsDocument,
  TeamDocsDocumentContent,
} from './schema.js'

interface TeamDocsDocumentInput {
  path: string
  sha: string | null
  size: number
  content: string
}

function teamDocsDir(id: string): string {
  assertSafeId(id, 'team-docs id')
  const root = resolve(join(sepilotdHome(), 'team-docs'))
  const dir = resolve(join(root, id))
  if (!dir.startsWith(`${root}/`) && dir !== root) {
    throw new Error('invalid team-docs id')
  }
  return dir
}

function normalizeDocPath(value: string): string {
  const normalized = value
    .replace(/\\/g, '/')
    .split('/')
    .filter((segment) => segment.length > 0)
    .join('/')
  if (!normalized || normalized.split('/').some((segment) => segment === '..')) {
    throw new Error('Invalid team docs path')
  }
  return normalized
}

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS team_docs_configs (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      description TEXT NOT NULL DEFAULT '',
      server_type TEXT NOT NULL,
      ghes_url TEXT NOT NULL DEFAULT '',
      token TEXT NOT NULL,
      owner TEXT NOT NULL,
      repo TEXT NOT NULL,
      branch TEXT NOT NULL,
      docs_path TEXT NOT NULL,
      enabled INTEGER NOT NULL DEFAULT 1,
      auto_sync INTEGER NOT NULL DEFAULT 0,
      sync_interval INTEGER NOT NULL DEFAULT 60,
      last_tested_at INTEGER,
      last_sync_at INTEGER,
      last_sync_status TEXT,
      last_sync_error TEXT,
      synced_documents INTEGER NOT NULL DEFAULT 0,
      updated_at INTEGER NOT NULL
    )`,
  ).run()
  db.prepare(
    `CREATE TABLE IF NOT EXISTS team_docs_documents (
      team_docs_id TEXT NOT NULL,
      path TEXT NOT NULL,
      sha TEXT,
      size INTEGER NOT NULL DEFAULT 0,
      synced_at INTEGER NOT NULL,
      PRIMARY KEY (team_docs_id, path),
      FOREIGN KEY (team_docs_id) REFERENCES team_docs_configs(id) ON DELETE CASCADE
    )`,
  ).run()
  db.prepare(
    'CREATE INDEX IF NOT EXISTS team_docs_documents_team_idx ON team_docs_documents(team_docs_id, path)',
  ).run()
}

function mapConfigRow(row: {
  id: string
  name: string
  description: string
  server_type: 'github.com' | 'ghes'
  ghes_url: string
  token: string
  owner: string
  repo: string
  branch: string
  docs_path: string
  enabled: number
  auto_sync: number
  sync_interval: number
  last_tested_at: number | null
  last_sync_at: number | null
  last_sync_status: 'success' | 'error' | null
  last_sync_error: string | null
  synced_documents: number
  updated_at: number
}): TeamDocsConfig {
  return {
    id: row.id,
    name: row.name,
    description: row.description,
    serverType: row.server_type,
    ghesUrl: row.ghes_url,
    token: row.token,
    owner: row.owner,
    repo: row.repo,
    branch: row.branch,
    docsPath: row.docs_path,
    enabled: Boolean(row.enabled),
    autoSync: Boolean(row.auto_sync),
    syncInterval: row.sync_interval,
    lastTestedAt: row.last_tested_at,
    lastSyncAt: row.last_sync_at,
    lastSyncStatus: row.last_sync_status,
    lastSyncError: row.last_sync_error,
    syncedDocuments: row.synced_documents,
    updatedAt: row.updated_at,
  }
}

export interface TeamDocsRepo {
  list(): TeamDocsConfig[]
  get(id: string): TeamDocsConfig | null
  upsert(input: TeamDocsConfigInput): TeamDocsConfig
  remove(id: string): void
  markTested(id: string): TeamDocsConfig | null
  markSyncResult(input: {
    id: string
    status: 'success' | 'error'
    error?: string | null
    syncedDocuments?: number
  }): TeamDocsConfig | null
  replaceDocuments(
    id: string,
    documents: TeamDocsDocumentInput[],
  ): TeamDocsDocument[]
  listDocuments(id: string, limit?: number): TeamDocsDocument[]
  readDocument(id: string, path: string): TeamDocsDocumentContent | null
}

export function createTeamDocsRepo(): TeamDocsRepo {
  const db = openDomainDb({ name: 'team-docs' })
  ensureSchema(db)

  return {
    list() {
      return (
        db
          .prepare(
            `SELECT
              id, name, description, server_type, ghes_url, token, owner, repo,
              branch, docs_path, enabled, auto_sync, sync_interval,
              last_tested_at, last_sync_at, last_sync_status, last_sync_error,
              synced_documents, updated_at
             FROM team_docs_configs
             ORDER BY updated_at DESC, name COLLATE NOCASE`,
          )
          .all() as Array<{
          id: string
          name: string
          description: string
          server_type: 'github.com' | 'ghes'
          ghes_url: string
          token: string
          owner: string
          repo: string
          branch: string
          docs_path: string
          enabled: number
          auto_sync: number
          sync_interval: number
          last_tested_at: number | null
          last_sync_at: number | null
          last_sync_status: 'success' | 'error' | null
          last_sync_error: string | null
          synced_documents: number
          updated_at: number
        }>
      ).map(mapConfigRow)
    },
    get(id) {
      const row = db
        .prepare(
          `SELECT
            id, name, description, server_type, ghes_url, token, owner, repo,
            branch, docs_path, enabled, auto_sync, sync_interval,
            last_tested_at, last_sync_at, last_sync_status, last_sync_error,
            synced_documents, updated_at
           FROM team_docs_configs
           WHERE id = ?`,
        )
        .get(id) as
        | {
            id: string
            name: string
            description: string
            server_type: 'github.com' | 'ghes'
            ghes_url: string
            token: string
            owner: string
            repo: string
            branch: string
            docs_path: string
            enabled: number
            auto_sync: number
            sync_interval: number
            last_tested_at: number | null
            last_sync_at: number | null
            last_sync_status: 'success' | 'error' | null
            last_sync_error: string | null
            synced_documents: number
            updated_at: number
          }
        | undefined
      return row ? mapConfigRow(row) : null
    },
    upsert(input) {
      const id = input.id ?? crypto.randomUUID()
      const now = Date.now()
      db.prepare(
        `INSERT INTO team_docs_configs (
          id, name, description, server_type, ghes_url, token, owner, repo,
          branch, docs_path, enabled, auto_sync, sync_interval, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          name=excluded.name,
          description=excluded.description,
          server_type=excluded.server_type,
          ghes_url=excluded.ghes_url,
          token=excluded.token,
          owner=excluded.owner,
          repo=excluded.repo,
          branch=excluded.branch,
          docs_path=excluded.docs_path,
          enabled=excluded.enabled,
          auto_sync=excluded.auto_sync,
          sync_interval=excluded.sync_interval,
          updated_at=excluded.updated_at`,
      ).run(
        id,
        input.name,
        input.description,
        input.serverType,
        input.ghesUrl,
        input.token,
        input.owner,
        input.repo,
        input.branch,
        input.docsPath,
        input.enabled ? 1 : 0,
        input.autoSync ? 1 : 0,
        input.syncInterval,
        now,
      )
      return this.get(id) as TeamDocsConfig
    },
    remove(id) {
      db.prepare('DELETE FROM team_docs_documents WHERE team_docs_id = ?').run(id)
      db.prepare('DELETE FROM team_docs_configs WHERE id = ?').run(id)
      rmSync(teamDocsDir(id), { recursive: true, force: true })
    },
    markTested(id) {
      db.prepare(
        'UPDATE team_docs_configs SET last_tested_at = ?, updated_at = ? WHERE id = ?',
      ).run(Date.now(), Date.now(), id)
      return this.get(id)
    },
    markSyncResult({ id, status, error = null, syncedDocuments }) {
      const current = this.get(id)
      if (!current) return null
      db.prepare(
        `UPDATE team_docs_configs
         SET last_sync_at = ?, last_sync_status = ?, last_sync_error = ?,
             synced_documents = ?, updated_at = ?
         WHERE id = ?`,
      ).run(
        Date.now(),
        status,
        error,
        syncedDocuments ?? current.syncedDocuments,
        Date.now(),
        id,
      )
      return this.get(id)
    },
    replaceDocuments(id, documents) {
      const root = teamDocsDir(id)
      rmSync(root, { recursive: true, force: true })
      mkdirSync(root, { recursive: true })
      db.prepare('DELETE FROM team_docs_documents WHERE team_docs_id = ?').run(id)
      const insert = db.prepare(
        `INSERT INTO team_docs_documents (team_docs_id, path, sha, size, synced_at)
         VALUES (?, ?, ?, ?, ?)`,
      )
      const syncedAt = Date.now()
      for (const document of documents) {
        const relativePath = normalizeDocPath(document.path)
        const target = join(root, relativePath)
        mkdirSync(dirname(target), { recursive: true })
        writeFileSync(target, document.content, 'utf-8')
        insert.run(
          id,
          relativePath,
          document.sha,
          document.size,
          syncedAt,
        )
      }
      return this.listDocuments(id, documents.length || 100)
    },
    listDocuments(id, limit = 50) {
      return (
        db
          .prepare(
            `SELECT path, sha, size, synced_at
             FROM team_docs_documents
             WHERE team_docs_id = ?
             ORDER BY path COLLATE NOCASE
             LIMIT ?`,
          )
          .all(id, limit) as Array<{
          path: string
          sha: string | null
          size: number
          synced_at: number
        }>
      ).map((row) => ({
        path: row.path,
        sha: row.sha,
        size: row.size,
        syncedAt: row.synced_at,
      }))
    },
    readDocument(id, path) {
      const normalizedPath = normalizeDocPath(path)
      const row = db
        .prepare(
          `SELECT path, sha, size, synced_at
           FROM team_docs_documents
           WHERE team_docs_id = ? AND path = ?`,
        )
        .get(id, normalizedPath) as
        | {
            path: string
            sha: string | null
            size: number
            synced_at: number
          }
        | undefined
      if (!row) return null
      try {
        const content = readFileSync(
          join(teamDocsDir(id), normalizedPath),
          'utf-8',
        )
        return {
          path: row.path,
          sha: row.sha,
          size: row.size,
          syncedAt: row.synced_at,
          content,
        }
      } catch {
        return null
      }
    },
  }
}
