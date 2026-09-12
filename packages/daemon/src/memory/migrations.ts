import type { SqliteDatabase } from '../db/sqlite.js'

interface Migration {
  version: number
  description: string
  up(db: SqliteDatabase): void
}

const MIGRATIONS: Migration[] = [
  {
    version: 1,
    description: 'Create base memory schema',
    up(db) {
      db.exec(`
        CREATE TABLE IF NOT EXISTS memories (
          id TEXT PRIMARY KEY,
          content TEXT NOT NULL,
          source TEXT,
          tags TEXT,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now')),
          session_id TEXT,
          metadata TEXT
        )
      `)

      db.exec(`
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
          content, tags, content=memories, content_rowid=rowid
        )
      `)

      db.exec(`
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
          INSERT INTO memories_fts(rowid, content, tags) VALUES (new.rowid, new.content, new.tags);
        END
      `)
      db.exec(`
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
          INSERT INTO memories_fts(memories_fts, rowid, content, tags) VALUES('delete', old.rowid, old.content, old.tags);
        END
      `)
      db.exec(`
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
          INSERT INTO memories_fts(memories_fts, rowid, content, tags) VALUES('delete', old.rowid, old.content, old.tags);
          INSERT INTO memories_fts(rowid, content, tags) VALUES (new.rowid, new.content, new.tags);
        END
      `)
    },
  },
  {
    version: 2,
    description: 'Add importance column to memories',
    up(db) {
      try {
        db.prepare('ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 0.5').run()
      } catch {
        // Column may already exist.
      }
    },
  },
  {
    version: 3,
    description: 'Add semantic embedding cache columns to memories',
    up(db) {
      const statements = [
        'ALTER TABLE memories ADD COLUMN embedding BLOB',
        'ALTER TABLE memories ADD COLUMN embedding_model TEXT',
        "ALTER TABLE memories ADD COLUMN embedding_state TEXT NOT NULL DEFAULT 'disabled'",
        'ALTER TABLE memories ADD COLUMN embedding_error TEXT',
        'ALTER TABLE memories ADD COLUMN embedding_updated_at TEXT',
      ]

      for (const statement of statements) {
        try {
          db.prepare(statement).run()
        } catch {
          // Column may already exist.
        }
      }

      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memories_embedding_model
        ON memories(embedding_model)
      `).run()
    },
  },
  {
    version: 4,
    description: 'Create semantic index state table',
    up(db) {
      db.exec(`
        CREATE TABLE IF NOT EXISTS semantic_index_state (
          singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
          provider_id TEXT,
          model TEXT,
          dimensions INTEGER,
          status TEXT NOT NULL DEFAULT 'disabled',
          last_error TEXT,
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
      `)

      db.prepare(`
        INSERT INTO semantic_index_state (
          singleton,
          status,
          updated_at
        )
        VALUES (1, 'disabled', datetime('now'))
        ON CONFLICT(singleton) DO NOTHING
      `).run()
    },
  },
  {
    version: 5,
    description: 'Add document memory tables and chunk metadata',
    up(db) {
      const statements = [
        'ALTER TABLE memories ADD COLUMN document_id TEXT',
        'ALTER TABLE memories ADD COLUMN chunk_index INTEGER',
        'ALTER TABLE memories ADD COLUMN chunk_count INTEGER',
        'ALTER TABLE memories ADD COLUMN chunk_start INTEGER',
        'ALTER TABLE memories ADD COLUMN chunk_end INTEGER',
        'ALTER TABLE memories ADD COLUMN chunk_title TEXT',
      ]

      for (const statement of statements) {
        try {
          db.prepare(statement).run()
        } catch {
          // Column may already exist.
        }
      }

      db.exec(`
        CREATE TABLE IF NOT EXISTS memory_documents (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          path TEXT,
          mime_type TEXT,
          source_file_id TEXT,
          tags TEXT,
          chunk_count INTEGER NOT NULL DEFAULT 0,
          content TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
      `)

      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memories_document_id
        ON memories(document_id)
      `).run()

      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memories_source_document
        ON memories(source, document_id)
      `).run()
    },
  },
  {
    version: 6,
    description: 'Add semantic memory audit trail',
    up(db) {
      db.exec(`
        CREATE TABLE IF NOT EXISTS memory_audit (
          id TEXT PRIMARY KEY,
          memory_id TEXT NOT NULL,
          action TEXT NOT NULL,
          actor TEXT NOT NULL,
          reason TEXT,
          before_json TEXT,
          after_json TEXT,
          created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
      `)

      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memory_audit_memory_created
        ON memory_audit(memory_id, created_at DESC)
      `).run()

      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memory_audit_created
        ON memory_audit(created_at DESC)
      `).run()
    },
  },
  {
    version: 7,
    description: 'Add memory access counter columns + indexes',
    up(db) {
      const statements = [
        'ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0',
        'ALTER TABLE memories ADD COLUMN last_accessed_at TEXT',
      ]
      for (const statement of statements) {
        try {
          db.prepare(statement).run()
        } catch {
          // Column may already exist.
        }
      }
      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memories_access_count
        ON memories(access_count DESC)
      `).run()
      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memories_last_accessed_at
        ON memories(last_accessed_at DESC)
      `).run()
    },
  },
  {
    version: 8,
    description: 'Add pinned flag for prune-immune memories',
    up(db) {
      try {
        db.prepare('ALTER TABLE memories ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0').run()
      } catch {
        // Column may already exist.
      }
      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memories_pinned
        ON memories(pinned)
      `).run()
    },
  },
  {
    version: 9,
    description: 'Add durable memory knowledge graph tables',
    up(db) {
      db.exec(`
        CREATE TABLE IF NOT EXISTS memory_graph_nodes (
          id TEXT PRIMARY KEY,
          label TEXT NOT NULL,
          kind TEXT NOT NULL,
          aliases TEXT NOT NULL DEFAULT '[]',
          tags TEXT NOT NULL DEFAULT '[]',
          evidence_memory_ids TEXT NOT NULL DEFAULT '[]',
          confidence REAL NOT NULL DEFAULT 0.5,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now')),
          last_seen_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
      `)

      db.exec(`
        CREATE TABLE IF NOT EXISTS memory_graph_edges (
          id TEXT PRIMARY KEY,
          from_node_id TEXT NOT NULL,
          to_node_id TEXT NOT NULL,
          relation TEXT NOT NULL,
          tags TEXT NOT NULL DEFAULT '[]',
          evidence_memory_ids TEXT NOT NULL DEFAULT '[]',
          confidence REAL NOT NULL DEFAULT 0.5,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now')),
          last_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
          FOREIGN KEY (from_node_id) REFERENCES memory_graph_nodes(id) ON DELETE CASCADE,
          FOREIGN KEY (to_node_id) REFERENCES memory_graph_nodes(id) ON DELETE CASCADE
        )
      `)

      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memory_graph_nodes_label_lc
        ON memory_graph_nodes(lower(label))
      `).run()
      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memory_graph_nodes_kind
        ON memory_graph_nodes(kind)
      `).run()
      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memory_graph_nodes_last_seen
        ON memory_graph_nodes(last_seen_at DESC)
      `).run()
      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memory_graph_edges_from
        ON memory_graph_edges(from_node_id)
      `).run()
      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memory_graph_edges_to
        ON memory_graph_edges(to_node_id)
      `).run()
      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memory_graph_edges_relation
        ON memory_graph_edges(relation)
      `).run()
      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memory_graph_edges_last_seen
        ON memory_graph_edges(last_seen_at DESC)
      `).run()
    },
  },
  {
    version: 10,
    description: 'Add memory_documents content hash for ingest dedup',
    up(db) {
      try {
        db.prepare('ALTER TABLE memory_documents ADD COLUMN content_hash TEXT').run()
      } catch {
        // Column may already exist.
      }
      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memory_documents_content_hash
        ON memory_documents(content_hash)
      `).run()
    },
  },
  {
    version: 11,
    description: 'Add created_at and embedding-backfill indexes for large stores',
    up(db) {
      // listRecent orders by created_at; without this a large store full-scans.
      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memories_created_at
        ON memories(created_at DESC)
      `).run()
      // The backfill selector scans for rows still needing an embedding; a
      // partial index over the un-embedded rows keeps each pass ~O(pending)
      // instead of O(n) (near-O(n^2) across many passes on a big store).
      db.prepare(`
        CREATE INDEX IF NOT EXISTS idx_memories_embedding_backfill
        ON memories(updated_at)
        WHERE embedding IS NULL
      `).run()
    },
  },
  {
    version: 12,
    description: 'Add bounded-retry bookkeeping for failed embeddings',
    up(db) {
      try {
        db.prepare('ALTER TABLE memories ADD COLUMN embedding_attempts INTEGER NOT NULL DEFAULT 0').run()
      } catch {
        // Column may already exist.
      }
      try {
        db.prepare('ALTER TABLE memories ADD COLUMN embedding_next_retry_at TEXT').run()
      } catch {
        // Column may already exist.
      }
    },
  },
  {
    version: 13,
    description: 'Add base_importance so importance recompute is deterministic (no ratchet saturation)',
    up(db) {
      // The old recompute did `importance = importance + age + access` every
      // run, which ratcheted to 1.0 within days and made the score
      // non-discriminating. Deterministic recompute needs a stable base to add
      // recency/access terms to. Seed it from the current importance so existing
      // stores keep their assigned baseline.
      try {
        db.prepare('ALTER TABLE memories ADD COLUMN base_importance REAL').run()
      } catch {
        // Column may already exist.
      }
      db.prepare(
        'UPDATE memories SET base_importance = COALESCE(importance, 0.5) WHERE base_importance IS NULL',
      ).run()
    },
  },
  {
    version: 14,
    description: 'Durable memory projection outbox and conservative graph invalidation',
    up(db) {
      db.exec(`
        CREATE TABLE memory_projection_changes (
          sequence INTEGER PRIMARY KEY AUTOINCREMENT,
          memory_id TEXT NOT NULL, old_content TEXT NOT NULL,
          old_tags TEXT NOT NULL, new_content TEXT
        );
        CREATE TABLE memory_retractions (
          memory_id TEXT PRIMARY KEY, content TEXT NOT NULL, tags TEXT NOT NULL, source_ids TEXT NOT NULL DEFAULT '[]',
          retracted_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TRIGGER memory_projection_update AFTER UPDATE OF content, tags, metadata ON memories
        WHEN old.content != new.content OR (COALESCE((SELECT group_concat(value, '|') FROM (SELECT value FROM json_each(CASE WHEN json_valid(old.tags) THEN old.tags ELSE '[]' END) WHERE value LIKE 'scope:%' ORDER BY value)), '') != COALESCE((SELECT group_concat(value, '|') FROM (SELECT value FROM json_each(CASE WHEN json_valid(new.tags) THEN new.tags ELSE '[]' END) WHERE value LIKE 'scope:%' ORDER BY value)), ''))
          OR COALESCE(json_extract(CASE WHEN json_valid(new.metadata) THEN new.metadata ELSE '{}' END, '$.memory.status'), 'active') = 'retracted'
        BEGIN
          INSERT INTO memory_projection_changes(memory_id, old_content, old_tags, new_content)
          VALUES(old.id, old.content, CASE WHEN json_valid(old.tags) THEN old.tags ELSE '[]' END,
            CASE WHEN (COALESCE((SELECT group_concat(value, '|') FROM (SELECT value FROM json_each(CASE WHEN json_valid(old.tags) THEN old.tags ELSE '[]' END) WHERE value LIKE 'scope:%' ORDER BY value)), '') != COALESCE((SELECT group_concat(value, '|') FROM (SELECT value FROM json_each(CASE WHEN json_valid(new.tags) THEN new.tags ELSE '[]' END) WHERE value LIKE 'scope:%' ORDER BY value)), '')) OR COALESCE(json_extract(CASE WHEN json_valid(new.metadata) THEN new.metadata ELSE '{}' END, '$.memory.status'), 'active') != 'active' THEN NULL ELSE new.content END);
          INSERT OR REPLACE INTO memory_retractions(memory_id, content, tags, source_ids)
            VALUES(old.id, old.content, CASE WHEN json_valid(old.tags) THEN old.tags ELSE '[]' END, COALESCE(json_extract(CASE WHEN json_valid(old.metadata) THEN old.metadata ELSE '{}' END, '$.memory.sourceIds'), '[]'));
          DELETE FROM memory_graph_edges WHERE id IN (
            SELECT e.id FROM memory_graph_edges e, json_each(e.evidence_memory_ids) j WHERE j.value = old.id
          );
          DELETE FROM memory_graph_nodes WHERE id IN (
            SELECT n.id FROM memory_graph_nodes n, json_each(n.evidence_memory_ids) j WHERE j.value = old.id
          );
        END;
        CREATE TRIGGER memory_projection_delete AFTER DELETE ON memories BEGIN
          INSERT INTO memory_projection_changes(memory_id, old_content, old_tags, new_content)
            VALUES(old.id, old.content, CASE WHEN json_valid(old.tags) THEN old.tags ELSE '[]' END, NULL);
          INSERT OR REPLACE INTO memory_retractions(memory_id, content, tags, source_ids)
            VALUES(old.id, old.content, CASE WHEN json_valid(old.tags) THEN old.tags ELSE '[]' END, COALESCE(json_extract(CASE WHEN json_valid(old.metadata) THEN old.metadata ELSE '{}' END, '$.memory.sourceIds'), '[]'));
          DELETE FROM memory_graph_edges WHERE id IN (
            SELECT e.id FROM memory_graph_edges e, json_each(e.evidence_memory_ids) j WHERE j.value = old.id
          );
          DELETE FROM memory_graph_nodes WHERE id IN (
            SELECT n.id FROM memory_graph_nodes n, json_each(n.evidence_memory_ids) j WHERE j.value = old.id
          );
        END;
      `)
    },
  },
  {
    version: 15,
    description: 'Persist owner memory reset boundaries',
    up(db) {
      db.exec(`CREATE TABLE memory_resets (owner TEXT PRIMARY KEY, reset_at TEXT NOT NULL)`)
    },
  },
]

export function runMigrations(db: SqliteDatabase): { from: number; to: number; applied: string[] } {
  const currentVersion = (db.pragma('user_version') as Array<{ user_version: number }>)[0]?.user_version ?? 0
  const applied: string[] = []

  for (const migration of MIGRATIONS) {
    if (migration.version <= currentVersion) continue
    migration.up(db)
    db.pragma(`user_version = ${migration.version}`)
    applied.push(`v${migration.version}: ${migration.description}`)
  }

  const newVersion = MIGRATIONS[MIGRATIONS.length - 1]?.version ?? 0
  return { from: currentVersion, to: newVersion, applied }
}
