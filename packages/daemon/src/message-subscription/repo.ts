import type { SqliteDatabase } from '../db/sqlite.js'
import { openDomainDb } from '../storage/domain-db.js'
import {
  DEFAULT_MESSAGE_SUBSCRIPTION_CONFIG,
  type MessageQueueStatus,
  type MessageStatus,
  type MessageSubscriptionConfig,
  type MessageSubscriptionItem,
  type SubscriptionStatus,
} from './schema.js'

interface SubscriptionStateRow {
  config_json: string
  is_connected: number
  last_polled: number | null
  last_error: string | null
  updated_at: number
}

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS message_subscription_state (
      id TEXT PRIMARY KEY,
      config_json TEXT NOT NULL,
      is_connected INTEGER NOT NULL DEFAULT 0,
      last_polled INTEGER,
      last_error TEXT,
      updated_at INTEGER NOT NULL
    )`,
  ).run()
  db.prepare(
    `CREATE TABLE IF NOT EXISTS message_subscription_messages (
      hash TEXT PRIMARY KEY,
      external_id TEXT,
      type TEXT NOT NULL,
      source TEXT NOT NULL,
      title TEXT NOT NULL,
      body TEXT NOT NULL,
      content TEXT NOT NULL,
      metadata_json TEXT NOT NULL DEFAULT '{}',
      timestamp INTEGER NOT NULL,
      queued_at INTEGER NOT NULL,
      status TEXT NOT NULL,
      processed_at INTEGER,
      error TEXT,
      retry_count INTEGER NOT NULL DEFAULT 0,
      conversation_id TEXT
    )`,
  ).run()
  db.prepare(
    'CREATE INDEX IF NOT EXISTS message_subscription_messages_status_idx ON message_subscription_messages(status, queued_at DESC)',
  ).run()
}

function readStateRow(
  db: SqliteDatabase,
): SubscriptionStateRow | null {
  return (
    (db
      .prepare(
        `SELECT config_json, is_connected, last_polled, last_error, updated_at
         FROM message_subscription_state
         WHERE id = 'default'`,
      )
      .get() as SubscriptionStateRow | undefined) ?? null
  )
}

function parseConfig(raw: string | null | undefined): MessageSubscriptionConfig {
  if (!raw) return DEFAULT_MESSAGE_SUBSCRIPTION_CONFIG
  try {
    return {
      ...DEFAULT_MESSAGE_SUBSCRIPTION_CONFIG,
      ...JSON.parse(raw),
    } as MessageSubscriptionConfig
  } catch {
    return DEFAULT_MESSAGE_SUBSCRIPTION_CONFIG
  }
}

function mapMessageRow(row: {
  hash: string
  external_id: string | null
  type: 'github_webhook' | 'community_post' | 'custom'
  source: string
  title: string
  body: string
  content: string
  metadata_json: string
  timestamp: number
  queued_at: number
  status: MessageStatus
  processed_at: number | null
  error: string | null
  retry_count: number
  conversation_id: string | null
}): MessageSubscriptionItem {
  return {
    hash: row.hash,
    id: row.external_id,
    type: row.type,
    source: row.source,
    title: row.title,
    body: row.body,
    content: row.content,
    metadata: JSON.parse(row.metadata_json || '{}') as Record<string, unknown>,
    timestamp: row.timestamp,
    queuedAt: row.queued_at,
    status: row.status,
    processedAt: row.processed_at,
    error: row.error,
    retryCount: row.retry_count,
    conversationId: row.conversation_id,
  }
}

export interface MessageSubscriptionRepo {
  getConfig(): MessageSubscriptionConfig
  saveConfig(next: MessageSubscriptionConfig): MessageSubscriptionConfig
  start(): MessageSubscriptionConfig
  stop(): MessageSubscriptionConfig
  getSubscriptionStatus(): SubscriptionStatus
  setSubscriptionStatus(next: Partial<SubscriptionStatus>): SubscriptionStatus
  queueStatus(): MessageQueueStatus
  getMessage(hash: string): MessageSubscriptionItem | null
  listMessages(status?: MessageStatus, limit?: number): MessageSubscriptionItem[]
  insertMessage(input: Omit<MessageSubscriptionItem, 'retryCount'> & {
    retryCount?: number
  }): boolean
  updateMessage(
    hash: string,
    next: Partial<
      Pick<
        MessageSubscriptionItem,
        'status' | 'processedAt' | 'error' | 'retryCount' | 'conversationId'
      >
    >,
  ): MessageSubscriptionItem | null
  reprocess(hash: string): MessageSubscriptionItem | null
  remove(hash: string): void
  cleanup(retentionDays: number): void
  trim(maxQueueSize: number): void
}

export function createMessageSubscriptionRepo(): MessageSubscriptionRepo {
  const db = openDomainDb({ name: 'message-subscription' })
  ensureSchema(db)

  function writeState(input: {
    config: MessageSubscriptionConfig
    isConnected: boolean
    lastPolled: number | null
    lastError: string | null
  }): void {
    db.prepare(
      `INSERT INTO message_subscription_state (
        id, config_json, is_connected, last_polled, last_error, updated_at
      ) VALUES ('default', ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        config_json=excluded.config_json,
        is_connected=excluded.is_connected,
        last_polled=excluded.last_polled,
        last_error=excluded.last_error,
        updated_at=excluded.updated_at`,
    ).run(
      JSON.stringify(input.config),
      input.isConnected ? 1 : 0,
      input.lastPolled,
      input.lastError,
      Date.now(),
    )
  }

  function currentState(): {
    config: MessageSubscriptionConfig
    status: SubscriptionStatus
  } {
    const row = readStateRow(db)
    return {
      config: parseConfig(row?.config_json),
      status: {
        isConnected: Boolean(row?.is_connected),
        lastPolled: row?.last_polled ?? null,
        lastError: row?.last_error ?? null,
      },
    }
  }

  return {
    getConfig() {
      return currentState().config
    },
    saveConfig(next) {
      const state = currentState()
      writeState({
        config: next,
        isConnected: next.enabled ? state.status.isConnected : false,
        lastPolled: state.status.lastPolled,
        lastError: state.status.lastError,
      })
      this.cleanup(next.retentionDays)
      this.trim(next.maxQueueSize)
      return this.getConfig()
    },
    start() {
      return this.saveConfig({
        ...this.getConfig(),
        enabled: true,
      })
    },
    stop() {
      const state = currentState()
      writeState({
        config: {
          ...state.config,
          enabled: false,
        },
        isConnected: false,
        lastPolled: state.status.lastPolled,
        lastError: state.status.lastError,
      })
      return this.getConfig()
    },
    getSubscriptionStatus() {
      return currentState().status
    },
    setSubscriptionStatus(next) {
      const state = currentState()
      writeState({
        config: state.config,
        isConnected: next.isConnected ?? state.status.isConnected,
        lastPolled: next.lastPolled ?? state.status.lastPolled,
        lastError:
          next.lastError === undefined ? state.status.lastError : next.lastError,
      })
      return this.getSubscriptionStatus()
    },
    queueStatus() {
      const counts = new Map<MessageStatus, number>()
      const rows = db
        .prepare(
          `SELECT status, COUNT(*) AS count
           FROM message_subscription_messages
           GROUP BY status`,
        )
        .all() as Array<{ status: MessageStatus; count: number }>
      for (const row of rows) counts.set(row.status, row.count)
      const latest = db
        .prepare(
          `SELECT MAX(processed_at) AS last_processed
           FROM message_subscription_messages
           WHERE processed_at IS NOT NULL`,
        )
        .get() as { last_processed: number | null }
      const state = currentState().status
      return {
        pending: counts.get('pending') ?? 0,
        processing: counts.get('processing') ?? 0,
        completed: counts.get('completed') ?? 0,
        failed: counts.get('failed') ?? 0,
        totalProcessed: (counts.get('completed') ?? 0) + (counts.get('failed') ?? 0),
        lastPolled: state.lastPolled,
        lastProcessed: latest.last_processed ?? null,
      }
    },
    getMessage(hash) {
      const row = db
        .prepare(
          `SELECT
             hash, external_id, type, source, title, body, content,
             metadata_json, timestamp, queued_at, status, processed_at,
             error, retry_count, conversation_id
           FROM message_subscription_messages
           WHERE hash = ?`,
        )
        .get(hash) as
        | {
            hash: string
            external_id: string | null
            type: 'github_webhook' | 'community_post' | 'custom'
            source: string
            title: string
            body: string
            content: string
            metadata_json: string
            timestamp: number
            queued_at: number
            status: MessageStatus
            processed_at: number | null
            error: string | null
            retry_count: number
            conversation_id: string | null
          }
        | undefined
      return row ? mapMessageRow(row) : null
    },
    listMessages(status, limit = 12) {
      const rows = status
        ? (db
            .prepare(
              `SELECT
                 hash, external_id, type, source, title, body, content,
                 metadata_json, timestamp, queued_at, status, processed_at,
                 error, retry_count, conversation_id
               FROM message_subscription_messages
               WHERE status = ?
               ORDER BY queued_at DESC
               LIMIT ?`,
            )
            .all(status, limit) as Array<{
            hash: string
            external_id: string | null
            type: 'github_webhook' | 'community_post' | 'custom'
            source: string
            title: string
            body: string
            content: string
            metadata_json: string
            timestamp: number
            queued_at: number
            status: MessageStatus
            processed_at: number | null
            error: string | null
            retry_count: number
            conversation_id: string | null
          }>)
        : (db
            .prepare(
              `SELECT
                 hash, external_id, type, source, title, body, content,
                 metadata_json, timestamp, queued_at, status, processed_at,
                 error, retry_count, conversation_id
               FROM message_subscription_messages
               ORDER BY queued_at DESC
               LIMIT ?`,
            )
            .all(limit) as Array<{
            hash: string
            external_id: string | null
            type: 'github_webhook' | 'community_post' | 'custom'
            source: string
            title: string
            body: string
            content: string
            metadata_json: string
            timestamp: number
            queued_at: number
            status: MessageStatus
            processed_at: number | null
            error: string | null
            retry_count: number
            conversation_id: string | null
          }>)
      return rows.map(mapMessageRow)
    },
    insertMessage(input) {
      const result = db
        .prepare(
          `INSERT OR IGNORE INTO message_subscription_messages (
             hash, external_id, type, source, title, body, content,
             metadata_json, timestamp, queued_at, status, processed_at,
             error, retry_count, conversation_id
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .run(
          input.hash,
          input.id,
          input.type,
          input.source,
          input.title,
          input.body,
          input.content,
          JSON.stringify(input.metadata),
          input.timestamp,
          input.queuedAt,
          input.status,
          input.processedAt,
          input.error,
          input.retryCount ?? 0,
          input.conversationId,
        )
      return result.changes > 0
    },
    updateMessage(hash, next) {
      const current = db
        .prepare(
          `SELECT
             hash, external_id, type, source, title, body, content,
             metadata_json, timestamp, queued_at, status, processed_at,
             error, retry_count, conversation_id
           FROM message_subscription_messages
           WHERE hash = ?`,
        )
        .get(hash) as
        | {
            hash: string
            external_id: string | null
            type: 'github_webhook' | 'community_post' | 'custom'
            source: string
            title: string
            body: string
            content: string
            metadata_json: string
            timestamp: number
            queued_at: number
            status: MessageStatus
            processed_at: number | null
            error: string | null
            retry_count: number
            conversation_id: string | null
          }
        | undefined
      if (!current) return null
      db.prepare(
        `UPDATE message_subscription_messages
         SET status = ?, processed_at = ?, error = ?, retry_count = ?, conversation_id = ?
         WHERE hash = ?`,
      ).run(
        next.status ?? current.status,
        next.processedAt === undefined ? current.processed_at : next.processedAt,
        next.error === undefined ? current.error : next.error,
        next.retryCount ?? current.retry_count,
        next.conversationId === undefined
          ? current.conversation_id
          : next.conversationId,
        hash,
      )
      return this.getMessage(hash)
    },
    reprocess(hash) {
      return this.updateMessage(hash, {
        status: 'pending',
        processedAt: null,
        error: null,
      })
    },
    remove(hash) {
      db.prepare('DELETE FROM message_subscription_messages WHERE hash = ?').run(hash)
    },
    cleanup(retentionDays) {
      const cutoff = Date.now() - retentionDays * 24 * 60 * 60 * 1000
      db.prepare(
        `DELETE FROM message_subscription_messages
         WHERE status IN ('completed', 'failed')
           AND COALESCE(processed_at, queued_at) < ?`,
      ).run(cutoff)
    },
    trim(maxQueueSize) {
      db.prepare(
        `DELETE FROM message_subscription_messages
         WHERE hash IN (
           SELECT hash
           FROM message_subscription_messages
           ORDER BY queued_at DESC
           LIMIT -1 OFFSET ?
         )`,
      ).run(maxQueueSize)
    },
  }
}
