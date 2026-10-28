import { mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'
import { randomUUID } from 'node:crypto'
import { openDatabase, type SqliteDatabase } from '../db/sqlite.js'
import { recoverSqliteDatabaseIfCorrupt } from './sqlite-recovery.js'
import { computeCostUsd } from '../providers/pricing.js'

export interface UsageRecord {
  sessionId: string
  provider: string
  model: string
  inputTokens: number
  outputTokens: number
  thinkingTokens?: number
  cacheReadTokens?: number
  cacheWriteTokens?: number
  inputTokensIncludeCache?: boolean
  firstTokenMs?: number
  costUsd?: number
  costKnown?: boolean
  durationMs?: number
  toolName?: string
}

export interface DailySummary {
  date: string
  provider: string
  model: string
  totalInputTokens: number
  totalOutputTokens: number
  totalCostUsd: number
  requestCount: number
}

export interface ProviderUsageSummary {
  provider: string
  model: string
  inputTokens: number
  outputTokens: number
  costUsd: number
  requestCount: number
}

interface UsageAggregateRow {
  input_tokens: number
  output_tokens: number
  cost_usd: number
  request_count: number
}

interface UsageCountRow {
  c: number
}

export class UsageTracker {
  private db: SqliteDatabase

  constructor(dbPath: string) {
    this.db = openDatabase(dbPath)
    this.db.pragma('journal_mode = WAL')
    this.initSchema()
  }

  static async create(dbPath: string): Promise<UsageTracker> {
    await mkdir(dirname(dbPath), { recursive: true })
    recoverSqliteDatabaseIfCorrupt(dbPath)
    return new UsageTracker(dbPath)
  }

  private initSchema(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS usage_records (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        input_tokens INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL,
        thinking_tokens INTEGER DEFAULT 0,
        cache_read_tokens INTEGER DEFAULT 0,
        cache_write_tokens INTEGER DEFAULT 0,
        cost_usd REAL,
        cost_known INTEGER DEFAULT 0,
        duration_ms INTEGER,
        tool_name TEXT,
        metadata TEXT
      )
    `)
    this.ensureColumn('usage_records', 'cost_known', 'INTEGER DEFAULT 0')
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS daily_summary (
        date TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        total_input_tokens INTEGER,
        total_output_tokens INTEGER,
        total_cost_usd REAL,
        request_count INTEGER,
        PRIMARY KEY (date, provider, model)
      )
    `)
    this.db.exec(`CREATE INDEX IF NOT EXISTS idx_usage_session ON usage_records(session_id)`)
    this.db.exec(`CREATE INDEX IF NOT EXISTS idx_usage_date ON usage_records(timestamp)`)
    this.db.exec(`CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage_records(provider, model)`)
  }

  private ensureColumn(table: string, column: string, definition: string): void {
    interface ColumnRow {
      name: string
    }
    const columns = this.db.prepare(`PRAGMA table_info(${table})`).all() as ColumnRow[]
    if (!columns.some((c) => c.name === column)) {
      this.db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition}`)
    }
  }

  record(usage: UsageRecord): void {
    let costUsd = usage.costUsd
    let costKnown = usage.costKnown ?? false
    if (costUsd === undefined) {
      const hasTokens =
        usage.inputTokens > 0 ||
        usage.outputTokens > 0 ||
        (usage.cacheReadTokens ?? 0) > 0 ||
        (usage.cacheWriteTokens ?? 0) > 0 ||
        (usage.thinkingTokens ?? 0) > 0
      if (hasTokens) {
        const computed = computeCostUsd(usage.provider, usage.model, {
          inputTokens: usage.inputTokens,
          outputTokens: usage.outputTokens,
          cacheReadTokens: usage.cacheReadTokens,
          cacheCreationTokens: usage.cacheWriteTokens,
          thinkingTokens: usage.thinkingTokens,
          inputTokensIncludeCache: usage.inputTokensIncludeCache,
        })
        costKnown = computed.costKnown
        costUsd = computed.costKnown ? computed.costUsd : null as unknown as number
      }
    }
    this.db.prepare(`
      INSERT INTO usage_records (id, session_id, provider, model, input_tokens, output_tokens, thinking_tokens, cache_read_tokens, cache_write_tokens, cost_usd, cost_known, duration_ms, tool_name, metadata)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      randomUUID(), usage.sessionId, usage.provider, usage.model,
      usage.inputTokens, usage.outputTokens, usage.thinkingTokens ?? 0,
      usage.cacheReadTokens ?? 0, usage.cacheWriteTokens ?? 0,
      costUsd ?? null, costKnown ? 1 : 0, usage.durationMs ?? null, usage.toolName ?? null,
      JSON.stringify({ inputTokensIncludeCache: usage.inputTokensIncludeCache, firstTokenMs: usage.firstTokenMs,
        cacheUsageReported: usage.cacheReadTokens !== undefined || usage.cacheWriteTokens !== undefined }),
    )
  }

  getCacheSummary(sessionId?: string): { reportedRequests: number; cacheReadTokens: number; cacheWriteTokens: number; readHitRatio: number | null; averageFirstTokenMs: number | null } {
    const row = this.db.prepare(`
      WITH samples AS (
        SELECT input_tokens, cache_read_tokens, cache_write_tokens,
          CASE WHEN json_valid(metadata) THEN metadata ELSE '{}' END AS details
        FROM usage_records ${sessionId === undefined ? '' : 'WHERE session_id = ?'}
      ), observed AS (
        SELECT *, json_extract(details, '$.cacheUsageReported') AS measured,
          json_extract(details, '$.inputTokensIncludeCache') AS inclusive
        FROM samples
      )
      SELECT COALESCE(SUM(CASE WHEN measured = 1 OR cache_read_tokens > 0 OR cache_write_tokens > 0 THEN 1 ELSE 0 END), 0) AS reported,
        COALESCE(SUM(cache_read_tokens), 0) AS reads, COALESCE(SUM(cache_write_tokens), 0) AS writes,
        COALESCE(SUM(CASE WHEN measured = 1 THEN cache_read_tokens ELSE 0 END), 0) AS measured_reads,
        SUM(CASE WHEN measured = 1 THEN input_tokens +
          CASE WHEN inclusive = 1 THEN 0 ELSE cache_read_tokens + cache_write_tokens END END) AS total,
        AVG(json_extract(details, '$.firstTokenMs')) AS latency
      FROM observed
    `).get(...(sessionId === undefined ? [] : [sessionId])) as { reported: number; reads: number; writes: number; measured_reads: number; total: number | null; latency: number | null }
    return { reportedRequests: row.reported, cacheReadTokens: row.reads, cacheWriteTokens: row.writes,
      readHitRatio: row.total === null ? null : row.total > 0 ? row.measured_reads / row.total : 0,
      averageFirstTokenMs: row.latency }
  }

  deleteBySession(sessionId: string): number {
    const result = this.db.prepare(`DELETE FROM usage_records WHERE session_id = ?`).run(sessionId)
    return Number(result.changes ?? 0)
  }

  deleteOlderThan(cutoffIso: string): number {
    const result = this.db.prepare(`DELETE FROM usage_records WHERE timestamp < ?`).run(cutoffIso)
    return Number(result.changes ?? 0)
  }

  getSessionUsage(sessionId: string): { inputTokens: number; outputTokens: number; costUsd: number; requestCount: number } {
    const row = this.db.prepare(`
      SELECT COALESCE(SUM(input_tokens), 0) as input_tokens,
             COALESCE(SUM(output_tokens), 0) as output_tokens,
             COALESCE(SUM(cost_usd), 0) as cost_usd,
             COUNT(*) as request_count
      FROM usage_records WHERE session_id = ?
    `).get(sessionId) as UsageAggregateRow
    return { inputTokens: row.input_tokens, outputTokens: row.output_tokens, costUsd: row.cost_usd, requestCount: row.request_count }
  }

  getDailySummaries(days?: number): DailySummary[] {
    const limit = days ?? 30
    interface DailySummaryRow {
      date: string
      provider: string
      model: string
      total_input_tokens: number | null
      total_output_tokens: number | null
      total_cost_usd: number | null
      request_count: number
    }
    const rows = this.db.prepare(`
      SELECT date(timestamp) as date, provider, model,
             SUM(input_tokens) as total_input_tokens,
             SUM(output_tokens) as total_output_tokens,
             SUM(cost_usd) as total_cost_usd,
             COUNT(*) as request_count
      FROM usage_records
      WHERE timestamp > datetime('now', '-' || ? || ' days')
      GROUP BY date(timestamp), provider, model
      ORDER BY date DESC
    `).all(limit) as DailySummaryRow[]
    return rows.map((row) => ({
      date: row.date,
      provider: row.provider,
      model: row.model,
      totalInputTokens: row.total_input_tokens ?? 0,
      totalOutputTokens: row.total_output_tokens ?? 0,
      totalCostUsd: row.total_cost_usd ?? 0,
      requestCount: row.request_count,
    }))
  }

  getTotalUsage(): { inputTokens: number; outputTokens: number; costUsd: number; requestCount: number } {
    const row = this.db.prepare(`
      SELECT COALESCE(SUM(input_tokens), 0) as input_tokens,
             COALESCE(SUM(output_tokens), 0) as output_tokens,
             COALESCE(SUM(cost_usd), 0) as cost_usd,
             COUNT(*) as request_count
      FROM usage_records
    `).get() as UsageAggregateRow
    return { inputTokens: row.input_tokens, outputTokens: row.output_tokens, costUsd: row.cost_usd, requestCount: row.request_count }
  }

  getProviderSummaries(): ProviderUsageSummary[] {
    interface ProviderUsageRow {
      provider: string
      model: string
      input_tokens: number | null
      output_tokens: number | null
      cost_usd: number | null
      request_count: number
    }

    const rows = this.db.prepare(`
      SELECT provider, model,
             SUM(input_tokens) as input_tokens,
             SUM(output_tokens) as output_tokens,
             SUM(cost_usd) as cost_usd,
             COUNT(*) as request_count
      FROM usage_records
      GROUP BY provider, model
      ORDER BY request_count DESC, input_tokens + output_tokens DESC, provider ASC, model ASC
    `).all() as ProviderUsageRow[]

    return rows.map((row) => ({
      provider: row.provider,
      model: row.model,
      inputTokens: row.input_tokens ?? 0,
      outputTokens: row.output_tokens ?? 0,
      costUsd: row.cost_usd ?? 0,
      requestCount: row.request_count,
    }))
  }

  getSnapshot(): { totalSessions: number; totalMessages: number; totalToolCalls: number } {
    try {
      const sessions = (this.db
        .prepare(
          `SELECT COUNT(DISTINCT session_id) AS c FROM usage_records`,
        )
        .get() as UsageCountRow).c
      const toolCalls = (this.db
        .prepare(
          `SELECT COUNT(*) AS c FROM usage_records WHERE tool_name IS NOT NULL`,
        )
        .get() as UsageCountRow).c
      const messages = (this.db
        .prepare(
          `SELECT COUNT(*) AS c FROM usage_records WHERE tool_name IS NULL`,
        )
        .get() as UsageCountRow).c
      return {
        totalSessions: sessions,
        totalMessages: messages,
        totalToolCalls: toolCalls,
      }
    } catch {
      return { totalSessions: 0, totalMessages: 0, totalToolCalls: 0 }
    }
  }

  close(): void {
    try {
      this.db.pragma('wal_checkpoint(TRUNCATE)')
    } catch {
    }
    this.db.close()
  }
}
