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
    // Additive migration for databases created before cost_known existed.
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
    // Compute cost from the maintained price table when the caller did not
    // supply an explicit cost. Unknown models stay honest: cost 0 with
    // cost_known=0 (never a fabricated known $0). Pure tool-call rows (no
    // tokens) are not billable and stay cost_known=0.
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
        })
        costKnown = computed.costKnown
        costUsd = computed.costKnown ? computed.costUsd : null as unknown as number
      }
    }
    this.db.prepare(`
      INSERT INTO usage_records (id, session_id, provider, model, input_tokens, output_tokens, thinking_tokens, cache_read_tokens, cache_write_tokens, cost_usd, cost_known, duration_ms, tool_name)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      randomUUID(), usage.sessionId, usage.provider, usage.model,
      usage.inputTokens, usage.outputTokens, usage.thinkingTokens ?? 0,
      usage.cacheReadTokens ?? 0, usage.cacheWriteTokens ?? 0,
      costUsd ?? null, costKnown ? 1 : 0, usage.durationMs ?? null, usage.toolName ?? null,
    )
  }

  /**
   * Delete all usage rows for a session. Used by complete session deletion so a
   * deleted session leaves no usage/cost residue behind (GDPR). Returns the
   * number of rows removed.
   */
  deleteBySession(sessionId: string): number {
    const result = this.db.prepare(`DELETE FROM usage_records WHERE session_id = ?`).run(sessionId)
    return Number(result.changes ?? 0)
  }

  /**
   * Delete usage rows older than an ISO cutoff. Used by the retention sweeper.
   * Returns the number of rows removed.
   */
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
    // 이전 구현은 usage.db 의 'sessions' / 'messages' / 'tool_calls' 테이블을
    // COUNT 했는데, 이 DB 의 스키마에는 그 테이블이 존재하지 않아 (initSchema는
    // usage_records 와 daily_summary 만 만든다) Settings → Agent Stats 의
    // Sessions / Messages / Tool calls 가 항상 0 으로 나오는 버그가 있었음.
    // 실제 데이터인 usage_records 에서 직접 집계:
    //   - totalSessions  = distinct session_id 수
    //   - totalToolCalls = tool_name 이 채워진 record (도구 실행 트레이스)
    //   - totalMessages  = tool_name 이 NULL 인 record (LLM 호출 = 1 turn).
    //     사용자 메시지는 별도 JSONL 세션 저널에 있어서 어시스턴트 측
    //     turn 수가 가장 가까운 의미.
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
    // Truncate the WAL back into the main db before closing so the -wal file
    // does not keep growing across restarts and the next boot has no replay
    // cost. Best-effort: a checkpoint failure must not block close.
    try {
      this.db.pragma('wal_checkpoint(TRUNCATE)')
    } catch {
      /* ignore checkpoint failure */
    }
    this.db.close()
  }
}
