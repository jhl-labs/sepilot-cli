import { z } from 'zod'
import { openDomainDb } from '../storage/domain-db.js'
import { knowledgeError } from './schema.js'

export const DailyTokenLimit = z.number().int().min(0).max(100_000_000)
/** UTC ledger: unresolved calls retain their reservation across failures and restarts. */
export function createKnowledgeBudget(now = Date.now) {
  const db = openDomainDb({ name: 'knowledge' })
  db.exec(`CREATE TABLE IF NOT EXISTS knowledge_budget_settings (id INTEGER PRIMARY KEY CHECK(id=1), limit_tokens INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS knowledge_budget_calls (id TEXT PRIMARY KEY, day TEXT NOT NULL, reserved INTEGER NOT NULL, actual INTEGER);
    CREATE INDEX IF NOT EXISTS knowledge_budget_day ON knowledge_budget_calls(day);`)
  const state = () => {
    const day = new Date(now()).toISOString().slice(0, 10)
    const limit =
      (
        db
          .prepare('SELECT limit_tokens AS value FROM knowledge_budget_settings WHERE id=1')
          .get() as { value: number } | undefined
      )?.value ?? 50_000
    const usage = db
      .prepare(
        `SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) AS chargedTokens,
      COALESCE(SUM(actual),0) AS measuredTokens, COALESCE(SUM(CASE WHEN actual IS NULL THEN reserved ELSE 0 END),0) AS reservedTokens
      FROM knowledge_budget_calls WHERE day=?`,
      )
      .get(day) as { chargedTokens: number; measuredTokens: number; reservedTokens: number }
    return { day, limit, ...usage, remainingTokens: Math.max(0, limit - usage.chargedTokens) }
  }
  return {
    state,
    backfill() {
      // Existing automatic calls from before the ledger still count on their original UTC day.
      db.exec(`INSERT OR IGNORE INTO knowledge_budget_calls(id,day,reserved,actual)
        SELECT c.id, strftime('%Y-%m-%d',c.started_at/1000,'unixepoch'),
          COALESCE(json_extract(c.data,'$.requestChars'),0)*4+3500,
          CASE WHEN c.input_tokens IS NOT NULL AND c.output_tokens IS NOT NULL AND c.input_tokens+c.output_tokens>0
            THEN c.input_tokens+c.output_tokens ELSE NULL END
        FROM knowledge_llm_calls c JOIN knowledge_activity a ON a.id=c.activity_id
        WHERE json_extract(a.data,'$.trigger')='automatic'`)
    },
    configure(limit: number) {
      db.prepare(
        'INSERT INTO knowledge_budget_settings VALUES (1,?) ON CONFLICT(id) DO UPDATE SET limit_tokens=excluded.limit_tokens',
      ).run(DailyTokenLimit.parse(limit))
      return state()
    },
    reserve(id: string, tokens: number) {
      return db.transaction(() => {
        const current = state()
        if (!Number.isSafeInteger(tokens) || tokens <= 0 || tokens > current.remainingTokens)
          knowledgeError(
            '자동 Wiki 작업의 일일 토큰 예산이 부족합니다. 예산을 조정하거나 다음 UTC 날짜에 다시 시도하세요.',
            429,
          )
        db.prepare('INSERT INTO knowledge_budget_calls VALUES (?,?,?,NULL)').run(
          id,
          current.day,
          tokens,
        )
        return tokens
      })()
    },
    settle(id: string, actual: number) {
      if (!Number.isFinite(actual) || actual <= 0) return
      db.prepare('UPDATE knowledge_budget_calls SET actual=? WHERE id=?').run(Math.ceil(actual), id)
    },
  }
}
