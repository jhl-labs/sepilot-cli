import { createHash, randomUUID } from 'node:crypto'
import { createKnowledgeBudget } from './budget.js'
import type { ILLMProvider, ChatResponse } from '@sepilotd/core'
import type {
  KnowledgeActivity,
  KnowledgeLlmCall,
  KnowledgeActivityPage,
} from '@sepilotd/api-client/daemon/types'
import { openDomainDb } from '../storage/domain-db.js'
import { redactSensitive } from '../memory/sensitive.js'
import type { UsageTracker } from '../memory/usage-tracker.js'

const preview = (value: unknown) =>
  redactSensitive(typeof value === 'string' ? value : (JSON.stringify(value) ?? '')).redacted.slice(
    0,
    12000,
  )
const number = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
export function createKnowledgeActivityStore() {
  const db = openDomainDb({ name: 'knowledge' })
  const budget = createKnowledgeBudget()
  db.exec(`CREATE TABLE IF NOT EXISTS knowledge_activity (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, status TEXT NOT NULL, data TEXT NOT NULL
  ); CREATE TABLE IF NOT EXISTS knowledge_llm_calls (
    id TEXT PRIMARY KEY, activity_id TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
    started_at INTEGER NOT NULL, input_tokens INTEGER, output_tokens INTEGER, data TEXT NOT NULL,
    FOREIGN KEY(activity_id) REFERENCES knowledge_activity(id)
  ); CREATE INDEX IF NOT EXISTS knowledge_calls_activity ON knowledge_llm_calls(activity_id);`)
  // Additive migration for activity databases created before indexed usage columns.
  const columns = new Set(
    (db.prepare('PRAGMA table_info(knowledge_llm_calls)').all() as { name: string }[]).map(
      (r) => r.name,
    ),
  )
  for (const [name, type] of [
    ['provider', 'TEXT'],
    ['model', 'TEXT'],
    ['started_at', 'INTEGER'],
    ['input_tokens', 'INTEGER'],
    ['output_tokens', 'INTEGER'],
  ]) {
    if (!columns.has(name!)) db.exec(`ALTER TABLE knowledge_llm_calls ADD COLUMN ${name} ${type}`)
  }
  db.exec(`UPDATE knowledge_llm_calls SET provider=json_extract(data,'$.provider'), model=json_extract(data,'$.model'),
    started_at=json_extract(data,'$.startedAt'), input_tokens=json_extract(data,'$.inputTokens'), output_tokens=json_extract(data,'$.outputTokens') WHERE provider IS NULL`)
  budget.backfill()
  const get = (id: string): KnowledgeActivity | undefined => {
    const row = db.prepare('SELECT sequence, data FROM knowledge_activity WHERE id=?').get(id) as
      | { sequence: number; data: string }
      | undefined
    return row ? { ...JSON.parse(row.data), sequence: row.sequence } : undefined
  }
  const patch = (id: string, input: Partial<KnowledgeActivity>) => {
    const existing = get(id)
    if (!existing) throw new Error('Knowledge activity not found')
    const next = { ...existing, ...input }
    db.prepare('UPDATE knowledge_activity SET status=?, data=? WHERE id=?').run(
      next.status,
      JSON.stringify(next),
      id,
    )
    return next
  }
  const event = (id: string, message: string) => {
    const existing = get(id)!
    patch(id, {
      events: [...existing.events, { at: Date.now(), message: preview(message) }].slice(-60),
    })
  }
  const begin = (input: {
    kind: string
    trigger?: KnowledgeActivity['trigger']
    parentId?: string
    queued?: boolean
    summary: string
    targets?: KnowledgeActivity['targets']
  }) => {
    const id = randomUUID()
    const record: KnowledgeActivity = {
      id,
      sequence: 0,
      kind: input.kind,
      trigger: input.trigger ?? 'manual',
      parentId: input.parentId ?? null,
      status: input.queued ? 'queued' : 'running',
      createdAt: Date.now(),
      startedAt: input.queued ? null : Date.now(),
      finishedAt: null,
      summary: preview(input.summary),
      targets: (input.targets ?? []).map((t) => ({
        ...t,
        title: t.title ? preview(t.title) : undefined,
      })),
      resultIds: [],
      resultPreview: null,
      error: null,
      events: [],
    }
    db.prepare('INSERT INTO knowledge_activity(id,status,data) VALUES (?,?,?)').run(
      id,
      record.status,
      JSON.stringify(record),
    )
    event(id, input.queued ? '작업 대기' : '작업 시작')
    return id
  }
  const finish = (id: string, result: { ids?: string[]; preview?: unknown } = {}) => {
    event(id, '작업 완료')
    patch(id, {
      status: 'succeeded',
      finishedAt: Date.now(),
      resultIds: result.ids ?? [],
      resultPreview: result.preview === undefined ? null : preview(result.preview),
    })
  }
  const fail = (id: string, error: unknown, interrupted = false) => {
    const message = preview(error instanceof Error ? error.message : String(error))
    event(id, interrupted ? '작업 중단' : '작업 실패')
    patch(id, {
      status: interrupted ? 'interrupted' : 'failed',
      finishedAt: Date.now(),
      error: message,
    })
  }
  const calls = (activityId: string): KnowledgeLlmCall[] =>
    db
      .prepare('SELECT data FROM knowledge_llm_calls WHERE activity_id=? ORDER BY rowid')
      .all(activityId)
      .map((r) => JSON.parse((r as { data: string }).data))
  const saveCall = (call: KnowledgeLlmCall) =>
    db
      .prepare(
        'INSERT INTO knowledge_llm_calls VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data, input_tokens=excluded.input_tokens, output_tokens=excluded.output_tokens',
      )
      .run(
        call.id,
        call.activityId,
        call.provider,
        call.model,
        call.startedAt,
        call.inputTokens,
        call.outputTokens,
        JSON.stringify(call),
      )
  return {
    get,
    patch,
    begin,
    event,
    finish,
    fail,
    calls,
    recover() {
      for (const row of db
        .prepare("SELECT id FROM knowledge_activity WHERE status IN ('running','queued')")
        .all() as { id: string }[]) {
        fail(
          row.id,
          '데몬 재시작으로 중단되었습니다. 사용량을 보고하지 않은 호출은 미확인으로 유지합니다.',
          true,
        )
        for (const call of calls(row.id))
          if (call.status === 'running')
            saveCall({
              ...call,
              status: 'interrupted',
              finishedAt: Date.now(),
              error: 'Daemon restarted before completion',
            })
      }
    },
    /** Observe each actual chat attempt, including guarded retries and late responses after timeout. */
    observe(
      provider: ILLMProvider,
      activityId: string,
      tracker?: Pick<UsageTracker, 'record'>,
    ): ILLMProvider {
      return new Proxy(provider, {
        get(target, property) {
          if (property !== 'chat') {
            const value = Reflect.get(target, property, target) as unknown
            return typeof value === 'function' ? value.bind(target) : value
          }
          const chat: ILLMProvider['chat'] = async (request, options) => {
            const requestText = JSON.stringify({
              model: request.model,
              messages: request.messages,
              maxTokens: request.maxTokens,
            })
            const redactedRequest = redactSensitive(requestText).redacted
            const call: KnowledgeLlmCall = {
              id: randomUUID(),
              activityId,
              provider: provider.id,
              model: request.model,
              status: 'running',
              startedAt: Date.now(),
              finishedAt: null,
              inputTokens: null,
              outputTokens: null,
              thinkingTokens: null,
              cacheReadTokens: null,
              cacheCreationTokens: null,
              requestPreview: redactedRequest.slice(0, 12000),
              requestSha256: createHash('sha256').update(redactedRequest).digest('hex'),
              requestChars: redactedRequest.length,
              responsePreview: null,
              error: null,
            }
            const automatic = get(activityId)?.trigger === 'automatic'
            if (automatic) {
              // UTF-8 bytes plus bounded output is a conservative admission estimate, not measured usage.
              const reservation =
                Buffer.byteLength(requestText, 'utf8') + (request.maxTokens ?? 3500)
              try {
                budget.reserve(call.id, reservation)
              } catch (error) {
                event(activityId, '일일 예산 부족 · LLM 호출을 시작하지 않았습니다.')
                throw error
              }
              event(activityId, `일일 예산 예약: ${reservation} 토큰 (추정)`)
            }
            saveCall(call)
            event(activityId, `LLM 호출 시작: ${provider.id} / ${request.model}`)
            let response: ChatResponse
            try {
              response = await target.chat(request, options)
            } catch (error) {
              saveCall({
                ...call,
                status: 'failed',
                finishedAt: Date.now(),
                error: preview(error instanceof Error ? error.message : String(error)),
              })
              event(activityId, 'LLM 호출 실패 · 사용량 미확인')
              throw error
            }
            const usage = response.usage
            const input = number(usage?.inputTokens),
              output = number(usage?.outputTokens)
            // Some adapters synthesize 0/0 for absent provider usage. Never present that as measured zero.
            const reported = input !== null && output !== null && input + output > 0
            const completed: KnowledgeLlmCall = {
              ...call,
              status: 'succeeded',
              finishedAt: Date.now(),
              inputTokens: reported ? input : null,
              outputTokens: reported ? output : null,
              thinkingTokens: number(usage?.thinkingTokens),
              cacheReadTokens: number(usage?.cacheReadTokens),
              cacheCreationTokens: number(usage?.cacheCreationTokens),
              responsePreview: preview(response.message.content),
            }
            saveCall(completed)
            if (automatic && reported) budget.settle(call.id, input + output)
            event(
              activityId,
              reported
                ? `LLM 응답 수신 · 입력 ${input} / 출력 ${output} 토큰`
                : 'LLM 응답 수신 · 사용량 미확인',
            )
            if (reported && tracker) {
              try {
                tracker.record({
                  sessionId: `knowledge:${activityId}`,
                  provider: provider.id,
                  model: request.model,
                  inputTokens: input,
                  outputTokens: output,
                  thinkingTokens: completed.thinkingTokens ?? undefined,
                  cacheReadTokens: completed.cacheReadTokens ?? undefined,
                  cacheWriteTokens: completed.cacheCreationTokens ?? undefined,
                  durationMs: completed.finishedAt! - completed.startedAt,
                  toolName: 'knowledge.curation',
                })
              } catch {
                event(activityId, '전체 사용량 통계 반영 실패 · 이 실행의 사용량은 보존됨')
              }
            }
            return response
          }
          return chat
        },
      })
    },
    page(limit = 30, before?: number): KnowledgeActivityPage {
      const rows = db
        .prepare(
          'SELECT id FROM knowledge_activity WHERE sequence < ? ORDER BY sequence DESC LIMIT ?',
        )
        .all(before ?? Number.MAX_SAFE_INTEGER, limit + 1) as { id: string }[]
      const items = rows.slice(0, limit).map((r) => get(r.id)!)
      const total = db
        .prepare(
          "SELECT COUNT(*) AS operations, COALESCE(SUM(status='running'),0) AS running, COALESCE(SUM(status='queued'),0) AS queued FROM knowledge_activity",
        )
        .get() as KnowledgeActivityPage['totals']
      const sql = `COUNT(*) AS calls, COALESCE(SUM(input_tokens),0) AS inputTokens, COALESCE(SUM(output_tokens),0) AS outputTokens, COALESCE(SUM(input_tokens IS NULL OR output_tokens IS NULL),0) AS unknownUsageCalls`
      const usage = db.prepare(`SELECT ${sql} FROM knowledge_llm_calls`).get() as Pick<
        KnowledgeActivityPage['totals'],
        'calls' | 'inputTokens' | 'outputTokens' | 'unknownUsageCalls'
      >
      const today = db
        .prepare(`SELECT ${sql} FROM knowledge_llm_calls WHERE started_at >= ?`)
        .get(new Date().setUTCHours(0, 0, 0, 0)) as typeof usage
      const byModel = db
        .prepare(`SELECT provider, model, ${sql} FROM knowledge_llm_calls GROUP BY provider,model`)
        .all() as KnowledgeActivityPage['totals']['byModel']
      return {
        items,
        nextCursor: rows.length > limit ? items.at(-1)!.sequence : null,
        totals: {
          ...total,
          ...usage,
          todayInputTokens: today.inputTokens,
          todayOutputTokens: today.outputTokens,
          byModel,
        },
      }
    },
    export() {
      return (
        db.prepare('SELECT id FROM knowledge_activity ORDER BY sequence').all() as { id: string }[]
      ).map((r) => ({ activity: get(r.id)!, calls: calls(r.id) }))
    },
  }
}
