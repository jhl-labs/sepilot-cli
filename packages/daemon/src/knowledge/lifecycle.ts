import { z } from 'zod'
import { openDomainDb } from '../storage/domain-db.js'
import type { KnowledgeRepo } from './repo.js'
import { KnowledgeRecord, KnowledgeWrite, knowledgeError } from './schema.js'
import { redactSensitive } from '../memory/sensitive.js'

export const VerificationInput = z
  .object({
    id: z.string().min(1).max(200),
    expectedRevision: z.number().int().positive(),
    action: z.enum(['schedule', 'verify']),
    nextReviewAt: z.number().int().nonnegative().max(8_640_000_000_000_000).nullable(),
    note: z.string().trim().min(1).max(1000),
  })
  .strict()
export const RestoreInput = z
  .object({
    snapshot: KnowledgeRecord,
    expectedRevision: z.number().int().positive().nullable(),
  })
  .strict()
export function createKnowledgeLifecycle(repo: KnowledgeRepo, now = Date.now) {
  const db = openDomainDb({ name: 'knowledge' })
  db.exec(`CREATE TABLE IF NOT EXISTS knowledge_verifications (id TEXT PRIMARY KEY, data TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS knowledge_verification_history (sequence INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT NOT NULL);`)
  type Verification = {
    id: string
    verifiedRevision: number | null
    verifiedAt: number | null
    nextReviewAt: number | null
    note: string
  }
  const get = (id: string): Verification => {
    const row = db.prepare('SELECT data FROM knowledge_verifications WHERE id=?').get(id) as
      | { data: string }
      | undefined
    return row
      ? JSON.parse(row.data)
      : { id, verifiedRevision: null, verifiedAt: null, nextReviewAt: null, note: '' }
  }
  const status = (record: KnowledgeRecord) => {
    const saved = get(record.id)
    return {
      ...saved,
      title: record.title,
      revision: record.revision,
      state:
        record.status !== 'accepted'
          ? 'inactive'
          : saved.verifiedRevision !== null && saved.verifiedRevision !== record.revision
            ? 'changed'
            : saved.nextReviewAt !== null && saved.nextReviewAt <= now()
              ? 'due'
              : saved.verifiedAt !== null
                ? 'verified'
                : 'unverified',
    }
  }
  return {
    status,
    list: () => repo.list().map(status),
    verify(raw: z.infer<typeof VerificationInput>) {
      return db.transaction(() => {
        const input = VerificationInput.parse(raw)
        const record = repo.get(input.id)
        if (!record || record.revision !== input.expectedRevision)
          knowledgeError('지식이 변경되었습니다. 최신 버전을 다시 확인하세요.', 409)
        if (record.status !== 'accepted') knowledgeError('승격된 지식만 재검증할 수 있습니다.')
        if (input.nextReviewAt !== null && input.nextReviewAt <= now())
          knowledgeError('다음 확인 시점은 미래여야 합니다.')
        const next: Verification = {
          ...get(input.id),
          nextReviewAt: input.nextReviewAt,
          note: redactSensitive(input.note).redacted,
          ...(input.action === 'verify'
            ? { verifiedAt: now(), verifiedRevision: record.revision }
            : {}),
        }
        db.prepare(
          'INSERT INTO knowledge_verifications VALUES (?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data',
        ).run(input.id, JSON.stringify(next))
        db.prepare('INSERT INTO knowledge_verification_history(data) VALUES (?)').run(
          JSON.stringify({
            ...next,
            action: input.action,
            at: now(),
            expectedRevision: record.revision,
          }),
        )
        return status(record)
      })()
    },
    restore(raw: z.infer<typeof RestoreInput>) {
      return db.transaction(() => {
        const input = RestoreInput.parse(raw)
        const current = repo.get(input.snapshot.id)
        if ((current?.revision ?? null) !== input.expectedRevision)
          knowledgeError('복원 대상이 변경되었습니다. 현재 지식과 다시 비교하세요.', 409)
        const snapshot = input.snapshot
        // Backups are untrusted input. Restore content as a candidate, never imported approval or execution history.
        const write = KnowledgeWrite.parse({
          id: snapshot.id,
          expectedRevision: current?.revision,
          title: redactSensitive(snapshot.title).redacted,
          body: redactSensitive(snapshot.body).redacted,
          kind: snapshot.kind,
          status: 'candidate',
          parentId: snapshot.parentId,
          tags: snapshot.tags.map((t) => redactSensitive(t).redacted),
          relations: snapshot.relations,
          reason: `백업 v${snapshot.revision}에서 선택 복원 · 출처와 내용을 재검토하세요.`,
        })
        return repo.write(
          write,
          snapshot.sources.map((s) => ({
            ...s,
            title: redactSensitive(s.title).redacted,
            excerpt: redactSensitive(s.excerpt).redacted,
          })),
        )
      })()
    },
    export() {
      return db
        .prepare('SELECT data FROM knowledge_verification_history ORDER BY sequence')
        .all()
        .map((row) => JSON.parse((row as { data: string }).data))
    },
  }
}
