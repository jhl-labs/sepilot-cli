import { createHash } from 'node:crypto'
import { z } from 'zod'
import { openDomainDb } from '../storage/domain-db.js'
import { redactSensitive } from '../memory/sensitive.js'
import {
  KnowledgeWrite,
  knowledgeError,
  type KnowledgeRecord,
  type KnowledgeSource,
} from './schema.js'
import type { KnowledgeRepo } from './repo.js'
import type { createKnowledgeActivityStore } from './activity.js'

export const ChatUpdateMode = z.enum(['off', 'suggest', 'automatic'])
export const KnowledgePatch = KnowledgeWrite.pick({
  title: true,
  body: true,
  kind: true,
  parentId: true,
  tags: true,
  relations: true,
})
  .partial()
  .strict()
export const KnowledgeEdit = z
  .object({
    action: z.enum(['update', 'archive', 'restore']),
    id: z.string().min(1).max(200),
    expectedRevision: z.number().int().positive(),
    reason: z.string().trim().min(1).max(1000),
    patch: KnowledgePatch.optional(),
  })
  .strict()
export const KnowledgeProposalInput = z
  .object({
    id: z.string().min(1).max(200).optional(),
    expectedRevision: z.number().int().positive().optional(),
    title: z.string().trim().min(1).max(300),
    body: z.string().trim().min(1).max(24000),
    reason: z.string().trim().min(1).max(1000),
  })
  .strict()
  .refine(
    (input) => Boolean(input.id) === Boolean(input.expectedRevision),
    'Updates require id and expectedRevision together',
  )

export interface ContentProposal {
  id: string
  status: 'pending' | 'accepted' | 'rejected'
  before: KnowledgeRecord | null
  proposed: z.infer<typeof KnowledgeProposalInput>
  source: KnowledgeSource
  createdAt: number
  resolvedAt: number | null
  resultId: string | null
  reviewReason?: string
}

export function writable(record: KnowledgeRecord): z.infer<typeof KnowledgeWrite> {
  const {
    sources: _sources,
    revision,
    createdAt: _created,
    updatedAt: _updated,
    ...fields
  } = record
  return { ...fields, expectedRevision: revision }
}

// Decision identity excludes model-written rationale and incidental source/session details.
// Preserve Markdown whitespace; only normalize line endings and Unicode representation.
function proposalContentKey(input: z.infer<typeof KnowledgeProposalInput>): string {
  return createHash('sha256')
    .update(
      JSON.stringify([
        input.id ?? null,
        input.expectedRevision ?? null,
        input.title.trim().normalize('NFC'),
        input.body.trim().replace(/\r\n/g, '\n').normalize('NFC'),
      ]),
    )
    .digest('hex')
}

/** One transactional service for chat and UI decisions; no hidden auxiliary LLM call. */
export function createKnowledgeMaintenance(
  repo: KnowledgeRepo,
  activity: ReturnType<typeof createKnowledgeActivityStore>,
  changed: (id: string) => void,
) {
  const db = openDomainDb({ name: 'knowledge' })
  db.exec(`CREATE TABLE IF NOT EXISTS knowledge_chat_policy (id INTEGER PRIMARY KEY CHECK(id=1), mode TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS knowledge_content_proposals (id TEXT PRIMARY KEY, status TEXT NOT NULL, data TEXT NOT NULL);`)
  const columns = db.prepare('PRAGMA table_info(knowledge_content_proposals)').all() as {
    name: string
  }[]
  if (!columns.some((column) => column.name === 'content_key'))
    db.exec('ALTER TABLE knowledge_content_proposals ADD COLUMN content_key TEXT')
  db.transaction(() => {
    const rows = db
      .prepare('SELECT id,data FROM knowledge_content_proposals WHERE content_key IS NULL')
      .all() as { id: string; data: string }[]
    const update = db.prepare('UPDATE knowledge_content_proposals SET content_key=? WHERE id=?')
    for (const row of rows)
      update.run(proposalContentKey((JSON.parse(row.data) as ContentProposal).proposed), row.id)
  })()
  db.exec(
    'CREATE INDEX IF NOT EXISTS knowledge_proposal_content_key ON knowledge_content_proposals(content_key)',
  )
  const mode = (): z.infer<typeof ChatUpdateMode> =>
    ChatUpdateMode.parse(
      (
        db.prepare('SELECT mode FROM knowledge_chat_policy WHERE id=1').get() as
          | { mode: string }
          | undefined
      )?.mode ?? 'suggest',
    )
  const get = (id: string): ContentProposal | undefined => {
    const row = db.prepare('SELECT data FROM knowledge_content_proposals WHERE id=?').get(id) as
      | { data: string }
      | undefined
    return row ? JSON.parse(row.data) : undefined
  }
  const reuseDecision = (proposal: ContentProposal): ContentProposal => {
    if (proposal.status === 'accepted') {
      const current = proposal.resultId ? repo.get(proposal.resultId) : undefined
      if (
        !current ||
        current.status !== 'accepted' ||
        current.title !== proposal.proposed.title ||
        current.body !== proposal.proposed.body
      ) {
        knowledgeError(
          '이전에 반영한 지식이 이후 수정되거나 보관되었습니다. 최신 본문을 읽고 새 요청으로 처리하세요.',
          409,
        )
      }
    }
    return proposal
  }
  const put = (proposal: ContentProposal) => {
    db.prepare(
      'INSERT INTO knowledge_content_proposals (id,status,data,content_key) VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,data=excluded.data,content_key=excluded.content_key',
    ).run(
      proposal.id,
      proposal.status,
      JSON.stringify(proposal),
      proposalContentKey(proposal.proposed),
    )
    return proposal
  }
  const recordsVersion = () =>
    (
      db.prepare('SELECT COALESCE(SUM(revision),0) AS version FROM knowledge_records').get() as {
        version: number
      }
    ).version
  const trace = <T>(
    kind: string,
    reason: string,
    task: () => T,
    automatic = false,
    parentId?: string,
  ): T => {
    const activityId = activity.begin({
      kind,
      parentId,
      trigger: automatic ? 'automatic' : 'manual',
      summary: reason,
    })
    try {
      const beforeVersion = kind === 'write' ? recordsVersion() : null
      const result = task()
      const r = result as { id?: string; resultId?: string }
      activity.finish(activityId, {
        ids: r.resultId ? [r.resultId] : r.id ? [r.id] : [],
        preview: result,
      })
      if (kind === 'write' && beforeVersion !== recordsVersion()) changed(activityId)
      return result
    } catch (error) {
      activity.fail(activityId, error)
      throw error
    }
  }
  const current = (id: string, revision: number) => {
    const record = repo.get(id)
    if (!record) knowledgeError('지식을 찾을 수 없습니다.', 404)
    if (record.revision !== revision)
      knowledgeError('지식이 변경되었습니다. 최신 본문을 읽고 다시 제안하세요.', 409)
    return record
  }
  const apply = (proposal: ContentProposal) => {
    const input = proposal.proposed
    const old = input.id ? current(input.id, input.expectedRevision!) : undefined
    if (old?.status === 'archived') knowledgeError('보관한 지식은 먼저 명시적으로 복원하세요.', 409)
    const saved = repo.write(
      old
        ? {
            ...writable(old),
            status: 'accepted',
            title: input.title,
            body: input.body,
            reason: input.reason,
          }
        : {
            id: `proposal-${proposal.id}`,
            title: input.title,
            body: input.body,
            kind: 'fact',
            status: 'accepted',
            parentId: null,
            tags: [],
            relations: [],
            reason: input.reason,
          },
      [proposal.source],
    )
    return put({ ...proposal, status: 'accepted', resolvedAt: Date.now(), resultId: saved.id })
  }
  return {
    mode,
    recordsVersion,
    getProposal: get,
    export: () => ({
      mode: mode(),
      proposals: (
        db.prepare('SELECT data FROM knowledge_content_proposals ORDER BY rowid').all() as {
          data: string
        }[]
      ).map((row) => JSON.parse(row.data) as ContentProposal),
    }),
    list: (includeResolved = false) =>
      (
        db
          .prepare(
            `SELECT data FROM knowledge_content_proposals ${includeResolved ? '' : "WHERE status='pending'"} ORDER BY rowid DESC${includeResolved ? ' LIMIT 100' : ''}`,
          )
          .all() as { data: string }[]
      ).map((row) => JSON.parse(row.data) as ContentProposal),
    configure(raw: unknown, parentId?: string) {
      const next = ChatUpdateMode.parse(raw)
      return trace(
        'settings',
        `대화 중 Wiki 갱신 방식: ${next}`,
        () => {
          db.prepare(
            'INSERT INTO knowledge_chat_policy VALUES (1,?) ON CONFLICT(id) DO UPDATE SET mode=excluded.mode',
          ).run(next)
          return { mode: next }
        },
        false,
        parentId,
      )
    },
    edit(raw: unknown, source?: KnowledgeSource, parentId?: string) {
      const input = KnowledgeEdit.parse(raw)
      if (input.action === 'update' && (!input.patch || !Object.keys(input.patch).length))
        knowledgeError('수정할 필드를 지정하세요.')
      if (input.action !== 'update' && input.patch)
        knowledgeError('보관·복원에는 수정 내용을 함께 넣지 마세요.')
      return trace(
        'write',
        redactSensitive(input.reason).redacted,
        () =>
          db.transaction(() => {
            const old = current(input.id, input.expectedRevision)
            if (input.action === 'restore' && old.status !== 'archived')
              knowledgeError('보관된 지식만 복원할 수 있습니다.', 409)
            if (input.action === 'update' && old.status === 'archived')
              knowledgeError('보관한 지식은 먼저 복원하세요.', 409)
            const patch = input.patch ?? {}
            const safePatch = {
              ...patch,
              ...(patch.title !== undefined
                ? { title: redactSensitive(patch.title).redacted }
                : {}),
              ...(patch.body !== undefined ? { body: redactSensitive(patch.body).redacted } : {}),
            }
            return repo.write(
              {
                ...writable(old),
                ...safePatch,
                status:
                  input.action === 'archive'
                    ? 'archived'
                    : input.action === 'restore'
                      ? 'accepted'
                      : old.status,
                reason: redactSensitive(input.reason).redacted,
              },
              source ? [source] : [],
            )
          })(),
        false,
        parentId,
      )
    },
    propose(raw: unknown, source: KnowledgeSource, userEvidence: boolean, parentId?: string) {
      const input = KnowledgeProposalInput.parse(raw)
      const safe = {
        ...input,
        title: redactSensitive(input.title).redacted,
        body: redactSensitive(input.body).redacted,
        reason: redactSensitive(input.reason).redacted,
      }
      const id = proposalContentKey(safe)
      const automatic = mode() === 'automatic' && userEvidence
      return trace(
        automatic ? 'write' : 'proposal',
        safe.reason,
        () =>
          db.transaction(() => {
            if (mode() === 'off')
              knowledgeError(
                '대화 중 Wiki 갱신 제안이 꺼져 있습니다. 명시적인 사용자 저장·수정 요청만 처리하세요.',
                409,
              )
            // Include pre-migration proposals without changing their externally referenced IDs.
            const previous = db
              .prepare(
                `SELECT data FROM knowledge_content_proposals WHERE content_key=?
              ORDER BY (status <> 'pending') DESC, json_extract(data, '$.resolvedAt') DESC, rowid DESC LIMIT 1`,
              )
              .get(id) as { data: string } | undefined
            if (previous) return reuseDecision(JSON.parse(previous.data) as ContentProposal)
            const count = db
              .prepare(
                "SELECT count(*) AS n FROM knowledge_content_proposals WHERE status='pending'",
              )
              .get() as { n: number }
            if (count.n >= 100) knowledgeError('대기 중인 갱신 제안을 먼저 검토하세요.', 409)
            if (
              !safe.id &&
              repo
                .list()
                .some(
                  (record) =>
                    record.title.trim().toLocaleLowerCase() ===
                      safe.title.trim().toLocaleLowerCase() ||
                    (record.status === 'archived' && record.body.trim() === safe.body.trim()),
                )
            )
              knowledgeError(
                '같은 제목의 지식 또는 보관된 원문이 있습니다. 해당 지식을 읽고 수정하거나 명시적으로 복원하세요.',
                409,
              )
            const old = safe.id ? current(safe.id, safe.expectedRevision!) : null
            if (old?.status === 'archived')
              knowledgeError('보관한 지식에는 자동 갱신을 제안하지 않습니다.', 409)
            const proposal: ContentProposal = {
              id,
              status: 'pending',
              before: old,
              proposed: safe,
              source,
              createdAt: Date.now(),
              resolvedAt: null,
              resultId: null,
            }
            const active = repo.list().filter((record) => record.status === 'accepted')
            let reviewReason = !userEvidence
              ? '사용자 발언 근거가 없어 검토가 필요합니다.'
              : undefined
            if (old && old.status !== 'accepted')
              reviewReason = '검토 대기 지식은 자동으로 승격하지 않습니다.'
            if (
              old &&
              active.some(
                (other) =>
                  other.id !== old.id &&
                  other.relations.some(
                    (edge) => edge.targetId === old.id && edge.type === 'supersedes',
                  ),
              )
            )
              reviewReason = '다른 지식으로 대체된 항목이므로 검토가 필요합니다.'
            if (
              old &&
              active.some(
                (other) =>
                  other.id !== old.id &&
                  (other.relations.some(
                    (edge) => edge.targetId === old.id && edge.type === 'contradicts',
                  ) ||
                    old.relations.some(
                      (edge) => edge.targetId === other.id && edge.type === 'contradicts',
                    )),
              )
            )
              reviewReason = '다른 지식과 충돌 관계가 있어 검토가 필요합니다.'
            if (mode() === 'automatic' && reviewReason) proposal.reviewReason = reviewReason
            return automatic && !reviewReason ? apply(proposal) : put(proposal)
          })(),
        automatic,
        parentId,
      )
    },
    decide(id: string, decision: 'accept' | 'reject', parentId?: string) {
      return trace(
        decision === 'accept' ? 'write' : 'review',
        `Wiki 갱신 제안 ${decision === 'accept' ? '승인' : '거절'}`,
        () =>
          db.transaction(() => {
            const proposal = get(id)
            if (!proposal) knowledgeError('갱신 제안을 찾을 수 없습니다.', 404)
            if (proposal.status === (decision === 'accept' ? 'accepted' : 'rejected'))
              return reuseDecision(proposal)
            if (proposal.status !== 'pending') knowledgeError('이미 처리된 갱신 제안입니다.', 409)
            return decision === 'accept'
              ? apply(proposal)
              : put({ ...proposal, status: 'rejected', resolvedAt: Date.now() })
          })(),
        false,
        parentId,
      )
    },
  }
}
