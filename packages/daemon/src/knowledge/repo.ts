import { createHash, randomUUID } from 'node:crypto'
import { openDomainDb } from '../storage/domain-db.js'
import {
  KnowledgeRecord,
  KnowledgeReview,
  KnowledgeWrite,
  knowledgeError,
  type KnowledgeSource,
} from './schema.js'

/** Canonical snapshots and revisions never participate in Memory eviction. */
export function createKnowledgeRepo() {
  const db = openDomainDb({ name: 'knowledge' })
  db.exec(`CREATE TABLE IF NOT EXISTS knowledge_records (
    id TEXT PRIMARY KEY, revision INTEGER NOT NULL, data TEXT NOT NULL
  ); CREATE TABLE IF NOT EXISTS knowledge_revisions (
    id TEXT NOT NULL, revision INTEGER NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (id, revision)
  ); CREATE TABLE IF NOT EXISTS knowledge_captures (
    fingerprint TEXT PRIMARY KEY, ids TEXT NOT NULL
  );`)
  db.exec(
    'CREATE TABLE IF NOT EXISTS knowledge_review (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT NOT NULL)',
  )
  db.exec(
    'CREATE TABLE IF NOT EXISTS knowledge_organization_cursor (id INTEGER PRIMARY KEY CHECK(id=1), last_id TEXT NOT NULL)',
  )
  const organizationCursor = () =>
    (
      db.prepare('SELECT last_id FROM knowledge_organization_cursor WHERE id=1').get() as
        | { last_id: string }
        | undefined
    )?.last_id ?? null
  const advanceOrganization = (id: string) => {
    db.prepare(
      'INSERT INTO knowledge_organization_cursor VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET last_id=excluded.last_id',
    ).run(id)
  }
  const review = (): KnowledgeReview => {
    const row = db.prepare('SELECT data FROM knowledge_review WHERE id=1').get() as
      | { data: string }
      | undefined
    return row
      ? KnowledgeReview.parse(JSON.parse(row.data))
      : { autoOrganize: false, status: 'idle', suggestions: [], error: null, updatedAt: 0 }
  }
  const saveReview = (patch: Partial<KnowledgeReview>) => {
    const next = KnowledgeReview.parse({ ...review(), ...patch, updatedAt: Date.now() })
    db.prepare(
      'INSERT INTO knowledge_review VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET data=excluded.data',
    ).run(JSON.stringify(next))
    return next
  }
  const decode = (row: unknown): KnowledgeRecord | undefined =>
    row ? KnowledgeRecord.parse(JSON.parse((row as { data: string }).data)) : undefined
  const get = (id: string) =>
    decode(db.prepare('SELECT data FROM knowledge_records WHERE id=?').get(id))
  const list = () =>
    db
      .prepare('SELECT data FROM knowledge_records ORDER BY rowid DESC')
      .all()
      .map((row) => decode(row)!)
  type Topology = Pick<KnowledgeRecord, 'id' | 'parentId' | 'kind' | 'status' | 'relations'>
  let batchView: Map<string, Topology> | null = null
  const topologyGet = (id: string): Topology | undefined => batchView?.get(id) ?? get(id)
  const topologyList = (): Topology[] => (batchView ? [...batchView.values()] : list())
  const write = (raw: KnowledgeWrite, sources: KnowledgeSource[] = []): KnowledgeRecord =>
    db.transaction(() => {
      const input = KnowledgeWrite.parse(raw)
      const id = input.id ?? randomUUID()
      const old = get(id)
      if (old ? input.expectedRevision !== old.revision : input.expectedRevision !== undefined) {
        knowledgeError('지식이 변경되었습니다. 새로고침한 뒤 다시 검토하세요.', 409)
      }
      if (input.parentId) {
        const parent = topologyGet(input.parentId)
        if (
          !parent ||
          parent.kind !== 'category' ||
          (parent.status === 'archived' && input.status !== 'archived')
        )
          knowledgeError('유효한 분류를 선택하세요.')
        const seen = new Set([id])
        let cursor: Topology | undefined = parent
        while (cursor) {
          if (seen.has(cursor.id)) knowledgeError('분류 트리는 순환할 수 없습니다.')
          seen.add(cursor.id)
          cursor = cursor.parentId ? topologyGet(cursor.parentId) : undefined
        }
      }
      if (
        old?.kind === 'category' &&
        input.kind !== 'category' &&
        topologyList().some((n) => n.parentId === id)
      ) {
        knowledgeError('하위 지식이 있는 분류의 유형은 바꿀 수 없습니다.')
      }
      if (
        input.status === 'archived' &&
        topologyList().some((n) => n.parentId === id && n.status !== 'archived')
      ) {
        knowledgeError('하위 지식을 먼저 이동하거나 보관하세요.')
      }
      for (const relation of input.relations) {
        if (relation.targetId === id || !topologyGet(relation.targetId))
          knowledgeError('관계 대상이 유효하지 않습니다.')
      }
      // Supersession is directional and acyclic, unlike ordinary related links.
      const pending = input.relations.filter((r) => r.type === 'supersedes').map((r) => r.targetId)
      const visited = new Set<string>()
      while (pending.length) {
        const target = pending.pop()!
        if (target === id) knowledgeError('대체 관계는 순환할 수 없습니다.')
        if (visited.has(target)) continue
        visited.add(target)
        pending.push(
          ...(topologyGet(target)?.relations ?? [])
            .filter((r) => r.type === 'supersedes')
            .map((r) => r.targetId),
        )
      }
      const { expectedRevision: _revision, ...fields } = input
      const record: KnowledgeRecord = {
        ...fields,
        id,
        tags: [...new Set(input.tags)],
        relations: [
          ...new Map(input.relations.map((r) => [`${r.type}:${r.targetId}`, r])).values(),
        ],
        // Keep the original source plus recent evidence; older complete snapshots remain in revisions.
        sources: (() => {
          const merged = [
            ...new Map(
              [...(old?.sources ?? []), ...sources].map((source) => [
                JSON.stringify([source.kind, source.id, source.excerpt]),
                source,
              ]),
            ).values(),
          ]
          return merged.length <= 20 ? merged : [merged[0]!, ...merged.slice(-19)]
        })(),
        revision: (old?.revision ?? 0) + 1,
        createdAt: old?.createdAt ?? Date.now(),
        updatedAt: Date.now(),
      }
      const data = JSON.stringify(KnowledgeRecord.parse(record))
      db.prepare(
        'INSERT INTO knowledge_records VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision, data=excluded.data',
      ).run(id, record.revision, data)
      db.prepare('INSERT INTO knowledge_revisions VALUES (?, ?, ?)').run(id, record.revision, data)
      return record
    })()
  return {
    get,
    list,
    organizationCursor,
    advanceOrganization,
    write,
    writeBatch(items: { input: KnowledgeWrite; sources: KnowledgeSource[] }[]) {
      return db.transaction(() => {
        const parsed = items.map(({ input, sources }) => ({
          input: { ...KnowledgeWrite.parse(input), id: input.id ?? randomUUID() },
          sources,
        }))
        if (new Set(parsed.map((e) => e.input.id)).size !== parsed.length)
          knowledgeError('중복 지식 ID입니다.')
        batchView = new Map(list().map((record) => [record.id, record]))
        for (const { input } of parsed) batchView.set(input.id, input)
        try {
          return parsed.map(({ input, sources }) => write(input, sources))
        } finally {
          batchView = null
        }
      })()
    },
    review,
    saveReview,
    history(id: string) {
      return db
        .prepare('SELECT data FROM knowledge_revisions WHERE id=? ORDER BY revision DESC')
        .all(id)
        .map((r) => decode(r)!)
    },
    capture(source: KnowledgeSource, drafts: KnowledgeWrite[]) {
      const fingerprint = createHash('sha256')
        .update(JSON.stringify([source.kind, source.id, source.excerpt]))
        .digest('hex')
      return db.transaction(() => {
        const previous = db
          .prepare('SELECT ids FROM knowledge_captures WHERE fingerprint=?')
          .get(fingerprint) as { ids: string } | undefined
        if (previous) return (JSON.parse(previous.ids) as string[]).map((id) => get(id)!)
        const records = drafts.map((draft) =>
          write({ ...draft, id: undefined, expectedRevision: undefined, status: 'candidate' }, [
            source,
          ]),
        )
        // Empty extraction can be retried after changing the model or guidance.
        if (records.length)
          db.prepare('INSERT INTO knowledge_captures VALUES (?, ?)').run(
            fingerprint,
            JSON.stringify(records.map((r) => r.id)),
          )
        return records
      })()
    },
  }
}
export type KnowledgeRepo = ReturnType<typeof createKnowledgeRepo>
