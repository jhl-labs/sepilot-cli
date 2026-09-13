import { createWikiRepo } from '../wiki/repo.js'
import { createKnowledgeRepo } from '../knowledge/repo.js'
import { openDomainDb } from '../storage/domain-db.js'
import { type Entry, encode, keyOf } from './model.js'

export function createDocsGitStore() {
  const docs = createWikiRepo()
  const wiki = createKnowledgeRepo()
  const store = {
    snapshot(): Record<string, string> {
      const entries: Entry[] = [
        ...docs
          .tree()
          .map(({ updatedAt: _time, ...record }) => ({ collection: 'docs' as const, record })),
        ...wiki
          .list()
          .map(
            ({
              revision: _rev,
              createdAt: _created,
              updatedAt: _updated,
              reason: _reason,
              ...record
            }) => ({ collection: 'wiki' as const, record }),
          ),
      ]
      return Object.fromEntries(entries.map((e) => [keyOf(e), encode(e)]))
    },
    apply(entries: Entry[], expected?: Record<string, string>) {
      openDomainDb({ name: 'wiki' }).transaction(() =>
        openDomainDb({ name: 'knowledge' }).transaction(() => {
          if (expected) {
            const current = store.snapshot()
            for (const entry of entries) {
              const key = keyOf(entry)
              if (current[key] !== expected[key])
                throw new Error('Local knowledge changed after review; compare again')
            }
          }
          const now = Date.now()
          const importedDocs = entries
            .filter((e) => e.collection === 'docs')
            .map((e) => ({ ...e.record, updatedAt: now }))
          if (importedDocs.length) {
            const merged = new Map(docs.tree().map((e) => [e.id, e]))
            for (const e of importedDocs) merged.set(e.id, e)
            const sorted = []
            const remaining = [...merged.values()]
            const done = new Set<string>()
            while (remaining.length) {
              const index = remaining.findIndex((e) => !e.parentId || done.has(e.parentId))
              if (index < 0) throw new Error('Document parent is missing or cyclic')
              const entry = remaining.splice(index, 1)[0]!
              sorted.push(entry)
              done.add(entry.id)
            }
            docs.importNodes(sorted, { replace: true })
          }
          wiki.writeBatch(
            entries
              .filter((e) => e.collection === 'wiki')
              .map(({ record }) => {
                const { sources, ...fields } = record
                return {
                  input: {
                    ...fields,
                    expectedRevision: wiki.get(record.id)?.revision,
                    reason: 'Git 동기화',
                  },
                  sources,
                }
              }),
          )
        })(),
      )()
    },
  }
  return store
}
