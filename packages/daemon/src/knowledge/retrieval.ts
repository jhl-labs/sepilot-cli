import { rankChatKnowledgeItems } from '../server/chat-knowledge.js'
import type { KnowledgeRecord } from './schema.js'

export function retrievePersonalKnowledge(
  records: KnowledgeRecord[],
  query: string,
  limit: number,
) {
  const accepted = records.filter((r) => r.status === 'accepted' && r.kind !== 'category')
  const superseded = new Set(
    accepted.flatMap((r) =>
      r.relations.filter((e) => e.type === 'supersedes').map((e) => e.targetId),
    ),
  )
  const byId = new Map(records.map((r) => [r.id, r]))
  return rankChatKnowledgeItems(
    query,
    accepted
      .filter((r) => !superseded.has(r.id))
      .map((r) => {
        const conflicts = accepted.filter(
          (other) =>
            !superseded.has(other.id) &&
            (other.relations.some((e) => e.type === 'contradicts' && e.targetId === r.id) ||
              r.relations.some((e) => e.type === 'contradicts' && e.targetId === other.id)),
        )
        return {
          id: r.id,
          source: 'Personal knowledge',
          title: r.title,
          contextNotes: [
            ...(conflicts.length
              ? [
                  `Unresolved contradiction: ${conflicts.length} competing claim(s). Neither side is settled; read the complete records before answering.`,
                  ...conflicts.map((c) => `${c.id}: ${c.title}`),
                ]
              : []),
            `Knowledge ID: ${r.id}; revision: ${r.revision}`,
            ...r.sources.map((s) => `Source: ${s.kind}:${s.id} — ${s.title}`),
          ],
          content: [
            `Knowledge ID: ${r.id}; revision: ${r.revision}; updated: ${new Date(r.updatedAt).toISOString()}`,
            r.body,
            ...r.sources.map((s) => `Source: ${s.kind}:${s.id} — ${s.title}`),
            ...r.relations
              .filter((e) => e.type !== 'contradicts')
              .map((e) => `Relationship: ${e.type} → ${byId.get(e.targetId)?.title ?? e.targetId}`),
            ...conflicts.map(
              (c) =>
                `Unresolved contradiction with ${c.title} (${c.id}): ${c.body.slice(0, 400)}. Do not treat either claim as settled.`,
            ),
          ].join('\n'),
          tags: [...r.tags, r.kind, ...(r.parentId ? [byId.get(r.parentId)?.title ?? ''] : [])],
        }
      }),
    limit,
  )
}
