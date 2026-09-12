import { createHash } from 'node:crypto'
import { z } from 'zod'
import { WikiPortableNode } from '../wiki/schema.js'
import { KnowledgeWrite, KnowledgeSource } from '../knowledge/schema.js'

export const Entry = z.discriminatedUnion('collection', [
  z
    .object({ collection: z.literal('docs'), record: WikiPortableNode.omit({ updatedAt: true }) })
    .strict(),
  z
    .object({
      collection: z.literal('wiki'),
      record: KnowledgeWrite.omit({ expectedRevision: true, reason: true }).extend({
        id: z.string().min(1).max(200),
        sources: z.array(KnowledgeSource).max(20),
      }),
    })
    .strict(),
])
export type Entry = z.infer<typeof Entry>
export const keyOf = (e: Entry) =>
  `${e.collection}/${createHash('sha256').update(e.record.id).digest('hex')}.md`
export function encode(e: Entry): string {
  const { body, ...metadata } = e.record
  return `<!-- sepilot-sync ${JSON.stringify({ collection: e.collection, record: metadata }, (_key, value) => (value && typeof value === 'object' && !Array.isArray(value) ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b))) : value))} -->\n${body}`
}
export function decode(text: string): Entry {
  const end = text.indexOf(' -->\n')
  if (!text.startsWith('<!-- sepilot-sync ') || end < 0)
    throw new Error('Invalid Sepilot Markdown header')
  const metadata = JSON.parse(text.slice('<!-- sepilot-sync '.length, end))
  return Entry.parse({ ...metadata, record: { ...metadata.record, body: text.slice(end + 5) } })
}
export const digest = (text: string | undefined) =>
  text === undefined ? null : createHash('sha256').update(text).digest('hex')
export interface Change {
  key: string
  title: string
  local: string | null
  remote: string | null
  action: 'upload' | 'download' | 'conflict'
}
export function plan(
  local: Record<string, string>,
  remote: Record<string, string>,
  base: Record<string, string>,
): Change[] {
  const changes: Change[] = []
  for (const key of new Set([
    ...Object.keys(local),
    ...Object.keys(remote),
    ...Object.keys(base),
  ])) {
    const l = local[key],
      r = remote[key],
      b = base[key]
    if (l === r) continue
    const action =
      b === undefined
        ? l === undefined
          ? 'download'
          : r === undefined
            ? 'upload'
            : 'conflict'
        : l === undefined || r === undefined
          ? 'conflict'
          : digest(l) === b
            ? 'download'
            : digest(r) === b
              ? 'upload'
              : 'conflict'
    changes.push({
      key,
      title: decode(l ?? r!).record.title,
      local: l ?? null,
      remote: r ?? null,
      action,
    })
  }
  return changes
}
