import type { SemanticSearchOptions } from '@sepilotd/core'
import { canReadLegacyGlobalMemory } from './scope.js'

/** SQL ownership is the same ACL as isMemoryVisibleInScope, before LIMIT. */
export function buildMemorySearchFilter(alias: string, options?: SemanticSearchOptions): {
  clause: string; params: Record<string, unknown>
} {
  const conditions: string[] = []
  const params: Record<string, unknown> = {}
  const tags = `json_each(CASE WHEN json_valid(${alias}.tags) THEN ${alias}.tags ELSE '[]' END)`
  const metadata = `CASE WHEN json_valid(${alias}.metadata) THEN ${alias}.metadata ELSE '{}' END`
  const exists = (predicate: string) => `EXISTS (SELECT 1 FROM ${tags} t WHERE ${predicate})`
  const has = (prefix: string) => exists(`lower(t.value) LIKE '${prefix}%'`)
  if (options?.scopeTags !== undefined) {
    params.memoryScope = JSON.stringify(options.scopeTags.map((tag) => tag.toLowerCase()))
    const match = (prefix: string) => exists(`lower(t.value) LIKE '${prefix}%' AND lower(t.value) IN (SELECT value FROM json_each(@memoryScope))`)
    // Namespace isolation is an additional boundary, including for public entries.
    // Count tags (not DISTINCT): malformed multi-namespace records fail closed,
    // matching the in-memory ACL before ranking and LIMIT can discard valid hits.
    const namespaces = options.scopeTags.filter((tag) => tag.toLowerCase().startsWith('scope:persona:'))
    if (namespaces.length === 0) conditions.push(`NOT ${has('scope:persona:')}`)
    else if (namespaces.length === 1) {
      conditions.push(`(SELECT count(*) FROM ${tags} t WHERE lower(t.value) LIKE 'scope:persona:%') = 1`)
      conditions.push(match('scope:persona:'))
    } else conditions.push('0')
    const legacy = options.scopeTags.length === 0 || canReadLegacyGlobalMemory(options.scopeTags)
    conditions.push(`(${exists("lower(t.value) = 'scope:public'")}
      OR (${legacy ? '1' : '0'} AND NOT ${has('scope:')})
      OR ${match('scope:user:')} OR ${match('scope:group:')}
      OR (NOT ${has('scope:user:')} AND (
        ${match('scope:channel:')} OR (NOT ${has('scope:channel:')} AND ${match('scope:')})
      )))`)
  }
  for (const [key, values, negate] of [
    ['requiredTags', options?.tags, false], ['excludedTags', options?.excludeTags, true],
  ] as const) {
    if (!values?.length) continue
    params[key] = JSON.stringify(values)
    if (negate) conditions.push(`NOT ${exists(`t.value IN (SELECT value FROM json_each(@${key}))`)}`)
    else if (options?.tagsLogic === 'or') conditions.push(exists(`t.value IN (SELECT value FROM json_each(@${key}))`))
    else conditions.push(`NOT EXISTS (SELECT 1 FROM json_each(@${key}) r WHERE NOT ${exists('t.value = r.value')})`)
  }
  if (!options?.includeInactive) {
    conditions.push(`COALESCE(json_extract(${metadata}, '$.memory.status'), 'active') = 'active'`)
    params.memoryAsOf = options?.asOf ?? new Date().toISOString()
    conditions.push(`(json_extract(${metadata}, '$.memory.validFrom') IS NULL OR julianday(json_extract(${metadata}, '$.memory.validFrom')) <= julianday(@memoryAsOf))`)
    conditions.push(`(json_extract(${metadata}, '$.memory.validUntil') IS NULL OR julianday(json_extract(${metadata}, '$.memory.validUntil')) > julianday(@memoryAsOf))`)
  }
  return { clause: conditions.map((condition) => `AND ${condition}`).join('\n'), params }
}
