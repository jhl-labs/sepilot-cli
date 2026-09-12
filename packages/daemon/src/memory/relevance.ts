import type { MemoryEntry } from '@sepilotd/core'

export interface RelevantMemorySelectionOptions {
  limit?: number
  threshold?: number
}

export interface RankedRelevantMemory {
  entry: MemoryEntry
  relevance: number
}

const MEMORY_TOKEN_RE = /[\p{L}\p{N}_-]+/gu

export function selectRelevantMemories(
  query: string,
  results: MemoryEntry[],
  options: RelevantMemorySelectionOptions = {},
): MemoryEntry[] {
  const limit = options.limit ?? 3
  const threshold = options.threshold ?? 0.15
  const seen = new Set<string>()

  return rankRelevantMemories(query, results)
    .filter(({ relevance }) => relevance >= threshold)
    .map(({ entry }) => entry)
    .filter((entry) => {
      const key = normalizeMemoryText(entry.content)
      if (!key || seen.has(key)) {
        return false
      }
      seen.add(key)
      return true
    })
    .slice(0, limit)
}

export function rankRelevantMemories(
  query: string,
  results: MemoryEntry[],
): RankedRelevantMemory[] {
  const normalizedQuery = normalizeMemoryText(query)
  const queryTokens = tokenizeMemoryText(query)

  return [...results]
    .map((entry) => ({
      entry,
      relevance: scoreRelevantMemory(normalizedQuery, queryTokens, entry),
    }))
    .sort((left, right) => right.relevance - left.relevance)
}

export function scoreRelevantMemory(
  normalizedQuery: string,
  queryTokens: string[],
  entry: MemoryEntry,
): number {
  const normalizedContent = normalizeMemoryText(entry.content)
  const contentTokens = tokenizeMemoryText(entry.content)
  const overlap = tokenOverlap(queryTokens, contentTokens)
  const baseScore = Math.max(0, entry.score ?? 0)
  const substringBonus = normalizedQuery.length >= 4 && normalizedContent.includes(normalizedQuery)
    ? 0.2
    : 0
  const sourceBias = entry.source === 'user'
    ? 0.08
    : entry.source === 'conversation'
      ? 0.05
      : entry.source === 'document'
        ? 0.03
        : 0.02
  const autoExtractedPenalty = entry.tags?.includes('source:auto') ? 0.04 : 0
  // Hybrid search can return a tail of weak embedding matches. Source bias
  // alone previously lifted those above the inclusion threshold even when the
  // query and memory shared no lexical evidence, which injected unrelated host
  // hardware/profile facts into focused repository turns. Preserve strong
  // cross-language semantic hits, but require them to clear a higher bar.
  // Agglutinative languages attach particles directly to a token, so an exact
  // identifier in the query ("E2E-1234에서") never equals the same identifier
  // in the memory ("E2E-1234:"). That is a tokenizer artifact, not missing
  // lexical evidence, so a distinctive affix match counts as a real overlap.
  const affixOverlap = overlap === 0 && substringBonus === 0
    && hasDistinctiveAffixMatch(queryTokens, contentTokens)
  const semanticOnlyPenalty = overlap === 0 && substringBonus === 0 && !affixOverlap ? 0.16 : 0

  return (baseScore * 0.65)
    + (overlap * 0.35)
    + substringBonus
    + sourceBias
    - autoExtractedPenalty
    - semanticOnlyPenalty
}

export function normalizeMemoryText(text: string): string {
  return text.toLowerCase().replace(/\s+/g, ' ').trim()
}

export function tokenizeMemoryText(text: string): string[] {
  return Array.from(
    new Set((text.toLowerCase().match(MEMORY_TOKEN_RE) ?? []).filter((token) => token.length >= 2)),
  )
}

const DISTINCTIVE_AFFIX_MIN_LENGTH = 4

/**
 * True when a distinctive token from one side is the prefix of a token on the
 * other side. Short tokens are excluded so common words cannot manufacture a
 * match, which keeps this a lexical-evidence signal rather than a fuzzy one.
 */
export function hasDistinctiveAffixMatch(left: string[], right: string[]): boolean {
  for (const leftToken of left) {
    if (leftToken.length < DISTINCTIVE_AFFIX_MIN_LENGTH) continue
    for (const rightToken of right) {
      if (rightToken.length < DISTINCTIVE_AFFIX_MIN_LENGTH) continue
      if (leftToken.startsWith(rightToken) || rightToken.startsWith(leftToken)) return true
    }
  }
  return false
}

export function tokenOverlap(left: string[], right: string[]): number {
  if (left.length === 0 || right.length === 0) return 0
  const rightSet = new Set(right)
  let matches = 0
  for (const token of left) {
    if (rightSet.has(token)) matches++
  }
  return matches / Math.max(1, Math.min(left.length, right.length))
}
