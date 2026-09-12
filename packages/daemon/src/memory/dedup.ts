import type { IDreamingMemoryStore, MemoryEntry } from '@sepilotd/core'
import { normalizeMemoryText, tokenizeMemoryText, tokenOverlap } from './relevance.js'

export interface DedupSearchQuery {
  text: string
  type: 'semantic' | 'keyword' | 'hybrid'
  minScore?: number
}

export interface DedupCandidateOptions {
  searchLimit?: number
  candidateLimit?: number
  minRelevance?: number
}

const DEDUP_STOPWORDS = new Set([
  'the', 'and', 'that', 'with', 'this', 'from', 'have', 'will', 'would', 'should',
  'could', 'into', 'after', 'before', 'about', 'than', 'then', 'they', 'them',
  'their', 'there', 'because', 'while', 'where', 'when', 'what', 'which', 'using',
  'use', 'used', 'into', 'onto', 'your', 'you', 'please', 'remember', 'need',
  'also', 'only', 'just', 'more', 'less', 'very', 'much', 'make', 'made',
  '그리고', '하지만', '그런데', '이것', '저것', '그것', '정도', '관련', '대한', '에서',
  '으로', '에게', '하면', '해야', '하는', '했다', '있다', '없다', '기억', '메모',
])

export async function findDedupCandidates(
  memoryStore: Pick<IDreamingMemoryStore, 'search'>,
  memory: Omit<MemoryEntry, 'score'>,
  processed: Set<string> = new Set(),
  options: DedupCandidateOptions = {},
): Promise<MemoryEntry[]> {
  const searchLimit = options.searchLimit ?? 6
  const candidateLimit = options.candidateLimit ?? 3
  const minRelevance = options.minRelevance ?? 0.3
  const queries = buildDedupQueries(memory.content)
  const ranked = new Map<string, { entry: MemoryEntry; relevance: number }>()

  for (const query of queries) {
    const similar = await memoryStore.search(query.text, {
      type: query.type,
      limit: searchLimit,
      minScore: query.minScore,
    })

    for (const candidate of similar) {
      if (candidate.id === memory.id || processed.has(candidate.id)) continue

      const relevance = scoreDedupCandidate(memory.content, candidate)
      if (relevance < minRelevance) continue

      const existing = ranked.get(candidate.id)
      if (!existing || relevance > existing.relevance) {
        ranked.set(candidate.id, { entry: candidate, relevance })
      }
    }
  }

  return Array.from(ranked.values())
    .sort((left, right) => right.relevance - left.relevance)
    .slice(0, candidateLimit)
    .map(({ entry }) => entry)
}

export function buildDedupQueries(content: string): DedupSearchQuery[] {
  const trimmed = content.trim()
  if (!trimmed) return []

  const queries: DedupSearchQuery[] = [{
    text: trimmed.slice(0, 240),
    type: 'hybrid',
    minScore: 0.08,
  }]

  const keywords = extractSalientMemoryTerms(trimmed).slice(0, 8).join(' ')
  if (keywords.length >= 6 && normalizeMemoryText(keywords) !== normalizeMemoryText(trimmed)) {
    queries.push({
      text: keywords,
      type: 'keyword',
    })
  }

  return queries
}

export function extractSalientMemoryTerms(text: string): string[] {
  const tokens = tokenizeMemoryText(text)
  return tokens.filter((token) => !DEDUP_STOPWORDS.has(token))
}

export function scoreDedupCandidate(primaryContent: string, candidate: MemoryEntry): number {
  const primaryTokens = tokenizeMemoryText(primaryContent)
  const candidateTokens = tokenizeMemoryText(candidate.content)
  const normalizedPrimary = normalizeMemoryText(primaryContent)
  const normalizedCandidate = normalizeMemoryText(candidate.content)
  const overlap = tokenOverlap(primaryTokens, candidateTokens)
  const baseScore = Math.max(0, candidate.score ?? 0)
  const substringBonus =
    normalizedPrimary.includes(normalizedCandidate) || normalizedCandidate.includes(normalizedPrimary)
      ? 0.2
      : 0

  return (baseScore * 0.55) + (overlap * 0.45) + substringBonus
}
