import type { SepilotdConfig } from '../config/schema.js'
import { buildMemoryHttpHeaders } from '../memory/vector-backend.js'

export interface RagSearchHitForRerank {
  documentId: string
  folderId: string
  title: string
  score: number
  snippet: string
  path?: string
}

export interface ResolvedRagSearchSettings {
  limit: number
  candidateLimit: number
  scoreThreshold: number
  rerank: NonNullable<NonNullable<SepilotdConfig['memory']['rag']>['rerank']>
}

const DEFAULT_RAG_SETTINGS = {
  defaultLimit: 8,
  candidateMultiplier: 5,
  scoreThreshold: 0,
  rerank: {
    enabled: false,
    provider: 'local' as const,
    headers: {},
    timeoutMs: 10_000,
    weight: 0.35,
  },
}

function clampInteger(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Math.trunc(value)))
}

function tokenize(value: string): string[] {
  return Array.from(
    new Set(
      value
        .toLowerCase()
        .match(/[\p{L}\p{N}]+/gu) ?? [],
    ),
  )
}

function localRerankScore(
  queryTokens: string[],
  hit: RagSearchHitForRerank,
  weight: number,
): number {
  if (queryTokens.length === 0) return hit.score
  const haystack = `${hit.title} ${hit.path ?? ''} ${hit.snippet}`.toLowerCase()
  const lexicalScore =
    queryTokens.filter((token) => haystack.includes(token)).length
    / queryTokens.length
  return (hit.score * (1 - weight)) + (lexicalScore * weight)
}

export function resolveRagSearchSettings(
  config: SepilotdConfig | undefined,
  requestedLimit: number | undefined,
): ResolvedRagSearchSettings {
  const rag = config?.memory.rag ?? DEFAULT_RAG_SETTINGS
  const limit = clampInteger(
    requestedLimit ?? rag.defaultLimit ?? DEFAULT_RAG_SETTINGS.defaultLimit,
    1,
    50,
  )
  const candidateMultiplier = clampInteger(
    rag.candidateMultiplier ?? DEFAULT_RAG_SETTINGS.candidateMultiplier,
    1,
    20,
  )
  return {
    limit,
    candidateLimit: clampInteger(limit * candidateMultiplier, limit, 250),
    scoreThreshold: Math.max(0, Math.min(1, rag.scoreThreshold ?? 0)),
    rerank: {
      ...DEFAULT_RAG_SETTINGS.rerank,
      ...(rag.rerank ?? {}),
    },
  }
}

export function localRerankRagHits(
  query: string,
  hits: RagSearchHitForRerank[],
  weight: number,
): RagSearchHitForRerank[] {
  const queryTokens = tokenize(query)
  return hits
    .map((hit, index) => ({
      hit: {
        ...hit,
        score: localRerankScore(queryTokens, hit, weight),
      },
      index,
    }))
    .sort((left, right) => (
      right.hit.score - left.hit.score || left.index - right.index
    ))
    .map((entry) => entry.hit)
}

async function customApiRerankRagHits(
  query: string,
  hits: RagSearchHitForRerank[],
  rerank: ResolvedRagSearchSettings['rerank'],
): Promise<RagSearchHitForRerank[] | null> {
  if (!rerank.endpoint) return null
  const headers = buildMemoryHttpHeaders({
    apiKey: rerank.apiKey,
    auth: rerank.auth,
    contentType: 'application/json',
    headers: rerank.headers,
  })

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), rerank.timeoutMs ?? 10_000)
  try {
    const response = await fetch(rerank.endpoint, {
      method: 'POST',
      headers,
      signal: controller.signal,
      body: JSON.stringify({
        query,
        model: rerank.model,
        hits,
      }),
    })
    if (!response.ok) return null
    const payload = await response.json() as {
      hits?: Array<{ documentId?: string; score?: number }>
      results?: Array<{ documentId?: string; score?: number }>
    }
    const order = payload.hits ?? payload.results ?? []
    if (order.length === 0) return null
    const byDocumentId = new Map(hits.map((hit) => [hit.documentId, hit]))
    const used = new Set<string>()
    const reranked = order.flatMap((item) => {
      if (!item.documentId) return []
      const hit = byDocumentId.get(item.documentId)
      if (!hit) return []
      used.add(item.documentId)
      return [{
        ...hit,
        score: typeof item.score === 'number' ? item.score : hit.score,
      }]
    })
    return [
      ...reranked,
      ...hits.filter((hit) => !used.has(hit.documentId)),
    ]
  } catch {
    return null
  } finally {
    clearTimeout(timer)
  }
}

export async function applyRagRerank(
  query: string,
  hits: RagSearchHitForRerank[],
  settings: ResolvedRagSearchSettings,
): Promise<RagSearchHitForRerank[]> {
  const filtered = settings.scoreThreshold > 0
    ? hits.filter((hit) => hit.score >= settings.scoreThreshold)
    : hits
  if (!settings.rerank.enabled) return filtered
  if (settings.rerank.provider === 'custom-api') {
    const reranked = await customApiRerankRagHits(
      query,
      filtered,
      settings.rerank,
    )
    if (reranked) return reranked
  }
  return localRerankRagHits(
    query,
    filtered,
    settings.rerank.weight ?? DEFAULT_RAG_SETTINGS.rerank.weight,
  )
}
