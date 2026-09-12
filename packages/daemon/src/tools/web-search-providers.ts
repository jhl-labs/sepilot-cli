/**
 * Keyed web-search backends.
 *
 * The keyless DuckDuckGo path is the default, but it answers automated queries
 * with an anti-bot challenge often enough that a personal assistant cannot rely
 * on it. Anyone who needs search to actually work configures a provider with an
 * API key here; DuckDuckGo stays as the no-configuration fallback.
 *
 * A configured provider never silently falls back to DuckDuckGo — a wrong key
 * or a spent quota has to look like a broken key, not like an empty web.
 */

import { getDispatcherCompatibleFetch } from '../providers/http-timeout.js'
import { raceWithAbort } from '../abort.js'

export type WebSearchProviderId =
  | 'auto'
  | 'duckduckgo'
  | 'brave'
  | 'tavily'
  | 'searxng'
  | 'ai-search'

export interface WebSearchProviderSettings {
  provider: WebSearchProviderId
  apiKey?: string
  /** Base URL of a self-hosted SearXNG or compatible AI Search instance. */
  endpoint?: string
}

export interface SearchHit {
  title: string
  url: string
  snippet?: string
  provenance?: string
}

export type SearchSourceKind = 'public-web' | 'internal-index'

export interface SearchFlowStageEvidence {
  id: string
  status?: string
}

export interface SearchTimingEvidence {
  name: string
  value: number
}

export interface SearchRetrievalEvidence {
  provider: Exclude<WebSearchProviderId, 'auto'>
  sourceKind: SearchSourceKind
  /** Valid-URL hits accepted from the provider before trusted-domain filtering and deduplication. */
  providerResultCount: number
  /** Provider-reported total matches, when the backend exposes one. */
  providerTotal?: number
  /** Provider-reported end-to-end retrieval latency, distinct from tool durationMs. */
  backendDurationMs?: number
  /** Bounded provider workflow evidence; arbitrary labels/details are intentionally omitted. */
  flow?: SearchFlowStageEvidence[]
  /** Raw provider flow length, so truncation or invalid entries remain visible. */
  flowStageCount?: number
  flowTruncated?: boolean
  /** Bounded numeric provider timing entries. Names retain any provider-declared unit suffix. */
  timings?: SearchTimingEvidence[]
  /** Raw provider timing-key count, including entries rejected by normalization. */
  timingCount?: number
  timingsTruncated?: boolean
}

export interface SearchProviderResult {
  hits: SearchHit[]
  evidence: SearchRetrievalEvidence
}

/** The backend a request will actually use, after env fallbacks and `auto`. */
export type ResolvedSearchProvider =
  | { id: 'duckduckgo' }
  | { id: 'brave'; apiKey: string }
  | { id: 'tavily'; apiKey: string }
  | { id: 'searxng'; endpoint: string }
  | { id: 'ai-search'; endpoint: string; apiKey?: string }

export class SearchProviderError extends Error {
  readonly retriable: boolean
  constructor(message: string, retriable: boolean) {
    super(message)
    this.name = 'SearchProviderError'
    this.retriable = retriable
  }
}

function searchFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  return getDispatcherCompatibleFetch()(input, init)
}

export const MAX_WEB_SEARCH_RESPONSE_BYTES = 2 * 1024 * 1024

export class SearchResponseTooLargeError extends Error {
  constructor(provider: string) {
    super(
      `${provider} search response exceeded the ${MAX_WEB_SEARCH_RESPONSE_BYTES}-byte limit`,
    )
    this.name = 'SearchResponseTooLargeError'
  }
}

function discardSearchResponseBody(res: Response): void {
  try {
    const cancellation = res.body?.cancel()
    if (cancellation) void cancellation.catch(() => undefined)
  } catch {
    // Best effort only. The request signal still owns transport cancellation.
  }
}

function discardSearchResponseReader(
  reader: ReadableStreamDefaultReader<Uint8Array>,
): void {
  try {
    void reader.cancel().catch(() => undefined)
  } catch {
    // Best effort only. The request signal still owns transport cancellation.
  }
}

export async function readSearchResponseText(
  provider: string,
  res: Response,
  signal?: AbortSignal,
): Promise<string> {
  const declaredLength = Number(res.headers.get('content-length'))
  if (Number.isFinite(declaredLength) && declaredLength > MAX_WEB_SEARCH_RESPONSE_BYTES) {
    discardSearchResponseBody(res)
    throw new SearchResponseTooLargeError(provider)
  }

  const reader = res.body?.getReader()
  if (!reader) {
    const text = await raceWithAbort(res.text(), signal, `${provider} search response aborted`)
    if (Buffer.byteLength(text, 'utf8') > MAX_WEB_SEARCH_RESPONSE_BYTES) {
      throw new SearchResponseTooLargeError(provider)
    }
    return text
  }

  const chunks: Uint8Array[] = []
  let totalBytes = 0
  try {
    for (;;) {
      const { done, value } = await raceWithAbort(
        reader.read(),
        signal,
        `${provider} search response aborted`,
      )
      if (done) break
      if (!value) continue
      totalBytes += value.byteLength
      if (totalBytes > MAX_WEB_SEARCH_RESPONSE_BYTES) {
        throw new SearchResponseTooLargeError(provider)
      }
      chunks.push(value)
    }
  } catch (error) {
    discardSearchResponseReader(reader)
    throw error
  }

  const bytes = new Uint8Array(totalBytes)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  return new TextDecoder('utf-8', { fatal: false }).decode(bytes)
}

export async function readSearchResponseJson<T>(
  provider: string,
  res: Response,
  signal?: AbortSignal,
): Promise<T> {
  const text = await readSearchResponseText(provider, res, signal)
  try {
    return JSON.parse(text) as T
  } catch {
    throw new SearchProviderError(`${provider} returned invalid JSON`, false)
  }
}

export function discardSearchResponse(res: Response): void {
  discardSearchResponseBody(res)
}

const trimmed = (value: string | undefined): string | undefined => {
  const text = value?.trim()
  return text ? text : undefined
}

/**
 * Config wins over the environment; the environment exists so a key can be kept
 * out of `config.yaml`. The provider-specific names are the ones each vendor's
 * own docs use, so an existing shell profile tends to just work.
 */
export function resolveSearchProvider(
  settings: WebSearchProviderSettings,
  env: NodeJS.ProcessEnv = process.env,
): ResolvedSearchProvider {
  const generic = trimmed(settings.apiKey) ?? trimmed(env.SEPILOTD_WEB_SEARCH_API_KEY)
  const brave = generic ?? trimmed(env.BRAVE_SEARCH_API_KEY)
  const tavily = generic ?? trimmed(env.TAVILY_API_KEY)
  const searxng = trimmed(settings.endpoint) ?? trimmed(env.SEARXNG_URL)
  const aiSearchEndpoint = trimmed(settings.endpoint) ?? trimmed(env.AI_SEARCH_URL)
  const aiSearchApiKey = generic ?? trimmed(env.AI_SEARCH_API_KEY)

  switch (settings.provider) {
    case 'brave':
      if (!brave) throw missingCredential('brave', 'an API key', 'webSearch.apiKey or BRAVE_SEARCH_API_KEY')
      return { id: 'brave', apiKey: brave }
    case 'tavily':
      if (!tavily) throw missingCredential('tavily', 'an API key', 'webSearch.apiKey or TAVILY_API_KEY')
      return { id: 'tavily', apiKey: tavily }
    case 'searxng':
      if (!searxng) throw missingCredential('searxng', 'an instance URL', 'webSearch.endpoint or SEARXNG_URL')
      return { id: 'searxng', endpoint: searxng }
    case 'ai-search':
      if (!aiSearchEndpoint) {
        throw missingCredential('ai-search', 'an instance URL', 'webSearch.endpoint or AI_SEARCH_URL')
      }
      return { id: 'ai-search', endpoint: aiSearchEndpoint, apiKey: aiSearchApiKey }
    case 'duckduckgo':
      return { id: 'duckduckgo' }
    case 'auto':
    default:
      // Only a key that names its provider can select one under `auto`; a
      // generic key would be ambiguous, so it needs an explicit provider.
      if (trimmed(env.BRAVE_SEARCH_API_KEY)) return { id: 'brave', apiKey: trimmed(env.BRAVE_SEARCH_API_KEY)! }
      if (trimmed(env.TAVILY_API_KEY)) return { id: 'tavily', apiKey: trimmed(env.TAVILY_API_KEY)! }
      if (trimmed(env.AI_SEARCH_URL)) {
        return {
          id: 'ai-search',
          endpoint: trimmed(env.AI_SEARCH_URL)!,
          apiKey: trimmed(env.AI_SEARCH_API_KEY),
        }
      }
      if (searxng) return { id: 'searxng', endpoint: searxng }
      return { id: 'duckduckgo' }
  }
}

function missingCredential(provider: string, what: string, where: string): SearchProviderError {
  return new SearchProviderError(
    `web search provider "${provider}" is selected but ${what} is not configured; set ${where}`,
    false,
  )
}

export async function runKeyedSearch(
  provider: Exclude<ResolvedSearchProvider, { id: 'duckduckgo' }>,
  query: string,
  maxResults: number,
  signal: AbortSignal | undefined,
): Promise<SearchProviderResult> {
  switch (provider.id) {
    case 'brave': {
      const hits = await searchBrave(provider.apiKey, query, maxResults, signal)
      return publicSearchResult('brave', hits)
    }
    case 'tavily': {
      const hits = await searchTavily(provider.apiKey, query, maxResults, signal)
      return publicSearchResult('tavily', hits)
    }
    case 'searxng': {
      const hits = await searchSearxng(provider.endpoint, query, maxResults, signal)
      return publicSearchResult('searxng', hits)
    }
    case 'ai-search':
      return searchAiSearch(provider.endpoint, provider.apiKey, query, maxResults, signal)
  }
}

function publicSearchResult(
  provider: 'brave' | 'tavily' | 'searxng',
  hits: SearchHit[],
): SearchProviderResult {
  return {
    hits,
    evidence: {
      provider,
      sourceKind: 'public-web',
      providerResultCount: hits.length,
    },
  }
}

async function readFailure(provider: string, res: Response): Promise<SearchProviderError> {
  // Quota and rate limits recover on their own; a rejected key does not, and
  // retrying it just burns the run's remaining iterations.
  const retriable = res.status === 429 || res.status >= 500
  const rejected = res.status === 401 || res.status === 403
  const detail = rejected
    ? provider === 'searxng'
      // SearXNG has no key: a refusal means the instance is not serving the
      // JSON API, which is off by default.
      ? ' — enable the json format in the instance settings.yml (search.formats)'
      : ' — check the API key'
    : res.status === 429
      ? ' — rate limited or out of quota'
      : ''
  discardSearchResponseBody(res)
  return new SearchProviderError(
    `${provider} search failed (HTTP ${res.status})${detail}`,
    retriable,
  )
}

async function searchBrave(
  apiKey: string,
  query: string,
  maxResults: number,
  signal: AbortSignal | undefined,
): Promise<SearchHit[]> {
  const url = new URL('https://api.search.brave.com/res/v1/web/search')
  url.searchParams.set('q', query)
  url.searchParams.set('count', String(maxResults))
  const res = await searchFetch(url, {
    signal,
    headers: {
      accept: 'application/json',
      'accept-encoding': 'gzip',
      'x-subscription-token': apiKey,
    },
  })
  if (!res.ok) throw await readFailure('brave', res)
  const data = await readSearchResponseJson<{
    web?: { results?: Array<{ title?: string; url?: string; description?: string }> }
  }>('brave', res, signal)
  return (data.web?.results ?? [])
    .map((result) => ({
      title: stripTags(result.title ?? ''),
      url: result.url ?? '',
      snippet: stripTags(result.description ?? ''),
    }))
    .filter((hit) => hit.url)
}

async function searchTavily(
  apiKey: string,
  query: string,
  maxResults: number,
  signal: AbortSignal | undefined,
): Promise<SearchHit[]> {
  const res = await searchFetch('https://api.tavily.com/search', {
    method: 'POST',
    signal,
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      query,
      max_results: maxResults,
      search_depth: 'basic',
    }),
  })
  if (!res.ok) throw await readFailure('tavily', res)
  const data = await readSearchResponseJson<{
    answer?: string
    results?: Array<{ title?: string; url?: string; content?: string }>
  }>('tavily', res, signal)
  const hits = (data.results ?? [])
    .map((result) => ({
      title: result.title ?? result.url ?? '',
      url: result.url ?? '',
      snippet: result.content ?? '',
    }))
    .filter((hit) => hit.url)
  return hits
}

async function searchSearxng(
  endpoint: string,
  query: string,
  maxResults: number,
  signal: AbortSignal | undefined,
): Promise<SearchHit[]> {
  const url = new URL('/search', endpoint.endsWith('/') ? endpoint : `${endpoint}/`)
  url.searchParams.set('q', query)
  url.searchParams.set('format', 'json')
  const res = await searchFetch(url, { signal, headers: { accept: 'application/json' } })
  if (!res.ok) throw await readFailure('searxng', res)
  const data = await readSearchResponseJson<{
    results?: Array<{ title?: string; url?: string; content?: string }>
  }>('searxng', res, signal)
  return (data.results ?? [])
    .slice(0, maxResults)
    .map((result) => ({
      title: result.title ?? result.url ?? '',
      url: result.url ?? '',
      snippet: result.content ?? '',
    }))
    .filter((hit) => hit.url)
}

async function searchAiSearch(
  endpoint: string,
  apiKey: string | undefined,
  query: string,
  maxResults: number,
  signal: AbortSignal | undefined,
): Promise<SearchProviderResult> {
  const url = new URL('/v1/search', endpoint.endsWith('/') ? endpoint : `${endpoint}/`)
  const headers = new Headers({
    accept: 'application/json',
    'content-type': 'application/json',
  })
  if (apiKey) headers.set('authorization', `Bearer ${apiKey}`)

  const res = await searchFetch(url, {
    method: 'POST',
    signal,
    headers,
    body: JSON.stringify({
      query,
      max_results: maxResults,
      search_type: 'hybrid',
      min_tier: 4,
      sort: 'relevance',
      enhance: false,
    }),
  })
  if (!res.ok) throw await readFailure('ai-search', res)

  const data = await readSearchResponseJson<{
    results?: Array<{
      title?: string
      url?: string
      snippet?: string
      content?: string
      tier?: number
      doc_type?: string
      source?: string
      domain?: string
      published_at?: string | null
    }>
    total?: number
    took_ms?: number
    flow?: unknown
    timings?: unknown
  }>('ai-search', res, signal)
  if (!Array.isArray(data.results)) {
    throw new SearchProviderError('ai-search response is missing a results array', false)
  }

  const providerTotal = nonNegativeInteger(data.total)
  const backendDurationMs = nonNegativeInteger(data.took_ms)
  if (providerTotal === undefined || backendDurationMs === undefined) {
    throw new SearchProviderError(
      'ai-search response is missing valid total/took_ms retrieval evidence',
      false,
    )
  }
  if (data.flow !== undefined && data.flow !== null && !Array.isArray(data.flow)) {
    throw new SearchProviderError('ai-search response contains an invalid flow array', false)
  }
  if (
    data.timings !== undefined
    && data.timings !== null
    && (!isRecord(data.timings) || Array.isArray(data.timings))
  ) {
    throw new SearchProviderError('ai-search response contains an invalid timings object', false)
  }

  const hits = data.results
    .slice(0, maxResults)
    .map((result) => {
      const provenance = [
        result.source ? `source=${result.source}` : '',
        result.domain ? `domain=${result.domain}` : '',
        Number.isFinite(result.tier) ? `tier=${result.tier}` : '',
        result.doc_type ? `type=${result.doc_type}` : '',
        result.published_at ? `published=${result.published_at}` : '',
      ].filter(Boolean).join(', ')
      return {
        title: stripTags(result.title ?? result.url ?? ''),
        url: result.url ?? '',
        snippet: stripTags(result.snippet ?? result.content ?? ''),
        provenance: provenance || undefined,
      }
    })
    .filter((hit) => hit.url)

  const flow = normalizeFlowEvidence(data.flow)
  const timings = normalizeTimingEvidence(data.timings)
  return {
    hits,
    evidence: {
      provider: 'ai-search',
      sourceKind: 'internal-index',
      providerResultCount: hits.length,
      providerTotal,
      backendDurationMs,
      ...(flow
        ? {
            flow: flow.items,
            flowStageCount: flow.total,
            ...(flow.truncated ? { flowTruncated: true } : {}),
          }
        : {}),
      ...(timings
        ? {
            timings: timings.items,
            timingCount: timings.total,
            ...(timings.truncated ? { timingsTruncated: true } : {}),
          }
        : {}),
    },
  }
}

const MAX_FLOW_STAGES = 20
const MAX_TIMING_ENTRIES = 24
const MAX_EVIDENCE_TEXT_LENGTH = 64

function nonNegativeInteger(value: unknown): number | undefined {
  return typeof value === 'number'
    && Number.isFinite(value)
    && Number.isInteger(value)
    && value >= 0
    ? value
    : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function boundedEvidenceText(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  const normalized = value.trim().split(/\s+/u).join(' ')
  return normalized ? normalized.slice(0, MAX_EVIDENCE_TEXT_LENGTH) : undefined
}

function normalizeFlowEvidence(value: unknown): {
  items: SearchFlowStageEvidence[]
  total: number
  truncated: boolean
} | undefined {
  if (!Array.isArray(value)) return undefined
  const items: SearchFlowStageEvidence[] = []
  for (const raw of value.slice(0, MAX_FLOW_STAGES)) {
    if (!isRecord(raw)) continue
    const id = boundedEvidenceText(raw.id)
    if (!id) continue
    const status = boundedEvidenceText(raw.status)
    items.push({ id, ...(status ? { status } : {}) })
  }
  return {
    items,
    total: value.length,
    truncated: value.length > MAX_FLOW_STAGES,
  }
}

function normalizeTimingEvidence(value: unknown): {
  items: SearchTimingEvidence[]
  total: number
  truncated: boolean
} | undefined {
  if (!isRecord(value) || Array.isArray(value)) return undefined
  const entries = Object.entries(value)
  const items: SearchTimingEvidence[] = []
  for (const [rawName, rawValue] of entries.slice(0, MAX_TIMING_ENTRIES)) {
    const name = boundedEvidenceText(rawName)
    if (!name || typeof rawValue !== 'number' || !Number.isFinite(rawValue) || rawValue < 0) continue
    items.push({ name, value: rawValue })
  }
  return {
    items,
    total: entries.length,
    truncated: entries.length > MAX_TIMING_ENTRIES,
  }
}

function stripTags(value: string): string {
  return value.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim()
}
