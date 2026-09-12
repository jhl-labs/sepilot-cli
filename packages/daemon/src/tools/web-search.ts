import type { ToolDefinitionRuntime, ToolResult } from './registry.js'
import { webSearchTrustedDomainSchema } from '../config/schema.js'
import {
  getAbortError,
  isAbortError,
  raceWithAbort,
  throwIfAborted,
} from '../abort.js'
import {
  discardSearchResponse,
  readSearchResponseJson,
  readSearchResponseText,
  resolveSearchProvider,
  runKeyedSearch,
  SearchProviderError,
  SearchResponseTooLargeError,
  type SearchHit,
  type SearchRetrievalEvidence,
  type SearchSourceKind,
  type WebSearchProviderSettings,
} from './web-search-providers.js'

export interface WebSearchToolOptions {
  getTrustedDomains?: () => readonly string[]
  /**
   * Which backend to search with. Read per execution so a settings change
   * takes effect without restarting the daemon, like the trusted domains.
   */
  getProviderSettings?: () => WebSearchProviderSettings
  /**
   * Bound the complete search attempt, including provider fallback work.
   * Search is a discovery aid: it must return control well before the global
   * tool backstop so the agent can try a known URL with webfetch/browser.
   */
  timeoutMs?: number
}

const DEFAULT_WEB_SEARCH_TIMEOUT_MS = 20_000

function configuredSearchCardinalityAliases(
  settings: WebSearchProviderSettings | undefined,
): string[] {
  if (!settings) return []
  const aliases = new Set<string>()
  if (settings.provider !== 'auto') aliases.add(settings.provider)
  const endpoint = settings.endpoint?.trim()
  if (endpoint) {
    try {
      const parsed = new URL(endpoint)
      if (parsed.hostname) aliases.add(parsed.hostname)
      if (parsed.host) aliases.add(parsed.host)
    } catch {
      // An invalid endpoint will fail through the normal provider contract.
      // It is not a safe structural identity for cardinality enforcement.
    }
  }
  return [...aliases]
}

type TrustedDomainPolicy = {
  restricted: boolean
  domains: string[]
}

export function createWebSearchTool(options: WebSearchToolOptions = {}): ToolDefinitionRuntime {
  return {
    name: 'web.search',
    description:
      "Keyword search through the daemon's configured search backend (not a real browser session). Set `sourceKind` to `internal-index` when the user specifically requests the configured internal knowledge index, or to `public-web` when public/official websites are required; omit it to use the configured provider. A public-web request bypasses an internal-only `ai-search` provider and uses the keyless public fallback, while an internal-index request fails closed if no internal provider is configured. If the user names the configured search service or provider as the place to search, use this tool; do not turn the query into a homepage GET with `webfetch`. Without a configured public provider this uses DuckDuckGo's keyless endpoints, which return sparse or empty results for recent news, model releases, benchmark numbers, and other time-sensitive or long-tail queries. When trusted search domains are configured, both the search query and returned source URLs are limited to those domains and their subdomains. Honor explicit methods: use this tool for search APIs, webfetch for a known URL HTTP read, and browser.remote_snapshot/browser.remote_action for a selected visible Chrome/Edge tab. Headless browser tools use a separate profile and do not satisfy a visible-tab request. For a combined request, search first and then open an observed result in the requested browser. Set provider when the user names a specific search provider; a mismatch fails without searching another provider.",
    cardinalityAliases: () => configuredSearchCardinalityAliases(
      options.getProviderSettings?.(),
    ),
    resumeSafety: 'replay-safe',
    scheduling: {
      mode: 'parallel-safe',
      resource: 'network',
      key: (input) => typeof input.query === 'string' ? input.query : null,
    },
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query' },
        provider: {
          type: 'string', enum: ['duckduckgo', 'brave', 'tavily', 'searxng', 'ai-search'],
          description: 'Required provider when explicitly named by the user. It must match the configured/resolved backend; does not change global settings or reuse credentials for another provider. Omit for automatic selection.',
        },
        maxResults: {
          type: 'number',
          minimum: 1,
          maximum: 10,
          description: 'Max results (default 5, range 1-10)',
        },
        sourceKind: {
          type: 'string',
          enum: ['public-web', 'internal-index'],
          description:
            'Required source boundary. Use public-web for public or official websites and internal-index for the configured internal knowledge index. Omit to use the configured provider.',
        },
      },
      required: ['query'],
    },
    async execute(input: Record<string, unknown>, context): Promise<ToolResult> {
      const start = Date.now()
      const query = typeof input.query === 'string' ? input.query.trim() : ''
      const requestedMaxResults =
        typeof input.maxResults === 'number' && Number.isFinite(input.maxResults)
          ? Math.trunc(input.maxResults)
          : 5
      const maxResults = Math.min(10, Math.max(1, requestedMaxResults))
      const requestedSourceKind = input.sourceKind === 'public-web'
        || input.sourceKind === 'internal-index'
        ? input.sourceKind satisfies SearchSourceKind
        : undefined
      if (!query) {
        return {
          output: 'query must be a non-empty string',
          status: 'error',
          durationMs: Date.now() - start,
          code: 'INVALID_QUERY_PERMANENT',
        }
      }

      const timeoutMs = Number.isFinite(options.timeoutMs)
        ? Math.max(1, Math.trunc(options.timeoutMs!))
        : DEFAULT_WEB_SEARCH_TIMEOUT_MS
      const controller = new AbortController()
      let timedOut = false
      const abortFromContext = () => controller.abort(context?.signal?.reason)
      if (context?.signal?.aborted) abortFromContext()
      else context?.signal?.addEventListener('abort', abortFromContext, { once: true })
      const timeout = setTimeout(() => {
        timedOut = true
        controller.abort(new Error(`web search timed out after ${timeoutMs}ms`))
      }, timeoutMs)
      timeout.unref?.()

      try {
        throwIfAborted(controller.signal, `Web search for ${query} aborted`)
        const domainPolicy = createTrustedDomainPolicy(options.getTrustedDomains?.() ?? [])
        const effectiveQuery = buildSearchQuery(query, domainPolicy)

        // No settings supplied means no provider configuration to honour at
        // all — including the environment, which the daemon reads only through
        // the settings it passes in. Keeps a bare tool instance deterministic.
        const settings = options.getProviderSettings?.()
        let provider = settings
          ? resolveSearchProvider(settings)
          : ({ id: 'duckduckgo' } as const)
        if (input.provider !== undefined && input.provider !== provider.id) {
          return {
            output: `The requested search provider is ${String(input.provider)}, but the configured/resolved provider is ${provider.id}. No search was sent. Configure the requested provider and its credentials, or ask the user before using another provider.`,
            status: 'error', durationMs: Date.now() - start,
            code: 'SEARCH_PROVIDER_UNAVAILABLE_PERMANENT',
          }
        }
        if (input.provider === 'ai-search' && requestedSourceKind === 'public-web') {
          return {
            output: 'The explicitly requested ai-search provider is an internal index and cannot satisfy public-web search. No search was sent. Clarify the requested source instead of substituting a provider.',
            status: 'error', durationMs: Date.now() - start,
            code: 'SEARCH_SOURCE_UNAVAILABLE_PERMANENT',
          }
        }
        if (requestedSourceKind === 'internal-index' && provider.id !== 'ai-search') {
          return {
            output:
              `Internal-index search was requested, but the configured web search provider is ${provider.id}. Configure ai-search or request public-web sources.`,
            status: 'error',
            durationMs: Date.now() - start,
            code: 'SEARCH_SOURCE_UNAVAILABLE_PERMANENT',
          }
        }
        if (requestedSourceKind === 'public-web' && provider.id === 'ai-search') {
          provider = { id: 'duckduckgo' }
        }
        if (provider.id !== 'duckduckgo') {
          const providerResult = await raceWithAbort(
            runKeyedSearch(provider, effectiveQuery, maxResults, controller.signal),
            controller.signal,
            `Web search for ${query} aborted`,
          )
          const formatted = formatSearchHits(providerResult.hits, maxResults, domainPolicy)
          const metadata = webSearchMetadata(providerResult.evidence, formatted.length)
          const retrievalEvidence = formatSearchRetrievalEvidence(
            providerResult.evidence,
            formatted.length,
          )
          if (formatted.length === 0) {
            return {
              output: [
                retrievalEvidence,
                `No results found for: ${query}${restrictionSuffix(domainPolicy)}`,
              ].join('\n\n'),
              status: 'success',
              durationMs: Date.now() - start,
              metadata,
            }
          }
          return {
            output: [retrievalEvidence, ...formatted].join('\n\n'),
            status: 'success',
            durationMs: Date.now() - start,
            metadata,
          }
        }

        // Use DuckDuckGo instant answer API (no API key needed)
        const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(effectiveQuery)}&format=json&no_html=1&skip_disambig=1`
        const res = await raceWithAbort(
          fetch(url, { signal: controller.signal }),
          controller.signal,
          `Web search for ${query} aborted`,
        )

        if (!res.ok) {
          discardSearchResponse(res)
          return { output: `Search failed: ${res.status}`, status: 'error', durationMs: Date.now() - start }
        }
        // 202 here is the same anti-bot response the HTML endpoint returns; the
        // body is not the JSON payload this path expects.
        if (res.status !== 200) {
          discardSearchResponse(res)
          throw new SearchBackendUnavailableError(res.status)
        }

        const data = await readSearchResponseJson<{
          Abstract?: string
          AbstractText?: string
          AbstractURL?: string
          RelatedTopics?: Array<{ Text?: string; FirstURL?: string }>
        }>('duckduckgo', res, controller.signal)
        const results: string[] = []

        // Abstract (main answer)
        const abstract = data.AbstractText || data.Abstract
        const abstractSource = normalizeSearchHref(data.AbstractURL ?? '', domainPolicy)
        if (abstract && (!domainPolicy.restricted || abstractSource)) {
          results.push(
            `${escapeMarkdownInlineText(abstract)}${abstractSource ? `\nSource: ${abstractSource}` : ''}`,
          )
        }

        // Related topics
        if (data.RelatedTopics) {
          let topicCount = 0
          for (const topic of flattenRelatedTopics(data.RelatedTopics)) {
            if (topicCount >= maxResults) break
            const source = normalizeSearchHref(topic.FirstURL ?? '', domainPolicy)
            if (!topic.Text || (domainPolicy.restricted && !source)) continue
            results.push(`- ${escapeMarkdownInlineText(topic.Text)}${source ? ` (${source})` : ''}`)
            topicCount += 1
          }
        }

        if (results.length === 0) {
          const htmlResults = await raceWithAbort(
            searchDuckDuckGoHtml(
              effectiveQuery,
              maxResults,
              domainPolicy,
              controller.signal,
            ),
            controller.signal,
            `Web search for ${query} aborted`,
          )
          results.push(...htmlResults)
        }

        if (results.length === 0) {
          const evidence: SearchRetrievalEvidence = {
            provider: 'duckduckgo',
            sourceKind: 'public-web',
            providerResultCount: 0,
          }
          return {
            output: [
              formatSearchRetrievalEvidence(evidence, 0),
              `No results found for: ${query}${restrictionSuffix(domainPolicy)}`,
            ].join('\n\n'),
            status: 'success',
            durationMs: Date.now() - start,
            metadata: webSearchMetadata(evidence, 0),
          }
        }

        const evidence: SearchRetrievalEvidence = {
          provider: 'duckduckgo',
          sourceKind: 'public-web',
          providerResultCount: results.length,
        }
        return {
          output: [
            formatSearchRetrievalEvidence(evidence, results.length),
            ...results,
          ].join('\n\n'),
          status: 'success',
          durationMs: Date.now() - start,
          metadata: webSearchMetadata(evidence, results.length),
        }
      } catch (err) {
        if (timedOut) {
          return {
            output: [
              `Web search timed out after ${timeoutMs}ms.`,
              'If the request or prior evidence provides a likely source URL, use webfetch or browser navigation directly instead of extending this search deadline.',
            ].join(' '),
            status: 'error',
            durationMs: Date.now() - start,
            code: 'TIMEOUT_TRANSIENT',
          }
        }
        if (isAbortError(err) || context?.signal?.aborted) {
          throw getAbortError(context?.signal, `Web search for ${query} aborted`)
        }
        const message = err instanceof Error ? err.message : String(err)
        if (err instanceof SearchResponseTooLargeError) {
          return {
            output: `Search error: ${message}`,
            status: 'error',
            durationMs: Date.now() - start,
            code: 'SEARCH_RESPONSE_TOO_LARGE_PERMANENT',
          }
        }
        // A configured provider's failure is reported as itself. Quietly
        // retrying the query on DuckDuckGo would turn a rejected API key into
        // "the web has nothing", which is exactly the class of silent
        // degradation this tool already guards against.
        if (err instanceof SearchProviderError) {
          return {
            output: `Search error: ${message}`,
            status: 'error',
            durationMs: Date.now() - start,
            code: err.retriable
              ? 'SEARCH_BACKEND_UNAVAILABLE_TRANSIENT'
              : 'SEARCH_PROVIDER_MISCONFIGURED_PERMANENT',
          }
        }
        if (err instanceof SearchBackendUnavailableError) {
          return {
            output: `Search error: ${message}`,
            status: 'error',
            durationMs: Date.now() - start,
            // Transient so the agent's own error-code guidance lets it retry
            // once rather than treating search as permanently gone.
            code: 'SEARCH_BACKEND_UNAVAILABLE_TRANSIENT',
          }
        }
        return { output: `Search error: ${message}`, status: 'error', durationMs: Date.now() - start }
      } finally {
        clearTimeout(timeout)
        context?.signal?.removeEventListener('abort', abortFromContext)
      }
    },
  }
}

function webSearchMetadata(
  evidence: SearchRetrievalEvidence,
  returnedResultCount: number,
): { webSearch: SearchRetrievalEvidence & { returnedResultCount: number } } {
  return {
    webSearch: {
      ...evidence,
      returnedResultCount,
    },
  }
}

/**
 * Tool metadata is authoritative for journals and policy, but the next model
 * turn only sees the tool output. Project the same bounded provider/source
 * boundary into that output so synthesis cannot silently relabel an internal
 * index as public web (or the reverse). Every field here comes from validated
 * daemon-owned enums and counts; arbitrary provider prose is never copied.
 */
function formatSearchRetrievalEvidence(
  evidence: SearchRetrievalEvidence,
  returnedResultCount: number,
): string {
  const sourceBoundary = evidence.sourceKind === 'internal-index'
    ? 'configured internal index'
    : 'public web'
  return [
    '[Search retrieval evidence]',
    `Provider: ${evidence.provider}`,
    `Source kind: ${evidence.sourceKind} (${sourceBoundary})`,
    `Results: provider=${evidence.providerResultCount}, returned=${returnedResultCount}`,
  ].join('\n')
}

function restrictionSuffix(domainPolicy: TrustedDomainPolicy): string {
  if (!domainPolicy.restricted) return ''
  return ` within trusted search domains: ${domainPolicy.domains.join(', ') || '(invalid configuration)'}`
}

/**
 * Keyed providers return structured hits, but the trusted-domain boundary has
 * to hold for them too: `site:` filters in the query are the provider's best
 * effort, not a guarantee, so every URL is checked again here.
 */
function formatSearchHits(
  hits: SearchHit[],
  maxResults: number,
  domainPolicy: TrustedDomainPolicy,
): string[] {
  const formatted: string[] = []
  const seen = new Set<string>()
  for (const hit of hits) {
    if (formatted.length >= maxResults) break
    const source = normalizeSearchHref(hit.url, domainPolicy)
    if (!source || seen.has(source)) continue
    seen.add(source)
    const title = escapeMarkdownInlineText(hit.title.trim() || source)
    const snippet = hit.snippet?.trim()
    const provenance = hit.provenance?.trim()
    formatted.push(
      `- ${title}\n  Source: ${source}`
      + `${provenance ? `\n  Provenance: ${escapeMarkdownInlineText(provenance)}` : ''}`
      + `${snippet ? `\n  ${escapeMarkdownInlineText(snippet)}` : ''}`,
    )
  }
  return formatted
}

type DuckDuckGoTopic = {
  Text?: string
  FirstURL?: string
  Topics?: DuckDuckGoTopic[]
}

function flattenRelatedTopics(topics: DuckDuckGoTopic[]): DuckDuckGoTopic[] {
  const flattened: DuckDuckGoTopic[] = []
  for (const topic of topics) {
    if (topic.Text) flattened.push(topic)
    if (topic.Topics?.length) flattened.push(...flattenRelatedTopics(topic.Topics))
  }
  return flattened
}

/**
 * DuckDuckGo answers a rate-limited or bot-suspected scrape with HTTP 202 and
 * an ordinary-looking landing page rather than an error status. `res.ok` is
 * true for 202, so without this check the challenge page is parsed as a result
 * set, yields nothing, and the caller is told "no results found" — which reads
 * as "the web has no answer" instead of "search is unavailable right now".
 */
function isSearchBackendChallenge(status: number, body: string): boolean {
  if (status !== 200) return true
  return /\banomaly\b|\bcaptcha\b|challenge-form/i.test(body)
}

export class SearchBackendUnavailableError extends Error {
  constructor(status: number) {
    super(
      `web search backend refused the request (HTTP ${status}); it is rate-limiting `
        + 'or blocking automated queries. Retry later, or configure a search '
        + 'provider with an API key.',
    )
    this.name = 'SearchBackendUnavailableError'
  }
}

async function searchDuckDuckGoHtml(
  query: string,
  maxResults: number,
  domainPolicy: TrustedDomainPolicy,
  signal: AbortSignal | undefined,
): Promise<string[]> {
  throwIfAborted(signal, `Web search for ${query} aborted`)
  const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`
  const res = await fetch(url, {
    signal,
    headers: { 'user-agent': 'sepilotd-web-search/1' },
  })
  if (!res.ok) {
    discardSearchResponse(res)
    return []
  }
  const html = await readSearchResponseText('duckduckgo', res, signal)
  if (isSearchBackendChallenge(res.status, html)) {
    throw new SearchBackendUnavailableError(res.status)
  }
  return extractHtmlSearchResults(html, maxResults, domainPolicy)
}

function extractHtmlSearchResults(
  html: string,
  maxResults: number,
  domainPolicy: TrustedDomainPolicy,
): string[] {
  const results: string[] = []
  const seen = new Set<string>()
  const anchorPattern = /<a\b[^>]*href=(["'])(.*?)\1[^>]*>([\s\S]*?)<\/a>/gi
  let match: RegExpExecArray | null
  while ((match = anchorPattern.exec(html)) && results.length < maxResults) {
    const title = escapeMarkdownInlineText(htmlToText(match[3]))
    const source = normalizeSearchHref(match[2], domainPolicy)
    if (!title || !source || seen.has(source)) continue
    seen.add(source)
    results.push(`- ${title}\n  Source: ${source}`)
  }
  return results
}

function normalizeSearchHref(
  rawHref: string,
  domainPolicy: TrustedDomainPolicy,
): string | null {
  if (!rawHref) return null
  let href = decodeHtmlEntities(rawHref)
  if (href.startsWith('//')) href = `https:${href}`
  if (href.startsWith('/')) href = `https://duckduckgo.com${href}`
  try {
    const url = new URL(href)
    const redirected = url.searchParams.get('uddg')
    const resolved = redirected ? new URL(redirected) : url
    if (resolved.protocol !== 'http:' && resolved.protocol !== 'https:') return null
    if (resolved.username || resolved.password) return null
    if (resolved.hostname.endsWith('duckduckgo.com') && !redirected) return null
    if (!isTrustedSearchHostname(resolved.hostname, domainPolicy)) return null
    return resolved.toString()
  } catch {
    return null
  }
}

function createTrustedDomainPolicy(rawDomains: readonly string[]): TrustedDomainPolicy {
  const domains = [
    ...new Set(
      rawDomains
        .map((domain) => normalizeTrustedDomain(domain))
        .filter((domain): domain is string => domain !== null),
    ),
  ]
  return {
    // Config validation prevents malformed domains. Keeping the restriction
    // active for any non-empty direct caller input makes the tool fail closed
    // if that boundary is bypassed.
    restricted: rawDomains.length > 0,
    domains,
  }
}

function normalizeTrustedDomain(rawDomain: string): string | null {
  const parsed = webSearchTrustedDomainSchema.safeParse(rawDomain)
  return parsed.success ? parsed.data : null
}

function buildSearchQuery(query: string, domainPolicy: TrustedDomainPolicy): string {
  if (domainPolicy.domains.length === 0) return query
  const siteFilters = domainPolicy.domains.map((domain) => `site:${domain}`)
  return siteFilters.length === 1
    ? `${query} ${siteFilters[0]}`
    : `${query} (${siteFilters.join(' OR ')})`
}

function isTrustedSearchHostname(
  rawHostname: string,
  domainPolicy: TrustedDomainPolicy,
): boolean {
  if (!domainPolicy.restricted) return true
  const hostname = rawHostname.toLowerCase().replace(/\.$/, '')
  return domainPolicy.domains.some(
    (domain) => hostname === domain || hostname.endsWith(`.${domain}`),
  )
}

function htmlToText(html: string): string {
  return decodeHtmlEntities(html)
    .replace(/<script[\s\S]*?<\/script>/gi, '')
    .replace(/<style[\s\S]*?<\/style>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/**
 * Search-provider labels and snippets are model-facing Markdown. Preserve
 * their visible text while preventing a literal pipe from becoming a table
 * delimiter when the assistant copies a result into a Markdown table.
 */
function escapeMarkdownInlineText(value: string): string {
  return value.replace(/\\/g, '\\\\').replace(/\|/g, '\\|')
}

function decodeHtmlEntities(value: string): string {
  return value
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
}
