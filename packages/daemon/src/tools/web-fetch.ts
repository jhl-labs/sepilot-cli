import type { Dispatcher } from 'undici/index.js'
import { getAbortError, raceWithAbort, throwIfAborted } from '../abort.js'
import { enforceOutputLimit } from './output-limit.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'
import {
  createEgressPolicy,
  EgressDeniedError,
  type EgressPolicyLogger,
} from '../utils/egress-policy.js'
import {
  assertPublicUrl,
  createManagedLoopbackDispatcher,
  createPinnedLookupDispatcher,
  fetchWithPinnedNetworkPolicy,
  type PublicUrlResolution,
} from '../utils/ssrf-guard.js'

const MAX_BYTES = 2 * 1024 * 1024
const TIMEOUT_MS = 20_000
const MAX_REDIRECTS = 5
const DISPATCHER_CLOSE_TIMEOUT_MS = 1_000
const DEFAULT_QUERY_CONTEXT_CHARS = 800
const MAX_QUERY_CONTEXT_CHARS = 2_000
const MAX_QUERY_WINDOWS = 12
const QUERY_MISS_EXCERPT_CHARS = 1_200
const MAX_QUERY_CHARS = 200
const MAX_LINK_MATCHES = 20

interface WebFetchQueryOptions {
  query?: string
  linkQuery?: string
  contextChars: number
}

export interface WebFetchToolOptions {
  checkUrl?: (url: string) => string | undefined
  egressAllowlist?: readonly string[] | null
  env?: Record<string, string | undefined>
  egressLogger?: EgressPolicyLogger
  allowLocalhost?: boolean
  isLocalhostAllowed?: (
    sessionId: string | undefined,
    rawUrl: string,
    workspaceRoot?: string,
  ) => boolean
  managedLoopbackSocketForUrl?: (
    sessionId: string | undefined,
    rawUrl: string,
    workspaceRoot?: string,
  ) => string | null
  fetcher?: typeof fetch
  timeoutMs?: number
}

function defaultWebFetcher(): typeof fetch {
  return fetchWithPinnedNetworkPolicy
}

function isLocalhostUrl(rawUrl: string): boolean {
  try {
    const hostname = new URL(rawUrl).hostname.replace(/^\[(.*)\]$/, '$1').toLowerCase()
    return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1'
  } catch {
    return false
  }
}

export function normalizeWebFetchInput(
  input: Record<string, unknown>,
): Record<string, unknown> {
  const rawUrl = String(input.url ?? '').trim()
  let url = rawUrl
  try {
    const parsed = new URL(rawUrl)
    parsed.hash = ''
    url = parsed.toString()
  } catch {
  }
  const query = typeof input.query === 'string'
    ? input.query.trim().slice(0, MAX_QUERY_CHARS)
    : ''
  const linkQuery = typeof input.linkQuery === 'string'
    ? input.linkQuery.trim().slice(0, MAX_QUERY_CHARS)
    : ''
  const requestedContextChars = typeof input.contextChars === 'number'
    && Number.isFinite(input.contextChars)
    ? Math.trunc(input.contextChars)
    : DEFAULT_QUERY_CONTEXT_CHARS
  const contextChars = Math.max(
    100,
    Math.min(MAX_QUERY_CONTEXT_CHARS, requestedContextChars),
  )
  return {
    url,
    ...(query ? { query, contextChars } : {}),
    ...(linkQuery ? { linkQuery } : {}),
    ...(input.refresh === true ? { refresh: true } : {}),
  }
}

export function createWebFetchTool(options: WebFetchToolOptions = {}): ToolDefinitionRuntime {
  const fetcher = options.fetcher ?? defaultWebFetcher()
  const egressPolicy = createEgressPolicy({
    allowlist: options.egressAllowlist,
    env: options.env,
    logger: options.egressLogger,
  })
  return {
    name: 'webfetch',
    description:
      'Fetch one already-known HTTP GET URL and return its textual content (HTML is stripped to text, JSON is returned verbatim). When a long successful result says middle lines were omitted, call the same URL with `query` set to one relevant literal term or phrase; this returns bounded matching character ranges from the complete fetched text. When an HTML page names a relevant destination but the stripped text hides its href, call the same URL with `linkQuery` set to literal anchor text or a URL fragment; this returns at most 20 resolved HTTP(S) links so you can follow the observed destination instead of guessing a path. Use only one of `query` and `linkQuery` per call. Each is a distinct observation, not a refresh. A successful same-URL result with the same extraction input is reused within the current user turn; set refresh=true only when an explicit elapsed-time, polling, or intervening-state boundary requires a fresh observation. This tool does not perform keyword search: when the user names a configured search service or provider, call `web.search` instead of constructing that service\'s homepage or query URL. Use webfetch for known static source pages, JSON/API GET endpoints, and raw text, including localhost services exposed by process.start managed loopback ports. Switch to browser.navigate or browser.extract when JavaScript must populate content (SPAs). Use terminal.run with network=loopback for other HTTP methods, headers, or request bodies. Will fail on auth-required pages, 5xx errors, non-text responses, or an extraction query with no literal match.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'web-fetch' },
    normalizeInput: normalizeWebFetchInput,
    observationCoverage: {
      covers: (observedInput, requestedInput) => {
        if (requestedInput.refresh === true) return false
        const observed = normalizeWebFetchInput(observedInput)
        const requested = normalizeWebFetchInput(requestedInput)
        return observed.url !== ''
          && observed.url === requested.url
          && observed.query === requested.query
          && observed.linkQuery === requested.linkQuery
          && observed.contextChars === requested.contextChars
      },
    },
    inputSchema: {
      type: 'object',
      properties: {
        url: { type: 'string' },
        query: {
          type: 'string',
          maxLength: MAX_QUERY_CHARS,
          description:
            'Optional case-insensitive literal term or phrase to extract from the complete decoded response. Use after a long result omitted the relevant middle; this is not a search-engine query.',
        },
        linkQuery: {
          type: 'string',
          maxLength: MAX_QUERY_CHARS,
          description:
            'Optional case-insensitive literal anchor text or URL fragment to match in an HTML response. Returns up to 20 resolved HTTP(S) links. Use when visible page text names a destination but hides its href; do not combine with query.',
        },
        contextChars: {
          type: 'integer',
          minimum: 100,
          maximum: MAX_QUERY_CONTEXT_CHARS,
          description:
            'Characters of surrounding context on each side of a query match. Defaults to 800 and applies only with query.',
        },
        refresh: {
          type: 'boolean',
          description:
            'Force a fresh GET only after an explicit polling/time/state-change boundary; omit for ordinary source verification so current-turn evidence can be reused.',
        },
      },
      required: ['url'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const normalizedInput = normalizeWebFetchInput(input)
      const url = String(normalizedInput.url ?? '')
      const query = typeof normalizedInput.query === 'string'
        ? normalizedInput.query
        : undefined
      const linkQuery = typeof normalizedInput.linkQuery === 'string'
        ? normalizedInput.linkQuery
        : undefined
      if (query && linkQuery) {
        return {
          output: 'Use only one extraction mode per request: query for text or linkQuery for HTML links.',
          status: 'error',
          durationMs: Date.now() - start,
          code: 'INVALID_EXTRACTION_MODE_PERMANENT',
        }
      }
      const queryOptions = query || linkQuery
        ? {
            ...(query ? { query } : {}),
            ...(linkQuery ? { linkQuery } : {}),
            contextChars: Number(normalizedInput.contextChars),
          }
        : undefined
      if (!/^https?:\/\//i.test(url)) {
        return {
          output: 'only http(s) urls are allowed',
          status: 'error',
          durationMs: Date.now() - start,
          code: 'INVALID_URL_PERMANENT',
        }
      }
      const ctrl = new AbortController()
      const timeoutMs = Number.isFinite(options.timeoutMs)
        ? Math.max(1, Math.trunc(options.timeoutMs!))
        : TIMEOUT_MS
      let timedOut = false
      const abortFromContext = () => ctrl.abort(context?.signal?.reason)
      if (context?.signal?.aborted) abortFromContext()
      else context?.signal?.addEventListener('abort', abortFromContext, { once: true })
      const t = setTimeout(() => {
        timedOut = true
        ctrl.abort(new Error(`webfetch timed out after ${timeoutMs}ms`))
      }, timeoutMs)
      t.unref?.()
      try {
        throwIfAborted(ctrl.signal, `Web fetch for ${url} aborted`)
        let currentUrl = url
        const localhostAllowed = options.allowLocalhost
          || options.isLocalhostAllowed?.(
            context?.sessionId,
            url,
            context?.workspaceRoot,
          )
        const allowedLocalhostOrigin = localhostAllowed && isLocalhostUrl(url)
          ? new URL(url).origin
          : null
        for (let hop = 0; ; hop += 1) {
          const accessDenied = options.checkUrl?.(currentUrl)
          if (accessDenied) return { output: accessDenied, status: 'error', code: 'ACCESS_DENIED_PERMANENT', durationMs: Date.now() - start }
          const current = new URL(currentUrl)
          const isAllowedLocalhost = Boolean(
            allowedLocalhostOrigin
            && current.origin === allowedLocalhostOrigin
            && isLocalhostUrl(currentUrl),
          )
          if (!isAllowedLocalhost) {
            if (isLocalhostUrl(currentUrl)) {
              return {
                output: [
                  `Local URL ${currentUrl} is not exposed by a live process.start managed-loopback service available to this session or strict workspace.`,
                  'Start or rediscover the server with process.start/process.sessions and declared loopback ports, then use webfetch or browser for GET requests. For curl or custom HTTP requests use terminal.run with network=loopback. If the process exited, inspect process.read and fix that failure before retrying HTTP.',
                ].join('\n'),
                status: 'error',
                durationMs: Date.now() - start,
                code: 'BLOCKED_LOCALHOST_PERMANENT',
              }
            }
            try {
              egressPolicy.assertAllowed(currentUrl, 'webfetch')
            } catch (e) {
              return {
                output: (e as Error).message,
                status: 'error',
                durationMs: Date.now() - start,
                code: e instanceof EgressDeniedError ? e.code : 'BLOCKED_URL_PERMANENT',
              }
            }
          }
          let resolution: PublicUrlResolution | null = null
          let dispatcher: Dispatcher | undefined
          if (isAllowedLocalhost) {
            dispatcher = createManagedLoopbackDispatcher(currentUrl)
          } else {
            try {
              resolution = await assertPublicUrl(currentUrl, ctrl.signal)
            } catch (e) {
              if (ctrl.signal.aborted) throw e
              return {
                output: (e as Error).message,
                status: 'error',
                durationMs: Date.now() - start,
                code: 'BLOCKED_URL_PERMANENT',
              }
            }
            dispatcher = createPinnedLookupDispatcher(resolution)
          }
          try {
            throwIfAborted(ctrl.signal, `Web fetch for ${currentUrl} aborted`)
            const res = await raceWithAbort(fetcher(currentUrl, {
                signal: ctrl.signal,
                headers: { 'user-agent': 'sepilotd-webfetch/1' },
                redirect: 'manual',
                ...(dispatcher ? { dispatcher } : {}),
                ...(isAllowedLocalhost
                  ? {
                      managedLoopbackSocketPath:
                        options.managedLoopbackSocketForUrl?.(
                          context?.sessionId,
                          currentUrl,
                          context?.workspaceRoot,
                        )
                        ?? undefined,
                    }
                  : {}),
              } as RequestInit & {
                dispatcher?: Dispatcher
                managedLoopbackSocketPath?: string
              }), ctrl.signal, `Web fetch for ${currentUrl} aborted`)
            if (res.status >= 300 && res.status < 400) {
              const loc = res.headers.get('location')
              if (loc) {
                if (hop >= MAX_REDIRECTS) {
                  await discardResponseBody(res, ctrl.signal)
                  return {
                    output: 'too many redirects',
                    status: 'error',
                    durationMs: Date.now() - start,
                    code: 'REDIRECT_LIMIT_PERMANENT',
                  }
                }
                let nextUrl: URL
                try {
                  nextUrl = new URL(loc, resolution?.url ?? current)
                } catch {
                  await discardResponseBody(res, ctrl.signal)
                  return {
                    output: 'invalid redirect URL',
                    status: 'error',
                    durationMs: Date.now() - start,
                    code: 'INVALID_REDIRECT_PERMANENT',
                  }
                }
                if ((resolution?.url ?? current).protocol === 'https:' && nextUrl.protocol !== 'https:') {
                  await discardResponseBody(res, ctrl.signal)
                  return {
                    output: 'HTTPS redirect downgrade is not allowed',
                    status: 'error',
                    durationMs: Date.now() - start,
                    code: 'REDIRECT_DOWNGRADE_PERMANENT',
                  }
                }
                await discardResponseBody(res, ctrl.signal)
                currentUrl = nextUrl.toString()
                continue
              }
            }
            return await handleResponse(
              res,
              start,
              ctrl.signal,
              queryOptions,
              resolution?.url.toString() ?? currentUrl,
            )
          } finally {
            await closeDispatcherBounded(dispatcher)
          }
        }
      } catch (e) {
        if (context?.signal?.aborted) {
          throw getAbortError(context.signal, `Web fetch for ${url} aborted`)
        }
        return {
          output: (e as Error).message,
          status: 'error',
          durationMs: Date.now() - start,
          code: timedOut ? 'TIMEOUT_TRANSIENT' : 'NETWORK_TRANSIENT',
        }
      } finally {
        clearTimeout(t)
        context?.signal?.removeEventListener('abort', abortFromContext)
      }
    },
  }
}

export async function closeDispatcherBounded(
  dispatcher: Pick<Dispatcher, 'close' | 'destroy'> | undefined,
  timeoutMs = DISPATCHER_CLOSE_TIMEOUT_MS,
): Promise<void> {
  if (!dispatcher) return

  let timeoutHandle: ReturnType<typeof setTimeout> | undefined
  let closePromise: Promise<boolean>
  try {
    closePromise = Promise.resolve(dispatcher.close()).then(
      () => true,
      () => false,
    )
  } catch {
    closePromise = Promise.resolve(false)
  }
  const closedGracefully = await Promise.race([
    closePromise,
    new Promise<false>((resolve) => {
      timeoutHandle = setTimeout(() => resolve(false), timeoutMs)
      timeoutHandle.unref?.()
    }),
  ])
  if (timeoutHandle) clearTimeout(timeoutHandle)
  if (closedGracefully) return

  try {
    void Promise.resolve(
      dispatcher.destroy(new Error('webfetch dispatcher close timed out')),
    ).catch(() => undefined)
  } catch {
  }
}

export function parseContentType(raw: string): { mime: string; charset: string } {
  const [mimeRaw, ...params] = raw.split(';')
  const mime = (mimeRaw ?? '').trim().toLowerCase()
  let charset = ''
  for (const param of params) {
    const match = param.trim().match(/^charset=(?:"([^"]+)"|([^"\s]+))$/i)
    if (match) charset = (match[1] ?? match[2] ?? '').trim().toLowerCase()
  }
  return { mime, charset }
}

export function isTextualContentType(mime: string): boolean {
  if (!mime) return true
  if (mime.startsWith('text/')) return true
  return /(?:json|xml|javascript|ecmascript|html|csv|graphql|yaml|x-ndjson|x-www-form-urlencoded|plain)/.test(
    mime,
  )
}

function decodeBody(bytes: Uint8Array, charset: string): string {
  const label = charset || 'utf-8'
  try {
    return new TextDecoder(label, { fatal: false }).decode(bytes)
  } catch {
    return new TextDecoder('utf-8', { fatal: false }).decode(bytes)
  }
}

async function readBodyBounded(
  res: Response,
  maxBytes: number,
  signal?: AbortSignal,
): Promise<{ bytes: Uint8Array; tooLarge: boolean }> {
  const reader = res.body?.getReader()
  if (!reader) return { bytes: new Uint8Array(0), tooLarge: false }
  const chunks: Uint8Array[] = []
  let total = 0
  try {
    for (;;) {
      const pendingRead = reader.read()
      const { done, value } = signal
        ? await raceWithAbort(pendingRead, signal)
        : await pendingRead
      if (done) break
      if (!value) continue
      total += value.byteLength
      if (total > maxBytes) {
        discardResponseReader(reader)
        return { bytes: new Uint8Array(0), tooLarge: true }
      }
      chunks.push(value)
    }
  } catch (error) {
    discardResponseReader(reader)
    throw error
  }
  const bytes = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  return { bytes, tooLarge: false }
}

function discardResponseReader(reader: ReadableStreamDefaultReader<Uint8Array>): void {
  try {
    void reader.cancel().catch(() => undefined)
  } catch {
  }
}

export async function handleResponse(
  res: Response,
  start: number,
  signal?: AbortSignal,
  queryOptions?: WebFetchQueryOptions,
  sourceUrl?: string,
): Promise<ToolResult> {
  if (!res.ok) {
    const transient = res.status >= 500 || res.status === 429 || res.status === 408
    await discardResponseBody(res, signal)
    return {
      output: 'HTTP ' + res.status,
      status: 'error',
      durationMs: Date.now() - start,
      code: transient ? `${res.status}_TRANSIENT` : `${res.status}_PERMANENT`,
    }
  }

  const { mime, charset } = parseContentType(res.headers.get('content-type') ?? '')
  if (!isTextualContentType(mime)) {
    await discardResponseBody(res, signal)
    return {
      output: `refusing to return non-text content (content-type: ${mime || 'unknown'}). Use a tool suited to binary content, or fetch a text/JSON URL.`,
      status: 'error',
      durationMs: Date.now() - start,
      code: 'NON_TEXT_CONTENT_PERMANENT',
    }
  }

  const declaredLength = Number(res.headers.get('content-length'))
  if (Number.isFinite(declaredLength) && declaredLength > MAX_BYTES) {
    await discardResponseBody(res, signal)
    return {
      output: 'response too large',
      status: 'error',
      durationMs: Date.now() - start,
      code: 'PAYLOAD_TOO_LARGE_PERMANENT',
    }
  }

  let body: string
  if (res.body) {
    const { bytes, tooLarge } = await readBodyBounded(res, MAX_BYTES, signal)
    if (tooLarge) {
      return {
        output: 'response too large',
        status: 'error',
        durationMs: Date.now() - start,
        code: 'PAYLOAD_TOO_LARGE_PERMANENT',
      }
    }
    body = decodeBody(bytes, charset)
  } else {
    const pendingText = res.text()
    body = signal ? await raceWithAbort(pendingText, signal) : await pendingText
    if (Buffer.byteLength(body, 'utf8') > MAX_BYTES) {
      return {
        output: 'response too large',
        status: 'error',
        durationMs: Date.now() - start,
        code: 'PAYLOAD_TOO_LARGE_PERMANENT',
      }
    }
  }

  const isHtml = mime.includes('html')
  const out = isHtml ? htmlToText(body) : body
  if (queryOptions?.linkQuery) {
    const isLinkText = !isHtml && (mime.includes('markdown') || mime.startsWith('text/') || /\.(md|markdown|txt)(?:$|[?#])/i.test(sourceUrl ?? ''))
    if (!isHtml && !isLinkText) {
      return {
        output: 'linkQuery reads links from HTML, Markdown or plain-text responses. Use query for JSON or other structured formats.',
        status: 'error',
        durationMs: Date.now() - start,
        code: 'LINK_QUERY_REQUIRES_HTML_PERMANENT',
      }
    }
    const extracted = extractWebFetchLinkMatches(
      body,
      queryOptions.linkQuery,
      sourceUrl,
      isHtml,
    )
    if (!extracted) {
      return {
        output: formatWebFetchQueryMiss({
          content: out,
          queryLabel: 'anchor query',
          query: queryOptions.linkQuery,
          guidance:
            'Choose different literal anchor text or a URL fragment, use query to inspect page text, or switch sources. Repeating the same URL/linkQuery is not useful without an intervening content change.',
        }),
        status: 'success',
        durationMs: Date.now() - start,
      }
    }
    return {
      output: extracted,
      status: 'success',
      durationMs: Date.now() - start,
    }
  }
  if (out.trim().length === 0) {
    return {
      output: [
        'The response contained no usable textual content after decoding.',
        'For a JavaScript-rendered page use browser.navigate/browser.extract; otherwise choose a different text or JSON source.',
      ].join('\n'),
      status: 'error',
      durationMs: Date.now() - start,
      code: 'EMPTY_CONTENT_PERMANENT',
    }
  }
  if (queryOptions?.query) {
    const extracted = extractWebFetchQueryMatches(
      out,
      queryOptions.query,
      queryOptions.contextChars,
    )
    if (!extracted) {
      return {
        output: formatWebFetchQueryMiss({
          content: out,
          queryLabel: 'literal query',
          query: queryOptions.query,
          guidance:
            'Choose a different literal term or phrase, or switch to another source. Repeating the same URL/query is not useful without an intervening content change.',
        }),
        status: 'success',
        durationMs: Date.now() - start,
      }
    }
    return {
      output: extracted,
      status: 'success',
      durationMs: Date.now() - start,
    }
  }
  const limited = enforceOutputLimit(out.slice(0, 200_000), {
    toolName: 'webfetch',
    resumeHint: 'request a smaller URL or extract a specific section',
  })
  return {
    output: limited.output,
    status: 'success',
    durationMs: Date.now() - start,
  }
}

const NON_CONTENT_OPENER = /<(script|style)\b/gi
const NON_CONTENT_OPENER_AT_SPLICE = /<(script|style)\b/i
const LONGEST_NON_CONTENT_OPENER = '<script'.length
const NON_CONTENT_CLOSERS: Readonly<Record<string, RegExp>> = {
  script: /<\/script\b/gi,
  style: /<\/style\b/gi,
}

export function stripNonContentElements(html: string): string {
  const kept: string[] = []
  let position = 0
  for (;;) {
    NON_CONTENT_OPENER.lastIndex = position
    const opener = NON_CONTENT_OPENER.exec(html)
    if (!opener) break
    if (opener.index > position) kept.push(html.slice(position, opener.index))
    let blockEnd = rawTextBlockEnd(html, opener[1]!, opener.index + opener[0].length)
    for (;;) {
      if (blockEnd < 0) return kept.join('')
      const spliced = openerSplicedAcrossCut(kept, html, blockEnd)
      if (!spliced) break
      dropKeptTail(kept, spliced.keptChars)
      blockEnd = rawTextBlockEnd(html, spliced.name, blockEnd + spliced.inputChars)
    }
    position = blockEnd
  }
  kept.push(html.slice(position))
  return kept.join('')
}

function rawTextBlockEnd(html: string, name: string, from: number): number {
  const closer = NON_CONTENT_CLOSERS[name.toLowerCase()]!
  closer.lastIndex = from
  const match = closer.exec(html)
  if (!match) return -1
  const tagEnd = html.indexOf('>', match.index + match[0].length)
  return tagEnd < 0 ? -1 : tagEnd + 1
}

function openerSplicedAcrossCut(
  kept: readonly string[],
  html: string,
  cut: number,
): { name: string; keptChars: number; inputChars: number } | null {
  let tail = ''
  for (let i = kept.length - 1; i >= 0 && tail.length < LONGEST_NON_CONTENT_OPENER; i -= 1) {
    tail = kept[i]!.slice(-(LONGEST_NON_CONTENT_OPENER - tail.length)) + tail
  }
  const window = tail + html.slice(cut, cut + LONGEST_NON_CONTENT_OPENER + 1)
  const match = NON_CONTENT_OPENER_AT_SPLICE.exec(window)
  if (!match || match.index >= tail.length) return null
  return {
    name: match[1]!,
    keptChars: tail.length - match.index,
    inputChars: match.index + match[0].length - tail.length,
  }
}

function dropKeptTail(kept: string[], chars: number): void {
  let remaining = chars
  while (remaining > 0 && kept.length > 0) {
    const last = kept.pop()!
    if (last.length > remaining) {
      kept.push(last.slice(0, last.length - remaining))
      return
    }
    remaining -= last.length
  }
}

function linkCandidates(body: string, isHtml: boolean): Array<{ rawHref: string; label: string }> {
  if (isHtml) {
    const contentHtml = stripNonContentElements(body)
    const candidates: Array<{ rawHref: string; label: string }> = []
    const anchorPattern = /<a\b([^>]*)>([\s\S]*?)<\/a\s*>/gi
    let anchor: RegExpExecArray | null
    while ((anchor = anchorPattern.exec(contentHtml)) !== null) {
      const attributes = anchor[1] ?? ''
      const rawHref = readHtmlAttribute(attributes, 'href')
      if (!rawHref) continue
      const fallbackLabel = readHtmlAttribute(attributes, 'aria-label') ?? readHtmlAttribute(attributes, 'title') ?? ''
      candidates.push({ rawHref: decodeHtmlEntities(rawHref.trim()), label: htmlToText(anchor[2] ?? '') || decodeHtmlEntities(fallbackLabel) })
    }
    return candidates
  }
  const candidates: Array<{ rawHref: string; label: string }> = []
  const markdown = /\[([^\]\n]{0,300})\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)/g
  let link: RegExpExecArray | null
  while ((link = markdown.exec(body)) !== null) candidates.push({ rawHref: link[2]!, label: link[1]!.trim() })
  const withoutMarkdownLinks = body.replace(markdown, ' ')
  for (const match of withoutMarkdownLinks.matchAll(/https?:\/\/[^\s<>"'`)\]]+/g)) {
    candidates.push({ rawHref: match[0].replace(/[.,;:!?]+$/, ''), label: '' })
  }
  return candidates
}

export function extractWebFetchLinkMatches(
  body: string,
  rawQuery: string,
  sourceUrl?: string,
  isHtml = true,
): string | null {
  const query = rawQuery.trim().slice(0, MAX_QUERY_CHARS)
  if (!query || !sourceUrl) return null
  let baseUrl: URL
  try {
    baseUrl = new URL(sourceUrl)
  } catch {
    return null
  }

  const lowerQuery = query.toLocaleLowerCase()
  const matches: Array<{ label: string; url: string }> = []
  const seenUrls = new Set<string>()
  let matchCount = 0

  for (const candidate of linkCandidates(body, isHtml)) {
    const rawHref = candidate.rawHref.trim()
    if (!rawHref || rawHref.startsWith('#')) continue
    let resolved: URL
    try {
      resolved = new URL(rawHref, baseUrl)
    } catch {
      continue
    }
    if (resolved.protocol !== 'http:' && resolved.protocol !== 'https:') continue
    const label = (candidate.label || resolved.toString()).replace(/\s+/g, ' ').trim().slice(0, 300)
    const absoluteUrl = resolved.toString()
    if (!`${label}\n${rawHref}\n${absoluteUrl}`.toLocaleLowerCase().includes(lowerQuery)) {
      continue
    }
    if (seenUrls.has(absoluteUrl)) continue
    seenUrls.add(absoluteUrl)
    matchCount += 1
    if (matches.length < MAX_LINK_MATCHES) {
      matches.push({ label, url: absoluteUrl })
    }
  }

  if (matchCount === 0) return null
  return [
    `[webfetch link query ${JSON.stringify(query)}: ${matchCount} matching resolved link${matchCount === 1 ? '' : 's'}; showing ${matches.length}]`,
    ...matches.map(({ label, url }) => `- ${label} -> ${url}`),
    ...(matchCount > matches.length
      ? [`[${matchCount - matches.length} additional matching link${matchCount - matches.length === 1 ? '' : 's'} omitted after the bounded link limit]`]
      : []),
  ].join('\n')
}

function readHtmlAttribute(attributes: string, name: string): string | null {
  const escapedName = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const pattern = new RegExp(
    `(?:^|\\s)${escapedName}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s"'=<>\u0060]+))`,
    'i',
  )
  const match = attributes.match(pattern)
  return match?.[1] ?? match?.[2] ?? match?.[3] ?? null
}

function decodeHtmlEntities(value: string): string {
  return value
    .replace(/&#(?:x([0-9a-f]+)|(\d+));?/gi, (_match, hex: string | undefined, decimal: string | undefined) => {
      const codePoint = Number.parseInt(hex ?? decimal ?? '', hex ? 16 : 10)
      if (!Number.isFinite(codePoint) || codePoint < 0 || codePoint > 0x10ffff) return ''
      try {
        return String.fromCodePoint(codePoint)
      } catch {
        return ''
      }
    })
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
}

export function extractWebFetchQueryMatches(
  content: string,
  rawQuery: string,
  rawContextChars = DEFAULT_QUERY_CONTEXT_CHARS,
): string | null {
  const query = rawQuery.trim().slice(0, MAX_QUERY_CHARS)
  if (!query) return null
  const contextChars = Math.max(
    100,
    Math.min(MAX_QUERY_CONTEXT_CHARS, Math.trunc(rawContextChars)),
  )
  const lowerContent = content.toLocaleLowerCase()
  const lowerQuery = query.toLocaleLowerCase()
  const ranges: Array<{ start: number; end: number }> = []
  let matchCount = 0
  let omittedWindowCount = 0
  let cursor = 0

  while (cursor <= lowerContent.length - lowerQuery.length) {
    const index = lowerContent.indexOf(lowerQuery, cursor)
    if (index < 0) break
    matchCount += 1
    const start = Math.max(0, index - contextChars)
    const end = Math.min(content.length, index + query.length + contextChars)
    const previous = ranges.at(-1)
    if (previous && start <= previous.end) {
      previous.end = Math.max(previous.end, end)
    } else if (ranges.length < MAX_QUERY_WINDOWS) {
      ranges.push({ start, end })
    } else {
      omittedWindowCount += 1
    }
    cursor = index + Math.max(1, lowerQuery.length)
  }

  if (matchCount === 0) return null
  const snippets = ranges.map((range, index) => [
    `[match window ${index + 1}: characters ${range.start + 1}-${range.end} of ${content.length}]`,
    content.slice(range.start, range.end).trim(),
  ].join('\n'))
  return [
    `[webfetch literal query ${JSON.stringify(query)}: ${matchCount} match${matchCount === 1 ? '' : 'es'} in ${content.length} decoded characters; showing ${ranges.length} bounded window${ranges.length === 1 ? '' : 's'}]`,
    ...snippets,
    ...(omittedWindowCount > 0
      ? [`[${omittedWindowCount} additional non-overlapping match window${omittedWindowCount === 1 ? '' : 's'} omitted after the bounded window limit]`]
      : []),
  ].join('\n\n')
}

export function formatWebFetchQueryMiss(opts: {
  content: string
  queryLabel: string
  query: string
  guidance: string
}): string {
  const { content, queryLabel, query, guidance } = opts
  const excerpt = content.slice(0, QUERY_MISS_EXCERPT_CHARS).trim()
  return [
    `[webfetch ${queryLabel} ${JSON.stringify(query)}: 0 matches in ${content.length} decoded characters]`,
    guidance,
    ...(excerpt
      ? [
          `[content head: characters 1-${Math.min(QUERY_MISS_EXCERPT_CHARS, content.length)} of ${content.length}]`,
          excerpt,
        ]
      : []),
  ].join('\n\n')
}

async function discardResponseBody(res: Response, signal?: AbortSignal): Promise<void> {
  try {
    const cancellation = res.body?.cancel(signal?.reason)
    if (cancellation) void cancellation.catch(() => undefined)
  } catch {
  }
}

function htmlToText(html: string): string {
  return stripNonContentElements(html)
    .replace(
      /<\/(?:tr|li|h[1-6]|p|div|section|article|header|footer|nav|table|tbody|thead|tfoot|caption|blockquote|pre|ul|ol|dl|dt|dd)>/gi,
      '\n',
    )
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/(?:td|th)>/gi, ' | ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&#(?:x([0-9a-f]+)|(\d+));?|&(?:nbsp|amp|lt|gt|quot|#39|apos);/gi, (entity) => decodeHtmlEntities(entity))
    .replace(/[ \t]+/g, ' ')
    .replace(/ *\n */g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}
