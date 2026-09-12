import type { Dispatcher } from 'undici'
import {
  assertPublicUrl,
  createPinnedLookupDispatcher,
  type PublicUrlResolution,
} from '../../utils/ssrf-guard.js'

const DEFAULT_MAX_REDIRECTS = 5
const DEFAULT_TIMEOUT_MS = 20_000
const MAX_SKILL_RESPONSE_BYTES = 2 * 1024 * 1024
const REDIRECT_STATUS_CODES = new Set([301, 302, 303, 307, 308])
const CROSS_ORIGIN_SAFE_HEADERS = new Set([
  'accept',
  'accept-encoding',
  'accept-language',
  'user-agent',
])

type CloseableDispatcher = Dispatcher & {
  close(): Promise<void>
}

export interface PublicSkillFetchRuntime {
  fetch?: typeof fetch
  resolveUrl?: typeof assertPublicUrl
  createDispatcher?: (
    resolution: PublicUrlResolution,
  ) => CloseableDispatcher
}

export interface PublicSkillFetchOptions extends PublicSkillFetchRuntime {
  assertAllowed?: (url: string) => void
  headers?: HeadersInit
  maxRedirects?: number
  signal?: AbortSignal
}

export interface PublicSkillTextResponse {
  ok: boolean
  status: number
  text: string
  url: string
}

/**
 * Fetch an HTTPS skill document without allowing a validated hostname to be
 * resolved again at connect time. Redirects are followed manually so every
 * hop is checked by both the configured source policy and the public-address
 * guard. The response body is consumed (or cancelled) before its dispatcher
 * is closed.
 */
export async function fetchPublicSkillText(
  rawUrl: string,
  options: PublicSkillFetchOptions = {},
): Promise<PublicSkillTextResponse> {
  const fetchFn = options.fetch ?? fetch
  const resolveUrl = options.resolveUrl ?? assertPublicUrl
  const createDispatcher = options.createDispatcher ?? createPinnedLookupDispatcher
  const maxRedirects = options.maxRedirects ?? DEFAULT_MAX_REDIRECTS
  const signal = options.signal ?? AbortSignal.timeout(DEFAULT_TIMEOUT_MS)
  let currentUrl = rawUrl
  let headers = new Headers(options.headers)

  for (let hop = 0; ; hop += 1) {
    options.assertAllowed?.(currentUrl)
    const resolution = await resolveUrl(currentUrl, signal)
    if (resolution.url.protocol !== 'https:') {
      throw new Error('skill source URLs and redirects must use https')
    }
    if (resolution.url.username || resolution.url.password) {
      throw new Error('skill source URLs must not contain credentials')
    }

    const dispatcher = createDispatcher(resolution)
    let response: Response | undefined
    try {
      response = await fetchFn(currentUrl, {
        headers,
        redirect: 'manual',
        signal,
        dispatcher,
      } as RequestInit & { dispatcher: Dispatcher })

      const location = REDIRECT_STATUS_CODES.has(response.status)
        ? response.headers.get('location')
        : null
      if (location) {
        if (hop >= maxRedirects) {
          throw new Error('too many skill source redirects')
        }
        const nextUrl = new URL(location, currentUrl).toString()
        if (new URL(nextUrl).origin !== new URL(currentUrl).origin) {
          headers = crossOriginSafeHeaders(headers)
        }
        currentUrl = nextUrl
        continue
      }

      if (!response.ok) {
        return {
          ok: false,
          status: response.status,
          text: '',
          url: currentUrl,
        }
      }

      return {
        ok: true,
        status: response.status,
        text: await readBoundedText(response),
        url: currentUrl,
      }
    } finally {
      await discardResponseBody(response)
      await dispatcher.close().catch(() => undefined)
    }
  }
}

async function readBoundedText(response: Response): Promise<string> {
  const contentLength = response.headers.get('content-length')
  if (contentLength != null) {
    if (!/^\d+$/u.test(contentLength)) {
      throw new Error('skill source content length is invalid')
    }
    const declaredBytes = Number(contentLength)
    if (!Number.isSafeInteger(declaredBytes) || declaredBytes > MAX_SKILL_RESPONSE_BYTES) {
      throw new Error(`skill source response exceeds ${MAX_SKILL_RESPONSE_BYTES} bytes`)
    }
  }

  const reader = response.body?.getReader()
  if (!reader) return ''
  const chunks: Uint8Array[] = []
  let totalBytes = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (!value) continue
    totalBytes += value.byteLength
    if (totalBytes > MAX_SKILL_RESPONSE_BYTES) {
      await reader.cancel().catch(() => undefined)
      throw new Error(`skill source response exceeds ${MAX_SKILL_RESPONSE_BYTES} bytes`)
    }
    chunks.push(value)
  }

  const bytes = new Uint8Array(totalBytes)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  return new TextDecoder('utf-8', { fatal: true }).decode(bytes)
}

function crossOriginSafeHeaders(headers: Headers): Headers {
  const safe = new Headers()
  for (const [name, value] of headers.entries()) {
    if (CROSS_ORIGIN_SAFE_HEADERS.has(name.toLowerCase())) {
      safe.set(name, value)
    }
  }
  return safe
}

async function discardResponseBody(response: Response | undefined): Promise<void> {
  try {
    await response?.body?.cancel()
  } catch {
    // Best effort. The per-hop dispatcher is closed immediately afterwards.
  }
}
