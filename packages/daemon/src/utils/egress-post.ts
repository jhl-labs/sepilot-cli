import type { Dispatcher } from 'undici'
import { isAbortError } from '../abort.js'
import { assertPublicUrl, createPinnedLookupDispatcher } from './ssrf-guard.js'

const EGRESS_POST_MAX_REDIRECTS = 5
type FetchInitWithDispatcher = RequestInit & { dispatcher?: Dispatcher }

export function permanentDeliveryError(message: string): Error & { retryable: false } {
  return Object.assign(new Error(message), { retryable: false as const })
}

export function isPermanentDeliveryError(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'retryable' in error
    && (error as { retryable?: unknown }).retryable === false
}

async function discardResponseBody(response: Response): Promise<void> {
  await response.body?.cancel().catch(() => undefined)
}

export async function postWithEgressGuard(
  initialUrl: string,
  init: { headers: Record<string, string>; body: string; signal: AbortSignal },
): Promise<{ ok: boolean; status: number | undefined }> {
  const allowPrivate = process.env.SEPILOTD_OUTBOUND_WEBHOOK_ALLOW_PRIVATE === '1'
  let parsedInitial: URL
  try {
    parsedInitial = new URL(initialUrl)
  } catch {
    throw permanentDeliveryError('invalid webhook URL')
  }
  if (
    (parsedInitial.protocol !== 'http:' && parsedInitial.protocol !== 'https:')
    || parsedInitial.username
    || parsedInitial.password
  ) {
    throw permanentDeliveryError('webhook URL must be credential-free http(s)')
  }

  const allowedOrigin = parsedInitial.origin
  let currentUrl = parsedInitial.toString()
  for (let redirects = 0; ; redirects += 1) {
    let dispatcher: Dispatcher | undefined
    let requestUrl = currentUrl
    if (!allowPrivate) {
      try {
        const resolution = await assertPublicUrl(currentUrl, init.signal)
        requestUrl = resolution.url.toString()
        dispatcher = createPinnedLookupDispatcher(resolution)
      } catch (error) {
        if (isAbortError(error)) throw error
        if (
          typeof error === 'object' && error !== null && 'retryable' in error
          && (error as { retryable?: unknown }).retryable === true
        ) {
          throw error
        }
        throw permanentDeliveryError(
          `blocked by egress guard: ${error instanceof Error ? error.message : String(error)}`,
        )
      }
    }

    let response: Response
    try {
      response = await fetch(requestUrl, {
        method: 'POST',
        headers: init.headers,
        body: init.body,
        signal: init.signal,
        redirect: 'manual',
        ...(dispatcher ? { dispatcher } : {}),
      } as FetchInitWithDispatcher)
    } catch (error) {
      await dispatcher?.close().catch(() => undefined)
      throw error
    }

    try {
      const isPreservingRedirect = response.status === 307 || response.status === 308
      const location = isPreservingRedirect ? response.headers?.get('location') : null
      if (!location) return { ok: response.ok, status: response.status }
      if (redirects >= EGRESS_POST_MAX_REDIRECTS) {
        throw permanentDeliveryError(
          `too many webhook redirects (${EGRESS_POST_MAX_REDIRECTS})`,
        )
      }

      let nextUrl: URL
      try {
        nextUrl = new URL(location, requestUrl)
      } catch {
        throw permanentDeliveryError('invalid webhook redirect URL')
      }
      if (
        nextUrl.origin !== allowedOrigin
        || nextUrl.username
        || nextUrl.password
      ) {
        throw permanentDeliveryError('cross-origin webhook redirects are not allowed')
      }
      currentUrl = nextUrl.toString()
    } finally {
      await discardResponseBody(response)
      await dispatcher?.close().catch(() => undefined)
    }
  }
}
