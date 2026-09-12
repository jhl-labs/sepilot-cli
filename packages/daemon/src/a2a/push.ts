import type { Dispatcher } from 'undici'
import { assertPublicUrl, createPinnedLookupDispatcher } from '../utils/ssrf-guard.js'
import type { A2APushDelivery } from './server.js'
import type {
  A2AJsonRpcResponse,
  A2AStreamResponse,
  A2ATaskPushNotificationConfig,
} from './types.js'

export interface A2AHttpPushDeliveryOptions {
  fetcher?: typeof fetch
  timeoutMs?: number
  /** Maximum redirect hops to follow before refusing. Each hop is re-validated. */
  maxRedirects?: number
}

const DEFAULT_TIMEOUT_MS = 15_000
const DEFAULT_MAX_REDIRECTS = 5

type FetchInitWithDispatcher = RequestInit & { dispatcher: Dispatcher }

export class A2AHttpPushDelivery implements A2APushDelivery {
  private readonly fetcher: typeof fetch
  private readonly maxRedirects: number

  constructor(private readonly options: A2AHttpPushDeliveryOptions = {}) {
    this.fetcher = options.fetcher ?? fetch
    this.maxRedirects = options.maxRedirects ?? DEFAULT_MAX_REDIRECTS
  }

  async deliver(config: A2ATaskPushNotificationConfig, event: A2AStreamResponse): Promise<void> {
    // The registration step only blocks literal private IPs, so a hostname
    // whose A record resolves to 169.254.169.254 / 127.0.0.1 / a private LAN
    // host slips through at registration and reaches `fetch` here untouched.
    // Re-validate the URL (with async DNS) at delivery time and walk redirects
    // manually so any 302 to a private target is also refused. Same shape as
    // tools/web-fetch.ts.
    const initialUrl = config.pushNotificationConfig.url
    const deliveryOrigin = new URL(initialUrl).origin
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS)
    const publicHeaders: Record<string, string> = {
      'content-type': 'application/json',
      accept: 'application/json',
    }
    const authenticatedHeaders = { ...publicHeaders }
    const auth = config.pushNotificationConfig.authentication
    if (auth?.scheme && auth.credentials) {
      authenticatedHeaders.authorization = `${auth.scheme} ${auth.credentials}`
    }
    const payload: A2AJsonRpcResponse = {
      jsonrpc: '2.0',
      result: event,
    }
    const body = JSON.stringify(payload)
    try {
      let currentUrl = initialUrl
      for (let hop = 0; ; hop += 1) {
        let resolution
        try {
          resolution = await assertPublicUrl(currentUrl, controller.signal)
        } catch (err) {
          const context = hop === 0 ? '' : ' on redirect'
          throw new Error(`A2A push delivery refused${context}: ${(err as Error).message}`)
        }
        const dispatcher = createPinnedLookupDispatcher(resolution)
        try {
          const headers = authenticatedHeaders
          const response = await this.fetcher(currentUrl, {
            method: 'POST',
            headers,
            signal: controller.signal,
            body,
            redirect: 'manual',
            dispatcher,
          } as FetchInitWithDispatcher)
          if (response.status >= 300 && response.status < 400) {
            const location = response.headers.get('location')
            if (location) {
              if (hop >= this.maxRedirects) {
                await discardResponseBody(response)
                throw new Error(
                  `A2A push delivery failed: too many redirects (${this.maxRedirects})`,
                )
              }
              let nextUrl: URL
              try {
                nextUrl = new URL(location, resolution.url)
              } catch {
                await discardResponseBody(response)
                throw new Error('A2A push delivery refused: invalid redirect URL')
              }
              if (resolution.url.protocol === 'https:' && nextUrl.protocol !== 'https:') {
                await discardResponseBody(response)
                throw new Error('A2A push delivery refused on redirect: HTTPS redirect downgrade')
              }
              if (nextUrl.origin !== deliveryOrigin) {
                await discardResponseBody(response)
                throw new Error(
                  'A2A push delivery refused: cross-origin redirects are not allowed',
                )
              }
              currentUrl = nextUrl.toString()
              await discardResponseBody(response)
              continue
            }
          }
          if (!response.ok) {
            await discardResponseBody(response)
            throw new Error(`A2A push delivery failed: HTTP ${response.status}`)
          }
          await discardResponseBody(response)
          return
        } finally {
          await dispatcher.close().catch(() => undefined)
        }
      }
    } finally {
      clearTimeout(timer)
    }
  }
}

async function discardResponseBody(response: Response): Promise<void> {
  await response.body?.cancel().catch(() => undefined)
}
