import type { ContentPart } from '@sepilotd/core'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'
import type { Dispatcher } from 'undici/index.js'
import { access, readFile, stat } from 'node:fs/promises'
import { getAbortError, throwIfAborted } from '../abort.js'
import { resolveToolPath } from './path-utils.js'
import {
  createEgressPolicy,
  EgressDeniedError,
  type EgressPolicy,
  type EgressPolicyLogger,
} from '../utils/egress-policy.js'
import {
  assertPublicUrl,
  createManagedLoopbackDispatcher,
  createPinnedLookupDispatcher,
  fetchWithPinnedNetworkPolicy,
} from '../utils/ssrf-guard.js'

let playwright: typeof import('playwright') | null = null
let browser: import('playwright').Browser | null = null
const MAX_BROWSER_SCREENSHOT_IMAGE_BYTES = 8_000_000
const MAX_BROWSER_RESOURCE_BYTES = 32 * 1024 * 1024
const MAX_BROWSER_PAGE_BYTES = 96 * 1024 * 1024
const MAX_BROWSER_REQUEST_BODY_BYTES = 8 * 1024 * 1024
const MAX_BROWSER_PAGE_REQUESTS = 512
const MAX_BROWSER_CONCURRENT_REQUESTS = 32
const BROWSER_RESOURCE_TIMEOUT_MS = 30_000
const MAX_BROWSER_NAVIGATION_REDIRECTS = 10
const GUARDED_CHROMIUM_ARGS = [
  '--disable-quic',
  // Chromium 151 enables Local Network Access checks for WebSockets. The
  // guarded page already enforces a stricter capability boundary: only the
  // exact localhost port backed by a managed process bridge may connect, and
  // every remote or unowned socket is closed. Disable Chromium's interactive
  // permission gate so that one authorized socket works in headless mode.
  '--disable-features=LocalNetworkAccessChecks,LocalNetworkAccessChecksWebSockets',
  '--force-webrtc-ip-handling-policy=disable_non_proxied_udp',
]
const BROWSER_SCREENSHOT_IMAGE_ATTACHMENT_ATTACHED =
  'Screenshot image attachment: attached for visual inspection'
const MAX_BROWSER_NAVIGATE_OUTPUT_CHARS = 50_000

export function formatBrowserNavigationResult(input: {
  url: string
  title: string
  renderedText: string
}): string {
  const url = input.url.replace(/\s+/g, ' ').trim()
  const title = input.title.replace(/\s+/g, ' ').trim()
  const header = [
    `URL: ${url || '(unknown)'}`,
    `Title: ${title || '(empty)'}`,
    '',
    'Rendered text:',
  ].join('\n')
  const remaining = Math.max(0, MAX_BROWSER_NAVIGATE_OUTPUT_CHARS - header.length - 1)
  return `${header}\n${input.renderedText.slice(0, remaining)}`
}

export interface BrowserToolOptions {
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
  browserExecutablePath?: string | null
  fetcher?: typeof fetch
}

interface BrowserLayoutAudit {
  viewport: { width: number; height: number }
  document: { width: number; height: number }
  body: { width: number; height: number; scrollHeight: number; computedHeight: string }
  horizontalOverflowPx: number
  visibleElementCount: number
  visibleViewportElementCount: number
  warnings: string[]
  examples: string[]
}

interface BrowserConsolePageAudit {
  consoleErrors: string[]
  consoleWarnings: string[]
  pageErrors: string[]
}

function isBrowserConnected(b: import('playwright').Browser): boolean {
  try {
    return b.isConnected()
  } catch {
    return false
  }
}

async function firstExistingExecutable(paths: readonly string[]): Promise<string | null> {
  for (const path of paths) {
    if (!path.trim()) continue
    try {
      await access(path)
      return path
    } catch {
      // Keep trying common system browser paths.
    }
  }
  return null
}

async function resolveBrowserExecutablePath(options: BrowserToolOptions): Promise<string | null> {
  if (options.browserExecutablePath?.trim()) {
    return options.browserExecutablePath.trim()
  }
  const env = options.env ?? process.env
  if (env.SEPILOTD_BROWSER_EXECUTABLE?.trim()) {
    return env.SEPILOTD_BROWSER_EXECUTABLE.trim()
  }
  return firstExistingExecutable([
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    `${env.PROGRAMFILES ?? 'C:\\Program Files'}\\Google\\Chrome\\Application\\chrome.exe`,
    `${env['PROGRAMFILES(X86)'] ?? 'C:\\Program Files (x86)'}\\Google\\Chrome\\Application\\chrome.exe`,
    `${env.LOCALAPPDATA ?? ''}\\Google\\Chrome\\Application\\chrome.exe`,
  ])
}

async function launchChromium(options: BrowserToolOptions): Promise<import('playwright').Browser> {
  if (!playwright) {
    throw new Error('Playwright not loaded')
  }
  try {
    return await playwright.chromium.launch({ headless: true, args: GUARDED_CHROMIUM_ARGS })
  } catch (primaryError) {
    const executablePath = await resolveBrowserExecutablePath(options)
    if (!executablePath) {
      throw primaryError
    }
    try {
      return await playwright.chromium.launch({
        headless: true,
        executablePath,
        args: GUARDED_CHROMIUM_ARGS,
      })
    } catch (fallbackError) {
      const primaryMessage =
        primaryError instanceof Error ? primaryError.message : String(primaryError)
      const fallbackMessage =
        fallbackError instanceof Error ? fallbackError.message : String(fallbackError)
      throw new Error(
        `Playwright browser launch failed, and system browser fallback at ${executablePath} also failed. Primary: ${primaryMessage}. Fallback: ${fallbackMessage}`,
      )
    }
  }
}

async function ensureBrowser(
  options: BrowserToolOptions = {},
): Promise<import('playwright').Browser> {
  if (browser) {
    if (isBrowserConnected(browser)) {
      return browser
    }
    browser = null
  }
  if (!playwright) {
    try {
      playwright = await import('playwright')
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error)
      throw new Error(
        'Playwright runtime could not be loaded. The standalone binary bundles '
          + `the runtime, while source installs require the playwright package. Original error: ${reason}`,
      )
    }
  }
  const launched = await launchChromium(options)
  launched.on('disconnected', () => {
    if (browser === launched) {
      browser = null
    }
  })
  browser = launched
  return launched
}

async function newGuardedPage(
  b: import('playwright').Browser,
  egressPolicy: EgressPolicy,
  allowedLocalhostOrigin?: string | null,
  managedLoopbackSocketPath?: string | null,
  fetcher: typeof fetch = fetchWithPinnedNetworkPolicy,
  executionSignal?: AbortSignal,
): Promise<import('playwright').Page> {
  const page = await b.newPage({ serviceWorkers: 'block' })
  await page.addInitScript(() => {
    class BlockedPeerTransport {
      constructor() {
        throw new DOMException('Direct peer transport is disabled', 'SecurityError')
      }
    }
    const target = globalThis as unknown as Record<string, unknown>
    for (const name of ['WebTransport', 'RTCPeerConnection', 'webkitRTCPeerConnection']) {
      Object.defineProperty(target, name, {
        configurable: false,
        enumerable: false,
        value: BlockedPeerTransport,
        writable: false,
      })
    }
  })
  const pageController = new AbortController()
  const pageSignal = executionSignal
    ? AbortSignal.any([executionSignal, pageController.signal])
    : pageController.signal
  const state: GuardedPageState = { activeRequests: 0, idleWaiters: new Set() }
  guardedPageStates.set(page, state)
  page.on('close', () => pageController.abort())
  const budget: BrowserNetworkBudget = {
    activeRequests: 0,
    responseBytes: 0,
    totalRequests: 0,
  }
  await page.routeWebSocket?.('**/*', async (route) => {
    // A session-owned managed-loopback endpoint is already capability-bound
    // by process.start. Permit its same-port localhost aliases so dev-server
    // HMR and application WebSockets work during browser QA, while every
    // remote or unowned WebSocket remains fail-closed.
    if (
      managedLoopbackSocketPath
      && isSameAllowedLocalhostEndpoint(route.url(), allowedLocalhostOrigin)
    ) {
      route.connectToServer()
      return
    }
    await route.close({ code: 1008, reason: 'WebSocket egress is disabled for guarded pages' })
  })
  await page.route('**/*', async (route) => {
    budget.totalRequests += 1
    if (budget.totalRequests > MAX_BROWSER_PAGE_REQUESTS) {
      await route.abort('blockedbyclient').catch(() => {})
      return
    }
    budget.activeRequests += 1
    state.activeRequests += 1
    if (budget.activeRequests > MAX_BROWSER_CONCURRENT_REQUESTS) {
      budget.activeRequests -= 1
      await route.abort('blockedbyclient').catch(() => {})
      return
    }
    try {
      await fulfillGuardedBrowserRequest(
        route,
        egressPolicy,
        allowedLocalhostOrigin,
        managedLoopbackSocketPath,
        fetcher,
        budget,
        page,
        state,
        pageSignal,
      )
    } catch (error) {
      state.lastRequestError = error instanceof Error ? error : new Error(String(error))
      await route.abort('blockedbyclient').catch(() => {})
    } finally {
      budget.activeRequests -= 1
      state.activeRequests -= 1
      if (state.activeRequests === 0) {
        for (const resolve of state.idleWaiters) resolve()
        state.idleWaiters.clear()
      }
    }
  })
  return page
}

type FetchInitWithDispatcher = RequestInit & {
  dispatcher?: Dispatcher
  managedLoopbackSocketPath?: string
}

interface BrowserNetworkBudget {
  activeRequests: number
  responseBytes: number
  totalRequests: number
}

interface GuardedPageState {
  pendingRedirect?: string
  lastRequestError?: Error
  activeRequests: number
  idleWaiters: Set<() => void>
}

const guardedPageStates = new WeakMap<import('playwright').Page, GuardedPageState>()

async function closeGuardedPage(page: import('playwright').Page): Promise<void> {
  if (page.isClosed()) return
  // DOMContentLoaded can be followed by an implicit favicon or module fetch.
  // Give those already-scheduled requests one bounded network-idle window so
  // closing the page does not tear down a single-threaded dev server mid-call.
  if (typeof page.waitForLoadState === 'function') {
    await page.waitForLoadState('networkidle', { timeout: 750 }).catch(() => {})
  }
  const state = guardedPageStates.get(page)
  const waitForIdle = async () => {
    if (!state || state.activeRequests === 0) return
    await new Promise<void>((resolve) => {
      const finish = () => {
        clearTimeout(timer)
        state.idleWaiters.delete(finish)
        resolve()
      }
      const timer = setTimeout(finish, 750)
      state.idleWaiters.add(finish)
    })
  }
  await waitForIdle()
  if (!page.isClosed()) await page.close().catch(() => {})
  // page.close aborts resources that Chromium scheduled after network-idle.
  // Wait for their guarded transports to release the server-side proxy too.
  await waitForIdle()
}

const REQUEST_HEADERS_TO_DROP = new Set([
  'accept-encoding',
  'connection',
  'content-length',
  'host',
  'keep-alive',
  'proxy-authorization',
  'proxy-connection',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
])

const RESPONSE_HEADERS_TO_DROP = new Set([
  'connection',
  'content-encoding',
  'content-length',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
])

function proxyRequestHeaders(headers: Record<string, string>): Record<string, string> {
  return Object.fromEntries(
    Object.entries(headers).filter(([name]) => !REQUEST_HEADERS_TO_DROP.has(name.toLowerCase())),
  )
}

function proxyResponseHeaders(
  headers: Headers,
  requestUrl: string,
  requestHeaders: Record<string, string>,
  allowedLocalhostOrigin?: string | null,
): Record<string, string> {
  const proxied = Object.fromEntries(
    [...headers.entries()].filter(([name]) => !RESPONSE_HEADERS_TO_DROP.has(name.toLowerCase())),
  )
  const setCookies = (headers as Headers & { getSetCookie?: () => string[] }).getSetCookie?.()
  if (setCookies?.length) {
    // Playwright's Chromium adapter splits newline-delimited Set-Cookie values
    // before fulfilling the route, preserving multiple cookie headers.
    proxied['set-cookie'] = setCookies.join('\n')
  }

  const localConnectSources = managedLoopbackConnectSources(allowedLocalhostOrigin)
  const networkCsp = [
    `connect-src 'self'${localConnectSources.length > 0 ? ` ${localConnectSources.join(' ')}` : ''}`,
    "worker-src 'none'",
  ].join('; ')
  const existingCsp = proxied['content-security-policy']?.trim()
  proxied['content-security-policy'] = existingCsp
    ? `${existingCsp}, ${networkCsp}`
    : networkCsp

  const requestOrigin = new Headers(requestHeaders).get('origin')?.trim()
  if (
    requestOrigin &&
    new URL(requestUrl).origin !== requestOrigin &&
    !('access-control-allow-origin' in proxied)
  ) {
    // Route.fulfill otherwise adds the request Origin automatically, which
    // would turn a server-side CORS denial into an allowed browser response.
    proxied['access-control-allow-origin'] = 'https://sepilotd-cors-denied.invalid'
  }
  return proxied
}

function isBrowserRedirectResponse(response: Response): boolean {
  return (
    [301, 302, 303, 307, 308].includes(response.status) &&
    response.headers.has('location')
  )
}

async function readBrowserResponseBody(
  response: Response,
  budget: BrowserNetworkBudget,
): Promise<Buffer> {
  const contentLength = response.headers.get('content-length')
  if (contentLength != null) {
    if (!/^\d+$/u.test(contentLength)) throw new Error('Invalid browser resource content length')
    const declaredBytes = Number(contentLength)
    if (
      !Number.isSafeInteger(declaredBytes) ||
      declaredBytes > MAX_BROWSER_RESOURCE_BYTES ||
      budget.responseBytes + declaredBytes > MAX_BROWSER_PAGE_BYTES
    ) {
      throw new Error('Browser resource exceeds the guarded page byte budget')
    }
  }
  if (!response.body) return Buffer.alloc(0)
  const reader = response.body.getReader()
  const chunks: Buffer[] = []
  let totalBytes = 0
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const chunk = Buffer.from(value)
      totalBytes += chunk.length
      budget.responseBytes += chunk.length
      if (totalBytes > MAX_BROWSER_RESOURCE_BYTES) {
        throw new Error(`Browser resource exceeds ${MAX_BROWSER_RESOURCE_BYTES} bytes`)
      }
      if (budget.responseBytes > MAX_BROWSER_PAGE_BYTES) {
        throw new Error(`Guarded page exceeds ${MAX_BROWSER_PAGE_BYTES} response bytes`)
      }
      chunks.push(chunk)
    }
    return Buffer.concat(chunks, totalBytes)
  } finally {
    await reader.cancel().catch(() => undefined)
  }
}

async function fulfillGuardedBrowserRequest(
  route: import('playwright').Route,
  egressPolicy: EgressPolicy,
  allowedLocalhostOrigin: string | null | undefined,
  managedLoopbackSocketPath: string | null | undefined,
  fetcher: typeof fetch,
  budget: BrowserNetworkBudget,
  page: import('playwright').Page,
  state: GuardedPageState,
  pageSignal: AbortSignal,
): Promise<void> {
  const request = route.request()
  const rawUrl = request.url()
  const signal = AbortSignal.any([
    pageSignal,
    AbortSignal.timeout(BROWSER_RESOURCE_TIMEOUT_MS),
  ])
  const allowedLocalhostRequest = isSameAllowedLocalhostOrigin(rawUrl, allowedLocalhostOrigin)
  if (!allowedLocalhostRequest) egressPolicy.assertAllowed(rawUrl, 'browser')
  const requestHeaders = await request.allHeaders()
  const requestOrigin = new Headers(requestHeaders).get('origin')?.trim()
  if (requestOrigin && new URL(rawUrl).origin !== requestOrigin) {
    // Playwright auto-fulfills intercepted CORS preflights. Forwarding the
    // actual cross-origin request would therefore bypass a server's preflight
    // denial and could trigger state changes even if its response stays
    // unreadable. Keep these requests fail-closed until a transport-level
    // pinned proxy can preserve Chromium's native CORS behavior.
    throw new Error('Cross-origin browser requests are blocked by the guarded transport')
  }

  let requestUrl = rawUrl
  let dispatcher: Dispatcher | undefined
  if (allowedLocalhostRequest) {
    dispatcher = createManagedLoopbackDispatcher(rawUrl)
  } else {
    const resolution = await assertPublicUrl(rawUrl, signal)
    requestUrl = resolution.url.toString()
    dispatcher = createPinnedLookupDispatcher(resolution)
  }

  const method = request.method().toUpperCase()
  const declaredRequestBytes = requestHeaders['content-length']
  if (declaredRequestBytes != null) {
    if (!/^\d+$/u.test(declaredRequestBytes)) {
      throw new Error('Invalid browser request content length')
    }
    if (Number(declaredRequestBytes) > MAX_BROWSER_REQUEST_BODY_BYTES) {
      throw new Error('Browser request body exceeds the guarded request limit')
    }
  }
  const body =
    method === 'GET' || method === 'HEAD' ? undefined : (request.postDataBuffer() ?? undefined)
  if (body && body.byteLength > MAX_BROWSER_REQUEST_BODY_BYTES) {
    throw new Error('Browser request body exceeds the guarded request limit')
  }
  let response: Response | null = null
  try {
    response = await fetcher(requestUrl, {
      method,
      headers: proxyRequestHeaders(requestHeaders),
      body,
      signal,
      redirect: 'manual',
      dispatcher,
      ...(allowedLocalhostRequest && managedLoopbackSocketPath
        ? { managedLoopbackSocketPath }
        : {}),
    } as FetchInitWithDispatcher)
    // Chromium does not surface a fulfilled redirect's next hop through a new
    // user Route. Passing a 3xx through would therefore bypass DNS pinning and
    // the egress policy. Keep redirects fail-closed until a fully pinned proxy
    // owns the entire browser transport. For the main document, hand the
    // validated target back to gotoPublicUrl so it can start a fresh guarded
    // navigation instead of letting Chromium follow invisibly.
    if (isBrowserRedirectResponse(response)) {
      const location = response.headers.get('location')
      const currentUrl = new URL(rawUrl)
      const nextUrlObject = new URL(location!, currentUrl)
      if (currentUrl.protocol === 'https:' && nextUrlObject.protocol !== 'https:') {
        throw new Error('HTTPS browser redirects must not downgrade to HTTP')
      }
      const nextUrl = nextUrlObject.toString()
      if (!isSameAllowedLocalhostOrigin(nextUrl, allowedLocalhostOrigin)) {
        egressPolicy.assertAllowed(nextUrl, 'browser')
      }
      await assertBrowserNetworkAllowed(nextUrl, allowedLocalhostOrigin, signal)
      if (
        request.isNavigationRequest() &&
        request.frame() === page.mainFrame()
      ) {
        state.pendingRedirect = nextUrl
      }
      throw new Error('Browser redirects are blocked by the guarded transport')
    }
    const hasResponseBody = method !== 'HEAD' && ![204, 205, 304].includes(response.status)
    const responseBody = hasResponseBody
      ? await readBrowserResponseBody(response, budget)
      : undefined
    await route.fulfill({
      status: response.status,
      headers: proxyResponseHeaders(
        response.headers,
        rawUrl,
        requestHeaders,
        allowedLocalhostOrigin,
      ),
      ...(responseBody ? { body: responseBody } : {}),
    })
  } finally {
    await response?.body?.cancel().catch(() => undefined)
    await dispatcher?.close().catch(() => undefined)
  }
}

async function assertBrowserUrlAllowed(
  rawUrl: string,
  egressPolicy: EgressPolicy,
  toolName: string,
  allowedLocalhostOrigin?: string | null,
): Promise<void> {
  if (!isSameAllowedLocalhostOrigin(rawUrl, allowedLocalhostOrigin)) {
    egressPolicy.assertAllowed(rawUrl, toolName)
  }
  await assertBrowserNetworkAllowed(rawUrl, allowedLocalhostOrigin)
}

async function gotoPublicUrl(
  page: import('playwright').Page,
  rawUrl: string,
  egressPolicy: EgressPolicy,
  toolName: string,
  allowedLocalhostOrigin?: string | null,
): Promise<void> {
  let currentUrl = rawUrl
  for (let redirects = 0; ; redirects += 1) {
    await assertBrowserUrlAllowed(currentUrl, egressPolicy, toolName, allowedLocalhostOrigin)
    const state = guardedPageStates.get(page)
    if (state) {
      state.pendingRedirect = undefined
      state.lastRequestError = undefined
    }
    let response: import('playwright').Response | null
    try {
      response = await page.goto(currentUrl, {
        timeout: 30000,
        waitUntil: 'domcontentloaded',
      })
    } catch (error) {
      const pendingRedirect = state?.pendingRedirect
      if (!pendingRedirect && state?.lastRequestError) {
        throw new Error(
          `Guarded browser request failed: ${formatBrowserErrorChain(state.lastRequestError)}`,
          { cause: state.lastRequestError },
        )
      }
      if (!pendingRedirect) throw error
      if (redirects >= MAX_BROWSER_NAVIGATION_REDIRECTS) {
        throw new Error(
          `Browser navigation exceeded ${MAX_BROWSER_NAVIGATION_REDIRECTS} redirects`,
        )
      }
      currentUrl = pendingRedirect
      continue
    }
    const status = response?.status()
    if (
      typeof status === 'number' &&
      [301, 302, 303, 307, 308].includes(status)
    ) {
      const location = await response?.headerValue('location')
      if (!location) {
        throw new Error(`Browser navigation returned HTTP ${status} without a Location header`)
      }
      if (redirects >= MAX_BROWSER_NAVIGATION_REDIRECTS) {
        throw new Error(
          `Browser navigation exceeded ${MAX_BROWSER_NAVIGATION_REDIRECTS} redirects`,
        )
      }
      currentUrl = new URL(location, response?.url() || currentUrl).toString()
      continue
    }
    if (typeof status === 'number' && status >= 400) {
      const statusText = response?.statusText()?.trim()
      const responseUrl = response?.url() || page.url()
      throw new Error(
        `Browser navigation returned HTTP ${status}${statusText ? ` ${statusText}` : ''} for ${responseUrl}`,
      )
    }
    await assertBrowserUrlAllowed(page.url(), egressPolicy, toolName, allowedLocalhostOrigin)
    return
  }
}

function formatBrowserErrorChain(error: Error): string {
  const messages: string[] = []
  const seen = new Set<unknown>()
  let current: unknown = error
  while (current instanceof Error && !seen.has(current) && messages.length < 4) {
    seen.add(current)
    const message = current.message.trim()
    if (message && messages[messages.length - 1] !== message) messages.push(message)
    current = current.cause
  }
  return messages.join(': ') || error.name
}

function resultForBrowserError(err: unknown, start: number): ToolResult {
  return {
    output: err instanceof Error ? err.message : String(err),
    status: 'error',
    durationMs: Date.now() - start,
    code: err instanceof EgressDeniedError ? err.code : undefined,
  }
}

function normalizeViewportDimension(value: unknown, fallback: number): number {
  const numeric =
    typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return fallback
  }
  return Math.max(100, Math.min(8000, Math.floor(numeric)))
}

function screenshotViewport(
  input: Record<string, unknown>,
): { width: number; height: number } | null {
  if (input.viewportWidth == null && input.viewportHeight == null) {
    return null
  }
  return {
    width: normalizeViewportDimension(input.viewportWidth, 1280),
    height: normalizeViewportDimension(input.viewportHeight, 720),
  }
}

function normalizeWaitMs(value: unknown, fallback = 0): number {
  const numeric =
    typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN
  if (!Number.isFinite(numeric) || numeric < 0) return fallback
  return Math.max(0, Math.min(10000, Math.floor(numeric)))
}

const DEFAULT_POST_INTERACTION_AUDIT_SETTLE_MS = 300

/**
 * A click/evaluate call that also captures a screenshot is normally asking
 * for the settled UI state, not an intermediate CSS-transition frame. An
 * immediate audit can combine the new foreground color with the old
 * background (or vice versa) and manufacture low-contrast findings that send
 * the agent into a repair loop. Give screenshot-backed interaction audits a
 * short default settle window while preserving exact caller control: an
 * explicit waitAfterMs=0 still captures the immediate/transient state, and
 * evaluate calls without a screenshot remain immediate.
 */
function normalizePostInteractionAuditWaitMs(input: Record<string, unknown>): number {
  const fallback = typeof input.path === 'string' && input.path.trim()
    ? DEFAULT_POST_INTERACTION_AUDIT_SETTLE_MS
    : 0
  return normalizeWaitMs(input.waitAfterMs, fallback)
}

function formatEvaluateResult(value: unknown): string {
  if (value === undefined) return 'undefined'
  if (typeof value === 'string') return value.slice(0, 50000)
  try {
    return JSON.stringify(value, null, 2).slice(0, 50000)
  } catch {
    return String(value).slice(0, 50000)
  }
}

function compactBrowserAuditLine(value: string, max = 240): string {
  const normalized = value.replace(/\s+/g, ' ').trim()
  return normalized.length <= max ? normalized : `${normalized.slice(0, max - 3).trimEnd()}...`
}

function installBrowserConsolePageAudit(page: import('playwright').Page): BrowserConsolePageAudit {
  const audit: BrowserConsolePageAudit = {
    consoleErrors: [],
    consoleWarnings: [],
    pageErrors: [],
  }
  page.on('console', (message) => {
    const type = message.type()
    if (type !== 'error' && type !== 'warning') return
    const location = message.location()
    const suffix = location.url
      ? ` (${location.url}:${location.lineNumber + 1}:${location.columnNumber + 1})`
      : ''
    const line = compactBrowserAuditLine(`console.${type}: ${message.text()}${suffix}`)
    if (type === 'error') {
      if (audit.consoleErrors.length < 12) audit.consoleErrors.push(line)
    } else if (audit.consoleWarnings.length < 12) {
      audit.consoleWarnings.push(line)
    }
  })
  page.on('pageerror', (error) => {
    if (audit.pageErrors.length < 12) {
      audit.pageErrors.push(compactBrowserAuditLine(`pageerror: ${error.message}`))
    }
  })
  return audit
}

function formatBrowserConsolePageAudit(audit: BrowserConsolePageAudit): string {
  const errors = [...audit.pageErrors, ...audit.consoleErrors]
  const errorLines =
    errors.length > 0
      ? errors.map((error) => `  - ${error}`).join('\n')
      : '  - none detected by browser console/page audit'
  const warningLines =
    audit.consoleWarnings.length > 0
      ? audit.consoleWarnings.map((warning) => `  - ${warning}`).join('\n')
      : '  - none detected by browser console/page audit'
  return ['Browser console/page audit:', '- errors:', errorLines, '- warnings:', warningLines].join(
    '\n',
  )
}

function createBrowserEgressPolicy(options: BrowserToolOptions): EgressPolicy {
  return createEgressPolicy({
    allowlist: options.egressAllowlist,
    env: options.env,
    logger: options.egressLogger,
  })
}

interface BrowserScreenshotImageAttachment {
  contentParts?: ContentPart[]
  auditLine: string
}

async function imageAttachmentForPngFile(path: string): Promise<BrowserScreenshotImageAttachment> {
  try {
    const fileStat = await stat(path)
    if (!fileStat.isFile()) {
      return {
        auditLine: 'Screenshot image attachment: omitted because the screenshot path is not a file',
      }
    }
    if (fileStat.size <= 0) {
      return {
        auditLine: 'Screenshot image attachment: omitted because the screenshot file is empty',
      }
    }
    if (fileStat.size > MAX_BROWSER_SCREENSHOT_IMAGE_BYTES) {
      return {
        auditLine: `Screenshot image attachment: omitted because the screenshot is ${fileStat.size} bytes, above the ${MAX_BROWSER_SCREENSHOT_IMAGE_BYTES} byte attachment limit`,
      }
    }
    const data = await readFile(path)
    return {
      contentParts: [
        {
          type: 'image',
          source: {
            type: 'base64',
            mediaType: 'image/png',
            data: data.toString('base64'),
          },
        },
      ],
      auditLine: BROWSER_SCREENSHOT_IMAGE_ATTACHMENT_ATTACHED,
    }
  } catch {
    return {
      auditLine:
        'Screenshot image attachment: omitted because the screenshot file could not be read',
    }
  }
}

async function captureOptionalScreenshot(
  page: import('playwright').Page,
  input: Record<string, unknown>,
  context: Parameters<NonNullable<ToolDefinitionRuntime['execute']>>[1],
): Promise<{
  outputPath: string
  contentParts?: ContentPart[]
  imageAttachmentAudit: string
  layoutAudit?: string
  fallbackMessage?: string
} | null> {
  if (typeof input.path !== 'string' || !input.path.trim()) {
    return null
  }
  const outputPath = resolveToolPath(input.path, context?.cwd)
  const capture = await saveBrowserScreenshot(page, outputPath, Boolean(input.fullPage))
  const imageAttachment = await imageAttachmentForPngFile(outputPath)
  return {
    outputPath,
    contentParts: imageAttachment.contentParts,
    imageAttachmentAudit: imageAttachment.auditLine,
    layoutAudit: await collectBrowserLayoutAudit(page, capture.fullPage),
    fallbackMessage: capture.fallbackMessage,
  }
}

function screenshotErrorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

function shouldRetryViewportScreenshot(err: unknown): boolean {
  const message = screenshotErrorMessage(err)
  return (
    message.includes('Page.captureScreenshot') || message.includes('Unable to capture screenshot')
  )
}

function summarizeScreenshotError(err: unknown): string {
  return screenshotErrorMessage(err).replace(/\s+/g, ' ').trim().slice(0, 300)
}

async function saveBrowserScreenshot(
  page: import('playwright').Page,
  outputPath: string,
  fullPage: boolean,
): Promise<{ fullPage: boolean; fallbackMessage?: string }> {
  try {
    await page.screenshot({
      path: outputPath,
      fullPage,
      type: 'png',
    })
    return { fullPage }
  } catch (err) {
    if (!fullPage || !shouldRetryViewportScreenshot(err)) throw err
    await page.screenshot({
      path: outputPath,
      fullPage: false,
      type: 'png',
    })
    return {
      fullPage: false,
      fallbackMessage: `Full-page screenshot failed (${summarizeScreenshotError(err)}); saved a viewport screenshot instead.`,
    }
  }
}

function isBrowserLayoutAudit(value: unknown): value is BrowserLayoutAudit {
  if (!value || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return (
    typeof record.horizontalOverflowPx === 'number' &&
    typeof record.visibleElementCount === 'number' &&
    Array.isArray(record.warnings) &&
    Array.isArray(record.examples) &&
    !!record.viewport &&
    typeof record.viewport === 'object' &&
    !!record.document &&
    typeof record.document === 'object' &&
    !!record.body &&
    typeof record.body === 'object'
  )
}

function formatBrowserLayoutAudit(value: unknown): string | null {
  if (!isBrowserLayoutAudit(value)) return null
  const warnings =
    value.warnings.length > 0
      ? value.warnings.map((warning) => `  - ${warning}`).join('\n')
      : '  - none detected by DOM/canvas layout audit'
  const examples =
    value.examples.length > 0 ? `\n- examples: ${value.examples.slice(0, 5).join('; ')}` : ''

  return [
    'Browser layout audit:',
    `- viewport: ${value.viewport.width}x${value.viewport.height}; document: ${value.document.width}x${value.document.height}; body: ${value.body.width}x${value.body.height} (scrollHeight ${value.body.scrollHeight}, computed ${value.body.computedHeight})`,
    `- horizontal overflow: ${value.horizontalOverflowPx}px; visible elements: ${value.visibleElementCount}; visible in viewport: ${value.visibleViewportElementCount}`,
    '- warnings:',
    warnings,
    examples,
  ]
    .filter(Boolean)
    .join('\n')
}

const BROWSER_LAYOUT_AUDIT_SOURCE = `
const doc = document.documentElement
const body = document.body
const viewportWidth = window.innerWidth
const viewportHeight = window.innerHeight
const documentWidth = Math.max(doc.scrollWidth, body?.scrollWidth ?? 0, doc.offsetWidth, body?.offsetWidth ?? 0)
const documentHeight = Math.max(doc.scrollHeight, body?.scrollHeight ?? 0, doc.offsetHeight, body?.offsetHeight ?? 0)
const bodyRect = body?.getBoundingClientRect()
const bodyStyle = body ? window.getComputedStyle(body) : null
const bodyHeight = Math.round(bodyRect?.height ?? 0)
const bodyWidth = Math.round(bodyRect?.width ?? 0)
const bodyScrollHeight = body?.scrollHeight ?? 0
const horizontalOverflowPx = Math.max(0, Math.round(documentWidth - viewportWidth))
const warnings = []
const examples = []

const escapeCss = (value) => {
  const css = globalThis.CSS
  return css?.escape ? css.escape(value) : value.replace(/[^a-zA-Z0-9_-]/g, '\\\\$&')
}
const labelFor = (element) => {
  if (element.id) return '#' + escapeCss(element.id)
  const tag = element.tagName.toLowerCase()
  const className = typeof element.className === 'string'
    ? element.className.trim().split(/\\s+/).slice(0, 3).filter(Boolean).map((name) => '.' + escapeCss(name)).join('')
    : ''
  return tag + className
}
const isVisible = (element, rect) => {
  if (rect.width < 1 || rect.height < 1) return false
  const style = window.getComputedStyle(element)
  return style.display !== 'none'
    && style.visibility !== 'hidden'
    && Number(style.opacity || '1') > 0.01
}

let visibleElementCount = 0
let visibleViewportElementCount = 0
let visibleElementsBeyondBody = 0
let maxVisibleBottom = 0
let maxVisibleRight = 0
let fixedViewportCoveringElements = 0
let smallInteractiveTargets = 0
let clippedControlTextElements = 0
let brokenImageElements = 0
let lowContrastTextElements = 0
let overlappingTextControlPairs = 0
const overlapCandidates = []
const allElements = Array.from(document.body?.querySelectorAll('*') ?? []).slice(0, 1000)
const roleFor = (element) => (element.getAttribute('role') ?? '').toLowerCase()
const inputTypeFor = (element) => (element.getAttribute('type') ?? '').toLowerCase()
const isButtonLikeControl = (element) => {
  const tag = element.tagName.toLowerCase()
  const role = roleFor(element)
  const inputType = inputTypeFor(element)
  if (tag === 'button' || tag === 'select' || tag === 'textarea' || tag === 'summary') return true
  if (tag === 'input') {
    return !['checkbox', 'radio', 'hidden', 'range', 'color', 'file'].includes(inputType)
  }
  if (['button', 'menuitem', 'tab', 'switch'].includes(role)) return true
  if (element.hasAttribute('onclick')) return true
  return false
}
const hasMeaningfulControlText = (element) => {
  const text = (element.innerText ?? element.textContent ?? '').replace(/\\s+/g, ' ').trim()
  const aria = (element.getAttribute('aria-label') ?? '').trim()
  const title = (element.getAttribute('title') ?? '').trim()
  return text.length >= 2 || aria.length >= 2 || title.length >= 2
}
const intersectsViewport = (rect) =>
  rect.bottom > 0
  && rect.top < viewportHeight
  && rect.right > 0
  && rect.left < viewportWidth
const viewportClippedRect = (rect) => {
  const left = Math.max(0, rect.left)
  const top = Math.max(0, rect.top)
  const right = Math.min(viewportWidth, rect.right)
  const bottom = Math.min(viewportHeight, rect.bottom)
  const width = Math.max(0, right - left)
  const height = Math.max(0, bottom - top)
  return { left, top, right, bottom, width, height, area: width * height }
}
const viewportClippedRects = (element, fallbackRect) => {
  const rects = Array.from(element.getClientRects?.() ?? [])
  const source = rects.length > 0 ? rects : [fallbackRect]
  return source
    .map(viewportClippedRect)
    .filter((rect) => rect.area >= 120 && rect.width >= 8 && rect.height >= 8)
}
const overlapBox = (left, right) => {
  const x1 = Math.max(left.left, right.left)
  const y1 = Math.max(left.top, right.top)
  const x2 = Math.min(left.right, right.right)
  const y2 = Math.min(left.bottom, right.bottom)
  const width = Math.max(0, x2 - x1)
  const height = Math.max(0, y2 - y1)
  return { width, height, area: width * height }
}
const parseRgbColor = (value) => {
  const match = String(value ?? '').match(/rgba?\\(([^)]+)\\)/i)
  if (!match) return null
  const parts = match[1].split(',').map((part) => part.trim())
  if (parts.length < 3) return null
  const alpha = parts[3] == null ? 1 : Number(parts[3])
  return {
    r: Number(parts[0]),
    g: Number(parts[1]),
    b: Number(parts[2]),
    a: Number.isFinite(alpha) ? alpha : 1,
  }
}
const relativeLuminance = (color) => {
  const channel = (value) => {
    const normalized = Math.max(0, Math.min(255, value)) / 255
    return normalized <= 0.03928
      ? normalized / 12.92
      : Math.pow((normalized + 0.055) / 1.055, 2.4)
  }
  return 0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b)
}
const contrastRatio = (left, right) => {
  const leftLuma = relativeLuminance(left)
  const rightLuma = relativeLuminance(right)
  const lighter = Math.max(leftLuma, rightLuma)
  const darker = Math.min(leftLuma, rightLuma)
  return (lighter + 0.05) / (darker + 0.05)
}
const effectiveBackgroundColor = (element) => {
  let current = element
  while (current) {
    const style = window.getComputedStyle(current)
    const background = parseRgbColor(style.backgroundColor)
    if (background && background.a >= 0.85) return background
    current = current.parentElement
  }
  return parseRgbColor(window.getComputedStyle(document.body ?? document.documentElement).backgroundColor)
    ?? { r: 255, g: 255, b: 255, a: 1 }
}
for (const element of allElements) {
  const rect = element.getBoundingClientRect()
  if (!isVisible(element, rect)) continue
  visibleElementCount += 1
  const absoluteBottom = Math.round(rect.bottom + window.scrollY)
  const absoluteRight = Math.round(rect.right + window.scrollX)
  maxVisibleBottom = Math.max(maxVisibleBottom, absoluteBottom)
  maxVisibleRight = Math.max(maxVisibleRight, absoluteRight)
  if (rect.bottom > 0 && rect.top < viewportHeight && rect.right > 0 && rect.left < viewportWidth) {
    visibleViewportElementCount += 1
  }
  if (bodyHeight > 0 && absoluteBottom > bodyHeight + 4) {
    visibleElementsBeyondBody += 1
    if (examples.length < 5) {
      examples.push(labelFor(element) + ' bottom ' + absoluteBottom + 'px exceeds body height ' + bodyHeight + 'px')
    }
  }
  const style = window.getComputedStyle(element)
  const inViewport = intersectsViewport(rect)
  if (
    (style.position === 'fixed' || style.position === 'sticky')
    && rect.width >= viewportWidth * 0.9
    && rect.height >= viewportHeight * 0.35
  ) {
    fixedViewportCoveringElements += 1
    if (examples.length < 5) {
      examples.push(labelFor(element) + ' ' + style.position + ' element covers ' + Math.round(rect.width) + 'x' + Math.round(rect.height) + 'px')
    }
  }
  const buttonLike = isButtonLikeControl(element)
  const meaningfulControlText = buttonLike && hasMeaningfulControlText(element)
  if (
    buttonLike
    && viewportWidth <= 767
    && inViewport
    && (rect.width < 32 || rect.height < 32)
  ) {
    smallInteractiveTargets += 1
    if (examples.length < 5) {
      examples.push(labelFor(element) + ' interactive target is only ' + Math.round(rect.width) + 'x' + Math.round(rect.height) + 'px in the mobile viewport')
    }
  }
  if (
    meaningfulControlText
    && (element.scrollWidth > element.clientWidth + 2 || element.scrollHeight > element.clientHeight + 2)
  ) {
    clippedControlTextElements += 1
    if (examples.length < 5) {
      examples.push(labelFor(element) + ' text/content size ' + element.scrollWidth + 'x' + element.scrollHeight + 'px exceeds its box ' + element.clientWidth + 'x' + element.clientHeight + 'px')
    }
  }
  if (
    element.tagName.toLowerCase() === 'img'
    && (element.currentSrc || element.getAttribute('src'))
    && element.complete
    && element.naturalWidth === 0
    && rect.width >= 16
    && rect.height >= 16
  ) {
    brokenImageElements += 1
    if (examples.length < 5) {
      examples.push(labelFor(element) + ' image failed to load; rendered box is ' + Math.round(rect.width) + 'x' + Math.round(rect.height) + 'px')
    }
  }
  const directText = Array.from(element.childNodes ?? [])
    .filter((node) => node.nodeType === Node.TEXT_NODE)
    .map((node) => node.textContent ?? '')
    .join(' ')
    .replace(/\\s+/g, ' ')
    .trim()
  if (
    directText.length >= 4
    && inViewport
    && rect.width >= 24
    && rect.height >= 10
  ) {
    const color = parseRgbColor(style.color)
    const background = effectiveBackgroundColor(element)
    if (color && background) {
      const ratio = contrastRatio(color, background)
      if (ratio < 3) {
        lowContrastTextElements += 1
        if (examples.length < 5) {
          examples.push(labelFor(element) + ' text contrast ratio is about ' + Math.round(ratio * 10) / 10 + ':1 against its background')
        }
      }
    }
  }
  if (
    (directText.length >= 4 || meaningfulControlText)
    && inViewport
    && rect.width >= 24
    && rect.height >= 10
    && overlapCandidates.length < 120
  ) {
    const rects = viewportClippedRects(element, rect)
    const area = rects.reduce((sum, candidateRect) => sum + candidateRect.area, 0)
    if (area >= 240) {
      overlapCandidates.push({ element, label: labelFor(element), rects, area })
    }
  }
}

for (let i = 0; i < overlapCandidates.length; i += 1) {
  const left = overlapCandidates[i]
  for (let j = i + 1; j < overlapCandidates.length; j += 1) {
    const right = overlapCandidates[j]
    if (left.element.contains(right.element) || right.element.contains(left.element)) continue
    let overlap = { width: 0, height: 0, area: 0 }
    for (const leftRect of left.rects) {
      for (const rightRect of right.rects) {
        const fragmentOverlap = overlapBox(leftRect, rightRect)
        if (fragmentOverlap.area > overlap.area) overlap = fragmentOverlap
      }
    }
    if (overlap.area < 120) continue
    const smallerArea = Math.min(left.area, right.area)
    const overlapRatio = smallerArea > 0 ? overlap.area / smallerArea : 0
    if (overlapRatio < 0.35 && overlap.area < 900) continue
    overlappingTextControlPairs += 1
    if (examples.length < 5) {
      examples.push(left.label + ' overlaps ' + right.label + ' by about ' + Math.round(overlap.width) + 'x' + Math.round(overlap.height) + 'px')
    }
  }
}

const colorDistance = (a, b) => {
  const dr = a[0] - b[0]
  const dg = a[1] - b[1]
  const db = a[2] - b[2]
  const da = a[3] - b[3]
  return Math.sqrt(dr * dr + dg * dg + db * db + da * da * 0.25)
}
const quietBandSize = (values, limit) => {
  let count = 0
  for (const value of values) {
    if (value > limit) break
    count += 1
  }
  return count
}
const summarizeCanvas = (canvas) => {
  const rect = canvas.getBoundingClientRect()
  if (!isVisible(canvas, rect)) return
  if (rect.width < 80 || rect.height < 80) return
  const sourceWidth = Math.max(1, Math.floor(canvas.width || rect.width))
  const sourceHeight = Math.max(1, Math.floor(canvas.height || rect.height))
  if (sourceWidth < 8 || sourceHeight < 8) return
  const sampleWidth = Math.max(8, Math.min(64, sourceWidth))
  const sampleHeight = Math.max(8, Math.min(64, sourceHeight))
  const sample = document.createElement('canvas')
  sample.width = sampleWidth
  sample.height = sampleHeight
  const ctx = sample.getContext('2d', { willReadFrequently: true })
  if (!ctx) return
  let data
  try {
    ctx.drawImage(canvas, 0, 0, sampleWidth, sampleHeight)
    data = ctx.getImageData(0, 0, sampleWidth, sampleHeight).data
  } catch {
    if (examples.length < 5) {
      examples.push(labelFor(canvas) + ' canvas could not be pixel-audited; it may be WebGL or cross-origin-tainted')
    }
    return
  }

  const buckets = new Map()
  const pixels = []
  let opaquePixels = 0
  let lumaSum = 0
  let lumaSquareSum = 0
  for (let index = 0; index < data.length; index += 4) {
    const pixel = [data[index], data[index + 1], data[index + 2], data[index + 3]]
    pixels.push(pixel)
    if (pixel[3] > 16) opaquePixels += 1
    const luma = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
    lumaSum += luma
    lumaSquareSum += luma * luma
    const key = [pixel[0] >> 4, pixel[1] >> 4, pixel[2] >> 4, pixel[3] >> 4].join(',')
    buckets.set(key, (buckets.get(key) ?? 0) + 1)
  }
  const totalPixels = sampleWidth * sampleHeight
  const dominant = [...buckets.entries()].sort((a, b) => b[1] - a[1])[0]?.[0]
  if (!dominant) return
  const dominantColor = dominant.split(',').map((part) => Number(part) * 16 + 8)
  const rowActivity = Array.from({ length: sampleHeight }, () => 0)
  const columnActivity = Array.from({ length: sampleWidth }, () => 0)
  let activePixels = 0
  for (let index = 0; index < pixels.length; index += 1) {
    const pixel = pixels[index]
    const distance = colorDistance(pixel, dominantColor)
    if (distance <= 42) continue
    activePixels += 1
    const y = Math.floor(index / sampleWidth)
    const x = index % sampleWidth
    rowActivity[y] += 1
    columnActivity[x] += 1
  }
  const activeRatio = activePixels / totalPixels
  const opacityRatio = opaquePixels / totalPixels
  const lumaMean = lumaSum / totalPixels
  const lumaVariance = Math.max(0, lumaSquareSum / totalPixels - lumaMean * lumaMean)
  const lumaStdDev = Math.sqrt(lumaVariance)
  const rowRatios = rowActivity.map((value) => value / sampleWidth)
  const columnRatios = columnActivity.map((value) => value / sampleHeight)
  const quietLimit = Math.max(0.01, Math.min(0.03, activeRatio * 0.4))
  const topQuietRows = quietBandSize(rowRatios, quietLimit)
  const bottomQuietRows = quietBandSize([...rowRatios].reverse(), quietLimit)
  const leftQuietColumns = quietBandSize(columnRatios, quietLimit)
  const rightQuietColumns = quietBandSize([...columnRatios].reverse(), quietLimit)
  const cssTopQuiet = Math.round(topQuietRows / sampleHeight * rect.height)
  const cssBottomQuiet = Math.round(bottomQuietRows / sampleHeight * rect.height)
  const cssLeftQuiet = Math.round(leftQuietColumns / sampleWidth * rect.width)
  const cssRightQuiet = Math.round(rightQuietColumns / sampleWidth * rect.width)
  const label = labelFor(canvas)

  if (opacityRatio < 0.01) {
    warnings.push(label + ' canvas appears blank or fully transparent by pixel audit')
  } else if (activeRatio < 0.005 && lumaStdDev < 2) {
    warnings.push(label + ' canvas has almost no pixel variation; inspect for a blank or stalled render')
  } else if (activeRatio < 0.025 && rect.width * rect.height > viewportWidth * viewportHeight * 0.2) {
    warnings.push(label + ' canvas has very low visual density (' + Math.round(activeRatio * 1000) / 10 + '% active pixels); inspect for an empty-looking primary surface')
  }
  const hasMeaningfulActiveContent = activeRatio >= 0.005 || lumaStdDev >= 2
  if (hasMeaningfulActiveContent && cssTopQuiet > Math.max(96, rect.height * 0.28)) {
    warnings.push(label + ' canvas has a low-detail top band of about ' + cssTopQuiet + 'px; inspect for excessive empty space in the primary surface')
  }
  if (hasMeaningfulActiveContent && cssBottomQuiet > Math.max(96, rect.height * 0.28)) {
    warnings.push(label + ' canvas has a low-detail bottom band of about ' + cssBottomQuiet + 'px; inspect for excessive empty space in the primary surface')
  }
  if (hasMeaningfulActiveContent && cssLeftQuiet > Math.max(96, rect.width * 0.28)) {
    warnings.push(label + ' canvas has a low-detail left band of about ' + cssLeftQuiet + 'px; inspect for poor framing or off-center content')
  }
  if (hasMeaningfulActiveContent && cssRightQuiet > Math.max(96, rect.width * 0.28)) {
    warnings.push(label + ' canvas has a low-detail right band of about ' + cssRightQuiet + 'px; inspect for poor framing or off-center content')
  }
  if (examples.length < 5 && (activeRatio < 0.05 || cssTopQuiet > 0 || cssBottomQuiet > 0)) {
    examples.push(label + ' canvas pixel audit: ' + Math.round(activeRatio * 1000) / 10 + '% active pixels, luma stddev ' + Math.round(lumaStdDev * 10) / 10)
  }
}

for (const canvas of Array.from(document.body?.querySelectorAll('canvas') ?? []).slice(0, 8)) {
  summarizeCanvas(canvas)
}

const bodyPaintShortfall = documentHeight - bodyHeight
const documentExtendsPastViewport = documentHeight > viewportHeight + Math.max(80, viewportHeight * 0.1)
if (horizontalOverflowPx > 2) {
  warnings.push('document is ' + horizontalOverflowPx + 'px wider than the viewport; inspect for horizontal overflow')
}
// A short content body inside a one-viewport document is normal: the root
// canvas still paints the viewport background, so treating the unused area as
// an unpainted full-page band creates a false defect on centered dashboards.
// The clipping/blank-band risk starts only when the document actually extends
// beyond the viewport while the body paint box does not.
if (documentExtendsPastViewport && bodyPaintShortfall > Math.max(80, viewportHeight * 0.1) && bodyHeight > 0) {
  warnings.push('document is ' + bodyPaintShortfall + 'px taller than the body paint height; full-page screenshots may show an unpainted or blank lower band')
}
if (documentExtendsPastViewport && visibleElementsBeyondBody > 0 && bodyPaintShortfall > Math.max(80, viewportHeight * 0.1)) {
  warnings.push(visibleElementsBeyondBody + ' visible element(s) extend below the body paint height; inspect for clipped content or a blank band after the first viewport')
}
if (capturesFullPage && maxVisibleBottom > 0 && documentHeight - maxVisibleBottom > Math.max(160, viewportHeight * 0.25)) {
  warnings.push('last visible element ends ' + (documentHeight - maxVisibleBottom) + 'px before the document bottom; inspect for excessive blank space')
}
if (maxVisibleRight > viewportWidth + 2) {
  warnings.push('visible content extends ' + Math.round(maxVisibleRight - viewportWidth) + 'px past the viewport right edge')
}
if (fixedViewportCoveringElements > 0) {
  warnings.push(fixedViewportCoveringElements + ' fixed/sticky element(s) cover a large viewport area; inspect for accidental overlays')
}
if (smallInteractiveTargets > 0) {
  warnings.push(smallInteractiveTargets + ' visible interactive control(s) are smaller than 32px in the mobile viewport; inspect touch target sizing')
}
if (clippedControlTextElements > 0) {
  warnings.push(clippedControlTextElements + ' visible control(s) appear to clip or overflow their text/content; inspect button/input label fitting')
}
if (brokenImageElements > 0) {
  warnings.push(brokenImageElements + ' visible image element(s) failed to load; inspect missing asset paths or broken media')
}
if (lowContrastTextElements > 0) {
  warnings.push(lowContrastTextElements + ' visible text element(s) have low color contrast; inspect readability against the background')
}
if (overlappingTextControlPairs > 0) {
  warnings.push(overlappingTextControlPairs + ' visible text/control overlap(s) detected; inspect text collisions or stacked controls')
}
if (visibleElementCount === 0) {
  warnings.push('no visible body elements were detected after rendering')
}

return {
  viewport: { width: viewportWidth, height: viewportHeight },
  document: { width: documentWidth, height: documentHeight },
  body: {
    width: bodyWidth,
    height: bodyHeight,
    scrollHeight: bodyScrollHeight,
    computedHeight: bodyStyle?.height ?? 'unknown',
  },
  horizontalOverflowPx,
  visibleElementCount,
  visibleViewportElementCount,
  warnings,
  examples,
}
`

async function collectBrowserLayoutAudit(
  page: import('playwright').Page,
  fullPage: boolean,
): Promise<string | undefined> {
  try {
    const auditFunction = new Function('capturesFullPage', BROWSER_LAYOUT_AUDIT_SOURCE) as (
      capturesFullPage: boolean,
    ) => BrowserLayoutAudit
    const audit = await page.evaluate(auditFunction, fullPage)
    return formatBrowserLayoutAudit(audit) ?? undefined
  } catch {
    return undefined
  }
}

function localhostOriginFor(
  rawUrl: string,
  options: BrowserToolOptions,
  sessionId?: string,
  workspaceRoot?: string,
): string | null {
  if (
    !options.allowLocalhost
    && !options.isLocalhostAllowed?.(sessionId, rawUrl, workspaceRoot)
  ) return null
  try {
    const parsed = new URL(rawUrl)
    return isLocalhost(parsed.hostname) ? parsed.origin : null
  } catch {
    return null
  }
}

function isLocalhost(hostname: string): boolean {
  const host = hostname.replace(/^\[(.*)\]$/, '$1').toLowerCase()
  return host === 'localhost' || host === '127.0.0.1' || host === '::1'
}

function isSameAllowedLocalhostOrigin(
  rawUrl: string,
  allowedOrigin: string | null | undefined,
): boolean {
  if (!allowedOrigin) return false
  try {
    const parsed = new URL(rawUrl)
    return parsed.origin === allowedOrigin && isLocalhost(parsed.hostname)
  } catch {
    return false
  }
}

function isSameAllowedLocalhostEndpoint(
  rawUrl: string,
  allowedOrigin: string | null | undefined,
): boolean {
  if (!allowedOrigin) return false
  try {
    const candidate = new URL(rawUrl)
    const allowed = new URL(allowedOrigin)
    if (!isLocalhost(candidate.hostname) || !isLocalhost(allowed.hostname)) return false
    const candidatePort = candidate.port || (candidate.protocol === 'wss:' || candidate.protocol === 'https:' ? '443' : '80')
    const allowedPort = allowed.port || (allowed.protocol === 'https:' ? '443' : '80')
    const protocolMatches = allowed.protocol === 'https:'
      ? candidate.protocol === 'https:' || candidate.protocol === 'wss:'
      : candidate.protocol === 'http:' || candidate.protocol === 'ws:'
    return protocolMatches && candidatePort === allowedPort
  } catch {
    return false
  }
}

function managedLoopbackConnectSources(
  allowedOrigin: string | null | undefined,
): string[] {
  if (!allowedOrigin) return []
  try {
    const allowed = new URL(allowedOrigin)
    if (!isLocalhost(allowed.hostname)) return []
    const secure = allowed.protocol === 'https:'
    const websocketScheme = secure ? 'wss:' : 'ws:'
    const port = allowed.port || (secure ? '443' : '80')
    return [
      `${websocketScheme}//localhost:${port}`,
      `${websocketScheme}//127.0.0.1:${port}`,
    ]
  } catch {
    return []
  }
}

async function assertBrowserNetworkAllowed(
  rawUrl: string,
  allowedLocalhostOrigin?: string | null,
  signal?: AbortSignal,
): Promise<void> {
  if (isSameAllowedLocalhostOrigin(rawUrl, allowedLocalhostOrigin)) {
    return
  }
  await assertPublicUrl(rawUrl, signal)
}

export function createBrowserNavigateTool(options: BrowserToolOptions = {}): ToolDefinitionRuntime {
  const egressPolicy = createBrowserEgressPolicy(options)
  return {
    name: 'browser.navigate',
    description:
      'Open `url` in a real headless Chromium browser and return rendered text. Use this whenever the user explicitly asks to use a browser, open Google or another specific site, visit/navigate to a URL, or when the page needs JavaScript to render content (SPAs, dashboards, modal content). It can inspect configured localhost dev-server URLs for local UI validation. When the user has NOT named a browser and the target is purely static HTML or a JSON endpoint, webfetch is cheaper. Set `waitFor` to a CSS selector that appears once content is ready. Will fail on auth-required or policy-blocked URLs (file://, chrome://).',
    resumeSafety: 'replay-safe',
    inputSchema: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'URL to navigate to' },
        waitFor: { type: 'string', description: 'CSS selector to wait for (optional)' },
      },
      required: ['url'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      let page: import('playwright').Page | null = null
      const onAbort = () => {
        if (page) {
          void page.close().catch(() => {})
        }
      }
      context?.signal?.addEventListener('abort', onAbort, { once: true })
      try {
        throwIfAborted(context?.signal, 'Browser navigation aborted')
        const allowedLocalhostOrigin = localhostOriginFor(
          String(input.url ?? ''),
          options,
          context?.sessionId,
          context?.workspaceRoot,
        )
        await assertBrowserUrlAllowed(
          String(input.url ?? ''),
          egressPolicy,
          'browser.navigate',
          allowedLocalhostOrigin,
        )
        const b = await ensureBrowser(options)
        page = await newGuardedPage(
          b,
          egressPolicy,
          allowedLocalhostOrigin,
          options.managedLoopbackSocketForUrl?.(
            context?.sessionId,
            String(input.url ?? ''),
            context?.workspaceRoot,
          ),
          options.fetcher,
          context?.signal,
        )
        throwIfAborted(context?.signal, 'Browser navigation aborted')
        await gotoPublicUrl(
          page,
          String(input.url ?? ''),
          egressPolicy,
          'browser.navigate',
          allowedLocalhostOrigin,
        )
        if (input.waitFor) await page.waitForSelector(input.waitFor as string, { timeout: 10000 })
        const [title, text] = await Promise.all([
          page.title(),
          page.innerText('body'),
        ])
        return {
          output: formatBrowserNavigationResult({
            url: page.url(),
            title,
            renderedText: text,
          }),
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (err) {
        if (context?.signal?.aborted) {
          throw getAbortError(context?.signal, 'Browser navigation aborted')
        }
        return resultForBrowserError(err, start)
      } finally {
        context?.signal?.removeEventListener('abort', onAbort)
        if (page && !page.isClosed()) {
          await closeGuardedPage(page)
        }
      }
    },
  }
}

export function createBrowserScreenshotTool(
  options: BrowserToolOptions = {},
): ToolDefinitionRuntime {
  const egressPolicy = createBrowserEgressPolicy(options)
  return {
    name: 'browser.screenshot',
    description:
      "Save a PNG of the rendered page to `path`, attach the screenshot image for the next model turn when the file is small/readable enough, and return a DOM/canvas layout audit plus browser console/page audit with overflow, blank-band, low-detail canvas warnings, small mobile touch-target warnings, clipped control-text warnings, broken visible-image warnings, low text-contrast warnings, text/control overlap warnings, console.error, and pageerror findings. Use when a local or remote browser-rendered artifact needs visual inspection (layout, responsive UI, charts, games, design verification). Set `viewportWidth` and `viewportHeight` when validating desktop/mobile responsive states. Set `waitFor` or `waitAfterMs` when validating SPAs, games, async media, animations, or delayed render states so the screenshot does not capture only a loading/blank frame. For rendered-UI completion evidence, the output must include `Screenshot image attachment: attached`; an omitted image attachment means the screenshot path alone was not visually inspected. Treat audit warnings/errors as visual or runtime defects to inspect or fix. For text content prefer browser.navigate or browser.extract, which are cheaper and don't pollute the workspace with binary files.",
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'URL to screenshot' },
        path: { type: 'string', description: 'File path to save screenshot' },
        waitFor: {
          type: 'string',
          description:
            'CSS selector to wait for before screenshotting async-rendered UI (optional)',
        },
        waitAfterMs: {
          type: 'number',
          description:
            'Milliseconds to wait after navigation/waitFor before screenshotting (0-10000, optional)',
        },
        fullPage: { type: 'boolean', description: 'Capture full page (default: false)' },
        viewportWidth: {
          type: 'number',
          description: 'Viewport width in CSS pixels for responsive screenshots (optional)',
        },
        viewportHeight: {
          type: 'number',
          description: 'Viewport height in CSS pixels for responsive screenshots (optional)',
        },
      },
      required: ['url', 'path'],
    },
    async recoverInterruptedExecution(input, context): Promise<ToolResult | null> {
      const path = typeof input.path === 'string' ? resolveToolPath(input.path, context.cwd) : ''
      if (!path) {
        return null
      }

      try {
        const fileStat = await stat(path)
        if (
          fileStat.isFile() &&
          fileStat.size > 0 &&
          fileStat.mtime.getTime() >= new Date(context.startedAt).getTime()
        ) {
          const imageAttachment = await imageAttachmentForPngFile(path)
          return {
            output: [
              `Screenshot saved to ${path}`,
              imageAttachment.auditLine,
              '[recovery] Existing screenshot file was recovered after an interrupted browser.screenshot call.',
              '[recovery] DOM/canvas layout audit and browser console/page audit were not recovered; rerun browser.screenshot for rendered-UI validation evidence.',
            ].join('\n'),
            status: 'success',
            durationMs: 0,
            ...(imageAttachment.contentParts ? { contentParts: imageAttachment.contentParts } : {}),
          }
        }
      } catch {
        return null
      }

      return null
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      let page: import('playwright').Page | null = null
      const onAbort = () => {
        if (page) {
          void page.close().catch(() => {})
        }
      }
      context?.signal?.addEventListener('abort', onAbort, { once: true })
      try {
        throwIfAborted(context?.signal, 'Browser screenshot aborted')
        const allowedLocalhostOrigin = localhostOriginFor(
          String(input.url ?? ''),
          options,
          context?.sessionId,
          context?.workspaceRoot,
        )
        await assertBrowserUrlAllowed(
          String(input.url ?? ''),
          egressPolicy,
          'browser.screenshot',
          allowedLocalhostOrigin,
        )
        if (typeof input.path !== 'string' || !input.path.trim()) {
          return {
            output: 'path is required',
            status: 'error',
            durationMs: Date.now() - start,
            code: 'INVALID_PATH_PERMANENT',
          }
        }
        const outputPath = resolveToolPath(input.path, context?.cwd)
        const viewport = screenshotViewport(input)
        const b = await ensureBrowser(options)
        page = await newGuardedPage(
          b,
          egressPolicy,
          allowedLocalhostOrigin,
          options.managedLoopbackSocketForUrl?.(
            context?.sessionId,
            String(input.url ?? ''),
            context?.workspaceRoot,
          ),
          options.fetcher,
          context?.signal,
        )
        const consolePageAudit = installBrowserConsolePageAudit(page)
        if (viewport) {
          await page.setViewportSize(viewport)
        }
        throwIfAborted(context?.signal, 'Browser screenshot aborted')
        await gotoPublicUrl(
          page,
          String(input.url ?? ''),
          egressPolicy,
          'browser.screenshot',
          allowedLocalhostOrigin,
        )
        if (input.waitFor) await page.waitForSelector(input.waitFor as string, { timeout: 10000 })
        const waitAfterMs = normalizeWaitMs(input.waitAfterMs)
        if (waitAfterMs > 0) await page.waitForTimeout(waitAfterMs)
        const capture = await saveBrowserScreenshot(page, outputPath, Boolean(input.fullPage))
        const layoutAudit = await collectBrowserLayoutAudit(page, capture.fullPage)
        const imageAttachment = await imageAttachmentForPngFile(outputPath)
        return {
          output: [
            `Screenshot saved to ${outputPath}`,
            imageAttachment.auditLine,
            capture.fallbackMessage,
            layoutAudit,
            formatBrowserConsolePageAudit(consolePageAudit),
          ]
            .filter(Boolean)
            .join('\n'),
          status: 'success',
          durationMs: Date.now() - start,
          ...(imageAttachment.contentParts ? { contentParts: imageAttachment.contentParts } : {}),
        }
      } catch (err) {
        if (context?.signal?.aborted) {
          throw getAbortError(context?.signal, 'Browser screenshot aborted')
        }
        return resultForBrowserError(err, start)
      } finally {
        context?.signal?.removeEventListener('abort', onAbort)
        if (page && !page.isClosed()) {
          await closeGuardedPage(page)
        }
      }
    },
  }
}

export function createBrowserClickTool(options: BrowserToolOptions = {}): ToolDefinitionRuntime {
  const egressPolicy = createBrowserEgressPolicy(options)
  return {
    name: 'browser.click',
    description:
      'Open `url` in a real headless Chromium browser, click a CSS `selector`, and return the post-click rendered text plus browser console/page audit. Use for validating interactive UI states such as opening menus, starting games, selecting tabs, or triggering buttons. For interaction smoke, use a specific selector for the target control such as #start, .start-button, [data-testid="start"], or button[aria-label="Start"], not broad selectors like button or #root button. For rendered-UI completion evidence, set `path` plus viewportWidth/viewportHeight so the output includes `Screenshot image attachment: attached` and a layout audit after the click; an omitted image attachment means the active state was not visually inspected.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'URL to open before clicking' },
        selector: { type: 'string', description: 'CSS selector to click' },
        waitFor: {
          type: 'string',
          description: 'CSS selector to wait for after navigation (optional)',
        },
        waitForAfter: {
          type: 'string',
          description: 'CSS selector to wait for after the click (optional)',
        },
        waitAfterMs: {
          type: 'number',
          description:
            'Milliseconds to wait after the click before reading/screenshotting (0-10000, optional)',
        },
        path: { type: 'string', description: 'Optional PNG path to save a post-click screenshot' },
        fullPage: {
          type: 'boolean',
          description: 'Capture full page when path is set (default: false)',
        },
        viewportWidth: {
          type: 'number',
          description: 'Viewport width in CSS pixels for responsive interaction checks (optional)',
        },
        viewportHeight: {
          type: 'number',
          description: 'Viewport height in CSS pixels for responsive interaction checks (optional)',
        },
      },
      required: ['url', 'selector'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      let page: import('playwright').Page | null = null
      const onAbort = () => {
        if (page) {
          void page.close().catch(() => {})
        }
      }
      context?.signal?.addEventListener('abort', onAbort, { once: true })
      try {
        throwIfAborted(context?.signal, 'Browser click aborted')
        const allowedLocalhostOrigin = localhostOriginFor(
          String(input.url ?? ''),
          options,
          context?.sessionId,
          context?.workspaceRoot,
        )
        await assertBrowserUrlAllowed(
          String(input.url ?? ''),
          egressPolicy,
          'browser.click',
          allowedLocalhostOrigin,
        )
        const selector = typeof input.selector === 'string' ? input.selector.trim() : ''
        if (!selector) {
          return {
            output: 'selector is required',
            status: 'error',
            durationMs: Date.now() - start,
            code: 'INVALID_SELECTOR',
          }
        }
        const viewport = screenshotViewport(input)
        const b = await ensureBrowser(options)
        page = await newGuardedPage(
          b,
          egressPolicy,
          allowedLocalhostOrigin,
          options.managedLoopbackSocketForUrl?.(
            context?.sessionId,
            String(input.url ?? ''),
            context?.workspaceRoot,
          ),
          options.fetcher,
          context?.signal,
        )
        const consolePageAudit = installBrowserConsolePageAudit(page)
        if (viewport) {
          await page.setViewportSize(viewport)
        }
        throwIfAborted(context?.signal, 'Browser click aborted')
        await gotoPublicUrl(
          page,
          String(input.url ?? ''),
          egressPolicy,
          'browser.click',
          allowedLocalhostOrigin,
        )
        if (input.waitFor) await page.waitForSelector(input.waitFor as string, { timeout: 10000 })
        await page.click(selector, { timeout: 10000 })
        if (input.waitForAfter)
          await page.waitForSelector(input.waitForAfter as string, { timeout: 10000 })
        const waitAfterMs = normalizePostInteractionAuditWaitMs(input)
        if (waitAfterMs > 0) await page.waitForTimeout(waitAfterMs)
        const screenshot = await captureOptionalScreenshot(page, input, context)
        const text = await page.innerText('body')
        return {
          output: [
            `Clicked ${selector} at ${page.url()}`,
            screenshot ? `Screenshot saved to ${screenshot.outputPath}` : '',
            screenshot?.imageAttachmentAudit,
            screenshot?.fallbackMessage,
            screenshot?.layoutAudit,
            formatBrowserConsolePageAudit(consolePageAudit),
            text.slice(0, 50000),
          ]
            .filter(Boolean)
            .join('\n'),
          status: 'success',
          durationMs: Date.now() - start,
          ...(screenshot?.contentParts ? { contentParts: screenshot.contentParts } : {}),
        }
      } catch (err) {
        if (context?.signal?.aborted) {
          throw getAbortError(context?.signal, 'Browser click aborted')
        }
        return resultForBrowserError(err, start)
      } finally {
        context?.signal?.removeEventListener('abort', onAbort)
        if (page && !page.isClosed()) {
          await closeGuardedPage(page)
        }
      }
    },
  }
}

export function createBrowserEvaluateTool(options: BrowserToolOptions = {}): ToolDefinitionRuntime {
  const egressPolicy = createBrowserEgressPolicy(options)
  return {
    name: 'browser.evaluate',
    description:
      'Open `url` in a real headless Chromium browser, run a JavaScript function body in the page context, and return the serializable result plus browser console/page audit. Use for deterministic UI validation or interaction that is awkward with selectors, such as checking computed styles, console-free DOM state, localStorage, canvas dimensions, or triggering a game state. For UI interaction smoke, target a specific control such as #start, .start-button, [data-testid="start"], or button[aria-label="Start"], or use a concrete keyboard/canvas script; avoid broad selectors like document.querySelector("button") or "#root button". For games or animated canvas/WebGL work, compare state before and after elapsed time or requestAnimationFrame ticks and return explicit evidence such as canvasChanged, frameDelta, positionChanged, spriteMoved, or gameStateChanged. For rendered-UI completion evidence, set `path` plus viewportWidth/viewportHeight so the output includes `Screenshot image attachment: attached` and a layout audit after evaluation; an omitted image attachment means the evaluated state was not visually inspected.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'URL to open before evaluating script' },
        script: {
          type: 'string',
          description:
            'JavaScript function body executed in the page context. Use `return ...` for a result; async code is allowed.',
        },
        waitFor: {
          type: 'string',
          description: 'CSS selector to wait for after navigation (optional)',
        },
        waitAfterMs: {
          type: 'number',
          description:
            'Milliseconds to wait after evaluation before reading/screenshotting (0-10000, optional)',
        },
        path: {
          type: 'string',
          description: 'Optional PNG path to save a post-evaluation screenshot',
        },
        fullPage: {
          type: 'boolean',
          description: 'Capture full page when path is set (default: false)',
        },
        viewportWidth: {
          type: 'number',
          description: 'Viewport width in CSS pixels for responsive interaction checks (optional)',
        },
        viewportHeight: {
          type: 'number',
          description: 'Viewport height in CSS pixels for responsive interaction checks (optional)',
        },
      },
      required: ['url', 'script'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      let page: import('playwright').Page | null = null
      const onAbort = () => {
        if (page) {
          void page.close().catch(() => {})
        }
      }
      context?.signal?.addEventListener('abort', onAbort, { once: true })
      try {
        throwIfAborted(context?.signal, 'Browser evaluate aborted')
        const allowedLocalhostOrigin = localhostOriginFor(
          String(input.url ?? ''),
          options,
          context?.sessionId,
          context?.workspaceRoot,
        )
        await assertBrowserUrlAllowed(
          String(input.url ?? ''),
          egressPolicy,
          'browser.evaluate',
          allowedLocalhostOrigin,
        )
        const script = typeof input.script === 'string' ? input.script : ''
        if (!script.trim()) {
          return {
            output: 'script is required',
            status: 'error',
            durationMs: Date.now() - start,
            code: 'INVALID_SCRIPT',
          }
        }
        if (script.length > 20000) {
          return {
            output: 'script exceeds 20000 characters',
            status: 'error',
            durationMs: Date.now() - start,
            code: 'INVALID_SCRIPT',
          }
        }
        const viewport = screenshotViewport(input)
        const b = await ensureBrowser(options)
        page = await newGuardedPage(
          b,
          egressPolicy,
          allowedLocalhostOrigin,
          options.managedLoopbackSocketForUrl?.(
            context?.sessionId,
            String(input.url ?? ''),
            context?.workspaceRoot,
          ),
          options.fetcher,
          context?.signal,
        )
        const consolePageAudit = installBrowserConsolePageAudit(page)
        if (viewport) {
          await page.setViewportSize(viewport)
        }
        throwIfAborted(context?.signal, 'Browser evaluate aborted')
        await gotoPublicUrl(
          page,
          String(input.url ?? ''),
          egressPolicy,
          'browser.evaluate',
          allowedLocalhostOrigin,
        )
        if (input.waitFor) await page.waitForSelector(input.waitFor as string, { timeout: 10000 })
        const value = await page.evaluate(async (source) => {
          const fn = new Function(`return (async () => {\n${source}\n})()`)
          return fn()
        }, script)
        const waitAfterMs = normalizePostInteractionAuditWaitMs(input)
        if (waitAfterMs > 0) await page.waitForTimeout(waitAfterMs)
        const screenshot = await captureOptionalScreenshot(page, input, context)
        return {
          output: [
            `Evaluation result at ${page.url()}:`,
            formatEvaluateResult(value),
            screenshot ? `Screenshot saved to ${screenshot.outputPath}` : '',
            screenshot?.imageAttachmentAudit,
            screenshot?.fallbackMessage,
            screenshot?.layoutAudit,
            formatBrowserConsolePageAudit(consolePageAudit),
          ]
            .filter(Boolean)
            .join('\n'),
          status: 'success',
          durationMs: Date.now() - start,
          ...(screenshot?.contentParts ? { contentParts: screenshot.contentParts } : {}),
        }
      } catch (err) {
        if (context?.signal?.aborted) {
          throw getAbortError(context?.signal, 'Browser evaluate aborted')
        }
        return resultForBrowserError(err, start)
      } finally {
        context?.signal?.removeEventListener('abort', onAbort)
        if (page && !page.isClosed()) {
          await closeGuardedPage(page)
        }
      }
    },
  }
}

export function createBrowserExtractTool(options: BrowserToolOptions = {}): ToolDefinitionRuntime {
  const egressPolicy = createBrowserEgressPolicy(options)
  return {
    name: 'browser.extract',
    description:
      'Run `selector` against the rendered DOM and return matched non-empty text. Faster and cheaper than browser.navigate when you only need a specific region. `selector` is a CSS selector. A missing match or whitespace-only match is a permanent failure for that exact URL/selector, so choose a different selector, use browser.navigate, or switch sources instead of retrying it.',
    resumeSafety: 'replay-safe',
    inputSchema: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'URL to extract from' },
        selector: { type: 'string', description: 'CSS selector to extract' },
      },
      required: ['url', 'selector'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      let page: import('playwright').Page | null = null
      const onAbort = () => {
        if (page) {
          void page.close().catch(() => {})
        }
      }
      context?.signal?.addEventListener('abort', onAbort, { once: true })
      try {
        throwIfAborted(context?.signal, 'Browser extract aborted')
        const allowedLocalhostOrigin = localhostOriginFor(
          String(input.url ?? ''),
          options,
          context?.sessionId,
          context?.workspaceRoot,
        )
        await assertBrowserUrlAllowed(
          String(input.url ?? ''),
          egressPolicy,
          'browser.extract',
          allowedLocalhostOrigin,
        )
        const b = await ensureBrowser(options)
        page = await newGuardedPage(
          b,
          egressPolicy,
          allowedLocalhostOrigin,
          options.managedLoopbackSocketForUrl?.(
            context?.sessionId,
            String(input.url ?? ''),
            context?.workspaceRoot,
          ),
          options.fetcher,
          context?.signal,
        )
        throwIfAborted(context?.signal, 'Browser extract aborted')
        await gotoPublicUrl(
          page,
          String(input.url ?? ''),
          egressPolicy,
          'browser.extract',
          allowedLocalhostOrigin,
        )
        const selector = String(input.selector ?? '')
        const element = await page.$(selector)
        if (!element) {
          return browserExtractEmptyResult(start, selector)
        }
        try {
          const text = await element.innerText()
          if (text.trim().length === 0) {
            return browserExtractEmptyResult(start, selector)
          }
          return { output: text.slice(0, 50000), status: 'success', durationMs: Date.now() - start }
        } finally {
          await element.dispose().catch(() => {})
        }
      } catch (err) {
        if (context?.signal?.aborted) {
          throw getAbortError(context?.signal, 'Browser extract aborted')
        }
        return resultForBrowserError(err, start)
      } finally {
        context?.signal?.removeEventListener('abort', onAbort)
        if (page && !page.isClosed()) {
          await closeGuardedPage(page)
        }
      }
    },
  }
}

function browserExtractEmptyResult(start: number, selector: string): ToolResult {
  return {
    output: [
      `No usable rendered text matched CSS selector ${JSON.stringify(selector)}.`,
      'Choose a different selector, use browser.navigate for the rendered page text, or switch to another source.',
    ].join('\n'),
    status: 'error',
    durationMs: Date.now() - start,
    code: 'EMPTY_CONTENT_PERMANENT',
  }
}

/** Cleanup browser on shutdown */
export async function closeBrowser(): Promise<void> {
  if (browser) {
    await browser.close()
    browser = null
  }
}
