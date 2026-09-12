export type Resolvable<T> = T | (() => T | Promise<T>)

/**
 * Error thrown by {@link ApiHttpClient.request} when the server returns
 * a non-OK status with a recognisable nested error envelope or legacy
 * top-level `{ error: code, message, ...details }` contract. Carries the
 * structured fields so callers (cli printers, daemon error mappers, gateway
 * tool wrappers) can route by code without parsing prose. Callers that do not
 * `instanceof`-check still get the contract message instead of raw JSON.
 */
export class ApiHttpError extends Error {
  readonly status: number
  readonly code: string | undefined
  readonly requestId: string | undefined
  readonly rawBody: string
  readonly details: Readonly<Record<string, unknown>> | undefined

  constructor(init: {
    status: number
    code?: string
    message: string
    requestId?: string
    rawBody: string
    details?: Record<string, unknown>
  }) {
    super(init.message)
    this.name = 'ApiHttpError'
    this.status = init.status
    this.code = init.code
    this.requestId = init.requestId
    this.rawBody = init.rawBody
    this.details = init.details
  }
}

interface ParsedErrorEnvelope {
  code?: string
  message?: string
  requestId?: string
  details?: Record<string, unknown>
}

function errorDetails(
  value: Record<string, unknown>,
  omitted: ReadonlySet<string>,
): Record<string, unknown> | undefined {
  const details = Object.fromEntries(
    Object.entries(value).filter(([key]) => !omitted.has(key)),
  )
  return Object.keys(details).length > 0 ? details : undefined
}

function tryExtractErrorEnvelope(text: string): ParsedErrorEnvelope | null {
  if (!text) return null
  try {
    const parsed = JSON.parse(text) as Record<string, unknown> | null
    if (!parsed || typeof parsed !== 'object') return null
    const err = parsed.error
    if (typeof err === 'string' && typeof parsed.message === 'string') {
      return {
        code: err,
        message: parsed.message,
        requestId: typeof parsed.requestId === 'string' ? parsed.requestId : undefined,
        details: errorDetails(parsed, new Set(['error', 'message', 'requestId'])),
      }
    }
    if (!err || typeof err !== 'object' || Array.isArray(err)) return null
    const errorRecord = err as Record<string, unknown>
    const code = errorRecord.code
    const message = errorRecord.message
    const requestId = errorRecord.requestId
    return {
      code: typeof code === 'string' ? code : undefined,
      message: typeof message === 'string' ? message : undefined,
      requestId: typeof requestId === 'string' ? requestId : undefined,
      details: errorDetails(errorRecord, new Set(['code', 'message', 'requestId'])),
    }
  } catch {
    return null
  }
}

/**
 * Memory scope identity attached to every outgoing request as
 * `X-Memory-Scope-*` headers. The daemon uses this to isolate memory
 * reads/writes per user / channel / session. All fields optional —
 * surfaces typically set `userId` only, channel adapters set
 * channelType/channelId.
 */
export interface MemoryScope {
  userId?: string
  channelType?: string
  channelId?: string
  sessionId?: string
  groupIds?: string[]
}

/**
 * Build the daemon's memory-scope request headers from a resolved scope.
 * HTTP and WebSocket transports share this mapping so memory isolation
 * cannot drift between otherwise equivalent chat paths.
 */
export function buildMemoryScopeHeaders(
  scope: MemoryScope | null | undefined,
): Record<string, string> {
  if (!scope) return {}

  const headers: Record<string, string> = {}
  const setIfPresent = (header: string, value: string | undefined) => {
    if (value) headers[header] = value.trim()
  }

  setIfPresent('X-Memory-Scope-User-Id', scope.userId)
  setIfPresent('X-Memory-Scope-Channel-Type', scope.channelType)
  setIfPresent('X-Memory-Scope-Channel-Id', scope.channelId)
  setIfPresent('X-Memory-Scope-Session-Id', scope.sessionId)

  const groups = scope.groupIds
    ?.map((entry) => entry?.trim())
    .filter((entry): entry is string => Boolean(entry))
    .join(',')
  if (groups) headers['X-Memory-Scope-Groups'] = groups

  return headers
}

export interface ApiHttpClientOptions {
  baseUrl: Resolvable<string>
  token?: Resolvable<string | null | undefined>
  defaultHeaders?: Resolvable<HeadersInit | undefined>
  fetch?: typeof fetch
  timeoutMs?: Resolvable<number | null | undefined>
  /** Optional UI/client surface label advertised as X-Sepilotd-Surface. */
  surface?: Resolvable<string | null | undefined>
  /**
   * Memory scope to advertise to the daemon. The client merges these
   * into the X-Memory-Scope-* headers automatically; per-call headers
   * still win when explicitly passed.
   */
  memoryScope?: Resolvable<MemoryScope | null | undefined>
}

export interface ApiRequestInit extends Omit<RequestInit, 'body' | 'headers'> {
  body?: BodyInit | object | unknown[] | null
  headers?: HeadersInit
  timeoutMs?: number
  /**
   * Optional transport for this one request. This is deliberately separate
   * from RequestInit: Node/Bun callers can align a long-lived JSON request's
   * dispatcher timeout without changing streaming fetches or every request
   * made by the client.
   */
  fetch?: typeof fetch
  idempotent?: boolean
  retry?: {
    maxAttempts?: number
    backoffMs?: number
  }
}

export class ApiTimeoutError extends Error {
  readonly timeoutMs: number

  constructor(timeoutMs: number) {
    super(`Request timed out after ${timeoutMs}ms`)
    this.name = 'ApiTimeoutError'
    this.timeoutMs = timeoutMs
  }
}

function resolve<T>(value: Resolvable<T> | undefined): Promise<T | undefined> {
  if (typeof value === 'function') {
    return Promise.resolve((value as () => T | Promise<T>)())
  }
  return Promise.resolve(value)
}

function joinUrl(baseUrl: string, path: string): string {
  const normalizedBase = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${normalizedBase}${normalizedPath}`
}

function normalizeBody(body: ApiRequestInit['body'], headers: Headers): BodyInit | undefined {
  if (body == null) return undefined
  if (typeof body === 'string') return body
  if (
    body instanceof ArrayBuffer ||
    body instanceof Blob ||
    body instanceof FormData ||
    body instanceof URLSearchParams
  ) {
    return body
  }
  if (ArrayBuffer.isView(body)) {
    return body as BodyInit
  }
  if (typeof ReadableStream !== 'undefined' && body instanceof ReadableStream) {
    return body
  }

  if (!headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  return JSON.stringify(body)
}

function resolveTimeoutMs(value: number | null | undefined): number | undefined {
  if (value == null) return undefined
  if (!Number.isFinite(value) || value <= 0) return undefined
  return Math.floor(value)
}

function composeSignal(
  signal: AbortSignal | null | undefined,
  timeoutMs: number | undefined,
): { signal?: AbortSignal; cleanup: () => void } {
  if (!signal && timeoutMs == null) {
    return { cleanup: () => {} }
  }

  const controller = new AbortController()
  let timeout: ReturnType<typeof setTimeout> | undefined
  const abortFromParent = () => {
    controller.abort(signal?.reason)
  }
  if (signal) {
    if (signal.aborted) {
      controller.abort(signal.reason)
    } else {
      signal.addEventListener('abort', abortFromParent, { once: true })
    }
  }
  if (timeoutMs != null) {
    timeout = setTimeout(() => {
      controller.abort(new ApiTimeoutError(timeoutMs))
    }, timeoutMs)
    timeout.unref?.()
  }

  return {
    signal: controller.signal,
    cleanup: () => {
      if (timeout) clearTimeout(timeout)
      signal?.removeEventListener('abort', abortFromParent)
    },
  }
}

function isIdempotentRequest(method: string, init: ApiRequestInit, headers: Headers): boolean {
  if (init.idempotent) return true
  if (method === 'GET' || method === 'HEAD') return true
  return headers.has('Idempotency-Key') || headers.has('X-Idempotency-Key')
}

function isRetryableStatus(status: number): boolean {
  return status >= 500 && status < 600
}

function isAbortError(error: unknown): boolean {
  return error != null
    && typeof error === 'object'
    && (error as { name?: unknown }).name === 'AbortError'
}

async function delay(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms))
}

export class ApiHttpClient {
  constructor(private readonly options: ApiHttpClientOptions) {}

  async getHeaders(extra?: HeadersInit): Promise<Headers> {
    const headers = new Headers(await resolve(this.options.defaultHeaders))
    const token = await resolve(this.options.token)
    if (token && !headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${token}`)
    }

    const surface = await resolve(this.options.surface)
    const normalizedSurface = surface?.trim()
    if (normalizedSurface && !headers.has('X-Sepilotd-Surface')) {
      headers.set('X-Sepilotd-Surface', normalizedSurface)
    }

    // Apply memoryScope before the per-call extras so explicit per-call
    // headers (X-Memory-Scope-*) still override.
    const scope = await resolve(this.options.memoryScope)
    for (const [header, value] of Object.entries(buildMemoryScopeHeaders(scope))) {
      if (!headers.has(header)) headers.set(header, value)
    }

    const extraHeaders = new Headers(extra)
    for (const [key, value] of extraHeaders.entries()) {
      headers.set(key, value)
    }
    return headers
  }

  async fetch(path: string, init: ApiRequestInit = {}): Promise<Response> {
    const headers = await this.getHeaders(init.headers)
    return this.fetchWithHeaders(path, init, headers)
  }

  private async fetchWithHeaders(
    path: string,
    init: ApiRequestInit,
    headers: Headers,
  ): Promise<Response> {
    const baseUrl = await resolve(this.options.baseUrl)
    if (!baseUrl) throw new Error('API base URL is not configured')

    const body = normalizeBody(init.body, headers)
    const {
      timeoutMs: _timeoutMs,
      fetch: requestFetch,
      idempotent: _idempotent,
      retry: _retry,
      ...fetchInit
    } = init

    return (requestFetch ?? this.options.fetch ?? fetch)(joinUrl(baseUrl, path), {
      ...fetchInit,
      headers,
      body,
    })
  }

  async request<T = unknown>(path: string, init: ApiRequestInit = {}): Promise<T> {
    const method = (init.method ?? 'GET').toUpperCase()
    const headers = await this.getHeaders(init.headers)
    const idempotent = isIdempotentRequest(method, init, headers)
    const retryMaxAttempts = init.retry?.maxAttempts ?? (idempotent ? 3 : 1)
    const maxAttempts = Math.max(1, Math.floor(retryMaxAttempts))
    const backoffMs = Math.max(0, Math.floor(init.retry?.backoffMs ?? 100))
    const defaultTimeoutMs = resolveTimeoutMs(await resolve(this.options.timeoutMs)) ?? 30_000
    const timeoutMs = resolveTimeoutMs(init.timeoutMs) ?? defaultTimeoutMs

    let lastError: unknown
    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      const composed = composeSignal(init.signal, timeoutMs)
      try {
        const res = await this.fetchWithHeaders(path, {
          ...init,
          headers,
          signal: composed.signal,
        }, headers)
        if (!res.ok && idempotent && isRetryableStatus(res.status) && attempt < maxAttempts) {
          await delay(backoffMs * attempt)
          continue
        }
        return await this.handleResponse<T>(res)
      } catch (error) {
        if (isAbortError(error) && composed.signal?.reason instanceof ApiTimeoutError) {
          lastError = composed.signal.reason
        } else {
          lastError = error
        }
        if (
          init.signal?.aborted ||
          lastError instanceof ApiTimeoutError ||
          !idempotent ||
          attempt >= maxAttempts
        ) {
          throw lastError
        }
        await delay(backoffMs * attempt)
      } finally {
        composed.cleanup()
      }
    }
    throw lastError
  }

  private async handleResponse<T = unknown>(res: Response): Promise<T> {
    if (!res.ok) {
      const text = await res.text()
      const envelope = tryExtractErrorEnvelope(text)
      const headerRequestId = res.headers.get('x-request-id') ?? undefined
      if (envelope?.message || envelope?.code) {
        throw new ApiHttpError({
          status: res.status,
          code: envelope.code,
          message: envelope.message ?? `${res.status} ${envelope.code ?? 'error'}`,
          requestId: envelope.requestId ?? headerRequestId,
          rawBody: text,
          details: envelope.details,
        })
      }
      // No envelope: keep the legacy `<status>: <body>` shape on
      // `Error.message` so callers that string-match (older surfaces)
      // keep working, but expose the structured status/rawBody on the
      // new ApiHttpError fields for everyone else.
      throw new ApiHttpError({
        status: res.status,
        message: `${res.status}: ${text || res.statusText}`,
        requestId: headerRequestId,
        rawBody: text,
      })
    }

    if (res.status === 204) return { ok: true } as T

    const contentType = res.headers.get('content-type') ?? ''
    if (contentType.includes('application/json')) {
      return res.json() as Promise<T>
    }

    return { data: await res.text() } as T
  }

  get<T = unknown>(path: string, init: Omit<ApiRequestInit, 'method'> = {}): Promise<T> {
    return this.request(path, { ...init, method: 'GET' })
  }

  post<T = unknown>(
    path: string,
    body?: ApiRequestInit['body'],
    init: Omit<ApiRequestInit, 'method' | 'body'> = {},
  ): Promise<T> {
    return this.request(path, { ...init, method: 'POST', body })
  }

  put<T = unknown>(
    path: string,
    body?: ApiRequestInit['body'],
    init: Omit<ApiRequestInit, 'method' | 'body'> = {},
  ): Promise<T> {
    return this.request(path, { ...init, method: 'PUT', body })
  }

  delete<T = unknown>(path: string, init: Omit<ApiRequestInit, 'method'> = {}): Promise<T> {
    return this.request(path, { ...init, method: 'DELETE' })
  }
}
