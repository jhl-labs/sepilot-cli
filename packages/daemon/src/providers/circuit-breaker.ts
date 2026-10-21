import type {
  ApiError,
  ChatRequest,
  ChatResponse,
  ILLMProvider,
  StreamChunk,
} from '@sepilotd/core'
import {
  abortableDelay,
  createAbortError,
  getAbortError,
  isAbortError,
} from '../abort.js'
import { tokenCalibration } from './token-calibration.js'
import { extractContent, isRetryableProviderHttpStatus } from './utils.js'
import { resolveRequestThinking } from './thinking-policy.js'

export interface ProviderCircuitBreakerOptions {
  failureThreshold?: number
  openDurationMs?: number
}

export interface ProviderCircuitSummary {
  trackedCircuits: number
  openCircuits: number
  halfOpenCircuits: number
}

interface CircuitState {
  status: 'closed' | 'open' | 'half_open'
  consecutiveFailures: number
  openedUntil?: number
  lastFailure?: string
  lastFailureAt?: string
  lastSuccessAt?: string
  probeInFlight: boolean
}

interface CircuitRequestToken {
  key: string
  providerId: string
  model: string
  phase: 'closed' | 'half_open'
}

export interface GuardedProviderChatOptions {
  provider: ILLMProvider
  request: ChatRequest
  signal?: AbortSignal
  breaker?: ProviderCircuitBreaker
  maxRetries?: number
  cache?: {
    get(request: ChatRequest): ChatResponse | null
    set(request: ChatRequest, response: ChatResponse): void
  }
}

export interface GuardedProviderStreamOptions {
  provider: ILLMProvider
  request: ChatRequest
  signal?: AbortSignal
  breaker?: ProviderCircuitBreaker
  maxRetries?: number
  allowEmptyStreamFallback?: boolean
}

export const PROVIDER_STREAM_HEARTBEAT_MS = 30_000
export const DEFAULT_PROVIDER_STREAM_IDLE_MS = 15 * 60_000
export const DEFAULT_PROVIDER_STREAM_FIRST_TOKEN_MS = 90_000
export const DEFAULT_BUFFERED_PROVIDER_FIRST_TOKEN_MS = 240_000

const bufferedStreamProfiles = new Map<string, boolean>()

function streamProfileKey(providerId: string, model: string): string {
  return `${providerId}|${model}`
}

export function classifyBufferedStream(input: {
  startedAt: number
  firstContentAt: number
  lastContentAt: number
  contentChunks: number
}): boolean | null {
  const silenceMs = input.firstContentAt - input.startedAt
  const spanMs = input.lastContentAt - input.firstContentAt
  if (input.contentChunks < 3) return null
  if (silenceMs < 3_000) return false
  return spanMs <= silenceMs * 0.1
}

export function recordStreamProfile(providerId: string, model: string, buffered: boolean | null): void {
  if (buffered === null) return
  bufferedStreamProfiles.set(streamProfileKey(providerId, model), buffered)
}

export function isBufferedStreamProfile(providerId: string, model: string): boolean {
  return bufferedStreamProfiles.get(streamProfileKey(providerId, model)) === true
}

export function resolveBufferedProviderFirstTokenMs(env: NodeJS.ProcessEnv = process.env): number {
  const parsed = Number.parseInt(env.SEPILOTD_PROVIDER_BUFFERED_FIRST_TOKEN_MS ?? '', 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_BUFFERED_PROVIDER_FIRST_TOKEN_MS
}
export const DEFAULT_PROVIDER_STREAM_ACTIVE_IDLE_MS = 60_000
export const DEFAULT_PROVIDER_STREAM_MAX_RETRIES = 1
export const DEFAULT_PROVIDER_CHAT_MAX_RETRIES = 1

type ProviderStreamWaitResult =
  | { kind: 'next'; result: IteratorResult<StreamChunk> }
  | { kind: 'error'; error: unknown }
  | { kind: 'heartbeat' }
  | { kind: 'timeout' }
  | { kind: 'abort' }

function waitForProviderStreamNext(
  nextPromise: Promise<IteratorResult<StreamChunk>>,
  signal?: AbortSignal,
  timeoutMs?: number | null,
): Promise<ProviderStreamWaitResult> {
  let heartbeatTimer: ReturnType<typeof setTimeout> | null = null
  let timeoutTimer: ReturnType<typeof setTimeout> | null = null
  let abortHandler: (() => void) | null = null

  return new Promise<ProviderStreamWaitResult>((resolve) => {
    const cleanup = () => {
      if (heartbeatTimer) {
        clearTimeout(heartbeatTimer)
        heartbeatTimer = null
      }
      if (timeoutTimer) {
        clearTimeout(timeoutTimer)
        timeoutTimer = null
      }
      if (signal && abortHandler) {
        signal.removeEventListener('abort', abortHandler)
        abortHandler = null
      }
    }

    const finish = (result: ProviderStreamWaitResult) => {
      cleanup()
      resolve(result)
    }

    nextPromise.then(
      (result) => finish({ kind: 'next', result }),
      (error) => finish({ kind: 'error', error }),
    )

    if (typeof timeoutMs === 'number') {
      if (timeoutMs <= 0) {
        finish({ kind: 'timeout' })
        return
      }
      timeoutTimer = setTimeout(() => {
        finish({ kind: 'timeout' })
      }, timeoutMs)
      timeoutTimer.unref?.()
    }

    if (timeoutMs === null || timeoutMs === undefined || timeoutMs > PROVIDER_STREAM_HEARTBEAT_MS) {
      heartbeatTimer = setTimeout(() => {
        finish({ kind: 'heartbeat' })
      }, PROVIDER_STREAM_HEARTBEAT_MS)
      heartbeatTimer.unref?.()
    }

    if (signal) {
      if (signal.aborted) {
        finish({ kind: 'abort' })
        return
      }
      abortHandler = () => finish({ kind: 'abort' })
      signal.addEventListener('abort', abortHandler, { once: true })
    }
  })
}

interface RequestAbortScope {
  signal?: AbortSignal
  timedOut(): boolean
  cleanup(): void
}

function createRequestAbortScope(
  parent: AbortSignal | undefined,
  timeoutMs: number | undefined,
): RequestAbortScope {
  if (!Number.isFinite(timeoutMs) || !timeoutMs || timeoutMs < 1) {
    return { signal: parent, timedOut: () => false, cleanup: () => {} }
  }

  const controller = new AbortController()
  let timeoutFired = false
  const abortFromParent = () => controller.abort(getAbortError(parent, 'Provider request aborted'))
  if (parent?.aborted) abortFromParent()
  else parent?.addEventListener('abort', abortFromParent, { once: true })

  const timer = setTimeout(() => {
    timeoutFired = true
    controller.abort(createAbortError(`Provider request timed out after ${timeoutMs}ms`))
  }, timeoutMs)
  timer.unref?.()

  return {
    signal: controller.signal,
    timedOut: () => timeoutFired,
    cleanup: () => {
      clearTimeout(timer)
      parent?.removeEventListener('abort', abortFromParent)
    },
  }
}

function getProviderStreamIdleMs(): number | null {
  const raw = process.env.SEPILOTD_PROVIDER_STREAM_IDLE_MS
    ?? process.env.SEPILOTD_STREAM_IDLE_MS
    ?? String(DEFAULT_PROVIDER_STREAM_IDLE_MS)
  const parsed = Number.parseInt(raw, 10)
  if (parsed === 0) {
    return null
  }
  if (!Number.isFinite(parsed) || parsed < 0) {
    return DEFAULT_PROVIDER_STREAM_IDLE_MS
  }
  return parsed
}

function getProviderStreamActiveIdleMs(globalIdleMs: number | null): number | null {
  if (globalIdleMs === null) return null
  const raw = process.env.SEPILOTD_PROVIDER_STREAM_ACTIVE_IDLE_MS
  if (raw === undefined) {
    if (
      process.env.SEPILOTD_PROVIDER_STREAM_IDLE_MS !== undefined
      || process.env.SEPILOTD_STREAM_IDLE_MS !== undefined
    ) {
      return globalIdleMs
    }
    return Math.min(globalIdleMs, DEFAULT_PROVIDER_STREAM_ACTIVE_IDLE_MS)
  }
  const parsed = Number.parseInt(raw, 10)
  if (parsed === 0) return null
  if (!Number.isFinite(parsed) || parsed < 0) {
    return Math.min(globalIdleMs, DEFAULT_PROVIDER_STREAM_ACTIVE_IDLE_MS)
  }
  return parsed
}

export function resolveProviderStreamFirstTokenMs(globalIdleMs = getProviderStreamIdleMs()): number | null {
  if (globalIdleMs === null) return null
  const raw = process.env.SEPILOTD_PROVIDER_STREAM_FIRST_TOKEN_MS
  if (raw === undefined) {
    if (
      process.env.SEPILOTD_PROVIDER_STREAM_IDLE_MS !== undefined
      || process.env.SEPILOTD_STREAM_IDLE_MS !== undefined
    ) {
      return globalIdleMs
    }
    return Math.min(globalIdleMs, DEFAULT_PROVIDER_STREAM_FIRST_TOKEN_MS)
  }
  const parsed = Number.parseInt(raw, 10)
  if (parsed === 0) return null
  if (!Number.isFinite(parsed) || parsed < 0) {
    return Math.min(globalIdleMs, DEFAULT_PROVIDER_STREAM_FIRST_TOKEN_MS)
  }
  return parsed
}

function getProviderStreamMaxRetries(): number {
  const raw = process.env.SEPILOTD_PROVIDER_STREAM_MAX_RETRIES
  if (raw === undefined) {
    return DEFAULT_PROVIDER_STREAM_MAX_RETRIES
  }

  const parsed = Number.parseInt(raw, 10)
  if (!Number.isFinite(parsed) || parsed < 0) {
    return DEFAULT_PROVIDER_STREAM_MAX_RETRIES
  }
  return parsed
}

function getProviderChatMaxRetries(): number {
  const raw = process.env.SEPILOTD_PROVIDER_CHAT_MAX_RETRIES
  if (raw === undefined || raw.trim() === '') {
    return DEFAULT_PROVIDER_CHAT_MAX_RETRIES
  }
  const parsed = Number(raw)
  if (Number.isFinite(parsed) && parsed >= 0) return Math.min(5, Math.floor(parsed))
  return DEFAULT_PROVIDER_CHAT_MAX_RETRIES
}

function providerRetryDelayMs(attempt: number): number {
  return Math.pow(2, attempt) * 1_000
}

export class ProviderUnavailableError extends Error {
  readonly code = 'SERVICE_UNAVAILABLE'

  constructor(
    readonly providerId: string,
    readonly model: string,
    readonly retryAfterMs: number,
  ) {
    super(
      `Provider ${providerId}/${model} is temporarily unavailable; circuit breaker is open for ${retryAfterMs}ms`,
    )
    this.name = 'ProviderUnavailableError'
  }
}

export class ProviderExecutionError extends Error {
  readonly code = 'PROVIDER_ERROR'

  constructor(
    readonly providerId: string,
    readonly model: string,
    message: string,
    readonly providerCode?: string,
    readonly providerStatus?: number,
  ) {
    super(message)
    this.name = 'ProviderExecutionError'
  }
}

export function isProviderUnavailableError(
  error: unknown,
): error is ProviderUnavailableError {
  return error instanceof ProviderUnavailableError
}

export function isProviderExecutionError(
  error: unknown,
): error is ProviderExecutionError {
  return error instanceof ProviderExecutionError
}

export function toProviderApiError(
  error: unknown,
): ApiError {
  if (isProviderUnavailableError(error)) {
    return {
      code: error.code,
      message: error.message,
      details: {
        source: 'provider',
        providerId: error.providerId,
        model: error.model,
      },
    }
  }

  if (isProviderExecutionError(error)) {
    const classification = error.providerCode ? ` [${error.providerCode}]` : ''
    return {
      code: error.code,
      message: `Provider ${error.providerId}/${error.model}${classification}: ${error.message}`,
      details: {
        source: 'provider',
        providerId: error.providerId,
        model: error.model,
        ...(error.providerCode ? { providerCode: error.providerCode } : {}),
        ...(error.providerStatus ? { status: error.providerStatus } : {}),
      },
    }
  }

  if (error instanceof Error) {
    return {
      code: 'INTERNAL_ERROR',
      message: error.message,
    }
  }

  return {
    code: 'INTERNAL_ERROR',
    message: normalizeProviderErrorMessage(error),
  }
}

function normalizeProviderErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }
  if (
    error
    && typeof error === 'object'
    && 'message' in error
    && typeof error.message === 'string'
  ) {
    return error.message
  }
  if (typeof error === 'string') return error
  if (error !== null && typeof error === 'object') {
    return 'Unknown provider error'
  }
  return String(error)
}

const CONTEXT_LENGTH_PATTERNS = [
  'context length',
  'context_length_exceeded',
  'maximum context',
  'prompt is too long',
  'input is too long',
  'too many tokens',
  'exceeds the maximum number of tokens',
  'request too large',
]

export function isContextLengthProviderError(error: unknown): boolean {
  const message = normalizeProviderErrorMessage(error).toLowerCase()
  return CONTEXT_LENGTH_PATTERNS.some((pattern) => message.includes(pattern))
}

const RETRYABLE_NETWORK_PATTERNS = [
  'econnreset',
  'econnrefused',
  'etimedout',
  'epipe',
  'eai_again',
  'socket hang up',
  'fetch failed',
  'und_err',
  'connection error',
  'other side closed',
]

function hasRetryableNetworkCauseCode(error: unknown): boolean {
  let current: unknown = error
  for (let depth = 0; depth < 4 && current && typeof current === 'object'; depth += 1) {
    const code = (current as { code?: unknown }).code
    if (typeof code === 'string') {
      const normalized = code.toLowerCase()
      if (RETRYABLE_NETWORK_PATTERNS.some((pattern) => normalized.startsWith(pattern))) return true
    }
    current = (current as { cause?: unknown }).cause
  }
  return false
}

export function extractProviderErrorStatusCode(error: unknown): number | undefined {
  if (error && typeof error === 'object') {
    const candidates = [
      (error as { status?: unknown }).status,
      (error as { statusCode?: unknown }).statusCode,
      (error as { response?: { status?: unknown } }).response?.status,
      (error as { apiError?: { details?: { status?: unknown } } }).apiError?.details?.status,
      ...(error instanceof ProviderExecutionError ? [error.providerStatus] : []),
    ]
    for (const candidate of candidates) {
      if (typeof candidate === 'number' && Number.isInteger(candidate) && candidate >= 100 && candidate < 600) {
        return candidate
      }
    }
  }
  const match = normalizeProviderErrorMessage(error).match(/\b([45]\d\d)\b/)
  return match ? Number(match[1]) : undefined
}

function structuredApiErrorCode(error: unknown): string | undefined {
  if (error instanceof ProviderExecutionError) return error.providerCode
  if (error && typeof error === 'object') {
    const apiError = (error as { apiError?: { code?: unknown } }).apiError
    if (apiError && typeof apiError.code === 'string') return apiError.code
  }
  return undefined
}

function isRetryableProviderError(error: unknown): boolean {
  if (error instanceof ProviderExecutionError && error.providerCode === 'STREAM_INCOMPLETE') return true
  const structured = structuredApiErrorCode(error)
  if (structured) {
    if (structured === 'RATE_LIMITED' || structured === 'SERVICE_UNAVAILABLE' || structured === 'TIMEOUT') {
      return true
    }
    if (
      structured === 'CONTEXT_LENGTH' ||
      structured === 'CONTENT_FILTER' ||
      structured === 'UNAUTHORIZED' ||
      structured === 'FORBIDDEN' ||
      structured === 'NOT_FOUND' ||
      structured === 'INVALID_REQUEST'
    ) {
      return false
    }
  }
  if (isContextLengthProviderError(error)) return false
  const statusCode = extractProviderErrorStatusCode(error)
  if (statusCode != null) return isRetryableProviderHttpStatus(statusCode)
  if (hasRetryableNetworkCauseCode(error)) return true
  const message = normalizeProviderErrorMessage(error).toLowerCase()
  return RETRYABLE_NETWORK_PATTERNS.some((pattern) => message.includes(pattern))
    || message.includes('rate')
    || message.includes('timeout')
}

function wrapProviderExecutionError(
  providerId: string,
  model: string,
  error: unknown,
): ProviderExecutionError {
  if (error instanceof ProviderExecutionError) {
    return error
  }
  return new ProviderExecutionError(
    providerId,
    model,
    normalizeProviderErrorMessage(error),
    structuredApiErrorCode(error),
    extractProviderErrorStatusCode(error),
  )
}

export class ProviderCircuitBreaker {
  private readonly failureThreshold: number
  private readonly openDurationMs: number
  private readonly circuits = new Map<string, CircuitState>()

  constructor(options: ProviderCircuitBreakerOptions = {}) {
    this.failureThreshold = options.failureThreshold ?? 5
    this.openDurationMs = options.openDurationMs ?? 30_000
  }

  beginRequest(
    providerId: string,
    model: string,
  ): CircuitRequestToken {
    const state = this.getOrCreateState(providerId, model)
    const now = Date.now()
    this.refreshExpiredOpenState(state, now)

    if (
      state.status === 'open'
      && typeof state.openedUntil === 'number'
      && state.openedUntil > now
    ) {
      throw new ProviderUnavailableError(
        providerId,
        model,
        state.openedUntil - now,
      )
    }

    if (state.status === 'open') {
      state.status = 'half_open'
      state.openedUntil = undefined
    }

    if (state.status === 'half_open') {
      if (state.probeInFlight) {
        throw new ProviderUnavailableError(
          providerId,
          model,
          this.openDurationMs,
        )
      }
      state.probeInFlight = true
      return { key: this.key(providerId, model), providerId, model, phase: 'half_open' }
    }

    return { key: this.key(providerId, model), providerId, model, phase: 'closed' }
  }

  cancel(token: CircuitRequestToken): void {
    if (token.phase !== 'half_open') {
      return
    }

    const state = this.circuits.get(token.key)
    if (!state) {
      return
    }

    state.probeInFlight = false
  }

  recordSuccess(token: CircuitRequestToken): void {
    const state = this.getOrCreateState(token.providerId, token.model)
    state.status = 'closed'
    state.consecutiveFailures = 0
    state.openedUntil = undefined
    state.lastSuccessAt = new Date().toISOString()
    state.probeInFlight = false
  }

  recordFailure(
    token: CircuitRequestToken,
    error: unknown,
  ): void {
    const state = this.getOrCreateState(token.providerId, token.model)
    state.lastFailure = normalizeProviderErrorMessage(error)
    state.lastFailureAt = new Date().toISOString()
    state.probeInFlight = false

    if (token.phase === 'half_open') {
      state.status = 'open'
      state.openedUntil = Date.now() + this.openDurationMs
      state.consecutiveFailures = this.failureThreshold
      return
    }

    state.consecutiveFailures += 1
    if (state.consecutiveFailures >= this.failureThreshold) {
      state.status = 'open'
      state.openedUntil = Date.now() + this.openDurationMs
      return
    }

    state.status = 'closed'
  }

  noteExternalFailure(
    providerId: string,
    model: string,
    reason: string,
  ): void {
    const state = this.getOrCreateState(providerId, model)
    state.lastFailure = reason
    state.lastFailureAt = new Date().toISOString()
    if (state.status === 'open') return
    if (state.status === 'half_open') {
      state.status = 'open'
      state.openedUntil = Date.now() + this.openDurationMs
      state.consecutiveFailures = this.failureThreshold
      state.probeInFlight = false
      return
    }
    state.consecutiveFailures += 1
    if (state.consecutiveFailures >= this.failureThreshold) {
      state.status = 'open'
      state.openedUntil = Date.now() + this.openDurationMs
    }
  }

  getSummary(): ProviderCircuitSummary {
    let openCircuits = 0
    let halfOpenCircuits = 0
    const now = Date.now()

    for (const state of this.circuits.values()) {
      this.refreshExpiredOpenState(state, now)
      if (state.status === 'open') {
        openCircuits += 1
      } else if (state.status === 'half_open') {
        halfOpenCircuits += 1
      }
    }

    return {
      trackedCircuits: this.circuits.size,
      openCircuits,
      halfOpenCircuits,
    }
  }

  private refreshExpiredOpenState(state: CircuitState, now: number): void {
    if (
      state.status === 'open'
      && (typeof state.openedUntil !== 'number' || state.openedUntil <= now)
    ) {
      state.status = 'half_open'
      state.openedUntil = undefined
      state.probeInFlight = false
    }
  }

  private getOrCreateState(providerId: string, model: string): CircuitState {
    const key = this.key(providerId, model)
    let state = this.circuits.get(key)
    if (!state) {
      state = {
        status: 'closed',
        consecutiveFailures: 0,
        probeInFlight: false,
      }
      this.circuits.set(key, state)
    }
    return state
  }

  private key(providerId: string, model: string): string {
    return `${providerId}:${model}`
  }
}

export async function guardedProviderChat(
  options: GuardedProviderChatOptions,
): Promise<ChatResponse> {
  options = { ...options, request: resolveRequestThinking(options.request) }
  const {
    provider,
    request,
    signal,
    breaker,
    maxRetries = getProviderChatMaxRetries(),
    cache,
  } = options

  const cached = cache?.get(request) ?? null
  if (cached) {
    return cached
  }

  const requestScope = createRequestAbortScope(signal, request.timeoutMs)
  try {
    for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
      const token = breaker?.beginRequest(provider.id, request.model)
      try {
        const response = await provider.chat(request, { signal: requestScope.signal })
        if (token) {
          breaker?.recordSuccess(token)
        }
        if (response.usage?.inputTokens) {
          tokenCalibration.observeRequest(provider.id, request.model, request, response.usage.inputTokens)
        }
        cache?.set(request, response)
        return response
      } catch (error) {
        if (requestScope.timedOut()) {
          if (token) breaker?.recordFailure(token, error)
          throw new ProviderExecutionError(
            provider.id,
            request.model,
            `provider-request-timeout after ${request.timeoutMs}ms`,
            'REQUEST_TIMEOUT',
          )
        }
        if ((signal?.aborted ?? false) || isAbortError(error)) {
          if (token) breaker?.cancel(token)
          throw getAbortError(signal, 'Provider request aborted')
        }

        if (token?.phase === 'half_open') {
          breaker?.recordFailure(token, error)
          throw wrapProviderExecutionError(provider.id, request.model, error)
        }

        const retryable = isRetryableProviderError(error)
        if (retryable && attempt < maxRetries) {
          await abortableDelay(providerRetryDelayMs(attempt), requestScope.signal)
          continue
        }

        if (token) breaker?.recordFailure(token, error)
        throw wrapProviderExecutionError(provider.id, request.model, error)
      }
    }

    throw new ProviderExecutionError(provider.id, request.model, 'Max retries exceeded')
  } finally {
    requestScope.cleanup()
  }
}

export async function* guardedProviderStream(
  options: GuardedProviderStreamOptions,
): AsyncGenerator<StreamChunk, void, void> {
  options = { ...options, request: resolveRequestThinking(options.request) }
  const {
    provider,
    request,
    signal,
    breaker,
    maxRetries = getProviderStreamMaxRetries(),
  } = options

  const model = provider.models.find((candidate) => candidate.id === request.model)
  if (model?.capabilities.streaming === false) {
    const fallbackTimeoutMs = resolveProviderStreamFirstTokenMs()
    const adaptedRequest = request.timeoutMs === undefined && fallbackTimeoutMs !== null
      ? { ...request, timeoutMs: fallbackTimeoutMs }
      : request
    const response = await guardedProviderChat({
      provider,
      request: adaptedRequest,
      signal,
      breaker,
      maxRetries,
    })
    if (response.thinking) {
      yield { type: 'thinking', text: response.thinking }
    }
    const content = extractContent(response.message)
    if (content) {
      yield { type: 'text', text: content }
    }
    for (const toolCall of response.message.toolCalls ?? []) {
      yield {
        type: 'tool_call_start',
        toolCall: { id: toolCall.id, name: toolCall.name },
      }
      yield {
        type: 'tool_call_delta',
        toolCallId: toolCall.id,
        delta: JSON.stringify(toolCall.arguments),
      }
      yield { type: 'tool_call_end', toolCallId: toolCall.id }
    }
    yield { type: 'usage', usage: response.usage }
    yield { type: 'done', finishReason: response.finishReason }
    return
  }

  const idleTimeoutMs = getProviderStreamIdleMs()
  const baseFirstTokenTimeoutMs = resolveProviderStreamFirstTokenMs(idleTimeoutMs)
  const firstTokenTimeoutMs = baseFirstTokenTimeoutMs !== null
    && process.env.SEPILOTD_PROVIDER_STREAM_FIRST_TOKEN_MS === undefined
    && isBufferedStreamProfile(provider.id, request.model)
    ? Math.max(baseFirstTokenTimeoutMs, resolveBufferedProviderFirstTokenMs())
    : baseFirstTokenTimeoutMs
  const activeIdleTimeoutMs = getProviderStreamActiveIdleMs(idleTimeoutMs)
  const requestScope = createRequestAbortScope(signal, request.timeoutMs)

  try {
    for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    const token = breaker?.beginRequest(provider.id, request.model)
    let failureRecorded = false
    let emittedProviderChunk = false
    let emittedActionableChunk = false
    const attemptStartedAt = Date.now()
    let generatedAnything = false
    let firstContentAt: number | null = null
    let lastContentAt = 0
    let contentChunks = 0
    let retryDelayMs = 0
    const iterator = provider.stream(request, { signal: requestScope.signal })[Symbol.asyncIterator]()

    try {
      let done = false
      while (!done) {
        const nextPromise = iterator.next()
        const waitStartedAt = Date.now()
        let nextChunk: StreamChunk | null = null
        let waitingNoticeEmitted = false

        while (!done && !nextChunk) {
          const currentIdleTimeoutMs = emittedProviderChunk
            ? activeIdleTimeoutMs
            : firstTokenTimeoutMs
          const remainingIdleMs = currentIdleTimeoutMs === null
            ? null
            : currentIdleTimeoutMs - (Date.now() - waitStartedAt)
          const waitResult = await waitForProviderStreamNext(
            nextPromise,
            requestScope.signal,
            remainingIdleMs,
          )
          switch (waitResult.kind) {
            case 'heartbeat':
              if (!waitingNoticeEmitted) {
                waitingNoticeEmitted = true
                yield {
                  type: 'thinking',
                  text: 'Still waiting for the model stream...',
                }
              }
              continue
            case 'timeout':
              throw new ProviderExecutionError(
                provider.id,
                request.model,
                emittedProviderChunk
                  ? `provider-stream-idle-timeout after ${currentIdleTimeoutMs}ms`
                  : `provider-stream-first-token-timeout after ${currentIdleTimeoutMs}ms`,
                emittedProviderChunk ? 'STREAM_IDLE_TIMEOUT' : 'FIRST_TOKEN_TIMEOUT',
              )
            case 'abort':
              throw getAbortError(signal, 'Provider request aborted')
            case 'error':
              throw waitResult.error
            case 'next':
              if (waitResult.result.done) {
                if (!emittedProviderChunk && options.allowEmptyStreamFallback) {
                  if (token) breaker?.cancel(token)
                  return
                }
                throw new ProviderExecutionError(
                  provider.id,
                  request.model,
                  'Provider stream ended before its terminal completion event',
                  'STREAM_INCOMPLETE',
                )
              } else {
                nextChunk = waitResult.result.value
              }
              break
          }
        }

        if (!nextChunk) {
          continue
        }

        const chunk = nextChunk
        if (chunk.type === 'usage' && chunk.usage.inputTokens) {
          tokenCalibration.observeRequest(provider.id, request.model, request, chunk.usage.inputTokens)
        }
        if (chunk.type === 'error') {
          throw new ProviderExecutionError(
            provider.id,
            request.model,
            chunk.error.message,
            chunk.error.code,
            extractProviderErrorStatusCode({ apiError: chunk.error }),
          )
        }

        emittedProviderChunk = true
        if (
          chunk.type === 'text'
          || chunk.type === 'tool_call_start'
          || chunk.type === 'tool_call_delta'
          || chunk.type === 'tool_call_end'
        ) {
          emittedActionableChunk = true
        }
        if (chunk.type === 'text' || chunk.type === 'thinking' || chunk.type === 'tool_call_start') {
          generatedAnything = true
        }
        if (chunk.type === 'text' || chunk.type === 'thinking' || chunk.type === 'tool_call_start' || chunk.type === 'tool_call_delta') {
          const now = Date.now()
          firstContentAt ??= now
          lastContentAt = now
          contentChunks += 1
        }
        yield chunk.type === 'usage' && generatedAnything && !chunk.usage.generationMs
          ? { ...chunk, usage: { ...chunk.usage, generationMs: Math.max(1, Date.now() - attemptStartedAt) } }
          : chunk
        if (chunk.type === 'done') {
          done = true
          if (firstContentAt !== null) {
            recordStreamProfile(provider.id, request.model, classifyBufferedStream({
              startedAt: attemptStartedAt, firstContentAt, lastContentAt, contentChunks,
            }))
          }
        }
      }

      if (token && !failureRecorded) {
        breaker?.recordSuccess(token)
      }
      return
    } catch (error) {
      if (requestScope.timedOut()) {
        if (token && !failureRecorded) breaker?.recordFailure(token, error)
        throw new ProviderExecutionError(
          provider.id,
          request.model,
          `provider-request-timeout after ${request.timeoutMs}ms`,
          'REQUEST_TIMEOUT',
        )
      }
      if ((signal?.aborted ?? false) || isAbortError(error)) {
        if (token) {
          breaker?.cancel(token)
        }
        throw getAbortError(signal, 'Provider request aborted')
      }

      const retryable = isRetryableProviderError(error)
      const firstTokenTimedOut = error instanceof ProviderExecutionError
        && error.providerCode === 'FIRST_TOKEN_TIMEOUT'
      if (
        token?.phase !== 'half_open'
        && !emittedActionableChunk
        && !firstTokenTimedOut
        && retryable
        && attempt < maxRetries
      ) {
        if (token) {
          breaker?.cancel(token)
        }
        retryDelayMs = providerRetryDelayMs(attempt)
      } else {
        if (token && !failureRecorded) {
          breaker?.recordFailure(token, error)
          failureRecorded = true
        }
        throw wrapProviderExecutionError(provider.id, request.model, error)
      }
    } finally {
      void iterator.return?.()
    }
    try {
      await abortableDelay(retryDelayMs, requestScope.signal)
    } catch (error) {
      if (requestScope.timedOut()) {
        if (token) breaker?.recordFailure(token, error)
        throw new ProviderExecutionError(
          provider.id,
          request.model,
          `provider-request-timeout after ${request.timeoutMs}ms`,
          'REQUEST_TIMEOUT',
        )
      }
      if (token) breaker?.cancel(token)
      throw getAbortError(signal, 'Provider request aborted')
    }
    }

    throw new ProviderExecutionError(provider.id, request.model, 'Max stream retries exceeded')
  } finally {
    requestScope.cleanup()
  }
}

export const __testables = {
  getProviderChatMaxRetries,
  getProviderStreamIdleMs,
  resolveProviderStreamFirstTokenMs,
  getProviderStreamMaxRetries,
  isRetryableProviderError,
  normalizeProviderErrorMessage,
  waitForProviderStreamNext,
  wrapProviderExecutionError,
}
