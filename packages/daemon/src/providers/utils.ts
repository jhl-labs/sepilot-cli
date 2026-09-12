import type { ApiError, ErrorCode, Message } from '@sepilotd/core'

export function extractContent(message: Message): string {
  if (typeof message.content === 'string') return message.content
  return message.content
    .filter((p) => p.type === 'text')
    .map((p) => (p as { type: 'text'; text: string }).text)
    .join('\n')
}

export class ProviderError extends Error {
  readonly apiError: ApiError

  constructor(apiError: ApiError) {
    super(apiError.message)
    this.name = 'ProviderError'
    this.apiError = apiError
  }
}

function extractStatusCode(err: unknown): number | undefined {
  if (err && typeof err === 'object') {
    const candidates = [
      (err as { status?: unknown }).status,
      (err as { statusCode?: unknown }).statusCode,
      (err as { response?: { status?: unknown } }).response?.status,
    ]
    for (const candidate of candidates) {
      if (typeof candidate === 'number' && candidate >= 100 && candidate < 600) {
        return candidate
      }
    }
  }
  return undefined
}

function extractString(err: unknown, key: string): string | undefined {
  if (err && typeof err === 'object') {
    const value = (err as Record<string, unknown>)[key]
    if (typeof value === 'string' && value.length > 0) return value
    // OpenAI SDK nests structured info under `error`.
    const nested = (err as { error?: unknown }).error
    if (nested && typeof nested === 'object') {
      const nestedValue = (nested as Record<string, unknown>)[key]
      if (typeof nestedValue === 'string' && nestedValue.length > 0) return nestedValue
    }
  }
  return undefined
}

const CONTEXT_LENGTH_HINTS = [
  'context length',
  'context_length_exceeded',
  'maximum context',
  'prompt is too long',
  'input is too long',
  'too many tokens',
  'exceeds the maximum number of tokens',
  'request too large',
  'reduce the length',
]

const CONTENT_FILTER_HINTS = [
  'content_filter',
  'content filter',
  'content policy',
  'content_policy',
  'safety',
  'responsibleai',
  'jailbreak',
  'blocked by',
]

/**
 * Classify a provider SDK error into a structured `ErrorCode` from the SDK's
 * own status / code / type before that information is lost. Falls back to
 * `PROVIDER_ERROR` when nothing structured is recognisable. Purely additive:
 * the human-readable message is preserved so existing string-based recovery
 * still works, and downstream retry logic can now key off the code.
 */
export function classifyProviderErrorCode(err: unknown): ErrorCode {
  const status = extractStatusCode(err)
  const code = extractString(err, 'code')?.toLowerCase()
  const type = extractString(err, 'type')?.toLowerCase()
  const message = (
    err instanceof Error ? err.message : extractString(err, 'message') ?? String(err)
  ).toLowerCase()

  const mentions = (hints: string[], ...values: Array<string | undefined>): boolean =>
    values.some((value) => value != null && hints.some((hint) => value.includes(hint)))

  if (mentions(CONTENT_FILTER_HINTS, code, type, message)) return 'CONTENT_FILTER'
  if (code === 'context_length_exceeded' || mentions(CONTEXT_LENGTH_HINTS, code, message)) {
    return 'CONTEXT_LENGTH'
  }
  if (code === 'rate_limit_exceeded' || code === 'insufficient_quota') return 'RATE_LIMITED'
  if (code === 'invalid_api_key' || type === 'authentication_error') return 'UNAUTHORIZED'

  if (status != null) {
    if (status === 401) return 'UNAUTHORIZED'
    if (status === 403) return 'FORBIDDEN'
    if (status === 404) return 'NOT_FOUND'
    if (status === 408 || status === 504) return 'TIMEOUT'
    if (status === 409) return 'CONFLICT'
    if (status === 429) return 'RATE_LIMITED'
    if (status === 400 && mentions(CONTEXT_LENGTH_HINTS, message)) return 'CONTEXT_LENGTH'
    if (status === 400 || status === 422) return 'INVALID_REQUEST'
    if (status === 500 || status === 502 || status === 503 || status === 529) {
      return 'SERVICE_UNAVAILABLE'
    }
  }

  return 'PROVIDER_ERROR'
}

export function toApiError(err: unknown): ApiError {
  const message = err instanceof Error ? err.message : String(err)
  const code = classifyProviderErrorCode(err)
  const status = extractStatusCode(err)
  return status != null ? { code, message, details: { status } } : { code, message }
}

export function wrapError(err: unknown): ProviderError {
  return new ProviderError(toApiError(err))
}
