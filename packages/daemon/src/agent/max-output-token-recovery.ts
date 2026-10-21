import type { AgentEvent } from '@sepilotd/core'

export function resolveModelOutputAllowance(options: {
  requested?: number
  modelMaximum?: number
  observedMaximum?: number
  fallback: number
}): number {
  const positive = (value: number | undefined): number | undefined =>
    Number.isSafeInteger(value) && value! > 0 ? value : undefined
  const model = positive(options.modelMaximum)
  const observed = positive(options.observedMaximum)
  return Math.min(
    positive(options.requested) ?? model ?? observed ?? options.fallback,
    model ?? Infinity,
    observed ?? Infinity,
  )
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  if (typeof error === 'string') return error
  if (!error || typeof error !== 'object') return String(error ?? '')
  const record = error as Record<string, unknown>
  for (const value of [record.message, record.error, record.apiError, record.cause]) {
    const nested = errorMessage(value)
    if (nested) return nested
  }
  return ''
}

export function detectProviderMaxOutputTokenLimit(
  error: unknown,
  requestedTokens: number | undefined,
): number | null {
  if (!requestedTokens || !Number.isFinite(requestedTokens)) return null
  const message = errorMessage(error).split(/\s+\(ref:/i, 1)[0] ?? ''
  if (
    !/(?:max[_ -]?tokens|max(?:imum)?\s+output\s+tokens?)/i.test(message)
    || !/(?:exceed|maximum|limit|at most|less than|too (?:large|high))/i.test(message)
  ) {
    return null
  }

  const explicit = message.match(/maximum\s+output\s+tokens?[^\d]{0,24}(\d[\d,_]*)/i)?.[1]
  const numericCandidates = (explicit ? [explicit] : [...message.matchAll(/\b\d[\d,_]*\b/g)].map((match) => match[0]))
    .map((value) => Number.parseInt(value.replace(/[,_]/g, ''), 10))
    .filter((value) => Number.isSafeInteger(value) && value >= 64 && value < requestedTokens)

  return numericCandidates.length > 0 ? Math.max(...numericCandidates) : null
}

export function buildMaxOutputTokenRecoveryEvent(
  requestedTokens: number,
  observedLimit: number,
): Extract<AgentEvent, { type: 'recovery' }> {
  return {
    type: 'recovery',
    scope: 'provider_protocol',
    kind: 'max_output_tokens',
    action: 'retry_with_observed_output_limit',
    message: `Provider rejected max_tokens=${requestedTokens}; retrying once with its observed limit ${observedLimit}.`,
    recoverable: true,
    details: {
      requestedTokens,
      observedLimit,
      graphMutation: false,
    },
  }
}

const OBSERVED_MAX_OUTPUT_TOKENS_TTL_MS = 60 * 60_000
const MAX_OBSERVED_MAX_OUTPUT_TOKEN_ENTRIES = 256
const observedMaxOutputTokens = new Map<string, { limit: number; at: number }>()

function observedLimitKey(providerId: string | undefined, model: string | undefined): string | null {
  return providerId && model ? `${providerId}\u0000${model}` : null
}

export function rememberObservedMaxOutputTokens(
  providerId: string | undefined,
  model: string | undefined,
  limit: number,
  now = Date.now(),
): void {
  const key = observedLimitKey(providerId, model)
  if (!key || !Number.isSafeInteger(limit) || limit <= 0) return
  for (const [existingKey, entry] of observedMaxOutputTokens) {
    if (now - entry.at > OBSERVED_MAX_OUTPUT_TOKENS_TTL_MS) observedMaxOutputTokens.delete(existingKey)
  }
  const current = observedMaxOutputTokens.get(key)
  if (current && current.limit <= limit && now - current.at <= OBSERVED_MAX_OUTPUT_TOKENS_TTL_MS) return
  while (observedMaxOutputTokens.size >= MAX_OBSERVED_MAX_OUTPUT_TOKEN_ENTRIES) {
    const oldest = observedMaxOutputTokens.keys().next().value
    if (oldest === undefined) break
    observedMaxOutputTokens.delete(oldest)
  }
  observedMaxOutputTokens.set(key, { limit, at: now })
}

export function recallObservedMaxOutputTokens(
  providerId: string | undefined,
  model: string | undefined,
  now = Date.now(),
): number | undefined {
  const key = observedLimitKey(providerId, model)
  if (!key) return undefined
  const entry = observedMaxOutputTokens.get(key)
  if (!entry) return undefined
  if (now - entry.at > OBSERVED_MAX_OUTPUT_TOKENS_TTL_MS) {
    observedMaxOutputTokens.delete(key)
    return undefined
  }
  return entry.limit
}

export function resetObservedMaxOutputTokens(): void {
  observedMaxOutputTokens.clear()
}
