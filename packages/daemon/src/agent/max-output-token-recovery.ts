import type { AgentEvent } from '@sepilotd/core'

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

/**
 * Recover a provider's real output-token ceiling from a rejected request.
 *
 * Model catalogs and OpenAI-compatible relays are frequently out of sync.
 * Only accept a smaller positive integer from an error that explicitly names
 * max_tokens or a maximum output-token limit. Context-window errors and model
 * retirement messages must not change this per-run capability observation.
 */
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
