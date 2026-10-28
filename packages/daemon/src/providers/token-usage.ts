import type { TokenUsage } from '@sepilotd/core'

export function accumulateTokenUsage(target: TokenUsage, source: TokenUsage): void {
  target.inputTokens += source.inputTokens
  target.outputTokens += source.outputTokens
  for (const field of ['thinkingTokens', 'cacheReadTokens', 'cacheCreationTokens', 'generationMs'] as const) {
    if (source[field] !== undefined) target[field] = (target[field] ?? 0) + source[field]
  }
  if (source.inputTokensIncludeCache !== undefined) target.inputTokensIncludeCache = source.inputTokensIncludeCache
  if (source.firstTokenMs !== undefined) target.firstTokenMs ??= source.firstTokenMs
}
