// ~4 chars ≈ 1 token, matching the daemon's context-manager heuristic
// (packages/daemon/src/agent/context-manager.ts). This estimates the
// *conversation the surface is holding* — not the full prompt, since the
// daemon prepends the system prompt + tool catalog + freshly-retrieved
// memory at request-build time — so the real context fill runs somewhat
// higher than this. It exists so the "ctx %" indicator tracks how much
// history is accumulating instead of the run's *cumulative* billed input
// tokens (which trivially exceeds the model window on any multi-step run
// because every react iteration re-sends the growing prompt).
const CHARS_PER_TOKEN = 4
// Rough envelope tokens per message (role marker, separators, etc).
const PER_MESSAGE_OVERHEAD_TOKENS = 4

/**
 * Structural shape of the surface message fields that contribute meaningfully
 * to context size. Kept deliberately minimal so this stays decoupled from the
 * exact `Message` type a caller is holding.
 */
export interface ConversationMessageLike {
  content: string
  toolResult?: string
}

export function estimateMessageTokens(message: ConversationMessageLike): number {
  const chars = message.content.length + (message.toolResult?.length ?? 0)
  return Math.ceil(chars / CHARS_PER_TOKEN) + PER_MESSAGE_OVERHEAD_TOKENS
}

export function estimateConversationTokens(
  messages: readonly ConversationMessageLike[],
): number {
  let total = 0
  for (const message of messages) {
    total += estimateMessageTokens(message)
  }
  return total
}

/**
 * `contextTokens / contextWindow` as a whole-number percent, or `null` when
 * the window is unknown. Not clamped to 100 — a value above 100 is a real
 * signal that the held history no longer fits and the agent is lossy-trimming
 * older turns on every request (time to /compact or /new).
 */
export function contextFillPercent(
  contextTokens: number,
  contextWindow: number | null | undefined,
): number | null {
  if (!contextWindow || contextWindow <= 0) {
    return null
  }
  return Math.min(999, Math.max(0, Math.round((contextTokens / contextWindow) * 100)))
}
