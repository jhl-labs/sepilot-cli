const CHARS_PER_TOKEN = 4
const PER_MESSAGE_OVERHEAD_TOKENS = 4

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

export function contextFillPercent(
  contextTokens: number,
  contextWindow: number | null | undefined,
): number | null {
  if (!contextWindow || contextWindow <= 0) {
    return null
  }
  return Math.min(999, Math.max(0, Math.round((contextTokens / contextWindow) * 100)))
}
