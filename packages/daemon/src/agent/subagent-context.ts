import type { Message } from '@sepilotd/core'
import { estimateMessagesTokens } from './session-compaction.js'

const FORK_MAX_TOKENS_ENV = 'SEPILOTD_SUBAGENT_FORK_MAX_TOKENS'
export const DEFAULT_FORK_MAX_TOKENS = 24_000
const MIN_FORK_TOKENS = 1_000
const CONTEXT_WINDOW_SHARE = 0.5
export const DEFAULT_FORK_TOOL_RESULT_CHARS = 4_000

export function resolveChildHistoryBudgetTokens(
  contextWindow: number | undefined,
  requested = Number(process.env[FORK_MAX_TOKENS_ENV] ?? DEFAULT_FORK_MAX_TOKENS),
): number {
  const configured = Number.isFinite(requested) && requested > 0
    ? Math.floor(requested)
    : DEFAULT_FORK_MAX_TOKENS
  const windowCap = contextWindow && contextWindow > 0
    ? Math.floor(contextWindow * CONTEXT_WINDOW_SHARE)
    : configured
  return Math.max(Math.min(MIN_FORK_TOKENS, windowCap), Math.min(configured, windowCap))
}

function textOf(content: Message['content']): string {
  if (typeof content === 'string') return content
  return content
    .map((part) => (part.type === 'text' ? part.text : `[${part.type} omitted]`))
    .join('\n')
}

export function boundConversationForChild(
  messages: readonly Message[],
  options: { maxTokens: number; maxToolResultChars?: number },
): Message[] {
  const maxToolResultChars = options.maxToolResultChars ?? DEFAULT_FORK_TOOL_RESULT_CHARS
  const normalized: Message[] = []
  for (const message of messages) {
    if (message.role === 'system') continue
    let content = textOf(message.content)
    if (message.role === 'tool' && content.length > maxToolResultChars) {
      content = `${content.slice(0, maxToolResultChars)}\n[tool result truncated for subagent context: ${content.length - maxToolResultChars} chars omitted]`
    }
    normalized.push({
      role: message.role,
      content,
      ...(message.toolCalls?.length ? { toolCalls: message.toolCalls.map((call) => ({ ...call })) } : {}),
      ...(message.toolCallId ? { toolCallId: message.toolCallId } : {}),
      ...(message.name ? { name: message.name } : {}),
    })
  }

  const selected: Message[] = []
  let used = 0
  for (let index = normalized.length - 1; index >= 0; index--) {
    const message = normalized[index]!
    const cost = estimateMessagesTokens([message])
    if (used + cost > options.maxTokens) break
    selected.unshift(message)
    used += cost
  }

  const callIds = new Set(selected.flatMap((message) => message.toolCalls?.map((call) => call.id) ?? []))
  const resultIds = new Set(
    selected.filter((message) => message.role === 'tool' && message.toolCallId).map((message) => message.toolCallId!),
  )
  const repaired: Message[] = []
  for (const message of selected) {
    if (message.role === 'tool') {
      if (message.toolCallId && callIds.has(message.toolCallId)) repaired.push(message)
      continue
    }
    if (message.toolCalls?.length) {
      const kept = message.toolCalls.filter((call) => resultIds.has(call.id))
      if (kept.length === 0) {
        const { toolCalls: _dropped, ...rest } = message
        if (textOf(rest.content).trim()) repaired.push(rest)
        continue
      }
      repaired.push({ ...message, toolCalls: kept })
      continue
    }
    if (textOf(message.content).trim()) repaired.push(message)
  }

  const firstUser = repaired.findIndex((message) => message.role === 'user')
  return firstUser > 0 ? repaired.slice(firstUser) : repaired
}
