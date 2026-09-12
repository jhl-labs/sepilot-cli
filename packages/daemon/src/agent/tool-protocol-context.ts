import type { Message, ToolCall } from '@sepilotd/core'

function hasOnlyEmptyTextContent(message: Message): boolean {
  return message.content === ''
    || (
      Array.isArray(message.content)
      && message.content.every((part) => part.type === 'text' && part.text.trim() === '')
    )
}

function mergeToolCalls(
  existing: ToolCall[],
  incoming: ToolCall[],
): ToolCall[] {
  const seenIds = new Set(existing.map((toolCall) => toolCall.id))
  return [
    ...existing,
    ...incoming.filter((toolCall) => !seenIds.has(toolCall.id)),
  ]
}

/**
 * Session journals persist each member of a parallel tool-call batch as an
 * individual event. On reload that can produce adjacent, empty assistant
 * messages even though providers require one assistant tool-use message
 * followed by all matching tool results. Coalesce only that unambiguous empty
 * fragment shape; substantive adjacent assistant messages remain untouched.
 */
export function coalesceAdjacentAssistantToolCallMessages(
  messages: readonly Message[],
): Message[] {
  const normalized: Message[] = []

  for (const message of messages) {
    const previous = normalized.at(-1)
    if (
      previous?.role === 'assistant'
      && message.role === 'assistant'
      && (previous.toolCalls?.length ?? 0) > 0
      && (message.toolCalls?.length ?? 0) > 0
      && hasOnlyEmptyTextContent(previous)
      && hasOnlyEmptyTextContent(message)
    ) {
      normalized[normalized.length - 1] = {
        ...previous,
        toolCalls: mergeToolCalls(previous.toolCalls!, message.toolCalls!),
        ...(
          previous.metadata || message.metadata
            ? { metadata: { ...previous.metadata, ...message.metadata } }
            : {}
        ),
      }
      continue
    }

    normalized.push(message)
  }

  return normalized
}
