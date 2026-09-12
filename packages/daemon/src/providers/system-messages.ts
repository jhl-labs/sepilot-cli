import type { Message } from '@sepilotd/core'
import { TURN_CONTEXT_HEADING, splitTurnContext } from '../agent/prompt-sections.js'
import { extractContent } from './utils.js'

/**
 * Delimiters for per-turn context carried inside the latest user message.
 * Retrieved memory, daily notes and sender context change between turns; if
 * they sat in the system slot every provider's cached prefix (tools, system
 * prompt, and the whole conversation history) would be invalidated on every
 * turn. Attaching them to the newest user message keeps the change at the
 * tail of the request, where prefix caches expect it. The block is applied at
 * the provider boundary only and never persisted into the session history.
 */
export const TURN_CONTEXT_OPEN = '[Turn context — supplied by the system for this turn, not written by the user]'
export const TURN_CONTEXT_CLOSE = '[End turn context]'

export interface PartitionedSystemMessages {
  /**
   * Provider-facing control text in precedence order: the request-level
   * prompt first, followed by inline system messages in conversation order.
   */
  systemText?: string
  /** All non-system messages in their original order and representation. */
  messages: Message[]
}

/**
 * Normalize Sepilot's provider-neutral message history for APIs that expose a
 * leading system/control slot.
 *
 * Agent recovery and supervision messages are intentionally appended to the
 * in-memory history at the point where they become relevant. Several strict
 * chat templates, however, reject any `system` role after the first user or
 * assistant message. Provider adapters must therefore separate control text
 * from conversational/tool protocol before serialization. This function:
 *
 * - preserves request-level and inline system precedence,
 * - removes only system-role messages,
 * - leaves user/assistant/tool ordering and content untouched, and
 * - omits a provider system slot when every control block is empty.
 *
 * Keeping this provider-boundary normalization shared prevents individual
 * adapters from drifting as new model families impose different chat-template
 * restrictions.
 */
export function partitionSystemMessages(
  messages: readonly Message[],
  systemPrompt?: string,
): PartitionedSystemMessages {
  const systemTexts: string[] = []
  const conversationMessages: Message[] = []

  const appendSystemText = (text: string | undefined): void => {
    if (text?.trim()) systemTexts.push(text)
  }

  const { staticText, turnText } = systemPrompt
    ? splitTurnContext(systemPrompt)
    : { staticText: undefined, turnText: undefined }
  appendSystemText(staticText)
  for (const message of messages) {
    if (message.role === 'system') {
      appendSystemText(extractContent(message))
      continue
    }
    conversationMessages.push(message)
  }

  const turnBody = turnText?.slice(TURN_CONTEXT_HEADING.length).trim()
  if (turnBody) {
    const latestUserIndex = conversationMessages.findLastIndex((message) => message.role === 'user')
    if (latestUserIndex >= 0) {
      conversationMessages[latestUserIndex] = withTurnContext(
        conversationMessages[latestUserIndex]!,
        turnBody,
      )
    } else {
      // No user turn to attach to (e.g. a bare completion): keep the old
      // system-slot placement rather than dropping the context.
      appendSystemText(turnText)
    }
  }

  return {
    ...(systemTexts.length > 0 ? { systemText: systemTexts.join('\n\n') } : {}),
    messages: conversationMessages,
  }
}

export function formatTurnContextBlock(turnBody: string): string {
  return `${TURN_CONTEXT_OPEN}\n${turnBody}\n${TURN_CONTEXT_CLOSE}`
}

function withTurnContext(message: Message, turnBody: string): Message {
  const block = formatTurnContextBlock(turnBody)
  if (typeof message.content === 'string') {
    return { ...message, content: `${block}\n\n${message.content}` }
  }
  return {
    ...message,
    content: [{ type: 'text', text: block }, ...message.content],
  }
}
