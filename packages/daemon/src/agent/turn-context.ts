import type { Message } from '@sepilotd/core'

export const CURRENT_AGENT_TURN_USER_METADATA_KEY = 'currentAgentTurnUserInput'
export const CURRENT_AGENT_TURN_INSTRUCTION_METADATA_KEY = 'currentAgentTurnInstruction'

function contentEquals(left: Message['content'], right: Message['content']): boolean {
  if (typeof left === 'string' || typeof right === 'string') return left === right
  return JSON.stringify(left) === JSON.stringify(right)
}

function clearCurrentAgentTurnMarkers(messages: Message[]): Message[] {
  return messages.map((message) => {
    if (message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] !== true) {
      return message
    }
    const metadata = { ...message.metadata }
    delete metadata[CURRENT_AGENT_TURN_USER_METADATA_KEY]
    delete metadata[CURRENT_AGENT_TURN_INSTRUCTION_METADATA_KEY]
    const messageWithoutMarker: Message = { ...message, metadata }
    if (Object.keys(metadata).length === 0) {
      delete messageWithoutMarker.metadata
    }
    return messageWithoutMarker
  })
}

function markCurrentAgentTurnUserMessage(
  messages: Message[],
  content: Message['content'],
  instruction?: string,
): Message[] {
  let currentIndex = -1
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message.role === 'system') {
      continue
    }
    if (message.role === 'user' && contentEquals(message.content, content)) {
      currentIndex = index
    }
    break
  }

  if (currentIndex < 0) {
    return [
      ...messages,
      {
        role: 'user',
        content,
        metadata: {
          [CURRENT_AGENT_TURN_USER_METADATA_KEY]: true,
          ...(instruction ? { [CURRENT_AGENT_TURN_INSTRUCTION_METADATA_KEY]: instruction } : {}),
        },
      },
    ]
  }

  return messages.map((message, index) => index === currentIndex
    ? {
        ...message,
        metadata: {
          ...message.metadata,
          [CURRENT_AGENT_TURN_USER_METADATA_KEY]: true,
          ...(instruction ? { [CURRENT_AGENT_TURN_INSTRUCTION_METADATA_KEY]: instruction } : {}),
        },
      }
    : message)
}

export function beginCurrentAgentTurnUserMessage(
  messages: Message[],
  content: Message['content'],
  instruction?: string,
): Message[] {
  return markCurrentAgentTurnUserMessage(
    clearCurrentAgentTurnMarkers(messages),
    content,
    instruction,
  )
}

export function ensureCurrentAgentTurnUserMessage(
  messages: Message[],
  content: Message['content'],
  instruction?: string,
): Message[] {
  const markedIndexes = messages.flatMap((message, index) =>
    message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true ? [index] : [],
  )
  const matchingIndexes = markedIndexes.filter((index) => {
    const message = messages[index]
    return message?.role === 'user' && contentEquals(message.content, content)
  })
  if (markedIndexes.length === 1 && matchingIndexes.length === 1) {
    return messages
  }

  const withoutMarkers = clearCurrentAgentTurnMarkers(messages)
  const currentIndex = matchingIndexes.at(-1)
  if (currentIndex !== undefined) {
    return withoutMarkers.map((message, index) => index === currentIndex
      ? {
          ...message,
          metadata: {
            ...message.metadata,
            [CURRENT_AGENT_TURN_USER_METADATA_KEY]: true,
            ...(instruction ? { [CURRENT_AGENT_TURN_INSTRUCTION_METADATA_KEY]: instruction } : {}),
          },
        }
      : message)
  }
  return markCurrentAgentTurnUserMessage(withoutMarkers, content, instruction)
}

const SIMPLE_AFFIRMATION_PATTERN = /^(?:yes|y|ok|okay|sure|go ahead|please do|do it|continue|proceed|sounds good|that works|응|네|예|ㅇㅇ|그래|좋아|좋습니다|좋아요|알겠어|알겠습니다|오케이)(?:[.!?~\s]+.*)?$/i
const ACTIONABLE_FOLLOWUP_PATTERN = /(?:그렇게|그거|그걸|그대로|방금|이어서|나머지).*(?:해|해줘|해주세요|진행|계속|하라고|해봐)|^(?:진행해|진행해줘|계속해|계속 진행해|해줘|해주세요|부탁해|부탁드립니다)(?:[.!?~\s]+.*)?$/i

function normalizeText(value: string): string {
  return replayConversationTextControls(value).trim().replace(/\s+/g, ' ')
}

const conversationGraphemeSegmenter = new Intl.Segmenter(undefined, {
  granularity: 'grapheme',
})

/**
 * Replay terminal replacement controls that may exist in legacy session
 * events. DEL/BS removes the preceding grapheme; other non-text C0 controls
 * are discarded before history is sent back to a model.
 */
export function replayConversationTextControls(input: string): string {
  let value = ''
  for (const ch of input) {
    if (ch === '\x7f' || ch === '\b') {
      const segments = [...conversationGraphemeSegmenter.segment(value)]
      const previous = segments.at(-1)
      value = previous ? value.slice(0, previous.index) : ''
      continue
    }
    if (ch < ' ' && ch !== '\n' && ch !== '\t') continue
    value += ch
  }
  return value.normalize('NFC')
}

function extractTextContent(message: Message): string {
  if (typeof message.content === 'string') {
    return message.content
  }

  if (Array.isArray(message.content)) {
    return message.content
      .filter((part): part is Extract<typeof part, { type: 'text' }> => part.type === 'text')
      .map((part) => part.text)
      .join(' ')
  }

  return ''
}

function truncate(value: string, limit: number): string {
  return value.length > limit
    ? `${value.slice(0, limit - 1)}…`
    : value
}

function findLatestAssistantReply(messages: Message[]): string | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message.role !== 'assistant') {
      continue
    }

    const content = normalizeText(extractTextContent(message))
    if (content) {
      return content
    }
  }

  return null
}

export function isLikelyShortApprovalFollowup(input: string): boolean {
  const normalized = normalizeText(input)
  if (!normalized || normalized.length > 80) {
    return false
  }

  return SIMPLE_AFFIRMATION_PATTERN.test(normalized)
    || ACTIONABLE_FOLLOWUP_PATTERN.test(normalized)
}

export function stripTrailingDuplicateUserMessage(
  input: string,
  previousMessages?: Message[],
): Message[] | undefined {
  if (!previousMessages?.length) {
    return previousMessages
  }

  const normalizedInput = normalizeText(input)
  if (!normalizedInput) {
    return previousMessages
  }

  let end = previousMessages.length
  while (end > 0) {
    const candidate = previousMessages[end - 1]
    if (candidate.role !== 'user') {
      break
    }

    const content = normalizeText(extractTextContent(candidate))
    if (content !== normalizedInput) {
      break
    }

    end -= 1
  }

  return end === previousMessages.length
    ? previousMessages
    : previousMessages.slice(0, end)
}

export function buildFollowupContinuityMessage(
  input: string,
  previousMessages?: Message[],
): Message | null {
  if (!previousMessages?.length || !isLikelyShortApprovalFollowup(input)) {
    return null
  }

  const lastAssistantReply = findLatestAssistantReply(previousMessages)
  if (!lastAssistantReply) {
    return null
  }

  return {
    role: 'system',
    content: [
      '[Follow-up continuity]',
      'The user\'s latest reply is brief and refers to the immediately preceding assistant message already present in conversation history.',
      `Latest user reply:\n${truncate(normalizeText(input), 120)}`,
      'If the previous reply proposed a next step, asked for confirmation, or offered to continue with a specific action, treat the user\'s latest reply as approval to perform that specific action now. Do not ask again unless the previous reply was genuinely ambiguous or offered multiple materially different options.',
    ].join('\n\n'),
  }
}

export function preparePreviousMessagesForTurn(
  input: string,
  previousMessages?: Message[],
): Message[] | undefined {
  const sanitized = stripTrailingDuplicateUserMessage(input, previousMessages)
  const continuityMessage = buildFollowupContinuityMessage(input, sanitized)

  return continuityMessage
    ? [...(sanitized ?? []), continuityMessage]
    : sanitized
}
