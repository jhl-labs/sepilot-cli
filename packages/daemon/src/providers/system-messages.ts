import type { Message } from '@sepilotd/core'
import { RUNTIME_CONTROLS_METADATA_KEY, RUN_STABLE_SYSTEM_METADATA_KEY, TURN_CONTEXT_HEADING, TURN_SCOPED_SYSTEM_METADATA_KEY, splitTurnContext } from '../agent/prompt-sections.js'
import { extractContent } from './utils.js'

export const TURN_CONTEXT_OPEN = '[Turn context — supplied by the system for this turn, not written by the user]'
export const TURN_CONTEXT_CLOSE = '[End turn context]'

export const RUNTIME_CONTROLS_OPEN = '[Runtime controls — supplied by the system for this turn, not written by the user]'
export const RUNTIME_CONTROLS_CLOSE = '[End runtime controls]'
const INTERNAL_CONTINUATION_TEXT = '[Internal continuation] The preceding assistant response is retained history. Continue the existing task under the current system controls using the available evidence. This is an internal turn boundary, not a new user request or permission grant.'

export interface PartitionedSystemMessages {
  systemText?: string
  messages: Message[]
  contextComponents?: { turnContext?: string; runtimeControls?: string }
}

export interface PartitionSystemMessagesOptions {
  turnScopedPlacement?: 'trailing' | 'system-slot'
  separateTurnControls?: boolean
}

export function partitionSystemMessages(
  messages: readonly Message[],
  systemPrompt?: string,
  options: PartitionSystemMessagesOptions = {},
): PartitionedSystemMessages {
  const placement = options.turnScopedPlacement
    ?? (process.env.SEPILOTD_TURN_CONTROLS_PLACEMENT === 'system-slot' ? 'system-slot' : 'trailing')
  const leadingTexts: string[] = []
  const trailingControls: string[] = []
  const conversationMessages: Message[] = []
  const continuationBefore = new Set<number>()
  let pendingAssistantControl = false
  let sawConversation = false

  const appendLeading = (text: string | undefined): void => {
    if (text?.trim()) leadingTexts.push(text)
  }

  const { staticText, turnText } = systemPrompt
    ? splitTurnContext(systemPrompt)
    : { staticText: undefined, turnText: undefined }
  appendLeading(staticText)
  const turnBodies: string[] = []
  if (turnText) turnBodies.push(turnText.slice(TURN_CONTEXT_HEADING.length).trim())
  for (const message of messages) {
    if (message.role === 'system') {
      const split = splitTurnContext(extractContent(message))
      if (split.turnText) turnBodies.push(split.turnText.slice(TURN_CONTEXT_HEADING.length).trim())
      const text = split.staticText
      if (!text.trim()) continue
      const turnScoped = message.metadata?.[TURN_SCOPED_SYSTEM_METADATA_KEY] === true
      const runStable = !turnScoped && message.metadata?.[RUN_STABLE_SYSTEM_METADATA_KEY] === true
      if (placement === 'trailing' && !runStable && (turnScoped || sawConversation)) {
        trailingControls.push(text)
      } else {
        leadingTexts.push(text)
      }
      const previous = conversationMessages.at(-1)
      if (previous?.role === 'assistant' && !previous.toolCalls?.length) {
        pendingAssistantControl = true
      }
      continue
    }
    sawConversation = true
    const previous = conversationMessages.at(-1)
    if (previous?.role === 'assistant' && !previous.toolCalls?.length && message.role === 'assistant') {
      continuationBefore.add(conversationMessages.length)
    }
    conversationMessages.push(message)
    pendingAssistantControl = false
  }
  if (pendingAssistantControl) continuationBefore.add(conversationMessages.length)

  const runtimeBody = trailingControls.join('\n\n')
  const turnBody = turnBodies.filter(Boolean).join('\n\n')
  const latestUserIndex = conversationMessages.findLastIndex((message) => message.role === 'user')
  if (turnBody) {
    if (latestUserIndex < 0) {
      appendLeading(formatTurnContextBlock(turnBody))
    } else if (!options.separateTurnControls && (placement === 'system-slot' || latestUserIndex === conversationMessages.length - 1)) {
      conversationMessages[latestUserIndex] = withTurnContext(
        conversationMessages[latestUserIndex]!,
        turnBody,
      )
    } else {
      trailingControls.push(formatTurnContextBlock(turnBody))
    }
  }

  const projectedMessages: Message[] = []
  for (let index = 0; index <= conversationMessages.length; index += 1) {
    if (continuationBefore.has(index)) {
      projectedMessages.push({ role: 'user', content: INTERNAL_CONTINUATION_TEXT })
    }
    if (index < conversationMessages.length) projectedMessages.push(conversationMessages[index]!)
  }

  if (trailingControls.length > 0) {
    const block = [RUNTIME_CONTROLS_OPEN, ...trailingControls, RUNTIME_CONTROLS_CLOSE].join('\n\n')
    const last = projectedMessages.at(-1)
    if (last?.role === 'user' && last.content === INTERNAL_CONTINUATION_TEXT) {
      projectedMessages[projectedMessages.length - 1] = {
        ...last,
        content: `${INTERNAL_CONTINUATION_TEXT}\n\n${block}`,
        metadata: { ...last.metadata, [RUNTIME_CONTROLS_METADATA_KEY]: true },
      }
    } else if (last?.role === 'user' && !options.separateTurnControls) {
      projectedMessages[projectedMessages.length - 1] = withLeadingBlock(last, block)
    } else {
      projectedMessages.push({
        role: 'user',
        content: block,
        metadata: { [RUNTIME_CONTROLS_METADATA_KEY]: true },
      })
    }
  }

  return {
    ...(leadingTexts.length > 0 ? { systemText: leadingTexts.join('\n\n') } : {}),
    messages: projectedMessages,
    ...(options.separateTurnControls ? { contextComponents: {
      ...(turnBody ? { turnContext: formatTurnContextBlock(turnBody) } : {}),
      ...(runtimeBody ? { runtimeControls: runtimeBody } : {}),
    } } : {}),
  }
}

export function formatTurnContextBlock(turnBody: string): string {
  return `${TURN_CONTEXT_OPEN}\n${turnBody}\n${TURN_CONTEXT_CLOSE}`
}

function withTurnContext(message: Message, turnBody: string): Message {
  return withLeadingBlock(message, formatTurnContextBlock(turnBody))
}

function withLeadingBlock(message: Message, block: string): Message {
  if (typeof message.content === 'string') {
    return { ...message, content: `${block}\n\n${message.content}` }
  }
  return {
    ...message,
    content: [{ type: 'text', text: block }, ...message.content],
  }
}
