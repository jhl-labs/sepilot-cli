import type { AgentEvent, Message, ToolCall } from '@sepilotd/core'

export const MAX_PROVIDER_MESSAGE_RECOVERY_REPAIRS = 2

export type ProviderMessageRecoveryKind =
  | 'missing_tool_result'
  | 'system_message_order'
  | 'thinking_block_order'
  | 'thinking_disabled_violation'
  | 'empty_message_content'

export interface ProviderMessageRecovery {
  kind: ProviderMessageRecoveryKind
  reason: string
  disableThinking: boolean
}

export interface ProviderMessageProtocolRepair {
  messages: Message[]
  repaired: boolean
  reasons: string[]
}

const PROVIDER_RECOVERY_MARKER = '[Provider protocol recovery]'

function normalizeSnippet(value: string, limit = 260): string {
  const normalized = value.trim().replace(/\s+/g, ' ')
  if (!normalized) return ''
  return normalized.length > limit ? `${normalized.slice(0, limit - 1)}...` : normalized
}

function getNestedMessage(value: unknown): string {
  if (!value) return ''
  if (typeof value === 'string') return value
  if (value instanceof Error) return value.message
  if (typeof value !== 'object') return String(value)

  const record = value as Record<string, unknown>
  const candidates = [
    record.message,
    record.error,
    record.data,
    record.response,
    record.cause,
    (record.data as Record<string, unknown> | undefined)?.error,
    (record.error as Record<string, unknown> | undefined)?.message,
    (record.response as Record<string, unknown> | undefined)?.data,
  ]

  for (const candidate of candidates) {
    const message = getNestedMessage(candidate)
    if (message) return message
  }

  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

export function detectProviderMessageRecovery(error: unknown): ProviderMessageRecovery | null {
  const rawMessage = getNestedMessage(error)
  const message = rawMessage.toLowerCase()
  if (!message) return null

  if (
    (
      message.includes('system message')
      || message.includes('system role')
      || message.includes('system prompt')
      || message.includes('developer message')
      || message.includes('developer role')
    )
    && (
      message.includes('at the beginning')
      || message.includes('at the start')
      || message.includes('first message')
      || message.includes('must be first')
      || message.includes('before any user')
    )
  ) {
    return {
      kind: 'system_message_order',
      reason: normalizeSnippet(rawMessage),
      disableThinking: false,
    }
  }

  if (
    message.includes('thinking is disabled') &&
    (message.includes('cannot contain') || message.includes('thinking'))
  ) {
    return {
      kind: 'thinking_disabled_violation',
      reason: normalizeSnippet(rawMessage),
      disableThinking: true,
    }
  }

  if (
    message.includes('thinking') &&
    (message.includes('first block') ||
      message.includes('must start with') ||
      message.includes('preceding') ||
      message.includes('preceeding') ||
      message.includes('final block') ||
      message.includes('cannot be thinking') ||
      (message.includes('expected') && message.includes('found')))
  ) {
    return {
      kind: 'thinking_block_order',
      reason: normalizeSnippet(rawMessage),
      disableThinking: true,
    }
  }

  if (
    (message.includes('tool_use') ||
      message.includes('tool call') ||
      message.includes('tool_calls')) &&
    (message.includes('tool_result') ||
      message.includes('tool result') ||
      message.includes('tool_call_id'))
  ) {
    return {
      kind: 'missing_tool_result',
      reason: normalizeSnippet(rawMessage),
      disableThinking: false,
    }
  }

  if (
    message.includes('empty') &&
    message.includes('message') &&
    (message.includes('content') || message.includes('text') || message.includes('parts'))
  ) {
    return {
      kind: 'empty_message_content',
      reason: normalizeSnippet(rawMessage),
      disableThinking: false,
    }
  }

  return null
}

function syntheticToolResult(toolCall: ToolCall): Message {
  return {
    role: 'tool',
    toolCallId: toolCall.id,
    content: [
      '[provider recovery]',
      `Tool result for ${toolCall.name || 'unknown tool'} was missing from the local provider transcript.`,
      'Treat the interrupted tool call as cancelled and continue from the latest available context.',
    ].join(' '),
  }
}

function hasEmptyContent(message: Message): boolean {
  if (typeof message.content === 'string') {
    return message.content.trim().length === 0
  }
  return (
    message.content.length === 0 ||
    message.content.every((part) => part.type === 'text' && part.text.trim().length === 0)
  )
}

function fillEmptyContent(message: Message): Message {
  if (
    message.role === 'assistant' &&
    (!message.toolCalls || message.toolCalls.length === 0) &&
    hasEmptyContent(message)
  ) {
    return {
      ...message,
      content: '[empty assistant message recovered]',
    }
  }

  if (message.role === 'user' && hasEmptyContent(message)) {
    return {
      ...message,
      content: '[empty user message recovered]',
    }
  }

  return message
}

export function repairProviderMessageProtocol(messages: Message[]): ProviderMessageProtocolRepair {
  const repairedMessages: Message[] = []
  const reasons = new Set<string>()
  const pendingToolCalls = new Map<string, ToolCall>()

  const flushMissingToolResults = () => {
    if (pendingToolCalls.size === 0) return
    for (const toolCall of pendingToolCalls.values()) {
      repairedMessages.push(syntheticToolResult(toolCall))
    }
    reasons.add('inserted cancelled tool_result messages for unmatched tool_use calls')
    pendingToolCalls.clear()
  }

  for (const original of messages) {
    const message = fillEmptyContent(original)
    if (message !== original) {
      reasons.add('filled empty message content')
    }

    if (message.role !== 'tool') {
      flushMissingToolResults()
    }

    repairedMessages.push(message)

    if (message.role === 'assistant' && message.toolCalls?.length) {
      pendingToolCalls.clear()
      for (const toolCall of message.toolCalls) {
        if (toolCall.id) {
          pendingToolCalls.set(toolCall.id, toolCall)
        }
      }
    } else if (message.role === 'tool' && message.toolCallId) {
      pendingToolCalls.delete(message.toolCallId)
    }
  }

  flushMissingToolResults()

  return {
    messages: repairedMessages,
    repaired: reasons.size > 0,
    reasons: [...reasons],
  }
}

export function buildProviderMessageRecoveryMessage(recovery: ProviderMessageRecovery): Message {
  const instructions = [
    `${PROVIDER_RECOVERY_MARKER}`,
    `The previous provider request was rejected because its message protocol looked invalid: ${recovery.reason || recovery.kind}.`,
    'The runtime will sanitize only provider-facing message shape on the next request; persisted conversation events and tool policy are unchanged.',
    recovery.disableThinking
      ? 'Thinking output will be disabled for the retry because the provider rejected thinking blocks.'
      : '',
    'Continue the user task from the latest available context. Do not mention this internal provider recovery unless the user asks about it.',
  ].filter(Boolean)

  return {
    role: 'system',
    content: instructions.join('\n'),
  }
}

export function buildProviderMessageRecoveryEvent(
  recovery: ProviderMessageRecovery,
): Extract<AgentEvent, { type: 'recovery' }> {
  return {
    type: 'recovery',
    scope: 'provider_protocol',
    kind: recovery.kind,
    action: recovery.disableThinking
      ? 'retry_sanitized_messages_without_thinking'
      : 'retry_sanitized_messages',
    message: `Provider message protocol recovery: ${recovery.kind}`,
    recoverable: true,
    details: {
      reason: recovery.reason,
      disableThinking: recovery.disableThinking,
      graphMutation: false,
    },
  }
}

export function countProviderMessageRecoveryPrompts(messages: Message[]): number {
  return messages.filter(
    (message) =>
      message.role === 'system' &&
      typeof message.content === 'string' &&
      message.content.includes(PROVIDER_RECOVERY_MARKER),
  ).length
}

export function shouldRecoverProviderMessageError(options: {
  recovery: ProviderMessageRecovery | null
  repairedCount: number
  maxRepairs?: number
  signal?: AbortSignal
  aborted: boolean
}): options is {
  recovery: ProviderMessageRecovery
  repairedCount: number
  maxRepairs?: number
  signal?: AbortSignal
  aborted: boolean
} {
  if (!options.recovery) return false
  if (options.aborted || (options.signal?.aborted ?? false)) return false
  return options.repairedCount < (options.maxRepairs ?? MAX_PROVIDER_MESSAGE_RECOVERY_REPAIRS)
}
