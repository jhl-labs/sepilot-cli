import type { AgentEvent, ChatResponse, Message } from '@sepilotd/core'
import { extractContent } from '../providers/utils.js'

export interface ToolTransportRecovery {
  reason: string
  source: 'provider_rejection' | 'empty_native_response' | 'empty_native_completion'
}

const TOOL_TRANSPORT_RECOVERY_MARKER = '[Native tool transport recovery]'

export function hasToolTransportRecoveryMessage(messages: Message[]): boolean {
  return messages.some((message) =>
    message.role === 'system'
    && typeof message.content === 'string'
    && message.content.includes(TOOL_TRANSPORT_RECOVERY_MARKER),
  )
}

function errorMessage(error: unknown): string {
  if (!error) return ''
  if (typeof error === 'string') return error
  if (error instanceof Error) return error.message
  if (typeof error !== 'object') return String(error)

  const record = error as Record<string, unknown>
  for (const candidate of [
    record.message,
    record.error,
    record.data,
    record.response,
    record.cause,
  ]) {
    const nested = errorMessage(candidate)
    if (nested) return nested
  }
  return ''
}

function normalizeReason(value: string, limit = 260): string {
  const normalized = value.trim().replace(/\s+/g, ' ')
  return normalized.length > limit
    ? `${normalized.slice(0, limit - 1)}...`
    : normalized
}

/**
 * Detect a provider rejecting the native function-calling request shape.
 * Matching requires both a tool-transport concept and an explicit
 * unavailable/invalid signal so ordinary tool execution errors do not change
 * transports.
 */
export function detectNativeToolTransportRejection(
  error: unknown,
): ToolTransportRecovery | null {
  const raw = errorMessage(error)
  const message = raw.toLowerCase()
  if (!message) return null

  const mentionsTransport =
    message.includes('tool_choice')
    || message.includes('tool calls')
    || message.includes('tool calling')
    || message.includes('function calling')
    || message.includes('function_call')
    || message.includes('tools parameter')
    || message.includes('tools field')
  const rejectsTransport =
    message.includes('not supported')
    || message.includes('does not support')
    || message.includes('unsupported')
    || message.includes('not available')
    || message.includes('unknown parameter')
    || message.includes('unrecognized request argument')

  if (!mentionsTransport || !rejectsTransport) return null
  return {
    reason: normalizeReason(raw),
    source: 'provider_rejection',
  }
}

/** A successful native turn with neither visible text nor tool calls is not a
 * usable protocol result. Reasoning-only output also qualifies: private
 * reasoning cannot complete the user turn or execute an action. */
export function detectEmptyNativeToolResponse(
  response: ChatResponse,
): ToolTransportRecovery | null {
  if (response.finishReason !== 'stop') return null
  if ((response.message.toolCalls?.length ?? 0) > 0) return null
  if (extractContent(response.message).trim()) return null
  return {
    reason: 'Provider completed a native-tool request without visible text or tool calls.',
    source: 'empty_native_response',
  }
}

export function buildToolTransportRecoveryMessage(
  recovery: ToolTransportRecovery,
): Message {
  return {
    role: 'system',
    content: [
      TOOL_TRANSPORT_RECOVERY_MARKER,
      `Native function calling was unusable (${recovery.source}): ${recovery.reason}`,
      'Retry once with the provider-neutral prompt tool protocol.',
      'Emit exactly one complete <sepilot_tool_call>{"name":"...","arguments":{...}}</sepilot_tool_call> when an action remains, or a complete ANSWER:/INCOMPLETE: final response when no tool is needed.',
      'Do not repeat or describe a tool call in ordinary prose.',
    ].join('\n'),
  }
}

export function buildToolTransportRecoveryEvent(
  recovery: ToolTransportRecovery,
): Extract<AgentEvent, { type: 'recovery' }> {
  return {
    type: 'recovery',
    scope: 'provider_protocol',
    kind: 'native_tool_transport',
    action: 'retry_with_prompt_tool_transport',
    message: 'Native tool transport was unusable; retrying with prompt tool transport.',
    recoverable: true,
    details: {
      reason: recovery.reason,
      source: recovery.source,
      graphMutation: false,
    },
  }
}
