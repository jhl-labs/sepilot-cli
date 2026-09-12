import type { ChatResponse } from '@sepilotd/core'

export const INTERRUPTED_USER_FACING_RESPONSE =
  'INCOMPLETE: The user-facing response stream was interrupted before a complete response could be validated. Retry or resume the run.'

export const INCOMPLETE_EXECUTION_CONTINUATION =
  'These results are partial execution evidence, not a completed answer or proof that the user\'s task is complete.\n'
  + 'On resume, compare the original request with retained evidence, identify unfinished work, and continue within the user\'s current instructions. Do not assume that only a final summary remains.'

interface RetainedToolResult {
  tool: string
  status: 'success' | 'error'
  output?: string
}

const MAX_RETAINED_TOOL_RESULTS = 6
const MAX_RETAINED_TOOL_RESULT_CHARS = 240

function compactRetainedToolResult(value: string | undefined): string {
  const compact = value?.trim().replace(/\s+/gu, ' ') ?? ''
  if (!compact) return 'No bounded output was retained.'
  return compact.length > MAX_RETAINED_TOOL_RESULT_CHARS
    ? `${compact.slice(0, MAX_RETAINED_TOOL_RESULT_CHARS - 1)}…`
    : compact
}

/**
 * Preserve daemon-owned current-turn execution evidence when the provider's
 * final prose cannot be published. This never reuses the partial provider
 * stream and never presents tool results as a completed answer.
 */
export function interruptedUserFacingResponse(
  toolResults: readonly RetainedToolResult[],
): string {
  if (toolResults.length === 0) return INTERRUPTED_USER_FACING_RESPONSE

  return [
    'INCOMPLETE: The final response was interrupted before a complete answer could be validated.',
    'Retained current-turn tool results:',
    ...toolResults.slice(-MAX_RETAINED_TOOL_RESULTS).map((result) => (
      `- [${result.status}] ${result.tool}: ${compactRetainedToolResult(result.output)}`
    )),
    INCOMPLETE_EXECUTION_CONTINUATION,
  ].join('\n')
}

export function canPublishUserFacingText(
  finishReason: ChatResponse['finishReason'],
): boolean {
  return finishReason === 'stop'
}

export function canExecuteCompletedToolCalls(
  finishReason: ChatResponse['finishReason'],
): boolean {
  return finishReason === 'stop' || finishReason === 'tool_use'
}
