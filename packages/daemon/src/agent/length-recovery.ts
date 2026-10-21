import type { AgentEvent } from '@sepilotd/core'
import type { AgentState, AgentTruncatedToolCall } from './graph/types.js'

export function lengthRecoveryInstruction(partialOutput: string): string {
  return partialOutput.trim()
    ? 'Your previous reply was cut off because it reached the output length limit. Continue exactly from where you left off; do not repeat what you already wrote.'
    : 'The output length limit was reached without a usable answer or executable tool call. There is no partial answer to continue. Resume from the actual user request and retained tool evidence: take the next concrete tool step, or give a concise complete answer if no action is needed. Keep planning brief; do not restart the entire design or invent completed work.'
}


const TRUNCATED_ATTEMPT_SIGNATURE_PREFIX = 'truncated-tool-call:'

function formatTruncatedToolCalls(toolCalls: ReadonlyArray<AgentTruncatedToolCall>): string {
  return toolCalls
    .map((call) => `${call.name} (~${Math.max(1, Math.round(call.streamedChars / 1024))} KiB streamed)`)
    .join(', ')
}

export function recordTruncatedToolCallAttempt(
  state: Pick<AgentState, 'failedAttempts'>,
  toolCalls: ReadonlyArray<AgentTruncatedToolCall>,
  maxOutputTokens: number | undefined,
  now: () => number = () => Date.now(),
): number {
  const list = (state.failedAttempts ??= [])
  const attempt = list.filter((entry) => entry.signature.startsWith(TRUNCATED_ATTEMPT_SIGNATURE_PREFIX)).length + 1
  const tool = toolCalls.map((call) => call.name).filter(Boolean).join('+') || 'tool-call'
  const limit = maxOutputTokens ? `the ${maxOutputTokens}-token output limit` : 'the output limit'
  list.push({
    signature: `${TRUNCATED_ATTEMPT_SIGNATURE_PREFIX}${attempt}`,
    tool,
    reason: `attempt ${attempt}: ${formatTruncatedToolCalls(toolCalls) || tool} exceeded ${limit} and was discarded unexecuted; split the artifact into smaller complete calls (skeleton first, then fs.append or separate files)`,
    ts: now(),
  })
  return attempt
}

export function buildTruncatedToolCallRecoveryEvent(
  toolCalls: ReadonlyArray<AgentTruncatedToolCall>,
  maxOutputTokens: number | undefined,
  attempt: number,
  continuation: { budget: number; used: number },
): AgentEvent {
  const exhausted = continuation.used >= continuation.budget
  return {
    type: 'recovery',
    scope: 'provider_protocol',
    kind: 'truncated_tool_call',
    action: exhausted ? 'return_control_to_graph' : 'request_bounded_tool_call',
    message: `${formatTruncatedToolCalls(toolCalls) || 'A tool call'} exceeded the output limit${maxOutputTokens ? ` (${maxOutputTokens} tokens)` : ''} and was discarded unexecuted (truncation ${attempt}); ${exhausted ? 'handing the failed attempt to the graph controller' : 'asking for a smaller complete call'}.`,
    recoverable: true,
    details: {
      toolNames: toolCalls.map((call) => call.name),
      streamedChars: toolCalls.map((call) => call.streamedChars),
      maxOutputTokens,
      attempt,
      budget: continuation.budget,
      used: continuation.used,
      graphMutation: false,
    },
  }
}
