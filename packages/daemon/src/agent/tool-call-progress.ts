import type { AgentEvent, ToolCall } from '@sepilotd/core'

const TOOL_CALL_PROGRESS_CHAR_INTERVAL = 8 * 1024

export interface ToolCallProgressTracker {
  start: (toolCall: Partial<ToolCall>) => void
  delta: (toolCallId: string, delta: string) => AgentEvent | null
}

export function createToolCallProgressTracker(): ToolCallProgressTracker {
  const toolNames = new Map<string, string>()
  const streamedChars = new Map<string, number>()
  const reportedChars = new Map<string, number>()

  return {
    start(toolCall) {
      if (!toolCall.id) {
        return
      }

      if (toolCall.name) {
        toolNames.set(toolCall.id, toolCall.name)
      }
      streamedChars.set(toolCall.id, 0)
      reportedChars.set(toolCall.id, 0)
    },
    delta(toolCallId, delta) {
      const nextChars = (streamedChars.get(toolCallId) ?? 0) + delta.length
      const previousReported = reportedChars.get(toolCallId) ?? 0
      streamedChars.set(toolCallId, nextChars)

      if (nextChars - previousReported < TOOL_CALL_PROGRESS_CHAR_INTERVAL) {
        return null
      }

      reportedChars.set(toolCallId, nextChars)
      const toolName = toolNames.get(toolCallId)
      const sizeKiB = Math.max(1, Math.round(nextChars / 1024))
      return {
        type: 'thinking',
        content: toolName
          ? `Preparing ${toolName} input (${sizeKiB} KiB streamed)...`
          : `Preparing tool input (${sizeKiB} KiB streamed)...`,
      }
    },
  }
}
