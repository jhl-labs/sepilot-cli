import type { ISessionStore } from '@sepilotd/core'
import { throwIfAborted } from '../abort.js'
import type { ToolDefinitionRuntime } from './registry.js'

export function createToolOutputReadTool(store: ISessionStore): ToolDefinitionRuntime {
  return {
    name: 'tool.output.read',
    description: 'Read a page of previously captured tool output in this session without re-running the original tool. Use the tool call ID (also shown on summarized output). offset is a zero-based character position; limit defaults to 4000, maximum 6000. Optional find searches for a case-sensitive literal at or after offset and starts the page at that match. Follow nextOffset to continue. This is a historical snapshot, not current file contents or a new validation result. Unavailable or ambiguous captures cannot be recovered.',
    resumeSafety: 'replay-safe',
    unavailableReason: () => store.getToolResult ? undefined : 'Session output readback is unavailable',
    inputSchema: {
      type: 'object', additionalProperties: false,
      properties: {
        toolCallId: { type: 'string', minLength: 1, maxLength: 128 },
        offset: { type: 'integer', minimum: 0 },
        limit: { type: 'integer', minimum: 1, maximum: 6000 },
        find: { type: 'string', minLength: 1, maxLength: 256 },
      },
      required: ['toolCallId'],
    },
    async execute(input, context) {
      const started = Date.now()
      const failure = (output: string) => ({ status: 'error' as const, durationMs: Date.now() - started, output })
      throwIfAborted(context?.signal)
      const { toolCallId, offset = 0, limit = 4000, find } = input
      if (Object.keys(input).some((key) => !['toolCallId', 'offset', 'limit', 'find'].includes(key))
        || typeof toolCallId !== 'string' || !toolCallId || toolCallId.length > 128
        || typeof offset !== 'number' || !Number.isSafeInteger(offset) || offset < 0
        || typeof limit !== 'number' || !Number.isSafeInteger(limit) || limit < 1 || limit > 6000
        || (find !== undefined && (typeof find !== 'string' || !find || find.length > 256))) {
        return failure('Invalid output readback arguments.')
      }
      if (!context?.sessionId || !store.getToolResult) {
        return failure('Session output readback is unavailable.')
      }
      const captured = await store.getToolResult(context.sessionId, toolCallId, { signal: context.signal })
      throwIfAborted(context.signal)
      if (!captured) {
        return failure('Captured output is unavailable or ambiguous in this session. No command was executed.')
      }
      const { call, result } = captured
      const start = find === undefined ? Math.min(offset, result.output.length) : result.output.indexOf(find, offset)
      const end = start < 0 ? -1 : Math.min(start + limit, result.output.length)
      const header = {
        snapshot: true, toolCallId, tool: call.tool, capturedAt: result.timestamp,
        originalStatus: result.status, totalChars: result.output.length,
        offset: start < 0 ? null : start,
        nextOffset: end >= 0 && end < result.output.length ? end : null,
        ...(find !== undefined ? { found: start >= 0 } : {}),
      }
      return {
        status: 'success',
        durationMs: Date.now() - started,
        output: `${JSON.stringify(header)}\nHistorical captured output, not a fresh execution or validation:\n${start < 0 ? '(literal not found)' : result.output.slice(start, end)}`,
      }
    },
  }
}
