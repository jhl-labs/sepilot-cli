import type { Message, ToolCall } from '@sepilotd/core'

/** Stop replaying clicks after two complete cycles with no observed change. */
export function repeatedBrowserClickWithoutProgress(messages: Message[]): boolean {
  const start = messages.findLastIndex(message => message.role === 'user')
  const calls = new Map<string, ToolCall>()
  const results: { call: ToolCall; output: string }[] = []
  for (const message of messages.slice(start + 1)) {
    for (const call of message.toolCalls ?? []) calls.set(call.id, call)
    if (message.role !== 'tool') continue
    const call = message.toolCallId ? calls.get(message.toolCallId) : undefined
    if (!call || message.metadata?.toolResultStatus !== 'success' || typeof message.content !== 'string') {
      results.length = 0
      continue
    }
    results.push({ call, output: message.content })
  }
  const tail = results.slice(-5)
  if (tail.length !== 5) return false
  const [before, first, middle, second, after] = tail
  if (![before, middle, after].every(result => result.call.name === 'browser.remote_snapshot')) return false
  if (before.output !== middle.output || middle.output !== after.output) return false
  const a = first.call.arguments
  const b = second.call.arguments
  return first.call.name === 'browser.remote_action' && second.call.name === 'browser.remote_action'
    && a.action === 'click' && b.action === 'click'
    && a.expectedUrl === b.expectedUrl && a.x === b.x && a.y === b.y
}
