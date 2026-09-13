import type { ChatRequest } from '@sepilotd/core'

/** Opt-in policy. Explicit levels are never escalated or silently lowered. */
export function resolveThinkingLevel(
  requested: string | undefined,
  context: { phase?: string; toolNames?: string[]; executionFailure?: boolean } = {},
): ChatRequest['thinkingLevel'] {
  if (requested !== 'auto') return requested as ChatRequest['thinkingLevel']
  const phase = context.phase ?? ''
  let level = 'medium'
  if (/final|conclusion|report/.test(phase)) level = 'low'
  else if (context.executionFailure || /review|diagnos/.test(phase)) level = 'high'
  else if (context.toolNames?.length && context.toolNames.every((name) =>
    ['fs.read', 'fs.list', 'fs.glob', 'fs.search', 'git.status', 'git.diff'].includes(name))) level = 'low'
  return level as ChatRequest['thinkingLevel']
}

/** No provider receives the daemon-only 'auto' policy as a wire value. */
export function resolveRequestThinking(request: ChatRequest): ChatRequest {
  return request.thinkingLevel === 'auto'
    ? { ...request, thinkingLevel: resolveThinkingLevel('auto', { toolNames: request.tools?.map((t) => t.name) }) }
    : request
}
