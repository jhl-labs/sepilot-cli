import type { AgentExecutionIntent } from '@sepilotd/core'

export const INSTANT_MEMORY_TOOL_NAMES = [
  'memory.remember',
  'memory.update',
  'memory.search',
  'memory.list',
  'memory.forget',
  'memory.graph.search',
  'memory.graph.neighbors',
  'memory.graph.page',
] as const

export const INSTANT_PERSONAL_TOOL_NAMES = [
  ...INSTANT_MEMORY_TOOL_NAMES,
  'knowledge.search',
  'knowledge.save',
  'knowledge.read',
  'knowledge.edit',
  'knowledge.review',
] as const

export function resolveInstantModeToolNames(
  _input: string,
  effectiveMode: string | undefined,
  requestedToolNames: readonly string[] | undefined,
  _semanticExecutionIntent?: AgentExecutionIntent,
  _semanticMode?: string,
  _activeRemoteBrowser = false,
): readonly string[] | undefined {
  if (effectiveMode !== 'instant') return requestedToolNames
  const permitted = requestedToolNames === undefined ? undefined : new Set(requestedToolNames)
  const names: readonly string[] = [...INSTANT_PERSONAL_TOOL_NAMES, 'browser.remote_snapshot']
  return names.filter((name) => !permitted || permitted.has(name))
}
