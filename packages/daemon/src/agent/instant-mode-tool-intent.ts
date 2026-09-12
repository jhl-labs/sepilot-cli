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

/** Mode projection can only narrow an explicit caller tool boundary. */
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
  // Keep session-scoped observation reachable after disconnect too, so the
  // tool can return its structured connection/control prerequisite. This does
  // not attach a tab or authorize mutation; explicit tool ceilings still win.
  const names: readonly string[] = [...INSTANT_PERSONAL_TOOL_NAMES, 'browser.remote_snapshot']
  return names.filter((name) => !permitted || permitted.has(name))
}
