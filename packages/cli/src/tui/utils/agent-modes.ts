import type { DaemonAgentDescriptor, DaemonAgentMode } from '@sepilotd/api-client'

export function nextAgentModeId(
  current: DaemonAgentMode,
  agents: DaemonAgentDescriptor[],
): DaemonAgentMode | null {
  if (agents.length === 0) return null
  const idx = agents.findIndex((agent) => agent.id === current)
  const nextIdx = idx === -1 ? 0 : (idx + 1) % agents.length
  return agents[nextIdx]?.id ?? null
}
