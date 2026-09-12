// Bundles the two pieces of agent-roster state App.tsx kept inline:
//
// - primaryAgentId — which of plan / build the next chat turn defaults
//   to (Shift+Tab cycles)
// - agentModes — the list of agent mode descriptors the daemon advertises
//   for the model + tool-set combination, used by the mode picker and
//   the /mode command
//
// Loading/error for agentModes lives in useModePicker because it's
// tied to that overlay's loading state, not the roster itself.

import { useState } from 'react'
import type { DaemonAgentDescriptor } from '@sepilotd/api-client'

type PrimaryAgentId = 'plan' | 'build'

export interface UseAgentRosterResult {
  primaryAgentId: PrimaryAgentId
  setPrimaryAgentId: React.Dispatch<React.SetStateAction<PrimaryAgentId>>
  agentModes: DaemonAgentDescriptor[]
  setAgentModes: React.Dispatch<
    React.SetStateAction<DaemonAgentDescriptor[]>
  >
}

export function useAgentRoster(): UseAgentRosterResult {
  const [primaryAgentId, setPrimaryAgentId] = useState<PrimaryAgentId>('build')
  const [agentModes, setAgentModes] = useState<DaemonAgentDescriptor[]>([])

  return {
    primaryAgentId,
    setPrimaryAgentId,
    agentModes,
    setAgentModes,
  }
}
