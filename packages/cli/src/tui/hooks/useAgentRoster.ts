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
