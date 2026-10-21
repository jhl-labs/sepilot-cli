import { useState } from 'react'
import { createSwarmClient, type SwarmClient } from '@sepilotd/api-client'

export interface UseSwarmRunResult {
  swarmClient: SwarmClient
  lastSwarmRunId: string | null
  setLastSwarmRunId: React.Dispatch<React.SetStateAction<string | null>>
}

export function useSwarmRun(opts: {
  baseUrl: string
  token: string | null
}): UseSwarmRunResult {
  const [swarmClient] = useState<SwarmClient>(() =>
    createSwarmClient({ baseUrl: opts.baseUrl, token: opts.token }),
  )
  const [lastSwarmRunId, setLastSwarmRunId] = useState<string | null>(null)
  return { swarmClient, lastSwarmRunId, setLastSwarmRunId }
}
