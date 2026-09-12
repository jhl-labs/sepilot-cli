// Two pieces of swarm-run state App.tsx kept inline next to the
// daemon clients:
//
// - swarmClient — singleton SwarmClient bound to the same daemon
//   baseUrl + token as the http client. Created once per session;
//   never reassigned.
// - lastSwarmRunId — id of the most recently started or referenced
//   /swarm run, used as the default when the operator omits <runId>
//   on follow-up subcommands.
//
// The hook accepts the construction inputs so the swarm client itself
// is created lazily (the same way App.tsx did with useState's lazy
// initialiser).

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
