// Bundles the three pieces of provider-registry state App.tsx needs
// at the top level: the configured provider list, the daemon-wide
// default provider id, and the daemon-wide default model id. The
// model picker / provider commands all read these together; collecting
// them in a hook means refreshing the registry (after an /update or a
// provider setup) is one destructured caller instead of three useState
// declarations split across the component body.
//
// providersLoading / providersError live in useModelPicker because
// they're tied to the picker overlay's loading state, not the registry
// itself.

import { useState } from 'react'
import type { DaemonProviderInfo } from '@sepilotd/api-client'

export interface UseProviderRegistryResult {
  providers: DaemonProviderInfo[]
  setProviders: React.Dispatch<React.SetStateAction<DaemonProviderInfo[]>>
  daemonDefaultProvider: string
  setDaemonDefaultProvider: React.Dispatch<React.SetStateAction<string>>
  daemonDefaultModel: string
  setDaemonDefaultModel: React.Dispatch<React.SetStateAction<string>>
}

export function useProviderRegistry(): UseProviderRegistryResult {
  const [providers, setProviders] = useState<DaemonProviderInfo[]>([])
  const [daemonDefaultProvider, setDaemonDefaultProvider] = useState('')
  const [daemonDefaultModel, setDaemonDefaultModel] = useState('')

  return {
    providers,
    setProviders,
    daemonDefaultProvider,
    setDaemonDefaultProvider,
    daemonDefaultModel,
    setDaemonDefaultModel,
  }
}
