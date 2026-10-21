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
