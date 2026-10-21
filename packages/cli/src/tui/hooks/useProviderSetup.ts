import { useCallback, useState } from 'react'

export interface UseProviderSetupResult<TSetup, TDelete> {
  providerSetup: TSetup | null
  setProviderSetup: React.Dispatch<React.SetStateAction<TSetup | null>>
  providerDeleteConfirm: TDelete | null
  setProviderDeleteConfirm: React.Dispatch<React.SetStateAction<TDelete | null>>
  providerSetupOpen: boolean
  providerDeleteOpen: boolean
  closeProviderSetup: () => void
  closeProviderDeleteConfirm: () => void
}

export function useProviderSetup<TSetup, TDelete>(): UseProviderSetupResult<
  TSetup,
  TDelete
> {
  const [providerSetup, setProviderSetup] = useState<TSetup | null>(null)
  const [providerDeleteConfirm, setProviderDeleteConfirm] =
    useState<TDelete | null>(null)

  const closeProviderSetup = useCallback(() => {
    setProviderSetup(null)
  }, [])

  const closeProviderDeleteConfirm = useCallback(() => {
    setProviderDeleteConfirm(null)
  }, [])

  return {
    providerSetup,
    setProviderSetup,
    providerDeleteConfirm,
    setProviderDeleteConfirm,
    providerSetupOpen: providerSetup !== null,
    providerDeleteOpen: providerDeleteConfirm !== null,
    closeProviderSetup,
    closeProviderDeleteConfirm,
  }
}
