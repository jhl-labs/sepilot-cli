// Bundles the two single-state slots that the provider setup wizard and
// the provider delete confirmation share. Each slot is a single object
// (or null when closed); the open flag is derived from `!== null`.
//
// The state shapes (ProviderSetupState, ProviderDeleteConfirmState) live
// in App.tsx because they reference ProviderSetupStep, ProviderWizard
// preset types, and ConfiguredProviderRecord — moving them here would
// either pull a chain of types across or require a separate types module.
// The generic parameters keep this file shape-agnostic so the hook does
// not need to know about any of those.

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
