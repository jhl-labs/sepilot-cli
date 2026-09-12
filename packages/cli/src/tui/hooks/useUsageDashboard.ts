// Bundles together the six useState, the request-ref, the derived buckets
// memo, and the close handler that the usage dashboard overlay needs.
//
// Why a hook and not a fully extracted component yet: loadUsageDashboard()
// in App.tsx still has cross-overlay coupling (it has to close the memory
// search panel before opening, see the mutual-exclusion comment there).
// Pulling that cross-cutting orchestration out is the next step. For now
// this hook just collapses the per-state plumbing and makes it possible
// to re-use the derived buckets / close behaviour from another component
// without copy/pasting six pieces of state.

import { useCallback, useMemo, useRef, useState } from 'react'
import type {
  DaemonDailyUsageSummary,
  DaemonUsageSummary,
} from '@sepilotd/api-client'
import { aggregateDailyUsage } from '../utils/usage.js'

export const USAGE_DASHBOARD_DEFAULT_DAYS = 7
export const USAGE_DASHBOARD_MAX_DAYS = 365

export interface UseUsageDashboardResult {
  usageDashboardOpen: boolean
  setUsageDashboardOpen: React.Dispatch<React.SetStateAction<boolean>>
  usageDashboardDays: number
  setUsageDashboardDays: React.Dispatch<React.SetStateAction<number>>
  usageDashboardLoading: boolean
  setUsageDashboardLoading: React.Dispatch<React.SetStateAction<boolean>>
  usageDashboardError: string | null
  setUsageDashboardError: React.Dispatch<React.SetStateAction<string | null>>
  usageSummary: DaemonUsageSummary | null
  setUsageSummary: React.Dispatch<
    React.SetStateAction<DaemonUsageSummary | null>
  >
  usageDaily: DaemonDailyUsageSummary[]
  setUsageDaily: React.Dispatch<
    React.SetStateAction<DaemonDailyUsageSummary[]>
  >
  usageRequestRef: React.MutableRefObject<number>
  usageBuckets: ReturnType<typeof aggregateDailyUsage>
  closeUsageDashboard: () => void
}

export function useUsageDashboard(): UseUsageDashboardResult {
  const [usageDashboardOpen, setUsageDashboardOpen] = useState(false)
  const [usageDashboardDays, setUsageDashboardDays] = useState<number>(
    USAGE_DASHBOARD_DEFAULT_DAYS,
  )
  const [usageDashboardLoading, setUsageDashboardLoading] = useState(false)
  const [usageDashboardError, setUsageDashboardError] = useState<string | null>(
    null,
  )
  const [usageSummary, setUsageSummary] = useState<DaemonUsageSummary | null>(
    null,
  )
  const [usageDaily, setUsageDaily] = useState<DaemonDailyUsageSummary[]>([])
  const usageRequestRef = useRef(0)

  const usageBuckets = useMemo(
    () => aggregateDailyUsage(usageDaily),
    [usageDaily],
  )

  // Bumping usageRequestRef invalidates any in-flight loadUsageDashboard()
  // call so a late-arriving response can no longer reopen / re-render the
  // panel after the user has dismissed it.
  const closeUsageDashboard = useCallback(() => {
    usageRequestRef.current += 1
    setUsageDashboardOpen(false)
    setUsageDashboardLoading(false)
    setUsageDashboardError(null)
  }, [])

  return {
    usageDashboardOpen,
    setUsageDashboardOpen,
    usageDashboardDays,
    setUsageDashboardDays,
    usageDashboardLoading,
    setUsageDashboardLoading,
    usageDashboardError,
    setUsageDashboardError,
    usageSummary,
    setUsageSummary,
    usageDaily,
    setUsageDaily,
    usageRequestRef,
    usageBuckets,
    closeUsageDashboard,
  }
}
