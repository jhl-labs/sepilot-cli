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
