import type { DaemonDailyUsageSummary, DaemonUsageSummary } from '@sepilotd/api-client'
import { useInput } from 'ink'
import { UsageDashboard } from '../../components/UsageDashboard.js'

export interface UsageDialogProps {
  summary: DaemonUsageSummary | null
  daily: DaemonDailyUsageSummary[]
  days: number
  loading: boolean
  error: string | null
  height: number
  onClose: () => void
}

export function UsageDialog(props: UsageDialogProps) {
  useInput((input, key) => {
    if (key.escape || input === 'q') props.onClose()
  })

  return (
    <UsageDashboard
      summary={props.summary}
      daily={props.daily}
      days={props.days}
      loading={props.loading}
      error={props.error}
      height={props.height}
      closeHint="Esc/q close"
    />
  )
}
