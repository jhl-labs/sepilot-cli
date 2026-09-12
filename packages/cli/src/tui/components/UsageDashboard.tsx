import { useMemo } from 'react'
import type {
  DaemonDailyUsageSummary,
  DaemonUsageSummary,
} from '@sepilotd/api-client'
import { Box, Text } from 'ink'
import { colors, symbols } from '../theme.js'
import {
  aggregateDailyUsage,
  aggregateUsageModels,
  formatUsageCount,
  usageBar,
} from '../utils/usage.js'

interface UsageDashboardProps {
  summary: DaemonUsageSummary | null
  daily: DaemonDailyUsageSummary[]
  days: number
  loading: boolean
  error: string | null
  height: number
  closeHint?: string
}

export function UsageDashboard({
  summary,
  daily,
  days,
  loading,
  error,
  height,
  closeHint = '/usage close',
}: UsageDashboardProps) {
  const buckets = useMemo(() => aggregateDailyUsage(daily), [daily])
  const modelBreakdown = useMemo(
    () => aggregateUsageModels(daily).slice(0, 3),
    [daily],
  )
  const reservedRows = 1 + (summary ? 1 : 0) + (modelBreakdown.length > 0 ? 1 : 0)
  const visibleBuckets = useMemo(
    () => buckets.slice(0, Math.max(0, height - reservedRows)),
    [buckets, height, reservedRows],
  )
  const maxCostUsd = useMemo(
    () => visibleBuckets.reduce(
      (max, bucket) => Math.max(max, bucket.costUsd),
      0,
    ),
    [visibleBuckets],
  )

  if (height < 4) {
    return null
  }

  return (
    <Box
      flexDirection="column"
      height={height}
      borderStyle="single"
      borderColor={colors.border}
      paddingX={1}
      marginBottom={1}
      overflow="hidden"
      width="100%"
      minWidth={0}
    >
      <Box justifyContent="space-between" width="100%" minWidth={0}>
        <Text color={colors.info} bold>
          Usage Dashboard
        </Text>
        <Box flexShrink={0} marginLeft={1}>
          <Text color={colors.dimText}>
            {days}d {symbols.separator} {closeHint}
          </Text>
        </Box>
      </Box>
      {summary && (
        <Text color={colors.muted} wrap="truncate-end">
          in {formatUsageCount(summary.inputTokens)} {symbols.separator} out {formatUsageCount(summary.outputTokens)} {symbols.separator} cost ${summary.costUsd.toFixed(4)} {symbols.separator} req {summary.requestCount.toLocaleString()}
        </Text>
      )}
      {modelBreakdown.length > 0 && (
        <Text color={colors.dimText} wrap="truncate-end">
          models {symbols.separator} {modelBreakdown.map((item) => (
            `${item.provider}/${item.model} $${item.costUsd.toFixed(4)}`
          )).join(` ${symbols.separator} `)}
        </Text>
      )}
      {loading && (
        <Text color={colors.dimText}>Loading usage data...</Text>
      )}
      {error && (
        <Text color={colors.error}>{error}</Text>
      )}
      {!loading && !error && visibleBuckets.length === 0 && (
        <Text color={colors.dimText}>No daily usage yet.</Text>
      )}
      {!loading && !error && visibleBuckets.map((bucket) => (
        <Text key={bucket.date} wrap="truncate-end">
          <Text color={colors.text}>{bucket.date}</Text>
          <Text color={colors.dimText}> {symbols.separator} {usageBar(bucket.costUsd, maxCostUsd, 10)} </Text>
          <Text color={colors.info}>${bucket.costUsd.toFixed(4)}</Text>
          <Text color={colors.dimText}> {symbols.separator} {formatUsageCount(bucket.inputTokens + bucket.outputTokens)} tok </Text>
          <Text color={colors.dimText}>{symbols.separator} {bucket.requestCount} req {symbols.separator} {bucket.topProvider}/{bucket.topModel}</Text>
        </Text>
      ))}
    </Box>
  )
}
