import type { DaemonDailyUsageSummary } from '@sepilotd/api-client'

export interface DailyUsageBucket {
  date: string
  inputTokens: number
  outputTokens: number
  costUsd: number
  requestCount: number
  topProvider: string
  topModel: string
  entries: DaemonDailyUsageSummary[]
}

export interface UsageModelBreakdown {
  provider: string
  model: string
  inputTokens: number
  outputTokens: number
  costUsd: number
  requestCount: number
}

function compareUsageRows(
  left: Pick<
    DaemonDailyUsageSummary,
    'totalCostUsd' | 'requestCount' | 'totalInputTokens' | 'totalOutputTokens'
  >,
  right: Pick<
    DaemonDailyUsageSummary,
    'totalCostUsd' | 'requestCount' | 'totalInputTokens' | 'totalOutputTokens'
  >,
): number {
  if (right.totalCostUsd !== left.totalCostUsd) {
    return right.totalCostUsd - left.totalCostUsd
  }
  if (right.requestCount !== left.requestCount) {
    return right.requestCount - left.requestCount
  }
  return (
    (right.totalInputTokens + right.totalOutputTokens)
    - (left.totalInputTokens + left.totalOutputTokens)
  )
}

export function aggregateDailyUsage(
  rows: DaemonDailyUsageSummary[],
): DailyUsageBucket[] {
  const buckets = new Map<string, DailyUsageBucket>()

  for (const row of rows) {
    const current = buckets.get(row.date)
    if (!current) {
      buckets.set(row.date, {
        date: row.date,
        inputTokens: row.totalInputTokens,
        outputTokens: row.totalOutputTokens,
        costUsd: row.totalCostUsd,
        requestCount: row.requestCount,
        topProvider: row.provider,
        topModel: row.model,
        entries: [row],
      })
      continue
    }

    current.inputTokens += row.totalInputTokens
    current.outputTokens += row.totalOutputTokens
    current.costUsd += row.totalCostUsd
    current.requestCount += row.requestCount
    current.entries.push(row)

    const currentTopEntry = current.entries
      .slice()
      .sort(compareUsageRows)[0]

    if (currentTopEntry) {
      current.topProvider = currentTopEntry.provider
      current.topModel = currentTopEntry.model
    }
  }

  return [...buckets.values()].sort((left, right) => (
    right.date.localeCompare(left.date)
  ))
}

export function aggregateUsageModels(
  rows: DaemonDailyUsageSummary[],
): UsageModelBreakdown[] {
  const models = new Map<string, UsageModelBreakdown>()

  for (const row of rows) {
    const key = `${row.provider}:${row.model}`
    const current = models.get(key)
    if (!current) {
      models.set(key, {
        provider: row.provider,
        model: row.model,
        inputTokens: row.totalInputTokens,
        outputTokens: row.totalOutputTokens,
        costUsd: row.totalCostUsd,
        requestCount: row.requestCount,
      })
      continue
    }

    current.inputTokens += row.totalInputTokens
    current.outputTokens += row.totalOutputTokens
    current.costUsd += row.totalCostUsd
    current.requestCount += row.requestCount
  }

  return [...models.values()].sort((left, right) => {
    if (right.costUsd !== left.costUsd) {
      return right.costUsd - left.costUsd
    }
    if (right.requestCount !== left.requestCount) {
      return right.requestCount - left.requestCount
    }
    return (
      (right.inputTokens + right.outputTokens)
      - (left.inputTokens + left.outputTokens)
    )
  })
}

export function formatUsageCount(value: number): string {
  if (value >= 1_000_000) return formatUsageCompact(value / 1_000_000, 'M')
  if (value >= 1_000) return formatUsageCompact(value / 1_000, 'k')
  return String(value)
}

function formatUsageCompact(value: number, suffix: string): string {
  const compact = value.toFixed(1)
  const normalized = compact.endsWith('.0')
    ? compact.slice(0, -2)
    : compact
  return `${normalized}${suffix}`
}

export function usageBar(
  value: number,
  maxValue: number,
  width = 12,
): string {
  if (width <= 0 || maxValue <= 0 || value <= 0) {
    return '-'.repeat(Math.max(0, width))
  }

  const filled = Math.max(1, Math.round((value / maxValue) * width))
  return `${'#'.repeat(filled)}${'-'.repeat(Math.max(0, width - filled))}`
}
