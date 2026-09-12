export interface McpPerToolEntry {
  calls: number
  errors: number
  totalDurationMs: number
}

export interface McpMetricsEntry {
  totalCalls: number
  errors: number
  totalDurationMs: number
  lastCallAt: string | null
  perTool: Record<string, McpPerToolEntry>
}

export interface McpMetricsSnapshot {
  servers: Record<string, McpMetricsEntry>
}

export function emptyEntry(): McpMetricsEntry {
  return {
    totalCalls: 0,
    errors: 0,
    totalDurationMs: 0,
    lastCallAt: null,
    perTool: {},
  }
}

export function recordCallInto(
  entry: McpMetricsEntry,
  tool: string,
  durationMs: number,
  status: 'success' | 'error',
): void {
  entry.totalCalls += 1
  if (status === 'error') entry.errors += 1
  entry.totalDurationMs += durationMs
  entry.lastCallAt = new Date().toISOString()
  const per = entry.perTool[tool] ?? { calls: 0, errors: 0, totalDurationMs: 0 }
  per.calls += 1
  if (status === 'error') per.errors += 1
  per.totalDurationMs += durationMs
  entry.perTool[tool] = per
}

export function snapshotEntry(entry: McpMetricsEntry): McpMetricsEntry {
  return {
    totalCalls: entry.totalCalls,
    errors: entry.errors,
    totalDurationMs: entry.totalDurationMs,
    lastCallAt: entry.lastCallAt,
    perTool: Object.fromEntries(
      Object.entries(entry.perTool).map(([k, v]) => [k, { ...v }]),
    ),
  }
}
