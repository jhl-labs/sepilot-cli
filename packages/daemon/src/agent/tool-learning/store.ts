export interface ToolStatsRecord {
  tool: string
  successCount: number
  errorCount: number
  totalCount: number
  successRate: number
  avgDurationMs: number
  recentErrorOutputs: string[]
}

interface InternalRecord {
  successCount: number
  errorCount: number
  durationSumMs: number
  recentErrorOutputs: string[]
}

const RECENT_ERRORS_LIMIT = 3

/**
 * Outcome of one tool execution. Carries identity and timing only — never the
 * tool output, which can hold user content and has no place in operational
 * telemetry.
 */
export interface ToolOutcomeObservation {
  sessionId: string
  tool: string
  status: 'success' | 'error'
  durationMs: number
}

export class ToolStatsStore {
  private readonly bySession = new Map<string, Map<string, InternalRecord>>()

  /**
   * Every tool execution already converges on `record`, so an observer here
   * sees the complete outcome stream without threading a second dependency
   * through the router/engine/graph layers.
   */
  constructor(private readonly onOutcome?: (outcome: ToolOutcomeObservation) => void) {}

  record(
    sessionId: string,
    tool: string,
    result: { status: 'success' | 'error'; output?: string; durationMs?: number },
  ): void {
    let bucket = this.bySession.get(sessionId)
    if (!bucket) {
      bucket = new Map()
      this.bySession.set(sessionId, bucket)
    }
    let record = bucket.get(tool)
    if (!record) {
      record = {
        successCount: 0,
        errorCount: 0,
        durationSumMs: 0,
        recentErrorOutputs: [],
      }
      bucket.set(tool, record)
    }
    if (result.status === 'success') {
      record.successCount += 1
    } else {
      record.errorCount += 1
      if (result.output) {
        record.recentErrorOutputs.push(result.output.slice(0, 200))
        if (record.recentErrorOutputs.length > RECENT_ERRORS_LIMIT) {
          record.recentErrorOutputs.shift()
        }
      }
    }
    record.durationSumMs += result.durationMs ?? 0

    // Telemetry is best-effort: a failing observer must never turn a completed
    // tool execution into a failed one.
    try {
      this.onOutcome?.({
        sessionId,
        tool,
        status: result.status,
        durationMs: result.durationMs ?? 0,
      })
    } catch {
      // ignored
    }
  }

  list(sessionId: string): ToolStatsRecord[] {
    const bucket = this.bySession.get(sessionId)
    if (!bucket) return []
    return [...bucket.entries()].map(([tool, record]) => toExternal(tool, record))
  }

  get(sessionId: string, tool: string): ToolStatsRecord | undefined {
    const record = this.bySession.get(sessionId)?.get(tool)
    return record ? toExternal(tool, record) : undefined
  }

  dispose(sessionId: string): void {
    this.bySession.delete(sessionId)
  }
}

function toExternal(tool: string, record: InternalRecord): ToolStatsRecord {
  const total = record.successCount + record.errorCount
  return {
    tool,
    successCount: record.successCount,
    errorCount: record.errorCount,
    totalCount: total,
    successRate: total > 0 ? record.successCount / total : 1,
    avgDurationMs: total > 0 ? record.durationSumMs / total : 0,
    recentErrorOutputs: [...record.recentErrorOutputs],
  }
}

export function summarizeProblemTools(
  records: ToolStatsRecord[],
  options: { minCalls?: number; threshold?: number; limit?: number } = {},
): ToolStatsRecord[] {
  const minCalls = options.minCalls ?? 2
  const threshold = options.threshold ?? 0.5
  const limit = options.limit ?? 4
  return records
    .filter((r) => r.totalCount >= minCalls && r.successRate < threshold)
    .sort((a, b) => a.successRate - b.successRate)
    .slice(0, limit)
}
