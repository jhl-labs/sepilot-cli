/**
 * Retention sweeper.
 *
 * Long-lived installs previously grew every store without bound — sessions,
 * derived memory, and usage rows only ever accumulated (only resume artifacts
 * aged out). This sweeper deletes items older than the configured window. It is
 * intentionally store-agnostic: callers inject small delete functions so the
 * unit is testable with fakes and does not import the route/runtime layer.
 *
 * Session deletion MUST route through the complete-deletion path (see
 * `cleanupSessionRuntimeState`) so aged-out sessions leave no orphaned
 * usage/memory/embeddings/uploads.
 */

export interface RetentionSweepConfig {
  enabled?: boolean
  sessionDays?: number
  memoryDays?: number
  usageDays?: number
}

export interface RetentionSweepDeps {
  now?: () => Date
  /** All known sessions with their last-activity timestamp (ISO). */
  listSessions?: () => Promise<Array<{ id: string; updatedAt: string }>>
  /** Complete deletion for one session (cleanup + store delete). */
  deleteSession?: (id: string) => Promise<void>
  /** Delete usage rows older than an ISO cutoff; returns rows removed. */
  deleteUsageOlderThan?: (cutoffIso: string) => number | Promise<number>
  /** Delete memory rows older than an ISO cutoff; returns rows removed. */
  deleteMemoriesOlderThan?: (cutoffIso: string) => number | Promise<number>
  logger?: {
    warn: (message: string, data?: Record<string, unknown>) => void
    info: (message: string, data?: Record<string, unknown>) => void
  }
}

export interface RetentionSweepResult {
  sessions: number
  memories: number
  usage: number
  skipped: boolean
}

function cutoffIso(now: Date, days: number): string {
  return new Date(now.getTime() - days * 24 * 60 * 60 * 1000).toISOString()
}

/** Whether retention sweeping is turned off (config flag or env kill-switch). */
export function isRetentionSweepDisabled(config: RetentionSweepConfig): boolean {
  if (process.env.SEPILOTD_RETENTION_SWEEP === '0') return true
  return config.enabled === false
}

export async function runRetentionSweep(
  deps: RetentionSweepDeps,
  config: RetentionSweepConfig,
): Promise<RetentionSweepResult> {
  const result: RetentionSweepResult = { sessions: 0, memories: 0, usage: 0, skipped: false }
  if (isRetentionSweepDisabled(config)) {
    result.skipped = true
    return result
  }
  const now = deps.now?.() ?? new Date()

  // Sessions — complete deletion for anything past the window.
  if (config.sessionDays && deps.listSessions && deps.deleteSession) {
    const cutoff = cutoffIso(now, config.sessionDays)
    let sessions: Array<{ id: string; updatedAt: string }> = []
    try {
      sessions = await deps.listSessions()
    } catch (error) {
      deps.logger?.warn?.('retention.list_sessions_failed', { err: String(error) })
    }
    for (const session of sessions) {
      if (!session.updatedAt || session.updatedAt >= cutoff) continue
      try {
        await deps.deleteSession(session.id)
        result.sessions += 1
      } catch (error) {
        deps.logger?.warn?.('retention.delete_session_failed', { id: session.id, err: String(error) })
      }
    }
  }

  // Memory — age out rows past the window.
  if (config.memoryDays && deps.deleteMemoriesOlderThan) {
    try {
      result.memories = await deps.deleteMemoriesOlderThan(cutoffIso(now, config.memoryDays))
    } catch (error) {
      deps.logger?.warn?.('retention.delete_memories_failed', { err: String(error) })
    }
  }

  // Usage — age out rows past the window.
  if (config.usageDays && deps.deleteUsageOlderThan) {
    try {
      result.usage = await deps.deleteUsageOlderThan(cutoffIso(now, config.usageDays))
    } catch (error) {
      deps.logger?.warn?.('retention.delete_usage_failed', { err: String(error) })
    }
  }

  if (result.sessions + result.memories + result.usage > 0) {
    deps.logger?.info?.('retention.sweep', {
      sessions: result.sessions,
      memories: result.memories,
      usage: result.usage,
    })
  }
  return result
}

/**
 * Start a periodic retention sweep. Returns a stop function. The timer is
 * unref'd so it never keeps the process alive on its own. A sweep already in
 * flight is not overlapped.
 */
export function startRetentionSweeper(
  deps: RetentionSweepDeps,
  config: RetentionSweepConfig,
  intervalMs: number,
): () => void {
  if (isRetentionSweepDisabled(config)) {
    return () => {}
  }
  let running = false
  const tick = async (): Promise<void> => {
    if (running) return
    running = true
    try {
      await runRetentionSweep(deps, config)
    } finally {
      running = false
    }
  }
  const timer = setInterval(() => {
    void tick()
  }, intervalMs)
  timer.unref?.()
  return () => clearInterval(timer)
}
