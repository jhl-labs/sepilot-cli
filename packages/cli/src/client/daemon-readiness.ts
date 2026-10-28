const DEFAULT_DAEMON_READY_TIMEOUT_MS = 60_000
const MIN_DAEMON_READY_TIMEOUT_MS = 5_000
const MAX_DAEMON_READY_TIMEOUT_MS = 300_000

export function resolveDaemonReadyTimeoutMs(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.SEPILOT_DAEMON_READY_MS?.trim()
  if (!raw) return DEFAULT_DAEMON_READY_TIMEOUT_MS
  const parsed = Number(raw)
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
    return DEFAULT_DAEMON_READY_TIMEOUT_MS
  }
  return Math.max(MIN_DAEMON_READY_TIMEOUT_MS, Math.min(MAX_DAEMON_READY_TIMEOUT_MS, parsed))
}
