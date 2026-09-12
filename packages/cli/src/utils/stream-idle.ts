export const DEFAULT_CLI_STREAM_IDLE_MS = 15 * 60_000

export function resolveCliStreamIdleMs(
  raw = process.env.SEPILOTD_STREAM_IDLE_MS,
): number {
  const parsed = Number.parseInt(raw ?? String(DEFAULT_CLI_STREAM_IDLE_MS), 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_CLI_STREAM_IDLE_MS
}
