/**
 * Synchronous JSON chat has no SSE heartbeat to refresh an idle deadline, so
 * it needs a bounded wall-clock envelope that is distinct from the generic
 * 30-second HTTP request timeout. Keep the default aligned with the CLI's
 * long-running stream envelope while allowing automation operators to tune it
 * independently.
 */
export const DEFAULT_CLI_SYNC_CHAT_TIMEOUT_MS = 15 * 60_000

export function resolveCliSyncChatTimeoutMs(
  raw = process.env.SEPILOTD_SYNC_CHAT_TIMEOUT_MS,
): number {
  const parsed = Number.parseInt(raw ?? String(DEFAULT_CLI_SYNC_CHAT_TIMEOUT_MS), 10)
  return Number.isFinite(parsed) && parsed > 0
    ? parsed
    : DEFAULT_CLI_SYNC_CHAT_TIMEOUT_MS
}
