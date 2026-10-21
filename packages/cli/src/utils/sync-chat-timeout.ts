export const DEFAULT_CLI_SYNC_CHAT_TIMEOUT_MS = 15 * 60_000

export function resolveCliSyncChatTimeoutMs(
  raw = process.env.SEPILOTD_SYNC_CHAT_TIMEOUT_MS,
): number {
  const parsed = Number.parseInt(raw ?? String(DEFAULT_CLI_SYNC_CHAT_TIMEOUT_MS), 10)
  return Number.isFinite(parsed) && parsed > 0
    ? parsed
    : DEFAULT_CLI_SYNC_CHAT_TIMEOUT_MS
}
