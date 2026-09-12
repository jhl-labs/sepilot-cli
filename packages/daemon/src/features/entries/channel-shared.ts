export function readRateLimitPerMinute(
  channelConfig: Record<string, unknown> | undefined,
): number | undefined {
  return typeof channelConfig?.rateLimitPerMinute === 'number'
    ? channelConfig.rateLimitPerMinute
    : undefined
}
