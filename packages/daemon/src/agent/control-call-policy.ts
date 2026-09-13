/**
 * Total control-transaction ceiling, separate from provider first-token and
 * active-stream idle guards. Native + JSON repair share this transaction.
 * An operator may disable it explicitly; run/call-count/output budgets and
 * cancellation remain independent. Never derive it from the first-token knob.
 */
export const DEFAULT_CONTROL_CALL_TIMEOUT_MS = 5 * 60_000

export function resolveControlCallTimeoutMs(
  raw = process.env.SEPILOTD_CONTROL_CALL_TIMEOUT_MS,
): number | null {
  if (raw === undefined || raw.trim() === '') return DEFAULT_CONTROL_CALL_TIMEOUT_MS
  const value = Number(raw)
  if (value === 0) return null
  return Number.isSafeInteger(value) && value > 0 && value <= 2_147_483_647
    ? value : DEFAULT_CONTROL_CALL_TIMEOUT_MS
}

export function resolveCallDeadline(
  explicit: number | null | undefined,
  fallback: number | null,
  remaining: number | null,
): number | null {
  const call = explicit === undefined ? fallback : explicit
  return call === null ? remaining : remaining === null ? call : Math.min(call, remaining)
}
