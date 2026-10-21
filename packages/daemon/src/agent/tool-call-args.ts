export interface ParsedToolCallArguments {
  ok: boolean
  truncated: boolean
  arguments: Record<string, unknown>
}

export function parseToolCallArguments(
  argsJson: string | null | undefined,
): ParsedToolCallArguments {
  if (argsJson == null || argsJson.trim() === '') {
    return { ok: true, truncated: false, arguments: {} }
  }
  try {
    const parsed = JSON.parse(argsJson) as unknown
    if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return { ok: true, truncated: false, arguments: parsed as Record<string, unknown> }
    }
    return { ok: false, truncated: true, arguments: {} }
  } catch {
    return { ok: false, truncated: true, arguments: {} }
  }
}

export function readLengthContinuationMax(): number {
  const raw = process.env.SEPILOTD_LENGTH_CONTINUE_MAX
  if (raw == null || raw.trim() === '') return 1
  const n = Number(raw)
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : 1
}
