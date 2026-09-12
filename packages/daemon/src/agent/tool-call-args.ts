// Shared parsing for streamed tool-call argument JSON.
//
// A tool call whose argument stream is a non-empty string that fails to parse
// is almost always TRUNCATED (the provider stopped mid-JSON, typically with
// finishReason==='length'). Executing such a call with `{}` silently discards
// the intended arguments — e.g. a `write_file` body — and causes data loss.
// Every streaming site funnels through this helper so that truncated calls are
// dropped and surfaced as a length signal rather than executed empty.

export interface ParsedToolCallArguments {
  // Whether argsJson yielded a usable object (or was legitimately empty).
  ok: boolean
  // Whether a non-empty args string failed to parse (suspected truncation).
  truncated: boolean
  arguments: Record<string, unknown>
}

export function parseToolCallArguments(
  argsJson: string | null | undefined,
): ParsedToolCallArguments {
  if (argsJson == null || argsJson.trim() === '') {
    // Legitimate no-argument tool call.
    return { ok: true, truncated: false, arguments: {} }
  }
  try {
    const parsed = JSON.parse(argsJson) as unknown
    if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return { ok: true, truncated: false, arguments: parsed as Record<string, unknown> }
    }
    // Parsed but not an object (bare number/string/array) — not valid tool args.
    return { ok: false, truncated: true, arguments: {} }
  } catch {
    // Non-empty args that do not parse — treat as a truncated tool call.
    return { ok: false, truncated: true, arguments: {} }
  }
}

// Bound on how many times a length-truncated turn may auto-continue.
// Default 1 keeps behaviour safe (one extra turn) while env allows tuning/off.
export function readLengthContinuationMax(): number {
  const raw = process.env.SEPILOTD_LENGTH_CONTINUE_MAX
  if (raw == null || raw.trim() === '') return 1
  const n = Number(raw)
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : 1
}
