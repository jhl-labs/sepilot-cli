const ANSI_RE = /\x1B(?:\[[0-9;?]*[A-Za-z]|\].*?(?:\x07|\x1B\\)|\([A-B]|[=>]|#[0-9]|[ -/]*[0-~])/g
const CODEX_DIM_PLACEHOLDER_RE = /^([^\r\n]*›[^\r\n]*?)\x1B\[(?:0;)?2(?:;[0-9]+)*m[^\r\n]*/gm

function stripCodexDimPlaceholders(raw: string): string {
  return raw.replace(CODEX_DIM_PLACEHOLDER_RE, '$1')
}

export function cleanAnsi(raw: string): string {
  return stripCodexDimPlaceholders(raw).replace(ANSI_RE, '')
}

export interface IncrementalSlice {
  text: string
  nextCursor: number
}

function commonPrefixLength(a: string, b: string): number {
  const max = Math.min(a.length, b.length)
  let i = 0
  while (i < max && a.charCodeAt(i) === b.charCodeAt(i)) i++
  return i
}

export function extractIncremental(cleaned: string, cursor: number): IncrementalSlice {
  const start = Math.max(0, Math.min(cursor, cleaned.length))
  return { text: cleaned.slice(start), nextCursor: cleaned.length }
}

export function extractIncrementalSince(previous: string, cleaned: string): IncrementalSlice {
  if (!previous) return { text: cleaned, nextCursor: cleaned.length }
  if (cleaned.startsWith(previous)) {
    return { text: cleaned.slice(previous.length), nextCursor: cleaned.length }
  }
  const start = commonPrefixLength(previous, cleaned)
  return { text: cleaned.slice(start), nextCursor: cleaned.length }
}
