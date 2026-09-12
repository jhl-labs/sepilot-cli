export interface SanitizeOptions {
  home: string
}

const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g
const BEARER_RE = /Bearer\s+[A-Za-z0-9._\-]+/g
const SK_TOKEN_RE = /sk-[A-Za-z0-9_\-]{12,}/g
const GITHUB_TOKEN_RE = /gh[posur]_[A-Za-z0-9]{20,}/g
const AWS_ACCESS_KEY_ID_RE = /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/g
const JWT_RE = /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/g
const PEM_PRIVATE_KEY_RE =
  /-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----/g
const CREDENTIAL_ASSIGNMENT_RE =
  /\b([A-Za-z0-9_.-]*(?:secret(?!s)|token(?!s)|password|passwd|pwd|api[_-]?key|access[_-]?key|private[_-]?key)[A-Za-z0-9_.-]*\s*[:=]\s*)(["']?)([^\s"',;`]+)(\2)/gi

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export function sanitizeText(input: string, opts: SanitizeOptions): string {
  let out = input
  if (opts.home) {
    out = out.replace(new RegExp(escapeRegExp(opts.home), 'g'), '<HOME>')
  }
  return out
    .replace(PEM_PRIVATE_KEY_RE, '<private-key>')
    .replace(
      CREDENTIAL_ASSIGNMENT_RE,
      (_match, prefix: string, quote: string) => `${prefix}${quote}<secret>${quote}`,
    )
    .replace(EMAIL_RE, '<email>')
    .replace(BEARER_RE, 'Bearer <token>')
    .replace(SK_TOKEN_RE, '<token>')
    .replace(GITHUB_TOKEN_RE, '<token>')
    .replace(AWS_ACCESS_KEY_ID_RE, '<token>')
    .replace(JWT_RE, '<token>')
}

// Walk an object/array tree and run sanitizeText on every string leaf. Used
// for session-event sanitisation so secret-shaped strings cannot ride out via
// nested fields (tool_call.input, tool_result.output, cowork_*.result/summary,
// delegation_state.detail, provider_attempt.errorMessage, etc.) that the old
// content-only sanitiser was leaving untouched.
function sanitizeValueDeep(value: unknown, opts: SanitizeOptions): unknown {
  if (typeof value === 'string') return sanitizeText(value, opts)
  if (Array.isArray(value)) return value.map((v) => sanitizeValueDeep(v, opts))
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = sanitizeValueDeep(v, opts)
    }
    return out as unknown
  }
  return value
}

export function sanitizeEvent<T extends object>(event: T, opts: SanitizeOptions): T {
  return sanitizeValueDeep(event, opts) as T
}
