import { isAbsolute, resolve } from 'node:path'
import { sepilotdHome } from '../storage/home.js'

export interface TraceRedactionInput {
  env?: NodeJS.ProcessEnv
  cwd?: string
  stateDir?: string
}

export interface TraceRedactionContext {
  env: NodeJS.ProcessEnv
  cwd: string
  stateDir: string
  pathAliases: Record<string, string>
  prefixes: Array<{ prefix: string; label: string; caseInsensitive: boolean }>
}

export interface RedactValueOptions {
  maxStringLength?: number
  maxDepth?: number
  maxArrayItems?: number
  maxObjectKeys?: number
}

export const TRACE_REDACTION_RULES = [
  'secret-like object keys are removed',
  'authorization, cookie, JWT, AWS key, URL userinfo, and token-like text are redacted',
  'email addresses are redacted',
  'state, home, and current-working-directory path prefixes are aliased',
  'strings, arrays, objects, and recursion depth are bounded',
]

const SECRET_KEY_RE =
  /(?:authorization|cookie|credential|key|password|passwd|secret|token|api[_-]?key|client[_-]?secret|private[_-]?key)/i
const PATH_KEY_RE =
  /(?:path|file|dir|cwd|root|home|workspace|stdout|stderr|log|logs|database|db)/i
const BEARER_RE = /\bBearer\s+[A-Za-z0-9._~+/=-]{8,}/giu
const BASIC_RE = /\bBasic\s+[A-Za-z0-9+/]+={0,2}/giu
const COOKIE_RE = /\b(?:Cookie|Set-Cookie)\s*:\s*[^\r\n]+/giu
const AWS_ACCESS_KEY_RE = /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/gu
const JWT_RE = /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/gu
const URL_USERINFO_RE = /\b([a-z][a-z0-9+.-]*:\/\/)([^/@\s:?#]+)(?::([^/@\s?#]+))?@/giu
const URL_PARAM_RE = /([?&])([^=&\s]+)=([^&#\s]+)/giu
const EMAIL_RE = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/giu
const NAMED_SECRET_RE =
  /\b(api[_-]?key|authorization|password|passwd|secret|token)(["']?\s*[:= ]\s*["']?)([^\s,;&"'}]{6,})/giu
const LONG_TOKEN_RE = /\b[A-Za-z0-9_./+=-]{32,}\b/gu
// A UUID is a correlation identifier — it joins an event to the tool call,
// run, or turn it belongs to — never a credential. LONG_TOKEN_RE matches a
// 36-character UUID, so without this carve-out every join key in a trace is
// rewritten to '[redacted-token]' and downstream consumers can no longer
// reconstruct causality. Credential shapes (opaque hex/base64 blobs, `sk-`
// style keys) are not UUID-shaped and stay redacted. Values under a
// secret-named key are handled earlier by NAMED_SECRET_RE and are unaffected.
const CORRELATION_UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu
const DEFAULT_MAX_STRING_LENGTH = 2_000
const DEFAULT_MAX_DEPTH = 8
const DEFAULT_MAX_ARRAY_ITEMS = 256
const DEFAULT_MAX_OBJECT_KEYS = 256
const BLOCKED_OBJECT_KEYS = new Set(['__proto__', 'prototype', 'constructor'])

export function isTraceRedactionEnabled(
  env: Record<string, string | undefined> = process.env,
): boolean {
  return env.SEPILOT_TRACE_REDACT !== '0'
}

function truncate(value: string, maxLength = DEFAULT_MAX_STRING_LENGTH): string {
  return value.length > maxLength
    ? `${value.slice(0, maxLength - 15)}...<truncated>`
    : value
}

function isWindowsAbsolutePath(value: string): boolean {
  return /^(?:[A-Za-z]:[\\/]|\\\\)/.test(value)
}

function normalizePrefix(value: string): string {
  return isWindowsAbsolutePath(value) ? value.replaceAll('\\', '/') : resolve(value)
}

function addPrefix(
  prefixes: Map<string, { prefix: string; label: string; caseInsensitive: boolean }>,
  value: string | undefined,
  label: string,
): void {
  if (!value) return
  const prefix = normalizePrefix(value)
  if (prefix.length <= 1) return
  if (!prefixes.has(prefix)) {
    prefixes.set(prefix, {
      prefix,
      label,
      caseInsensitive: isWindowsAbsolutePath(prefix),
    })
  }
}

export function createTraceRedactionContext(
  input: TraceRedactionInput = {},
): TraceRedactionContext {
  const env = input.env ?? process.env
  const stateDir = input.stateDir ?? sepilotdHome()
  const cwd = input.cwd ?? process.cwd()
  const prefixes = new Map<string, { prefix: string; label: string; caseInsensitive: boolean }>()
  addPrefix(prefixes, stateDir, '$SEPILOTD_HOME')
  addPrefix(prefixes, env.HOME, '~')
  addPrefix(prefixes, env.USERPROFILE, '~')
  addPrefix(prefixes, cwd, '$PWD')
  const sorted = [...prefixes.values()].sort((a, b) => b.prefix.length - a.prefix.length)
  const pathAliases: Record<string, string> = {}
  for (const prefix of sorted) {
    pathAliases[prefix.label] = `${prefix.label} path prefix`
  }
  return { env, cwd, stateDir, prefixes: sorted, pathAliases }
}

function hasPrefix(value: string, prefix: string, caseInsensitive: boolean): boolean {
  return caseInsensitive
    ? value.toLowerCase().startsWith(prefix.toLowerCase())
    : value.startsWith(prefix)
}

function replacePathPrefix(value: string, context: TraceRedactionContext): string {
  const candidates = isWindowsAbsolutePath(value)
    ? [value.replaceAll('\\', '/')]
    : [value, isAbsolute(value) ? resolve(value) : value]
  for (const candidate of candidates) {
    for (const prefix of context.prefixes) {
      if (!hasPrefix(candidate, prefix.prefix, prefix.caseInsensitive)) continue
      const rest = candidate.slice(prefix.prefix.length)
      if (rest && rest[0] !== '/' && rest[0] !== '\\') continue
      return `${prefix.label}${rest.replaceAll('\\', '/')}`
    }
  }
  return value
}

function replacePathPrefixesInText(value: string, context: TraceRedactionContext): string {
  let output = value
  for (const prefix of context.prefixes) {
    const variants = new Set([
      prefix.prefix,
      prefix.prefix.replaceAll('\\', '/'),
      isWindowsAbsolutePath(prefix.prefix) ? prefix.prefix.replaceAll('/', '\\') : prefix.prefix,
    ])
    for (const variant of variants) {
      output = output.split(variant).join(prefix.label)
    }
  }
  return output
}

export function redactUrlParams(value: string): string {
  return value.replace(URL_PARAM_RE, (_match, separator, key, rawValue) => {
    return SECRET_KEY_RE.test(String(key))
      ? `${separator}${key}=[redacted]`
      : `${separator}${key}=${rawValue}`
  })
}

export function redactSensitiveText(
  value: string,
  context = createTraceRedactionContext(),
  options: Pick<RedactValueOptions, 'maxStringLength'> = {},
): string {
  return truncate(replacePathPrefixesInText(value, context)
    .replace(BEARER_RE, 'Bearer [redacted]')
    .replace(BASIC_RE, 'Basic [redacted]')
    .replace(COOKIE_RE, '[redacted-cookie-header]')
    .replace(AWS_ACCESS_KEY_RE, '[redacted-aws-key]')
    .replace(JWT_RE, '[redacted-jwt]')
    .replace(URL_USERINFO_RE, '$1[redacted]@')
    .replace(EMAIL_RE, '[redacted-email]')
    .replace(NAMED_SECRET_RE, '$1$2[redacted]')
    .replace(LONG_TOKEN_RE, (match) => (CORRELATION_UUID_RE.test(match) ? match : '[redacted-token]')), options.maxStringLength)
}

export function redactSecretKeys(
  value: unknown,
  context = createTraceRedactionContext(),
  key = '',
  depth = 0,
  options: RedactValueOptions = {},
): unknown {
  const maxDepth = options.maxDepth ?? DEFAULT_MAX_DEPTH
  const maxArrayItems = options.maxArrayItems ?? DEFAULT_MAX_ARRAY_ITEMS
  const maxObjectKeys = options.maxObjectKeys ?? DEFAULT_MAX_OBJECT_KEYS

  if (depth >= maxDepth) return '[max-depth]'
  if (value === null || typeof value === 'number' || typeof value === 'boolean') {
    return value
  }
  if (typeof value === 'string') {
    if (SECRET_KEY_RE.test(key)) return '[redacted]'
    if (PATH_KEY_RE.test(key)) return replacePathPrefix(value, context)
    return redactSensitiveText(redactUrlParams(value), context, options)
  }
  if (typeof value === 'bigint' || typeof value === 'symbol') {
    return String(value)
  }
  if (Array.isArray(value)) {
    const items = value
      .slice(0, maxArrayItems)
      .map((item) => redactSecretKeys(item, context, key, depth + 1, options))
    if (value.length > maxArrayItems) {
      items.push(`[+${value.length - maxArrayItems} more items]`)
    }
    return items
  }
  if (typeof value === 'object') {
    const output: Record<string, unknown> = {}
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([entryKey]) => !BLOCKED_OBJECT_KEYS.has(entryKey))
      .sort(([left], [right]) => left.localeCompare(right))
    for (const [entryKey, entryValue] of entries.slice(0, maxObjectKeys)) {
      // Counters such as inputTokens/outputTokens are numeric telemetry, not
      // credentials. Preserve primitive non-string values even when the key
      // contains "token" so trace stats remain type-safe and summable.
      output[entryKey] = SECRET_KEY_RE.test(entryKey)
        && typeof entryValue !== 'number'
        && typeof entryValue !== 'boolean'
        && entryValue !== null
        ? '[redacted]'
        : redactSecretKeys(entryValue, context, entryKey, depth + 1, options)
    }
    if (entries.length > maxObjectKeys) {
      output.__truncated_keys__ = entries.length - maxObjectKeys
    }
    return output
  }
  if (typeof value === 'undefined') return null
  return String(value)
}
