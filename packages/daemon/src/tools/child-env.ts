const ALWAYS_PASSTHROUGH_ENV = new Set([
  'PATH',
  'HOME',
  'SHELL',
  'USER',
  'LOGNAME',
  'LANG',
  'LANGUAGE',
  'TERM',
  'TZ',
  'TMPDIR',
  'TMP',
  'TEMP',
  'PWD',
  'COLUMNS',
  'LINES',
  'HOSTNAME',
])

// Structural (not name-list) secret detectors. A child spawned by terminal.run
// must NOT inherit the daemon's provider keys, cloud credentials, or channel
// tokens — otherwise `terminal.run{executable:'env'}` (or any subprocess that
// prints its environment) dumps every secret straight into the model context.
// We keep the essentials above and drop anything whose KEY has a secret-shaped
// segment or whose VALUE looks like a credential.
const SECRET_KEY_SEGMENTS = new Set([
  'KEY',
  'KEYS',
  'APIKEY',
  'TOKEN',
  'TOKENS',
  'SECRET',
  'SECRETS',
  'PASSWORD',
  'PASSWD',
  'PASSPHRASE',
  'CREDENTIAL',
  'CREDENTIALS',
  'CREDS',
  'PRIVATEKEY',
])

const SECRET_VALUE_PATTERNS: RegExp[] = [
  // Well-known credential prefixes (OpenAI/Anthropic sk-, GitHub, Slack, AWS).
  /\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}/,
  /\bgh[pousr]_[A-Za-z0-9]{20,}/,
  /\bgithub_pat_[A-Za-z0-9_]{20,}/,
  /\bxox[baprs]-[A-Za-z0-9-]{10,}/,
  /\bA(?:KIA|SIA)[A-Z0-9]{16}\b/,
  // JWT: three base64url segments separated by dots.
  /\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}/,
  // URL carrying inline credentials (scheme://user:pass@host).
  /[a-z][a-z0-9+.-]*:\/\/[^/\s:@]+:[^/\s:@]+@/i,
]

function hasSecretShapedKey(key: string): boolean {
  const segments = key.toUpperCase().split(/[^A-Z0-9]+/)
  return segments.some((segment) => SECRET_KEY_SEGMENTS.has(segment))
}

function hasSecretShapedValue(value: string): boolean {
  if (value.length < 12) return false
  return SECRET_VALUE_PATTERNS.some((pattern) => pattern.test(value))
}

function resolveEnvPassthrough(raw: string | undefined): Set<string> {
  if (!raw) return new Set()
  return new Set(
    raw
      .split(/[,\s]+/)
      .map((entry) => entry.trim())
      .filter(Boolean),
  )
}

/**
 * Produce a scrubbed environment for a terminal child process. Keeps the
 * essential shell variables plus an optional operator allowlist, and drops
 * anything whose key or value is secret-shaped. This is a structural
 * secret-shape filter, not a content-meaning heuristic: it never inspects the
 * command being run, only the environment being handed to it.
 */
export function scrubChildEnv(
  source: NodeJS.ProcessEnv = process.env,
  options: { passthrough?: string } = {},
): NodeJS.ProcessEnv {
  const passthrough = resolveEnvPassthrough(
    options.passthrough ?? process.env.SEPILOTD_TERMINAL_ENV_PASSTHROUGH,
  )
  const scrubbed: NodeJS.ProcessEnv = {}
  for (const [key, value] of Object.entries(source)) {
    if (value == null) continue
    // Git's indexed config is atomic. It can contain auth headers; never
    // inherit a partial tuple or let a broad passthrough expose its values.
    if (key === 'GIT_CONFIG_COUNT' || /^GIT_CONFIG_(?:KEY|VALUE)_\d+$/.test(key)
      || key === 'GIT_CONFIG_PARAMETERS') continue
    if (ALWAYS_PASSTHROUGH_ENV.has(key) || key.startsWith('LC_') || passthrough.has(key)) {
      scrubbed[key] = value
      continue
    }
    if (hasSecretShapedKey(key) || hasSecretShapedValue(value)) {
      continue
    }
    scrubbed[key] = value
  }
  return scrubbed
}
