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
  /\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}/,
  /\bgh[pousr]_[A-Za-z0-9]{20,}/,
  /\bgithub_pat_[A-Za-z0-9_]{20,}/,
  /\bxox[baprs]-[A-Za-z0-9-]{10,}/,
  /\bA(?:KIA|SIA)[A-Z0-9]{16}\b/,
  /\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}/,
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
