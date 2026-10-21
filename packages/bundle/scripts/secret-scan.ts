import { readFileSync } from 'node:fs'

export interface ScanResult {
  ok: boolean
  findings: string[]
}

const PLACEHOLDER_MARKERS = [
  'test-token',
  'example.invalid',
  'your-token-here',
  'placeholder',
  'dummy',
  'replace_me',
  'replace-me',
  'changeme',
  'change-me',
  'xxxxxxxx',
  '<token>',
  'example.com',
  'your_token',
  'your-key',
  'your_key',
  'fake',
  'sample',
  'redacted',
]

function isPlaceholder(value: string): boolean {
  const v = value.toLowerCase()
  if (PLACEHOLDER_MARKERS.some((m) => v.includes(m))) return true
  const body = v.replace(/^(sk-ant-|sk-|ghp_|gho_|ghs_|github_pat_|akia|aiza|xox[baprs]-)/i, '')
  if (/^x+$/.test(body)) return true
  if (/^0+$/.test(body)) return true
  return false
}

interface PatternSpec {
  label: string
  regex: RegExp
  allowPlaceholder?: boolean
}

interface Base64DataUri {
  payloadStart: number
  payloadEnd: number
  decoded: string
}

const BASE64_DATA_URI_RE = /data:[^,\s"'`]{0,256};base64,([A-Za-z0-9+/]+={0,2})/g
const MAX_EMBEDDED_DATA_URI_DEPTH = 4

function findBase64DataUris(text: string): Base64DataUri[] {
  const uris: Base64DataUri[] = []
  BASE64_DATA_URI_RE.lastIndex = 0

  let match: RegExpExecArray | null = BASE64_DATA_URI_RE.exec(text)
  while (match !== null) {
    const payload = match[1]
    const payloadStart = match.index + match[0].length - payload.length
    uris.push({
      payloadStart,
      payloadEnd: payloadStart + payload.length,
      decoded: Buffer.from(payload, 'base64').toString('latin1'),
    })
    match = BASE64_DATA_URI_RE.exec(text)
  }

  return uris
}

const PATTERNS: PatternSpec[] = [
  {
    label: 'PEM private key block',
    regex: /-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----/g,
    allowPlaceholder: false,
  },
  {
    label: 'PGP private key block',
    regex: /-----BEGIN PGP PRIVATE KEY(?: BLOCK)?-----/g,
    allowPlaceholder: false,
  },
  { label: 'Slack token', regex: /xox[baprs]-[A-Za-z0-9-]{10,}/g },
  { label: 'GitHub personal access token (classic)', regex: /ghp_[A-Za-z0-9]{36}/g },
  { label: 'GitHub fine-grained PAT', regex: /github_pat_[A-Za-z0-9_]{60,}/g },
  { label: 'GitHub OAuth token', regex: /gho_[A-Za-z0-9]{36}/g },
  { label: 'GitHub server-to-server token', regex: /ghs_[A-Za-z0-9]{36}/g },
  { label: 'Anthropic API key', regex: /(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_-]{20,}/g },
  { label: 'OpenAI-style API key', regex: /(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}/g },
  { label: 'AWS access key id', regex: /AKIA[0-9A-Z]{16}/g, allowPlaceholder: false },
  { label: 'Google API key', regex: /AIza[0-9A-Za-z_-]{35}/g },
  {
    label: 'Bearer credential',
    regex: /\bBearer\s+[A-Za-z0-9][A-Za-z0-9._~+/-]{19,}={0,2}/gi,
  },
  {
    label: 'OAuth client secret assignment',
    regex:
      /\b(?:client_secret|clientSecret)\s*["'`]?\s*[:=]\s*(["'`])[A-Za-z0-9][A-Za-z0-9._~+/-]{19,}={0,2}\1/gi,
  },
  {
    label: 'OAuth client secret assignment',
    regex:
      /(?:^|\r?\n)[ \t]*(?:client_secret|clientSecret)[ \t]*[:=][ \t]*[A-Za-z0-9][A-Za-z0-9._~+/-]{19,}={0,2}[ \t]*(?=#|\r?$)/gim,
  },
  {
    label: 'sepilot daemon/gateway token assignment',
    regex:
      /\b(?:daemonToken|gatewayToken|daemon_token|gateway_token|SEPILOTD_TOKEN|SEPILOTD?_(?:DAEMON|GATEWAY)_TOKEN)\s*["'`]?\s*[:=]\s*["'`]?[A-Fa-f0-9]{32,}(?![A-Za-z0-9_-])/gi,
    allowPlaceholder: false,
  },
  { label: 'npm registry auth token line', regex: /\/\/registry\.npmjs\.org\/:_authToken=\S+/g },
  { label: 'npm `_authToken=` assignment', regex: /(?:^|[\s"'`])_authToken\s*=\s*\S+/g },
]

function scanText(text: string, findings: string[], seen: Set<string>, embeddedDepth = 0): void {
  const dataUris = findBase64DataUris(text)
  const isEncodedDataUriCharacter = (index: number): boolean =>
    dataUris.some(({ payloadStart, payloadEnd }) => index >= payloadStart && index < payloadEnd)

  const antKeys = new Set<string>()

  for (const spec of PATTERNS) {
    spec.regex.lastIndex = 0
    let m: RegExpExecArray | null = spec.regex.exec(text)
    while (m !== null) {
      if (isEncodedDataUriCharacter(m.index)) {
        m = spec.regex.exec(text)
        continue
      }
      const raw = m[0]
      const value = raw.replace(/^[\s"'`]+/, '')
      if (spec.label === 'OpenAI-style API key') {
        if (value.startsWith('sk-ant-') || antKeys.has(value) || /^sk-[-_]*$/.test(value)) {
          m = spec.regex.exec(text)
          continue
        }
      }
      if (spec.label === 'Anthropic API key') antKeys.add(value)
      const allowPlaceholder = spec.allowPlaceholder ?? true
      if (allowPlaceholder && isPlaceholder(value)) {
        m = spec.regex.exec(text)
        continue
      }
      const shown = `[REDACTED] (${value.length} chars)`
      const key = `${spec.label}:${shown}`
      if (!seen.has(key)) {
        seen.add(key)
        findings.push(`${spec.label}: ${shown}`)
      }
      m = spec.regex.exec(text)
    }
  }

  if (embeddedDepth >= MAX_EMBEDDED_DATA_URI_DEPTH) return
  for (const { decoded } of dataUris) {
    scanText(decoded, findings, seen, embeddedDepth + 1)
  }
}

export function scanArtifactBytes(
  bytes: Uint8Array,
  displayPath = '<in-memory artifact>',
): ScanResult {
  const text = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('latin1')
  const findings: string[] = []
  scanText(text, findings, new Set<string>())

  if (findings.length > 0) {
    return {
      ok: false,
      findings: [
        `scanned ${displayPath} (${bytes.byteLength} bytes) — ${findings.length} secret-like match(es):`,
        ...findings,
      ],
    }
  }
  return { ok: true, findings: [] }
}

export function scanArtifact(filePath: string): ScanResult {
  try {
    return scanArtifactBytes(readFileSync(filePath), filePath)
  } catch (err) {
    return {
      ok: false,
      findings: [`cannot read ${filePath}: ${err instanceof Error ? err.message : String(err)}`],
    }
  }
}
