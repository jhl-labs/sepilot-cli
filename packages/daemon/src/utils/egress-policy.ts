import picomatch from 'picomatch'
import { createLogger } from '../logger.js'

const ENV_ALLOWLIST = 'SEPILOTD_EGRESS_ALLOWLIST'
const HIGH_ENTROPY_TOKEN = /[A-Za-z0-9+_=-]{24,}/g

const defaultLog = createLogger('egress-policy')

export class EgressDeniedError extends Error {
  code = 'EGRESS_DENIED_PERMANENT' as const
}

export interface EgressPolicyLogger {
  warn(message: string, meta?: Record<string, unknown>): void
}

export interface EgressPolicyOptions {
  allowlist?: readonly string[] | null
  env?: Record<string, string | undefined>
  logger?: EgressPolicyLogger
}

export interface EgressPolicy {
  allowlist: readonly string[]
  enforced: boolean
  assertAllowed(rawUrl: string, toolName: string): URL
}

function cleanList(items: readonly string[] | undefined): string[] {
  return Array.from(new Set(
    (items ?? [])
      .map((item) => item.trim())
      .filter(Boolean),
  ))
}

function parseEnvAllowlist(env: Record<string, string | undefined>): string[] {
  return cleanList(
    (env[ENV_ALLOWLIST] ?? '')
      .split(/[,\s]+/u)
      .filter(Boolean),
  )
}

export function resolveEgressAllowlist(
  allowlist: readonly string[] | null | undefined,
  env: Record<string, string | undefined> = process.env,
): string[] {
  return cleanList([
    ...(allowlist ?? []),
    ...parseEnvAllowlist(env),
  ])
}

function normalizeHost(hostname: string): string {
  return hostname.replace(/^\[(.*)\]$/, '$1').toLowerCase()
}

function hostMatches(host: string, pattern: string): boolean {
  try {
    return picomatch.isMatch(host, pattern, {
      nocase: true,
      dot: true,
    })
  } catch {
    return false
  }
}

function looksHighEntropy(token: string): boolean {
  if (token.length < 24) return false
  const unique = new Set(token).size
  return unique >= 12 && /[a-z]/.test(token) && /[A-Z]/.test(token) && /\d/.test(token)
}

function warnOnHighEntropyUrlComponent(
  parsed: URL,
  toolName: string,
  logger: EgressPolicyLogger,
): void {
  const candidates = [
    ...parsed.pathname.split(/[/?#]+/u),
    ...Array.from(parsed.searchParams.keys()),
    ...Array.from(parsed.searchParams.values()),
  ]
  for (const candidate of candidates) {
    for (const match of candidate.matchAll(HIGH_ENTROPY_TOKEN)) {
      const token = match[0]
      if (!looksHighEntropy(token)) continue
      logger.warn('Potential high-entropy outbound URL component', {
        tool: toolName,
        host: normalizeHost(parsed.hostname),
        tokenLength: token.length,
      })
      return
    }
  }
}

export function createEgressPolicy(options: EgressPolicyOptions = {}): EgressPolicy {
  const env = options.env ?? process.env
  const allowlist = resolveEgressAllowlist(options.allowlist, env)
  const logger = options.logger ?? defaultLog
  return {
    allowlist,
    enforced: allowlist.length > 0,
    assertAllowed(rawUrl: string, toolName: string): URL {
      let parsed: URL
      try {
        parsed = new URL(rawUrl)
      } catch {
        throw new Error('invalid url')
      }

      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        return parsed
      }

      warnOnHighEntropyUrlComponent(parsed, toolName, logger)

      if (allowlist.length === 0) return parsed

      const host = normalizeHost(parsed.hostname)
      if (allowlist.some((pattern) => hostMatches(host, pattern))) {
        return parsed
      }

      throw new EgressDeniedError(`egress denied for host ${host}`)
    },
  }
}
