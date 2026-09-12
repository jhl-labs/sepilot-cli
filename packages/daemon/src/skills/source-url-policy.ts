import { SkillSourceUrlNotAllowedError } from './errors.js'

export interface SkillSourceUrlPolicyConfig {
  enforceUrlAllowlist?: boolean
  allowedHosts?: string[]
  allowedUrlPrefixes?: string[]
  allowLocalPaths?: boolean
}

function cleanList(items: readonly string[] | undefined): string[] {
  return Array.from(new Set(
    (items ?? [])
      .map((item) => item.trim())
      .filter(Boolean),
  ))
}

function isLocalPath(value: string): boolean {
  return !/^[a-z][a-z0-9+.-]*:\/\//i.test(value) && !/^git@/i.test(value)
}

function parseSshGitHost(value: string): string | null {
  const match = value.match(/^[^@]+@([^:/]+)[:/]/)
  return match?.[1]?.toLowerCase() ?? null
}

function normalizeHost(host: string): string {
  return host.trim().toLowerCase()
}

function hostMatches(host: string, allowed: string): boolean {
  const normalized = normalizeHost(allowed)
  if (!normalized) return false
  if (normalized.startsWith('*.')) {
    const suffix = normalized.slice(1)
    return host.endsWith(suffix) && host !== normalized.slice(2)
  }
  return host === normalized
}

function urlMatchesPrefix(url: URL, prefix: string): boolean {
  let parsed: URL
  try {
    parsed = new URL(prefix)
  } catch {
    return false
  }

  if (parsed.protocol !== url.protocol) return false
  if (parsed.hostname.toLowerCase() !== url.hostname.toLowerCase()) return false
  if (parsed.port !== url.port) return false

  const prefixPath = parsed.pathname.replace(/\/+$/, '')
  const path = url.pathname.replace(/\/+$/, '')
  if (prefixPath && prefixPath !== '/') {
    if (path !== prefixPath && !path.startsWith(`${prefixPath}/`)) return false
  }

  if (parsed.search && parsed.search !== url.search) return false
  if (parsed.hash && parsed.hash !== url.hash) return false
  return true
}

export class SkillSourceUrlPolicy {
  allowedHosts: string[] = []
  allowedUrlPrefixes: string[] = []
  allowLocalPaths = false
  enforceUrlAllowlist = false

  constructor(config: SkillSourceUrlPolicyConfig = {}) {
    this.update(config)
  }

  update(config: SkillSourceUrlPolicyConfig = {}): void {
    this.allowedHosts = cleanList(config.allowedHosts)
    this.allowedUrlPrefixes = cleanList(config.allowedUrlPrefixes)
    this.allowLocalPaths = config.allowLocalPaths === true
    this.enforceUrlAllowlist = config.enforceUrlAllowlist === true
      || this.allowedHosts.length > 0
      || this.allowedUrlPrefixes.length > 0
  }

  get enforced(): boolean {
    return this.enforceUrlAllowlist
  }

  assertAllowed(rawUrl: string, kind: string): void {
    const value = rawUrl.trim()
    if (!value) throw new SkillSourceUrlNotAllowedError(rawUrl, `${kind} source URL is empty`)
    if (!this.enforced) return

    if (isLocalPath(value)) {
      if (this.allowLocalPaths) return
      throw new SkillSourceUrlNotAllowedError(value, 'local paths are not allowlisted')
    }

    const sshHost = parseSshGitHost(value)
    if (sshHost) {
      throw new SkillSourceUrlNotAllowedError(
        value,
        `SSH git sources are not allowed; use an HTTPS URL for host ${sshHost}`,
      )
    }

    let parsed: URL
    try {
      parsed = new URL(value)
    } catch {
      throw new SkillSourceUrlNotAllowedError(value, 'source is not a valid URL')
    }

    if (parsed.protocol !== 'https:') {
      throw new SkillSourceUrlNotAllowedError(value, 'only https:// skill sources are allowed')
    }

    const host = parsed.hostname.toLowerCase()
    if (this.allowedHosts.some((allowed) => hostMatches(host, allowed))) return
    if (this.allowedUrlPrefixes.some((prefix) => urlMatchesPrefix(parsed, prefix))) return

    throw new SkillSourceUrlNotAllowedError(
      value,
      `host ${host} does not match security.skillSources allowlist`,
    )
  }
}
