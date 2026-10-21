export interface BrowserSitePolicy {
  allow: string[]
  deny: string[]
}

export type BrowserSiteDecision = { kind: 'deny' } | { kind: 'allow' } | { kind: 'prompt' }

export const EMPTY_BROWSER_SITE_POLICY: BrowserSitePolicy = { allow: [], deny: [] }

function canonicalHost(value: string): string {
  return value.trim().toLowerCase().replace(/\.+$/, '')
}

export function hostMatches(host: string, pattern: string): boolean {
  const candidate = canonicalHost(host)
  const rule = canonicalHost(pattern)
  if (!candidate || !rule) return false
  if (rule.startsWith('*.')) {
    const suffix = rule.slice(1)
    return candidate.endsWith(suffix) && candidate.length > suffix.length
  }
  return candidate === rule
}

function hostOf(url: string): string | null {
  try {
    return canonicalHost(new URL(url).hostname) || null
  } catch {
    return null
  }
}

export function decideBrowserSite(url: string, policy: BrowserSitePolicy): BrowserSiteDecision {
  const host = hostOf(url)
  if (!host) return { kind: 'prompt' }
  if (policy.deny.some((pattern) => hostMatches(host, pattern))) return { kind: 'deny' }
  if (policy.allow.some((pattern) => hostMatches(host, pattern))) return { kind: 'allow' }
  return { kind: 'prompt' }
}
