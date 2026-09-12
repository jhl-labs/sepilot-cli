import { dirname, isAbsolute, normalize } from 'node:path'
import { createHash } from 'node:crypto'
import picomatch from 'picomatch'
import type { ApprovalRule } from '@sepilotd/core'

function terminalPattern(input: Record<string, unknown>): string {
  const executable = typeof input.executable === 'string' ? input.executable : ''
  const rawArgs = Array.isArray(input.args) ? input.args : []
  const firstArg = typeof rawArgs[0] === 'string' ? rawArgs[0] : ''
  if (!executable) return 'terminal.run *'
  if (!firstArg) return `${executable} *`
  return `${executable} ${firstArg} *`
}

function terminalRawTarget(input: Record<string, unknown>): string {
  const executable = typeof input.executable === 'string' ? input.executable : ''
  const rawArgs = Array.isArray(input.args)
    ? input.args.filter((value): value is string => typeof value === 'string')
    : []
  if (!executable) return ''
  return rawArgs.length > 0 ? `${executable} ${rawArgs.join(' ')}` : executable
}

function pathDirectoryPattern(input: Record<string, unknown>): string {
  const path = typeof input.path === 'string' ? input.path : ''
  if (!path) return '**'
  const normalized = normalize(path)
  const parent = dirname(normalized)
  if (!parent || parent === '.' || parent === normalized) {
    return `${normalized}`
  }
  return isAbsolute(parent) ? `${parent}/**` : `${parent}/**`
}

function browserUrlPattern(input: Record<string, unknown>): string {
  const url = typeof input.url === 'string' ? input.url : ''
  if (!url) return '*'
  try {
    const parsed = new URL(url)
    return `${parsed.origin}/**`
  } catch {
    return url
  }
}

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalize)
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, entry]) => entry !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => [key, canonicalize(entry)]),
    )
  }
  return value
}

function exactPattern(input: Record<string, unknown>): string {
  const serialized = JSON.stringify(canonicalize(input))
  const digest = createHash('sha256').update(serialized).digest('hex')
  return `exact:sha256:${digest}`
}

function legacyPatternFor(tool: string, input: Record<string, unknown>): string | null {
  switch (tool) {
    case 'terminal.run':
      return terminalPattern(input)
    case 'fs.append':
    case 'fs.write':
    case 'fs.read':
      return pathDirectoryPattern(input)
    case 'browser.navigate':
    case 'browser.screenshot':
    case 'browser.click':
    case 'browser.evaluate':
    case 'browser.extract':
      return browserUrlPattern(input)
    default:
      return null
  }
}

export function describeRuleFor(
  tool: string,
  input: Record<string, unknown>,
): ApprovalRule {
  // Generated remembered rules are exact by default. Operators can still edit
  // a rule into a reviewed wildcard, and legacy executable/directory/origin
  // patterns remain matchable below, but one click must never silently widen
  // into "every call of this tool".
  return { tool, pattern: exactPattern(input) }
}

function safeIsMatch(target: string, pattern: string): boolean {
  if (!target) return false
  try {
    return picomatch.isMatch(target, pattern, { nocase: false, dot: true })
  } catch {
    return false
  }
}

export function matchesRule(
  rule: ApprovalRule,
  tool: string,
  input: Record<string, unknown>,
): boolean {
  if (rule.tool !== tool) return false
  if (rule.pattern === '*') return true
  if (rule.pattern.startsWith('exact:sha256:')) {
    return rule.pattern === exactPattern(input)
  }

  if (tool !== 'terminal.run') {
    const candidate = legacyPatternFor(tool, input)
    if (!candidate) return false
    if (candidate === rule.pattern) return true
    return safeIsMatch(candidate, rule.pattern)
  }

  const raw = terminalRawTarget(input)
  if (safeIsMatch(raw, rule.pattern)) return true
  return terminalPattern(input) === rule.pattern
}
