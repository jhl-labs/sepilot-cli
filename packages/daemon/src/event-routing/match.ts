import type { EventRule } from './config-schema.js'
import { summarizeEvent, type RoutedEvent } from './types.js'

export function resolvePath(root: unknown, path: string): unknown {
  let current: unknown = root
  for (const segment of path.split('.')) {
    if (current === null || typeof current !== 'object') return undefined
    if (!Object.hasOwn(current, segment)) return undefined
    current = (current as Record<string, unknown>)[segment]
  }
  return current
}

export function eventView(event: RoutedEvent): Record<string, unknown> {
  return { event: summarizeEvent(event), payload: event.payload }
}

export function sourceMatches(pattern: string, source: string): boolean {
  return pattern.endsWith('*') ? source.startsWith(pattern.slice(0, -1)) : pattern === source
}

type Scalar = string | number | boolean

function isScalar(value: unknown): value is Scalar {
  return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
}

function whereMatches(expected: Scalar | Scalar[], actual: unknown): boolean {
  if (!isScalar(actual)) return false
  return Array.isArray(expected) ? expected.includes(actual) : expected === actual
}

export function ruleMatches(rule: EventRule, event: RoutedEvent): boolean {
  if (!rule.enabled) return false
  if (!sourceMatches(rule.match.source, event.source)) return false
  if (rule.match.types.length > 0 && !rule.match.types.includes(event.type)) return false
  const view = eventView(event)
  return Object.entries(rule.match.where).every(([path, expected]) => whereMatches(expected, resolvePath(view, path)))
}

export function matchRules(rules: readonly EventRule[], event: RoutedEvent): EventRule[] {
  return rules.filter((rule) => ruleMatches(rule, event))
}
