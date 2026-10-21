import { createTraceRedactionContext, redactSensitiveText } from '../../observability/trace-redaction.js'
import { sourceMatches } from '../match.js'
import type { EventRoutingConfig } from '../config-schema.js'

export { EVENT_ROUTING_SURFACE } from '../origin.js'

export function anyRuleMatchesSource(config: EventRoutingConfig, source: string): boolean {
  if (!config.enabled) return false
  return config.rules.some((rule) => rule.enabled && sourceMatches(rule.match.source, source))
}

export function redactForEvent(text: string | undefined, max: number): string | undefined {
  if (text === undefined) return undefined
  return redactSensitiveText(text, createTraceRedactionContext(), { maxStringLength: max })
}
