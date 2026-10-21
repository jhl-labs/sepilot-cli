import type { EventRoutingConfig, EventRule } from './config-schema.js'
import type { EventDeliveryOutbox } from './delivery-outbox.js'
import { eventView, matchRules } from './match.js'
import type { RuleRunStore } from './rule-run-store.js'
import { DEFAULT_PASSTHROUGH_TEMPLATE, renderTemplate, RESULT_VALUE_MAX_CHARS } from './template.js'
import { summarizeEvent, type RoutedEvent } from './types.js'

export const DEFAULT_LLM_TEMPLATE = '{{result}}'

export function eventTitle(event: RoutedEvent): string {
  return event.subject ?? `${event.source} ${event.type}`
}

export function enqueueRuleDeliveries(input: {
  event: RoutedEvent
  rule: EventRule
  outbox: EventDeliveryOutbox
  result?: string
  now?: number
}): void {
  const { event, rule, outbox, result } = input
  const template = rule.template ?? (result === undefined ? DEFAULT_PASSTHROUGH_TEMPLATE : DEFAULT_LLM_TEMPLATE)
  const vars = { ...eventView(event), rule: { id: rule.id }, ...(result === undefined ? {} : { result }) }
  const text = renderTemplate(template, vars, { maxCharsByPath: { result: RESULT_VALUE_MAX_CHARS } })
  const summary = summarizeEvent(event)
  for (const destinationId of rule.deliver) {
    outbox.enqueue({ eventId: event.id, ruleId: rule.id, destinationId, title: eventTitle(event), text, event: summary }, input.now)
  }
}

export function routeEvent(input: {
  event: RoutedEvent
  config: EventRoutingConfig
  outbox: EventDeliveryOutbox
  ruleRuns: RuleRunStore
  now?: number
}): string[] {
  const matched = matchRules(input.config.rules, input.event)
  for (const rule of matched) {
    if (rule.process.mode === 'llm') {
      input.ruleRuns.create({ eventId: input.event.id, ruleId: rule.id }, input.now)
    } else {
      enqueueRuleDeliveries({ event: input.event, rule, outbox: input.outbox, now: input.now })
    }
  }
  return matched.map((rule) => rule.id)
}
