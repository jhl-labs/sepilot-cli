import type { IChannel } from '@sepilotd/core'
import type { SqliteDatabase } from '../db/sqlite.js'
import { createLogger } from '../logger.js'
import { createDeliver, type DeliverFn } from './adapters.js'
import type { EventDestination, EventRoutingConfig, EventRule, LlmProcess } from './config-schema.js'
import { createDeliveryOutbox, DELIVERY_MAX_ATTEMPTS, nextDeliveryBackoffMs, type DeliveryRecord, type EventDeliveryOutbox } from './delivery-outbox.js'
import { createEventStore, EVENT_MAX_ATTEMPTS, openEventRoutingDb, type EventStore } from './event-store.js'
import { buildLlmTurnMessage } from './llm-prompt.js'
import type { LlmTurnResult, LlmTurnRunner } from './llm-runner.js'
import { enqueueRuleDeliveries, eventTitle, routeEvent } from './router.js'
import { createRuleRunStore, RULE_RUN_MAX_ATTEMPTS, type RuleRunRecord, type RuleRunStore } from './rule-run-store.js'
import type { DeliveryOutcome, RoutedEvent, RoutedEventInput } from './types.js'

const log = createLogger('event-routing')
export const DELIVERY_TIMEOUT_MS = 60_000
export const EVENT_RETENTION_MS = 14 * 24 * 60 * 60_000
export const EVENT_RETENTION_MAX_ROWS = 10_000
export const RULE_RUN_RETENTION_MAX_ROWS = 10_000
export const DEAD_DELIVERY_RETENTION_MS = 30 * 24 * 60 * 60_000
const EVENT_RETRY_BACKOFF_MS = 5_000
const PRUNE_INTERVAL_MS = 60 * 60_000
export const LLM_FAILURE_PREFIX = 'LLM processing failed: '
const RULE_RUN_RETRY_BACKOFF_MS = 30_000
const RULE_RUN_STALE_GRACE_MS = 120_000
const RATE_WINDOW_MS = 60_000
const MAX_CLAIMS_PER_PASS = 50

export interface EventRoutingServiceDeps {
  getConfig: () => EventRoutingConfig
  channels: () => readonly IChannel[]
  deliver?: DeliverFn
  db?: SqliteDatabase
  now?: () => number
  intervalMs?: number
  llmRunner?: () => LlmTurnRunner | undefined
  timeoutSignal?: (ms: number) => AbortSignal
}

export interface EventRoutingService {
  readonly store: EventStore
  readonly outbox: EventDeliveryOutbox
  readonly ruleRuns: RuleRunStore
  ingest(input: RoutedEventInput): { eventId: string; duplicate: boolean }
  processDueEvents(limit?: number): Promise<number>
  processDueRuleRuns(): Promise<number>
  settleLlmRuns(): Promise<void>
  drainDeliveries(limit?: number): Promise<number>
  scheduleDrain(): void
  testDestination(destination: EventDestination, text?: string): Promise<DeliveryOutcome>
  start(): void
  stop(): Promise<void>
}

export function createEventRoutingService(deps: EventRoutingServiceDeps): EventRoutingService {
  const db = deps.db ?? openEventRoutingDb()
  const store = createEventStore(db)
  const outbox = createDeliveryOutbox(db)
  const deliver = deps.deliver ?? createDeliver({ channels: deps.channels })
  const now = deps.now ?? Date.now
  const intervalMs = deps.intervalMs ?? 5_000
  const timeoutSignal = deps.timeoutSignal ?? ((ms: number) => AbortSignal.timeout(ms))
  const ruleRuns = createRuleRunStore(db)
  const inFlight = new Map<string, { controller: AbortController; done: Promise<void> }>()
  let timer: NodeJS.Timeout | null = null
  let lastPruneAt = 0
  let active: Promise<void> | null = null
  let stopped = false

  const processDueEvents = async (limit = 50): Promise<number> => {
    const config = deps.getConfig()
    if (!config.enabled) return 0
    const claimed = store.claimDue(now(), limit)
    for (const record of claimed) {
      try {
        store.markRouted(record.id, routeEvent({ event: record, config, outbox, ruleRuns, now: now() }), now())
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        const retryAt = record.attempt >= EVENT_MAX_ATTEMPTS ? null : now() + EVENT_RETRY_BACKOFF_MS * record.attempt
        store.markFailed(record.id, message, retryAt, now())
        log.warn('event routing failed', { eventId: record.id, attempt: record.attempt, error: message })
      }
    }
    return claimed.length
  }

  const runRule = async (
    runner: LlmTurnRunner,
    record: RuleRunRecord,
    rule: EventRule,
    process: LlmProcess,
    event: RoutedEvent,
    timeoutMs: number,
    signal: AbortSignal,
  ): Promise<void> => {
    const result = await runner({
      ruleId: rule.id,
      eventId: event.id,
      sessionId: record.sessionId!,
      title: `[event] ${rule.id} ${eventTitle(event)}`.slice(0, 120),
      message: buildLlmTurnMessage(process.prompt, event),
      ...(process.bot ? { bot: process.bot } : {}),
      ...(process.project ? { projectId: process.project } : {}),
      ...(process.network ? { network: true } : {}),
      timeoutMs,
      signal,
    }).catch((error: unknown): LlmTurnResult => ({ ok: false, retryable: true, error: error instanceof Error ? error.message : String(error) }))
    if (result.ok) {
      ruleRuns.markSucceeded(record.id, result.text, now())
      enqueueRuleDeliveries({ event, rule, outbox, result: result.text, now: now() })
      return
    }
    if (stopped && signal.aborted) {
      ruleRuns.requeue(record.id, now())
      return
    }
    if (result.retryable && record.attempt < RULE_RUN_MAX_ATTEMPTS) {
      ruleRuns.markRetry(record.id, result.error, now() + RULE_RUN_RETRY_BACKOFF_MS * record.attempt, now())
      return
    }
    ruleRuns.markFailed(record.id, result.error, now())
    if (rule.deliverOnFailure) {
      enqueueRuleDeliveries({ event, rule, outbox, result: `${LLM_FAILURE_PREFIX}${result.error}`, now: now() })
    }
  }

  const processDueRuleRuns = async (): Promise<number> => {
    const config = deps.getConfig()
    if (stopped || !config.enabled) return 0
    const runner = deps.llmRunner?.()
    if (!runner) return 0
    const staleBefore = now() - config.limits.llmRunTimeoutMs - RULE_RUN_STALE_GRACE_MS
    let claimed = 0
    while (inFlight.size < config.limits.maxConcurrentLlmRuns && claimed < MAX_CLAIMS_PER_PASS) {
      const [record] = ruleRuns.claimDue(now(), 1, staleBefore, [...inFlight.keys()])
      if (!record) break
      claimed += 1
      const rule = config.rules.find((candidate) => candidate.id === record.ruleId)
      const process = rule?.process
      if (!rule?.enabled || !process || process.mode !== 'llm') {
        ruleRuns.markSkipped(record.id, `rule removed, disabled or no longer llm: ${record.ruleId}`, now())
        continue
      }
      const event = store.get(record.eventId)
      if (!event) {
        ruleRuns.markFailed(record.id, `event not found: ${record.eventId}`, now())
        continue
      }
      if (record.attempt > RULE_RUN_MAX_ATTEMPTS) {
        const error = 'stale run exceeded max attempts'
        ruleRuns.markFailed(record.id, error, now())
        if (rule.deliverOnFailure) {
          enqueueRuleDeliveries({ event, rule, outbox, result: `${LLM_FAILURE_PREFIX}${error}`, now: now() })
        }
        continue
      }
      if (rule.rateLimitPerMinute !== undefined
        && ruleRuns.countStartedSince(rule.id, now() - RATE_WINDOW_MS, record.id) >= rule.rateLimitPerMinute) {
        ruleRuns.markThrottled(record.id, now())
        continue
      }
      const controller = new AbortController()
      const signal = AbortSignal.any([controller.signal, timeoutSignal(config.limits.llmRunTimeoutMs)])
      const done = runRule(runner, record, rule, process, event, config.limits.llmRunTimeoutMs, signal)
        .catch((error: unknown) => log.warn('llm rule run failed', { runId: record.id, error: String(error) }))
        .finally(() => { inFlight.delete(record.id) })
      inFlight.set(record.id, { controller, done })
    }
    return claimed
  }

  const settleLlmRuns = async (): Promise<void> => {
    while (inFlight.size > 0) await Promise.allSettled([...inFlight.values()].map((run) => run.done))
  }

  const settle = (record: DeliveryRecord, outcome: DeliveryOutcome): void => {
    if (outcome.ok) {
      outbox.markDelivered(record.id, now())
    } else if (!outcome.retryable || record.attempt >= DELIVERY_MAX_ATTEMPTS) {
      outbox.markDead(record.id, outcome.error, now())
    } else {
      outbox.markRetry(record.id, outcome.error, now() + nextDeliveryBackoffMs(record.attempt), now())
    }
  }

  const drainDeliveries = async (limit = 20): Promise<number> => {
    const config = deps.getConfig()
    if (!config.enabled) return 0
    const claimed = outbox.claimDue(now(), limit)
    await Promise.all(claimed.map(async (record) => {
      const destination = config.destinations.find((candidate) => candidate.id === record.destinationId)
      if (!destination) return outbox.markDead(record.id, `destination removed: ${record.destinationId}`, now())
      if (!destination.enabled) return outbox.markDead(record.id, `destination disabled: ${record.destinationId}`, now())
      const outcome = await deliver(destination, record, {
        deliveryId: record.id,
        attempt: record.attempt,
        signal: AbortSignal.timeout(DELIVERY_TIMEOUT_MS),
      }).catch((error: unknown): DeliveryOutcome => ({ ok: false, retryable: true, error: error instanceof Error ? error.message : String(error) }))
      settle(record, outcome)
    }))
    return claimed.length
  }

  const tick = (): void => {
    if (stopped || active) return
    active = (async () => {
      await processDueEvents()
      await processDueRuleRuns()
      await drainDeliveries()
      if (now() - lastPruneAt >= PRUNE_INTERVAL_MS) {
        lastPruneAt = now()
        store.prune(now() - EVENT_RETENTION_MS, EVENT_RETENTION_MAX_ROWS)
        outbox.prune(now() - EVENT_RETENTION_MS, now() - DEAD_DELIVERY_RETENTION_MS)
        ruleRuns.prune(now() - EVENT_RETENTION_MS, RULE_RUN_RETENTION_MAX_ROWS)
      }
    })()
      .catch((error: unknown) => log.warn('event routing tick failed', { error: String(error) }))
      .finally(() => { active = null })
  }

  return {
    store,
    outbox,
    ruleRuns,
    ingest(input) {
      const { record, duplicate } = store.append(input, now())
      if (!duplicate && timer) setImmediate(tick)
      return { eventId: record.id, duplicate }
    },
    processDueEvents,
    processDueRuleRuns,
    settleLlmRuns,
    drainDeliveries,
    scheduleDrain() {
      if (timer) setImmediate(tick)
    },
    testDestination(destination, text = 'sepilot event routing test message') {
      const at = now()
      return deliver(destination, {
        ruleId: 'test',
        title: 'sepilot test',
        text,
        event: { id: 'test', source: 'event-routing', type: 'test', subject: null, occurredAt: at },
      }, { deliveryId: `test-${at}`, attempt: 1, signal: AbortSignal.timeout(DELIVERY_TIMEOUT_MS) })
    },
    start() {
      if (timer) return
      stopped = false
      timer = setInterval(tick, intervalMs)
      timer.unref?.()
      tick()
    },
    async stop() {
      stopped = true
      if (timer) clearInterval(timer)
      timer = null
      for (const run of inFlight.values()) run.controller.abort()
      await active
      await settleLlmRuns()
    },
  }
}
