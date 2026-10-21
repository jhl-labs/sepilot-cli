import { readFileSync } from 'fs'
import chalk from 'chalk'
import YAML from 'yaml'
import {
  ApiHttpError,
  type DaemonEventRoutingDelivery,
  type DaemonEventRoutingDeliveryStatus,
  type DaemonEventRoutingDestination,
  type DaemonEventRoutingDestinationInput,
  type DaemonEventRoutingEvent,
  type DaemonEventRoutingEventDetail,
  type DaemonEventRoutingEventStatus,
  type DaemonEventRoutingRule,
  type DaemonEventRoutingRuleInput,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { getOutputFormat, output, outputError } from '../output/formatter.js'
import { detectCliLocale } from '../utils/locale.js'

const EVENTS_COPY = {
  en: {
    unavailable: 'Event routing is unavailable on this daemon.',
    noDestinations: 'No event routing destinations configured.\nUse `sepilot events destinations add <id> --kind <kind> ...` to add one.',
    noRules: 'No event routing rules configured.\nUse `sepilot events rules add --file <path>` to add one.',
    noEvents: 'No routed events recorded yet.',
    noDeliveries: 'No routed deliveries recorded yet.',
    destinationSaved: (id: string) => `Event routing destination saved: ${id}`,
    destinationRemoved: (id: string) => `Event routing destination removed: ${id}`,
    ruleSaved: (id: string) => `Event routing rule saved: ${id}`,
    ruleRemoved: (id: string) => `Event routing rule removed: ${id}`,
    deliveryReplayed: (id: string) => `Delivery replayed: ${id}`,
    deliveryAcked: (id: string) => `Delivery acknowledged: ${id}`,
    ruleInputRequired: 'Provide the rule definition with --file <path.json|yaml> or --json \'<rule>\'',
    invalidJson: (message: string) => `Invalid --json rule payload: ${message}`,
    ruleFileReadError: (path: string, message: string) => `Could not read rule file ${path}: ${message}`,
    destinationMessengerRequiresPresetUrl: '--preset and --target are required for --kind messenger-webhook',
    destinationHttpRequiresUrl: '--target is required for --kind http',
    destinationChannelRequiresTypeTarget: '--channel-type and --channel-target are required for --kind channel',
    destinationKindInvalid: (kind: string) => `Unknown destination kind: ${kind}`,
    summary: (destinations: number, rules: number) =>
      `Event routing: ${destinations} destination(s), ${rules} rule(s).\nUse \`sepilot events destinations\` / \`sepilot events rules\` for details.`,
    enabled: 'enabled',
    disabled: 'disabled',
    destinationTestSucceeded: (id: string) => `Destination test succeeded: ${id}`,
    destinationTestFailed: (id: string, error: string) => `Destination test failed: ${id} (${error})`,
    invalidLimit: (value: string) => `Invalid --limit value: ${value} (must be a positive integer)`,
    invalidEventStatus: (value: string) => `Invalid --status value: ${value} (must be one of received, processing, processed, unmatched, failed)`,
    invalidDeliveryStatus: (value: string) => `Invalid --status value: ${value} (must be one of pending, delivering, delivered, dead)`,
    invalidPayload: (message: string) => `Invalid --payload payload: ${message}`,
    invalidPayloadFile: (path: string, message: string) => `Could not parse --payload-file ${path}: ${message}`,
    invalidHeaderPair: (pair: string) => `Invalid --header value: ${pair} (expected format key=value)`,
  },
  ko: {
    unavailable: '이 데몬에서는 이벤트 라우팅을 사용할 수 없습니다.',
    noDestinations: '구성된 이벤트 라우팅 대상이 없습니다.\n`sepilot events destinations add <id> --kind <kind> ...`로 추가하세요.',
    noRules: '구성된 이벤트 라우팅 규칙이 없습니다.\n`sepilot events rules add --file <path>`로 추가하세요.',
    noEvents: '아직 기록된 라우팅 이벤트가 없습니다.',
    noDeliveries: '아직 기록된 라우팅 전달이 없습니다.',
    destinationSaved: (id: string) => `이벤트 라우팅 대상 저장됨: ${id}`,
    destinationRemoved: (id: string) => `이벤트 라우팅 대상 제거됨: ${id}`,
    ruleSaved: (id: string) => `이벤트 라우팅 규칙 저장됨: ${id}`,
    ruleRemoved: (id: string) => `이벤트 라우팅 규칙 제거됨: ${id}`,
    deliveryReplayed: (id: string) => `전달 재시도됨: ${id}`,
    deliveryAcked: (id: string) => `전달 확인됨: ${id}`,
    ruleInputRequired: '--file <path.json|yaml> 또는 --json \'<rule>\'로 규칙 정의를 제공하세요',
    invalidJson: (message: string) => `잘못된 --json 규칙 페이로드: ${message}`,
    ruleFileReadError: (path: string, message: string) => `규칙 파일을 읽을 수 없습니다 ${path}: ${message}`,
    destinationMessengerRequiresPresetUrl: '--kind messenger-webhook에는 --preset과 --target이 필요합니다',
    destinationHttpRequiresUrl: '--kind http에는 --target이 필요합니다',
    destinationChannelRequiresTypeTarget: '--kind channel에는 --channel-type과 --channel-target이 필요합니다',
    destinationKindInvalid: (kind: string) => `알 수 없는 대상 종류: ${kind}`,
    summary: (destinations: number, rules: number) =>
      `이벤트 라우팅: 대상 ${destinations}개, 규칙 ${rules}개.\n자세한 내용은 \`sepilot events destinations\` / \`sepilot events rules\`를 사용하세요.`,
    enabled: '활성',
    disabled: '비활성',
    destinationTestSucceeded: (id: string) => `대상 테스트 성공: ${id}`,
    destinationTestFailed: (id: string, error: string) => `대상 테스트 실패: ${id} (${error})`,
    invalidLimit: (value: string) => `잘못된 --limit 값: ${value} (양의 정수여야 합니다)`,
    invalidEventStatus: (value: string) => `잘못된 --status 값: ${value} (received, processing, processed, unmatched, failed 중 하나여야 합니다)`,
    invalidDeliveryStatus: (value: string) => `잘못된 --status 값: ${value} (pending, delivering, delivered, dead 중 하나여야 합니다)`,
    invalidPayload: (message: string) => `잘못된 --payload 페이로드: ${message}`,
    invalidPayloadFile: (path: string, message: string) => `--payload-file을 파싱할 수 없습니다 ${path}: ${message}`,
    invalidHeaderPair: (pair: string) => `잘못된 --header 값: ${pair} (key=value 형식이어야 합니다)`,
  },
} as const

const EVENT_STATUSES = ['received', 'processing', 'processed', 'unmatched', 'failed'] as const
const DELIVERY_STATUSES = ['pending', 'delivering', 'delivered', 'dead'] as const

function eventsCopy() {
  return EVENTS_COPY[detectCliLocale()] ?? EVENTS_COPY.en
}

export interface EventsMutationOptions {
  url?: string
}

export interface EventsDestinationsAddOptions extends EventsMutationOptions {
  kind: 'messenger-webhook' | 'http' | 'channel' | 'notification'
  preset?: 'mattermost' | 'slack' | 'discord'
  target?: string
  secret?: string
  header?: string[]
  channelType?: string
  channelTarget?: string
  topic?: string
  disabled?: boolean
}

export interface EventsRuleFileOptions extends EventsMutationOptions {
  file?: string
  json?: string
}

export interface EventsRulesDryRunOptions extends EventsMutationOptions {
  source: string
  type: string
  payload?: string
  payloadFile?: string
}

export interface EventsLogOptions extends EventsMutationOptions {
  source?: string
  status?: DaemonEventRoutingEventStatus
  limit?: string
}

export interface EventsDeliveriesOptions extends EventsMutationOptions {
  status?: DaemonEventRoutingDeliveryStatus
  limit?: string
}

async function withEventsErrorHandling<T>(
  fn: () => Promise<T>,
  opts: { unavailableOn404?: boolean } = {},
): Promise<T> {
  try {
    return await fn()
  } catch (err) {
    if (err instanceof ApiHttpError && (err.status === 503 || (opts.unavailableOn404 && err.status === 404))) {
      const copy = eventsCopy()
      outputError({ ok: false, error: 'event-routing-unavailable' }, () => copy.unavailable)
      process.exit(1)
    }
    throw err
  }
}

function printRuleValidationErrorAndExit(err: unknown): never {
  if (err instanceof ApiHttpError && err.status === 400) {
    if (getOutputFormat() === 'json') {
      console.log(JSON.stringify({ ok: false, error: err.message }, null, 2))
    } else {
      console.error(chalk.red(err.message))
    }
    process.exit(1)
  }
  throw err
}

function parseHeaderPairs(pairs: string[]): Record<string, string> {
  const copy = eventsCopy()
  const headers: Record<string, string> = {}
  for (const pair of pairs) {
    const idx = pair.indexOf('=')
    if (idx <= 0) {
      throw new Error(copy.invalidHeaderPair(pair))
    }
    headers[pair.slice(0, idx)] = pair.slice(idx + 1)
  }
  return headers
}

function parseLimitOption(limit: string | undefined): number | undefined {
  if (limit === undefined) return undefined
  const copy = eventsCopy()
  const value = Number(limit)
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(copy.invalidLimit(limit))
  }
  return value
}

function parseEventStatusOption(status: string | undefined): DaemonEventRoutingEventStatus | undefined {
  if (status === undefined) return undefined
  const copy = eventsCopy()
  if (!(EVENT_STATUSES as readonly string[]).includes(status)) {
    throw new Error(copy.invalidEventStatus(status))
  }
  return status as DaemonEventRoutingEventStatus
}

function parseDeliveryStatusOption(status: string | undefined): DaemonEventRoutingDeliveryStatus | undefined {
  if (status === undefined) return undefined
  const copy = eventsCopy()
  if (!(DELIVERY_STATUSES as readonly string[]).includes(status)) {
    throw new Error(copy.invalidDeliveryStatus(status))
  }
  return status as DaemonEventRoutingDeliveryStatus
}

function buildDestinationInput(
  id: string,
  options: EventsDestinationsAddOptions,
  existing?: DaemonEventRoutingDestination,
): DaemonEventRoutingDestinationInput {
  const copy = eventsCopy()
  const enabled = !options.disabled
  const sameKindExisting = existing && existing.kind === options.kind ? existing : undefined
  switch (options.kind) {
    case 'messenger-webhook': {
      const existingUrl = sameKindExisting?.kind === 'messenger-webhook' ? sameKindExisting.url : undefined
      const url = options.target ?? existingUrl
      if (!options.preset || !url) {
        throw new Error(copy.destinationMessengerRequiresPresetUrl)
      }
      return { id, kind: 'messenger-webhook', enabled, preset: options.preset, url }
    }
    case 'http': {
      const existingHttp = sameKindExisting?.kind === 'http' ? sameKindExisting : undefined
      const url = options.target ?? existingHttp?.url
      if (!url) {
        throw new Error(copy.destinationHttpRequiresUrl)
      }
      const secret = options.secret ?? existingHttp?.secret
      const explicitHeaders = parseHeaderPairs(options.header ?? [])
      const headers = existingHttp
        ? { ...existingHttp.headers, ...explicitHeaders }
        : explicitHeaders
      return { id, kind: 'http', enabled, url, secret, headers }
    }
    case 'channel': {
      const existingChannel = sameKindExisting?.kind === 'channel' ? sameKindExisting : undefined
      const channelType = options.channelType ?? existingChannel?.channelType
      const channelTarget = options.channelTarget ?? existingChannel?.channelTarget
      if (!channelType || !channelTarget) {
        throw new Error(copy.destinationChannelRequiresTypeTarget)
      }
      return { id, kind: 'channel', enabled, channelType, channelTarget }
    }
    case 'notification': {
      const existingNotification = sameKindExisting?.kind === 'notification' ? sameKindExisting : undefined
      return { id, kind: 'notification', enabled, topic: options.topic ?? existingNotification?.topic }
    }
    default:
      throw new Error(copy.destinationKindInvalid(String(options.kind)))
  }
}

const REDACTED_PLACEHOLDER = '***redacted***'

function redactedDisplay(value: string): string {
  return value === REDACTED_PLACEHOLDER ? value : REDACTED_PLACEHOLDER
}

function formatDestination(destination: DaemonEventRoutingDestination): string {
  const copy = eventsCopy()
  const status = destination.enabled ? chalk.green(copy.enabled) : chalk.gray(copy.disabled)
  const head = `  ${chalk.bold(destination.id)}  [${destination.kind}]  ${status}`
  let tail: string
  switch (destination.kind) {
    case 'messenger-webhook':
      tail = `      preset=${destination.preset} url=${destination.url}`
      break
    case 'http': {
      const headerEntries = Object.entries(destination.headers ?? {})
        .map(([key, value]) => `${key}=${redactedDisplay(value)}`)
        .join(',')
      tail = `      url=${destination.url}`
        + (destination.secret ? ` secret=${redactedDisplay(destination.secret)}` : '')
        + (headerEntries ? ` headers=${headerEntries}` : '')
      break
    }
    case 'channel':
      tail = `      channelType=${destination.channelType} channelTarget=${destination.channelTarget}`
      break
    case 'notification':
      tail = `      topic=${destination.topic ?? '(default)'}`
      break
  }
  return `${head}\n${tail}`
}

function formatRule(rule: DaemonEventRoutingRule): string {
  const copy = eventsCopy()
  const status = rule.enabled ? chalk.green(copy.enabled) : chalk.gray(copy.disabled)
  const head = `  ${chalk.bold(rule.id)}  ${status}`
  const tail = `      match.source=${rule.match.source} types=${rule.match.types.join(',') || '*'} deliver=${rule.deliver.join(',')} process=${rule.process.mode}`
  return `${head}\n${tail}`
}

function formatEvent(event: DaemonEventRoutingEvent): string {
  return `  ${chalk.bold(event.id)}  [${event.status}]  ${event.source}/${event.type}${event.subject ? ` subject=${event.subject}` : ''}`
}

function formatEventDetail(detail: DaemonEventRoutingEventDetail): string {
  const lines = [formatEvent(detail.event)]
  for (const run of detail.ruleRuns) {
    lines.push(`    rule=${run.ruleId} status=${run.status} attempt=${run.attempt}${run.error ? ` error=${run.error}` : ''}`)
  }
  for (const delivery of detail.deliveries) {
    lines.push(`    delivery=${delivery.id} destination=${delivery.destinationId} status=${delivery.status}`)
  }
  return lines.join('\n')
}

function formatDelivery(delivery: DaemonEventRoutingDelivery): string {
  return `  ${chalk.bold(delivery.id)}  [${delivery.status}]  destination=${delivery.destinationId} event=${delivery.event.source}/${delivery.event.type}`
}

function parseRuleDefinition(options: EventsRuleFileOptions): Record<string, unknown> {
  const copy = eventsCopy()
  if (options.json) {
    try {
      return JSON.parse(options.json) as Record<string, unknown>
    } catch (err) {
      throw new Error(copy.invalidJson(err instanceof Error ? err.message : String(err)))
    }
  }
  if (options.file) {
    let raw: string
    try {
      raw = readFileSync(options.file, 'utf8')
    } catch (err) {
      throw new Error(copy.ruleFileReadError(options.file, err instanceof Error ? err.message : String(err)))
    }
    try {
      if (options.file.endsWith('.json')) {
        return JSON.parse(raw) as Record<string, unknown>
      }
      return YAML.parse(raw) as Record<string, unknown>
    } catch (err) {
      throw new Error(copy.ruleFileReadError(options.file, err instanceof Error ? err.message : String(err)))
    }
  }
  throw new Error(copy.ruleInputRequired)
}

export async function eventsDefaultCommand(options: EventsMutationOptions): Promise<void> {
  const copy = eventsCopy()
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    const [destinations, rules] = await Promise.all([
      client.eventRoutingDestinations(),
      client.eventRoutingRules(),
    ])
    output({ destinations, rules }, (data) => copy.summary(data.destinations.length, data.rules.length))
  }, { unavailableOn404: true })
}

export async function eventsDestinationsListCommand(options: EventsMutationOptions): Promise<void> {
  const copy = eventsCopy()
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    const destinations = await client.eventRoutingDestinations()
    output(destinations, (list) => (list.length === 0 ? copy.noDestinations : list.map(formatDestination).join('\n')))
  }, { unavailableOn404: true })
}

export async function eventsDestinationsAddCommand(
  id: string,
  options: EventsDestinationsAddOptions,
): Promise<void> {
  const copy = eventsCopy()
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    const existing = await client.eventRoutingDestinations()
    const existingDestination = existing.find((d) => d.id === id)
    const input = buildDestinationInput(id, options, existingDestination)
    const result = existingDestination
      ? await client.updateEventRoutingDestination(id, input)
      : await client.createEventRoutingDestination(input)
    output({ ok: true, destination: result }, () => `${copy.destinationSaved(id)}\n${formatDestination(result)}`)
  })
}

export async function eventsDestinationsRemoveCommand(id: string, options: EventsMutationOptions): Promise<void> {
  const copy = eventsCopy()
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    await client.deleteEventRoutingDestination(id)
    output({ ok: true, id }, () => copy.destinationRemoved(id))
  })
}

export async function eventsDestinationsTestCommand(id: string, options: EventsMutationOptions): Promise<void> {
  const copy = eventsCopy()
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    const result = await client.testEventRoutingDestination(id)
    output(result, (r) => (r.ok ? chalk.green(copy.destinationTestSucceeded(id)) : chalk.red(copy.destinationTestFailed(id, String(r.error)))))
  })
}

export async function eventsRulesListCommand(options: EventsMutationOptions): Promise<void> {
  const copy = eventsCopy()
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    const rules = await client.eventRoutingRules()
    output(rules, (list) => (list.length === 0 ? copy.noRules : list.map(formatRule).join('\n')))
  }, { unavailableOn404: true })
}

export async function eventsRulesAddCommand(options: EventsRuleFileOptions): Promise<void> {
  const copy = eventsCopy()
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    const definition = parseRuleDefinition(options) as unknown as DaemonEventRoutingRuleInput
    try {
      const rule = await client.createEventRoutingRule(definition)
      output({ ok: true, rule }, () => `${copy.ruleSaved(rule.id)}\n${formatRule(rule)}`)
    } catch (err) {
      printRuleValidationErrorAndExit(err)
    }
  })
}

export async function eventsRulesUpdateCommand(id: string, options: EventsRuleFileOptions): Promise<void> {
  const copy = eventsCopy()
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    const definition = parseRuleDefinition(options) as unknown as DaemonEventRoutingRuleInput
    try {
      const rule = await client.updateEventRoutingRule(id, { ...definition, id })
      output({ ok: true, rule }, () => `${copy.ruleSaved(rule.id)}\n${formatRule(rule)}`)
    } catch (err) {
      printRuleValidationErrorAndExit(err)
    }
  })
}

export async function eventsRulesRemoveCommand(id: string, options: EventsMutationOptions): Promise<void> {
  const copy = eventsCopy()
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    await client.deleteEventRoutingRule(id)
    output({ ok: true, id }, () => copy.ruleRemoved(id))
  })
}

export async function eventsRulesDryRunCommand(id: string, options: EventsRulesDryRunOptions): Promise<void> {
  const copy = eventsCopy()
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    let payload: Record<string, unknown> | undefined
    if (options.payloadFile) {
      let raw: string
      try {
        raw = readFileSync(options.payloadFile, 'utf8')
      } catch (err) {
        throw new Error(copy.invalidPayloadFile(options.payloadFile, err instanceof Error ? err.message : String(err)))
      }
      try {
        payload = JSON.parse(raw) as Record<string, unknown>
      } catch (err) {
        throw new Error(copy.invalidPayloadFile(options.payloadFile, err instanceof Error ? err.message : String(err)))
      }
    } else if (options.payload) {
      try {
        payload = JSON.parse(options.payload) as Record<string, unknown>
      } catch (err) {
        throw new Error(copy.invalidPayload(err instanceof Error ? err.message : String(err)))
      }
    }
    const result = await client.dryRunEventRoutingRule(id, {
      source: options.source,
      type: options.type,
      payload,
    })
    output(result, (r) => JSON.stringify(r, null, 2))
  })
}

export async function eventsLogCommand(options: EventsLogOptions): Promise<void> {
  const copy = eventsCopy()
  const status = parseEventStatusOption(options.status as unknown as string | undefined)
  const limit = parseLimitOption(options.limit)
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    const events = await client.eventRoutingEvents({
      source: options.source,
      status,
      limit,
    })
    output(events, (list) => (list.length === 0 ? copy.noEvents : list.map(formatEvent).join('\n')))
  }, { unavailableOn404: true })
}

export async function eventsShowCommand(eventId: string, options: EventsMutationOptions): Promise<void> {
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    const detail = await client.eventRoutingEvent(eventId)
    output(detail, formatEventDetail)
  })
}

export async function eventsDeliveriesCommand(options: EventsDeliveriesOptions): Promise<void> {
  const copy = eventsCopy()
  const status = parseDeliveryStatusOption(options.status as unknown as string | undefined)
  const limit = parseLimitOption(options.limit)
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    const deliveries = await client.eventRoutingDeliveries({
      status,
      limit,
    })
    output(deliveries, (list) => (list.length === 0 ? copy.noDeliveries : list.map(formatDelivery).join('\n')))
  }, { unavailableOn404: true })
}

export async function eventsDeliveriesReplayCommand(id: string, options: EventsMutationOptions): Promise<void> {
  const copy = eventsCopy()
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    const delivery = await client.replayEventRoutingDelivery(id)
    output({ ok: true, delivery }, () => `${copy.deliveryReplayed(id)}\n${formatDelivery(delivery)}`)
  })
}

export async function eventsDeliveriesAckCommand(id: string, options: EventsMutationOptions): Promise<void> {
  const copy = eventsCopy()
  await withEventsErrorHandling(async () => {
    const client = new DaemonClient(options.url)
    await client.ackEventRoutingDelivery(id)
    output({ ok: true, id }, () => copy.deliveryAcked(id))
  })
}

export const __testables = {
  buildDestinationInput,
  formatDestination,
  formatRule,
  formatEvent,
  formatDelivery,
  parseHeaderPairs,
  parseRuleDefinition,
}
