import * as chrono from 'chrono-node'
import cronParser from 'cron-parser'
import { parseReminderTime } from '../memory/reminders.js'

export class SchedulerParseError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'SchedulerParseError'
  }
}

export type ParsedSchedule =
  | { kind: 'oneshot'; runAt: number }
  | { kind: 'recurring'; cron: string; nextRunAt: number }

export interface ParseWhenOptions {
  now?: number
  timezone?: string
}

const CRON_RE = /^(\S+\s+){4}\S+$/
const INTERVAL_RE = /^(\d+)\s*(s|m|h|d|초|분|시간|일)$/i
const EVERY_RE = /^@every\s+(\d+)\s*(s|m|h|d)$/i
const CRON_NICKNAMES = new Set([
  '@yearly', '@annually', '@monthly', '@weekly', '@daily', '@midnight', '@hourly',
])
const INTERVAL_UNIT_MS: Record<string, number> = {
  s: 1000, m: 60_000, h: 3_600_000, d: 86_400_000,
  초: 1000, 분: 60_000, 시간: 3_600_000, 일: 86_400_000,
}

const KO_RECURRING_RE =
  /(매일|매주|매달|매월|매시간)\s*(?:(오전|오후)\s*)?(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?(?:\s*(월|화|수|목|금|토|일)요일)?/

const EN_RECURRING_RE =
  /every\s+(?:(monday|tuesday|wednesday|thursday|friday|saturday|sunday|day|hour)|(\d+)\s+(minute|hour|day)s?)\s*(?:at\s+)?(\d{1,2})?(?::(\d{2}))?\s*(am|pm)?/i

const KO_RECURRING_INTERVAL_RE =
  /^(?:매\s*)?(\d+)\s*(초|분|시간|일)\s*(?:마다|간격(?:으로)?|단위(?:로)?|에\s*한\s*번씩)$/

const DAY_OF_WEEK: Record<string, number> = {
  sunday: 0, monday: 1, tuesday: 2, wednesday: 3, thursday: 4, friday: 5, saturday: 6,
  일: 0, 월: 1, 화: 2, 수: 3, 목: 4, 금: 5, 토: 6,
}

const KO_INTERVAL_RE = /^(\d+)\s*(초|분|시간|일)\s*후$/

function intervalUnitCode(unit: string): 's' | 'm' | 'h' | 'd' {
  switch (unit.toLowerCase()) {
    case 's':
    case '초':
      return 's'
    case 'm':
    case 'minute':
    case '분':
      return 'm'
    case 'h':
    case 'hour':
    case '시간':
      return 'h'
    case 'd':
    case 'day':
    case '일':
      return 'd'
    default:
      return 'm'
  }
}

const MAX_RECURRING_INTERVAL_MS = 5 * 365 * 86_400_000

function recurringInterval(value: number, unit: string, now: number): ParsedSchedule {
  if (!Number.isFinite(value) || value <= 0) {
    throw new SchedulerParseError(`recurring interval must be positive: ${value}${unit}`)
  }
  const code = intervalUnitCode(unit)
  const unitMs = INTERVAL_UNIT_MS[code]
  const intervalMs = value * unitMs
  if (!Number.isFinite(intervalMs) || intervalMs > MAX_RECURRING_INTERVAL_MS) {
    throw new SchedulerParseError(
      `recurring interval exceeds 5-year maximum: ${value}${unit}`,
    )
  }
  return { kind: 'recurring', cron: `@every ${value}${code}`, nextRunAt: now + intervalMs }
}

export function parseEveryInterval(cron: string): number | null {
  const m = cron.match(EVERY_RE)
  if (!m) return null
  return parseInt(m[1], 10) * (INTERVAL_UNIT_MS[m[2].toLowerCase()] ?? 60_000)
}

const CRON_NICKNAME_ALIASES: Readonly<Record<string, string>> = {
  '@midnight': '@daily',
  '@annually': '@yearly',
}

function normalizeCronNickname(cron: string): string {
  const lowered = cron.trim().toLowerCase()
  const prefixed = CRON_NICKNAMES.has(lowered) ? lowered : `@${lowered}`
  if (!CRON_NICKNAMES.has(prefixed)) return cron
  return CRON_NICKNAME_ALIASES[prefixed] ?? prefixed
}

function isCronParserExpression(cron: string): boolean {
  return CRON_RE.test(cron) || CRON_NICKNAMES.has(normalizeCronNickname(cron))
}

export function nextRecurringRun(cron: string, after: number, timezone?: string): number {
  const everyMs = parseEveryInterval(cron)
  if (everyMs != null) return after + everyMs
  if (isCronParserExpression(cron)) {
    const interval = cronParser.parseExpression(normalizeCronNickname(cron), {
      currentDate: new Date(after),
      ...(timezone ? { tz: timezone } : {}),
    })
    return interval.next().getTime()
  }
  throw new SchedulerParseError(`invalid recurring expression: ${cron}`)
}

export function nextScheduledRun(cron: string, completedAt: number, timezone: string | undefined, metadata: Record<string, unknown> | null | undefined): number {
  const cadence = metadata?.schedulerInterval as { version?: unknown; mode?: unknown; anchor?: unknown } | undefined
  if (cadence && (cadence.version !== 1 || !['fixed-rate', 'fixed-delay'].includes(String(cadence.mode)))) {
    throw new SchedulerParseError('invalid scheduler interval policy')
  }
  if (cadence?.mode !== 'fixed-rate') return nextRecurringRun(cron, completedAt, timezone)
  const interval = parseEveryInterval(cron)
  if (!interval || typeof cadence.anchor !== 'number' || !Number.isFinite(cadence.anchor)) {
    throw new SchedulerParseError('fixed-rate requires an @every interval and a finite anchor')
  }
  return cadence.anchor + Math.max(1, Math.floor((completedAt - cadence.anchor) / interval) + 1) * interval
}

export function parseWhen(text: string, opts: ParseWhenOptions = {}): ParsedSchedule {
  const now = opts.now ?? Date.now()
  const timezone = opts.timezone
  const trimmed = text.trim()

  if (trimmed.toLowerCase() === '@now') return { kind: 'oneshot', runAt: now }

  const everyMatch = trimmed.match(EVERY_RE)
  if (everyMatch) {
    const value = parseInt(everyMatch[1], 10)
    return recurringInterval(value, everyMatch[2].toLowerCase(), now)
  }

  if (CRON_NICKNAMES.has(normalizeCronNickname(trimmed))) {
    const nick = normalizeCronNickname(trimmed)
    try {
      const interval = cronParser.parseExpression(nick, {
        currentDate: new Date(now),
        ...(timezone ? { tz: timezone } : {}),
      })
      return { kind: 'recurring', cron: nick, nextRunAt: interval.next().getTime() }
    } catch {
      throw new SchedulerParseError(`invalid cron nickname: ${trimmed}`)
    }
  }

  if (CRON_RE.test(trimmed)) {
    try {
      const interval = cronParser.parseExpression(trimmed, {
        currentDate: new Date(now),
        ...(timezone ? { tz: timezone } : {}),
      })
      return { kind: 'recurring', cron: trimmed, nextRunAt: interval.next().getTime() }
    } catch {
      throw new SchedulerParseError(`invalid cron: ${trimmed}`)
    }
  }

  const intervalMatch = trimmed.match(INTERVAL_RE)
  if (intervalMatch) {
    const value = parseInt(intervalMatch[1], 10)
    const unitMs = INTERVAL_UNIT_MS[intervalMatch[2].toLowerCase()] ?? 60_000
    return { kind: 'oneshot', runAt: now + value * unitMs }
  }

  const koIntervalMatch = trimmed.match(KO_INTERVAL_RE)
  if (koIntervalMatch) {
    const value = parseInt(koIntervalMatch[1], 10)
    const unitMs = INTERVAL_UNIT_MS[koIntervalMatch[2]] ?? 60_000
    return { kind: 'oneshot', runAt: now + value * unitMs }
  }

  const koRecurringInterval = trimmed.match(KO_RECURRING_INTERVAL_RE)
  if (koRecurringInterval) {
    return recurringInterval(parseInt(koRecurringInterval[1], 10), koRecurringInterval[2], now)
  }

  const koRecurring = trimmed.match(KO_RECURRING_RE)
  if (koRecurring) {
    const [, period, ampm, hourStr, minuteStr, dayOfWeek] = koRecurring
    let hour = parseInt(hourStr, 10)
    if (ampm === '오후' && hour < 12) hour += 12
    if (ampm === '오전' && hour === 12) hour = 0
    const minute = minuteStr ? parseInt(minuteStr, 10) : 0
    const dow = dayOfWeek ? DAY_OF_WEEK[dayOfWeek] : '*'
    let cron: string
    if (period === '매일') cron = `${minute} ${hour} * * *`
    else if (period === '매주' && dow !== '*') cron = `${minute} ${hour} * * ${dow}`
    else if (period === '매달' || period === '매월') cron = `${minute} ${hour} 1 * *`
    else if (period === '매시간') cron = `${minute} * * * *`
    else throw new SchedulerParseError(`cannot interpret korean recurring: ${trimmed}`)
    const interval = cronParser.parseExpression(cron, {
      currentDate: new Date(now),
      ...(timezone ? { tz: timezone } : {}),
    })
    return { kind: 'recurring', cron, nextRunAt: interval.next().getTime() }
  }

  const enRecurring = trimmed.match(EN_RECURRING_RE)
  if (enRecurring) {
    const [, dayWord, intervalNum, intervalUnit, hourStr, minuteStr, ampm] = enRecurring
    if (dayWord) {
      let hour = hourStr ? parseInt(hourStr, 10) : 0
      if (ampm?.toLowerCase() === 'pm' && hour < 12) hour += 12
      if (ampm?.toLowerCase() === 'am' && hour === 12) hour = 0
      const minute = minuteStr ? parseInt(minuteStr, 10) : 0
      const lower = dayWord.toLowerCase()
      let cron: string
      if (lower === 'day') cron = `${minute} ${hour} * * *`
      else if (lower === 'hour') cron = `${minute} * * * *`
      else cron = `${minute} ${hour} * * ${DAY_OF_WEEK[lower]}`
      const interval = cronParser.parseExpression(cron, {
        currentDate: new Date(now),
        ...(timezone ? { tz: timezone } : {}),
      })
      return { kind: 'recurring', cron, nextRunAt: interval.next().getTime() }
    }
    if (intervalNum && intervalUnit) {
      return recurringInterval(parseInt(intervalNum, 10), intervalUnit, now)
    }
  }

  const zoned = parseReminderTime(trimmed, new Date(now), timezone)
  if (zoned) {
    const zonedTs = zoned.getTime()
    if (zonedTs <= now) throw new SchedulerParseError(`time is in the past: ${trimmed}`)
    return { kind: 'oneshot', runAt: zonedTs }
  }

  const parsers = [chrono.casual]
  for (const parser of parsers) {
    const date = parser?.parseDate?.(trimmed, new Date(now), { forwardDate: true })
    if (date) {
      const ts = date.getTime()
      if (ts <= now) throw new SchedulerParseError(`time is in the past: ${trimmed}`)
      return { kind: 'oneshot', runAt: ts }
    }
  }

  throw new SchedulerParseError(`cannot interpret: ${trimmed}`)
}

interface JobLike {
  kind: 'oneshot' | 'recurring'
  runAt: number | null
  cron: string | null
}

export function formatWhen(job: JobLike, now: number = Date.now()): string {
  if (job.kind === 'recurring' && job.cron) return job.cron
  if (job.kind === 'oneshot' && job.runAt) {
    const deltaMs = job.runAt - now
    if (deltaMs < 60_000) return `in ${Math.round(deltaMs / 1000)} seconds`
    if (deltaMs < 3_600_000) return `in ${Math.round(deltaMs / 60_000)} minutes`
    if (deltaMs < 86_400_000) return `in ${Math.round(deltaMs / 3_600_000)} hours`
    return new Date(job.runAt).toISOString()
  }
  return '<unknown>'
}
