import { randomUUID } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import { createLogger } from '../logger.js'
import { isMemoryWritableInScope } from './scope.js'

const log = createLogger('memory:reminders')

export interface Reminder {
  id: string
  dueAt: string
  content: string
  scopeTags: string[]
  channelType?: string
  chatId?: string
  createdAt: string
  firedAt?: string
  cancelledAt?: string
  reason?: string
  /**
   * IANA timezone (e.g. 'Asia/Seoul') the wall-clock time in the original
   * request was expressed in. When set, callers should resolve the due instant
   * with parseReminderTime(value, now, timezone) so "tomorrow 9am" means 9am in
   * the user's zone, not the server's. Absent → server-local (legacy behavior).
   */
  timezone?: string
}

export interface RemindersStore {
  list(now?: Date): Promise<Reminder[]>
  listForScope(scopeTags: string[], now?: Date): Promise<Reminder[]>
  add(input: Omit<Reminder, 'id' | 'createdAt' | 'firedAt' | 'cancelledAt'>): Promise<Reminder>
  cancel(id: string, reason?: string): Promise<Reminder | null>
  markFired(id: string, firedAt?: string): Promise<Reminder | null>
  due(now?: Date): Promise<Reminder[]>
  close(): Promise<void>
}

const FILE_HEADER = '# sepilotd reminders.jsonl — one Reminder per line. Do not edit by hand.'

/**
 * Append-only JSONL-backed reminders store. Each mutation writes a new
 * line with the full Reminder snapshot; on init we replay to compute
 * the latest state for each id.
 */
export class JsonlRemindersStore implements RemindersStore {
  private readonly reminders = new Map<string, Reminder>()
  private initialized = false

  constructor(private readonly path: string) {}

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return
    await mkdir(dirname(this.path), { recursive: true })
    try {
      const content = await readFile(this.path, 'utf-8')
      for (const line of content.split('\n')) {
        const trimmed = line.trim()
        if (!trimmed || trimmed.startsWith('#')) continue
        try {
          const entry = JSON.parse(trimmed) as Reminder
          if (entry.id) this.reminders.set(entry.id, entry)
        } catch {
          // skip malformed line
        }
      }
    } catch {
      // file missing — start fresh
      try {
        await writeFile(this.path, `${FILE_HEADER}\n`, 'utf-8')
      } catch (err) {
        log.warn('Failed to seed reminders file', { error: String(err) })
        throw err
      }
    }
    this.initialized = true
  }

  private async appendRecord(reminder: Reminder): Promise<void> {
    await mkdir(dirname(this.path), { recursive: true })
    const line = `${JSON.stringify(reminder)}\n`
    try {
      await writeFile(this.path, line, { flag: 'a', encoding: 'utf-8' })
    } catch (err) {
      log.warn('Failed to append reminder record', { error: String(err) })
      throw err
    }
  }

  async list(): Promise<Reminder[]> {
    await this.ensureInitialized()
    return Array.from(this.reminders.values()).sort(
      (a, b) => Date.parse(a.dueAt) - Date.parse(b.dueAt),
    )
  }

  async listForScope(scopeTags: string[]): Promise<Reminder[]> {
    const all = await this.list()
    return all.filter((reminder) => {
      if (reminder.firedAt || reminder.cancelledAt) return false
      return isMemoryWritableInScope(reminder.scopeTags, scopeTags)
    }).sort((a, b) => Date.parse(a.dueAt) - Date.parse(b.dueAt))
  }

  async add(
    input: Omit<Reminder, 'id' | 'createdAt' | 'firedAt' | 'cancelledAt'>,
  ): Promise<Reminder> {
    await this.ensureInitialized()
    const id = randomUUID()
    const reminder: Reminder = {
      ...input,
      id,
      createdAt: new Date().toISOString(),
    }
    await this.appendRecord(reminder)
    this.reminders.set(id, reminder)
    return reminder
  }

  async cancel(id: string, reason?: string): Promise<Reminder | null> {
    await this.ensureInitialized()
    const existing = this.reminders.get(id)
    if (!existing || existing.firedAt || existing.cancelledAt) return null
    const updated: Reminder = {
      ...existing,
      cancelledAt: new Date().toISOString(),
      reason: reason ?? existing.reason,
    }
    await this.appendRecord(updated)
    this.reminders.set(id, updated)
    return updated
  }

  async markFired(id: string, firedAt = new Date().toISOString()): Promise<Reminder | null> {
    await this.ensureInitialized()
    const existing = this.reminders.get(id)
    if (!existing || existing.firedAt || existing.cancelledAt) return null
    const updated: Reminder = {
      ...existing,
      firedAt,
    }
    await this.appendRecord(updated)
    this.reminders.set(id, updated)
    return updated
  }

  async due(now: Date = new Date()): Promise<Reminder[]> {
    await this.ensureInitialized()
    const cutoff = now.getTime()
    return Array.from(this.reminders.values()).filter((reminder) => {
      if (reminder.firedAt || reminder.cancelledAt) return false
      return Date.parse(reminder.dueAt) <= cutoff
    })
  }

  async close(): Promise<void> {
    this.initialized = false
    this.reminders.clear()
  }
}

export interface RemindersSchedulerOptions {
  intervalMs?: number
  onDue: (reminder: Reminder) => Promise<void> | void
}

/** Periodically sweeps a RemindersStore for due reminders and fires onDue. */
export class RemindersScheduler {
  private readonly store: RemindersStore
  private readonly options: Required<RemindersSchedulerOptions>
  private timer?: ReturnType<typeof setInterval>
  private firing = false

  constructor(store: RemindersStore, options: RemindersSchedulerOptions) {
    this.store = store
    this.options = {
      intervalMs: options.intervalMs ?? 60_000,
      onDue: options.onDue,
    }
  }

  start(): void {
    if (this.timer) return
    this.timer = setInterval(() => {
      void this.sweep()
    }, this.options.intervalMs)
    this.timer.unref?.()
  }

  stop(): void {
    if (!this.timer) return
    clearInterval(this.timer)
    this.timer = undefined
  }

  /** Visible to tests so they can flush deterministically. */
  async sweep(now: Date = new Date()): Promise<number> {
    if (this.firing) return 0
    this.firing = true
    let fired = 0
    try {
      const due = await this.store.due(now)
      for (const reminder of due) {
        // Claim the reminder BEFORE delivering. markFired is idempotent and
        // returns null if the reminder was already fired or cancelled, so a
        // crash after delivery — or a re-entrant sweep of the same overdue
        // backlog after downtime — can never re-deliver it. This trades a lost
        // delivery on a hard crash mid-onDue for never spamming the user with
        // duplicates, which is the safer default for notifications.
        const claimed = await this.store.markFired(reminder.id, now.toISOString())
        if (!claimed) continue
        try {
          await this.options.onDue(reminder)
          fired += 1
        } catch (err) {
          log.warn('reminder fire failed', {
            reminderId: reminder.id,
            error: String(err),
          })
        }
      }
    } finally {
      this.firing = false
    }
    return fired
  }
}

const RELATIVE_DURATION_PATTERN = /^(\d+)\s*(s|sec|secs|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|초|분|시간|일)$/i
const ISO_DATE_TIME_HINT = /\d{4}-\d{2}-\d{2}/

const KO_DAY_OFFSETS: Record<string, number> = {
  '오늘': 0, '내일': 1, '모레': 2, '글피': 3,
  '어제': -1, // never accepted but kept for completeness; rejected later
}
const EN_DAY_OFFSETS: Record<string, number> = {
  today: 0, tomorrow: 1, tonight: 0,
}

const KO_WEEKDAYS: Record<string, number> = {
  '일요일': 0, '일': 0,
  '월요일': 1, '월': 1,
  '화요일': 2, '화': 2,
  '수요일': 3, '수': 3,
  '목요일': 4, '목': 4,
  '금요일': 5, '금': 5,
  '토요일': 6, '토': 6,
}
const EN_WEEKDAYS: Record<string, number> = {
  sunday: 0, sun: 0,
  monday: 1, mon: 1,
  tuesday: 2, tue: 2, tues: 2,
  wednesday: 3, wed: 3,
  thursday: 4, thu: 4, thurs: 4,
  friday: 5, fri: 5,
  saturday: 6, sat: 6,
}

/**
 * Parse "in 30m", "in 2 hours", "tomorrow 9am", "내일 오후 3시",
 * "다음주 월요일 9시", "next monday 3pm", "2026-05-09T15:00:00Z",
 * or a Date-acceptable ISO string into an absolute timestamp.
 * Returns null on failure (callers should reject INVALID_INPUT).
 *
 * When `timezone` (IANA name) is given, wall-clock expressions ("tomorrow 9am",
 * "월요일 15시") are interpreted in that zone and returned as the correct UTC
 * instant. Timezone-independent forms (relative durations, ISO/epoch) are
 * unaffected. Omitting `timezone` preserves the exact legacy server-local
 * behavior.
 */
export function parseReminderTime(
  value: string,
  now: Date = new Date(),
  timezone?: string,
): Date | null {
  if (!timezone) return parseReminderTimeNaive(value, now)
  const raw = value.trim()
  if (!raw) return null
  // Relative durations and ISO/epoch timestamps resolve to the same absolute
  // instant in every zone — parse them against the real `now`.
  if (isTimezoneIndependent(raw)) return parseReminderTimeNaive(raw, now)
  // Wall-clock expressions: parse the calendar/clock components as if `now` were
  // the user's local time (so the correct day/weekday is picked in their zone),
  // then map those wall components back to a real UTC instant for that zone.
  try {
    const naiveNow = zonedNaive(now, timezone)
    const naive = parseReminderTimeNaive(raw, naiveNow)
    if (!naive) return null
    return naiveWallToUtc(naive, timezone)
  } catch {
    // Invalid IANA zone or Intl failure — fall back to server-local parsing.
    return parseReminderTimeNaive(value, now)
  }
}

function isTimezoneIndependent(raw: string): boolean {
  if (ISO_DATE_TIME_HINT.test(raw) || /^\d{10,}$/.test(raw)) return true
  const lower = raw.toLowerCase()
  const inMatch = lower.match(/^(?:in|after)\s+(.+)$/)
  const durationCandidate = (inMatch?.[1] ?? lower).replace(/\s+/g, '')
  return RELATIVE_DURATION_PATTERN.test(durationCandidate)
}

/**
 * Return a Date whose LOCAL getters (getHours, getDate, …) read back `date`'s
 * wall-clock in `timeZone`. Used to run the naive parser as if the process were
 * in the user's zone.
 */
function zonedNaive(date: Date, timeZone: string): Date {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).formatToParts(date)
  const get = (type: string): number => Number(parts.find((part) => part.type === type)?.value)
  let hour = get('hour')
  if (hour === 24) hour = 0 // Intl can emit '24' for midnight in some engines.
  return new Date(get('year'), get('month') - 1, get('day'), hour, get('minute'), get('second'))
}

/**
 * Interpret the LOCAL wall-clock components of `naive` as a wall-clock in
 * `timeZone` and return the corresponding real UTC instant.
 */
function naiveWallToUtc(naive: Date, timeZone: string): Date {
  const asUtc = Date.UTC(
    naive.getFullYear(),
    naive.getMonth(),
    naive.getDate(),
    naive.getHours(),
    naive.getMinutes(),
    naive.getSeconds(),
  )
  // Offset (ms) that the zone runs ahead of UTC at this instant.
  const offset = zoneOffsetMs(new Date(asUtc), timeZone)
  return new Date(asUtc - offset)
}

function zoneOffsetMs(date: Date, timeZone: string): number {
  const naive = zonedNaive(date, timeZone)
  const asUtc = Date.UTC(
    naive.getFullYear(),
    naive.getMonth(),
    naive.getDate(),
    naive.getHours(),
    naive.getMinutes(),
    naive.getSeconds(),
  )
  return asUtc - date.getTime()
}

function parseReminderTimeNaive(value: string, now: Date = new Date()): Date | null {
  const raw = value.trim()
  if (!raw) return null

  // Bare day-anchor words
  const lower = raw.toLowerCase()
  if (raw === '내일' || lower === 'tomorrow') {
    const next = startOfDay(now)
    next.setDate(next.getDate() + 1)
    next.setHours(9, 0, 0, 0)
    return next
  }
  if (raw === '오늘' || lower === 'today') {
    const today = new Date(now)
    today.setMinutes(0, 0, 0)
    today.setHours(today.getHours() + 1)
    return today
  }
  if (raw === '모레') {
    const next = startOfDay(now)
    next.setDate(next.getDate() + 2)
    next.setHours(9, 0, 0, 0)
    return next
  }

  // "in 30m" / "in 2 hours" / "after 1d"
  const inMatch = lower.match(/^(?:in|after)\s+(.+)$/)
  const durationCandidate = inMatch?.[1] ?? lower
  const durationMatch = durationCandidate.replace(/\s+/g, '').match(RELATIVE_DURATION_PATTERN)
  if (durationMatch) {
    const amount = Number(durationMatch[1])
    const unit = durationMatch[2].toLowerCase()
    if (Number.isFinite(amount) && amount > 0) {
      const ms = amount * unitToMs(unit)
      return new Date(now.getTime() + ms)
    }
  }

  // ISO-like timestamp
  if (ISO_DATE_TIME_HINT.test(raw) || /^\d{10,}$/.test(raw)) {
    const parsed = new Date(raw)
    if (!Number.isNaN(parsed.valueOf()) && parsed.getTime() > now.getTime() - 60_000) {
      return parsed
    }
  }

  // Korean: "내일 오후 3시", "오늘 23시", "내일 9:30", "내일 오전 9시 30분"
  const koDayTime = matchKoDayTime(raw, now)
  if (koDayTime) return koDayTime

  // Korean: "월요일 9시", "다음주 화요일 오후 3시", "다음 월요일 9시"
  const koWeekdayTime = matchKoWeekdayTime(raw, now)
  if (koWeekdayTime) return koWeekdayTime

  // English: "tomorrow 3pm", "today at 9", "tonight 8pm"
  const enDayTime = matchEnDayTime(lower, now)
  if (enDayTime) return enDayTime

  // English: "next monday", "next monday 3pm", "monday 9am"
  const enWeekdayTime = matchEnWeekdayTime(lower, now)
  if (enWeekdayTime) return enWeekdayTime

  return null
}

function startOfDay(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate())
}

function applyTimeOfDay(target: Date, hours: number, minutes: number, now: Date): Date {
  const result = new Date(target)
  result.setHours(hours, minutes, 0, 0)
  if (result.getTime() <= now.getTime()) {
    // Same-day request that's already past — caller patterns handle their own rules.
  }
  return result
}

function parseHourMinute(input: string): { hour: number; minute: number } | null {
  const text = input.trim()
  if (!text) return null

  // 한국어 시·분 표현
  const koAmpm = text.match(/^(오전|오후)\s*(\d{1,2})(?:시)?(?:\s*(\d{1,2})\s*분)?$/)
  if (koAmpm) {
    let hour = Number(koAmpm[2])
    const minute = koAmpm[3] ? Number(koAmpm[3]) : 0
    if (!isHour(hour) || !isMinute(minute)) return null
    if (koAmpm[1] === '오후' && hour < 12) hour += 12
    if (koAmpm[1] === '오전' && hour === 12) hour = 0
    return { hour, minute }
  }
  const koHourMin = text.match(/^(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?$/)
  if (koHourMin) {
    const hour = Number(koHourMin[1])
    const minute = koHourMin[2] ? Number(koHourMin[2]) : 0
    if (!isHour(hour) || !isMinute(minute)) return null
    return { hour, minute }
  }

  // 영어 H[:MM] (am|pm)?
  const enAmpm = text.match(/^(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)$/i)
  if (enAmpm) {
    let hour = Number(enAmpm[1])
    const minute = enAmpm[2] ? Number(enAmpm[2]) : 0
    // am/pm only accepts 1-12 in the hour slot.
    if (hour < 1 || hour > 12) return null
    if (!isMinute(minute)) return null
    const isPm = /^p/i.test(enAmpm[3])
    if (isPm && hour < 12) hour += 12
    if (!isPm && hour === 12) hour = 0
    return { hour, minute }
  }

  // 24h: 9:30 / 09 / 23:15
  const numeric = text.match(/^(\d{1,2})(?::(\d{2}))?$/)
  if (numeric) {
    const hour = Number(numeric[1])
    const minute = numeric[2] ? Number(numeric[2]) : 0
    if (!isHour(hour) || !isMinute(minute)) return null
    return { hour, minute }
  }

  return null
}

function isHour(value: number): boolean {
  return Number.isInteger(value) && value >= 0 && value <= 23
}

function isMinute(value: number): boolean {
  return Number.isInteger(value) && value >= 0 && value <= 59
}

function matchKoDayTime(raw: string, now: Date): Date | null {
  const m = raw.match(/^(오늘|내일|모레|글피)\s+(.+)$/)
  if (!m) return null
  const offset = KO_DAY_OFFSETS[m[1]]
  if (typeof offset !== 'number' || offset < 0) return null
  const time = parseHourMinute(m[2])
  if (!time) return null
  const target = startOfDay(now)
  target.setDate(target.getDate() + offset)
  return applyTimeOfDay(target, time.hour, time.minute, now)
}

function matchKoWeekdayTime(raw: string, now: Date): Date | null {
  const m = raw.match(/^(?:(다음주|이번주|다음)\s+)?(월요일|화요일|수요일|목요일|금요일|토요일|일요일|월|화|수|목|금|토|일)(?:\s+(.+))?$/)
  if (!m) return null
  const weekday = KO_WEEKDAYS[m[2]]
  if (typeof weekday !== 'number') return null
  const target = nextWeekday(now, weekday, m[1] === '다음주' || m[1] === '다음')
  const time = m[3] ? parseHourMinute(m[3]) : { hour: 9, minute: 0 }
  if (!time) return null
  return applyTimeOfDay(target, time.hour, time.minute, now)
}

function matchEnDayTime(lower: string, now: Date): Date | null {
  const m = lower.match(/^(today|tomorrow|tonight)(?:\s+at)?\s+(.+)$/)
  if (!m) return null
  const offset = EN_DAY_OFFSETS[m[1]]
  if (typeof offset !== 'number') return null
  const time = parseHourMinute(m[2])
  if (!time) return null
  if (m[1] === 'tonight' && time.hour < 12) time.hour += 12
  const target = startOfDay(now)
  target.setDate(target.getDate() + offset)
  return applyTimeOfDay(target, time.hour, time.minute, now)
}

function matchEnWeekdayTime(lower: string, now: Date): Date | null {
  const m = lower.match(/^(?:(next|this)\s+)?(sunday|monday|tuesday|wednesday|thursday|friday|saturday|sun|mon|tue|tues|wed|thu|thurs|fri|sat)(?:\s+(?:at\s+)?(.+))?$/)
  if (!m) return null
  const weekday = EN_WEEKDAYS[m[2]]
  if (typeof weekday !== 'number') return null
  const target = nextWeekday(now, weekday, m[1] === 'next')
  const time = m[3] ? parseHourMinute(m[3]) : { hour: 9, minute: 0 }
  if (!time) return null
  return applyTimeOfDay(target, time.hour, time.minute, now)
}

function nextWeekday(now: Date, target: number, _forceNextWeek: boolean): Date {
  // Treat "next monday" / "다음주 월요일" / bare "monday" the same way:
  // the upcoming Monday after today. If today is the target weekday, jump
  // a full week — never schedule a reminder for "today, but earlier".
  const start = startOfDay(now)
  const today = start.getDay()
  let delta = (target - today + 7) % 7
  if (delta === 0) delta = 7
  const result = new Date(start)
  result.setDate(start.getDate() + delta)
  return result
}

function unitToMs(unit: string): number {
  switch (unit) {
    case 's':
    case 'sec':
    case 'secs':
    case 'seconds':
    case '초':
      return 1000
    case 'm':
    case 'min':
    case 'mins':
    case 'minute':
    case 'minutes':
    case '분':
      return 60_000
    case 'h':
    case 'hr':
    case 'hrs':
    case 'hour':
    case 'hours':
    case '시간':
      return 3_600_000
    case 'd':
    case 'day':
    case 'days':
    case '일':
      return 86_400_000
    default:
      return 60_000
  }
}
