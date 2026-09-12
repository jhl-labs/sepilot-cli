/**
 * Backward-looking time parser for memory queries.
 *
 * Where `parseReminderTime` (in reminders.ts) is FUTURE-looking ("fire
 * this in 5 minutes"), this helper resolves PAST relative phrases into
 * absolute ISO timestamps so `memory.search ?createdAfter=yesterday`
 * etc. work without the caller having to compute Date math.
 *
 * Supported forms (case-insensitive English; bare Korean):
 *
 *   today / 오늘                  — start of today (00:00 local)
 *   yesterday / 어제              — start of yesterday
 *   <N>(s|m|h|d|w) ago            — exact relative offset
 *   <N> (seconds|minutes|hours|
 *        days|weeks) ago          — same, spelled out
 *   <N>(초|분|시간|일|주) 전       — Korean equivalents
 *   last week / 지난주             — exactly 7 days ago
 *   last month / 지난달            — exactly 30 days ago
 *   <ISO-8601 timestamp>          — passed straight through
 *
 * Returns null on unrecognised input so callers can return INVALID_INPUT
 * with a precise error message instead of silently treating it as "now".
 */

const ENGLISH_OFFSET_RE = /^(\d+)\s*(s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?|d|days?|w|weeks?)\s+ago$/i
const KOREAN_OFFSET_RE = /^(\d+)\s*(초|분|시간|시|일|주)\s*전$/
const ISO_DATE_TIME_HINT = /\d{4}-\d{2}-\d{2}/

export function parseRelativePastTimestamp(
  value: string,
  now: Date = new Date(),
): Date | null {
  const raw = value.trim()
  if (!raw) return null

  const lower = raw.toLowerCase()

  if (raw === '오늘' || lower === 'today') {
    return startOfDay(now)
  }
  if (raw === '어제' || lower === 'yesterday') {
    const d = startOfDay(now)
    d.setDate(d.getDate() - 1)
    return d
  }
  if (raw === '그저께' || raw === '그제') {
    const d = startOfDay(now)
    d.setDate(d.getDate() - 2)
    return d
  }
  if (raw === '지난주' || lower === 'last week') {
    return new Date(now.getTime() - 7 * 24 * 3600_000)
  }
  if (raw === '지난달' || lower === 'last month') {
    return new Date(now.getTime() - 30 * 24 * 3600_000)
  }
  if (raw === '작년' || lower === 'last year') {
    return new Date(now.getTime() - 365 * 24 * 3600_000)
  }

  const englishMatch = lower.match(ENGLISH_OFFSET_RE)
  if (englishMatch) {
    const amount = Number(englishMatch[1])
    const unitMs = englishUnitToMs(englishMatch[2])
    if (Number.isFinite(amount) && amount > 0 && unitMs > 0) {
      return new Date(now.getTime() - amount * unitMs)
    }
  }

  const koreanMatch = raw.match(KOREAN_OFFSET_RE)
  if (koreanMatch) {
    const amount = Number(koreanMatch[1])
    const unitMs = koreanUnitToMs(koreanMatch[2])
    if (Number.isFinite(amount) && amount > 0 && unitMs > 0) {
      return new Date(now.getTime() - amount * unitMs)
    }
  }

  if (ISO_DATE_TIME_HINT.test(raw) || /^\d{10,}$/.test(raw)) {
    const parsed = new Date(raw)
    if (!Number.isNaN(parsed.valueOf())) return parsed
  }

  return null
}

function startOfDay(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate())
}

function englishUnitToMs(unit: string): number {
  const u = unit.toLowerCase()
  if (u.startsWith('w')) return 7 * 24 * 3600_000
  if (u.startsWith('d')) return 24 * 3600_000
  if (u.startsWith('h')) return 3600_000
  if (u.startsWith('m') && u !== 'mo' && u !== 'mos') return 60_000
  if (u.startsWith('s')) return 1_000
  return 0
}

function koreanUnitToMs(unit: string): number {
  switch (unit) {
    case '초': return 1_000
    case '분': return 60_000
    case '시간':
    case '시': return 3600_000
    case '일': return 24 * 3600_000
    case '주': return 7 * 24 * 3600_000
    default: return 0
  }
}
