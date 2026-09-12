import { createHash } from 'node:crypto'
import { journalInventory, readJournal, validateJournalDate } from './journal-lifecycle.js'

export interface JournalSearchOptions {
  maxMatches?: number
  withinDays?: number
  contextLines?: number
  cursor?: string
  since?: string
  until?: string
}
export interface JournalSearchHit {
  date: string
  section?: string
  line: string
  lineNumber: number
  contextBefore?: string[]
  contextAfter?: string[]
}
const integer = (value: number | undefined, fallback: number, min: number, max: number) => {
  if (value === undefined) return fallback
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new Error(`Expected an integer from ${min} to ${max}`)
  return value
}

/** Search every retained date, with bounded IO and an explicit continuation.
 * The cursor is a position, not a snapshot: concurrent edits require a fresh search.
 */
export async function searchJournalPage(root: string, query: string, options: JournalSearchOptions = {}) {
  const normalized = query.trim().toLowerCase()
  if (!normalized) throw new Error('A query is required')
  const limit = integer(options.maxMatches, 25, 1, 200)
  const context = integer(options.contextLines, 0, 0, 10)
  const recent = integer(options.withinDays, Number.MAX_SAFE_INTEGER, 1, Number.MAX_SAFE_INTEGER)
  for (const date of [options.since, options.until]) if (date !== undefined) validateJournalDate(date)
  if (options.since && options.until && options.since > options.until) throw new Error('since must not follow until')
  const fingerprint = createHash('sha256').update(JSON.stringify([normalized, options.since, options.until, recent])).digest('hex')
  let position: { date: string; line: number } | undefined
  if (options.cursor !== undefined) {
    try {
      if (options.cursor.length > 1024) throw new Error()
      const value = JSON.parse(Buffer.from(options.cursor, 'base64url').toString('utf8'))
      validateJournalDate(value.date)
      if (value.version !== 1 || value.query !== fingerprint || !Number.isSafeInteger(value.line) || value.line < 0) throw new Error()
      position = value
    } catch { throw new Error('Invalid journal cursor; restart the search with the same query and date range') }
  }
  const dates = (await journalInventory(root)).filter(({ date }) => (!options.since || date >= options.since) && (!options.until || date <= options.until)).slice(0, recent)
  const pending = dates.filter(({ date }) => !position || date <= position.date)
  const matches: JournalSearchHit[] = []
  let scannedDates = 0
  const continuation = (date: string, line: number) => Buffer.from(JSON.stringify({ version: 1, query: fingerprint, date, line })).toString('base64url')
  for (const { date } of pending) {
    if (scannedDates >= 100) return { matches, scannedDates, nextCursor: continuation(date, 0) }
    const content = await readJournal(root, date)
    scannedDates++
    if (content === undefined) continue
    const lines = content.split('\n')
    let section: string | undefined
    for (let i = 0; i < lines.length; i++) {
      const heading = lines[i]!.match(/^##\s+(.+?)\s*$/)
      if (heading) { section = heading[1]!.trim(); continue }
      if (position?.date === date && i < position.line) continue
      if (!lines[i]!.toLowerCase().includes(normalized)) continue
      if (matches.length >= limit) return { matches, scannedDates, nextCursor: continuation(date, i) }
      matches.push({ date, section, line: lines[i]!.trim(), lineNumber: i + 1,
        ...(context ? {
          contextBefore: lines.slice(Math.max(0, i - context), i).map((line) => line.trimEnd()),
          contextAfter: lines.slice(i + 1, i + 1 + context).map((line) => line.trimEnd()),
        } : {}),
      })
    }
  }
  return { matches, scannedDates, nextCursor: null }
}
