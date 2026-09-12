import { searchJournalPage, type JournalSearchOptions } from './journal-search.js'
import { journalInventory, pendingJournalContext, readJournal, restoreJournal, withJournalLock } from './journal-lifecycle.js'
import { randomUUID } from 'node:crypto'
import { constants } from 'node:fs'
import { access, mkdir, readFile, writeFile, rename, rm } from 'node:fs/promises'
import { join } from 'node:path'

export interface FileMemoryPromptContext {
  longTermMemory?: string
  todayNote?: string
  yesterdayNote?: string
}

export interface FileMemorySection {
  title: string
  content: string
}

export interface FileMemoryOptions {
  onLongTermChange?: (before: string, after: string) => Promise<void>
  beforePromptRead?: () => Promise<void>
  maxLongTermChars?: number
  maxDailyNoteChars?: number
}

const DEFAULT_OPTIONS: Required<FileMemoryOptions> = {
  onLongTermChange: async () => {},
  beforePromptRead: async () => {},
  maxLongTermChars: 4000,
  maxDailyNoteChars: 2000,
}

const DAILY_PROMPT_PRIORITY_SECTIONS = [
  'Open Loop Queue',
  'Open Loops',
  'Backlog',
  'Reflection Ledger',
  'REM Summary',
  'Tasks',
  'Questions',
]

export class FileMemory {
  private readonly options: Required<FileMemoryOptions>
  private readonly hasChangeListener: boolean
  private ownerScopeTags?: string[]

  setOwnerScopeTags(tags: string[]): void { this.ownerScopeTags = [...tags] }

  constructor(
    private readonly memoryDir: string,
    options: FileMemoryOptions = {},
  ) {
    this.options = { ...DEFAULT_OPTIONS, ...options }
    this.hasChangeListener = Boolean(options.onLongTermChange)
  }

  async init(now = new Date()): Promise<void> {
    await this.recoverLongTermChange()
    await mkdir(this.dailyDir(), { recursive: true })
    if (this.ownerScopeTags) await writeFile(join(this.memoryDir, '.scope.json'), JSON.stringify(this.ownerScopeTags), { mode: 0o600 })
    await this.ensureFile(
      this.getMemoryPath(),
      '# Long-Term Memory\n\n',
    )
    await this.ensureFile(
      this.getDailyNotePath(now),
      `# ${formatDateKey(now)}\n\n`,
    )
  }

  /** Clear this bucket only; the global bucket also contains other owners' scopes. */
  async clearAll(reconcileSemantic = true): Promise<void> {
    await this.recoverLongTermChange()
    await mkdir(this.memoryDir, { recursive: true })
    await this.writeLongTerm(await readFile(this.getMemoryPath(), 'utf8').catch((error: NodeJS.ErrnoException) => {
      if (error.code === 'ENOENT') return ''
      throw error
    }), '# Long-Term Memory\n\n', reconcileSemantic)
    await withJournalLock(this.memoryDir, async () => {
      await rm(this.dailyDir(), { recursive: true, force: true })
      await rm(join(this.memoryDir, 'journal-archive'), { recursive: true, force: true })
      await rm(join(this.memoryDir, 'journal-maintenance.json'), { force: true })
      await rm(join(this.memoryDir, 'consolidation'), { recursive: true, force: true })
    })
  }

  getMemoryPath(): string {
    return join(this.memoryDir, 'MEMORY.md')
  }

  getDailyNotePath(date: Date): string {
    return join(this.dailyDir(), `${formatDateKey(date)}.md`)
  }

  async getDailyNoteReadPath(date: Date): Promise<string> {
    const entry = (await journalInventory(this.memoryDir)).find((item) => item.date === formatDateKey(date))
    return entry?.archived ? join(this.memoryDir, 'journal-archive', `${entry.date}.md.gz`) : this.getDailyNotePath(date)
  }

  async getPromptContext(now = new Date()): Promise<FileMemoryPromptContext> {
    await this.recoverLongTermChange()
    await this.options.beforePromptRead()
    const today = new Date(now)
    const yesterday = new Date(now)
    yesterday.setDate(yesterday.getDate() - 1)

    const pending = await pendingJournalContext(this.memoryDir, formatDateKey(yesterday), Math.min(700, Math.floor(this.options.maxDailyNoteChars / 2)))
    const [longTermMemory, todayNote, yesterdayNote] = await Promise.all([
      this.readPromptFile(
        this.getMemoryPath(),
        this.options.maxLongTermChars,
        [],
        { stripPrivate: true },
      ),
      this.readPromptFile(
        this.getDailyNotePath(today),
        Math.max(0, this.options.maxDailyNoteChars - pending.length - (pending ? 2 : 0)),
        DAILY_PROMPT_PRIORITY_SECTIONS,
      ),
      this.readPromptFile(
        this.getDailyNotePath(yesterday),
        this.options.maxDailyNoteChars,
        DAILY_PROMPT_PRIORITY_SECTIONS,
      ),
    ])

    return {
      longTermMemory,
      todayNote: [pending, todayNote].filter(Boolean).join('\n\n') || undefined,
      yesterdayNote,
    }
  }

  async readMemory(): Promise<string | undefined> {
    return this.readTextFile(this.getMemoryPath())
  }

  async readMemorySections(): Promise<FileMemorySection[]> {
    const content = await this.readTextFile(this.getMemoryPath())
    return parseMarkdownSections(content ?? '')
  }

  async readMemorySection(sectionTitle: string): Promise<string | undefined> {
    const content = await this.readTextFile(this.getMemoryPath())
    if (!content) {
      return undefined
    }

    const section = findMarkdownSection(content, sectionTitle)
    if (!section) {
      return undefined
    }

    const body = normalizeMarkdownSectionBody(content.slice(section.bodyStart, section.bodyEnd))
    return body.length > 0 ? body : undefined
  }

  async mergeMemorySectionItems(
    sectionTitle: string,
    items: string[],
  ): Promise<number> {
    await this.init()

    const currentItems = parseBulletItems(
      await this.readMemorySection(sectionTitle),
    )
    const seen = new Set(
      currentItems.map((item) => normalizeBulletItem(item)),
    )

    const merged = [...currentItems]
    let added = 0

    for (const item of items) {
      const normalized = normalizeBulletItem(item)
      if (!normalized || seen.has(normalized)) {
        continue
      }
      seen.add(normalized)
      merged.push(item.trim().replace(/^-+\s*/, ''))
      added++
    }

    if (added === 0) {
      return 0
    }

    await this.replaceMemorySection(
      sectionTitle,
      merged.map((item) => `- ${item}`).join('\n'),
    )

    return added
  }

  async readDailyNote(date = new Date()): Promise<string | undefined> {
    return readJournal(this.memoryDir, formatDateKey(date))
  }

  async searchDailyNotesPage(query: string, options: JournalSearchOptions = {}) {
    return searchJournalPage(this.memoryDir, query, options)
  }

  /** Search across daily notes for a substring (case-insensitive). Walks
   *  the recent dates first so the freshest matches surface early. Returns
   *  up to `maxMatches` hits as { date, section, line, [contextBefore], [contextAfter] }.
   *  Caller can pass `withinDays` to bound the scan (default 90) and
   *  `contextLines` to capture N lines on each side of the match (default 0). */
  async searchDailyNotes(
    query: string,
    options: { maxMatches?: number; withinDays?: number; contextLines?: number } = {},
  ): Promise<Array<{
    date: string
    section?: string
    line: string
    lineNumber: number
    contextBefore?: string[]
    contextAfter?: string[]
  }>> {
    const trimmed = query.trim()
    if (!trimmed) return []
    const lowerQuery = trimmed.toLowerCase()
    const maxMatches = Math.max(1, Math.min(Math.floor(options.maxMatches ?? 25), 200))
    const contextLines = Math.max(0, Math.min(Math.floor(options.contextLines ?? 0), 10))
    const dates = await this.listDailyNoteDates(options.withinDays ?? 90)
    const hits: Array<{
      date: string
      section?: string
      line: string
      lineNumber: number
      contextBefore?: string[]
      contextAfter?: string[]
    }> = []
    for (const dateKey of dates) {
      if (hits.length >= maxMatches) break
      const isoMatch = dateKey.match(/^(\d{4})-(\d{2})-(\d{2})$/)
      if (!isoMatch) continue
      const date = new Date(Number(isoMatch[1]), Number(isoMatch[2]) - 1, Number(isoMatch[3]))
      const content = await this.readDailyNote(date)
      if (!content) continue
      const lines = content.split('\n')
      let currentSection: string | undefined
      for (let i = 0; i < lines.length && hits.length < maxMatches; i++) {
        const line = lines[i]
        const sectionMatch = line.match(/^##\s+(.+?)\s*$/)
        if (sectionMatch) {
          currentSection = sectionMatch[1].trim()
          continue
        }
        if (line.toLowerCase().includes(lowerQuery)) {
          const hit: typeof hits[number] = {
            date: dateKey,
            section: currentSection,
            line: line.trim(),
            lineNumber: i + 1,
          }
          if (contextLines > 0) {
            const beforeStart = Math.max(0, i - contextLines)
            const afterEnd = Math.min(lines.length, i + 1 + contextLines)
            if (beforeStart < i) {
              hit.contextBefore = lines.slice(beforeStart, i).map((line) => line.trimEnd())
            }
            if (afterEnd > i + 1) {
              hit.contextAfter = lines.slice(i + 1, afterEnd).map((line) => line.trimEnd())
            }
          }
          hits.push(hit)
        }
      }
    }
    return hits
  }

  /** Enumerate the dates that actually have a daily note on disk, in
   *  reverse chronological order. Returns up to `limit` entries (default
   *  14, hard cap 90). Used by memory.daily.list to surface the user's
   *  recent journal without reading every file. */
  async listDailyNoteDates(limit: number = 14): Promise<string[]> {
    const safeLimit = Math.max(1, Math.min(Math.floor(limit), 90))
    return (await journalInventory(this.memoryDir)).map((entry) => entry.date).slice(0, safeLimit)
  }

  async readDailySection(
    sectionTitle: string,
    date = new Date(),
  ): Promise<string | undefined> {
    const content = await this.readDailyNote(date)
    if (!content) {
      return undefined
    }

    const section = findMarkdownSection(content, sectionTitle)
    if (!section) {
      return undefined
    }

    const body = content.slice(section.bodyStart, section.bodyEnd).trim()
    return body.length > 0 ? body : undefined
  }

  async appendToDailySection(
    sectionTitle: string,
    entry: string,
    date = new Date(),
  ): Promise<void> {
    const normalizedEntry = entry.trim()
    if (!normalizedEntry) {
      return
    }

    await this.updateDailySection(
      sectionTitle,
      (existing) => existing ? `${existing}\n\n${normalizedEntry}` : normalizedEntry,
      date,
    )
  }

  async replaceDailySection(
    sectionTitle: string,
    content: string,
    date = new Date(),
  ): Promise<void> {
    const normalizedContent = content.trim()
    if (!normalizedContent) {
      return
    }

    await this.updateDailySection(
      sectionTitle,
      () => normalizedContent,
      date,
    )
  }

  async replaceMemorySection(
    sectionTitle: string,
    content: string,
  ): Promise<void> {
    const normalizedContent = content.trimEnd()
    if (!normalizedContent.trim()) {
      return
    }

    await this.init()

    await this.updateNamedSection(
      this.getMemoryPath(),
      sectionTitle,
      () => normalizedContent,
      '# Long-Term Memory\n\n',
    )
  }

  /**
   * Remove a single bullet item from a memory section. The item is matched by
   * substring (case-insensitive) against the normalized bullet text — the
   * caller does not need to know the exact whitespace/punctuation. If the
   * section becomes empty afterwards, the entire section is removed.
   *
   * Returns `removed: false` when the section is missing or no item matched
   * (to support agent retries with different phrasing).
   */
  async removeMemorySectionItem(
    sectionTitle: string,
    itemQuery: string,
  ): Promise<{ removed: boolean; sectionDeleted: boolean; remainingItems: number; ambiguous?: boolean }> {
    const query = normalizeBulletItem(itemQuery)
    if (!query) {
      return { removed: false, sectionDeleted: false, remainingItems: 0 }
    }

    const currentItems = parseBulletItems(await this.readMemorySection(sectionTitle))
    if (currentItems.length === 0) {
      return { removed: false, sectionDeleted: false, remainingItems: 0 }
    }

    let matchIndex = currentItems.findIndex(
      (item) => normalizeBulletItem(item) === query,
    )
    if (matchIndex < 0) {
      const matches = currentItems.flatMap((item, index) => normalizeBulletItem(item).includes(query) ? [index] : [])
      if (matches.length > 1) return { removed: false, ambiguous: true, sectionDeleted: false, remainingItems: currentItems.length }
      matchIndex = matches[0] ?? -1
    }
    if (matchIndex < 0) {
      return { removed: false, sectionDeleted: false, remainingItems: currentItems.length }
    }

    const next = [...currentItems.slice(0, matchIndex), ...currentItems.slice(matchIndex + 1)]
    if (next.length === 0) {
      const sectionDeleted = await this.deleteMemorySection(sectionTitle)
      return { removed: true, sectionDeleted, remainingItems: 0 }
    }

    await this.replaceMemorySection(
      sectionTitle,
      next.map((item) => `- ${item}`).join('\n'),
    )
    return { removed: true, sectionDeleted: false, remainingItems: next.length }
  }

  async deleteMemorySection(sectionTitle: string): Promise<boolean> {
    await this.init()
    const currentContent =
      (await this.readTextFile(this.getMemoryPath()))
      ?? '# Long-Term Memory\n\n'
    const section = findMarkdownSection(currentContent, sectionTitle)
    if (!section) {
      return false
    }

    const before = currentContent.slice(0, section.start).trimEnd()
    const after = currentContent.slice(section.end).trimStart()
    let nextContent = before
    if (after.length > 0) {
      nextContent += nextContent.length > 0 ? `\n\n${after}` : after
    }
    nextContent = nextContent.trimEnd()
    const persisted = `${nextContent || '# Long-Term Memory'}\n`
    await this.writeLongTerm(currentContent, persisted)
    return true
  }

  private async recoverLongTermChange(): Promise<void> {
    if (!this.hasChangeListener) return
    const pending = join(this.memoryDir, 'MEMORY.pending.json')
    let before: string
    try { before = JSON.parse(await readFile(pending, 'utf8')).before as string }
    catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return; throw error }
    const after = await this.readTextFile(this.getMemoryPath()) ?? before
    await this.options.onLongTermChange(before, after)
    await rm(pending, { force: true })
  }

  private async writeLongTerm(before: string, after: string, notify = true): Promise<void> {
    const pending = join(this.memoryDir, 'MEMORY.pending.json')
    if (this.hasChangeListener && notify) {
      const temp = `${pending}.${randomUUID()}.tmp`
      await writeFile(temp, JSON.stringify({ before }), { mode: 0o600 })
      await rename(temp, pending)
    }
    const temporary = `${this.getMemoryPath()}.${randomUUID()}.tmp`
    await writeFile(temporary, after, { mode: 0o600 })
    await rename(temporary, this.getMemoryPath())
    if (this.hasChangeListener && notify) await this.recoverLongTermChange()
  }

  private dailyDir(): string {
    return join(this.memoryDir, 'daily')
  }

  private async ensureFile(path: string, initialContent: string): Promise<void> {
    try {
      await access(path, constants.F_OK)
    } catch {
      await writeFile(path, initialContent, 'utf-8')
    }
  }

  private async readPromptFile(
    path: string,
    maxChars: number,
    prioritySectionTitles: string[] = [],
    options: { stripPrivate?: boolean } = {},
  ): Promise<string | undefined> {
    try {
      let raw = await readFile(path, 'utf-8')
      if (options.stripPrivate) {
        // The strip pass must see the `<!-- private -->` markers, so
        // it has to run before sanitizePromptContent strips ALL HTML
        // comments.
        raw = stripPrivateSections(raw)
      }
      const content = sanitizePromptContent(raw).trim()
      if (!hasSubstantiveContent(content)) {
        return undefined
      }
      return trimMarkdownSections(content, maxChars, prioritySectionTitles)
    } catch {
      return undefined
    }
  }

  private async readTextFile(path: string): Promise<string | undefined> {
    try {
      return await readFile(path, 'utf-8')
    } catch {
      return undefined
    }
  }

  private async updateDailySection(
    sectionTitle: string,
    update: (existing: string) => string,
    date: Date,
  ): Promise<void> {
    await withJournalLock(this.memoryDir, async () => {
      await restoreJournal(this.memoryDir, formatDateKey(date))
      await this.init(date)
      await this.updateNamedSection(
        this.getDailyNotePath(date),
        sectionTitle,
        update,
        `# ${formatDateKey(date)}\n\n`,
      )
    })
  }

  private async updateNamedSection(
    path: string,
    sectionTitle: string,
    update: (existing: string) => string,
    initialContent: string,
  ): Promise<void> {
    const currentContent =
      (await this.readTextFile(path))
      ?? initialContent
    const section = findMarkdownSection(currentContent, sectionTitle)
    const existingBody = section
      ? normalizeMarkdownSectionBody(currentContent.slice(section.bodyStart, section.bodyEnd))
      : ''
    const nextBody = update(existingBody).trimEnd()

    if (!nextBody.trim()) {
      return
    }

    const nextSection = `## ${sectionTitle}\n\n${nextBody}\n`
    let nextContent: string

    if (section) {
      const before = currentContent.slice(0, section.start).trimEnd()
      const after = currentContent.slice(section.end).trimStart()
      nextContent = before
      if (nextContent.length > 0) {
        nextContent += '\n\n'
      }
      nextContent += nextSection.trimEnd()
      if (after.length > 0) {
        nextContent += `\n\n${after}`
      }
      nextContent += '\n'
    } else {
      nextContent = `${currentContent.trimEnd()}\n\n${nextSection.trimEnd()}\n`
    }

    if (path === this.getMemoryPath()) await this.writeLongTerm(currentContent, nextContent)
    else await writeFile(path, nextContent, 'utf-8')
  }
}

function formatDateKey(date: Date): string {
  const year = String(date.getFullYear())
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function hasSubstantiveContent(content: string): boolean {
  const stripped = content
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/^#{1,6}\s+.*$/gm, '')
    .trim()

  return stripped.length > 0
}

function sanitizePromptContent(content: string): string {
  return content
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/\n{3,}/g, '\n\n')
}

/**
 * Drop sections whose body starts with `<!-- private -->` (or its
 * `private:high` variant) from the system-prompt inject path. The
 * section header is also removed so the model is not nudged into
 * asking about it. Specific entries inside a private section are
 * still reachable via `memory.search` / `memory.read` / agent tools
 * — this only changes the default inject, not retrieval.
 *
 * Round 6 long-term-memory-exposure plan, mitigation candidate 3
 * ("Sensitivity tagging").
 */
const PRIVATE_MARKER = /<!--\s*private(?::[a-zA-Z0-9_-]+)?\s*-->/i

function stripPrivateSections(content: string): string {
  if (!PRIVATE_MARKER.test(content)) return content
  const sections = content.includes('\n#')
    ? content.split(/\n(?=#{1,6}\s)/)
    : [content]
  const kept: string[] = []
  for (const section of sections) {
    if (PRIVATE_MARKER.test(section)) continue
    kept.push(section)
  }
  return kept.join('\n')
}

function trimMarkdownSections(
  content: string,
  maxChars: number,
  prioritySectionTitles: string[] = [],
): string {
  if (content.length <= maxChars) {
    return content
  }

  const parsedSections = content.includes('\n#')
    ? content.split(/\n(?=#{1,6}\s)/)
    : [content]
  const sections = orderPromptSections(parsedSections, prioritySectionTitles)

  let total = 0
  const kept: string[] = []

  for (const section of sections) {
    const nextLength = total + section.length + (kept.length > 0 ? 1 : 0)
    if (nextLength <= maxChars) {
      kept.push(section)
      total = nextLength
      continue
    }

    if (kept.length === 0) {
      return truncateText(section, maxChars)
    }

    kept.push('...[truncated]')
    return kept.join('\n')
  }

  return kept.join('\n')
}

function orderPromptSections(
  sections: string[],
  prioritySectionTitles: string[],
): string[] {
  if (prioritySectionTitles.length === 0 || sections.length < 3) {
    return sections
  }

  const titlePriority = new Map(
    prioritySectionTitles.map((title, index) => [title.toLowerCase(), index]),
  )
  const header: string[] = []
  const prioritized: Array<{ section: string; priority: number; index: number }> = []
  const remaining: Array<{ section: string; index: number }> = []

  sections.forEach((section, index) => {
    const title = section.match(/^##\s+(.+?)\s*$/m)?.[1]?.trim().toLowerCase()
    if (!title) {
      header.push(section)
      return
    }
    const priority = titlePriority.get(title)
    if (priority == null) {
      remaining.push({ section, index })
      return
    }
    prioritized.push({ section, priority, index })
  })

  prioritized.sort((left, right) => (
    left.priority - right.priority
    || left.index - right.index
  ))

  return [
    ...header,
    ...prioritized.map((entry) => entry.section),
    ...remaining.map((entry) => entry.section),
  ]
}

function truncateText(content: string, maxChars: number): string {
  const suffix = '\n...[truncated]'
  if (maxChars <= suffix.length) {
    return suffix.slice(0, maxChars)
  }
  return `${content.slice(0, maxChars - suffix.length).trimEnd()}${suffix}`
}

interface MarkdownSectionRange {
  start: number
  end: number
  bodyStart: number
  bodyEnd: number
}

function findMarkdownSection(
  content: string,
  sectionTitle: string,
): MarkdownSectionRange | null {
  const headerPattern = new RegExp(`^## ${escapeRegExp(sectionTitle)}\\s*$`, 'm')
  const headerMatch = headerPattern.exec(content)
  if (!headerMatch || headerMatch.index === undefined) {
    return null
  }

  const start = headerMatch.index
  const headerEnd = start + headerMatch[0].length
  const nextSectionPattern = /^##\s/mg
  nextSectionPattern.lastIndex = headerEnd
  const nextSectionMatch = nextSectionPattern.exec(content)
  const end = nextSectionMatch?.index ?? content.length

  let bodyStart = headerEnd
  while (content[bodyStart] === '\n') {
    bodyStart++
  }

  let bodyEnd = end
  while (bodyEnd > bodyStart && content[bodyEnd - 1] === '\n') {
    bodyEnd--
  }

  return { start, end, bodyStart, bodyEnd }
}

function parseMarkdownSections(content: string): FileMemorySection[] {
  const sections: FileMemorySection[] = []
  const headerPattern = /^##\s+(.+?)\s*$/gm
  let match: RegExpExecArray | null

  while ((match = headerPattern.exec(content)) !== null) {
    const title = match[1]?.trim()
    if (!title) {
      continue
    }
    const headerEnd = match.index + match[0].length
    const nextHeaderPattern = /^##\s/mg
    nextHeaderPattern.lastIndex = headerEnd
    const next = nextHeaderPattern.exec(content)
    const end = next?.index ?? content.length
    sections.push({
      title,
      content: normalizeMarkdownSectionBody(content.slice(headerEnd, end)),
    })
  }

  return sections
}

function normalizeMarkdownSectionBody(content: string): string {
  const withoutHeaderGap = content.startsWith('\r\n')
    ? content.slice(2)
    : content.startsWith('\n')
      ? content.slice(1)
      : content
  return withoutHeaderGap.trimEnd()
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function parseBulletItems(content: string | undefined): string[] {
  if (!content) {
    return []
  }

  return content
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.startsWith('- '))
    .map((line) => line.slice(2).trim())
    .filter((line) => line.length > 0)
}

function normalizeBulletItem(content: string): string {
  return content
    .trim()
    .replace(/^-+\s*/, '')
    .replace(/\s+/g, ' ')
    .toLowerCase()
}
