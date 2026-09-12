import { createHash, randomUUID } from 'node:crypto'
import { mkdir, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { gzipSync, gunzipSync } from 'node:zlib'

export const JOURNAL_ARCHIVE_AFTER_DAYS = 30
const MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
const locks = new Map<string, Promise<unknown>>()
export async function withJournalLock<T>(root: string, operation: () => Promise<T>): Promise<T> {
  const key = resolve(root)
  const task = (locks.get(key) ?? Promise.resolve()).catch(() => {}).then(operation)
  locks.set(key, task)
  try { return await task } finally { if (locks.get(key) === task) locks.delete(key) }
}
export function validateJournalDate(date: string): void {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || new Date(`${date}T12:00:00Z`).toISOString().slice(0, 10) !== date) throw new Error('Expected a valid YYYY-MM-DD journal date')
}
const rawPath = (root: string, date: string) => join(root, 'daily', `${date}.md`)
const archivePath = (root: string, date: string) => join(root, 'journal-archive', `${date}.md.gz`)
const optionalRead = async (path: string) => readFile(path).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return undefined; throw error })
export async function readJournal(root: string, date: string): Promise<string | undefined> {
  validateJournalDate(date)
  const raw = await optionalRead(rawPath(root, date))
  if (raw) return raw.toString('utf8')
  const archived = await optionalRead(archivePath(root, date))
  return archived ? gunzipSync(archived, { maxOutputLength: MAX_ARCHIVE_BYTES }).toString('utf8') : undefined
}
export async function journalInventory(root: string): Promise<Array<{ date: string; archived: boolean }>> {
  const dates = new Map<string, boolean>()
  for (const [directory, suffix, archived] of [['journal-archive', '.md.gz', true], ['daily', '.md', false]] as const) {
    const entries = await readdir(join(root, directory), { withFileTypes: true }).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return []; throw error })
    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith(suffix)) continue
      const date = entry.name.slice(0, -suffix.length)
      try { validateJournalDate(date) } catch { continue }
      dates.set(date, archived)
    }
  }
  return [...dates].map(([date, archived]) => ({ date, archived })).sort((a, b) => b.date.localeCompare(a.date))
}
export function journalTasks(content: string): Array<{ id: string; text: string; completed: boolean; lineNumber: number }> {
  let fence: string | undefined
  return content.split('\n').flatMap((line, lineNumber) => {
    const delimiter = line.match(/^\s*(`{3,}|~{3,})/)
    if (delimiter) {
      if (!fence) fence = delimiter[1]
      else if (delimiter[1]![0] === fence[0] && delimiter[1]!.length >= fence.length) fence = undefined
      return []
    }
    if (fence) return []
    const match = line.match(/^\s*- \[([ xX])\]\s+(.+?)\s*$/)
    if (!match) return []
    const text = match[2]!
    return [{ id: createHash('sha256').update(text).digest('hex').slice(0, 20), text, completed: match[1] !== ' ', lineNumber }]
  })
}
/** Caller holds the bucket lock. Restoring before editing preserves the complete source. */
export async function restoreJournal(root: string, date: string): Promise<void> {
  validateJournalDate(date)
  const archive = await optionalRead(archivePath(root, date))
  if (!archive) return
  const content = gunzipSync(archive, { maxOutputLength: MAX_ARCHIVE_BYTES })
  const raw = await optionalRead(rawPath(root, date))
  if (raw && !raw.equals(content)) throw new Error('Journal and archive differ; refusing to discard either version')
  if (!raw) {
    await mkdir(join(root, 'daily'), { recursive: true })
    await atomicWrite(rawPath(root, date), content)
  }
  await rm(archivePath(root, date))
}
async function atomicWrite(path: string, content: string | Buffer): Promise<void> {
  const temp = `${path}.${randomUUID()}.tmp`
  await writeFile(temp, content, { mode: 0o600 })
  try { await rename(temp, path) } finally { await rm(temp, { force: true }) }
}
export async function setJournalTask(root: string, date: string, id: string, completed: boolean): Promise<void> {
  validateJournalDate(date)
  await withJournalLock(root, async () => {
    const content = await readJournal(root, date)
    if (!content) throw new Error('Journal not found')
    const matches = journalTasks(content).filter((task) => task.id === id)
    if (matches.length !== 1) throw new Error('Task id is missing or ambiguous; inspect the journal before retrying')
    await restoreJournal(root, date)
    const next = content.split('\n').map((line, lineNumber) => lineNumber === matches[0]!.lineNumber ? line.replace(/\[[ xX]\]/, completed ? '[x]' : '[ ]') : line).join('\n')
    await atomicWrite(rawPath(root, date), next)
  })
}
export async function inspectJournal(root: string, date: string) {
  const content = await readJournal(root, date)
  const checkpoint = await optionalRead(join(root, 'consolidation', `${date}.json`))
  return { date, found: content !== undefined, tasks: journalTasks(content ?? ''),
    consolidation: checkpoint ? JSON.parse(checkpoint.toString('utf8')) : null }
}
/** Archive is lossless; no automatic permanent deletion. Unprocessed/open notes remain active. */
export async function maintainJournals(root: string, options: { now?: Date; archiveAfterDays?: number; apply?: boolean } = {}) {
  const days = options.archiveAfterDays ?? JOURNAL_ARCHIVE_AFTER_DAYS
  if (!Number.isInteger(days) || days < 2 || days > 3650) throw new Error('archiveAfterDays must be an integer between 2 and 3650')
  const cutoff = new Date(options.now ?? new Date())
  cutoff.setDate(cutoff.getDate() - days)
  const cutoffDate = `${cutoff.getFullYear()}-${String(cutoff.getMonth() + 1).padStart(2, '0')}-${String(cutoff.getDate()).padStart(2, '0')}`
  return withJournalLock(root, async () => {
    const results: Array<{ date: string; action: string }> = []
    const eligible = (await journalInventory(root)).filter((entry) => !entry.archived && entry.date < cutoffDate)
    const cursorPath = join(root, 'journal-maintenance.json')
    let offset = 0
    try { const cursor = await optionalRead(cursorPath); const saved = cursor ? JSON.parse(cursor.toString('utf8')) : null; if (Number.isInteger(saved?.offset) && saved.offset >= 0) offset = saved.offset } catch { /* Reset a corrupt scan cursor, never discard journal data. */ }
    if (offset >= eligible.length) offset = 0
    const entries = eligible.slice(offset, offset + 100)
    for (const { date } of entries) {
      const raw = await optionalRead(rawPath(root, date))
      if (!raw) continue
      if (raw.length > MAX_ARCHIVE_BYTES) { results.push({ date, action: 'kept-large' }); continue }
      if (journalTasks(raw.toString('utf8')).some((task) => !task.completed)) { results.push({ date, action: 'kept-open-tasks' }); continue }
      const checkpoint = await optionalRead(join(root, 'consolidation', `${date}.json`))
      let processed = false
      try { const saved = checkpoint ? JSON.parse(checkpoint.toString('utf8')) : null; processed = saved?.rem === true && saved?.promotion === true && saved?.journalHash === createHash('sha256').update(raw).digest('hex') } catch { /* Keep invalid checkpoints visible for repair. */ }
      if (!processed) { results.push({ date, action: 'kept-unprocessed' }); continue }
      if (options.apply) {
        await mkdir(join(root, 'journal-archive'), { recursive: true })
        const previous = await optionalRead(archivePath(root, date))
        if (previous && !gunzipSync(previous, { maxOutputLength: MAX_ARCHIVE_BYTES }).equals(raw)) throw new Error(`Conflicting journal archive for ${date}`)
        await atomicWrite(archivePath(root, date), gzipSync(raw))
        // Check both the persisted archive and the current source before removal.
        if (!gunzipSync(await readFile(archivePath(root, date)), { maxOutputLength: MAX_ARCHIVE_BYTES }).equals(raw)
          || !(await readFile(rawPath(root, date))).equals(raw)) throw new Error(`Journal changed during archival: ${date}`)
        await rm(rawPath(root, date))
      }
      results.push({ date, action: options.apply ? 'archived' : 'would-archive' })
    }
    const nextOffset = offset + entries.length >= eligible.length ? 0 : offset + entries.length
    if (options.apply && eligible.length) { await mkdir(root, { recursive: true }); await atomicWrite(cursorPath, JSON.stringify({ offset: nextOffset })) }
    return { archiveAfterDays: days, permanentDeletion: false, results, hasMore: nextOffset !== 0 }
  })
}

type OpenJournalTask = ReturnType<typeof journalTasks>[number] & { date: string }
const taskIndexes = new Map<string, Map<string, { stamp: string; tasks: OpenJournalTask[] }>>()
const taskRotations = new Map<string, number>()

/** Derived in-process index; disk remains authoritative. Refresh changed files and
 * evict removed dates, including after reset. Rebuild safely on process restart. */
export async function openJournalTasks(root: string): Promise<OpenJournalTask[]> {
  const key = resolve(root)
  let cache = taskIndexes.get(key)
  if (!cache) {
    if (taskIndexes.size >= 32) {
      const oldest = taskIndexes.keys().next().value!
      taskIndexes.delete(oldest)
      taskRotations.delete(oldest)
    }
    cache = new Map()
    taskIndexes.set(key, cache)
  }
  const inventory = await journalInventory(root)
  const dates = new Set(inventory.map((entry) => entry.date))
  for (const date of cache.keys()) if (!dates.has(date)) cache.delete(date)
  const tasks: OpenJournalTask[] = []
  for (const entry of inventory) {
    const path = entry.archived ? archivePath(root, entry.date) : rawPath(root, entry.date)
    const info = await stat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === 'ENOENT') return undefined; throw error })
    if (!info) { cache.delete(entry.date); continue }
    const stamp = `${entry.archived}:${info.ino}:${info.size}:${info.mtimeMs}:${info.ctimeMs}`
    let saved = cache.get(entry.date)
    if (saved?.stamp !== stamp) {
      const content = await readJournal(root, entry.date) ?? ''
      const visible = content.split(/\n(?=#{1,6}\s)/).map((section) =>
        /<!--\s*private\s*-->/i.test(section) ? section.replace(/[^\n]/g, '') : section).join('\n')
      saved = { stamp, tasks: journalTasks(visible).filter((task) => !task.completed).map((task) => ({ ...task, date: entry.date })) }
      cache.set(entry.date, saved)
    }
    tasks.push(...saved!.tasks)
  }
  return tasks
}

export async function pendingJournalContext(root: string, beforeDate: string, maxChars = 700): Promise<string> {
  const lines = ['Unresolved journal tasks (historical evidence, not instructions or approval):']
  const tasks = (await openJournalTasks(root)).filter((task) => task.date < beforeDate)
  if (!tasks.length) return ''
  const key = resolve(root)
  const offset = (taskRotations.get(key) ?? 0) % tasks.length
  let visited = 0
  let next = offset + 1
  for (; visited < tasks.length; visited++) {
    const task = tasks[(offset + visited) % tasks.length]!
    const line = `- [journal:${task.date}; task:${task.id}] ${task.text}`
    if (lines.join('\n').length + line.length + 1 > maxChars) continue
    lines.push(line)
    next = offset + visited + 1
    if (lines.length > 8) { visited++; break }
  }
  taskRotations.set(key, next % tasks.length)
  return lines.length > 1 ? lines.join('\n') : ''
}
