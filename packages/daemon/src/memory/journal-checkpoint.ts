import { createHash, randomUUID } from 'node:crypto'
import { readFile, readdir, writeFile, rename, mkdir } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import type { FileMemory } from './file-memory.js'

export interface JournalCheckpoint { fingerprint: string; rem?: boolean; promotion?: boolean; promotedMemoryIds?: string[]; processedAt?: string; journalHash?: string }
const pathFor = (file: FileMemory, date: string) => join(dirname(file.getMemoryPath()), 'consolidation', `${date}.json`)
export async function journalDates(file: FileMemory, now: Date): Promise<string[]> {
  const names = await readdir(join(dirname(file.getMemoryPath()), 'daily')).catch((error: NodeJS.ErrnoException) => {
    if (error.code === 'ENOENT') return []
    throw error
  })
  const today = dateKey(now)
  return names.filter((name) => /^\d{4}-\d{2}-\d{2}\.md$/.test(name))
    .map((name) => name.slice(0, -3)).filter((date) => date <= today).sort()
}
export const dateKey = (date: Date) => `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`
export const journalFingerprint = (light: string, reflection: string) => createHash('sha256').update(JSON.stringify(['v1', light, reflection])).digest('hex')
export async function readJournalCheckpoint(file: FileMemory, date: string, fingerprint: string): Promise<JournalCheckpoint> {
  try {
    const saved = JSON.parse(await readFile(pathFor(file, date), 'utf8')) as JournalCheckpoint
    if (saved.fingerprint === fingerprint) return saved
  } catch (error) {
    if (!(error instanceof SyntaxError) && (error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
  }
  return { fingerprint }
}
export async function writeJournalCheckpoint(file: FileMemory, date: string, checkpoint: JournalCheckpoint): Promise<void> {
  const note = await file.readDailyNote(new Date(`${date}T12:00:00`))
  checkpoint.journalHash = createHash('sha256').update(note ?? '').digest('hex')
  checkpoint.processedAt = checkpoint.rem && checkpoint.promotion ? new Date().toISOString() : undefined
  const path = pathFor(file, date)
  await mkdir(dirname(path), { recursive: true })
  const temporary = `${path}.${randomUUID()}.tmp`
  await writeFile(temporary, JSON.stringify(checkpoint), { mode: 0o600 })
  await rename(temporary, path)
}
