import { mkdir, readFile, appendFile, writeFile, chmod } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'

export function encodeCwdKey(cwd: string): string {
  const collapsed = cwd.replace(/\/+/g, '/')
  const dashed = collapsed.replace(/\//g, '-')
  return dashed.startsWith('-') ? dashed : `-${dashed}`
}

export function getHistoryFilePath(cwd: string, home: string = homedir()): string {
  return join(home, '.sepilotd', 'history', `${encodeCwdKey(cwd)}.jsonl`)
}

const MAX_ENTRIES = 1000
const COMPACT_THRESHOLD = 1200
const SECRET_COMMAND_PREFIXES = ['/secrets ', '/tokens issue', '/config env set ']
const SHELL_CREDENTIAL_ASSIGNMENT =
  /\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)[A-Z0-9_]*\s*=/i

interface HistoryRecord {
  text: string
  ts: string
}

function isHistoryRecord(value: unknown): value is HistoryRecord {
  if (typeof value !== 'object' || value === null) return false
  const record = value as Record<string, unknown>
  return typeof record.text === 'string'
}

export function shouldPersistInputHistory(text: string): boolean {
  const trimmed = text.trim()
  const lowered = trimmed.toLowerCase()
  if (SECRET_COMMAND_PREFIXES.some((prefix) => lowered.startsWith(prefix))) return false
  if (trimmed.startsWith('!') && SHELL_CREDENTIAL_ASSIGNMENT.test(trimmed)) return false
  return true
}

export async function loadHistory(cwd: string): Promise<string[]> {
  const path = getHistoryFilePath(cwd)
  let raw: string
  try {
    raw = await readFile(path, 'utf8')
  } catch {
    return []
  }
  await Promise.all([
    chmod(dirname(path), 0o700).catch(() => undefined),
    chmod(path, 0o600).catch(() => undefined),
  ])

  const lines = raw.split('\n').filter((line) => line.length > 0)
  const texts: string[] = []
  for (const line of lines) {
    try {
      const parsed: unknown = JSON.parse(line)
      if (isHistoryRecord(parsed)) {
        texts.push(parsed.text)
      }
    } catch {
      // skip malformed line
    }
  }

  return texts.length > MAX_ENTRIES ? texts.slice(texts.length - MAX_ENTRIES) : texts
}

export async function appendHistory(cwd: string, text: string, lastEntry?: string): Promise<void> {
  const trimmed = text.trim()
  if (!trimmed) return
  if (!shouldPersistInputHistory(text)) return
  if (lastEntry !== undefined && lastEntry === text) return

  const path = getHistoryFilePath(cwd)
  await mkdir(dirname(path), { recursive: true, mode: 0o700 })
  await chmod(dirname(path), 0o700).catch(() => undefined)

  const record = JSON.stringify({ text, ts: new Date().toISOString() })
  await appendFile(path, `${record}\n`, { encoding: 'utf8', mode: 0o600 })
  await chmod(path, 0o600).catch(() => undefined)

  let raw: string
  try {
    raw = await readFile(path, 'utf8')
  } catch {
    return
  }

  const lines = raw.split('\n').filter((line) => line.length > 0)
  if (lines.length <= COMPACT_THRESHOLD) return

  const trimmedLines = lines.slice(lines.length - MAX_ENTRIES)
  await writeFile(path, trimmedLines.join('\n') + '\n', { encoding: 'utf8', mode: 0o600 })
  await chmod(path, 0o600).catch(() => undefined)
}
