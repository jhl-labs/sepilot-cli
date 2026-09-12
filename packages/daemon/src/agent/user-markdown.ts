import { readdir, readFile, mkdir, stat } from 'node:fs/promises'
import { basename, join } from 'node:path'
import type { z } from 'zod'
import { parse as parseYaml } from 'yaml'

const FRONTMATTER_REGEX = /^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n?/

export function parseMarkdownFrontmatter<TFrontmatter>(
  fileName: string,
  contents: string,
  schema: z.ZodType<TFrontmatter>,
): { frontmatter: TFrontmatter; body: string } {
  const match = contents.match(FRONTMATTER_REGEX)
  let frontmatter: Record<string, unknown> = {}
  let body = contents
  if (match) {
    body = contents.slice(match[0].length)
    try {
      const parsed = parseYaml(match[1])
      if (parsed && typeof parsed === 'object') {
        frontmatter = parsed as Record<string, unknown>
      }
    } catch (err) {
      throw new Error(`${fileName}: invalid YAML frontmatter: ${(err as Error).message}`)
    }
  }

  const validated = schema.safeParse(frontmatter)
  if (!validated.success) {
    throw new Error(`${fileName}: invalid frontmatter: ${validated.error.message}`)
  }

  return { frontmatter: validated.data, body }
}

export function resolveMarkdownId(
  fileName: string,
  configuredName: string | undefined,
  label: string,
): string {
  const id = configuredName?.trim() || basename(fileName, '.md')
  if (!/^[a-z0-9][a-z0-9-_]*$/i.test(id)) {
    throw new Error(`${fileName}: ${label} id must match [a-z0-9][a-z0-9-_]*`)
  }
  return id
}

export function requireMarkdownBody(
  fileName: string,
  body: string,
  label: string,
): string {
  const trimmed = body.trim()
  if (trimmed.length === 0) {
    throw new Error(`${fileName}: ${label} body is empty`)
  }
  return trimmed
}

export async function ensureUserMarkdownDir(path: string): Promise<void> {
  try {
    await mkdir(path, { recursive: true })
  } catch (err) {
    const code = (err as { code?: string }).code
    if (code !== 'EEXIST') throw err
  }
}

export async function loadUserMarkdownRecords<TRecord>(options: {
  dir: string
  log: { warn: (message: string, data?: Record<string, unknown>) => void }
  duplicateLabel: string
  records: Map<string, TRecord>
  parse: (entry: string, filePath: string, contents: string) => TRecord
  getId: (record: TRecord) => string
}): Promise<TRecord[]> {
  await ensureUserMarkdownDir(options.dir)
  let entries: string[]
  try {
    entries = await readdir(options.dir)
  } catch (err) {
    const code = (err as { code?: string }).code
    if (code === 'ENOENT') return []
    throw err
  }

  options.records.clear()
  const records: TRecord[] = []
  for (const entry of entries) {
    if (!entry.endsWith('.md')) continue
    const filePath = join(options.dir, entry)
    const info = await stat(filePath).catch(() => null)
    if (!info?.isFile()) continue
    try {
      const contents = await readFile(filePath, 'utf-8')
      const record = options.parse(entry, filePath, contents)
      const id = options.getId(record)
      if (options.records.has(id)) {
        throw new Error(`${entry}: duplicate ${options.duplicateLabel} id "${id}"`)
      }
      options.records.set(id, record)
      records.push(record)
    } catch (err) {
      options.log.warn(`skip ${entry}`, {
        entry,
        error: err instanceof Error ? err.message : String(err),
      })
    }
  }
  return records
}
