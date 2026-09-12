import { writeFile, unlink } from 'node:fs/promises'
import { join } from 'node:path'
import { z } from 'zod'
import { createLogger } from '../../logger.js'
import {
  ensureUserMarkdownDir,
  loadUserMarkdownRecords,
  parseMarkdownFrontmatter,
  requireMarkdownBody,
  resolveMarkdownId,
} from '../user-markdown.js'

const log = createLogger('user-command')

const FrontmatterSchema = z.object({
  name: z.string().min(1).optional(),
  description: z.string().min(1).optional(),
  args: z.enum(['none', 'optional', 'required']).optional(),
  agent: z.string().min(1).optional(),
  model: z.string().min(1).optional(),
})

export interface UserCommandSpec {
  id: string
  name: string
  description: string
  args: 'none' | 'optional' | 'required'
  agent?: string
  model?: string
  body: string
}

export interface UserCommandRecord {
  id: string
  filePath: string
  spec: UserCommandSpec
}

export function parseUserCommandMarkdown(
  fileName: string,
  contents: string,
): UserCommandSpec {
  const { frontmatter: fm, body } = parseMarkdownFrontmatter(fileName, contents, FrontmatterSchema)
  const id = resolveMarkdownId(fileName, fm.name, 'command')
  const commandBody = requireMarkdownBody(fileName, body, 'command')
  return {
    id,
    name: fm.name?.trim() || id,
    description: fm.description ?? `Custom command ${id}`,
    args: fm.args ?? 'optional',
    agent: fm.agent,
    model: fm.model,
    body: commandBody,
  }
}

export interface UserCommandStoreDeps {
  commandsDir: string
}

export class UserCommandStore {
  private records = new Map<string, UserCommandRecord>()

  constructor(private readonly deps: UserCommandStoreDeps) {}

  async ensureDir(): Promise<void> {
    await ensureUserMarkdownDir(this.deps.commandsDir)
  }

  async loadAll(): Promise<UserCommandRecord[]> {
    return loadUserMarkdownRecords({
      dir: this.deps.commandsDir,
      log,
      duplicateLabel: 'command',
      records: this.records,
      parse: (entry, filePath, contents) => {
        const spec = parseUserCommandMarkdown(entry, contents)
        return { id: spec.id, filePath, spec }
      },
      getId: (record) => record.id,
    })
  }

  list(): UserCommandRecord[] {
    return [...this.records.values()]
  }

  get(id: string): UserCommandRecord | undefined {
    return this.records.get(id)
  }

  async writeCommand(spec: UserCommandSpec): Promise<UserCommandRecord> {
    await this.ensureDir()
    const filePath = join(this.deps.commandsDir, `${spec.id}.md`)
    const lines: string[] = []
    lines.push(`name: ${JSON.stringify(spec.name)}`)
    lines.push(`description: ${JSON.stringify(spec.description)}`)
    lines.push(`args: ${spec.args}`)
    if (spec.agent) lines.push(`agent: ${JSON.stringify(spec.agent)}`)
    if (spec.model) lines.push(`model: ${JSON.stringify(spec.model)}`)
    const contents = `---\n${lines.join('\n')}\n---\n\n${spec.body.trim()}\n`
    await writeFile(filePath, contents, 'utf-8')
    const record: UserCommandRecord = { id: spec.id, filePath, spec }
    this.records.set(spec.id, record)
    return record
  }

  async deleteCommand(id: string): Promise<boolean> {
    const record = this.records.get(id)
    if (!record) return false
    try {
      await unlink(record.filePath)
    } catch (err) {
      const code = (err as { code?: string }).code
      if (code !== 'ENOENT') throw err
    }
    this.records.delete(id)
    return true
  }
}

export interface ResolvedCommand {
  id: string
  prompt: string
  agent?: string
  model?: string
}

export function resolveCommand(
  spec: UserCommandSpec,
  rawArgs: string,
): ResolvedCommand {
  const trimmedArgs = rawArgs.trim()
  if (spec.args === 'none' && trimmedArgs.length > 0) {
    throw new Error(`Command "${spec.id}" does not accept arguments`)
  }
  if (spec.args === 'required' && trimmedArgs.length === 0) {
    throw new Error(`Command "${spec.id}" requires arguments`)
  }
  const prompt = spec.body
    .replace(/\$ARGS\b/g, trimmedArgs)
    .replace(/\$ARG_FIRST\b/g, trimmedArgs.split(/\s+/, 1)[0] ?? '')
  return {
    id: spec.id,
    prompt,
    agent: spec.agent,
    model: spec.model,
  }
}
