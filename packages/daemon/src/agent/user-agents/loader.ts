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
import type { GraphBuilder, GraphAgentInfo, GraphAgentRegistry } from '../graph/registry.js'
import type { Deps } from '../graph/nodes.js'
import type { AgentGraph } from '../graph/engine.js'

const log = createLogger('user-agent')

const FrontmatterSchema = z.object({
  name: z.string().min(1).optional(),
  description: z.string().min(1).optional(),
  base: z.string().min(1).optional(),
  model: z.string().min(1).optional(),
  temperature: z.number().min(0).max(2).optional(),
  topP: z.number().min(0).max(1).optional(),
  maxTokens: z.number().int().positive().optional(),
  maxIterations: z.number().int().positive().optional(),
})

export interface UserAgentSpec {
  id: string
  name: string
  description: string
  base: string
  model?: string
  temperature?: number
  topP?: number
  maxTokens?: number
  maxIterations?: number
  systemPrompt: string
}

export interface UserAgentRecord {
  id: string
  filePath: string
  spec: UserAgentSpec
}

export interface BaseGraphBuilders {
  get(base: string): GraphBuilder | undefined
  list(): string[]
}

export interface UserAgentLoaderDeps {
  agentsDir: string
  baseBuilders: BaseGraphBuilders
  defaultBase?: string
}

export function parseUserAgentMarkdown(
  fileName: string,
  contents: string,
  defaultBase = 'enhanced',
): UserAgentSpec {
  const { frontmatter: fm, body } = parseMarkdownFrontmatter(fileName, contents, FrontmatterSchema)
  const id = resolveMarkdownId(fileName, fm.name, 'agent')
  const systemPrompt = requireMarkdownBody(fileName, body, 'agent system prompt')
  return {
    id,
    name: fm.name?.trim() || id,
    description: fm.description ?? `User-defined agent ${id}`,
    base: fm.base ?? defaultBase,
    model: fm.model,
    temperature: fm.temperature,
    topP: fm.topP,
    maxTokens: fm.maxTokens,
    maxIterations: fm.maxIterations,
    systemPrompt,
  }
}

export class UserAgentLoader {
  private records = new Map<string, UserAgentRecord>()

  constructor(private readonly deps: UserAgentLoaderDeps) {}

  async ensureDir(): Promise<void> {
    await ensureUserMarkdownDir(this.deps.agentsDir)
  }

  async loadAll(): Promise<UserAgentRecord[]> {
    return loadUserMarkdownRecords({
      dir: this.deps.agentsDir,
      log,
      duplicateLabel: 'agent',
      records: this.records,
      parse: (entry, filePath, contents) => {
        const spec = parseUserAgentMarkdown(entry, contents, this.deps.defaultBase)
        return { id: spec.id, filePath, spec }
      },
      getId: (record) => record.id,
    })
  }

  list(): UserAgentRecord[] {
    return [...this.records.values()]
  }

  get(id: string): UserAgentRecord | undefined {
    return this.records.get(id)
  }

  async writeAgent(spec: UserAgentSpec): Promise<UserAgentRecord> {
    await this.ensureDir()
    const filePath = join(this.deps.agentsDir, `${spec.id}.md`)
    const frontmatter = serializeFrontmatter(spec)
    const contents = `---\n${frontmatter}---\n\n${spec.systemPrompt.trim()}\n`
    await writeFile(filePath, contents, 'utf-8')
    const record: UserAgentRecord = { id: spec.id, filePath, spec }
    this.records.set(spec.id, record)
    return record
  }

  async deleteAgent(id: string): Promise<boolean> {
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

function serializeFrontmatter(spec: UserAgentSpec): string {
  const lines: string[] = []
  lines.push(`name: ${JSON.stringify(spec.name)}`)
  lines.push(`description: ${JSON.stringify(spec.description)}`)
  if (spec.base) lines.push(`base: ${JSON.stringify(spec.base)}`)
  if (spec.model) lines.push(`model: ${JSON.stringify(spec.model)}`)
  if (spec.temperature !== undefined) lines.push(`temperature: ${spec.temperature}`)
  if (spec.topP !== undefined) lines.push(`topP: ${spec.topP}`)
  if (spec.maxTokens !== undefined) lines.push(`maxTokens: ${spec.maxTokens}`)
  if (spec.maxIterations !== undefined) lines.push(`maxIterations: ${spec.maxIterations}`)
  return lines.join('\n') + '\n'
}

export function buildUserAgentInfo(
  record: UserAgentRecord,
  baseBuilders: BaseGraphBuilders,
  defaultBase = 'enhanced',
): GraphAgentInfo | null {
  const baseId = record.spec.base || defaultBase
  const baseBuilder = baseBuilders.get(baseId) ?? baseBuilders.get(defaultBase)
  if (!baseBuilder) return null

  const wrappedBuilder: GraphBuilder = (deps: Deps): AgentGraph => {
    const composedSystemPrompt = [
      deps.systemPrompt ?? '',
      `\n[User agent: ${record.spec.name}]\n${record.spec.systemPrompt}`,
    ].join('\n')
    return baseBuilder({ ...deps, systemPrompt: composedSystemPrompt })
  }

  return {
    id: record.spec.id,
    name: record.spec.name,
    description: record.spec.description,
    builder: wrappedBuilder,
    source: 'user',
    limits: record.spec.maxIterations
      ? { maxIterations: record.spec.maxIterations }
      : undefined,
  }
}

export function registerUserAgents(
  registry: GraphAgentRegistry,
  records: UserAgentRecord[],
  baseBuilders: BaseGraphBuilders,
  defaultBase = 'enhanced',
): { registered: string[]; skipped: string[] } {
  const registered: string[] = []
  const skipped: string[] = []
  for (const record of records) {
    const info = buildUserAgentInfo(record, baseBuilders, defaultBase)
    if (!info) {
      skipped.push(record.id)
      continue
    }
    registry.register(info)
    registered.push(info.id)
  }
  return { registered, skipped }
}
