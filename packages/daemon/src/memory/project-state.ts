import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { createLogger } from '../logger.js'

/**
 * Project-scoped state board (long-term layer of the P022 state board).
 *
 * A named-block markdown file per repo/project lives at
 * `<stateDir>/<projectHash>.md`, where `projectHash` is the privacy-preserving
 * hash from `memory/scope.ts#hashProjectPath` (never a raw cwd — that embeds
 * the OS username). It is the project-scoped analog of Claude Code's
 * auto-MEMORY.md and opencode/Letta shared memory blocks:
 *
 *   ## decisions
 *   - chose Fastify over Express for the daemon HTTP layer
 *   ## learnings
 *   - the auth store is redis-backed, not in-process
 *   ## open-questions
 *   - is the migration idempotent?
 *   ## conventions
 *   - no semicolons; single quotes; .js import suffixes
 *
 * Two write paths, both structural (no content heuristics):
 *   (a) autonomous write-back — dreaming/knowledge-extraction append durable
 *       decisions/learnings/open-questions when the active scope is a project;
 *   (b) explicit agent/user edits — `writeBlock` replaces a named block on an
 *       explicit "remember this for the project" instruction.
 *
 * Every write broadcasts, so web/cli/desktop sharing one daemon observe the
 * same block (opencode's "state = persist + broadcast").
 */

const log = createLogger('project-state')

/** Canonical named blocks. Free-form block names are allowed too, but these are
 *  the ones the autonomous write-back and board rendering understand. */
export const PROJECT_STATE_BLOCKS = ['decisions', 'learnings', 'open-questions', 'conventions'] as const
export type ProjectStateBlockName = (typeof PROJECT_STATE_BLOCKS)[number]

export interface ProjectStateBlock {
  name: string
  /** Block body, trimmed of surrounding blank lines. */
  body: string
}

export interface ProjectStateWriteEvent {
  projectHash: string
  path: string
  block: string
  /** 'replace' for an explicit block overwrite, 'append' for auto write-back. */
  op: 'replace' | 'append'
}

export type ProjectStateBroadcaster = (event: ProjectStateWriteEvent) => void

/** projectHash is used directly as a filename segment, so it must be a bare
 *  token — never a path. hashProjectPath emits lowercase hex, so anything with
 *  a separator or dot is a caller bug (or an attempt to escape stateDir). */
const SAFE_PROJECT_HASH = /^[a-z0-9_-]+$/

function assertSafeProjectHash(projectHash: string): void {
  if (!SAFE_PROJECT_HASH.test(projectHash)) {
    throw new Error(`Unsafe projectHash for project-state filename: ${JSON.stringify(projectHash)}`)
  }
}

const BLOCK_HEADING = /^##\s+(.+?)\s*$/

/** Parse a project-state markdown document into ordered named blocks. Content
 *  before the first `## ` heading is ignored (only headed blocks are state). */
export function parseProjectStateBlocks(markdown: string): ProjectStateBlock[] {
  const blocks: ProjectStateBlock[] = []
  let current: { name: string; lines: string[] } | undefined
  for (const line of markdown.split('\n')) {
    const heading = BLOCK_HEADING.exec(line)
    if (heading) {
      if (current) blocks.push({ name: current.name, body: joinBody(current.lines) })
      current = { name: heading[1].trim().toLowerCase(), lines: [] }
      continue
    }
    if (current) current.lines.push(line)
  }
  if (current) blocks.push({ name: current.name, body: joinBody(current.lines) })
  return blocks
}

function joinBody(lines: string[]): string {
  return lines.join('\n').replace(/^\n+/, '').replace(/\s+$/, '')
}

/** Serialize named blocks back to a stable markdown document. */
export function serializeProjectStateBlocks(blocks: ProjectStateBlock[]): string {
  const parts = blocks
    .filter((block) => block.name.trim().length > 0)
    .map((block) => {
      const body = block.body.trim()
      return body.length > 0 ? `## ${block.name}\n${body}\n` : `## ${block.name}\n`
    })
  return parts.length > 0 ? `${parts.join('\n')}` : ''
}

/** Split a block body into its bullet items (lines starting with `- `). */
function bodyItems(body: string): string[] {
  return body
    .split('\n')
    .map((line) => line.replace(/^[-*]\s+/, '').trim())
    .filter((line) => line.length > 0)
}

export class ProjectStateStore {
  constructor(
    private readonly stateDir: string,
    private readonly broadcast?: ProjectStateBroadcaster,
  ) {}

  filePath(projectHash: string): string {
    assertSafeProjectHash(projectHash)
    return join(this.stateDir, `${projectHash}.md`)
  }

  async readBlocks(projectHash: string): Promise<ProjectStateBlock[]> {
    const raw = await this.readRaw(projectHash)
    return raw ? parseProjectStateBlocks(raw) : []
  }

  async readBlock(projectHash: string, name: string): Promise<string | undefined> {
    const target = name.trim().toLowerCase()
    const found = (await this.readBlocks(projectHash)).find((block) => block.name === target)
    return found?.body || undefined
  }

  /**
   * Replace (upsert) a whole named block, preserving every other block and
   * their order. This is the explicit agent/user edit path. Broadcasts.
   */
  async writeBlock(projectHash: string, name: string, body: string): Promise<void> {
    const target = name.trim().toLowerCase()
    if (!target) throw new Error('project-state block name is required')
    const blocks = await this.readBlocks(projectHash)
    const existing = blocks.find((block) => block.name === target)
    if (existing) existing.body = body.trim()
    else blocks.push({ name: target, body: body.trim() })
    await this.persist(projectHash, blocks)
    this.emit(projectHash, target, 'replace')
  }

  /**
   * Append bullet items to a named block, deduping against items already
   * present (case-insensitive). This is the autonomous write-back path.
   * Returns the number of items actually added; broadcasts only when > 0.
   */
  async appendBlockItems(projectHash: string, name: string, items: string[]): Promise<number> {
    const target = name.trim().toLowerCase()
    if (!target) throw new Error('project-state block name is required')
    const clean = items.map((item) => item.replace(/\s+/g, ' ').trim()).filter((item) => item.length > 0)
    if (clean.length === 0) return 0

    const blocks = await this.readBlocks(projectHash)
    let block = blocks.find((entry) => entry.name === target)
    if (!block) {
      block = { name: target, body: '' }
      blocks.push(block)
    }
    const seen = new Set(bodyItems(block.body).map((item) => item.toLowerCase()))
    const added: string[] = []
    for (const item of clean) {
      const key = item.toLowerCase()
      if (seen.has(key)) continue
      seen.add(key)
      added.push(item)
    }
    if (added.length === 0) return 0
    const bullets = added.map((item) => `- ${item}`).join('\n')
    block.body = block.body.trim().length > 0 ? `${block.body.trim()}\n${bullets}` : bullets
    await this.persist(projectHash, blocks)
    this.emit(projectHash, target, 'append')
    return added.length
  }

  /** Render the project state for injection into the board / system prompt, or
   *  null when there is nothing recorded for the project yet. */
  async render(projectHash: string): Promise<string | null> {
    const blocks = (await this.readBlocks(projectHash)).filter((block) => block.body.trim().length > 0)
    if (blocks.length === 0) return null
    const lines = ['[Project state board]', `Durable project memory (repo ${projectHash}).`]
    for (const block of blocks) {
      lines.push(`## ${block.name}`, block.body.trim())
    }
    return lines.join('\n')
  }

  private async readRaw(projectHash: string): Promise<string | undefined> {
    try {
      return await readFile(this.filePath(projectHash), 'utf-8')
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
      throw error
    }
  }

  private async persist(projectHash: string, blocks: ProjectStateBlock[]): Promise<void> {
    await mkdir(this.stateDir, { recursive: true })
    await writeFile(this.filePath(projectHash), serializeProjectStateBlocks(blocks), 'utf-8')
  }

  private emit(projectHash: string, block: string, op: ProjectStateWriteEvent['op']): void {
    if (!this.broadcast) return
    try {
      this.broadcast({ projectHash, path: this.filePath(projectHash), block, op })
    } catch (error) {
      log.warn('project-state broadcast failed', { projectHash, block, error: String(error) })
    }
  }
}
