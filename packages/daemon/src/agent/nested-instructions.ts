import { open, realpath, stat } from 'node:fs/promises'
import { dirname, join, relative, resolve, sep } from 'node:path'

/**
 * Lazily surface directory-scoped instruction files (AGENTS.md / CLAUDE.md)
 * that live *below* the session cwd.
 *
 * The system prompt already carries the instruction files on the cwd → home
 * chain. Instruction files in subdirectories are only relevant once the agent
 * works there, and models trained on nested-instruction conventions otherwise
 * spend tool calls probing `packages/x/CLAUDE.md` to find out whether one
 * exists. Injecting the file with the first tool result that touches its
 * directory makes that probe unnecessary and keeps the injection scoped to
 * the directories actually visited. Each directory is checked at most once
 * per session.
 */

export const NESTED_INSTRUCTION_FILENAMES = ['AGENTS.md', 'CLAUDE.md'] as const

const DEFAULT_PER_FILE_LIMIT = 4000
const DEFAULT_MAX_SESSIONS = 256
const MAX_DIRECTORIES_PER_CALL = 8
const TRUNCATED_SUFFIX = '\n… [truncated]'

/** Tools whose arguments name a filesystem location worth scoping to. */
const FILESYSTEM_TOOL_PREFIX = 'fs.'

export interface NestedInstructionFile {
  path: string
  directory: string
  content: string
}

export interface NestedInstructionLookup {
  sessionId: string
  /** Absolute or cwd-relative path the tool call targeted. */
  targetPath: string
  cwd?: string
  workspaceRoot?: string
}

function isPathInside(candidate: string, parent: string): boolean {
  const rel = relative(parent, candidate)
  return rel === '' || (!rel.startsWith('..') && !rel.startsWith(sep) && !/^[A-Za-z]:/u.test(rel))
}

function normalizeContent(raw: string): string {
  const noBom = raw.startsWith('\uFEFF') ? raw.slice(1) : raw
  return noBom.replace(/\r\n/g, '\n').replace(/\s+$/u, '')
}

function truncate(content: string, limit: number): string {
  if (content.length <= limit) return content
  if (limit <= TRUNCATED_SUFFIX.length) return content.slice(0, Math.max(0, limit))
  return content.slice(0, limit - TRUNCATED_SUFFIX.length) + TRUNCATED_SUFFIX
}

async function readBounded(
  path: string,
  limit: number,
  boundary: string,
): Promise<string | null> {
  try {
    const resolved = await realpath(path)
    if (!isPathInside(resolved, boundary)) return null
    if (!(await stat(resolved)).isFile()) return null
    const handle = await open(resolved, 'r')
    try {
      const buffer = Buffer.allocUnsafe(Math.max(4, (limit + 1) * 4))
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0)
      return truncate(normalizeContent(buffer.subarray(0, bytesRead).toString('utf8')), limit)
    } finally {
      await handle.close()
    }
  } catch {
    return null
  }
}

/**
 * Resolve the filesystem location a tool call is about, or null when the
 * tool is not filesystem-scoped. `path` wins over `cwd` so fs.read/fs.edit
 * scope to the file's directory while fs.list/fs.glob scope to their cwd.
 */
export function nestedInstructionTargetPath(
  toolName: string,
  args: Record<string, unknown> | undefined,
): string | null {
  if (!toolName.startsWith(FILESYSTEM_TOOL_PREFIX) || !args) return null
  for (const key of ['path', 'cwd'] as const) {
    const value = args[key]
    if (typeof value === 'string' && value.trim().length > 0) return value
  }
  return null
}

export class NestedInstructionInjector {
  private readonly checkedDirectories = new Map<string, Set<string>>()
  private readonly perFileLimit: number
  private readonly maxSessions: number

  constructor(options: { perFileCharLimit?: number; maxSessions?: number } = {}) {
    this.perFileLimit = options.perFileCharLimit ?? DEFAULT_PER_FILE_LIMIT
    this.maxSessions = options.maxSessions ?? DEFAULT_MAX_SESSIONS
  }

  forget(sessionId: string): void {
    this.checkedDirectories.delete(sessionId)
  }

  private sessionSet(sessionId: string): Set<string> {
    const existing = this.checkedDirectories.get(sessionId)
    if (existing) {
      // Refresh insertion order so long-lived sessions are not evicted first.
      this.checkedDirectories.delete(sessionId)
      this.checkedDirectories.set(sessionId, existing)
      return existing
    }
    while (this.checkedDirectories.size >= this.maxSessions) {
      const oldest = this.checkedDirectories.keys().next().value
      if (oldest === undefined) break
      this.checkedDirectories.delete(oldest)
    }
    const created = new Set<string>()
    this.checkedDirectories.set(sessionId, created)
    return created
  }

  /**
   * Instruction files for every not-yet-checked directory strictly below cwd
   * on the way to `targetPath`, least specific first. Directories above or
   * equal to cwd are the system prompt's responsibility and are skipped.
   */
  async collect(lookup: NestedInstructionLookup): Promise<NestedInstructionFile[]> {
    if (!lookup.cwd) return []
    let cwd: string
    try {
      cwd = await realpath(resolve(lookup.cwd))
    } catch {
      return []
    }
    let boundary = cwd
    if (lookup.workspaceRoot) {
      try {
        boundary = await realpath(resolve(lookup.workspaceRoot))
      } catch {
        return []
      }
    }
    const target = resolve(cwd, lookup.targetPath)
    if (!isPathInside(target, cwd) || target === cwd) return []

    let targetDir = dirname(target)
    try {
      if ((await stat(target)).isDirectory()) targetDir = target
    } catch {
      // A missing target still scopes to the directories that do exist.
    }

    const directories: string[] = []
    for (let dir = targetDir; dir !== cwd && isPathInside(dir, cwd); dir = dirname(dir)) {
      directories.push(dir)
      if (directories.length >= MAX_DIRECTORIES_PER_CALL) break
    }
    directories.reverse()

    const checked = this.sessionSet(lookup.sessionId)
    const files: NestedInstructionFile[] = []
    for (const directory of directories) {
      if (checked.has(directory)) continue
      checked.add(directory)
      for (const filename of NESTED_INSTRUCTION_FILENAMES) {
        const path = join(directory, filename)
        const content = await readBounded(path, this.perFileLimit, boundary)
        if (content !== null && content.trim().length > 0) {
          files.push({ path, directory, content })
        }
      }
    }
    return files
  }
}

export function formatNestedInstructions(
  files: readonly NestedInstructionFile[],
  cwd: string,
): string | null {
  if (files.length === 0) return null
  const parts: string[] = []
  for (const file of files) {
    const dir = relative(cwd, file.directory).split(sep).join('/')
    const rel = relative(cwd, file.path).split(sep).join('/')
    parts.push(`[Directory instructions ${rel} — apply while working under ${dir}/]`)
    parts.push(file.content)
  }
  return parts.join('\n')
}

let defaultInjector: NestedInstructionInjector | null = null

export function getDefaultNestedInstructionInjector(): NestedInstructionInjector {
  defaultInjector ??= new NestedInstructionInjector()
  return defaultInjector
}
