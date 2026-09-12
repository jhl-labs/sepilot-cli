import { open, readdir, realpath, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { parseFrontmatter } from '../agent/custom/frontmatter.js'

export interface AgentsMdFile {
  path: string
  content: string
}

export interface CollectAgentsMdOptions {
  home?: string
  cwd?: string | null
  /** Optional strict root that parent traversal must never cross. */
  boundaryRoot?: string | null
  perFileCharLimit?: number
  totalCharLimit?: number
  maxParentHops?: number
}

const DEFAULT_PER_FILE_LIMIT = 4000
const DEFAULT_TOTAL_LIMIT = 12000
const DEFAULT_MAX_PARENT_HOPS = 10
const TRUNCATED_SUFFIX = '\n… [truncated]'
// Both instruction filenames are read and merged (P022-T9). The system prompt
// already claims both AGENTS.md and CLAUDE.md are honored; this makes reality
// match. AGENTS.md is listed first so it wins ordering ties within a directory.
const AGENTS_FILENAMES = ['AGENTS.md', 'CLAUDE.md'] as const
const GLOBAL_SUBDIR = '.sepilotd'
const RULE_DIRS = [
  '.sepilotd/rules',
  '.agent/rules',
  '.agents/rules',
  '.claude/rules',
  '.codex/rules',
  '.opencode/rules',
  '.roo/rules',
  '.cursor/rules',
  'agent/rules',
] as const
const GLOBAL_RULE_DIRS = RULE_DIRS.filter((dir) => dir !== 'agent/rules')
const RULE_FILE_EXTENSIONS = new Set(['.md', '.mdc'])
const MAX_RULE_DEPTH = 8
const MAX_RULE_FILES_PER_ROOT = 100
const MAX_GLOBAL_RULE_FILES = 128
const MAX_WORKSPACE_RULE_FILES = 256
const MAX_RULE_SCAN_CHARS = 256_000

interface RuleScanBudget {
  remainingFiles: number
  remainingChars: number
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

function boundedReadBytes(charLimit: number): number {
  // UTF-8 uses at most four bytes per Unicode scalar. One extra character is
  // enough to detect truncation without reading a multi-megabyte rule file.
  return Math.max(4, (Math.max(0, charLimit) + 1) * 4)
}

async function readIfPresent(
  path: string,
  perFileLimit: number,
  boundaryRoot?: string | null,
): Promise<string | null> {
  try {
    const resolvedPath = await realpath(path)
    if (boundaryRoot && !isPathInside(resolvedPath, boundaryRoot)) return null
    const handle = await open(resolvedPath, 'r')
    try {
      const buffer = Buffer.allocUnsafe(boundedReadBytes(perFileLimit))
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0)
      const raw = buffer.subarray(0, bytesRead).toString('utf8')
      return truncate(normalizeContent(raw), perFileLimit)
    } finally {
      await handle.close()
    }
  } catch {
    return null
  }
}

function compactMetadataValue(value: unknown, limit = 240): string {
  const text = Array.isArray(value) ? value.join(', ') : String(value ?? '')
  const compact = text.trim().replace(/\s+/gu, ' ')
  return compact.length > limit ? `${compact.slice(0, limit - 1)}…` : compact
}

/** Preserve Cursor-compatible .mdc applicability instead of applying it universally. */
function formatScopedMdcRule(content: string, perFileLimit: number): string {
  const parsed = parseFrontmatter(content)
  if (parsed.body === content) return content
  const globs = compactMetadataValue(parsed.data.globs)
  const alwaysApply = parsed.data.alwaysApply === true
  const description = compactMetadataValue(parsed.data.description)
  const applicability = globs
    ? `Apply only while working on paths matching: ${globs}`
    : alwaysApply
      ? 'Apply to every relevant task.'
      : 'Apply only when its subject is relevant; this rule is not marked alwaysApply.'
  const lines = [
    `[Scoped rule metadata: ${applicability}]`,
    description ? `[Description: ${description}]` : '',
    parsed.body.trim(),
  ].filter(Boolean)
  return truncate(lines.join('\n'), perFileLimit)
}

async function resolveCwd(cwd: string | null | undefined): Promise<string | null> {
  if (!cwd) return null
  try {
    const real = await realpath(cwd)
    const info = await stat(real)
    return info.isDirectory() ? real : null
  } catch {
    return null
  }
}

function isPathInside(candidate: string, parent: string): boolean {
  const relativePath = relative(resolve(parent), resolve(candidate))
  return (
    relativePath === ''
    || (
      !!relativePath
      && relativePath !== '..'
      && !relativePath.startsWith(`..${sep}`)
      && !isAbsolute(relativePath)
    )
  )
}

export async function collectAgentsMd(
  options: CollectAgentsMdOptions = {},
): Promise<AgentsMdFile[]> {
  const home = options.home ?? homedir()
  const perFileLimit = options.perFileCharLimit ?? DEFAULT_PER_FILE_LIMIT
  const totalLimit = options.totalCharLimit ?? DEFAULT_TOTAL_LIMIT
  const maxHops = options.maxParentHops ?? DEFAULT_MAX_PARENT_HOPS
  const boundaryRequested = typeof options.boundaryRoot === 'string'
    && options.boundaryRoot.trim().length > 0
  const resolvedBoundary = boundaryRequested
    ? await resolveCwd(options.boundaryRoot)
    : null
  const stableBoundary = !boundaryRequested || (
    resolvedBoundary !== null
    && resolvedBoundary === resolve(options.boundaryRoot as string)
  )

  const collected: AgentsMdFile[] = []
  const globalPaths = new Set<string>()
  const scanCharBudget = Math.min(
    MAX_RULE_SCAN_CHARS,
    Math.max(DEFAULT_TOTAL_LIMIT, totalLimit * 4),
  )
  const globalRuleBudget: RuleScanBudget = {
    remainingFiles: MAX_GLOBAL_RULE_FILES,
    remainingChars: scanCharBudget,
  }
  const workspaceRuleBudget: RuleScanBudget = {
    remainingFiles: MAX_WORKSPACE_RULE_FILES,
    remainingChars: scanCharBudget,
  }

  const readRulesDir = async (
    dir: string,
    boundaryRoot?: string | null,
    budget: RuleScanBudget = workspaceRuleBudget,
  ): Promise<AgentsMdFile[]> => {
    const files: AgentsMdFile[] = []

    const visit = async (current: string, depth: number): Promise<void> => {
      if (
        depth > MAX_RULE_DEPTH
        || files.length >= MAX_RULE_FILES_PER_ROOT
        || budget.remainingFiles <= 0
        || budget.remainingChars <= 0
      ) return
      let entries
      try {
        const resolvedDir = await realpath(current)
        if (boundaryRoot && !isPathInside(resolvedDir, boundaryRoot)) return
        entries = await readdir(resolvedDir, { withFileTypes: true })
      } catch {
        return
      }

      entries.sort((a, b) => a.name.localeCompare(b.name))
      for (const entry of entries) {
        if (
          files.length >= MAX_RULE_FILES_PER_ROOT
          || budget.remainingFiles <= 0
          || budget.remainingChars <= 0
        ) return
        // Do not follow symlinked directories or files. Local rule roots are
        // untrusted project content and recursive symlinks can escape a strict
        // workspace or create loops. Global control-plane roots do not need
        // symlink compatibility either.
        if (entry.isSymbolicLink()) continue
        const candidate = join(current, entry.name)
        if (entry.isDirectory()) {
          await visit(candidate, depth + 1)
          continue
        }
        const extension = entry.name.slice(entry.name.lastIndexOf('.')).toLowerCase()
        if (!entry.isFile() || !RULE_FILE_EXTENSIONS.has(extension)) continue
        const rawContent = await readIfPresent(candidate, perFileLimit, boundaryRoot)
        if (rawContent !== null) {
          const content = extension === '.mdc'
            ? formatScopedMdcRule(rawContent, perFileLimit)
            : rawContent
          const boundedContent = truncate(content, budget.remainingChars)
          budget.remainingFiles -= 1
          budget.remainingChars = Math.max(0, budget.remainingChars - boundedContent.length)
          files.push({ path: candidate, content: boundedContent })
        }
      }
    }

    await visit(dir, 0)
    return files
  }

  // Read every instruction filename present in a directory, in filename order.
  const readDir = async (
    dir: string,
    boundaryRoot?: string | null,
  ): Promise<AgentsMdFile[]> => {
    const files: AgentsMdFile[] = []
    for (const filename of AGENTS_FILENAMES) {
      const candidate = join(dir, filename)
      const content = await readIfPresent(candidate, perFileLimit, boundaryRoot)
      if (content !== null) files.push({ path: candidate, content })
    }
    for (const relativeRuleDir of RULE_DIRS) {
      files.push(...await readRulesDir(join(dir, relativeRuleDir), boundaryRoot, workspaceRuleBudget))
    }
    return files
  }

  // 1. Global control-plane instructions. These are user-level agent config,
  // not workspace files, and therefore remain active inside a strict
  // workspace without widening any tool filesystem capability.
  const globalDir = join(home, GLOBAL_SUBDIR)
  for (const filename of AGENTS_FILENAMES) {
    const candidate = join(globalDir, filename)
    const content = await readIfPresent(candidate, perFileLimit)
    if (content !== null) {
      collected.push({ path: candidate, content })
      globalPaths.add(candidate)
    }
  }
  for (const relativeRuleDir of GLOBAL_RULE_DIRS) {
    const rules = await readRulesDir(join(home, relativeRuleDir), undefined, globalRuleBudget)
    for (const rule of rules) {
      collected.push(rule)
      globalPaths.add(rule.path)
    }
  }

  // 2. Walk-up from cwd to home, reading both filenames at each level.
  const resolvedCwd = await resolveCwd(options.cwd)
  const stableCwd = !boundaryRequested || (
    resolvedCwd !== null
    && resolvedCwd === resolve(options.cwd as string)
  )
  if (resolvedCwd && stableBoundary && stableCwd && (!boundaryRequested || resolvedBoundary)) {
    const resolvedHome = resolve(home)
    const walkUpLevels: AgentsMdFile[][] = []
    let current = resolvedCwd
    let hops = 0
    while (hops <= maxHops && (!resolvedBoundary || isPathInside(current, resolvedBoundary))) {
      const level = await readDir(current, resolvedBoundary)
      if (level.length > 0) walkUpLevels.push(level)
      if (resolvedBoundary && current === resolvedBoundary) break
      if (current === resolvedHome || !isPathInside(current, resolvedHome)) break
      const parent = dirname(current)
      if (parent === current) break
      if (resolvedBoundary && !isPathInside(parent, resolvedBoundary)) break
      current = parent
      hops += 1
    }
    // Collected deepest-first; emit least-specific → most-specific by reversing
    // the levels (filename order within each level is preserved).
    walkUpLevels.reverse()
    for (const level of walkUpLevels) collected.push(...level)
  }

  // 2b. Dedupe by normalized content — a CLAUDE.md that is a copy/symlink of an
  // AGENTS.md (common) must not be injected twice. First occurrence (the
  // least-specific / AGENTS.md-first) wins.
  const seenContent = new Set<string>()
  const deduped: AgentsMdFile[] = []
  for (const file of collected) {
    if (seenContent.has(file.content)) continue
    seenContent.add(file.content)
    deduped.push(file)
  }
  collected.length = 0
  collected.push(...deduped)

  // 3. Enforce total budget. Workspace rules may displace less-specific
  // workspace rules, but never evict user-level global control-plane files.
  let total = collected.reduce((sum, f) => sum + f.content.length, 0)
  while (total > totalLimit) {
    const workspaceIndexes = collected.flatMap((file, index) =>
      globalPaths.has(file.path) ? [] : [index])
    if (workspaceIndexes.length <= 1) break
    const workspaceIndex = workspaceIndexes[0]!
    const [dropped] = collected.splice(workspaceIndex, 1)
    total -= dropped?.content.length ?? 0
  }
  if (total > totalLimit) {
    const workspaceIndex = collected.findIndex((file) => !globalPaths.has(file.path))
    if (workspaceIndex >= 0) {
      const file = collected[workspaceIndex]!
      const globalTotal = collected.reduce(
        (sum, candidate) => sum + (globalPaths.has(candidate.path) ? candidate.content.length : 0),
        0,
      )
      const workspaceBudget = Math.max(0, totalLimit - globalTotal)
      if (workspaceBudget === 0) {
        collected.splice(workspaceIndex, 1)
      } else {
        collected[workspaceIndex] = {
          ...file,
          content: truncate(file.content, workspaceBudget),
        }
      }
      total = collected.reduce((sum, candidate) => sum + candidate.content.length, 0)
    }
  }
  if (total > totalLimit && collected.length > 0) {
    // Global-only overflow is shared fairly so every control-plane source
    // remains represented rather than silently deleting the earliest file.
    const perFileBudget = Math.max(1, Math.floor(totalLimit / collected.length))
    for (let index = 0; index < collected.length; index += 1) {
      const file = collected[index]!
      collected[index] = { ...file, content: truncate(file.content, perFileBudget) }
    }
  }

  return collected
}

export function formatAgentsMdSection(
  files: readonly AgentsMdFile[],
): string | null {
  if (files.length === 0) return null
  const parts = ['Agent instructions (global and workspace):']
  files.forEach((file, index) => {
    parts.push(`--- ${file.path} ---`)
    parts.push(file.content)
    if (index < files.length - 1) parts.push('')
  })
  return parts.join('\n')
}
