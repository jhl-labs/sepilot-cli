import { lstat, readdir, readFile, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join, basename, dirname, extname, resolve } from 'node:path'
import { strictWorkspacePathViolation } from '../../security/policy-engine.js'
import { parseFrontmatter } from './frontmatter.js'

export interface CustomDef {
  id: string
  data: Record<string, unknown>
  body: string
  source: string
}

export type CustomKind = 'agents' | 'commands'

const PROJECT_CONFIG_DIRS = ['.sepilotd', '.agent', '.agents', '.claude', '.codex', '.opencode', '.roo']
const GLOBAL_CONFIG_DIRS = ['.sepilotd', '.claude']
const MAX_PARENT_HOPS = 10

async function isDirectory(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isDirectory()
  } catch {
    return false
  }
}

async function projectConfigRoots(
  kind: CustomKind,
  cwd: string | null,
  home: string,
  boundaryRoot?: string,
): Promise<string[]> {
  if (cwd === null) return []
  const roots: string[] = []
  const boundary = boundaryRoot ? resolve(boundaryRoot) : undefined
  const resolvedHome = resolve(home)
  let current = resolve(cwd)
  let hops = 0

  while (hops <= MAX_PARENT_HOPS) {
    if (boundary && strictWorkspacePathViolation(current, boundary)) break
    for (const configDir of PROJECT_CONFIG_DIRS) {
      roots.push(join(current, configDir, kind))
    }
    if (boundary && current === boundary) break
    if (await isDirectory(join(current, '.git'))) break
    if (current === resolvedHome) break
    const parent = dirname(current)
    if (parent === current) break
    current = parent
    hops += 1
  }
  return roots
}

export async function discoverCustomDefs(
  kind: CustomKind,
  /**
   * `cwd: null` skips the project-level roots entirely (request without a
   * cwd on a shared daemon); `undefined` keeps the legacy process.cwd()
   * default for direct callers.
   */
  opts: { home?: string; cwd?: string | null; boundaryRoot?: string } = {},
): Promise<CustomDef[]> {
  const home = opts.home ?? homedir()
  const cwd = opts.cwd === null ? null : (opts.cwd ?? process.cwd())
  const projectRoots = await projectConfigRoots(kind, cwd, home, opts.boundaryRoot)
  // User-level custom definitions are agent control-plane configuration. They
  // remain available inside a strict workspace but do not widen tool access.
  const roots: Array<{ path: string; enforceBoundary: boolean }> = [
    ...projectRoots.map((path) => ({ path, enforceBoundary: true })),
    ...GLOBAL_CONFIG_DIRS.map((configDir) => ({
      path: join(home, configDir, kind),
      enforceBoundary: false,
    })),
  ]
  const byId = new Map<string, CustomDef>()
  for (const root of roots) {
    if (root.enforceBoundary && opts.boundaryRoot && strictWorkspacePathViolation(root.path, opts.boundaryRoot)) {
      continue
    }
    let entries
    try {
      if ((await lstat(root.path)).isSymbolicLink()) continue
      entries = await readdir(root.path, { withFileTypes: true })
      entries.sort((a, b) => a.name.localeCompare(b.name))
    } catch {
      continue
    }
    for (const entry of entries) {
      // Custom definitions are single regular Markdown files. Refusing
      // symlinks avoids silently importing prompts from outside the selected
      // workspace (or from mutable credential/config locations).
      if (!entry.isFile() || entry.isSymbolicLink() || extname(entry.name) !== '.md') continue
      const source = join(root.path, entry.name)
      if (root.enforceBoundary && opts.boundaryRoot && strictWorkspacePathViolation(source, opts.boundaryRoot)) {
        continue
      }
      const id = basename(entry.name, '.md')
      if (byId.has(id)) continue
      try {
        const raw = await readFile(source, 'utf8')
        const fm = parseFrontmatter(raw)
        byId.set(id, { id, data: fm.data, body: fm.body, source })
      } catch {
        // One unreadable or malformed definition must not break all commands
        // and agents for the turn.
      }
    }
  }
  return Array.from(byId.values())
}
