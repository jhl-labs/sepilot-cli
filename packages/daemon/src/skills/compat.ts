import { lstat, readdir, realpath, stat } from 'node:fs/promises'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

const HOME_RELATIVE_SKILL_ROOTS = [
  '.agents/skills',
  '.claude/skills',
  '.codex/skills',
  '.config/opencode/skill',
  '.config/opencode/skills',
  '.opencode/skill',
  '.opencode/skills',
  '.roo/skills',
]

const PROJECT_CONFIG_DIRS = ['.agents', '.claude', '.codex', '.opencode', '.roo']

function skillDirNames(configDir: string): string[] {
  return configDir === '.opencode' ? ['skill', 'skills'] : ['skills']
}

export async function defaultCompatibleSkillRoots(
  env: NodeJS.ProcessEnv = process.env,
): Promise<string[]> {
  const roots: string[] = []
  const home = env.HOME || env.USERPROFILE
  if (home) {
    for (const rel of HOME_RELATIVE_SKILL_ROOTS) {
      roots.push(join(home, rel))
    }
    if (env.CODEX_HOME) {
      roots.push(join(env.CODEX_HOME, 'skills'))
    }
    if (env.XDG_CONFIG_HOME) {
      roots.push(join(env.XDG_CONFIG_HOME, 'opencode', 'skill'))
      roots.push(join(env.XDG_CONFIG_HOME, 'opencode', 'skills'))
    }
    if (env.OPENCODE_CONFIG_DIR) {
      roots.push(join(env.OPENCODE_CONFIG_DIR, 'skill'))
      roots.push(join(env.OPENCODE_CONFIG_DIR, 'skills'))
    }
    roots.push(...await expandModeSpecificRoots(join(home, '.agents')))
    roots.push(...await expandModeSpecificRoots(join(home, '.roo')))
  }
  return uniquePaths(roots)
}

export async function projectCompatibleSkillRoots(
  cwd: string,
  boundaryRoot?: string,
): Promise<string[]> {
  const roots: string[] = []
  for (const dir of await ancestorDirs(cwd, boundaryRoot)) {
    for (const configDir of PROJECT_CONFIG_DIRS) {
      const root = join(dir, configDir)
      for (const name of skillDirNames(configDir)) roots.push(join(root, name))
      if (configDir === '.agents' || configDir === '.roo') {
        roots.push(...await expandModeSpecificRoots(root, boundaryRoot))
      }
    }
  }
  return uniquePaths(roots)
}

export async function gitCompatibleSkillRoots(repoRoot: string): Promise<string[]> {
  const roots = [
    join(repoRoot, 'skills'),
    join(repoRoot, '.agents', 'skills'),
    join(repoRoot, '.claude', 'skills'),
    join(repoRoot, '.codex', 'skills'),
    join(repoRoot, '.opencode', 'skill'),
    join(repoRoot, '.opencode', 'skills'),
    join(repoRoot, '.roo', 'skills'),
  ]
  roots.push(...await expandModeSpecificRoots(join(repoRoot, '.agents')))
  roots.push(...await expandModeSpecificRoots(join(repoRoot, '.roo')))
  return uniquePaths(roots)
}

async function ancestorDirs(start: string, boundaryRoot?: string): Promise<string[]> {
  const dirs: string[] = []
  let current = resolve(start)
  const boundary = boundaryRoot ? resolve(boundaryRoot) : undefined
  if (boundary && !isPathWithin(boundary, current)) return []
  while (true) {
    dirs.push(current)
    if (boundary && current === boundary) break
    if (await pathExists(join(current, '.git'))) break
    const parent = dirname(current)
    if (parent === current) break
    if (boundary && !isPathWithin(boundary, parent)) break
    current = parent
  }
  return dirs.reverse()
}

function isPathWithin(root: string, target: string): boolean {
  const rel = relative(root, target)
  return rel === '' || (
    rel !== '..'
    && !rel.startsWith(`..${sep}`)
    && !isAbsolute(rel)
  )
}

async function expandModeSpecificRoots(
  configRoot: string,
  boundaryRoot?: string,
): Promise<string[]> {
  let entries
  try {
    if (boundaryRoot) {
      const link = await lstat(configRoot)
      if (link.isSymbolicLink()) return []
      const [canonicalConfigRoot, canonicalBoundary] = await Promise.all([
        realpath(configRoot),
        realpath(boundaryRoot),
      ])
      if (!isPathWithin(canonicalBoundary, canonicalConfigRoot)) return []
    }
    entries = await readdir(configRoot, { withFileTypes: true })
  } catch {
    return []
  }
  const roots: string[] = []
  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    if (/^skills-[a-z0-9-]+$/i.test(entry.name)) {
      roots.push(join(configRoot, entry.name))
    }
  }
  return roots
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path)
    return true
  } catch {
    return false
  }
}

function uniquePaths(paths: string[]): string[] {
  return Array.from(new Set(paths.map((path) => resolve(path))))
}
