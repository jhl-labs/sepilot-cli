import { realpath } from 'node:fs/promises'
import { basename, dirname, isAbsolute, relative, resolve, sep } from 'node:path'
import { resolveToolPath } from './path-utils.js'

export type PagesTemplate = 'astro-mdx'

export const PAGES_ASTRO_MDX_TEMPLATE_PATHS = [
  'package.json',
  'astro.config.mjs',
  '.gitignore',
  'public/.nojekyll',
  'public/assets/.gitkeep',
  'src/layouts/Layout.astro',
  'src/pages/index.astro',
  'src/pages/posts/first-note.mdx',
  'src/pages/wiki/index.mdx',
  'src/pages/decks/demo.astro',
  'src/styles/global.css',
] as const

export type PagesAstroMdxTemplatePath = typeof PAGES_ASTRO_MDX_TEMPLATE_PATHS[number]

export interface PagesScaffoldPlannedTarget {
  absolutePath: string
  relativePath: string
  templatePath?: PagesAstroMdxTemplatePath
  workflow: boolean
}

export interface PagesScaffoldPathPlan {
  repoRoot: string
  siteRoot: string
  sitePath: string
  siteName: string
  template: PagesTemplate
  defaultBranch: string
  includeWorkflow: boolean
  overwrite: boolean
  targets: PagesScaffoldPlannedTarget[]
}

const DEFAULT_SITE_PATH = 'site'
const BLOCKED_SCAFFOLD_PARTS = new Set([
  '.git',
  '.sepilotd',
  'node_modules',
  'release-assets',
])

/**
 * Resolves every filesystem target that pages.scaffold can create. Both the
 * policy engine and the executor use this plan so a newly added template file
 * cannot silently bypass strict-workspace validation.
 */
export function resolvePagesScaffoldPathPlan(
  input: Record<string, unknown>,
  cwd?: string,
): PagesScaffoldPathPlan {
  const repoRoot = resolveToolPath(
    typeof input.repoPath === 'string' && input.repoPath.trim() ? input.repoPath : '.',
    cwd,
  )
  const sitePath = normalizeRelativePath(input.sitePath, DEFAULT_SITE_PATH)
  const siteRoot = resolve(repoRoot, fromPortablePath(sitePath))
  assertInside(repoRoot, siteRoot, 'sitePath must stay inside repoPath')
  assertAllowedScaffoldPath(sitePath)

  const template = normalizeTemplate(input.template)
  const siteName = normalizeSiteName(input.siteName, basename(repoRoot) || 'Pages Studio')
  const defaultBranch = normalizeBranch(input.defaultBranch)
  const includeWorkflow = input.includeWorkflow !== false
  const targets: PagesScaffoldPlannedTarget[] = PAGES_ASTRO_MDX_TEMPLATE_PATHS.map(
    (templatePath) => {
      const absolutePath = resolve(siteRoot, fromPortablePath(templatePath))
      assertInside(siteRoot, absolutePath, 'Template file escaped sitePath')
      return {
        absolutePath,
        relativePath: toPortablePath(relative(repoRoot, absolutePath)),
        templatePath,
        workflow: false,
      }
    },
  )

  if (includeWorkflow) {
    const absolutePath = resolve(repoRoot, '.github', 'workflows', 'pages-studio.yml')
    targets.push({
      absolutePath,
      relativePath: toPortablePath(relative(repoRoot, absolutePath)),
      workflow: true,
    })
  }

  return {
    repoRoot,
    siteRoot,
    sitePath,
    siteName,
    template,
    defaultBranch,
    includeWorkflow,
    overwrite: input.overwrite === true,
    targets,
  }
}

export function resolvePagesScaffoldPolicyTargets(
  input: Record<string, unknown>,
  cwd?: string,
): string[] {
  const plan = resolvePagesScaffoldPathPlan(input, cwd)
  return [plan.repoRoot, plan.siteRoot, ...plan.targets.map((target) => target.absolutePath)]
}

/**
 * Revalidates the immutable workspace root and a concrete output immediately
 * before filesystem mutation. Existing symlinks/junctions are resolved, while
 * missing suffixes are appended to the nearest canonical existing ancestor.
 */
export async function assertPagesScaffoldExecutionBoundary(
  workspaceRoot: string,
  targetPath: string,
): Promise<void> {
  const rawRoot = resolve(workspaceRoot)
  let canonicalRoot: string
  try {
    canonicalRoot = await realpath(rawRoot)
  } catch {
    throw new Error('Strict workspace root could not be safely validated')
  }

  if (relative(rawRoot, canonicalRoot) !== '') {
    throw new Error(
      'Strict workspace root changed after it was selected; reselect the workspace before continuing',
    )
  }

  let canonicalTarget: string
  try {
    canonicalTarget = await canonicalPotentialPath(targetPath)
  } catch {
    throw new Error('Strict workspace could not safely validate the pages.scaffold target')
  }

  if (!isInsideOrEqual(canonicalTarget, canonicalRoot)) {
    throw new Error(
      'Strict workspace blocks pages.scaffold: a planned output resolves outside the selected workspace',
    )
  }
}

async function canonicalPotentialPath(rawPath: string): Promise<string> {
  let current = resolve(rawPath)
  const missingParts: string[] = []

  while (true) {
    try {
      const canonicalAncestor = await realpath(current)
      return resolve(canonicalAncestor, ...missingParts.reverse())
    } catch (error) {
      if (!isMissingPathError(error)) throw error
      const parent = dirname(current)
      if (parent === current) throw error
      missingParts.push(basename(current))
      current = parent
    }
  }
}

function isMissingPathError(error: unknown): boolean {
  if (!error || typeof error !== 'object' || !('code' in error)) return false
  return error.code === 'ENOENT' || error.code === 'ENOTDIR'
}

function isInsideOrEqual(candidate: string, root: string): boolean {
  const rel = relative(root, candidate)
  return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel))
}

function normalizeTemplate(raw: unknown): PagesTemplate {
  if (raw == null || raw === '') return 'astro-mdx'
  if (raw === 'astro-mdx') return raw
  throw new Error('Unsupported Pages template. Supported templates: astro-mdx')
}

function normalizeSiteName(raw: unknown, fallback: string): string {
  if (typeof raw !== 'string') return fallback
  const trimmed = raw.trim()
  return trimmed || fallback
}

function normalizeBranch(raw: unknown): string {
  if (typeof raw !== 'string' || !raw.trim()) return 'main'
  const branch = raw.trim()
  if (!/^[A-Za-z0-9._/-]+$/u.test(branch) || branch.includes('..')) {
    throw new Error('defaultBranch contains unsupported characters')
  }
  return branch
}

function normalizeRelativePath(raw: unknown, fallback: string): string {
  const value = typeof raw === 'string' && raw.trim() ? raw.trim().replaceAll('\\', '/') : fallback
  if (value === '.') return '.'
  if (value.startsWith('/') || /^[A-Za-z]:\//u.test(value)) {
    throw new Error('sitePath must be relative to repoPath')
  }
  const parts = value.split('/').filter((part) => part && part !== '.')
  if (parts.some((part) => part === '..')) {
    throw new Error('sitePath must not contain ..')
  }
  return parts.length ? parts.join('/') : '.'
}

function fromPortablePath(path: string): string {
  if (path === '.') return '.'
  return path.split('/').join(sep)
}

function toPortablePath(path: string): string {
  return path.split(sep).join('/')
}

function assertInside(root: string, child: string, message: string): void {
  const rel = relative(root, child)
  if (rel === '' || (!rel.startsWith('..') && !isAbsolute(rel))) return
  throw new Error(message)
}

function assertAllowedScaffoldPath(sitePath: string): void {
  for (const part of sitePath.split('/')) {
    if (BLOCKED_SCAFFOLD_PARTS.has(part)) {
      throw new Error(`Refusing to scaffold Pages Studio inside ${part}`)
    }
  }
}
