import type { Dirent } from 'node:fs'
import { readdir, stat } from 'node:fs/promises'
import { basename, dirname, extname, join, relative, resolve, sep } from 'node:path'

/**
 * Turn a filesystem ENOENT into evidence the model can act on in its next
 * turn instead of guessing again.
 *
 * A bare "File not found" costs one more model round trip per guess. Listing
 * the nearest existing ancestor and pointing at same-named files elsewhere in
 * the workspace usually turns the second guess into the right call. All
 * enumeration is bounded (entries, directories, depth, wall clock) and stays
 * inside the caller-supplied root so a hint never widens what the tool could
 * have listed on its own.
 */

const MAX_LISTED_ENTRIES = 24
const MAX_CANDIDATES = 5
const MAX_SCAN_DIRS = 400
const MAX_SCAN_DEPTH = 4
const SCAN_TIME_BUDGET_MS = 120
const SKIP_DIR_NAMES = new Set([
  '.git',
  '.hg',
  '.svn',
  '.turbo',
  '.next',
  '.cache',
  'node_modules',
  'dist',
  'build',
  'out',
  'coverage',
  'target',
  '__pycache__',
])

export interface MissingPathHintOptions {
  /**
   * Ceiling for enumeration. Hints never describe anything outside it, and
   * without one no enumeration happens at all.
   */
  root?: string
  signal?: AbortSignal
  now?: () => number
  /** Test seam for the wall-clock budget. */
  scanTimeBudgetMs?: number
}

export interface MissingPathHint {
  /** Nearest existing ancestor directory of the requested path. */
  nearestDirectory?: string
  /** First path segment (relative to nearestDirectory) that does not exist. */
  missingSegment?: string
  /** Bounded, sorted entries of nearestDirectory (directories end with "/"). */
  entries: string[]
  omittedEntries: number
  /** Same-named (or same-stem) files elsewhere under the root, root-relative. */
  candidates: string[]
}

function isInside(candidate: string, parent: string): boolean {
  const rel = relative(parent, candidate)
  return rel === '' || (!rel.startsWith('..') && !rel.startsWith(sep) && !/^[A-Za-z]:/u.test(rel))
}

function displayPath(target: string, root: string): string {
  const rel = relative(root, target)
  if (rel === '') return '.'
  return rel.split(sep).join('/')
}

function formatEntry(entry: Dirent): string {
  if (entry.isDirectory()) return `${entry.name}/`
  return entry.name
}

async function directoryExists(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isDirectory()
  } catch {
    return false
  }
}

async function fileExists(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isFile()
  } catch {
    return false
  }
}

function stemOf(name: string): string {
  const ext = extname(name)
  return ext ? name.slice(0, -ext.length) : name
}

export async function collectMissingPathHint(
  resolvedPath: string,
  options: MissingPathHintOptions = {},
): Promise<MissingPathHint | null> {
  if (!options.root) return null
  const root = resolve(options.root)
  const target = resolve(resolvedPath)
  if (!isInside(target, root) || target === root) return null
  if (!(await directoryExists(root))) return null

  const now = options.now ?? (() => Date.now())
  const deadline = now() + (options.scanTimeBudgetMs ?? SCAN_TIME_BUDGET_MS)
  const aborted = (): boolean => options.signal?.aborted === true || now() > deadline

  // 1. Nearest existing ancestor and the first missing segment below it.
  let nearest = dirname(target)
  let missingChild = target
  while (nearest !== root && isInside(nearest, root) && !(await directoryExists(nearest))) {
    missingChild = nearest
    nearest = dirname(nearest)
  }
  if (!(await directoryExists(nearest))) return null

  const hint: MissingPathHint = {
    nearestDirectory: nearest,
    missingSegment: basename(missingChild),
    entries: [],
    omittedEntries: 0,
    candidates: [],
  }

  let nearestEntries: Dirent[] = []
  try {
    nearestEntries = (await readdir(nearest, { withFileTypes: true }))
      .filter((entry) => !entry.name.startsWith('.'))
      .sort((left, right) => left.name.localeCompare(right.name))
  } catch {
    nearestEntries = []
  }
  hint.entries = nearestEntries.slice(0, MAX_LISTED_ENTRIES).map(formatEntry)
  hint.omittedEntries = Math.max(0, nearestEntries.length - MAX_LISTED_ENTRIES)

  // 2. Candidates: same name up the ancestor chain, near-name matches in the
  //    nearest directory, then a bounded breadth-first scan under the root.
  const wanted = basename(target)
  const wantedLower = wanted.toLowerCase()
  const wantedStem = stemOf(wantedLower)
  const seen = new Set<string>()
  const addCandidate = (path: string): void => {
    if (hint.candidates.length >= MAX_CANDIDATES) return
    const rel = displayPath(path, root)
    if (seen.has(rel)) return
    seen.add(rel)
    hint.candidates.push(rel)
  }

  for (let dir = nearest; isInside(dir, root); dir = dirname(dir)) {
    if (dir !== dirname(target)) {
      const candidate = join(dir, wanted)
      if (await fileExists(candidate)) addCandidate(candidate)
    }
    if (dir === root) break
  }

  for (const entry of nearestEntries) {
    if (!entry.isFile()) continue
    const lower = entry.name.toLowerCase()
    if (lower === wantedLower || stemOf(lower) === wantedStem) {
      addCandidate(join(nearest, entry.name))
    }
  }

  if (hint.candidates.length < MAX_CANDIDATES && !aborted()) {
    const queue: Array<{ dir: string; depth: number }> = [{ dir: root, depth: 0 }]
    let scanned = 0
    while (queue.length > 0 && scanned < MAX_SCAN_DIRS && hint.candidates.length < MAX_CANDIDATES) {
      if (aborted()) break
      const { dir, depth } = queue.shift()!
      scanned += 1
      let entries: Dirent[]
      try {
        entries = await readdir(dir, { withFileTypes: true })
      } catch {
        continue
      }
      for (const entry of entries) {
        if (entry.isSymbolicLink()) continue
        if (entry.isDirectory()) {
          if (depth < MAX_SCAN_DEPTH && !SKIP_DIR_NAMES.has(entry.name) && !entry.name.startsWith('.')) {
            queue.push({ dir: join(dir, entry.name), depth: depth + 1 })
          }
          continue
        }
        if (entry.isFile() && entry.name.toLowerCase() === wantedLower) {
          addCandidate(join(dir, entry.name))
        }
      }
    }
  }

  return hint
}

export function formatMissingPathHint(
  hint: MissingPathHint | null,
  root: string | undefined,
): string[] {
  if (!hint || !root) return []
  const resolvedRoot = resolve(root)
  const lines: string[] = []
  if (hint.nearestDirectory) {
    const dir = displayPath(hint.nearestDirectory, resolvedRoot)
    const where = dir === '.' ? 'the working directory' : `${dir}/`
    const entries = hint.entries.length > 0
      ? `${hint.entries.join(', ')}${hint.omittedEntries > 0 ? `, … +${hint.omittedEntries} more` : ''}`
      : '(empty)'
    lines.push(
      `Nearest existing directory is ${where}; it has no "${hint.missingSegment ?? ''}". Entries: ${entries}`,
    )
  }
  if (hint.candidates.length > 0) {
    lines.push(`Same-named files under the working directory: ${hint.candidates.join(', ')}`)
  }
  if (lines.length > 0) {
    lines.push('Do not retry this path. Use one of the listed paths, or fs.glob/fs.search to locate the file.')
  }
  return lines
}

/**
 * Convenience for tool error paths: message lines to append after the
 * primary "not found" line. Never throws.
 */
export async function describeMissingPath(
  resolvedPath: string,
  options: MissingPathHintOptions = {},
): Promise<string[]> {
  try {
    const hint = await collectMissingPathHint(resolvedPath, options)
    return formatMissingPathHint(hint, options.root)
  } catch {
    return []
  }
}
