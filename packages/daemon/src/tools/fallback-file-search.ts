import { readdir } from 'node:fs/promises'
import { join, relative } from 'node:path'
import { throwIfAborted } from '../abort.js'

export const FALLBACK_SEARCH_SKIP_DIRS = new Set([
  '.git',
  'node_modules',
  'dist',
  'coverage',
  '__pycache__',
  '.pytest_cache',
  '.mypy_cache',
  '.tox',
  '.venv',
  'venv',
])

export function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function globToRegExp(glob: string): RegExp {
  let source = ''
  for (let index = 0; index < glob.length; index += 1) {
    const char = glob[index]!
    const next = glob[index + 1]
    if (char === '*') {
      if (next === '*') {
        const afterNext = glob[index + 2]
        if (afterNext === '/') {
          source += '(?:.*/)?'
          index += 2
        } else {
          source += '.*'
          index += 1
        }
      } else {
        source += '[^/]*'
      }
    } else if (char === '?') {
      source += '[^/]'
    } else {
      source += escapeRegExp(char)
    }
  }
  return new RegExp(`^${source}$`)
}

export function matchesGlob(relativePath: string, glob: string | undefined): boolean {
  const normalized = relativePath.split('\\').join('/')
  const pattern = glob?.trim()
  if (!pattern) {
    return true
  }
  const regex = globToRegExp(pattern)
  if (!pattern.includes('/')) {
    return regex.test(normalized.split('/').at(-1) ?? normalized)
  }
  return regex.test(normalized)
}

export function isEnoent(error: unknown): boolean {
  return Boolean(
    error
    && typeof error === 'object'
    && 'code' in error
    && error.code === 'ENOENT',
  )
}

export async function collectFallbackFiles(options: {
  root: string
  current?: string
  glob?: string
  signal?: AbortSignal
  abortMessage: string
  skipDirs?: ReadonlySet<string>
}): Promise<string[]> {
  const {
    root,
    current = root,
    glob,
    signal,
    abortMessage,
    skipDirs = FALLBACK_SEARCH_SKIP_DIRS,
  } = options
  throwIfAborted(signal, abortMessage)
  const entries = await readdir(current, { withFileTypes: true })
  const files: string[] = []
  for (const entry of entries) {
    const path = join(current, entry.name)
    if (entry.isDirectory()) {
      if (skipDirs.has(entry.name)) {
        continue
      }
      files.push(...await collectFallbackFiles({
        root,
        current: path,
        glob,
        signal,
        abortMessage,
        skipDirs,
      }))
    } else if (entry.isFile()) {
      const rel = relative(root, path).split('\\').join('/')
      if (matchesGlob(rel, glob)) {
        files.push(path)
      }
    }
  }
  return files
}
