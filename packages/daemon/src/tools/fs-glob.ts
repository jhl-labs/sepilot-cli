import { spawn } from 'node:child_process'
import { readdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { throwIfAborted } from '../abort.js'
import { classifyToolCwd, rejectToolCwd, resolveToolCwd } from './path-utils.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

const DEFAULT_GLOB_LIMIT = 1_000
const MAX_GLOB_LIMIT = 10_000
const MAX_STDERR_CHARS = 8_192
const MAX_GLOB_PATTERN_EXPANSIONS = 64
const DEFAULT_EXCLUDED_DIRS = [
  'node_modules',
  'dist',
  'build',
  'out',
  'coverage',
]

interface GlobMatches {
  matches: string[]
  truncated: boolean
  offset: number
  nextOffset?: number
}

function normalizePatterns(input: Record<string, unknown>): string[] {
  const patterns = input.patterns
  if (Array.isArray(patterns)) {
    return patterns
      .filter((value): value is string => typeof value === 'string')
      .map((value) => value.trim())
      .filter(Boolean)
  }
  if (typeof input.pattern === 'string' && input.pattern.trim()) {
    return [input.pattern.trim()]
  }
  return []
}

function normalizeLimit(input: Record<string, unknown>): number {
  const raw = input.limit ?? input.maxResults
  const numeric = typeof raw === 'number'
    ? raw
    : typeof raw === 'string' && raw.trim()
      ? Number(raw)
      : DEFAULT_GLOB_LIMIT

  if (!Number.isFinite(numeric) || numeric <= 0) {
    return DEFAULT_GLOB_LIMIT
  }
  return Math.min(MAX_GLOB_LIMIT, Math.floor(numeric))
}

function normalizeOffset(input: Record<string, unknown>): number {
  const raw = input.offset
  const numeric = typeof raw === 'number'
    ? raw
    : typeof raw === 'string' && raw.trim()
      ? Number(raw)
      : 0

  if (!Number.isFinite(numeric) || numeric <= 0) {
    return 0
  }
  return Math.floor(numeric)
}

function formatGlobOutput(result: GlobMatches, limit: number): string {
  if (result.matches.length === 0) {
    return result.offset > 0
      ? `[no matches after offset ${result.offset}]`
      : '[no matches]'
  }

  const lines = [...result.matches]
  if (result.truncated) {
    const nextOffset = result.nextOffset ?? result.offset + result.matches.length
    lines.push(
      `[truncated after ${limit} matches; continue with offset ${nextOffset}, narrow pattern, or pass a higher limit]`,
    )
  }
  return lines.join('\n')
}

function escapeRegexChar(char: string): string {
  return /[\\^$+?.()|[\]{}]/.test(char) ? `\\${char}` : char
}

function expandGlobPatternAlternatives(pattern: string): string[] {
  let expanded = [pattern]
  while (true) {
    const next: string[] = []
    let changed = false
    for (const candidate of expanded) {
      const alternative = /\{([^{}]*,[^{}]*)\}/.exec(candidate)
      if (!alternative) {
        next.push(candidate)
        continue
      }
      changed = true
      const options = alternative[1]!.split(',')
      if (next.length + options.length > MAX_GLOB_PATTERN_EXPANSIONS) {
        // A partial expansion would create false absence claims for the
        // discarded alternatives. Keep the original literal and fail closed.
        return [pattern]
      }
      for (const option of options) {
        next.push(
          `${candidate.slice(0, alternative.index)}${option}${candidate.slice(alternative.index + alternative[0].length)}`,
        )
      }
    }
    expanded = next
    if (!changed) {
      return expanded
    }
  }
}

function globPatternToRegExp(pattern: string): RegExp {
  const normalized = pattern.replaceAll('\\', '/')
  // ripgrep treats a slash-less pattern (e.g. `*.ts`) as a basename match at any
  // depth; the Node fallback must do the same, otherwise a bare pattern only
  // matches top-level files and the two backends diverge.
  let source = normalized.includes('/') ? '^' : '^(?:.*/)?'

  for (let index = 0; index < normalized.length; index += 1) {
    const char = normalized[index]
    const next = normalized[index + 1]

    if (char === '*') {
      if (next === '*') {
        while (normalized[index + 1] === '*') {
          index += 1
        }
        if (normalized[index + 1] === '/') {
          source += '(?:.*/)?'
          index += 1
        } else {
          source += '.*'
        }
        continue
      }
      source += '[^/]*'
      continue
    }

    if (char === '?') {
      source += '[^/]'
      continue
    }

    source += escapeRegexChar(char ?? '')
  }

  return new RegExp(`${source}$`)
}

interface GitignoreRule {
  negated: boolean
  dirOnly: boolean
  regex: RegExp
}

function gitignorePatternToRule(rawLine: string): GitignoreRule | null {
  let line = rawLine.trim()
  if (!line || line.startsWith('#')) return null
  const negated = line.startsWith('!')
  if (negated) line = line.slice(1)
  let dirOnly = false
  if (line.endsWith('/')) {
    dirOnly = true
    line = line.slice(0, -1)
  }
  const anchored = line.startsWith('/') || line.slice(0, -1).includes('/')
  if (line.startsWith('/')) line = line.slice(1)
  if (!line) return null

  let source = ''
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index]!
    if (char === '*') {
      if (line[index + 1] === '*') {
        while (line[index + 1] === '*') index += 1
        if (line[index + 1] === '/') {
          source += '(?:.*/)?'
          index += 1
        } else {
          source += '.*'
        }
      } else {
        source += '[^/]*'
      }
    } else if (char === '?') {
      source += '[^/]'
    } else {
      source += escapeRegexChar(char)
    }
  }
  // A slash-less pattern matches at any depth (basename); an anchored one is
  // rooted at the search dir. Either way a matched directory ignores its whole
  // subtree, so allow an optional trailing path segment.
  const prefix = anchored ? '^' : '^(?:.*/)?'
  return { negated, dirOnly, regex: new RegExp(`${prefix}${source}(?:/.*)?$`) }
}

async function loadGitignoreRules(cwd: string): Promise<GitignoreRule[]> {
  let text: string
  try {
    text = await readFile(join(cwd, '.gitignore'), 'utf8')
  } catch {
    return []
  }
  const rules: GitignoreRule[] = []
  for (const line of text.split(/\r?\n/)) {
    const rule = gitignorePatternToRule(line)
    if (rule) rules.push(rule)
  }
  return rules
}

function isGitignored(
  relativePath: string,
  isDirectory: boolean,
  rules: GitignoreRule[],
): boolean {
  let ignored = false
  for (const rule of rules) {
    if (rule.dirOnly && !isDirectory) continue
    if (rule.regex.test(relativePath)) {
      ignored = !rule.negated
    }
  }
  return ignored
}

async function listFilesForGlob(
  cwd: string,
  hidden: boolean,
  regexes: RegExp[],
  excludedDirs: ReadonlySet<string>,
  gitignoreRules: GitignoreRule[],
  limit: number,
  offset: number,
  signal: AbortSignal | undefined,
): Promise<GlobMatches> {
  const matches: string[] = []
  let truncated = false
  let seen = 0

  async function walk(directory: string, prefix: string): Promise<void> {
    if (truncated) {
      return
    }
    throwIfAborted(signal, 'Filesystem glob aborted')
    const entries = await readdir(directory, { withFileTypes: true })
    entries.sort((left, right) => left.name.localeCompare(right.name))

    for (const entry of entries) {
      if (truncated) {
        return
      }
      if (!hidden && entry.name.startsWith('.')) {
        continue
      }
      // ripgrep never descends into .git even with --hidden; match that.
      if (entry.isDirectory() && entry.name === '.git') {
        continue
      }
      if (entry.isDirectory() && excludedDirs.has(entry.name)) {
        continue
      }

      const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name
      if (isGitignored(relativePath, entry.isDirectory(), gitignoreRules)) {
        continue
      }
      const fullPath = join(directory, entry.name)
      if (entry.isDirectory()) {
        await walk(fullPath, relativePath)
      } else if (entry.isFile() && regexes.some((regex) => regex.test(relativePath))) {
        seen += 1
        if (seen <= offset) {
          continue
        }
        if (matches.length >= limit) {
          truncated = true
          return
        }
        matches.push(relativePath)
      }
    }
  }

  await walk(cwd, '')
  return {
    matches,
    truncated,
    offset,
    nextOffset: truncated ? offset + matches.length : undefined,
  }
}

async function globWithoutRipgrep(
  cwd: string,
  patterns: string[],
  hidden: boolean,
  excludedDirs: ReadonlySet<string>,
  includeIgnored: boolean,
  limit: number,
  offset: number,
  signal: AbortSignal | undefined,
): Promise<GlobMatches> {
  const regexes = patterns
    .flatMap(expandGlobPatternAlternatives)
    .map(globPatternToRegExp)
  // ripgrep honors .gitignore by default; mirror that in the fallback so the
  // two backends return the same set. includeIgnored disables it on both sides.
  const gitignoreRules = includeIgnored ? [] : await loadGitignoreRules(cwd)
  return listFilesForGlob(
    cwd,
    hidden,
    regexes,
    excludedDirs,
    gitignoreRules,
    limit,
    offset,
    signal,
  )
}

interface GlobPatternGroup {
  patterns: string[]
  hidden: boolean
}

function patternExplicitlyTargetsHiddenPath(pattern: string): boolean {
  const normalized = pattern.replaceAll('\\', '/').replace(/^\.\/+/, '')
  return normalized.split('/').some((segment) =>
    segment.startsWith('.') && segment !== '.' && segment !== '..',
  )
}

function groupPatternsByHiddenScope(
  patterns: string[],
  includeAllHidden: boolean,
): GlobPatternGroup[] {
  if (includeAllHidden) {
    return [{ patterns, hidden: true }]
  }

  const visiblePatterns: string[] = []
  const explicitHiddenPatterns: string[] = []
  for (const pattern of patterns) {
    if (patternExplicitlyTargetsHiddenPath(pattern)) {
      explicitHiddenPatterns.push(pattern)
    } else {
      visiblePatterns.push(pattern)
    }
  }

  return [
    ...(visiblePatterns.length > 0
      ? [{ patterns: visiblePatterns, hidden: false }]
      : []),
    ...(explicitHiddenPatterns.length > 0
      ? [{ patterns: explicitHiddenPatterns, hidden: true }]
      : []),
  ]
}

function mergeGlobMatches(
  results: GlobMatches[],
  limit: number,
  offset: number,
): GlobMatches {
  const combined = [...new Set(results.flatMap((result) => result.matches))].sort()
  const matches = combined.slice(offset, offset + limit)
  const truncated = combined.length > offset + limit
    || results.some((result) => result.truncated)
  return {
    matches,
    truncated,
    offset,
    nextOffset: truncated ? offset + matches.length : undefined,
  }
}

async function runGlobPatternGroups(
  groups: GlobPatternGroup[],
  limit: number,
  offset: number,
  runGroup: (
    group: GlobPatternGroup,
    groupLimit: number,
    groupOffset: number,
  ) => Promise<GlobMatches>,
): Promise<GlobMatches> {
  if (groups.length === 1) {
    return runGroup(groups[0]!, limit, offset)
  }

  // Apply pagination after merging both path-sorted, disjoint scopes. Paging
  // each scope first could hide an earlier explicit dot-directory match behind
  // a later visible match (or vice versa).
  const groupLimit = Math.min(Number.MAX_SAFE_INTEGER, offset + limit)
  const results: GlobMatches[] = []
  for (const group of groups) {
    results.push(await runGroup(group, groupLimit, 0))
  }
  return mergeGlobMatches(results, limit, offset)
}

function patternExplicitlyTargetsDir(pattern: string, directory: string): boolean {
  const normalized = pattern.replaceAll('\\', '/').replace(/^\.\/+/, '')
  return normalized === directory
    || normalized.startsWith(`${directory}/`)
    || normalized.includes(`/${directory}/`)
    || normalized.includes(`**/${directory}/`)
}

function defaultExcludedDirsForGlob(patterns: string[], includeIgnored: boolean): string[] {
  if (includeIgnored) {
    return []
  }
  return DEFAULT_EXCLUDED_DIRS.filter((directory) =>
    !patterns.some((pattern) => patternExplicitlyTargetsDir(pattern, directory)),
  )
}

function isMissingRipgrep(error: unknown): boolean {
  return (
    typeof error === 'object'
    && error !== null
    && 'code' in error
    && error.code === 'ENOENT'
  )
}

function appendLimitedStderr(current: string, chunk: Buffer): string {
  if (current.length >= MAX_STDERR_CHARS) {
    return current
  }
  return (current + chunk.toString('utf8')).slice(0, MAX_STDERR_CHARS)
}

async function runRipgrepGlob(
  cwd: string,
  args: string[],
  limit: number,
  offset: number,
  signal: AbortSignal | undefined,
): Promise<GlobMatches> {
  return new Promise((resolve, reject) => {
    const child = spawn('rg', args, { cwd, signal })
    const matches: string[] = []
    let seen = 0
    let stdoutRemainder = ''
    let stderr = ''
    let truncated = false
    let settled = false

    function stopAfterLimit() {
      if (truncated) {
        return
      }
      truncated = true
      child.kill('SIGTERM')
    }

    function addLine(line: string) {
      if (!line) {
        return
      }
      seen += 1
      if (seen <= offset) {
        return
      }
      if (matches.length >= limit) {
        stopAfterLimit()
        return
      }
      matches.push(line)
    }

    child.stdout.on('data', (chunk: Buffer) => {
      stdoutRemainder += chunk.toString('utf8')
      const lines = stdoutRemainder.split(/\r?\n/)
      stdoutRemainder = lines.pop() ?? ''
      for (const line of lines) {
        addLine(line)
        if (truncated) {
          break
        }
      }
    })

    child.stdout.on('end', () => {
      if (!truncated && stdoutRemainder) {
        addLine(stdoutRemainder)
      }
    })

    child.stderr.on('data', (chunk: Buffer) => {
      stderr = appendLimitedStderr(stderr, chunk)
    })

    child.on('error', (error) => {
      if (settled) {
        return
      }
      settled = true
      reject(error)
    })

    child.on('close', (code, termSignal) => {
      if (settled) {
        return
      }
      settled = true
      if (truncated) {
        resolve({
          matches,
          truncated: true,
          offset,
          nextOffset: offset + matches.length,
        })
        return
      }
      if (code === 0) {
        resolve({ matches, truncated: false, offset })
        return
      }
      if (code === 1) {
        resolve({ matches: [], truncated: false, offset })
        return
      }
      reject(new Error(
        stderr.trim()
        || `rg exited with code ${code ?? 'null'}${termSignal ? ` signal ${termSignal}` : ''}`,
      ))
    })
  })
}

export function createFsGlobTool(): ToolDefinitionRuntime {
  return {
    name: 'fs.glob',
    description: 'List files matching one or more glob patterns under a directory. A literal dot-directory segment such as ".github" is included for that pattern; use hidden=true only when broad patterns should include hidden paths too.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'filesystem' },
    observationCoverage: {
      covers: (observedInput, requestedInput, context) => {
        const observedCwd = resolveToolCwd(observedInput.cwd, context.cwd)
        const requestedCwd = resolveToolCwd(requestedInput.cwd, context.cwd)
        if (observedCwd !== requestedCwd) return false
        const observedPatterns = normalizePatterns(observedInput).sort()
        const requestedPatterns = normalizePatterns(requestedInput).sort()
        if (JSON.stringify(observedPatterns) !== JSON.stringify(requestedPatterns)) return false
        if ((observedInput.hidden === true) !== (requestedInput.hidden === true)) return false
        if ((observedInput.includeIgnored === true) !== (requestedInput.includeIgnored === true)) return false
        const observedOffset = normalizeOffset(observedInput)
        const requestedOffset = normalizeOffset(requestedInput)
        const observedEnd = observedOffset + normalizeLimit(observedInput)
        const requestedEnd = requestedOffset + normalizeLimit(requestedInput)
        return observedOffset <= requestedOffset && observedEnd >= requestedEnd
      },
    },
    inputSchema: {
      type: 'object',
      properties: {
        cwd: { type: 'string', description: 'Directory to search. Supports ~/ paths. Defaults to the active session cwd.' },
        pattern: { type: 'string', description: 'Single glob pattern, for example "**/*.ts".' },
        patterns: { type: 'array', items: { type: 'string' }, description: 'Multiple glob patterns.' },
        hidden: { type: 'boolean', description: 'Let all patterns include hidden files. Explicit dot-directory patterns already include only their matching hidden paths.' },
        includeIgnored: {
          type: 'boolean',
          description: 'Include dependency/build output directories that fs.glob normally skips for broad repository discovery, such as node_modules and dist.',
        },
        limit: { type: 'number', description: `Maximum number of matches to return. Defaults to ${DEFAULT_GLOB_LIMIT}; maximum ${MAX_GLOB_LIMIT}.` },
        maxResults: { type: 'number', description: 'Alias for limit.' },
        offset: { type: 'number', description: 'Zero-based match offset for paging through broad results after a truncated response.' },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const cwd = resolveToolCwd(input.cwd, context?.cwd)
      const patterns = normalizePatterns(input)
      const limit = normalizeLimit(input)
      const offset = normalizeOffset(input)
      const excludedDirs = defaultExcludedDirsForGlob(patterns, input.includeIgnored === true)
      if (patterns.length === 0) {
        return {
          output: 'pattern or patterns is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      const patternGroups = groupPatternsByHiddenScope(patterns, input.hidden === true)

      try {
        throwIfAborted(context?.signal, 'Filesystem glob aborted')
        const cwdKind = await classifyToolCwd(cwd)
        if (cwdKind !== 'directory') {
          const rejection = rejectToolCwd(cwd, cwdKind, { scopeParameter: 'pattern' })
          return {
            output: rejection.output,
            status: 'error',
            durationMs: Date.now() - start,
            code: rejection.code,
          }
        }
        const result = await runGlobPatternGroups(
          patternGroups,
          limit,
          offset,
          async (group, groupLimit, groupOffset) => {
            // Do not inherit RIPGREP_CONFIG_PATH. A user config can enable
            // --follow and make a workspace glob traverse an outside symlink.
            const args = ['--no-config', '--files', '--sort', 'path']
            if (group.hidden) {
              args.push('--hidden')
            }
            for (const pattern of group.patterns) {
              args.push('-g', pattern)
            }
            for (const directory of excludedDirs) {
              args.push('-g', `!**/${directory}/**`)
            }
            return runRipgrepGlob(
              cwd,
              args,
              groupLimit,
              groupOffset,
              context?.signal,
            )
          },
        )
        return {
          output: formatGlobOutput(result, limit),
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (error) {
        if (isMissingRipgrep(error)) {
          const result = await runGlobPatternGroups(
            patternGroups,
            limit,
            offset,
            (group, groupLimit, groupOffset) => globWithoutRipgrep(
              cwd,
              group.patterns,
              group.hidden,
              new Set(excludedDirs),
              input.includeIgnored === true,
              groupLimit,
              groupOffset,
              context?.signal,
            ),
          )
          return {
            output: formatGlobOutput(result, limit),
            status: 'success',
            durationMs: Date.now() - start,
          }
        }
        if (
          typeof error === 'object'
          && error !== null
          && 'code' in error
          && error.code === 1
        ) {
          return {
            output: '[no matches]',
            status: 'success',
            durationMs: Date.now() - start,
          }
        }
        const message = error instanceof Error ? error.message : String(error)
        return {
          output: message,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
    },
  }
}
