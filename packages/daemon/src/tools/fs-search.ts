import { readFile } from 'node:fs/promises'
import { basename, dirname, relative } from 'node:path'
import { throwIfAborted } from '../abort.js'
import { enforceOutputLimit, resolveToolOutputMaxBytes } from './output-limit.js'
import { collectFallbackFiles, isEnoent } from './fallback-file-search.js'
import { classifyToolCwd, rejectToolCwd, resolveToolCwd } from './path-utils.js'

// `limit` bounds matches per file; a broad query over a large tree still
// returned tens of thousands of characters under the byte cap. Bound the
// whole result too and say which files were left out, so the model narrows
// the query instead of reading a truncated dump.
const MAX_TOTAL_MATCH_LINES = 200
import { runRipgrepCapture } from './ripgrep-run.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

function buildSearchMatcher(query: string, fixedStrings: boolean): (line: string) => number {
  if (fixedStrings) {
    return (line) => line.indexOf(query)
  }

  let regex: RegExp
  try {
    regex = new RegExp(query)
  } catch {
    return (line) => line.indexOf(query)
  }
  return (line) => {
    const match = regex.exec(line)
    return match?.index ?? -1
  }
}

async function fallbackSearch(options: {
  cwd: string
  query: string
  glob?: string
  fixedStrings: boolean
  limit: number
  signal?: AbortSignal
}): Promise<string> {
  const matcher = buildSearchMatcher(options.query, options.fixedStrings)
  const files = await collectFallbackFiles({
    root: options.cwd,
    glob: options.glob,
    signal: options.signal,
    abortMessage: 'Filesystem search aborted',
  })
  const lines: string[] = []
  for (const file of files) {
    throwIfAborted(options.signal, 'Filesystem search aborted')
    let text: string
    try {
      text = await readFile(file, 'utf-8')
    } catch {
      continue
    }
    if (text.includes('\0')) {
      continue
    }
    const rel = relative(options.cwd, file).split('\\').join('/')
    let count = 0
    const fileLines = text.split(/\r?\n/)
    for (let index = 0; index < fileLines.length; index += 1) {
      const line = fileLines[index]!
      const column = matcher(line)
      if (column < 0) {
        continue
      }
      lines.push(`${rel}:${index + 1}:${column + 1}:${line}`)
      count += 1
      if (count >= options.limit) {
        break
      }
    }
  }
  return lines.join('\n')
}

export function createFsSearchTool(): ToolDefinitionRuntime {
  return {
    name: 'fs.search',
    description:
      'Search file contents under a directory with ripgrep and return matching lines as `path:line:col:text`. `limit` caps matches per file (default 100, max 500). Prefer fs.read for content of a known file and fs.glob for filename-only matches; reach for fs.search when you need to locate text across an unknown set of files. Narrow with `glob: "*.ts"` and `fixedStrings: true` first; broad regex over a large repo is slow and noisy.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'filesystem' },
    inputSchema: {
      type: 'object',
      properties: {
        cwd: { type: 'string', description: 'Directory to search. Supports ~/ paths. Defaults to the active session cwd. A file path scopes the search to that single file.' },
        query: { type: 'string', description: 'Text or regex to search for.' },
        glob: { type: 'string', description: 'Optional glob filter such as "*.ts".' },
        fixedStrings: { type: 'boolean', description: 'Treat query as literal text. Defaults to true.' },
        hidden: { type: 'boolean', description: 'Include hidden files.' },
        limit: {
          type: 'number',
          description:
            'Maximum number of matches PER FILE (default 100, capped at 500). When a file hits this limit the result is truncated for that file — narrow the query if that happens.',
        },
      },
      required: ['query'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const requestedCwd = resolveToolCwd(input.cwd, context?.cwd)
      const query = typeof input.query === 'string' ? input.query.trim() : ''
      if (!query) {
        return {
          output: 'query is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }

      const limit = typeof input.limit === 'number' && Number.isFinite(input.limit)
        ? Math.max(1, Math.min(500, Math.trunc(input.limit)))
        : 100
      const args = [
        // Do not inherit RIPGREP_CONFIG_PATH. A user config can enable
        // --follow and make a workspace search traverse an outside symlink.
        '--no-config',
        '--line-number',
        '--column',
        '--color',
        'never',
        '--max-count',
        String(limit),
      ]
      if ((input.fixedStrings as boolean | undefined) !== false) {
        args.push('--fixed-strings')
      }
      if (input.hidden === true) {
        args.push('--hidden')
      }
      try {
        throwIfAborted(context?.signal, 'Filesystem search aborted')
        const cwdKind = await classifyToolCwd(requestedCwd)
        if (cwdKind === 'missing') {
          const rejection = rejectToolCwd(requestedCwd, 'missing')
          return {
            output: rejection.output,
            status: 'error',
            durationMs: Date.now() - start,
            code: rejection.code,
          }
        }
        // "Search this file" is an unambiguous scope. Honour it instead of
        // sending the model on another round trip: search from the file's
        // directory with the file itself as the only target.
        const fileScope = cwdKind === 'file' ? basename(requestedCwd) : null
        const cwd = fileScope ? dirname(requestedCwd) : requestedCwd
        if (!fileScope && typeof input.glob === 'string' && input.glob.trim()) {
          args.push('-g', input.glob)
        }
        // A model-controlled query can begin with "--". Without the option
        // terminator ripgrep interprets values such as --follow as flags and can
        // traverse a workspace symlink into an outside directory.
        args.push('--', query, fileScope ?? '.')
        const scopeNote = fileScope
          ? `[fs.search: cwd ${requestedCwd} is a file; searched only that file, paths are relative to ${cwd}]\n`
          : ''
        // Stream rg with a byte cap instead of a 10MB execFile maxBuffer that
        // throws (losing every match) on huge result sets.
        const maxBytes = resolveToolOutputMaxBytes('fs.search')
        let stdout: string
        let streamTruncated = false
        try {
          const result = await runRipgrepCapture({
            args,
            cwd,
            maxBytes,
            signal: context?.signal,
          })
          stdout = result.stdout
          streamTruncated = result.truncated
        } catch (error) {
          if (!isEnoent(error)) {
            throw error
          }
          stdout = await fallbackSearch({
            cwd,
            query,
            glob: fileScope ?? (typeof input.glob === 'string' ? input.glob : undefined),
            fixedStrings: (input.fixedStrings as boolean | undefined) !== false,
            limit,
            signal: context?.signal,
          })
        }
        const text = stdout.trim()
        if (!text) {
          return {
            output: '[no matches]',
            status: 'success',
            durationMs: Date.now() - start,
          }
        }
        // Detect any file that hit the per-file cap so the agent knows
        // the result for that file is incomplete and can decide to
        // narrow the query rather than treating it as exhaustive.
        const allLines = text.split('\n').filter(Boolean)
        const matchesByFile = new Map<string, number>()
        for (const line of allLines) {
          const file = line.split(':', 1)[0]
          matchesByFile.set(file, (matchesByFile.get(file) ?? 0) + 1)
        }
        const lines = allLines.slice(0, MAX_TOTAL_MATCH_LINES)
        let totalHint = ''
        if (allLines.length > MAX_TOTAL_MATCH_LINES) {
          const shownFiles = new Set(lines.map((line) => line.split(':', 1)[0]))
          const omittedFiles = [...matchesByFile.entries()].filter(([file]) => !shownFiles.has(file))
          const omittedPreview = omittedFiles
            .slice(0, 8)
            .map(([file, count]) => `${file} (${count})`)
            .join(', ')
          totalHint = `\n[fs.search: showing ${lines.length} of ${allLines.length} matching lines across ${matchesByFile.size} files; ${omittedFiles.length} file${omittedFiles.length === 1 ? '' : 's'} omitted entirely${omittedPreview ? ` — ${omittedPreview}${omittedFiles.length > 8 ? ', …' : ''}` : ''}. Narrow the query, restrict the path with \`glob\`, or search a subdirectory.]`
        }
        const cappedFiles = [...matchesByFile.entries()]
          .filter(([, count]) => count >= limit)
          .map(([file]) => file)
        const cappedHint = cappedFiles.length > 0
          ? `\n[fs.search: ${cappedFiles.length} file${cappedFiles.length === 1 ? '' : 's'} hit the per-file limit of ${limit} matches (e.g. ${cappedFiles.slice(0, 3).join(', ')}). Narrow the query or pass a smaller \`glob\` to see what was elided.]`
          : ''
        const streamHint = streamTruncated
          ? `\n[fs.search: results truncated at ${maxBytes} bytes; narrow the query, restrict the path with \`glob\`, or lower \`limit\` to see the rest.]`
          : ''
        const limited = enforceOutputLimit(scopeNote + lines.join('\n') + totalHint + cappedHint + streamHint, {
          toolName: 'fs.search',
          resumeHint: 'narrow the query, restrict the path with `glob`, or lower `limit`',
        })
        return {
          output: limited.output,
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (error) {
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
