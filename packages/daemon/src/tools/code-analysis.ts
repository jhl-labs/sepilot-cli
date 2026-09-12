import { readFile } from 'node:fs/promises'
import { relative } from 'node:path'
import type { LspClient } from '../lsp/client.js'
import type { LspDiagnostic } from '../lsp/diagnostics.js'
import { throwIfAborted } from '../abort.js'
import { enforceOutputLimit, resolveToolOutputMaxBytes } from './output-limit.js'
import { classifyToolCwd, rejectToolCwd, resolveToolCwd, resolveToolPath } from './path-utils.js'
import { collectFallbackFiles, escapeRegExp, isEnoent } from './fallback-file-search.js'
import { runRipgrepCapture } from './ripgrep-run.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

export interface CodeDiagnosticsToolDeps {
  getClient: (language: string) => LspClient | null
  /** File reader for opening the document with the server. Defaults to fs. */
  readFile?: (path: string) => Promise<string>
  /** Settle window (ms) after a fresh didOpen before reading diagnostics. */
  settleMs?: number
}

const DEFAULT_DIAGNOSTICS_SETTLE_MS = 400

function fileUriToPath(uri: string): string | null {
  if (!uri.startsWith('file://')) return null
  try {
    return decodeURIComponent(new URL(uri).pathname)
  } catch {
    return null
  }
}

function formatDiagnostic(diagnostic: LspDiagnostic): string {
  const severity = diagnostic.severity ? `[${diagnostic.severity}] ` : ''
  const source = diagnostic.source ? ` (${diagnostic.source})` : ''
  return `L${diagnostic.line}: ${severity}${diagnostic.message}${source}`
}

function classifyDependency(moduleName: string): 'local' | 'external' {
  return moduleName.startsWith('.') || moduleName.startsWith('/')
    ? 'local'
    : 'external'
}

interface DependencyMatch {
  module: string
  kind: string
  line: number
  scope: 'local' | 'external'
}

// Whole-file ES/CommonJS patterns. `[^'";]*?` between `import`/`export` and
// `from` spans newlines but never crosses a quote or statement terminator, so a
// multi-line `import {\n a,\n b,\n} from 'mod'` is captured while two adjacent
// statements never bridge into one another.
const WHOLE_FILE_DEPENDENCY_PATTERNS: Array<{ pattern: RegExp; kind: string }> = [
  { pattern: /\bimport\b[^'";]*?\bfrom\s*["']([^"']+)["']/g, kind: 'import' },
  { pattern: /\bexport\b[^'";]*?\bfrom\s*["']([^"']+)["']/g, kind: 'export_from' },
  { pattern: /\brequire\(\s*["']([^"']+)["']\s*\)/g, kind: 'require' },
  { pattern: /\bimport\(\s*["']([^"']+)["']\s*\)/g, kind: 'dynamic_import' },
]

function lineNumberAtIndex(content: string, index: number): number {
  let line = 1
  const upper = Math.min(index, content.length)
  for (let cursor = 0; cursor < upper; cursor += 1) {
    if (content[cursor] === '\n') line += 1
  }
  return line
}

function collectDependencyMatches(content: string): DependencyMatch[] {
  const matches: DependencyMatch[] = []
  const push = (moduleName: string, kind: string, line: number) => {
    const normalized = moduleName.trim()
    if (!normalized) return
    matches.push({ module: normalized, kind, line, scope: classifyDependency(normalized) })
  }

  for (const { pattern, kind } of WHOLE_FILE_DEPENDENCY_PATTERNS) {
    for (const match of content.matchAll(pattern)) {
      if (match[1]) {
        push(match[1], kind, lineNumberAtIndex(content, match.index ?? 0))
      }
    }
  }

  // Python `from pkg import (\n a,\n b,\n)` keeps the module on the `from` line,
  // so line-oriented matching still captures the dependency. Go/Rust remain
  // line-oriented too.
  const lines = content.split(/\r?\n/)
  let goImportBlock = false
  lines.forEach((line, index) => {
    const lineNumber = index + 1

    if (/^\s*import\s*\(\s*$/.test(line)) {
      goImportBlock = true
      return
    }
    if (goImportBlock && /^\s*\)\s*$/.test(line)) {
      goImportBlock = false
      return
    }

    const pythonImport = line.match(/^\s*import\s+([A-Za-z0-9_.,\s]+)/)
    if (pythonImport?.[1]) {
      pythonImport[1]
        .split(',')
        .map((item) => item.trim().split(/\s+as\s+/i)[0]?.trim() ?? '')
        .filter(Boolean)
        .forEach((item) => push(item, 'python_import', lineNumber))
    }
    const pythonFrom = line.match(/^\s*from\s+([A-Za-z0-9_.]+)\s+import\b/)
    if (pythonFrom?.[1]) {
      push(pythonFrom[1], 'python_from', lineNumber)
    }

    const goSingle = line.match(/^\s*import\s+"([^"]+)"/)
    if (goSingle?.[1]) {
      push(goSingle[1], 'go_import', lineNumber)
    }
    if (goImportBlock) {
      const goBlock = line.match(/^\s*"([^"]+)"\s*$/)
      if (goBlock?.[1]) {
        push(goBlock[1], 'go_import', lineNumber)
      }
    }

    const rustUse = line.match(/^\s*use\s+([^;]+);/)
    if (rustUse?.[1]) {
      const root = rustUse[1].trim().split('::')[0]?.trim()
      if (root) {
        push(root, 'rust_use', lineNumber)
      }
    }
  })

  return matches
}

async function fallbackCodeSymbols(options: {
  cwd: string
  pattern: string
  glob?: string
  limit: number
  signal?: AbortSignal
}): Promise<string> {
  const matcher = new RegExp(options.pattern)
  const files = await collectFallbackFiles({
    root: options.cwd,
    glob: options.glob,
    signal: options.signal,
    abortMessage: 'Code symbol search aborted',
  })
  const matches: string[] = []
  for (const file of files) {
    throwIfAborted(options.signal, 'Code symbol search aborted')
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
    const lines = text.split(/\r?\n/)
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index]!
      const match = matcher.exec(line)
      if (!match) {
        continue
      }
      matches.push(`${rel}:${index + 1}:${match.index + 1}:${line}`)
      count += 1
      if (count >= options.limit) {
        break
      }
    }
  }
  return matches.join('\n')
}

export function createCodeDiagnosticsTool(
  deps: CodeDiagnosticsToolDeps,
): ToolDefinitionRuntime {
  return {
    name: 'code.diagnostics',
    description: 'Return the latest LSP diagnostics for a file URI.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'lsp' },
    inputSchema: {
      type: 'object',
      properties: {
        uri: { type: 'string', description: 'file:// URI' },
        language: { type: 'string', description: 'Language id such as typescript, python, rust, or go.' },
      },
      required: ['uri', 'language'],
    },
    async execute(input): Promise<ToolResult> {
      const start = Date.now()
      const uri = typeof input.uri === 'string' ? input.uri : ''
      const language = typeof input.language === 'string' ? input.language : ''
      if (!uri || !language) {
        return {
          output: 'uri and language are required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }

      const client = deps.getClient(language)
      if (!client) {
        return {
          output: `no language server available for '${language}'`,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }

      // Open the document with the server so it publishes diagnostics for
      // it — without this, a freshly-started server has nothing to report
      // and diagnostics were a permanent dead path. Best-effort: a URI
      // that isn't a readable file simply falls through to whatever the
      // collector already holds.
      if (typeof client.ensureOpen === 'function') {
        const path = fileUriToPath(uri)
        if (path) {
          const read = deps.readFile ?? ((p: string) => readFile(p, 'utf-8'))
          const text = await read(path).catch(() => null)
          if (text !== null && client.ensureOpen(uri, language, text)) {
            const settleMs = deps.settleMs ?? DEFAULT_DIAGNOSTICS_SETTLE_MS
            if (settleMs > 0) {
              await new Promise((resolve) => setTimeout(resolve, settleMs))
            }
          }
        }
      }

      const diagnostics = client.diagnostics.get(uri)
      const raw = diagnostics.length === 0
        ? 'no diagnostics'
        : diagnostics.map(formatDiagnostic).join('\n')
      const limited = enforceOutputLimit(raw, {
        toolName: 'code.diagnostics',
        resumeHint: 'open the file in the editor or narrow the URI',
      })
      return {
        output: limited.output,
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }
}

export interface CodeSymbolsToolDeps {
  /**
   * Optional language-server accessor. When a `language` is supplied and a
   * server is available, `workspace/symbol` results are merged ahead of the
   * ripgrep matches (precise declarations, labeled [lsp]). Any failure or
   * missing server falls back to the ripgrep-only behavior.
   */
  getClient?: (language: string) => LspClient | null
}

const SYMBOL_KIND_NAMES: Record<number, string> = {
  5: 'class', 6: 'method', 8: 'field', 9: 'constructor', 10: 'enum',
  11: 'interface', 12: 'function', 13: 'variable', 14: 'constant', 23: 'struct',
}

async function lspWorkspaceSymbolBlock(
  deps: CodeSymbolsToolDeps,
  language: string,
  symbol: string,
): Promise<string | null> {
  const client = deps.getClient?.(language)
  if (!client || typeof client.workspaceSymbol !== 'function') return null
  try {
    const results = await client.workspaceSymbol(symbol)
    const lines = results
      .filter((s) => s.name === symbol || s.name.includes(symbol))
      .slice(0, 20)
      .map((s) => {
        const loc = s.location
        const kind = SYMBOL_KIND_NAMES[s.kind] ? `${SYMBOL_KIND_NAMES[s.kind]} ` : ''
        return `[lsp] ${kind}${s.name} ${loc.uri}:${loc.range.start.line + 1}:${loc.range.start.character + 1}`
      })
    return lines.length > 0 ? lines.join('\n') : null
  } catch {
    return null
  }
}

export function createCodeSymbolsTool(deps: CodeSymbolsToolDeps = {}): ToolDefinitionRuntime {
  return {
    name: 'code.symbols',
    description: 'Search for symbol declarations or references across the workspace. Pass `language` to use precise LSP workspace/symbol results when a server is available.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'filesystem' },
    inputSchema: {
      type: 'object',
      properties: {
        cwd: { type: 'string', description: 'Directory to search. Supports ~/ paths. Defaults to the active session cwd.' },
        symbol: { type: 'string', description: 'Symbol name to search for.' },
        glob: { type: 'string', description: 'Optional glob filter such as "*.ts".' },
        declarationOnly: { type: 'boolean', description: 'Prefer declaration-style matches.' },
        language: { type: 'string', description: 'Language id (typescript, python, ...) to use the LSP server for precise results.' },
        limit: { type: 'number', description: 'Maximum number of matches to return. Defaults to 50.' },
      },
      required: ['symbol'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const cwd = resolveToolCwd(input.cwd, context?.cwd)
      const symbol = typeof input.symbol === 'string' ? input.symbol.trim() : ''
      if (!symbol) {
        return {
          output: 'symbol is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }

      const escaped = escapeRegExp(symbol)
      const pattern = input.declarationOnly === true
        ? `\\b(?:function|class|interface|type|enum|const|let|var|def)\\s+${escaped}\\b`
        : `\\b${escaped}\\b`
      const limit = typeof input.limit === 'number' && Number.isFinite(input.limit)
        ? Math.max(1, Math.min(500, Math.trunc(input.limit)))
        : 50
      const args = [
        // Ignore RIPGREP_CONFIG_PATH so a user-level --follow cannot escape
        // a strict workspace through a symlink during fallback symbol search.
        '--no-config',
        '--line-number',
        '--column',
        '--color',
        'never',
        '--max-count',
        String(limit),
      ]
      if (typeof input.glob === 'string' && input.glob.trim()) {
        args.push('-g', input.glob)
      }
      args.push(pattern, '.')

      try {
        throwIfAborted(context?.signal, 'Code symbol search aborted')
        const cwdKind = await classifyToolCwd(cwd)
        if (cwdKind !== 'directory') {
          const rejection = rejectToolCwd(cwd, cwdKind, { scopeParameter: 'glob' })
          return {
            output: rejection.output,
            status: 'error',
            durationMs: Date.now() - start,
            code: rejection.code,
          }
        }
        // Stream rg with a byte cap instead of a 10MB execFile maxBuffer that
        // throws (losing every match) on huge result sets.
        const maxBytes = resolveToolOutputMaxBytes('code.symbols')
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
          stdout = await fallbackCodeSymbols({
            cwd,
            pattern,
            glob: typeof input.glob === 'string' ? input.glob : undefined,
            limit,
            signal: context?.signal,
          })
        }
        const language = typeof input.language === 'string' ? input.language.trim() : ''
        const lspBlock = language
          ? await lspWorkspaceSymbolBlock(deps, language, symbol)
          : null
        const ripgrepBlock = stdout.trim()
        const streamHint = streamTruncated
          ? `\n[code.symbols: results truncated at ${maxBytes} bytes; tighten the symbol, add a \`glob\`, or lower \`limit\` to see the rest.]`
          : ''
        const merged = (lspBlock
          ? `${lspBlock}${ripgrepBlock ? `\n${ripgrepBlock}` : ''}`
          : ripgrepBlock || '[no matches]') + streamHint
        const limited = enforceOutputLimit(merged, {
          toolName: 'code.symbols',
          resumeHint: 'tighten the symbol, add a `glob`, or lower `limit`',
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

export function createCodeDependenciesTool(): ToolDefinitionRuntime {
  return {
    name: 'code.dependencies',
    description: 'Extract import and dependency references from a source file.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'filesystem' },
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Source file path.' },
      },
      required: ['path'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      // Resolve relative paths against the active session cwd, not the daemon's
      // process.cwd(), so `code.dependencies { path: 'src/x.ts' }` reads the
      // file the agent means.
      const path = typeof input.path === 'string'
        ? resolveToolPath(input.path.trim(), context?.cwd)
        : ''
      if (!path) {
        return {
          output: 'path is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }

      try {
        const content = await readFile(path, 'utf-8')
        const matches = collectDependencyMatches(content)
        const externalModules = [...new Set(matches
          .filter((match) => match.scope === 'external')
          .map((match) => match.module))]
        const localModules = [...new Set(matches
          .filter((match) => match.scope === 'local')
          .map((match) => match.module))]

        const raw = JSON.stringify({
          path,
          totalMatches: matches.length,
          externalModules,
          localModules,
          dependencies: matches,
        }, null, 2)
        const limited = enforceOutputLimit(raw, {
          toolName: 'code.dependencies',
          resumeHint: 'analyze a smaller file or split the source',
        })
        return {
          output: limited.output,
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (error) {
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
