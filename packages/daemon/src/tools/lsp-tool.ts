import type {
  LspClient,
  LspDocumentSymbol,
  LspLocation,
} from '../lsp/client.js'
import type { LspDiagnostic } from '../lsp/diagnostics.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

export interface LspToolDeps {
  getClient: (language: string) => LspClient | null
}

function formatDiagnostic(d: LspDiagnostic): string {
  const severity = d.severity ? `[${d.severity}] ` : ''
  const source = d.source ? ` (${d.source})` : ''
  return `L${d.line}: ${severity}${d.message}${source}`
}

function formatLocation(loc: LspLocation): string {
  return `${loc.uri}:${loc.range.start.line + 1}:${loc.range.start.character + 1}`
}

/** Flatten an LSP Hover `contents` (string | MarkedString | MarkupContent | array). */
function formatHoverContents(contents: unknown): string {
  if (contents == null) return ''
  if (typeof contents === 'string') return contents
  if (Array.isArray(contents)) return contents.map(formatHoverContents).filter(Boolean).join('\n')
  if (typeof contents === 'object') {
    const obj = contents as { value?: unknown; language?: unknown }
    if (typeof obj.value === 'string') return obj.value
  }
  return ''
}

/**
 * Pick a small set of symbol positions worth running `references` on.
 * Strategy: ask documentSymbol for top-level declarations, prefer
 * exported / file-level kinds (function, class, interface, …), cap at 3
 * to bound the round-trip count.
 */
async function collectReferencePositions(
  client: LspClient,
  uri: string,
): Promise<Array<{ line: number; character: number }>> {
  let symbols: LspDocumentSymbol[] = []
  try {
    symbols = (await client.documentSymbol(uri)) ?? []
  } catch {
    return []
  }
  // Prefer kinds that typically appear in import sites: function (12),
  // class (5), interface (11), enum (10), variable (13). We stay
  // shallow — top-level only — and skip property/field/parameter kinds.
  const interestingKinds = new Set([5, 6, 10, 11, 12, 13])
  const picked: Array<{ line: number; character: number }> = []
  for (const sym of symbols) {
    if (picked.length >= 3) break
    if (!interestingKinds.has(sym.kind)) continue
    const pos = sym.selectionRange?.start ?? sym.range?.start
    if (!pos) continue
    picked.push({ line: pos.line, character: pos.character })
  }
  return picked
}

export function createLspTool(deps: LspToolDeps): ToolDefinitionRuntime {
  return {
    name: 'lsp',
    description:
      'Query a language server for a file URI. action="diagnostics" (default) '
      + 'returns the latest diagnostics; action="references" reports symbol callers; '
      + 'action="definition"/"hover" need a 0-based line+character and report the '
      + 'declaration site(s) / type info for that position. Use language identifiers '
      + 'like "typescript", "python", "rust", "go".',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'lsp' },
    inputSchema: {
      type: 'object',
      properties: {
        uri: { type: 'string', description: 'file:// URI' },
        language: { type: 'string', description: 'typescript | python | rust | go' },
        action: {
          type: 'string',
          enum: ['diagnostics', 'references', 'definition', 'hover'],
          description: 'Query kind. Defaults to "diagnostics" for back-compat. definition/hover require line+character.',
        },
        line: { type: 'number', description: '0-based line for definition/hover.' },
        character: { type: 'number', description: '0-based column for definition/hover.' },
      },
      required: ['uri', 'language'],
    },
    async execute(input): Promise<ToolResult> {
      const start = Date.now()
      const uri = typeof input.uri === 'string' ? input.uri : ''
      const language = typeof input.language === 'string' ? input.language : ''
      const action =
        input.action === 'references' || input.action === 'definition' || input.action === 'hover'
          ? input.action
          : 'diagnostics'
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
      if (action === 'definition' || action === 'hover') {
        const line = typeof input.line === 'number' ? input.line : null
        const character = typeof input.character === 'number' ? input.character : null
        if (line === null || character === null) {
          return {
            output: `action="${action}" requires line and character (0-based position)`,
            status: 'error',
            durationMs: Date.now() - start,
          }
        }
        try {
          if (action === 'definition') {
            const locs = await client.definition(uri, line, character)
            return {
              output: locs.length === 0 ? '[no definition]' : locs.map(formatLocation).join('\n'),
              status: 'success',
              durationMs: Date.now() - start,
            }
          }
          const hover = await client.hover(uri, line, character)
          const text = formatHoverContents(hover?.contents)
          return {
            output: text.trim().length === 0 ? '[no hover info]' : text,
            status: 'success',
            durationMs: Date.now() - start,
          }
        } catch (err) {
          return {
            output: err instanceof Error ? err.message : String(err),
            status: 'error',
            durationMs: Date.now() - start,
            code: action === 'definition' ? 'LSP_DEFINITION_FAILED' : 'LSP_HOVER_FAILED',
          }
        }
      }
      if (action === 'references') {
        try {
          const positions = await collectReferencePositions(client, uri)
          if (positions.length === 0) {
            return {
              output: '[no references]',
              status: 'success',
              durationMs: Date.now() - start,
            }
          }
          const seen = new Set<string>()
          const lines: string[] = []
          for (const pos of positions) {
            const refs = await client.references(uri, pos.line, pos.character)
            for (const loc of refs) {
              const key = `${loc.uri}:${loc.range.start.line}:${loc.range.start.character}`
              if (seen.has(key)) continue
              seen.add(key)
              lines.push(formatLocation(loc))
            }
          }
          return {
            output: lines.length === 0 ? '[no references]' : lines.join('\n'),
            status: 'success',
            durationMs: Date.now() - start,
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err)
          return {
            output: message,
            status: 'error',
            durationMs: Date.now() - start,
            code: 'LSP_REFERENCES_FAILED',
          }
        }
      }
      const diags = client.diagnostics.get(uri)
      return {
        output: diags.length === 0 ? 'no diagnostics' : diags.map(formatDiagnostic).join('\n'),
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }
}
