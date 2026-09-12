import { readFile, stat } from 'node:fs/promises'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'
import { resolveToolPath } from './path-utils.js'

// Upper bound on a notebook file read. `.ipynb` is JSON, so an oversized file
// (e.g. a notebook with megabytes of embedded base64 image outputs, or a
// mislabeled multi-GB file) would be read fully into memory and then
// JSON.parsed — a daemon OOM. Stat the file first and refuse before reading.
const MAX_NOTEBOOK_SIZE = 25 * 1024 * 1024 // 25MB

function toText(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }
  if (Array.isArray(value)) {
    return value
      .map((entry) => typeof entry === 'string' ? entry : '')
      .join('')
  }
  return ''
}

function truncateText(value: string, max: number): string {
  return value.length > max
    ? `${value.slice(0, max - 1)}…`
    : value
}

export function createNotebookInspectTool(): ToolDefinitionRuntime {
  return {
    name: 'notebook.inspect',
    description: 'Inspect a Jupyter notebook and summarize its cells, metadata, and outputs.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'filesystem' },
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Notebook path, usually ending with .ipynb.' },
        includeSources: { type: 'boolean', description: 'Include truncated cell source text in the output.' },
        cellLimit: { type: 'number', description: 'Maximum number of cells to include. Defaults to 20.' },
        maxCharsPerCell: { type: 'number', description: 'Maximum number of source characters per cell. Defaults to 400.' },
      },
      required: ['path'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const path = typeof input.path === 'string' && input.path.trim()
        ? resolveToolPath(input.path.trim(), context?.cwd)
        : ''
      if (!path) {
        return {
          output: 'path is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }

      const includeSources = input.includeSources !== false
      const cellLimit = typeof input.cellLimit === 'number' && Number.isFinite(input.cellLimit)
        ? Math.max(1, Math.min(200, Math.trunc(input.cellLimit)))
        : 20
      const maxCharsPerCell = typeof input.maxCharsPerCell === 'number' && Number.isFinite(input.maxCharsPerCell)
        ? Math.max(40, Math.min(10_000, Math.trunc(input.maxCharsPerCell)))
        : 400

      try {
        // Pre-read size guard: refuse an oversized notebook before it is read
        // into memory and parsed.
        try {
          const info = await stat(path)
          if (info.isFile() && info.size > MAX_NOTEBOOK_SIZE) {
            return {
              output: `Notebook too large to inspect: ${info.size} bytes (max ${MAX_NOTEBOOK_SIZE}).`,
              status: 'error',
              code: 'FILE_TOO_LARGE_PERMANENT',
              durationMs: Date.now() - start,
            }
          }
        } catch {
          // Let the downstream read surface a proper ENOENT/permission error.
        }
        const raw = JSON.parse(await readFile(path, 'utf-8')) as {
          metadata?: Record<string, unknown>
          nbformat?: number
          nbformat_minor?: number
          cells?: Array<Record<string, unknown>>
        }
        const cells = Array.isArray(raw.cells) ? raw.cells : []
        const codeCells = cells.filter((cell) => cell.cell_type === 'code')
        const markdownCells = cells.filter((cell) => cell.cell_type === 'markdown')
        const rawCells = cells.filter((cell) => cell.cell_type === 'raw')

        const summarizedCells = cells.slice(0, cellLimit).map((cell, index) => {
          const source = toText(cell.source)
          const outputs = Array.isArray(cell.outputs) ? cell.outputs : []
          const outputPreview = outputs
            .map((output) => {
              if (typeof output?.text === 'string') {
                return output.text
              }
              if (Array.isArray(output?.text)) {
                return output.text.join('')
              }
              if (typeof output?.data === 'object' && output.data !== null) {
                const textPlain = (output.data as Record<string, unknown>)['text/plain']
                return toText(textPlain)
              }
              return ''
            })
            .filter(Boolean)
            .join('\n')

          return {
            index,
            type: typeof cell.cell_type === 'string' ? cell.cell_type : 'unknown',
            executionCount: typeof cell.execution_count === 'number'
              ? cell.execution_count
              : null,
            sourceLength: source.length,
            outputCount: outputs.length,
            sourcePreview: includeSources
              ? truncateText(source.trim(), maxCharsPerCell)
              : undefined,
            outputPreview: outputPreview
              ? truncateText(outputPreview.trim(), maxCharsPerCell)
              : undefined,
          }
        })

        return {
          output: JSON.stringify({
            path,
            nbformat: raw.nbformat ?? null,
            nbformatMinor: raw.nbformat_minor ?? null,
            kernelspec: raw.metadata?.kernelspec ?? null,
            languageInfo: raw.metadata?.language_info ?? null,
            cellCount: cells.length,
            codeCellCount: codeCells.length,
            markdownCellCount: markdownCells.length,
            rawCellCount: rawCells.length,
            cells: summarizedCells,
          }, null, 2),
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
