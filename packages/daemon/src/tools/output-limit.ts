/**
 * Tool output size enforcement helper.
 *
 * Tools whose output is forwarded verbatim to the LLM (terminal.run,
 * web-fetch, fs.search, code.analysis) can produce arbitrarily large
 * payloads — large enough to blow out the context window of the next
 * LLM turn. `enforceOutputLimit` truncates UTF-8 byte-aware to a cap
 * and appends a structured hint so the agent knows the result was
 * elided and how to re-run with a narrower scope.
 *
 * The fallback cap defaults to 100KB and can be overridden globally via
 * the `SEPILOTD_TOOL_OUTPUT_MAX_BYTES` environment variable. Noisy tools
 * get narrower built-in defaults when no override is set; operators can
 * override one tool with `SEPILOTD_TOOL_OUTPUT_MAX_BYTES_<TOOL_NAME>`,
 * where non-alphanumeric characters become underscores
 * (e.g. `SEPILOTD_TOOL_OUTPUT_MAX_BYTES_FS_SEARCH`).
 */

const DEFAULT_MAX_BYTES = 100_000
const TOOL_DEFAULT_MAX_BYTES: Record<string, number> = {
  webfetch: 40_000,
  'web-fetch': 40_000,
  'fs.search': 80_000,
  'code.symbols': 80_000,
  'code.dependencies': 80_000,
  'code.diagnostics': 60_000,
  'lsp.diagnostics': 60_000,
  'git.diff': 60_000,
  'git.log': 60_000,
  'git.status': 60_000,
  'terminal.run': 40_000,
}

export interface OutputLimitOptions {
  /** Hard cap in bytes. Falls back to {@link resolveToolOutputMaxBytes}(). */
  maxBytes?: number
  /** Tool name used in the truncation hint (e.g. 'terminal.run'). */
  toolName?: string
  /**
   * Tool-specific suggestion for narrowing output (e.g.
   * "pipe through head/tail/grep, or run with a more specific command").
   * Falls back to a generic phrase when omitted.
   */
  resumeHint?: string
}

export interface EnforcedOutput {
  /** Possibly-truncated output ready to emit. */
  output: string
  /** True when the input exceeded `limitBytes`. */
  truncated: boolean
  /** Byte length of the original input. */
  originalBytes: number
  /** Byte cap that was applied. */
  limitBytes: number
}

function parsePositiveByteCount(value: string | undefined): number | null {
  if (!value) return null
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0) return null
  return Math.floor(parsed)
}

/**
 * Resolve the default tool output cap.
 *
 * - Reads `SEPILOTD_TOOL_OUTPUT_MAX_BYTES` (positive integer) when set.
 * - Falls back to 100KB when the env var is unset, non-numeric, or <= 0.
 */
export function resolveDefaultMaxBytes(): number {
  return parsePositiveByteCount(process.env.SEPILOTD_TOOL_OUTPUT_MAX_BYTES) ?? DEFAULT_MAX_BYTES
}

function outputLimitEnvName(toolName: string): string {
  const suffix = toolName
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
  return `SEPILOTD_TOOL_OUTPUT_MAX_BYTES_${suffix}`
}

/**
 * Resolve the effective cap for a tool. Precedence:
 * explicit `maxBytes` in the caller > tool-specific env > global env >
 * built-in per-tool default > generic default.
 */
export function resolveToolOutputMaxBytes(toolName?: string): number {
  const normalizedToolName = toolName?.trim()
  if (normalizedToolName) {
    const specific = parsePositiveByteCount(process.env[outputLimitEnvName(normalizedToolName)])
    if (specific !== null) return specific
  }

  const global = parsePositiveByteCount(process.env.SEPILOTD_TOOL_OUTPUT_MAX_BYTES)
  if (global !== null) return global

  if (normalizedToolName) {
    return TOOL_DEFAULT_MAX_BYTES[normalizedToolName] ?? DEFAULT_MAX_BYTES
  }

  return DEFAULT_MAX_BYTES
}

/**
 * Truncate `raw` to `options.maxBytes` UTF-8 bytes (or the tool's
 * effective default) and append a one-line truncation hint so the agent can
 * decide how to re-run with a narrower scope.
 *
 * The byte slice happens on the UTF-8 buffer; `Buffer.toString('utf8')`
 * cleans up any partial multibyte char at the boundary.
 */
export function enforceOutputLimit(raw: string, options: OutputLimitOptions = {}): EnforcedOutput {
  const maxBytes = options.maxBytes ?? resolveToolOutputMaxBytes(options.toolName)
  const buffer = Buffer.from(raw, 'utf8')
  const originalBytes = buffer.byteLength

  if (originalBytes <= maxBytes) {
    return {
      output: raw,
      truncated: false,
      originalBytes,
      limitBytes: maxBytes,
    }
  }

  const sliced = buffer.subarray(0, maxBytes).toString('utf8')
  const tool = options.toolName ?? 'tool'
  const hint = options.resumeHint ?? `narrow ${tool} output (filter/offset/limit) and re-run`
  const tail = `\n\n[${tool}: output truncated at ${maxBytes} bytes; original was ${originalBytes} bytes — ${hint}]`

  return {
    output: sliced + tail,
    truncated: true,
    originalBytes,
    limitBytes: maxBytes,
  }
}
