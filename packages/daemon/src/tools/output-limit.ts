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
  maxBytes?: number
  toolName?: string
  resumeHint?: string
}

export interface EnforcedOutput {
  output: string
  truncated: boolean
  originalBytes: number
  limitBytes: number
}

function parsePositiveByteCount(value: string | undefined): number | null {
  if (!value) return null
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0) return null
  return Math.floor(parsed)
}

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
