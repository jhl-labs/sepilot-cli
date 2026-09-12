export type SessionExportFormat = 'markdown' | 'json'

export interface SessionExportOptions {
  format: SessionExportFormat
  outputPath: string | null
}

const SESSION_EXPORT_FORMAT_ALIASES: Record<string, SessionExportFormat> = {
  markdown: 'markdown',
  md: 'markdown',
  json: 'json',
}

export function parseSessionExportArgs(
  args: string[],
): SessionExportOptions {
  const [firstArg, ...restArgs] = args
  const normalizedFormat = firstArg
    ? SESSION_EXPORT_FORMAT_ALIASES[firstArg.trim().toLowerCase()]
    : undefined

  if (normalizedFormat) {
    const outputPath = restArgs.join(' ').trim()
    return {
      format: normalizedFormat,
      outputPath: outputPath.length > 0 ? outputPath : null,
    }
  }

  const outputPath = args.join(' ').trim()
  return {
    format: 'markdown',
    outputPath: outputPath.length > 0 ? outputPath : null,
  }
}

export function buildSessionExportFilename(
  sessionId: string,
  format: SessionExportFormat,
): string {
  // Strip an existing `session-` prefix before slicing so an id like
  // `session-controlled` doesn't produce `session-session-.{md,json}`
  // — the literal pattern that left stray fixture files in the
  // working tree during prior test runs.
  const trimmed = sessionId.replace(/^session-/, '')
  const safe = trimmed.replace(/[^A-Za-z0-9_-]/g, '-').slice(0, 8) || 'unknown'
  return `session-${safe}.${format === 'json' ? 'json' : 'md'}`
}
