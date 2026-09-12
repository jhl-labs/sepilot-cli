const DEFAULT_MAX_LINES = 120
const DEFAULT_MAX_CHARS = 8_000

export function truncateJsonOutput(
  text: string,
  options: { maxLines?: number; maxChars?: number } = {},
): string {
  const maxLines = options.maxLines ?? DEFAULT_MAX_LINES
  const maxChars = options.maxChars ?? DEFAULT_MAX_CHARS
  let result = text
  let truncated = false
  const lines = result.split('\n')

  if (lines.length > maxLines) {
    result = lines.slice(0, maxLines).join('\n')
    truncated = true
  }

  if (result.length > maxChars) {
    result = result.slice(0, maxChars)
    truncated = true
  }

  if (!truncated) return text

  const omittedLines = Math.max(0, lines.length - maxLines)
  const lineSuffix = omittedLines > 0 ? `+${omittedLines} lines ` : ''
  return `${result}\n... (${lineSuffix}truncated - narrow the query or use the non-interactive CLI with --json)`
}

export function maskSecretValue(value: string): string {
  if (value.length < 12) return '***'
  return `${value.slice(0, 4)}...${value.slice(-4)}`
}
