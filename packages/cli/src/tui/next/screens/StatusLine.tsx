import { Box, Text } from 'ink'
import stringWidth from 'string-width'
import { colors } from '../../theme.js'

export interface StatusLineProps {
  model: string
  mode: string
  autonomy: string
  contextPercent: number | null
  contextTokens?: number | null
  contextWindow?: number | null
  contextEstimated?: boolean
  costLabel: string | null
  progressLabel?: string | null
  attention?: boolean
  streamStartedAt?: number | null
  error: string | null
  width: number
}

export function StatusLine({
  model,
  mode,
  autonomy,
  contextPercent,
  contextTokens = null,
  contextWindow = null,
  contextEstimated = false,
  costLabel,
  progressLabel = null,
  attention = false,
  streamStartedAt = null,
  error,
  width,
}: StatusLineProps) {
  const contextLabel = formatContextLabel(
    contextTokens,
    contextWindow,
    contextPercent,
    contextEstimated,
  )
  const compactContextLabel = formatContextLabel(
    contextTokens,
    contextWindow,
    contextPercent,
    contextEstimated,
    true,
  )
  // Active runs already rerender from the shared activity clock. Reading the
  // wall clock here keeps elapsed time current without starting a competing
  // interval that causes another full Ink redraw.
  const elapsedLabel = streamStartedAt == null ? null : formatElapsed(Date.now() - streamStartedAt)
  const detailedParts = attention
    ? [progressLabel, model, `mode ${mode}`, `access ${autonomy}`, contextLabel, costLabel]
    : [model, `mode ${mode}`, `access ${autonomy}`, contextLabel, elapsedLabel ? `elapsed ${elapsedLabel}` : null, costLabel]
  const compactParts = attention
    ? [progressLabel, compactModelLabel(model), mode, autonomy, compactContextLabel]
    : [compactModelLabel(model), mode, autonomy, compactContextLabel, elapsedLabel, costLabel]
  const detailedStatus = joinStatusParts(detailedParts)
  const fittedStatus = stringWidth(detailedStatus) <= width
    ? detailedStatus
    : joinStatusParts(compactParts)
  const status = error
    ? error
    : fittedStatus

  return (
    <Box width={width}>
      <Text
        color={error ? colors.error : attention ? colors.warning : colors.dimText}
        bold={attention && !error}
      >
        {truncateToWidth(status, width)}
      </Text>
    </Box>
  )
}

function formatContextLabel(
  contextTokens: number | null,
  contextWindow: number | null,
  contextPercent: number | null,
  estimated: boolean,
  compact = false,
): string | null {
  const prefix = compact ? 'ctx' : 'context'
  if (contextTokens != null && contextWindow != null && contextWindow > 0) {
    const percent = contextPercent == null
      ? Math.round((contextTokens / contextWindow) * 100)
      : contextPercent
    return compact
      ? `${prefix} ${estimated ? '~' : ''}${formatTokens(contextTokens)}/${formatTokens(contextWindow)} ${percent}%`
      : `${prefix} ${estimated ? '~' : ''}${formatTokens(contextTokens)}/${formatTokens(contextWindow)} (${percent}%)`
  }
  if (contextPercent != null) return `${prefix} ${estimated ? '~' : ''}${contextPercent}%`
  if (contextTokens != null) return `${prefix} ${estimated ? '~' : ''}${formatTokens(contextTokens)}`
  if (contextWindow != null && contextWindow > 0) return `${prefix} ?/${formatTokens(contextWindow)}`
  return null
}

function compactModelLabel(model: string): string {
  const separator = model.lastIndexOf('/')
  return separator >= 0 ? model.slice(separator + 1) : model
}

function joinStatusParts(parts: Array<string | null>): string {
  return parts.filter((part): part is string => Boolean(part)).join(' · ')
}

function formatTokens(tokens: number): string {
  if (!Number.isFinite(tokens) || tokens < 0) return '0'
  if (tokens >= 1_000_000) {
    const value = tokens / 1_000_000
    return `${Number.isInteger(value) ? value : value.toFixed(1)}M`
  }
  if (tokens >= 1_000) {
    const value = tokens / 1_000
    return `${Number.isInteger(value) ? value : value.toFixed(1)}k`
  }
  return String(Math.round(tokens))
}

function formatElapsed(elapsedMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1_000))
  if (totalSeconds < 60) return `${totalSeconds}s`
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes < 60) return `${minutes}m${String(seconds).padStart(2, '0')}s`
  const hours = Math.floor(minutes / 60)
  return `${hours}h${String(minutes % 60).padStart(2, '0')}m`
}

function truncateToWidth(value: string, width: number): string {
  const available = Math.max(0, width)
  if (stringWidth(value) <= available) return value
  if (available === 0) return ''
  if (available === 1) return '…'

  let result = ''
  for (const character of value) {
    if (stringWidth(result) + stringWidth(character) + 1 > available) break
    result += character
  }
  return `${result}…`
}
