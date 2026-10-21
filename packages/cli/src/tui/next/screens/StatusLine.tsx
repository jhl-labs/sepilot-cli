import { Box, Text } from 'ink'
import stringWidth from 'string-width'
import { colors } from '../../theme.js'
import { privilegeColor } from '../privilege.js'

export interface StatusLineProps {
  model: string
  mode: string
  bot?: string
  engineOverride?: boolean
  tokensPerSecond?: number | null
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
  bot,
  engineOverride,
  tokensPerSecond,
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
  const elapsedLabel = streamStartedAt == null ? null : formatElapsed(Date.now() - streamStartedAt)
  const tpsLabel = tokensPerSecond && tokensPerSecond > 0 ? `${tokensPerSecond} tps` : null
  const contextColor = contextPercent == null ? colors.dimText
    : contextPercent >= 85 ? colors.error
    : contextPercent >= 60 ? colors.warning
    : colors.success
  const part = (text: string | null, color: string, bold = false): StatusPart | null => text ? { text, color, bold } : null
  const modeParts = bot === undefined
    ? [part(`mode ${mode}`, colors.info)]
    : engineOverride
      ? [part(`bot ${bot}`, colors.info), part(`engine ${mode}`, colors.warning)]
      : [part(`bot ${bot}`, colors.info)]
  const compactModeParts = bot === undefined
    ? [part(mode, colors.info)]
    : engineOverride ? [part(bot, colors.info), part(mode, colors.warning)] : [part(bot, colors.info)]
  const progressPart = part(progressLabel, colors.warning, true)
  const detailedParts = attention
    ? [progressPart, part(model, colors.primary), ...modeParts, part(`privilege ${autonomy}`, privilegeColor(autonomy), autonomy === 'autonomous'), part(contextLabel, contextColor), part(tpsLabel, colors.success), part(costLabel, colors.dimText)]
    : [part(model, colors.primary), ...modeParts, part(`privilege ${autonomy}`, privilegeColor(autonomy), autonomy === 'autonomous'), part(contextLabel, contextColor), part(tpsLabel, colors.success), part(elapsedLabel ? `elapsed ${elapsedLabel}` : null, colors.dimText), part(costLabel, colors.dimText)]
  const compactParts = attention
    ? [progressPart, part(compactModelLabel(model), colors.primary), ...compactModeParts, part(autonomy, privilegeColor(autonomy), autonomy === 'autonomous'), part(compactContextLabel, contextColor), part(tpsLabel, colors.success)]
    : [part(compactModelLabel(model), colors.primary), ...compactModeParts, part(autonomy, privilegeColor(autonomy), autonomy === 'autonomous'), part(compactContextLabel, contextColor), part(tpsLabel, colors.success), part(elapsedLabel, colors.dimText), part(costLabel, colors.dimText)]
  const detailed = detailedParts.filter((entry): entry is StatusPart => entry !== null)
  const compact = compactParts.filter((entry): entry is StatusPart => entry !== null)
  const fitted = stringWidth(joinStatusParts(detailed.map((entry) => entry.text))) <= width ? detailed : compact
  const fittedText = joinStatusParts(fitted.map((entry) => entry.text))

  if (error || stringWidth(fittedText) > width) {
    return (
      <Box width={width}>
        <Text color={error ? colors.error : colors.dimText}>
          {truncateToWidth(error ?? fittedText, width)}
        </Text>
      </Box>
    )
  }
  return (
    <Box width={width}>
      <Text>
        {fitted.map((entry, index) => (
          <Text key={index}>
            {index > 0 ? <Text color={colors.border}>{' · '}</Text> : null}
            <Text color={entry.color} bold={entry.bold}>{entry.text}</Text>
          </Text>
        ))}
      </Text>
    </Box>
  )
}

interface StatusPart {
  text: string
  color: string
  bold: boolean
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
