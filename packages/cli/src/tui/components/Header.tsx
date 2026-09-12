import { Box, Text, useStdout } from 'ink'
import stringWidth from 'string-width'
import { colors } from '../theme.js'
import { clearLineTail } from '../utils/terminal-line.js'

interface HeaderProps {
  version: string
  model: string
  provider: string
  contextWindow?: number | null
  maxOutputTokens?: number | null
  projectName?: string | null
  width?: number
}

const DOT = '\u00B7'
const SEPARATOR = `  ${DOT}  `
const HEADER_HORIZONTAL_PADDING = 2

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

function truncateToWidth(text: string, maxWidth: number): string {
  if (maxWidth <= 0) return ''
  if (stringWidth(text) <= maxWidth) return text

  const ellipsis = '…'
  const ellipsisWidth = stringWidth(ellipsis)
  if (maxWidth <= ellipsisWidth) return ellipsis

  let output = ''
  for (const char of text) {
    if (stringWidth(output) + stringWidth(char) + ellipsisWidth > maxWidth) break
    output += char
  }
  return `${output}${ellipsis}`
}

function fitHeaderLine({
  version,
  projectName,
  model,
  provider,
  contextWindow,
  maxOutputTokens,
  width,
}: Required<Pick<HeaderProps, 'version' | 'model' | 'provider'>> &
  Pick<HeaderProps, 'projectName' | 'contextWindow' | 'maxOutputTokens'> & {
    width?: number
  }): string {
  const baseLeft = `sepilot${SEPARATOR}v${version}`
  const fullLeft = projectName ? `${baseLeft}${SEPARATOR}${projectName}` : baseLeft
  const modelProvider = model && provider ? `${model} @ ${provider}` : model || provider
  const rightParts = [
    modelProvider,
    contextWindow && contextWindow > 0 ? `ctx ${formatTokens(contextWindow)}` : null,
    maxOutputTokens && maxOutputTokens > 0 ? `out ${formatTokens(maxOutputTokens)}` : null,
  ].filter((part): part is string => Boolean(part))
  const rightCandidates = [
    rightParts.join(SEPARATOR),
    modelProvider,
    model,
    '',
  ].filter((candidate, index, candidates) => (
    candidate || index === candidates.length - 1
  ))

  if (width == null || width <= 0) {
    return rightCandidates[0]
      ? `${fullLeft} ${rightCandidates[0]}`
      : fullLeft
  }

  const contentWidth = Math.max(0, width - HEADER_HORIZONTAL_PADDING)
  if (contentWidth <= 0) return ''

  for (const right of rightCandidates) {
    const rightWidth = stringWidth(right)
    const gapWidth = right ? 1 : 0
    const leftWidth = contentWidth - rightWidth - gapWidth
    if (leftWidth < Math.min(stringWidth(baseLeft), contentWidth)) continue

    const left = truncateToWidth(fullLeft, leftWidth)
    const used = stringWidth(left) + rightWidth
    const gap = right ? ' '.repeat(Math.max(1, contentWidth - used)) : ''
    return `${left}${gap}${right}`
  }

  return truncateToWidth(baseLeft, contentWidth)
}

export function Header({
  version,
  model,
  provider,
  contextWindow = null,
  maxOutputTokens = null,
  projectName = null,
  width,
}: HeaderProps) {
  const { stdout } = useStdout()
  const headerWidth = width ?? stdout.columns
  const line = fitHeaderLine({
    version,
    model,
    provider,
    contextWindow,
    maxOutputTokens,
    projectName,
    width: headerWidth,
  })

  return (
    <Box
      width={headerWidth ?? '100%'}
      borderStyle="single"
      borderColor={colors.border}
      borderBottom
      borderTop={false}
      borderLeft={false}
      borderRight={false}
      paddingX={1}
      overflow="hidden"
    >
      <Text color={colors.muted} wrap="truncate-end">{clearLineTail(line)}</Text>
    </Box>
  )
}
