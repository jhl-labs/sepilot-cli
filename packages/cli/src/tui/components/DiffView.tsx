import { Box, Text } from 'ink'
import { colors } from '../theme.js'
import {
  buildDiffLines,
  truncatePreviewItems,
} from '../utils/tooling.js'
import { terminalFileLink } from '../utils/file-link.js'

interface DiffViewProps {
  path?: string
  previousContent: string | null | undefined
  nextContent: string
  maxLines?: number
}

interface UnifiedDiffViewProps {
  diff: string
  maxLines?: number
}

export function UnifiedDiffView({ diff, maxLines }: UnifiedDiffViewProps) {
  const lines = diff.endsWith('\n') ? diff.slice(0, -1).split('\n') : diff.split('\n')
  const visibleDiff = maxLines == null
    ? { visible: lines, omitted: 0 }
    : truncatePreviewItems(lines, maxLines)

  return (
    <Box flexDirection="column">
      {visibleDiff.visible.map((line, index) => (
        <Text
          key={`${index}-${line.slice(0, 24)}`}
          color={
            line.startsWith('+') && !line.startsWith('+++')
              ? colors.success
              : line.startsWith('-') && !line.startsWith('---')
                ? colors.error
                : line.startsWith('@@')
                  ? colors.info
                  : colors.dimText
          }
          wrap="truncate-end"
        >
          {line || ' '}
        </Text>
      ))}
      {visibleDiff.omitted > 0 && (
        <Text color={colors.dimText} wrap="truncate-end">
          … {visibleDiff.omitted} more diff line{visibleDiff.omitted === 1 ? '' : 's'}
        </Text>
      )}
    </Box>
  )
}

export function DiffView({
  path,
  previousContent,
  nextContent,
  maxLines,
}: DiffViewProps) {
  const lines = buildDiffLines(previousContent, nextContent)
  const visibleDiff = maxLines == null
    ? { visible: lines, omitted: 0 }
    : truncatePreviewItems(lines, maxLines)

  return (
    <Box flexDirection="column">
      {path && (
        <Text wrap="truncate-end">
          <Text color={colors.info} underline>file {terminalFileLink(path)}</Text>
          <Text color={colors.dimText}> · Ctrl+O full file</Text>
        </Text>
      )}
      {previousContent === undefined && (
        <Text color={colors.dimText} wrap="truncate-end">
          Previous file contents unavailable. Showing requested write payload.
        </Text>
      )}
      {lines.length === 0 && (
        <Text color={colors.dimText} wrap="truncate-end">No textual changes detected.</Text>
      )}
      {visibleDiff.visible.map((line, index) => (
        <Text
          key={`${line.prefix}-${index}`}
          color={
            line.prefix === '+'
              ? colors.success
              : line.prefix === '-'
                ? colors.error
                : colors.dimText
          }
          wrap="truncate-end"
        >
          {line.prefix} {line.value || ' '}
        </Text>
      ))}
      {visibleDiff.omitted > 0 && (
        <Text color={colors.dimText} wrap="truncate-end">
          ... {visibleDiff.omitted} more diff line{visibleDiff.omitted === 1 ? '' : 's'}
        </Text>
      )}
    </Box>
  )
}
