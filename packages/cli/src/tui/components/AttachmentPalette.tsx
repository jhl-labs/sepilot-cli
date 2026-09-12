import { Box, Text } from 'ink'
import { colors } from '../theme.js'
import { calculateVisibleWindow } from '../utils/layout.js'
import type { AttachmentCandidate } from '../utils/attachments.js'

interface AttachmentPaletteProps {
  items: AttachmentCandidate[]
  selectedIndex: number
  maxVisibleItems?: number
  reindexNotice?: string | null
  width?: number
}

export function AttachmentPalette({
  items,
  selectedIndex,
  maxVisibleItems = items.length,
  reindexNotice = null,
  width = 80,
}: AttachmentPaletteProps) {
  const { start, end, clampedIndex } = calculateVisibleWindow(
    items.length,
    selectedIndex,
    maxVisibleItems,
  )
  const visibleItems = items.slice(start, end)
  const contentWidth = Math.max(20, width - 4)
  const nameWidth = Math.min(
    38,
    Math.max(18, Math.floor(contentWidth * 0.42)),
  )

  return (
    <Box
      flexDirection="column"
      borderStyle="single"
      borderColor={colors.primary}
      paddingX={1}
      marginBottom={0}
      width="100%"
      minWidth={0}
    >
      <Text color={colors.primary} bold>
        Files
      </Text>
      {reindexNotice && (
        <Text color={colors.dimText} wrap="truncate-end">{reindexNotice}</Text>
      )}
      {visibleItems.map((item, index) => {
        const absoluteIndex = start + index
        const selected = absoluteIndex === clampedIndex
        const display = item.isDirectory ? `${item.basename}/` : item.basename
        return (
          <Box key={item.path} gap={1} width="100%" minWidth={0}>
            <Box width={nameWidth} flexShrink={0} minWidth={0}>
              <Text
                color={selected ? colors.text : colors.dimText}
                backgroundColor={selected ? colors.primary : undefined}
                bold={selected}
                wrap="truncate-end"
              >
                {' '}
                {display}
                {' '}
              </Text>
            </Box>
            <Box flexGrow={1} flexShrink={1} minWidth={0}>
              {item.dir && (
                <Text color={colors.dimText} wrap="truncate-end">{item.dir}</Text>
              )}
            </Box>
          </Box>
        )
      })}
    </Box>
  )
}
