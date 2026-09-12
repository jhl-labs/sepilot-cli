import { Box, Text } from 'ink'
import { colors } from '../theme.js'
import type { CommandPaletteItem } from '../utils/command-autocomplete.js'
import { calculateVisibleWindow } from '../utils/layout.js'

interface CommandPaletteProps {
  selectedIndex: number
  items: CommandPaletteItem[]
  maxVisibleItems?: number
  width?: number
}

export function CommandPalette({
  selectedIndex,
  items,
  maxVisibleItems = items.length,
  width = 80,
}: CommandPaletteProps) {
  const { start, end, clampedIndex } = calculateVisibleWindow(
    items.length,
    selectedIndex,
    maxVisibleItems,
  )
  const visibleItems = items.slice(start, end)
  const contentWidth = Math.max(20, width - 4)
  const labelWidth = Math.min(
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
        Commands
      </Text>
      {visibleItems.map((item, index) => {
        const absoluteIndex = start + index
        const selected = absoluteIndex === clampedIndex
        return (
          <Box key={item.insertValue} gap={1} width="100%" minWidth={0}>
            <Box width={labelWidth} flexShrink={0} minWidth={0}>
              <Text
                color={selected ? colors.text : colors.dimText}
                backgroundColor={selected ? colors.primary : undefined}
                bold={selected}
                wrap="truncate-end"
              >
                {' '}{item.label}{' '}
              </Text>
            </Box>
            <Box flexGrow={1} flexShrink={1} minWidth={0}>
              <Text color={colors.dimText} wrap="truncate-end">
                {item.description}
              </Text>
            </Box>
          </Box>
        )
      })}
      {items.length === 0 && (
        <Text color={colors.dimText} wrap="truncate-end">No matching commands</Text>
      )}
    </Box>
  )
}
