import type { DaemonSkill } from '@sepilotd/api-client'
import { Box, Text } from 'ink'
import { colors, symbols } from '../theme.js'
import { calculateVisibleWindow } from '../utils/layout.js'

interface SkillPaletteProps {
  items: DaemonSkill[]
  selectedIndex: number
  loading?: boolean
  error?: string | null
  query?: string
  maxVisibleItems?: number
  width?: number
}

function skillLabels(skill: DaemonSkill): string {
  return [
    ...(skill.tools ?? []).slice(0, 2).map((tool) => `tool:${tool}`),
    ...(skill.tags ?? []).slice(0, 3).map((tag) => `#${tag}`),
  ].join(' ')
}

export function SkillPalette({
  items,
  selectedIndex,
  loading = false,
  error = null,
  query = '',
  maxVisibleItems = items.length,
  width = 80,
}: SkillPaletteProps) {
  const { start, end, clampedIndex } = calculateVisibleWindow(
    items.length,
    selectedIndex,
    maxVisibleItems,
  )
  const visibleItems = items.slice(start, end)
  const contentWidth = Math.max(20, width - 4)
  const nameWidth = Math.min(30, Math.max(16, Math.floor(contentWidth * 0.32)))
  const versionWidth = 10

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
        Skills
      </Text>
      {loading ? <Text color={colors.dimText}>Loading installed skills...</Text> : null}
      {error ? (
        <Text color={colors.error} wrap="truncate-end">
          {error}
        </Text>
      ) : null}
      {!loading && !error && items.length === 0 ? (
        <Text color={colors.dimText} wrap="truncate-end">
          {query.trim() ? `No enabled skills match "${query}".` : 'No enabled skills installed.'}
        </Text>
      ) : null}
      {!loading &&
        !error &&
        visibleItems.map((item, index) => {
          const absoluteIndex = start + index
          const selected = absoluteIndex === clampedIndex
          const labels = skillLabels(item)
          return (
            <Box key={item.id} gap={1} width="100%" minWidth={0}>
              <Box width={nameWidth} flexShrink={0} minWidth={0}>
                <Text
                  color={selected ? colors.text : colors.dimText}
                  backgroundColor={selected ? colors.primary : undefined}
                  bold={selected}
                  wrap="truncate-end"
                >
                  {' '}
                  {item.id}{' '}
                </Text>
              </Box>
              <Box width={versionWidth} flexShrink={0} minWidth={0}>
                <Text color={selected ? colors.text : colors.dimText} wrap="truncate-end">
                  v{item.version}
                </Text>
              </Box>
              <Box flexGrow={1} flexShrink={1} minWidth={0}>
                <Text color={selected ? colors.text : colors.dimText} wrap="truncate-end">
                  {item.name} {symbols.separator} {item.description}
                  {labels ? ` ${symbols.separator} ${labels}` : ''}
                </Text>
              </Box>
            </Box>
          )
        })}
    </Box>
  )
}
