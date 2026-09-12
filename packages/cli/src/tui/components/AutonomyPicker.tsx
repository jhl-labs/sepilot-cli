import { useEffect } from 'react'
import type { AutonomyLevel } from '../utils/autonomy.js'
import { AUTONOMY_OPTIONS } from '../utils/autonomy.js'
import { Box, Text, useInput } from 'ink'
import { colors } from '../theme.js'
import { isReturnKey } from '../utils/key.js'
import { calculateVisibleWindow } from '../utils/layout.js'
import { PickerRow } from './PickerPrimitives.js'

interface AutonomyPickerProps {
  currentLevel: AutonomyLevel
  selectedIndex: number
  maxVisibleItems?: number
  onSelectIndex: (index: number) => void
  onConfirm: (level: AutonomyLevel) => void
  onClose: () => void
}

export function AutonomyPicker({
  currentLevel,
  selectedIndex,
  maxVisibleItems = AUTONOMY_OPTIONS.length,
  onSelectIndex,
  onConfirm,
  onClose,
}: AutonomyPickerProps) {
  const clampedIndex = Math.max(0, Math.min(selectedIndex, AUTONOMY_OPTIONS.length - 1))
  const { start, end } = calculateVisibleWindow(
    AUTONOMY_OPTIONS.length,
    clampedIndex,
    maxVisibleItems,
  )
  const visibleOptions = AUTONOMY_OPTIONS.slice(start, end)
  const quickPickMax = Math.min(AUTONOMY_OPTIONS.length, 9)

  useEffect(() => {
    if (clampedIndex !== selectedIndex) {
      onSelectIndex(clampedIndex)
    }
  }, [clampedIndex, onSelectIndex, selectedIndex])

  useInput((input, key) => {
    if (key.escape) {
      onClose()
      return
    }

    const quickPick = Number(input)
    if (Number.isInteger(quickPick) && quickPick >= 1 && quickPick <= quickPickMax) {
      const option = AUTONOMY_OPTIONS[quickPick - 1]
      if (option) {
        onSelectIndex(quickPick - 1)
        onConfirm(option.id)
      }
      return
    }

    if (key.upArrow) {
      onSelectIndex(Math.max(0, clampedIndex - 1))
      return
    }

    if (key.downArrow) {
      onSelectIndex(Math.min(AUTONOMY_OPTIONS.length - 1, clampedIndex + 1))
      return
    }

    if ((isReturnKey(input, key) || input === '\t') && AUTONOMY_OPTIONS[clampedIndex]) {
      onConfirm(AUTONOMY_OPTIONS[clampedIndex]!.id)
    }
  })

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={colors.primary}
      paddingX={1}
      marginY={1}
      width="100%"
      minWidth={0}
    >
      <Text color={colors.primary} bold>
        Autonomy Picker
      </Text>
      {visibleOptions.map((option, index) => {
        const optionIndex = start + index
        const selected = optionIndex === clampedIndex
        const isCurrent = option.id === currentLevel
        return (
          <PickerRow
            key={option.id}
            marker={`${optionIndex + 1}`}
            selected={selected}
            current={isCurrent}
          >
            {option.icon} {option.label} ({option.id}) {option.description}
          </PickerRow>
        )
      })}
      <Text color={colors.dimText}>
        ↑/↓ move Enter select 1-{quickPickMax} quick pick Tab select Esc close
      </Text>
    </Box>
  )
}
