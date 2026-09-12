import { useEffect, useMemo } from 'react'
import { Box, Text, useInput } from 'ink'
import { relative } from 'node:path'
import { colors } from '../theme.js'
import { isReturnKey } from '../utils/key.js'
import { calculateVisibleWindow } from '../utils/layout.js'

export interface FilePickerItem {
  absolutePath: string
  label: string
  isDirectory: boolean
}

interface FilePickerProps {
  rootDir: string
  currentDir: string
  items: FilePickerItem[]
  selectedIndex: number
  selectedPaths: string[]
  selectedLabels: string[]
  loading: boolean
  error: string | null
  maxVisibleItems?: number
  onSelectIndex: (index: number) => void
  onOpenDirectory: (item: FilePickerItem) => void
  onNavigateUp: () => void
  onToggleFile: (item: FilePickerItem) => void
  onConfirm: () => void
  onClose: () => void
}

function formatDirectory(rootDir: string, currentDir: string): string {
  const relativeDir = relative(rootDir, currentDir)
  if (!relativeDir) return '.'
  if (!relativeDir.startsWith('..')) return `./${relativeDir}`
  return currentDir
}

function summarizeSelections(selectedPaths: string[]): string {
  if (selectedPaths.length === 0) return 'none'
  if (selectedPaths.length <= 3) return selectedPaths.join(', ')
  return `${selectedPaths.slice(0, 3).join(', ')} +${selectedPaths.length - 3}`
}

export function FilePicker({
  rootDir,
  currentDir,
  items,
  selectedIndex,
  selectedPaths,
  selectedLabels,
  loading,
  error,
  maxVisibleItems = 12,
  onSelectIndex,
  onOpenDirectory,
  onNavigateUp,
  onToggleFile,
  onConfirm,
  onClose,
}: FilePickerProps) {
  const clampedIndex = items.length === 0
    ? 0
    : Math.max(0, Math.min(selectedIndex, items.length - 1))

  useEffect(() => {
    if (clampedIndex !== selectedIndex) {
      onSelectIndex(clampedIndex)
    }
  }, [clampedIndex, onSelectIndex, selectedIndex])

  const { start: firstVisibleIndex, end } = useMemo(
    () => calculateVisibleWindow(items.length, clampedIndex, maxVisibleItems),
    [clampedIndex, items.length, maxVisibleItems],
  )
  const visibleItems = useMemo(
    () => items.slice(firstVisibleIndex, end),
    [end, firstVisibleIndex, items],
  )

  useInput((input, key) => {
    if (key.escape) {
      onClose()
      return
    }

    if (key.upArrow) {
      onSelectIndex(Math.max(0, clampedIndex - 1))
      return
    }

    if (key.downArrow) {
      onSelectIndex(
        items.length === 0
          ? 0
          : Math.min(items.length - 1, clampedIndex + 1),
      )
      return
    }

    if (key.leftArrow || key.backspace || key.delete) {
      onNavigateUp()
      return
    }

    const activeItem = items[clampedIndex]
    if (!activeItem) return

    if (key.rightArrow && activeItem.isDirectory) {
      onOpenDirectory(activeItem)
      return
    }

    if (isReturnKey(input, key)) {
      if (activeItem.isDirectory) {
        onOpenDirectory(activeItem)
      } else {
        onToggleFile(activeItem)
      }
      return
    }

    if (input === ' ' && !activeItem.isDirectory) {
      onToggleFile(activeItem)
      return
    }

    if (input === '\t') {
      onConfirm()
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
        File Picker
      </Text>
      <Text color={colors.dimText} wrap="truncate-end">dir {formatDirectory(rootDir, currentDir)}</Text>
      {loading && (
        <Text color={colors.dimText}>Loading files...</Text>
      )}
      {error && (
        <Text color={colors.error}>{error}</Text>
      )}
      {!loading && items.length === 0 && !error && (
        <Text color={colors.dimText}>No files in this directory.</Text>
      )}
      {!loading && !error && visibleItems.map((item, index) => {
        const absoluteIndex = firstVisibleIndex + index
        const selected = absoluteIndex === clampedIndex
        const checked = selectedPaths.includes(item.absolutePath)
        return (
          <Box key={item.absolutePath} gap={1} width="100%" minWidth={0}>
            <Text
              color={selected ? colors.text : colors.dimText}
              backgroundColor={selected ? colors.primary : undefined}
              bold={selected}
            >
              {' '}
              {item.isDirectory ? 'dir' : checked ? '[x]' : '[ ]'}
              {' '}
            </Text>
            <Box flexGrow={1} flexShrink={1} minWidth={0}>
              <Text color={selected ? colors.text : colors.muted} wrap="truncate-end">
                {item.label}
              </Text>
            </Box>
          </Box>
        )
      })}
      <Text color={colors.dimText} wrap="truncate-end">
        selected {selectedLabels.length}: {summarizeSelections(selectedLabels)}
      </Text>
      <Text color={colors.dimText}>
        ↑/↓ move  Enter open/toggle  Space select  ←/Backspace parent  Tab attach  Esc close
      </Text>
    </Box>
  )
}
