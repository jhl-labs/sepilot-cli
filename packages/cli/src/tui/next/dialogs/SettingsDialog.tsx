import { Box, Text, useInput } from 'ink'
import { useState } from 'react'
import { colors } from '../../theme.js'
import type { SettingsCategory } from '../commands/types.js'
import { commandsByCategory } from '../commands/registry.js'
import { itemsForCategory, scopeBadge, settingsCategories } from '../settings/schema.js'

const SETTINGS_COMMANDS = new Set([
  'model.pick', 'providers.open', 'mode.pick', 'thinking.pick', 'maxTokens.pick',
  'autonomy.pick', 'mcp.open', 'skills.open', 'memory.open', 'rag.open',
  'theme.pick', 'diagnostics.doctor', 'usage.open',
])

const CATEGORY_LABELS: Record<SettingsCategory, string> = {
  model: 'Model & Provider',
  agent: 'Agent & Mode',
  permissions: 'Permissions',
  mcp: 'MCP Servers',
  skills: 'Skills',
  memory: 'Memory & RAG',
  appearance: 'Appearance',
  diagnostics: 'Diagnostics',
}

export interface SettingsDialogProps {
  width: number
  onSelectItem(itemId: string): void
  onClose(): void
  errorText?: string | null
}

export function SettingsDialog({
  width,
  onSelectItem,
  onClose,
  errorText,
}: SettingsDialogProps) {
  const categories = settingsCategories()
  const [categoryIndex, setCategoryIndex] = useState(0)
  const [itemIndex, setItemIndex] = useState(0)
  const [column, setColumn] = useState<'category' | 'item'>('category')
  const category = categories[categoryIndex]
  const items = [
    ...itemsForCategory(category),
    ...commandsByCategory(category)
      .filter((command) => !SETTINGS_COMMANDS.has(command.id))
      .map((command) => ({
        id: `command:${command.id}`,
        category,
        label: command.title,
        scope: 'command' as const,
        kind: 'action' as const,
      })),
  ]

  useInput((input, key) => {
    if (input === 'q') {
      onClose()
      return
    }
    if (key.rightArrow) {
      setColumn('item')
      setItemIndex(0)
      return
    }
    if (key.leftArrow) {
      setColumn('category')
      return
    }
    if (key.upArrow) {
      if (column === 'category') {
        setCategoryIndex((index) => Math.max(0, index - 1))
        setItemIndex(0)
      } else {
        setItemIndex((index) => Math.max(0, index - 1))
      }
      return
    }
    // Ink reports escape as part of some terminal arrow sequences, so handle
    // directional keys before a standalone Escape.
    if (key.escape) {
      if (column === 'item') setColumn('category')
      else onClose()
      return
    }
    if (key.downArrow) {
      if (column === 'category') {
        setCategoryIndex((index) => Math.min(categories.length - 1, index + 1))
        setItemIndex(0)
      } else {
        setItemIndex((index) => Math.min(items.length - 1, index + 1))
      }
      return
    }
    if (!key.return) return
    if (column === 'category') {
      setColumn('item')
      setItemIndex(0)
      return
    }
    const item = items[itemIndex]
    if (item) onSelectItem(item.id)
  })

  const categoryWidth = Math.max(20, Math.floor(width / 3))
  return (
    <Box
      flexDirection="column"
      width={width}
      borderStyle="round"
      borderColor={colors.primary}
      paddingX={1}
    >
      <Text color={colors.primary}>Settings</Text>
      <Box flexDirection="row">
        <Box flexDirection="column" width={categoryWidth}>
          {categories.map((entry, index) => (
            <Text
              key={entry}
              color={column === 'category' && index === categoryIndex ? colors.primary : colors.text}
            >
              {`${index === categoryIndex ? '› ' : '  '}${CATEGORY_LABELS[entry]}`}
            </Text>
          ))}
        </Box>
        <Box flexDirection="column" flexGrow={1}>
          {items.map((item, index) => (
            <Text
              key={item.id}
              color={column === 'item' && index === itemIndex ? colors.primary : colors.text}
            >
              {`${column === 'item' && index === itemIndex ? '› ' : '  '}${item.label} ${scopeBadge(item.scope)}`}
            </Text>
          ))}
        </Box>
      </Box>
      {errorText ? <Text color={colors.error}>{errorText}</Text> : null}
      <Text color={colors.dimText}>↑↓ move · → items · ↵ select · esc back/close · q close</Text>
    </Box>
  )
}
