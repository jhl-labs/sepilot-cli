import { useEffect, useMemo } from 'react'
import type { DaemonProviderInfo } from '@sepilotd/api-client'
import { Box, Text, useInput } from 'ink'
import { colors, symbols } from '../theme.js'
import { isReturnKey } from '../utils/key.js'
import { calculateVisibleWindow } from '../utils/layout.js'

export interface ProviderManagerRow {
  id: string
  name: string
  type: string | null
  modelCount: number
  defaultModel: string | null
  healthStatus: DaemonProviderInfo['health']['status']
  healthMessage?: string
  missingEnvVars?: string[]
  isCurrent: boolean
  isDefault: boolean
}

export function buildProviderManagerRows(options: {
  providers: DaemonProviderInfo[]
  configuredTypes?: Record<string, string>
  currentProvider: string
  defaultProvider: string
  defaultModel: string
}): ProviderManagerRow[] {
  return options.providers
    .map((provider) => ({
      id: provider.id,
      name: provider.name,
      type: options.configuredTypes?.[provider.id] ?? null,
      modelCount: provider.models.length,
      defaultModel: provider.id === options.defaultProvider
        ? options.defaultModel || null
        : provider.models[0]?.id ?? null,
      healthStatus: provider.health.status,
      healthMessage: provider.health.message,
      missingEnvVars: provider.health.missingEnvVars,
      isCurrent: provider.id === options.currentProvider,
      isDefault: provider.id === options.defaultProvider,
    }))
    .sort((left, right) => {
      if (left.isDefault !== right.isDefault) return left.isDefault ? -1 : 1
      if (left.isCurrent !== right.isCurrent) return left.isCurrent ? -1 : 1
      return left.id.localeCompare(right.id)
    })
}

function healthLabel(row: ProviderManagerRow): { text: string; color: string } {
  if (row.healthStatus === 'env_missing') {
    return {
      text: row.missingEnvVars?.length ? `missing ${row.missingEnvVars.join(', ')}` : 'env missing',
      color: colors.warning,
    }
  }
  if (row.healthStatus === 'unavailable') {
    return { text: row.healthMessage ?? 'unavailable', color: colors.error }
  }
  return { text: 'ready', color: colors.success }
}

interface ProviderManagerProps {
  rows: ProviderManagerRow[]
  selectedIndex: number
  loading: boolean
  error: string | null
  maxVisibleItems?: number
  onSelectIndex: (index: number) => void
  onAdd: () => void
  onEdit: (providerId: string) => void
  onSetDefault: (providerId: string) => void
  onDelete: (providerId: string) => void
  onClose: () => void
}

const ID_WIDTH = 18

export function ProviderManager({
  rows,
  selectedIndex,
  loading,
  error,
  maxVisibleItems = 10,
  onSelectIndex,
  onAdd,
  onEdit,
  onSetDefault,
  onDelete,
  onClose,
}: ProviderManagerProps) {
  const addRowIndex = rows.length
  const totalRows = rows.length + 1
  const clampedIndex = Math.max(0, Math.min(selectedIndex, totalRows - 1))
  const { start, end } = useMemo(
    () => calculateVisibleWindow(totalRows, clampedIndex, maxVisibleItems),
    [clampedIndex, maxVisibleItems, totalRows],
  )
  const selectedRow = rows[clampedIndex] ?? null

  useEffect(() => {
    if (clampedIndex !== selectedIndex) onSelectIndex(clampedIndex)
  }, [clampedIndex, onSelectIndex, selectedIndex])

  useInput((input, key) => {
    if (key.escape) { onClose(); return }
    if (key.upArrow) { onSelectIndex(Math.max(0, clampedIndex - 1)); return }
    if (key.downArrow) { onSelectIndex(Math.min(totalRows - 1, clampedIndex + 1)); return }
    if (input === 'a' || input === '+') { onAdd(); return }
    if (loading) return
    if (isReturnKey(input, key) || input === 'e') {
      if (clampedIndex === addRowIndex) onAdd()
      else if (selectedRow) onEdit(selectedRow.id)
      return
    }
    if (input === 'd' && selectedRow && !selectedRow.isDefault) { onSetDefault(selectedRow.id); return }
    if ((input === 'x' || key.delete) && selectedRow && rows.length > 1) onDelete(selectedRow.id)
  })

  const footer = selectedRow
    ? `Enter edit  d default  x delete  a add  Esc close`
    : 'Enter add  Esc close'

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
      <Text color={colors.primary} bold>Providers</Text>
      <Text color={colors.dimText}>
        Daemon providers and their default model. [default] is used for new sessions.
      </Text>
      {loading ? <Text color={colors.dimText}>Loading providers...</Text> : null}
      {error ? <Text color={colors.error}>{error}</Text> : null}
      {!loading && rows.length === 0 ? (
        <Text color={colors.dimText}>No providers configured yet. Add one to start chatting.</Text>
      ) : null}
      {rows.slice(start, Math.min(end, rows.length)).map((row, offset) => {
        const selected = start + offset === clampedIndex
        const health = healthLabel(row)
        return (
          <Box key={row.id} gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
            <Text color={selected ? colors.primary : colors.dimText}>{selected ? '›' : ' '}</Text>
            <Box width={ID_WIDTH} flexShrink={0}>
              <Text color={selected ? colors.primary : colors.text} bold={selected} wrap="truncate-end">
                {row.id}
              </Text>
            </Box>
            <Box flexGrow={1} flexShrink={1} minWidth={0}>
              <Text color={colors.muted} wrap="truncate-end">
                {[row.type ?? row.name, `${row.modelCount} model${row.modelCount === 1 ? '' : 's'}`, row.defaultModel]
                  .filter(Boolean)
                  .join(` ${symbols.separator} `)}
              </Text>
            </Box>
            {row.isDefault ? <Text color={colors.warning}>[default]</Text> : null}
            {row.isCurrent ? <Text color={colors.success}>[session]</Text> : null}
            <Text color={health.color}>[{health.text}]</Text>
          </Box>
        )
      })}
      {end > addRowIndex ? (
        <Box gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
          <Text color={clampedIndex === addRowIndex ? colors.primary : colors.dimText}>
            {clampedIndex === addRowIndex ? '›' : ' '}
          </Text>
          <Text color={clampedIndex === addRowIndex ? colors.primary : colors.success} bold={clampedIndex === addRowIndex}>
            + Add provider...
          </Text>
        </Box>
      ) : null}
      <Text color={colors.dimText}>{footer}</Text>
    </Box>
  )
}
