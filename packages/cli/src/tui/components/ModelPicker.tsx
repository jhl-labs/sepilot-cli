import { useEffect, useMemo } from 'react'
import type { DaemonProviderInfo } from '@sepilotd/api-client'
import { Box, Text, useInput } from 'ink'
import { formatProviderModelBadges } from '../../utils/provider-display.js'
import { colors, symbols } from '../theme.js'
import {
  buildProviderModelPickerList,
  type ProviderModelPickerItem,
} from '../utils/provider-models.js'
import { isReturnKey } from '../utils/key.js'
import { calculateVisibleWindow } from '../utils/layout.js'
import { ControlSafeTextInput } from './ControlSafeTextInput.js'

export type ModelPickerConfirmTarget = 'session' | 'default'

interface ModelPickerProps {
  query: string
  providers: DaemonProviderInfo[]
  currentProvider: string
  currentModel: string
  defaultProvider: string
  defaultModel: string
  mru?: string[]
  selectedIndex: number
  loading: boolean
  error: string | null
  maxVisibleItems?: number
  onQueryChange: (value: string) => void
  onSelectIndex: (index: number) => void
  onConfirm: (selection: ProviderModelPickerItem, target: ModelPickerConfirmTarget) => void
  onAddProvider: () => void
  onClose: () => void
}

function formatProviderHealthBadge(item: {
  healthStatus?: 'ready' | 'env_missing' | 'unavailable'
  missingEnvVars?: string[]
}): { label: string; tone: 'success' | 'warning' | 'error' | 'info' } | null {
  if (item.healthStatus === 'env_missing') {
    const envLabel = item.missingEnvVars?.length
      ? `env:${item.missingEnvVars.join(',')}`
      : 'env missing'
    return {
      label: envLabel,
      tone: 'warning',
    }
  }
  if (item.healthStatus === 'unavailable') {
    return {
      label: 'unavailable',
      tone: 'error',
    }
  }
  return null
}

function renderPickerRow(
  item: ReturnType<typeof buildProviderModelPickerList>['ordered'][number],
  selected: boolean,
) {
  const badgeColor = (tone: 'primary' | 'success' | 'warning' | 'error' | 'info') => {
    if (selected) {
      return colors.text
    }
    switch (tone) {
      case 'success':
        return colors.success
      case 'warning':
        return colors.warning
      case 'error':
        return colors.error
      case 'info':
        return colors.info
      default:
        return colors.primary
    }
  }

  const renderStatusBadge = (
    label: string,
    tone: 'primary' | 'success' | 'warning' | 'error' | 'info',
  ) => <Text color={badgeColor(tone)}>[{label}]</Text>

  if (item.kind === 'action') {
    const badge = item.shortcut
      ? String(item.shortcut)
      : item.action === 'edit-provider'
        ? 'edit'
        : item.action === 'set-default-provider'
          ? 'default'
          : item.action === 'sync-session-default'
            ? 'sync'
            : item.action === 'delete-provider'
              ? 'delete'
              : 'add'
    const actionTone =
      item.action === 'delete-provider'
        ? 'error'
        : item.action === 'set-default-provider'
          ? 'warning'
          : item.action === 'sync-session-default'
            ? 'success'
            : item.action === 'edit-provider'
              ? 'info'
              : 'success'

    return (
      <Box
        key={`${item.action}:${item.providerId ?? item.presetType ?? 'setup'}`}
        gap={1}
        width="100%"
        minWidth={0}
        height={1}
        overflow="hidden"
      >
        <Text
          color={selected ? colors.text : badgeColor(actionTone)}
          backgroundColor={selected ? colors.primary : undefined}
          bold={selected}
        >
          {' '}
          {badge}{' '}
        </Text>
        <Box flexGrow={1} flexShrink={1} minWidth={0} height={1} overflow="hidden">
          <Text color={selected ? colors.text : colors.muted} wrap="truncate-end">
            {item.label} {symbols.separator} {item.detail}
          </Text>
        </Box>
        {item.isCurrentProvider && renderStatusBadge('current', 'success')}
        {item.isDefaultProvider && renderStatusBadge('default', 'warning')}
        {(() => {
          const healthBadge = formatProviderHealthBadge(item)
          return healthBadge ? renderStatusBadge(healthBadge.label, healthBadge.tone) : null
        })()}
      </Box>
    )
  }

  const label = item.shortcut ? `${item.shortcut}` : item.modelId
  const badgeText = item.model ? formatProviderModelBadges(item.model) : ''
  const providerLabel =
    item.providerName !== item.providerId
      ? `${item.providerName} (${item.providerId})`
      : item.providerId
  const availabilityLabel = item.synthetic ? 'not configured on daemon' : providerLabel
  const healthBadge = item.synthetic
    ? { label: 'unavailable', tone: 'error' as const }
    : formatProviderHealthBadge(item)

  return (
    <Box
      key={`${item.providerId}:${item.modelId}`}
      gap={1}
      width="100%"
      minWidth={0}
      height={1}
      overflow="hidden"
    >
      <Text
        color={selected ? colors.text : colors.dimText}
        backgroundColor={selected ? colors.primary : undefined}
        bold={selected}
      >
        {' '}
        {label}{' '}
      </Text>
      <Box flexGrow={1} flexShrink={1} minWidth={0} height={1} overflow="hidden">
        <Text color={selected ? colors.text : colors.muted} wrap="truncate-end">
          {item.providerId}/{item.modelId} {symbols.separator} {availabilityLabel}
          {badgeText}
        </Text>
      </Box>
      {item.isCurrent && renderStatusBadge('current', 'success')}
      {item.isDefault && renderStatusBadge('default', 'warning')}
      {item.synthetic && renderStatusBadge('unavailable', 'error')}
      {!item.synthetic && healthBadge && renderStatusBadge(healthBadge.label, healthBadge.tone)}
    </Box>
  )
}

export function ModelPicker({
  query,
  providers,
  currentProvider,
  currentModel,
  defaultProvider,
  defaultModel,
  mru = [],
  selectedIndex,
  loading,
  error,
  maxVisibleItems = 12,
  onQueryChange,
  onSelectIndex,
  onConfirm,
  onAddProvider,
  onClose,
}: ModelPickerProps) {
  const { ordered } = useMemo(
    () =>
      buildProviderModelPickerList(providers, currentProvider, currentModel, query, {
        defaultProviderId: defaultProvider,
        defaultModelId: defaultModel,
        mru,
      }),
    [currentModel, currentProvider, defaultModel, defaultProvider, mru, providers, query],
  )
  // The add-provider row is an extra virtual row appended after the built
  // list, so it shares the same up/down navigation and Enter handling as
  // every other row without needing its own selection model.
  const addProviderIndex = ordered.length
  const totalRows = ordered.length + 1
  const clampedIndex = Math.max(0, Math.min(selectedIndex, totalRows - 1))
  const hasQuery = query.trim().length > 0
  const commandLikeQuery = /^[!/]/.test(query.trim())
  const { start, end } = useMemo(
    () => calculateVisibleWindow(totalRows, clampedIndex, maxVisibleItems),
    [clampedIndex, maxVisibleItems, totalRows],
  )
  const visibleItems = ordered.slice(start, Math.min(end, ordered.length))
  const showAddProviderRow = end > addProviderIndex
  const providerHealthSummary = useMemo(
    () =>
      providers
        .map((provider) => {
          const flags = [provider.id]
          if (provider.id === currentProvider) {
            flags.push('current')
          }
          if (provider.id === defaultProvider) {
            flags.push('default')
          }
          if (provider.health.status === 'env_missing') {
            flags.push(`env:${provider.health.missingEnvVars.join(',') || 'missing'}`)
          } else if (provider.health.status === 'unavailable') {
            flags.push('unavailable')
          } else {
            flags.push('ready')
          }
          return flags.join(' ')
        })
        .join(` ${symbols.separator} `),
    [currentProvider, defaultProvider, providers],
  )

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

    if (input === 'a' && key.ctrl) {
      onAddProvider()
      return
    }

    if (input === 'd' && key.ctrl) {
      if (ordered[clampedIndex]) {
        onConfirm(ordered[clampedIndex]!, 'default')
      }
      return
    }

    if (!hasQuery && /^[1-9]$/.test(input)) {
      const quickPick = ordered[Number(input) - 1]
      if (quickPick) {
        onSelectIndex(Number(input) - 1)
        onConfirm(quickPick, 'session')
      }
      return
    }

    if (key.upArrow) {
      onSelectIndex(Math.max(0, clampedIndex - 1))
      return
    }

    if (key.downArrow) {
      onSelectIndex(Math.min(totalRows - 1, clampedIndex + 1))
      return
    }

    if (isReturnKey(input, key) || input === '\t') {
      if (clampedIndex === addProviderIndex) {
        onAddProvider()
        return
      }
      if (ordered[clampedIndex]) {
        onConfirm(ordered[clampedIndex]!, 'session')
      }
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
        Model & Provider Picker
      </Text>
      <Box>
        <Text color={colors.dimText}>search </Text>
        <ControlSafeTextInput
          value={query}
          onChange={onQueryChange}
          placeholder="provider, model..."
        />
      </Box>
      <Text color={colors.dimText}>
        current {symbols.separator} {currentProvider}/{currentModel}
      </Text>
      <Text color={colors.dimText} wrap="truncate-end">
        default {symbols.separator} {defaultProvider || 'none'}/{defaultModel || '?'}
      </Text>
      <Box gap={1} width="100%" minWidth={0}>
        <Box flexGrow={1} flexShrink={1} minWidth={0}>
          <Text color={colors.dimText} wrap="truncate-end">
            current = active session, default = daemon startup default, unavailable = not configured
            now
          </Text>
        </Box>
      </Box>
      {providers.length > 0 && (
        <Text color={colors.dimText} wrap="truncate-end">
          providers {symbols.separator} {providerHealthSummary}
        </Text>
      )}
      {loading && <Text color={colors.dimText}>Loading configured providers...</Text>}
      {error && <Text color={colors.error}>{error}</Text>}
      {commandLikeQuery && (
        <Text color={colors.warning}>
          Picker search is active. Press Esc to close before typing slash or shell commands.
        </Text>
      )}
      {!loading && !error && ordered.length === 0 && (
        <Text color={colors.dimText}>
          No configured models matched. Use the picker actions below or run `sepilot init` outside
          the TUI.
        </Text>
      )}
      {!loading &&
        !error &&
        visibleItems.map((item, index) => renderPickerRow(item, start + index === clampedIndex))}
      {!loading && !error && showAddProviderRow && (
        <Box gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
          <Text
            color={addProviderIndex === clampedIndex ? colors.text : colors.success}
            backgroundColor={addProviderIndex === clampedIndex ? colors.primary : undefined}
            bold={addProviderIndex === clampedIndex}
          >
            {' '}
            +{' '}
          </Text>
          <Text
            color={addProviderIndex === clampedIndex ? colors.text : colors.muted}
            wrap="truncate-end"
          >
            Add provider...
          </Text>
        </Box>
      )}
      <Text color={colors.dimText}>Enter 세션 · Ctrl+D 기본값 저장 · Ctrl+A provider 추가 · Esc 닫기</Text>
    </Box>
  )
}
