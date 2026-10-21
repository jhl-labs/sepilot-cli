import { useEffect, useMemo, useState } from 'react'
import type { DaemonProviderInfo } from '@sepilotd/api-client'
import { Box, Text, useInput } from 'ink'
import { formatProviderModelBadges } from '../../utils/provider-display.js'
import { colors, symbols } from '../theme.js'
import {
  buildProviderModelPickerList,
  buildProviderPickerList,
  type ProviderModelPickerItem,
  type ProviderPickerItem,
  type ProviderPickerOption,
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
  preferredProvider?: string
  preferredModel?: string
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

type PickerView = 'providers' | 'models'

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

function badgeColor(
  selected: boolean,
  tone: 'primary' | 'success' | 'warning' | 'error' | 'info',
): string {
  if (selected) return colors.text
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

function renderStatusBadge(
  selected: boolean,
  label: string,
  tone: 'primary' | 'success' | 'warning' | 'error' | 'info',
) {
  return <Text color={badgeColor(selected, tone)}>[{label}]</Text>
}

function renderPickerRow(
  item: Exclude<ProviderModelPickerItem, ProviderPickerOption>,
  selected: boolean,
) {
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
          color={selected ? colors.text : badgeColor(false, actionTone)}
          backgroundColor={selected ? colors.primary : undefined}
          bold={selected}
        >
          {' '}{badge}{' '}
        </Text>
        <Box flexGrow={1} flexShrink={1} minWidth={0} height={1} overflow="hidden">
          <Text color={selected ? colors.text : colors.muted} wrap="truncate-end">
            {item.label} {symbols.separator} {item.detail}
          </Text>
        </Box>
        {item.isCurrentProvider && renderStatusBadge(selected, 'current', 'success')}
        {item.isDefaultProvider && renderStatusBadge(selected, 'default', 'warning')}
        {(() => {
          const healthBadge = formatProviderHealthBadge(item)
          return healthBadge ? renderStatusBadge(selected, healthBadge.label, healthBadge.tone) : null
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
        {' '}{label}{' '}
      </Text>
      <Box flexGrow={1} flexShrink={1} minWidth={0} height={1} overflow="hidden">
        <Text color={selected ? colors.text : colors.muted} wrap="truncate-end">
          {item.providerId}/{item.modelId} {symbols.separator} {availabilityLabel}
          {badgeText}
        </Text>
      </Box>
      {item.isCurrent && renderStatusBadge(selected, 'current', 'success')}
      {item.isDefault && renderStatusBadge(selected, 'default', 'warning')}
      {item.synthetic && renderStatusBadge(selected, 'unavailable', 'error')}
      {!item.synthetic && healthBadge && renderStatusBadge(selected, healthBadge.label, healthBadge.tone)}
    </Box>
  )
}

function renderProviderRow(item: ProviderPickerOption, selected: boolean) {
  const modelLabel = `${item.modelCount} model${item.modelCount === 1 ? '' : 's'}`
  const defaultModel = item.defaultModelId
    ? `${item.isDefault ? 'default' : 'first'} ${item.defaultModelId}`
    : 'no model catalog'
  const providerLabel = item.providerName !== item.providerId
    ? `${item.providerName} (${item.providerId})`
    : item.providerId
  const healthBadge = formatProviderHealthBadge(item)

  return (
    <Box
      key={item.providerId}
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
        {' '}{item.shortcut ?? '›'}{' '}
      </Text>
      <Box flexGrow={1} flexShrink={1} minWidth={0} height={1} overflow="hidden">
        <Text color={selected ? colors.text : colors.muted} wrap="truncate-end">
          {providerLabel} {symbols.separator} {modelLabel} {symbols.separator} {defaultModel}
        </Text>
      </Box>
      {item.isCurrent && renderStatusBadge(selected, 'session', 'success')}
      {item.isDefault && renderStatusBadge(selected, 'default', 'warning')}
      {healthBadge && renderStatusBadge(selected, healthBadge.label, healthBadge.tone)}
    </Box>
  )
}

function isModelItem(item: ProviderModelPickerItem): item is Extract<ProviderModelPickerItem, { kind: 'model' }> {
  return item.kind === 'model'
}

export function ModelPicker({
  query,
  providers,
  currentProvider,
  currentModel,
  defaultProvider,
  defaultModel,
  preferredProvider,
  preferredModel,
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
  const [view, setView] = useState<PickerView>('providers')
  const [selectedProviderId, setSelectedProviderId] = useState(
    preferredProvider || currentProvider || defaultProvider,
  )
  const [searchMode, setSearchMode] = useState(query.trim().length > 0)
  const providerList = useMemo(
    () => buildProviderPickerList(providers, currentProvider, defaultProvider, defaultModel, query),
    [currentProvider, defaultModel, defaultProvider, providers, query],
  )
  const selectedProvider = providers.find((provider) => provider.id === selectedProviderId) ?? null
  const modelList = useMemo(() => {
    if (!selectedProvider) return []
    return buildProviderModelPickerList(
      [selectedProvider],
      currentProvider,
      currentModel,
      query,
      { defaultProviderId: defaultProvider, defaultModelId: defaultModel, mru },
    ).ordered.filter(isModelItem)
  }, [currentModel, currentProvider, defaultModel, defaultProvider, mru, query, selectedProvider])
  const ordered: ProviderPickerItem[] | ProviderModelPickerItem[] = view === 'providers'
    ? providerList.ordered
    : modelList
  const addProviderIndex = view === 'providers' ? ordered.length : -1
  const totalRows = ordered.length + (view === 'providers' ? 1 : 0)
  const clampedIndex = totalRows === 0
    ? 0
    : Math.max(0, Math.min(selectedIndex, totalRows - 1))
  const hasQuery = query.trim().length > 0
  const commandLikeQuery = searchMode && /^[!/]/.test(query.trim())
  const { start, end } = useMemo(
    () => calculateVisibleWindow(totalRows, clampedIndex, maxVisibleItems),
    [clampedIndex, maxVisibleItems, totalRows],
  )
  const visibleItems = ordered.slice(start, Math.min(end, ordered.length))
  const showAddProviderRow = view === 'providers' && end > addProviderIndex
  const providerIssueSummary = useMemo(
    () =>
      providers
        .filter((provider) => provider.health.status !== 'ready')
        .map((provider) => (
          provider.health.status === 'env_missing'
            ? `${provider.id} missing ${provider.health.missingEnvVars.join(',') || 'env'}`
            : `${provider.id} unavailable`
        ))
        .join(', '),
    [providers],
  )

  useEffect(() => {
    if (
      selectedProviderId
      && providers.some((provider) => provider.id === selectedProviderId)
    ) {
      return
    }
    setSelectedProviderId(preferredProvider || currentProvider || defaultProvider || providers[0]?.id || '')
  }, [currentProvider, defaultProvider, preferredProvider, providers, selectedProviderId])

  useEffect(() => {
    if (query.trim()) setSearchMode(true)
  }, [query])

  useEffect(() => {
    if (clampedIndex !== selectedIndex) onSelectIndex(clampedIndex)
  }, [clampedIndex, onSelectIndex, selectedIndex])

  const enterProvider = (item: ProviderPickerOption) => {
    setSelectedProviderId(item.providerId)
    setView('models')
    setSearchMode(false)
    onQueryChange('')

    const preferredModelForProvider = item.providerId === preferredProvider
      ? preferredModel
      : item.providerId === currentProvider
        ? currentModel
        : item.providerId === defaultProvider
          ? defaultModel
          : item.provider.models[0]?.id
    const nextIndex = buildProviderModelPickerList(
      [item.provider],
      currentProvider,
      currentModel,
      '',
      { defaultProviderId: defaultProvider, defaultModelId: defaultModel, mru },
    ).ordered.findIndex((candidate) => (
      candidate.kind === 'model' && candidate.modelId === preferredModelForProvider
    ))
    onSelectIndex(nextIndex >= 0 ? nextIndex : 0)
  }

  const returnToProviders = () => {
    setView('providers')
    setSearchMode(false)
    onQueryChange('')
    const providerIndex = providerList.ordered.findIndex((item) => (
      item.kind === 'provider' && item.providerId === selectedProviderId
    ))
    onSelectIndex(providerIndex >= 0 ? providerIndex : 0)
  }

  useInput((input, key) => {
    if (key.escape) {
      if (searchMode) {
        setSearchMode(false)
        onQueryChange('')
      } else {
        onClose()
      }
      return
    }

    if (input === 'a' && key.ctrl) {
      onAddProvider()
      return
    }

    if (searchMode) {
      if (key.upArrow) {
        onSelectIndex(Math.max(0, clampedIndex - 1))
      } else if (key.downArrow) {
        onSelectIndex(Math.min(Math.max(0, totalRows - 1), clampedIndex + 1))
      } else if (isReturnKey(input, key)) {
        if (view === 'providers') {
          const item = ordered[clampedIndex]
          if (item?.kind === 'provider') enterProvider(item)
          else if (item?.kind === 'action') onConfirm(item, 'session')
        } else {
          const item = modelList[clampedIndex]
          if (item) onConfirm(item, 'session')
        }
      }
      return
    }

    if (input === 'f') {
      setSearchMode(true)
      onQueryChange('')
      return
    }

    if (view === 'models' && (key.leftArrow || key.backspace || key.delete || input === 'b')) {
      returnToProviders()
      return
    }

    if (view === 'models' && (input === 'd' || input === '\u0004')) {
      const item = modelList[clampedIndex]
      if (item) onConfirm(item, 'default')
      return
    }

    if (!hasQuery && /^[1-9]$/.test(input)) {
      const quickPick = ordered[Number(input) - 1]
      if (view === 'providers' && quickPick?.kind === 'provider') {
        onSelectIndex(Number(input) - 1)
        enterProvider(quickPick)
      } else if (view === 'models' && quickPick?.kind === 'model') {
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
      onSelectIndex(Math.min(Math.max(0, totalRows - 1), clampedIndex + 1))
      return
    }

    if (isReturnKey(input, key) || input === '\t') {
      if (view === 'providers') {
        if (clampedIndex === addProviderIndex) {
          onAddProvider()
          return
        }
        const item = providerList.ordered[clampedIndex]
        if (item?.kind === 'provider') enterProvider(item)
        else if (item?.kind === 'action') onConfirm(item, 'session')
      } else {
        const item = modelList[clampedIndex]
        if (item) onConfirm(item, 'session')
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
      <Text color={colors.primary} bold>Models</Text>
      {searchMode ? (
        <Box>
          <Text color={colors.dimText}>search </Text>
          <ControlSafeTextInput
            value={query}
            onChange={onQueryChange}
            placeholder="provider, model..."
          />
        </Box>
      ) : (
        <Text color={colors.dimText}>f search provider/model</Text>
      )}
      <Text color={colors.dimText} wrap="truncate-end">
        session {currentProvider}/{currentModel}
        {' '}{symbols.separator}{' '}
        default {defaultProvider || 'none'}/{defaultModel || '?'}
        {providerIssueSummary ? ` ${symbols.separator} ${providerIssueSummary}` : ''}
      </Text>
      {view === 'providers' ? (
        <Text color={colors.dimText}>Choose a provider</Text>
      ) : (
        <Text color={colors.dimText} wrap="truncate-end">
          Provider {selectedProvider?.name ?? (selectedProviderId || 'unknown')}
          {selectedProviderId ? ` (${selectedProviderId})` : ''}
        </Text>
      )}
      {loading && <Text color={colors.dimText}>Loading configured providers...</Text>}
      {error && <Text color={colors.error}>{error}</Text>}
      {commandLikeQuery && (
        <Text color={colors.warning}>
          Picker search is active. Press Esc to exit search before typing slash or shell commands.
        </Text>
      )}
      {!loading && !error && ordered.length === 0 && (
        <Text color={colors.dimText}>
          {view === 'providers'
            ? 'No configured providers matched. Pick an action below to add or repair a provider.'
            : 'No models matched for this provider. Press Esc to return to providers.'}
        </Text>
      )}
      {!loading && !error && visibleItems.map((item, index) => (
        item.kind === 'provider'
          ? renderProviderRow(item, start + index === clampedIndex)
          : renderPickerRow(item, start + index === clampedIndex)
      ))}
      {!loading && !error && showAddProviderRow && (
        <Box gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
          <Text
            color={addProviderIndex === clampedIndex ? colors.text : colors.success}
            backgroundColor={addProviderIndex === clampedIndex ? colors.primary : undefined}
            bold={addProviderIndex === clampedIndex}
          >
            {' '}+{' '}
          </Text>
          <Text
            color={addProviderIndex === clampedIndex ? colors.text : colors.muted}
            wrap="truncate-end"
          >
            Add provider...
          </Text>
        </Box>
      )}
      <Text color={colors.dimText}>
        {view === 'providers'
          ? 'Enter choose provider · f search · Ctrl+A add provider · Esc close'
          : 'Enter use in session · d save as default · ←/Backspace providers · f search · Esc close'}
      </Text>
    </Box>
  )
}
