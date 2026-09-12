import { useEffect, useMemo } from 'react'
import type { DaemonAgentDescriptor, DaemonAgentMode } from '@sepilotd/api-client'
import { Text, useInput } from 'ink'
import { colors, symbols } from '../theme.js'
import { isReturnKey } from '../utils/key.js'
import { calculateVisibleWindow } from '../utils/layout.js'
import { PickerFrame, PickerRow } from './PickerPrimitives.js'

interface ModePickerProps {
  query: string
  agents: DaemonAgentDescriptor[]
  currentMode: DaemonAgentMode
  selectedIndex: number
  loading: boolean
  error: string | null
  maxVisibleItems?: number
  onQueryChange: (value: string) => void
  onSelectIndex: (index: number) => void
  onConfirm: (agent: DaemonAgentDescriptor) => void
  onClose: () => void
}

export interface ModePickerItem {
  agent: DaemonAgentDescriptor
  shortcut?: number
  isCurrent: boolean
}

interface ModePickerList {
  ordered: ModePickerItem[]
}

function normalizeAgents(agents: DaemonAgentDescriptor[], currentMode: DaemonAgentMode): DaemonAgentDescriptor[] {
  const deduped = Array.from(new Map(agents.map((agent) => [agent.id, agent] as const)).values())

  if (!deduped.some((agent) => agent.id === currentMode)) {
    deduped.unshift({
      id: currentMode,
      name: currentMode,
      description: 'Current mode from config or existing session',
    })
  }

  return deduped
}

export function buildModePickerList(
  agents: DaemonAgentDescriptor[],
  currentMode: DaemonAgentMode,
  query: string,
): ModePickerList {
  const trimmedQuery = query.trim().toLowerCase()
  const normalizedAgents = normalizeAgents(agents, currentMode)
  const filteredAgents = trimmedQuery
    ? normalizedAgents.filter(
        (agent) =>
          agent.id.toLowerCase().includes(trimmedQuery) ||
          agent.name.toLowerCase().includes(trimmedQuery) ||
          agent.description.toLowerCase().includes(trimmedQuery),
      )
    : normalizedAgents

  const currentAgentIndex = filteredAgents.findIndex((agent) => agent.id === currentMode)
  const orderedAgents =
    currentAgentIndex > 0
      ? [filteredAgents[currentAgentIndex]!, ...filteredAgents.filter((agent) => agent.id !== currentMode)]
      : filteredAgents

  return {
    ordered: orderedAgents.map((agent, index) => ({
      agent,
      shortcut: !trimmedQuery && index < 9 ? index + 1 : undefined,
      isCurrent: agent.id === currentMode,
    })),
  }
}

export function findModeMatch(
  agents: DaemonAgentDescriptor[],
  currentMode: DaemonAgentMode,
  query: string,
): { match: DaemonAgentDescriptor | null; ambiguousMatches: DaemonAgentDescriptor[] } {
  const normalizedQuery = query.trim().toLowerCase()
  if (!normalizedQuery) {
    return { match: null, ambiguousMatches: [] }
  }

  const normalizedAgents = normalizeAgents(agents, currentMode)
  const exactMatch = normalizedAgents.find(
    (agent) => agent.id.toLowerCase() === normalizedQuery || agent.name.toLowerCase() === normalizedQuery,
  )
  if (exactMatch) {
    return { match: exactMatch, ambiguousMatches: [] }
  }

  const partialMatches = normalizedAgents.filter(
    (agent) =>
      agent.id.toLowerCase().includes(normalizedQuery) ||
      agent.name.toLowerCase().includes(normalizedQuery) ||
      agent.description.toLowerCase().includes(normalizedQuery),
  )
  if (partialMatches.length === 1) {
    return { match: partialMatches[0] ?? null, ambiguousMatches: [] }
  }

  return {
    match: null,
    ambiguousMatches: partialMatches,
  }
}

function renderModeRow(item: ModePickerItem, selected: boolean) {
  return (
    <PickerRow
      key={item.agent.id}
      marker={item.shortcut ? `${item.shortcut}` : item.agent.id}
      selected={selected}
      current={item.isCurrent}
    >
      {`${item.agent.name} ${symbols.separator} ${item.agent.description}`}
    </PickerRow>
  )
}

export function ModePicker({
  query,
  agents,
  currentMode,
  selectedIndex,
  loading,
  error,
  maxVisibleItems = 12,
  onQueryChange,
  onSelectIndex,
  onConfirm,
  onClose,
}: ModePickerProps) {
  const { ordered } = useMemo(() => buildModePickerList(agents, currentMode, query), [agents, currentMode, query])
  const clampedIndex = ordered.length === 0 ? 0 : Math.max(0, Math.min(selectedIndex, ordered.length - 1))
  const hasQuery = query.trim().length > 0
  const { start, end } = useMemo(
    () => calculateVisibleWindow(ordered.length, clampedIndex, maxVisibleItems),
    [clampedIndex, maxVisibleItems, ordered.length],
  )
  const visibleItems = ordered.slice(start, end)

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

    if (!hasQuery && /^[1-9]$/.test(input)) {
      const quickPick = ordered[Number(input) - 1]
      if (quickPick) {
        onSelectIndex(Number(input) - 1)
        onConfirm(quickPick.agent)
      }
      return
    }

    if (key.upArrow) {
      onSelectIndex(Math.max(0, clampedIndex - 1))
      return
    }

    if (key.downArrow) {
      onSelectIndex(ordered.length === 0 ? 0 : Math.min(ordered.length - 1, clampedIndex + 1))
      return
    }

    if ((isReturnKey(input, key) || input === '\t') && ordered[clampedIndex]) {
      onConfirm(ordered[clampedIndex].agent)
    }
  })

  return (
    <PickerFrame
      title="Mode Picker"
      query={query}
      placeholder="id, name, description..."
      loading={loading}
      loadingLabel="Loading agent modes..."
      error={error}
      isEmpty={ordered.length === 0}
      emptyLabel="No agent modes matched."
      onQueryChange={onQueryChange}
      footer={<Text color={colors.dimText}>↑/↓ move Enter select Tab select Esc close</Text>}
    >
      {!loading && !error && visibleItems.map((item, index) => renderModeRow(item, start + index === clampedIndex))}
    </PickerFrame>
  )
}
