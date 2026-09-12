import { useEffect, useMemo, useState } from 'react'
import { Box, Text, useInput } from 'ink'
import type { DaemonSessionMeta } from '@sepilotd/api-client'
import { colors, symbols } from '../theme.js'
import { calculateVisibleWindow } from '../utils/layout.js'
import { isReturnKey } from '../utils/key.js'
import type { SessionExportFormat } from '../utils/session-commands.js'
import { PickerFrame, PickerRow } from './PickerPrimitives.js'

const MAX_RECENT_SESSIONS = 5

interface SessionPickerProps {
  query: string
  sessions: DaemonSessionMeta[]
  projectSessionIds: string[]
  currentSessionId: string | null
  recentSessionIds: string[]
  selectedIndex: number
  loading: boolean
  error: string | null
  maxVisibleItems?: number
  onQueryChange: (value: string) => void
  onSelectIndex: (index: number) => void
  onConfirm: (session: DaemonSessionMeta) => void
  onBranch: (session: DaemonSessionMeta) => void
  onCompact: (session: DaemonSessionMeta) => void
  onDelete: (session: DaemonSessionMeta) => void
  onExport: (session: DaemonSessionMeta, format: SessionExportFormat) => void
  onRetry?: () => void
  onClose: () => void
}

interface SessionPickerItem {
  session: DaemonSessionMeta
  shortcut?: number
  isCurrent: boolean
}

interface SessionPickerList {
  ordered: SessionPickerItem[]
  project: SessionPickerItem[]
  recent: SessionPickerItem[]
}

type SessionPickerActionId = 'load' | 'branch' | 'compact' | 'export-markdown' | 'export-json' | 'delete'

type SessionDeleteConfirmAction = 'cancel' | 'delete'

interface SessionPickerAction {
  id: SessionPickerActionId
  label: string
  description: string
}

interface SessionPickerActionHandlers {
  onConfirm: (session: DaemonSessionMeta) => void
  onBranch: (session: DaemonSessionMeta) => void
  onCompact: (session: DaemonSessionMeta) => void
  onDelete: (session: DaemonSessionMeta) => void
  onExport: (session: DaemonSessionMeta, format: SessionExportFormat) => void
}

const SESSION_PICKER_ACTIONS: SessionPickerAction[] = [
  {
    id: 'load',
    label: 'Continue',
    description: 'Load this session and keep working from its latest context.',
  },
  {
    id: 'branch',
    label: 'Fork',
    description: 'Create a branch at the current tail; the source stays unchanged.',
  },
  {
    id: 'compact',
    label: 'Compact',
    description: 'Summarize the selected session to reduce context pressure.',
  },
  {
    id: 'export-markdown',
    label: 'Export MD',
    description: 'Write a markdown export into the workspace.',
  },
  {
    id: 'export-json',
    label: 'Export JSON',
    description: 'Write a JSON export into the workspace.',
  },
  {
    id: 'delete',
    label: 'Delete',
    description: 'Delete the selected session after confirmation.',
  },
]

export function shouldRetrySessionPickerInput(
  input: string,
  key: { return?: boolean; name?: unknown; sequence?: unknown },
  opts: { loading: boolean; error: string | null },
): boolean {
  return Boolean(opts.error)
    && !opts.loading
    && (input.toLowerCase() === 'r' || isReturnKey(input, key))
}

function formatRelativeDate(value: string): string {
  const timestamp = Date.parse(value)
  if (!Number.isFinite(timestamp)) return value

  const diffMs = Date.now() - timestamp
  const diffMinutes = Math.max(0, Math.floor(diffMs / 60000))
  if (diffMinutes < 1) return 'just now'
  if (diffMinutes < 60) return `${diffMinutes}m ago`
  const diffHours = Math.floor(diffMinutes / 60)
  if (diffHours < 24) return `${diffHours}h ago`
  const diffDays = Math.floor(diffHours / 24)
  if (diffDays < 30) return `${diffDays}d ago`
  return new Date(timestamp).toISOString().slice(0, 10)
}

function formatCost(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '$0'
  return value >= 1 ? `$${value.toFixed(2)}` : `$${value.toFixed(4)}`
}

function formatTokenCount(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0'
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}m`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`
  return value.toLocaleString()
}

function summarizeSession(session: DaemonSessionMeta): string {
  const title = session.title?.trim() || '(untitled)'
  const totalTokens = session.totalTokens.input + session.totalTokens.output
  return `${title} ${symbols.separator} ${session.status} ${symbols.separator} ${session.provider}/${session.model} ${symbols.separator} ${session.messageCount} msgs ${symbols.separator} ${formatTokenCount(totalTokens)} tok ${symbols.separator} ${formatRelativeDate(session.updatedAt)}`
}

function buildInspectorLines(session: DaemonSessionMeta, isCurrent: boolean): string[] {
  const totalTokens = session.totalTokens.input + session.totalTokens.output
  const lines = [
    `${session.title?.trim() || '(untitled)'}${isCurrent ? '  current context' : ''}`,
    `session ${session.id.slice(0, 8)}  ${symbols.separator}  ${session.status}  ${symbols.separator}  updated ${formatRelativeDate(session.updatedAt)}`,
    `workspace ${session.cwd ?? 'unbound'}`,
    `stack ${session.provider}/${session.model}  ${symbols.separator}  ${session.messageCount} messages  ${symbols.separator}  ${totalTokens.toLocaleString()} tokens  ${symbols.separator}  ${formatCost(session.totalCost)}`,
    `actions Continue  ${symbols.separator}  Fork tail  ${symbols.separator}  Compact  ${symbols.separator}  Export`,
  ]

  if (isCurrent) {
    lines.push(`current-only shortcuts /context  ${symbols.separator}  /rewind 1  ${symbols.separator}  /resume`)
  }

  if (session.tags.length > 0) {
    lines.push(`tags ${session.tags.join(', ')}`)
  }

  return lines
}

export function executeSessionPickerAction(
  action: SessionPickerActionId,
  session: DaemonSessionMeta,
  handlers: SessionPickerActionHandlers,
) {
  switch (action) {
    case 'load':
      handlers.onConfirm(session)
      break
    case 'branch':
      handlers.onBranch(session)
      break
    case 'compact':
      handlers.onCompact(session)
      break
    case 'export-markdown':
      handlers.onExport(session, 'markdown')
      break
    case 'export-json':
      handlers.onExport(session, 'json')
      break
    case 'delete':
      handlers.onDelete(session)
      break
  }
}

export function buildSessionPickerList(
  sessions: DaemonSessionMeta[],
  projectSessionIds: string[],
  currentSessionId: string | null,
  recentSessionIds: string[],
  query: string,
): SessionPickerList {
  const trimmedQuery = query.trim()
  const projectOrder = new Map(projectSessionIds.map((sessionId, index) => [sessionId, index] as const))
  const recentOrder = new Map(recentSessionIds.map((sessionId, index) => [sessionId, index] as const))
  const projectSessions = trimmedQuery
    ? []
    : sessions
        .filter((session) => projectOrder.has(session.id))
        .sort(
          (left, right) =>
            (projectOrder.get(left.id) ?? Number.MAX_SAFE_INTEGER) -
            (projectOrder.get(right.id) ?? Number.MAX_SAFE_INTEGER),
        )
  const projectIds = new Set(projectSessions.map((session) => session.id))
  const recentSessions = trimmedQuery
    ? []
    : sessions
        .filter((session) => recentOrder.has(session.id) && !projectIds.has(session.id))
        .sort(
          (left, right) =>
            (recentOrder.get(left.id) ?? Number.MAX_SAFE_INTEGER) -
            (recentOrder.get(right.id) ?? Number.MAX_SAFE_INTEGER),
        )
        .slice(0, MAX_RECENT_SESSIONS)

  const quickPickSessions = projectSessions.length > 0 ? projectSessions : recentSessions
  const quickPickById = new Map(
    quickPickSessions.slice(0, MAX_RECENT_SESSIONS).map((session, index) => [session.id, index + 1] as const),
  )
  const orderedSessions = trimmedQuery
    ? sessions
    : [
        ...projectSessions,
        ...recentSessions,
        ...sessions.filter(
          (session) => !projectIds.has(session.id) && !recentSessions.some((item) => item.id === session.id),
        ),
      ]

  const ordered = orderedSessions.map((session) => ({
    session,
    shortcut: quickPickById.get(session.id),
    isCurrent: session.id === currentSessionId,
  }))

  return {
    ordered,
    project: trimmedQuery ? [] : ordered.filter((item) => projectIds.has(item.session.id)),
    recent: trimmedQuery
      ? []
      : ordered.filter((item) => !projectIds.has(item.session.id) && recentOrder.has(item.session.id)),
  }
}

function renderSessionRow(item: SessionPickerItem, selected: boolean) {
  return (
    <PickerRow
      key={item.session.id}
      marker={item.shortcut ? `${item.shortcut}` : item.session.id.slice(0, 8)}
      selected={selected}
      current={item.isCurrent}
    >
      {summarizeSession(item.session)}
    </PickerRow>
  )
}

export function SessionPicker({
  query,
  sessions,
  projectSessionIds,
  currentSessionId,
  recentSessionIds,
  selectedIndex,
  loading,
  error,
  maxVisibleItems = 12,
  onQueryChange,
  onSelectIndex,
  onConfirm,
  onBranch,
  onCompact,
  onDelete,
  onExport,
  onRetry,
  onClose,
}: SessionPickerProps) {
  const [actionsFocused, setActionsFocused] = useState(false)
  const [actionIndex, setActionIndex] = useState(0)
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false)
  const [deleteConfirmActionIndex, setDeleteConfirmActionIndex] = useState(0)
  const { ordered, project, recent } = useMemo(
    () => buildSessionPickerList(sessions, projectSessionIds, currentSessionId, recentSessionIds, query),
    [currentSessionId, projectSessionIds, query, recentSessionIds, sessions],
  )

  const clampedIndex = ordered.length === 0 ? 0 : Math.max(0, Math.min(selectedIndex, ordered.length - 1))
  const hasQuery = query.trim().length > 0
  const { start, end } = useMemo(
    () => calculateVisibleWindow(ordered.length, clampedIndex, maxVisibleItems),
    [clampedIndex, maxVisibleItems, ordered.length],
  )
  const visibleOrdered = ordered.slice(start, end)
  const projectBoundary = hasQuery ? 0 : project.length
  const recentBoundary = hasQuery ? 0 : project.length + recent.length
  const visibleProject = hasQuery ? [] : visibleOrdered.filter((_, index) => start + index < projectBoundary)
  const visibleRecent = hasQuery
    ? []
    : visibleOrdered.filter((_, index) => {
        const absoluteIndex = start + index
        return absoluteIndex >= projectBoundary && absoluteIndex < recentBoundary
      })
  const visibleAll = hasQuery ? visibleOrdered : visibleOrdered.filter((_, index) => start + index >= recentBoundary)
  const selectedItem = ordered[clampedIndex] ?? null
  const selectedSession = selectedItem?.session ?? null
  const selectedAction = SESSION_PICKER_ACTIONS[actionIndex] ?? SESSION_PICKER_ACTIONS[0]
  const selectedDeleteConfirmAction: SessionDeleteConfirmAction = deleteConfirmActionIndex === 1 ? 'delete' : 'cancel'
  const inspectorLines = selectedSession ? buildInspectorLines(selectedSession, selectedItem?.isCurrent ?? false) : []
  const actionDescription = deleteConfirmOpen
    ? `Confirm delete for ${selectedSession?.id.slice(0, 8)}.`
    : selectedAction.description

  useEffect(() => {
    if (clampedIndex !== selectedIndex) {
      onSelectIndex(clampedIndex)
    }
  }, [clampedIndex, onSelectIndex, selectedIndex])

  useEffect(() => {
    if (!selectedSession) {
      setActionsFocused(false)
      setActionIndex(0)
      setDeleteConfirmOpen(false)
      setDeleteConfirmActionIndex(0)
    }
  }, [selectedSession])

  useEffect(() => {
    setActionsFocused(false)
    setDeleteConfirmOpen(false)
    setDeleteConfirmActionIndex(0)
  }, [query])

  useEffect(() => {
    setDeleteConfirmOpen(false)
    setDeleteConfirmActionIndex(0)
  }, [actionIndex, clampedIndex])

  const runAction = (action: SessionPickerActionId) => {
    if (!selectedSession) {
      return
    }

    if (action === 'delete') {
      setActionsFocused(true)
      setDeleteConfirmOpen(true)
      setDeleteConfirmActionIndex(0)
      return
    }
    executeSessionPickerAction(action, selectedSession, {
      onConfirm,
      onBranch,
      onCompact,
      onDelete,
      onExport,
    })
    setDeleteConfirmOpen(false)
    setDeleteConfirmActionIndex(0)
  }

  useInput((input, key) => {
    if (deleteConfirmOpen) {
      if (key.escape || key.backspace || key.delete) {
        setDeleteConfirmOpen(false)
        setDeleteConfirmActionIndex(0)
        return
      }

      if (key.leftArrow || key.upArrow) {
        setDeleteConfirmActionIndex(0)
        return
      }

      if (key.rightArrow || key.downArrow) {
        setDeleteConfirmActionIndex(1)
        return
      }

      if (key.tab) {
        setDeleteConfirmActionIndex((current) => (current === 0 ? 1 : 0))
        return
      }

      if (isReturnKey(input, key) && selectedSession) {
        if (selectedDeleteConfirmAction === 'delete') {
          executeSessionPickerAction('delete', selectedSession, {
            onConfirm,
            onBranch,
            onCompact,
            onDelete,
            onExport,
          })
        }
        setDeleteConfirmOpen(false)
        setDeleteConfirmActionIndex(0)
      }
      return
    }

    if (key.escape) {
      onClose()
      return
    }

    if (shouldRetrySessionPickerInput(input, key, { loading, error })) {
      onRetry?.()
      return
    }

    if (key.tab) {
      if (selectedSession) {
        setActionsFocused((focused) => !focused)
        setDeleteConfirmOpen(false)
        setDeleteConfirmActionIndex(0)
      }
      return
    }

    if (actionsFocused && key.leftArrow) {
      setActionIndex((current) => (current === 0 ? SESSION_PICKER_ACTIONS.length - 1 : current - 1))
      return
    }

    if (actionsFocused && key.rightArrow) {
      setActionIndex((current) => (current >= SESSION_PICKER_ACTIONS.length - 1 ? 0 : current + 1))
      return
    }

    if (!actionsFocused && !hasQuery && /^[1-5]$/.test(input)) {
      const quickPickSource = project.length > 0 ? project : recent
      const quickPick = quickPickSource[Number(input) - 1]
      if (quickPick) {
        const quickPickIndex = ordered.findIndex((item) => item.session.id === quickPick.session.id)
        if (quickPickIndex >= 0) {
          onSelectIndex(quickPickIndex)
        }
        onConfirm(quickPick.session)
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

    if (isReturnKey(input, key) && selectedSession) {
      runAction(actionsFocused ? selectedAction.id : 'load')
    }
  })

  return (
    <PickerFrame
      title="Session Timeline"
      query={query}
      placeholder="title, provider, model, tag, id..."
      loading={loading}
      loadingLabel="Loading sessions..."
      error={error}
      isEmpty={ordered.length === 0}
      emptyLabel="No matching sessions."
      queryInputActive={!error}
      onQueryChange={onQueryChange}
      footer={
        <Text color={colors.dimText}>
          {deleteConfirmOpen
            ? '←/→ move  Enter select  Tab toggle  Backspace cancel  Esc cancel'
            : error && !loading
              ? 'r retry  Enter retry  Esc close'
              : `↑/↓ sessions  Enter ${actionsFocused ? 'run action' : 'continue'}  Tab ${actionsFocused ? 'search' : 'actions'}  Esc close${actionsFocused ? '  ←/→ action' : ''}${!actionsFocused && !hasQuery && (project.length > 0 || recent.length > 0) ? '  1-5 quick continue' : ''}`}
        </Text>
      }
    >
      {!loading && !error && visibleRecent.length > 0 && (
        <>
          {visibleProject.length > 0 && (
            <>
              <Text color={colors.dimText}>Project Context</Text>
              {visibleProject.map((item, index) => renderSessionRow(item, start + index === clampedIndex))}
            </>
          )}
          <Text color={colors.dimText}>Recent Context</Text>
          {visibleRecent.map((item, index) =>
            renderSessionRow(item, start + visibleProject.length + index === clampedIndex),
          )}
          {visibleAll.length > 0 && <Text color={colors.dimText}>All Sessions</Text>}
        </>
      )}
      {!loading && !error && visibleProject.length > 0 && visibleRecent.length === 0 && (
        <>
          <Text color={colors.dimText}>Project Context</Text>
          {visibleProject.map((item, index) => renderSessionRow(item, start + index === clampedIndex))}
          {visibleAll.length > 0 && <Text color={colors.dimText}>All Sessions</Text>}
        </>
      )}
      {!loading &&
        !error &&
        visibleAll.map((item, index) => {
          const rowIndex = hasQuery ? start + index : start + visibleProject.length + visibleRecent.length + index
          return renderSessionRow(item, rowIndex === clampedIndex)
        })}
      {!loading && !error && selectedSession && (
        <>
          <Text color={colors.dimText}>Selected Session Context</Text>
          {inspectorLines.map((line) => (
            <Text key={line} color={colors.muted} wrap="truncate-end">
              {line}
            </Text>
          ))}
          <Text color={colors.dimText}>Actions</Text>
          <Box gap={1} flexWrap="wrap">
            {SESSION_PICKER_ACTIONS.map((action, index) => {
              const selected = actionsFocused && index === actionIndex
              return (
                <Text
                  key={action.id}
                  color={selected ? colors.text : colors.muted}
                  backgroundColor={selected ? colors.primary : undefined}
                  bold={selected}
                >
                  {' '}
                  {action.label}{' '}
                </Text>
              )
            })}
          </Box>
          <Text color={colors.dimText}>{actionDescription}</Text>
          {deleteConfirmOpen ? (
            <>
              <Text color={colors.warning} wrap="truncate-end">
                Delete {selectedSession.id.slice(0, 8)} from local session history?
              </Text>
              <Box gap={1} minWidth={0} width="100%">
                <Text
                  color={selectedDeleteConfirmAction === 'cancel' ? colors.text : colors.dimText}
                  backgroundColor={selectedDeleteConfirmAction === 'cancel' ? colors.primary : undefined}
                  bold={selectedDeleteConfirmAction === 'cancel'}
                >
                  {selectedDeleteConfirmAction === 'cancel' ? '>' : ' '} Cancel{' '}
                </Text>
                <Text
                  color={selectedDeleteConfirmAction === 'delete' ? colors.text : colors.error}
                  backgroundColor={selectedDeleteConfirmAction === 'delete' ? colors.warning : undefined}
                  bold={selectedDeleteConfirmAction === 'delete'}
                >
                  {selectedDeleteConfirmAction === 'delete' ? '>' : ' '} Delete{' '}
                </Text>
              </Box>
            </>
          ) : null}
        </>
      )}
    </PickerFrame>
  )
}
