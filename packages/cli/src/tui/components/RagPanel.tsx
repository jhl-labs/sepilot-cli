import { useEffect, useMemo } from 'react'
import type {
  DaemonRagFolder,
  DaemonRagSearchHit,
  DaemonRagSyncResult,
  DaemonRagVectorDbInfo,
} from '@sepilotd/api-client'
import { Box, Text, useInput } from 'ink'
import { colors, symbols } from '../theme.js'
import { isReturnKey } from '../utils/key.js'
import { calculateVisibleWindow } from '../utils/layout.js'
import {
  formatRagSyncResult,
  formatRagVectorInfo,
} from '../utils/rag.js'

interface RagPanelProps {
  query: string
  sources: DaemonRagFolder[]
  hits: DaemonRagSearchHit[]
  vectorInfo: DaemonRagVectorDbInfo | null
  syncResult: DaemonRagSyncResult | null
  loading: boolean
  error: string | null
  selectedIndex: number
  height: number
  onSelectIndex: (index: number) => void
  onSync: () => void
  onClose: () => void
}

function compact(value: string | undefined, limit: number): string {
  const text = value?.trim().replace(/\s+/g, ' ') ?? ''
  if (!text) return ''
  return text.length > limit ? `${text.slice(0, Math.max(0, limit - 3))}...` : text
}

function sourceLine(source: DaemonRagFolder, selected: boolean) {
  const syncLabel = source.lastSyncStatus
    ? source.lastSyncStatus === 'success' ? 'synced' : 'error'
    : 'not synced'
  return (
    <Text key={source.id} wrap="truncate-end">
      <Text color={selected ? colors.text : colors.dimText}>
        {selected ? '>' : '-'}{' '}
      </Text>
      <Text color={selected ? colors.primary : colors.text}>
        {source.name}
      </Text>
      <Text color={colors.dimText}>
        {' '}{symbols.separator} {source.documents} docs {symbols.separator} {syncLabel}
      </Text>
      {source.path && (
        <Text color={colors.muted}> {symbols.separator} {compact(source.path, 48)}</Text>
      )}
    </Text>
  )
}

function hitLine(hit: DaemonRagSearchHit) {
  return (
    <Text key={`${hit.folderId}:${hit.documentId}:${hit.score}`} wrap="truncate-end">
      <Text color={colors.info}>{Math.round(hit.score * 100)}%</Text>
      <Text color={colors.dimText}> {symbols.separator} </Text>
      <Text color={colors.text}>{compact(hit.title, 36)}</Text>
      {hit.path && (
        <Text color={colors.muted}> {symbols.separator} {compact(hit.path, 36)}</Text>
      )}
      <Text color={colors.dimText}> {symbols.separator} {compact(hit.snippet, 70)}</Text>
    </Text>
  )
}

export function RagPanel({
  query,
  sources,
  hits,
  vectorInfo,
  syncResult,
  loading,
  error,
  selectedIndex,
  height,
  onSelectIndex,
  onSync,
  onClose,
}: RagPanelProps) {
  const clampedIndex = sources.length === 0
    ? 0
    : Math.max(0, Math.min(selectedIndex, sources.length - 1))
  const hitRows = Math.min(hits.length, Math.max(0, Math.floor((height - 7) / 2)))
  const sourceRows = Math.max(1, height - 7 - hitRows)
  const { start, end } = useMemo(
    () => calculateVisibleWindow(sources.length, clampedIndex, sourceRows),
    [clampedIndex, sourceRows, sources.length],
  )
  const visibleSources = sources.slice(start, end)

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
    if (key.upArrow) {
      onSelectIndex(Math.max(0, clampedIndex - 1))
      return
    }
    if (key.downArrow) {
      onSelectIndex(
        sources.length === 0
          ? 0
          : Math.min(sources.length - 1, clampedIndex + 1),
      )
      return
    }
    if (isReturnKey(input, key)) {
      onSync()
    }
  })

  if (height < 6) return null

  return (
    <Box
      flexDirection="column"
      height={height}
      borderStyle="single"
      borderColor={colors.border}
      paddingX={1}
      marginBottom={1}
      overflow="hidden"
      width="100%"
      minWidth={0}
    >
      <Box justifyContent="space-between" width="100%" minWidth={0}>
        <Text color={colors.info} bold>Local RAG</Text>
        <Text color={colors.dimText}>
          {sources.length} sources {symbols.separator} /rag close
        </Text>
      </Box>
      <Text color={colors.muted} wrap="truncate-end">
        {formatRagVectorInfo(vectorInfo)}
      </Text>
      {query && (
        <Text color={colors.muted} wrap="truncate-end">
          query {symbols.separator} {compact(query, 72)}
        </Text>
      )}
      {syncResult && (
        <Text color={syncResult.ok ? colors.success : colors.warning} wrap="truncate-end">
          {formatRagSyncResult(syncResult).replace(/\n/g, '  ')}
        </Text>
      )}
      {loading && <Text color={colors.dimText}>Loading RAG...</Text>}
      {error && <Text color={colors.error}>{error}</Text>}
      {!loading && !error && hits.slice(0, hitRows).map(hitLine)}
      {!loading && !error && hits.length === 0 && query && (
        <Text color={colors.dimText}>No RAG matches for this query.</Text>
      )}
      {!loading && !error && visibleSources.map((source, index) => (
        sourceLine(source, start + index === clampedIndex)
      ))}
      {!loading && !error && sources.length === 0 && (
        <Text color={colors.dimText}>No local RAG sources. Use /rag add &lt;path&gt;.</Text>
      )}
      <Text color={colors.dimText}>↑/↓ source  Enter sync  Esc close</Text>
    </Box>
  )
}
