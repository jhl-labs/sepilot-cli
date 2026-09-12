import { useMemo } from 'react'
import type {
  DaemonMemoryEntry,
  DaemonMemorySemanticStatus,
} from '@sepilotd/api-client'
import { Box, Text } from 'ink'
import { colors, symbols } from '../theme.js'
import {
  compactMemoryContent,
  formatMemoryScore,
  summarizeMemoryTags,
} from '../utils/memory.js'

interface MemorySearchPanelProps {
  query: string
  results: DaemonMemoryEntry[]
  status: DaemonMemorySemanticStatus | null
  loading: boolean
  error: string | null
  height: number
}

function getStatusColor(status: DaemonMemorySemanticStatus | null): string {
  if (!status) return colors.dimText
  switch (status.status) {
    case 'ready':
      return colors.success
    case 'disabled':
    case 'reindex_required':
      return colors.warning
    case 'degraded':
      return colors.error
    case 'backfilling':
      return colors.info
    default:
      return colors.dimText
  }
}

export function MemorySearchPanel({
  query,
  results,
  status,
  loading,
  error,
  height,
}: MemorySearchPanelProps) {
  const statusBits = useMemo(() => {
    if (!status) return []

    const bits: string[] = [status.status]
    if (status.configuredProviderId && status.configuredModel) {
      bits.push(`${status.configuredProviderId}/${status.configuredModel}`)
    }
    if (status.pendingCount > 0) {
      bits.push(`pending:${status.pendingCount}`)
    }
    if (status.failedCount > 0) {
      bits.push(`failed:${status.failedCount}`)
    }
    return bits
  }, [status])
  const reservedRows = 2 + (statusBits.length > 0 ? 1 : 0)
  const visibleResults = useMemo(
    () => results.slice(0, Math.max(0, height - reservedRows)),
    [height, reservedRows, results],
  )

  if (height < 5) {
    return null
  }

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
        <Text color={colors.info} bold>
          Memory Search
        </Text>
        <Box flexShrink={0} marginLeft={1}>
          <Text color={colors.dimText}>
            {results.length} hits {symbols.separator} /memory close
          </Text>
        </Box>
      </Box>
      <Text color={colors.muted} wrap="truncate-end">
        query {symbols.separator} {compactMemoryContent(query, 60)}
      </Text>
      {statusBits.length > 0 && (
        <Text color={getStatusColor(status)} wrap="truncate-end">
          {statusBits.join(` ${symbols.separator} `)}
        </Text>
      )}
      {loading && (
        <Text color={colors.dimText}>Searching memory...</Text>
      )}
      {error && (
        <Text color={colors.error}>{error}</Text>
      )}
      {!loading && !error && visibleResults.length === 0 && (
        <Text color={colors.dimText}>No memory matches for this query.</Text>
      )}
      {!loading && !error && visibleResults.map((entry) => {
        const tags = summarizeMemoryTags(entry.tags)
        return (
          <Text key={entry.id} wrap="truncate-end">
            <Text color={colors.info}>{formatMemoryScore(entry.score)}</Text>
            <Text color={colors.dimText}> {symbols.separator} {entry.source}</Text>
            {tags && (
              <Text color={colors.warning}> {symbols.separator} {tags}</Text>
            )}
            <Text color={colors.text}> {symbols.separator} {compactMemoryContent(entry.content, tags ? 52 : 64)}</Text>
          </Text>
        )
      })}
    </Box>
  )
}
