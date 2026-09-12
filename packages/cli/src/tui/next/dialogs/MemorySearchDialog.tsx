import type { DaemonMemoryEntry, DaemonMemorySemanticStatus } from '@sepilotd/api-client'
import { Box, Text, useInput } from 'ink'
import { useState } from 'react'
import { ControlSafeTextInput } from '../../components/ControlSafeTextInput.js'
import { colors } from '../../theme.js'
import { compactMemoryContent, formatMemoryScore } from '../../utils/memory.js'

export interface MemorySearchDialogProps {
  initialQuery: string
  width: number
  status: DaemonMemorySemanticStatus | null
  results: DaemonMemoryEntry[]
  loading: boolean
  error: string | null
  onSearch(query: string): void
  onClose(): void
}

export function MemorySearchDialog({
  initialQuery,
  width,
  status,
  results,
  loading,
  error,
  onSearch,
  onClose,
}: MemorySearchDialogProps) {
  const [query, setQuery] = useState(initialQuery)
  useInput((_, key) => {
    if (key.escape) onClose()
  })

  return (
    <Box flexDirection="column" width={width} borderStyle="round" borderColor={colors.info} paddingX={1}>
      <Text color={colors.info} bold>Memory Search</Text>
      <Text color={colors.dimText}>
        {status ? `semantic ${status.status}${status.configuredProviderId && status.configuredModel ? ` · ${status.configuredProviderId}/${status.configuredModel}` : ''}` : 'semantic status unavailable'}
      </Text>
      <Box>
        <Text color={colors.dimText}>query </Text>
        <ControlSafeTextInput
          value={query}
          onChange={setQuery}
          onSubmit={(value) => { if (value.trim()) onSearch(value.trim()) }}
          placeholder="search durable memory…"
          focus={!loading}
        />
      </Box>
      {loading ? <Text color={colors.dimText}>Searching…</Text> : null}
      {error ? <Text color={colors.error}>{error}</Text> : null}
      {!loading && !error && results.length === 0 ? <Text color={colors.dimText}>Enter a query to search memory.</Text> : null}
      {!loading && !error ? results.slice(0, 10).map((entry) => (
        <Text key={entry.id} wrap="truncate-end">
          <Text color={colors.info}>{formatMemoryScore(entry.score)}</Text>
          <Text color={colors.dimText}>{` · ${entry.source} · `}</Text>
          <Text>{compactMemoryContent(entry.content, 80)}</Text>
        </Text>
      )) : null}
      <Text color={colors.dimText}>enter search · esc close</Text>
    </Box>
  )
}
