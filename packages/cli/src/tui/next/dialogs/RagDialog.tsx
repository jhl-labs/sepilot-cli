import type {
  DaemonRagFolder,
  DaemonRagSearchHit,
  DaemonRagVectorDbInfo,
} from '@sepilotd/api-client'
import { Box, Text, useInput } from 'ink'
import { useState } from 'react'
import { ControlSafeTextInput } from '../../components/ControlSafeTextInput.js'
import { colors } from '../../theme.js'
import { formatRagVectorInfo } from '../../utils/rag.js'

export interface RagDialogProps {
  initialQuery: string
  sources: DaemonRagFolder[]
  hits: DaemonRagSearchHit[]
  vectorInfo: DaemonRagVectorDbInfo | null
  loading: boolean
  error: string | null
  width: number
  onSearch(query: string): void
  onClose(): void
}

function compact(value: string, limit: number): string {
  const text = value.trim().replace(/\s+/g, ' ')
  return text.length > limit ? `${text.slice(0, Math.max(0, limit - 1))}…` : text
}

export function RagDialog({
  initialQuery,
  sources,
  hits,
  vectorInfo,
  loading,
  error,
  width,
  onSearch,
  onClose,
}: RagDialogProps) {
  const [query, setQuery] = useState(initialQuery)
  useInput((_, key) => {
    if (key.escape) onClose()
  })

  return (
    <Box flexDirection="column" width={width} borderStyle="round" borderColor={colors.info} paddingX={1}>
      <Text color={colors.info} bold>{`Local RAG · ${sources.length} sources`}</Text>
      <Text color={colors.dimText}>{formatRagVectorInfo(vectorInfo)}</Text>
      <Box>
        <Text color={colors.dimText}>query </Text>
        <ControlSafeTextInput
          value={query}
          onChange={setQuery}
          onSubmit={(value) => { if (value.trim()) onSearch(value.trim()) }}
          placeholder="search indexed knowledge…"
          focus={!loading}
        />
      </Box>
      {loading ? <Text color={colors.dimText}>Loading RAG…</Text> : null}
      {error ? <Text color={colors.error}>{error}</Text> : null}
      {!loading && !error && hits.length > 0 ? hits.slice(0, 8).map((hit) => (
        <Text key={`${hit.folderId}:${hit.documentId}`} wrap="truncate-end">
          <Text color={colors.info}>{`${Math.round(hit.score * 100)}%`}</Text>
          <Text>{` · ${compact(hit.title, 36)} · ${compact(hit.snippet, 80)}`}</Text>
        </Text>
      )) : null}
      {!loading && !error && query && hits.length === 0 ? <Text color={colors.dimText}>No RAG matches for this query.</Text> : null}
      {!loading && !error && !query ? sources.slice(0, 8).map((source) => (
        <Text key={source.id} wrap="truncate-end">
          <Text>{source.name}</Text>
          <Text color={colors.dimText}>{` · ${source.documents} docs · ${source.lastSyncStatus ?? 'not synced'}${source.path ? ` · ${compact(source.path, 48)}` : ''}`}</Text>
        </Text>
      )) : null}
      {!loading && !error && !query && sources.length === 0 ? <Text color={colors.dimText}>No local RAG sources. Use `sepilot rag source add`.</Text> : null}
      <Text color={colors.dimText}>enter search · esc close</Text>
    </Box>
  )
}
