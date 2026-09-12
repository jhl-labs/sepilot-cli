import { useEffect, useMemo } from 'react'
import type { MarketplaceSkillSearchResult } from '@sepilotd/api-client'
import { Box, Text, useInput } from 'ink'
import { colors, symbols } from '../theme.js'
import { isReturnKey } from '../utils/key.js'
import { calculateVisibleWindow } from '../utils/layout.js'
import { getSkillCategoryTitle } from '../utils/skill-store.js'
import { ControlSafeTextInput } from './ControlSafeTextInput.js'

interface SkillStorePickerProps {
  query: string
  searchedQuery: string
  results: MarketplaceSkillSearchResult[]
  selectedIndex: number
  loading: boolean
  installingSource: string | null
  error: string | null
  message: string | null
  maxVisibleItems?: number
  onQueryChange: (value: string) => void
  onSearch: (query: string) => void
  onSelectIndex: (index: number) => void
  onInstall: (result: MarketplaceSkillSearchResult) => void
  onClose: () => void
}

function skillLabels(result: MarketplaceSkillSearchResult): string {
  const labels = [
    ...(result.metadata.tools ?? []).slice(0, 3).map((tool) => `tool:${tool}`),
    ...(result.metadata.tags ?? []).slice(0, 4).map((tag) => `#${tag}`),
  ]
  return labels.join(' ')
}

function buildSkillDetailLines(result: MarketplaceSkillSearchResult): string[] {
  const metadata = result.metadata
  return [
    `${metadata.name} v${metadata.version}`,
    metadata.description,
    `category ${getSkillCategoryTitle(metadata)}`,
    `catalog ${result.marketplace} ${symbols.separator} source ${result.source}`,
    metadata.author ? `author ${metadata.author}` : null,
    skillLabels(result) || null,
  ].filter((line): line is string => Boolean(line))
}

function renderSkillRow(
  result: MarketplaceSkillSearchResult,
  selected: boolean,
  installing: boolean,
) {
  const metadata = result.metadata
  const status = result.installed ? 'installed' : installing ? 'installing' : 'available'
  const category = getSkillCategoryTitle(metadata)

  return (
    <Box key={result.source} gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
      <Text
        color={selected ? colors.text : colors.dimText}
        backgroundColor={selected ? colors.primary : undefined}
        bold={selected}
      >
        {' '}
        {selected ? '>' : '-'}{' '}
      </Text>
      <Box flexGrow={1} flexShrink={1} minWidth={0} height={1} overflow="hidden">
        <Text color={selected ? colors.text : colors.muted} wrap="truncate-end">
          {metadata.name} v{metadata.version} {symbols.separator} {category} {symbols.separator}{' '}
          {metadata.description}
        </Text>
      </Box>
      <Text color={selected ? colors.text : result.installed ? colors.primary : colors.dimText}>
        {status}
      </Text>
    </Box>
  )
}

export function SkillStorePicker({
  query,
  searchedQuery,
  results,
  selectedIndex,
  loading,
  installingSource,
  error,
  message,
  maxVisibleItems = 10,
  onQueryChange,
  onSearch,
  onSelectIndex,
  onInstall,
  onClose,
}: SkillStorePickerProps) {
  const clampedIndex =
    results.length === 0 ? 0 : Math.max(0, Math.min(selectedIndex, results.length - 1))
  const trimmedQuery = query.trim()
  const queryDirty = trimmedQuery !== searchedQuery.trim()
  const selectedResult = results[clampedIndex] ?? null
  const actionLabel =
    trimmedQuery.length > 0 && (queryDirty || results.length === 0)
      ? 'search'
      : selectedResult?.installed
        ? 'already installed'
        : 'install'
  const { start, end } = useMemo(
    () => calculateVisibleWindow(results.length, clampedIndex, maxVisibleItems),
    [clampedIndex, maxVisibleItems, results.length],
  )
  const visibleResults = results.slice(start, end)
  const detailLines = selectedResult ? buildSkillDetailLines(selectedResult) : []

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
      onSelectIndex(results.length === 0 ? 0 : Math.min(results.length - 1, clampedIndex + 1))
      return
    }

    if (isReturnKey(input, key)) {
      if (trimmedQuery.length > 0 && (queryDirty || results.length === 0)) {
        onSearch(trimmedQuery)
        return
      }
      if (selectedResult && !selectedResult.installed && installingSource === null) {
        onInstall(selectedResult)
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
        Skill Catalog
      </Text>
      <Box>
        <Text color={colors.dimText}>search </Text>
        <ControlSafeTextInput
          value={query}
          onChange={onQueryChange}
          placeholder="blog, review, terminal, code..."
        />
      </Box>
      {message && (
        <Text color={colors.dimText} wrap="truncate-end">
          {message}
        </Text>
      )}
      {loading && <Text color={colors.dimText}>Searching skills...</Text>}
      {error && <Text color={colors.error}>{error}</Text>}
      {!loading && !error && results.length === 0 && searchedQuery && (
        <Text color={colors.dimText}>No skills matched "{searchedQuery}".</Text>
      )}
      {!loading &&
        !error &&
        visibleResults.map((result, index) =>
          renderSkillRow(
            result,
            start + index === clampedIndex,
            installingSource === result.source,
          ),
        )}
      {!loading && !error && selectedResult && (
        <>
          <Text color={colors.dimText}>Selected Skill</Text>
          {detailLines.map((line) => (
            <Text key={line} color={colors.muted} wrap="truncate-end">
              {line}
            </Text>
          ))}
        </>
      )}
      <Text color={colors.dimText}>↑/↓ move Enter {actionLabel} Esc close</Text>
    </Box>
  )
}
