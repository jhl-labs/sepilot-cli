import type { ReactNode } from 'react'
import { Box, Text } from 'ink'
import { colors } from '../theme.js'
import { ControlSafeTextInput } from './ControlSafeTextInput.js'

interface PickerFrameProps {
  title: string
  query: string
  placeholder: string
  loading: boolean
  loadingLabel: string
  error: string | null
  isEmpty: boolean
  emptyLabel: string
  queryInputActive?: boolean
  footer?: ReactNode
  children: ReactNode
  onQueryChange: (value: string) => void
}

export function PickerFrame({
  title,
  query,
  placeholder,
  loading,
  loadingLabel,
  error,
  isEmpty,
  emptyLabel,
  queryInputActive = true,
  footer,
  children,
  onQueryChange,
}: PickerFrameProps) {
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
        {title}
      </Text>
      <Box>
        <Text color={colors.dimText}>search </Text>
        {queryInputActive ? (
          <ControlSafeTextInput value={query} onChange={onQueryChange} placeholder={placeholder} />
        ) : (
          <Text color={query ? colors.text : colors.dimText}>{query || placeholder}</Text>
        )}
      </Box>
      {loading && <Text color={colors.dimText}>{loadingLabel}</Text>}
      {error && <Text color={colors.error}>{error}</Text>}
      {!loading && isEmpty && !error && <Text color={colors.dimText}>{emptyLabel}</Text>}
      {children}
      {footer}
    </Box>
  )
}

interface PickerRowProps {
  marker: string
  selected: boolean
  current?: boolean
  children: ReactNode
}

export function PickerRow({ marker, selected, current = false, children }: PickerRowProps) {
  return (
    <Box gap={1} width="100%" minWidth={0} height={1} overflow="hidden">
      <Text
        color={selected ? colors.text : colors.dimText}
        backgroundColor={selected ? colors.primary : undefined}
        bold={selected}
      >
        {' '}
        {marker}{' '}
      </Text>
      <Box flexGrow={1} flexShrink={1} minWidth={0} height={1} overflow="hidden">
        <Text color={selected ? colors.text : colors.muted} wrap="truncate-end">
          {children}
        </Text>
      </Box>
      {current && <Text color={selected ? colors.text : colors.primary}>current</Text>}
    </Box>
  )
}
