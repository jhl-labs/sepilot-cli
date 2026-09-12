import { useState } from 'react'
import { Box, Text, useInput } from 'ink'
import { colors, symbols } from '../theme.js'
import { isReturnKey } from '../utils/key.js'

interface ProviderDeleteModalProps {
  providerId: string
  providerName: string
  nextProviderId: string
  nextModel: string
  currentProviderId: string
  currentModel: string
  willSwitchCurrentSession: boolean
  busy: boolean
  error: string | null
  onConfirm: () => void
  onClose: () => void
}

export function ProviderDeleteModal({
  providerId,
  providerName,
  nextProviderId,
  nextModel,
  currentProviderId,
  currentModel,
  willSwitchCurrentSession,
  busy,
  error,
  onConfirm,
  onClose,
}: ProviderDeleteModalProps) {
  const [selectedActionIndex, setSelectedActionIndex] = useState(0)
  const selectedAction = selectedActionIndex === 1 ? 'delete' : 'cancel'

  useInput((input, key) => {
    if (busy) {
      return
    }

    if (key.escape || key.backspace || key.delete) {
      onClose()
      return
    }

    if (key.leftArrow || key.upArrow) {
      setSelectedActionIndex(0)
      return
    }

    if (key.rightArrow || key.downArrow) {
      setSelectedActionIndex(1)
      return
    }

    if (input === '\t') {
      setSelectedActionIndex((current) => current === 0 ? 1 : 0)
      return
    }

    if (isReturnKey(input, key)) {
      if (selectedAction === 'delete') {
        onConfirm()
      } else {
        onClose()
      }
    }
  })

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={colors.warning}
      paddingX={1}
      marginY={1}
      width="100%"
      minWidth={0}
    >
      <Text color={colors.warning} bold>
        Delete Provider
      </Text>
      <Text>
        {providerName} {symbols.separator} {providerId}
      </Text>
      <Text color={colors.dimText}>
        Remove this provider from daemon config and reassign defaults.
      </Text>
      <Text color={colors.muted} wrap="truncate-end">
        Next daemon default {symbols.separator} {nextProviderId}/{nextModel}
      </Text>
      {willSwitchCurrentSession ? (
        <Text color={colors.muted} wrap="truncate-end">
          Current session {symbols.separator} {currentProviderId}/{currentModel} {'->'} {nextProviderId}/{nextModel}
        </Text>
      ) : (
        <Text color={colors.dimText} wrap="truncate-end">
          Current session stays on {currentProviderId}/{currentModel}.
        </Text>
      )}
      {error ? (
        <Text color={colors.error}>{error}</Text>
      ) : null}
      {busy ? (
        <Text color={colors.dimText}>Deleting provider...</Text>
      ) : null}
      {!busy ? (
        <Box gap={1} minWidth={0} width="100%">
          <Text
            color={selectedAction === 'cancel' ? colors.text : colors.dimText}
            backgroundColor={selectedAction === 'cancel' ? colors.primary : undefined}
            bold={selectedAction === 'cancel'}
          >
            {selectedAction === 'cancel' ? '>' : ' '} Cancel{' '}
          </Text>
          <Text
            color={selectedAction === 'delete' ? colors.text : colors.error}
            backgroundColor={selectedAction === 'delete' ? colors.warning : undefined}
            bold={selectedAction === 'delete'}
          >
            {selectedAction === 'delete' ? '>' : ' '} Delete{' '}
          </Text>
        </Box>
      ) : null}
      <Text color={colors.dimText}>
        {busy
          ? 'Please wait'
          : '←/→ move  Enter select  Tab toggle  Backspace cancel  Esc close'}
      </Text>
    </Box>
  )
}
