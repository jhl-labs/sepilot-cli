import React from 'react'
import { Box, Text } from 'ink'
import { colors, symbols } from '../theme.js'

interface ThinkingIndicatorProps {
  text: string
}

export const ThinkingIndicator = React.memo(function ThinkingIndicator({
  text,
}: ThinkingIndicatorProps) {
  return (
    <Box gap={1} marginLeft={2}>
      <Text color={colors.warning}>{symbols.thinking}</Text>
      <Text color={colors.dimText}>
        {text ? text.slice(0, 80) : 'Thinking...'}
      </Text>
    </Box>
  )
})
