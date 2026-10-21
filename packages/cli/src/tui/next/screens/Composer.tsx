import { Box, Text } from 'ink'
import { colors } from '../../theme.js'
import { clampGraphemeOffset } from '../../utils/graphemes.js'
import { IME_CURSOR_MARKER } from '../runtime/ime-cursor.js'
import { deriveComposerInputPresentation } from '../../utils/composer-mode.js'

export interface ComposerProps {
  value: string
  cursorOffset?: number
  busy: boolean
  activityColor?: string
  width: number
  hardwareCursor?: boolean
}

export function Composer({
  value,
  cursorOffset = value.length,
  busy,
  activityColor = colors.primary,
  width,
  hardwareCursor = false,
}: ComposerProps) {
  const presentation = deriveComposerInputPresentation(value, cursorOffset)
  const shellMode = presentation.mode === 'shell'
  const safeCursorOffset = clampGraphemeOffset(
    presentation.editorValue,
    presentation.editorCursorOffset,
  )
  const beforeCursor = presentation.editorValue.slice(0, safeCursorOffset)
  const afterCursor = presentation.editorValue.slice(safeCursorOffset)
  return (
    <Box flexDirection="column" width={width}>
      <Box
        borderStyle="round"
        borderColor={shellMode ? colors.warning : busy ? activityColor : colors.primary}
        paddingX={1}
      >
        <Text color={shellMode ? colors.warning : colors.dimText} bold={shellMode}>
          {shellMode ? '$ ' : '› '}
        </Text>
        <Text>{`${beforeCursor}${hardwareCursor ? IME_CURSOR_MARKER : '│'}${afterCursor}`}</Text>
        {shellMode && presentation.editorValue.length === 0 ? (
          <Text color={colors.dimText}> Type a local shell command...</Text>
        ) : null}
      </Box>
    </Box>
  )
}
