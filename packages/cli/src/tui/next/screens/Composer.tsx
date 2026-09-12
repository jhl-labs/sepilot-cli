import { Box, Text } from 'ink'
import { colors } from '../../theme.js'
import { clampGraphemeOffset } from '../../utils/graphemes.js'
import { IME_CURSOR_MARKER } from '../runtime/ime-cursor.js'
import { deriveComposerInputPresentation } from '../../utils/composer-mode.js'

export interface ComposerProps {
  value: string
  cursorOffset?: number
  leaderArmed: boolean
  leaderHintText: string
  busy: boolean
  busyLabel?: string | null
  activityFrame?: string | null
  activityColor?: string
  width: number
  hardwareCursor?: boolean
}

export function Composer({
  value,
  cursorOffset = value.length,
  leaderArmed,
  leaderHintText,
  busy,
  busyLabel = null,
  activityFrame = null,
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
      {leaderArmed && leaderHintText ? <Text color={colors.dimText}>{leaderHintText}</Text> : null}
      {shellMode ? <Text color={colors.warning}>SHELL · Enter:run · Esc:chat · !!:rerun</Text> : null}
      {busy ? (
        <Text color={activityColor} bold>
          {activityFrame ?? '⠋'} {busyLabel ?? 'Working…'}
        </Text>
      ) : null}
    </Box>
  )
}
