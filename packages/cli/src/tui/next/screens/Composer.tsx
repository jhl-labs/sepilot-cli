import { Box, Text } from 'ink'
import wrapAnsi from 'wrap-ansi'
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
  maxEditorRows?: number
}

export function composerViewport(value: string, cursorOffset: number, width: number, maxRows = 6): { text: string; rows: number } {
  const presentation = deriveComposerInputPresentation(value, cursorOffset)
  const offset = clampGraphemeOffset(presentation.editorValue, presentation.editorCursorOffset)
  const content = `${presentation.editorValue.slice(0, offset)}${IME_CURSOR_MARKER}${presentation.editorValue.slice(offset)}`
  const lines = wrapAnsi(content, Math.max(1, width - 6), { hard: true, trim: false }).split('\n')
  const limit = Math.max(1, Math.floor(maxRows))
  const caretRow = Math.max(0, lines.findIndex(line => line.includes(IME_CURSOR_MARKER)))
  const start = Math.max(0, Math.min(caretRow - Math.floor(limit / 2), lines.length - limit))
  const visible = lines.slice(start, start + limit)
  return { text: visible.join('\n'), rows: visible.length }
}

export function Composer({
  value,
  cursorOffset = value.length,
  busy,
  activityColor = colors.primary,
  width,
  hardwareCursor = false,
  maxEditorRows = 6,
}: ComposerProps) {
  const presentation = deriveComposerInputPresentation(value, cursorOffset)
  const shellMode = presentation.mode === 'shell'
  const viewport = composerViewport(value, cursorOffset, width, maxEditorRows)
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
        <Text wrap="truncate-end">{hardwareCursor ? viewport.text : viewport.text.replaceAll(IME_CURSOR_MARKER, '│')}</Text>
        {shellMode && presentation.editorValue.length === 0 ? (
          <Text color={colors.dimText} wrap="truncate-end"> Type a local shell command...</Text>
        ) : null}
      </Box>
    </Box>
  )
}
