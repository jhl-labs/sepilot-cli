import { Box, Text, useInput } from 'ink'
import { useState } from 'react'
import wrapAnsi from 'wrap-ansi'
import { colors } from '../../theme.js'

export interface InfoDialogProps {
  title: string
  body: string
  width: number
  height?: number
  onClose(): void
}

export function InfoDialog({ title, body, width, height, onClose }: InfoDialogProps) {
  const [position, setPosition] = useState(0)
  const lines = wrapAnsi(body, Math.max(1, width - 4), { hard: true, trim: false }).split('\n')
  const dialogHeight = height === undefined ? undefined : Math.min(height, lines.length + 4)
  const pageSize = dialogHeight === undefined ? lines.length : Math.max(1, dialogHeight - 4)
  const maxOffset = Math.max(0, lines.length - pageSize)
  const offset = Math.min(position, maxOffset)
  useInput((input, key) => {
    if (key.escape || key.return || input === 'q') onClose()
    else if (key.upArrow) setPosition(Math.max(0, offset - 1))
    else if (key.downArrow) setPosition(Math.min(maxOffset, offset + 1))
    else if (key.pageUp) setPosition(Math.max(0, offset - pageSize))
    else if (key.pageDown) setPosition(Math.min(maxOffset, offset + pageSize))
    else if (input === 'g') setPosition(0)
    else if (input === 'G') setPosition(maxOffset)
  })

  return (
    <Box
      flexDirection="column"
      width={width}
      height={dialogHeight}
      borderStyle="round"
      borderColor={colors.primary}
      paddingX={1}
    >
      <Text color={colors.primary} bold wrap="truncate-end">{title}</Text>
      <Text wrap="truncate-end">{lines.slice(offset, offset + pageSize).join('\n')}</Text>
      <Text color={colors.dimText} wrap="truncate-end">{maxOffset > 0 ? `${offset + 1}–${Math.min(lines.length, offset + pageSize)}/${lines.length} · ↑↓/PgUp/PgDn · ` : ''}enter/esc/q close</Text>
    </Box>
  )
}
