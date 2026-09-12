import { Box, Text, useInput } from 'ink'
import { useEffect, useMemo, useState } from 'react'
import { colors } from '../../theme.js'

export interface FilePreviewDialogProps {
  path: string
  content: string
  width: number
  height: number
  onClose(): void
}

export function FilePreviewDialog({
  path,
  content,
  width,
  height,
  onClose,
}: FilePreviewDialogProps) {
  const lines = useMemo(() => content.split('\n'), [content])
  const visibleRows = Math.max(1, height - 4)
  const maxOffset = Math.max(0, lines.length - visibleRows)
  const [offset, setOffset] = useState(0)
  const lineNumberWidth = String(Math.max(1, lines.length)).length

  useEffect(() => {
    setOffset((current) => Math.min(current, maxOffset))
  }, [maxOffset])

  useInput((input, key) => {
    if (key.escape || (key.ctrl && input.toLowerCase() === 'o')) {
      onClose()
      return
    }
    if (key.upArrow) setOffset((current) => Math.max(0, current - 1))
    else if (key.downArrow) setOffset((current) => Math.min(maxOffset, current + 1))
    else if (key.pageUp) setOffset((current) => Math.max(0, current - visibleRows))
    else if (key.pageDown) setOffset((current) => Math.min(maxOffset, current + visibleRows))
    else if (input === 'g') setOffset(0)
    else if (input === 'G') setOffset(maxOffset)
  })

  return (
    <Box
      flexDirection="column"
      width={width}
      height={height}
      borderStyle="round"
      borderColor={colors.info}
      paddingX={1}
    >
      <Text color={colors.info} bold wrap="truncate-end">File · {path}</Text>
      {lines.slice(offset, offset + visibleRows).map((line, index) => (
        <Text key={`${offset + index}-${line.slice(0, 24)}`} wrap="truncate-end">
          <Text color={colors.dimText}>{String(offset + index + 1).padStart(lineNumberWidth)} │ </Text>
          {line || ' '}
        </Text>
      ))}
      <Text color={colors.dimText}>
        {`${offset + 1}-${Math.min(lines.length, offset + visibleRows)}/${lines.length} · ↑↓/PgUp/PgDn scroll · g/G ends · Esc/Ctrl+O close`}
      </Text>
    </Box>
  )
}
