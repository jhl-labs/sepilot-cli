import { Box, Text, useInput } from 'ink'
import { useMemo, useState } from 'react'
import wrapAnsi from 'wrap-ansi'
import type { Message } from '../../types.js'
import { colors } from '../../theme.js'

export function ActivityDialog({ reasoning, messages, isStreaming, width, height, onClose }: {
  reasoning: string
  messages: Message[]
  isStreaming: boolean
  width: number
  height: number
  onClose(): void
}) {
  const [tab, setTab] = useState<'reasoning' | 'activity'>(() => reasoning.trim() ? 'reasoning' : 'activity')
  const [position, setPosition] = useState<number | null>(null)
  const content = useMemo(() => {
    if (tab === 'reasoning') return reasoning || 'No reasoning text received. Tab opens execution activity.'
    const start = messages.findLastIndex(({ role }) => role === 'user')
    return messages.slice(Math.max(0, start)).map((message) => {
      const tool = message.toolCall
      return tool
        ? `[${tool.status}] ${tool.name}\n${tool.arguments}\n${tool.output ?? ''}`
        : `${message.role} · ${message.content}`
    }).join('\n\n') || 'No activity received for this turn.'
  }, [tab, reasoning, messages])
  const lines = useMemo(() => wrapAnsi(content, Math.max(1, width - 4), { hard: true, trim: false }).split('\n'), [content, width])
  const pageSize = Math.max(1, height - 5)
  const maxOffset = Math.max(0, lines.length - pageSize)
  const offset = position === null ? maxOffset : Math.min(position, maxOffset)
  useInput((input, key) => {
    if (key.escape || (key.ctrl && input.toLowerCase() === 'o')) { onClose(); return }
    if (key.tab) { setTab(tab === 'reasoning' ? 'activity' : 'reasoning'); setPosition(null); return }
    if (key.upArrow) setPosition(Math.max(0, offset - 1))
    else if (key.downArrow) setPosition(offset + 1 >= maxOffset ? null : offset + 1)
    else if (key.pageUp) setPosition(Math.max(0, offset - pageSize))
    else if (key.pageDown) setPosition(offset + pageSize >= maxOffset ? null : offset + pageSize)
    else if (input === 'g') setPosition(0)
    else if (input === 'G') setPosition(null)
  })
  return (
    <Box flexDirection="column" width={width} height={height} borderStyle="round" borderColor={colors.info} paddingX={1}>
      <Text bold color={colors.info} wrap="truncate-end">{tab === 'reasoning' ? 'Thinking' : 'Execution activity'} · {isStreaming ? 'live' : 'latest turn'} · Tab switch</Text>
      <Text color={colors.dimText} wrap="truncate-end">{offset + 1}–{Math.min(lines.length, offset + pageSize)}/{lines.length} · {position === null ? 'following latest' : 'scroll paused · G resumes'}</Text>
      {lines.slice(offset, offset + pageSize).map((line, index) => <Text key={index} wrap="truncate-end">{line || ' '}</Text>)}
      <Text color={colors.dimText} wrap="truncate-end">↑↓/PgUp/PgDn · g/G ends · Ctrl+O/Esc close</Text>
    </Box>
  )
}
