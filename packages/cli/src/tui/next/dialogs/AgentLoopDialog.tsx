import { Box, Text, useInput } from 'ink'
import { useEffect, useMemo, useState } from 'react'
import type { GraphTraceEntry } from '../../types.js'
import { colors } from '../../theme.js'

export interface AgentLoopDialogProps {
  entries: GraphTraceEntry[]
  currentPhase?: string | null
  isStreaming: boolean
  streamStatus?: string | null
  width: number
  height: number
  onClose(): void
}

function durationLabel(durationMs: number | undefined): string {
  if (durationMs === undefined) return ''
  if (durationMs < 1000) return `${Math.round(durationMs)}ms`
  return `${(durationMs / 1000).toFixed(durationMs < 10_000 ? 1 : 0)}s`
}

function statusVisual(status: GraphTraceEntry['status']): {
  glyph: string
  color: string
} {
  if (status === 'completed') return { glyph: '●', color: colors.success }
  if (status === 'error') return { glyph: '!', color: colors.error }
  if (status === 'interrupted') return { glyph: '⊘', color: colors.dimText }
  return { glyph: '◐', color: colors.info }
}

export function AgentLoopDialog({
  entries,
  currentPhase,
  isStreaming,
  streamStatus,
  width,
  height,
  onClose,
}: AgentLoopDialogProps) {
  const visibleRows = Math.max(1, height - 5)
  const maxOffset = Math.max(0, entries.length - visibleRows)
  const [offset, setOffset] = useState(maxOffset)

  useEffect(() => {
    setOffset(maxOffset)
  }, [maxOffset])

  useInput((input, key) => {
    if (key.escape || input === 'q') {
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

  const active = useMemo(
    () => [...entries].reverse().find((entry) => entry.status === 'running'),
    [entries],
  )
  const phase = active?.phase ?? currentPhase ?? 'unassigned'
  const runLabel = active
    ? `Current · ${active.node} · ${active.lifecycleState ?? 'running'}${active.iteration === undefined ? '' : ` · iteration ${active.iteration}`}`
    : isStreaming
      ? `Waiting for graph node · ${streamStatus ?? 'running'}`
      : 'Run idle · showing the latest actual traversal'

  return (
    <Box
      flexDirection="column"
      width={width}
      height={height}
      borderStyle="round"
      borderColor={colors.info}
      paddingX={1}
    >
      <Text color={colors.info} bold>Agent loop · phase {phase}</Text>
      <Text color={active ? colors.info : colors.dimText} wrap="truncate-end">{runLabel}</Text>
      {entries.length === 0 ? (
        <Text color={colors.dimText}>No graph traversal has been emitted for this run yet.</Text>
      ) : entries.slice(offset, offset + visibleRows).map((entry, index) => {
        const visual = statusVisual(entry.status)
        const duration = durationLabel(entry.durationMs)
        return (
          <Text key={`${offset + index}-${entry.node}-${entry.status}`} wrap="truncate-end">
            <Text color={visual.color}>{visual.glyph} </Text>
            <Text bold={entry.status === 'running'}>{entry.node}</Text>
            <Text color={colors.dimText}>
              {`${entry.phase ? ` · ${entry.phase}` : ''}${entry.iteration === undefined ? '' : ` · #${entry.iteration}`}${duration ? ` · ${duration}` : ''}${entry.nextEdge ? ` → ${entry.nextEdge}` : ''}`}
            </Text>
          </Text>
        )
      })}
      <Text color={colors.dimText}>
        {entries.length > 0 ? `${offset + 1}-${Math.min(entries.length, offset + visibleRows)}/${entries.length} · ` : ''}actual visited nodes · ↑↓/PgUp/PgDn · g/G ends · Esc/q/F12 close
      </Text>
    </Box>
  )
}
