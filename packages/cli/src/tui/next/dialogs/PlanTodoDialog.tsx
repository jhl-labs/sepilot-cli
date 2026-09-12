import type { PlannerWorkingMemory } from '@sepilotd/api-client'
import { Box, Text, useInput } from 'ink'
import { useEffect, useMemo, useState } from 'react'
import type { RunWorkProgress } from '../../types.js'
import { colors } from '../../theme.js'

interface WorkRow {
  key: string
  kind: 'section' | 'item' | 'detail'
  text: string
  status?: string
  depth?: number
}

export interface PlanTodoDialogProps {
  workingMemory?: PlannerWorkingMemory | null
  progress?: RunWorkProgress | null
  isStreaming: boolean
  width: number
  height: number
  onClose(): void
}

function flattenPlan(
  steps: PlannerWorkingMemory['plan'],
  depth = 0,
): Array<{ id: string; title: string; status: string; depth: number }> {
  return steps.flatMap((step) => [
    { id: step.id, title: step.title, status: step.status, depth },
    ...flattenPlan(step.children ?? [], depth + 1),
  ])
}

function statusVisual(status: string | undefined): { glyph: string; color: string } {
  if (status === 'done' || status === 'completed') return { glyph: '●', color: colors.success }
  if (status === 'in_progress' || status === 'running') return { glyph: '◐', color: colors.info }
  if (status === 'blocked' || status === 'failed') return { glyph: '!', color: colors.error }
  if (status === 'skipped' || status === 'cancelled') return { glyph: '⊘', color: colors.dimText }
  return { glyph: '○', color: colors.pending }
}

export function PlanTodoDialog({
  workingMemory,
  progress,
  isStreaming,
  width,
  height,
  onClose,
}: PlanTodoDialogProps) {
  const rows = useMemo<WorkRow[]>(() => {
    const storedPlan = flattenPlan(workingMemory?.plan ?? [])
    const plan = progress?.plan?.length
      ? progress.plan.map((step, index) => ({ ...step, id: `live-${index}` }))
      : storedPlan
    const todos = progress?.todos ?? []
    const next: WorkRow[] = []

    if (workingMemory?.taskSummary) {
      next.push({ key: 'task', kind: 'detail', text: `Task · ${workingMemory.taskSummary}` })
    }
    if (workingMemory?.currentStepRationale) {
      next.push({ key: 'rationale', kind: 'detail', text: `Why now · ${workingMemory.currentStepRationale}` })
    }
    next.push({ key: 'plan-section', kind: 'section', text: 'Plan' })
    if (plan.length === 0) {
      next.push({ key: 'plan-empty', kind: 'detail', text: '  No structured plan has been emitted.' })
    } else {
      for (const step of plan) {
        next.push({
          key: `plan-${step.id}`,
          kind: 'item',
          text: step.title,
          status: step.status,
          depth: step.depth,
        })
      }
    }
    next.push({ key: 'todo-section', kind: 'section', text: 'Todo' })
    if (todos.length === 0) {
      next.push({ key: 'todo-empty', kind: 'detail', text: '  No structured todo items have been emitted.' })
    } else {
      todos.forEach((todo, index) => next.push({
        key: `todo-${index}`,
        kind: 'item',
        text: todo.content,
        status: todo.status,
        depth: 0,
      }))
    }
    return next
  }, [progress, workingMemory])

  const visibleRows = Math.max(1, height - 5)
  const maxOffset = Math.max(0, rows.length - visibleRows)
  const [offset, setOffset] = useState(0)

  useEffect(() => {
    setOffset((current) => Math.min(current, maxOffset))
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

  const flattened = flattenPlan(workingMemory?.plan ?? [])
  const planTotal = progress?.planTotal ?? flattened.length
  const planDone = progress?.planDone ?? flattened.filter((step) => step.status === 'done').length
  const todosTotal = progress?.todosTotal ?? 0
  const todosDone = progress?.todosDone ?? 0
  const criteriaTotal = progress?.criteriaTotal ?? 0

  return (
    <Box
      flexDirection="column"
      width={width}
      height={height}
      borderStyle="round"
      borderColor={colors.primary}
      paddingX={1}
    >
      <Text color={colors.primary} bold>Plan &amp; todo · {isStreaming ? 'live' : 'latest snapshot'}</Text>
      <Text color={colors.dimText}>
        criteria {criteriaTotal} · plan {planDone}/{planTotal} · todos {todosDone}/{todosTotal}
      </Text>
      {rows.slice(offset, offset + visibleRows).map((row) => {
        if (row.kind === 'section') {
          return <Text key={row.key} color={colors.primary} bold>{row.text}</Text>
        }
        if (row.kind === 'detail') {
          return <Text key={row.key} color={colors.dimText} wrap="truncate-end">{row.text}</Text>
        }
        const visual = statusVisual(row.status)
        return (
          <Text key={row.key} wrap="truncate-end">
            {'  '.repeat(row.depth ?? 0)}
            <Text color={visual.color}>{visual.glyph} </Text>
            <Text bold={row.status === 'in_progress'}>{row.text}</Text>
            <Text color={colors.dimText}> · {row.status ?? 'pending'}</Text>
          </Text>
        )
      })}
      <Text color={colors.dimText}>
        {`${offset + 1}-${Math.min(rows.length, offset + visibleRows)}/${rows.length} · ↑↓/PgUp/PgDn · g/G ends · Esc/q/F11 close`}
      </Text>
    </Box>
  )
}
