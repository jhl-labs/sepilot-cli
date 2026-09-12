import { Box, Text } from 'ink'
import type {
  PlannerHierarchicalStep,
  PlannerStepStatus,
  PlannerWorkingMemory,
} from '@sepilotd/api-client'

interface PlannerProgressPanelProps {
  workingMemory: PlannerWorkingMemory
  height?: number
  maxRows?: number
}

const STATUS_GLYPH: Record<PlannerStepStatus, string> = {
  pending: '○',
  in_progress: '◐',
  done: '●',
  blocked: '!',
  skipped: '⊘',
}

const STATUS_COLOR: Record<PlannerStepStatus, string> = {
  pending: 'gray',
  in_progress: 'cyan',
  done: 'green',
  blocked: 'red',
  skipped: 'yellow',
}

interface FlatRow {
  key: string
  glyph: string
  color: string
  indent: number
  title: string
  detail?: string
  highlight: boolean
}

function flatten(
  steps: PlannerHierarchicalStep[] | undefined,
  currentSubtaskId: string | undefined,
  depth = 0,
  rows: FlatRow[] = [],
): FlatRow[] {
  for (const step of steps ?? []) {
    rows.push({
      key: step.id,
      glyph: STATUS_GLYPH[step.status],
      color: STATUS_COLOR[step.status],
      indent: depth,
      title: step.title,
      detail: step.detail,
      highlight: step.id === currentSubtaskId,
    })
    if (step.children && step.children.length > 0) {
      flatten(step.children, currentSubtaskId, depth + 1, rows)
    }
  }
  return rows
}

export function PlannerProgressPanel({
  workingMemory,
  height,
  maxRows = 12,
}: PlannerProgressPanelProps) {
  const rows = flatten(workingMemory.plan, workingMemory.currentSubtaskId).slice(0, maxRows)
  const overflow =
    flatten(workingMemory.plan, workingMemory.currentSubtaskId).length - rows.length
  const risks = workingMemory.risks ?? []
  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor="cyan"
      paddingX={1}
      height={height}
      width="100%"
      minWidth={0}
    >
      <Text color="cyan" bold>
        Planner
      </Text>
      {workingMemory.taskSummary
        ? <Text wrap="truncate-end">{workingMemory.taskSummary}</Text>
        : null}
      {rows.length === 0
        ? <Text color="gray">No plan yet.</Text>
        : rows.map((row) => (
            <Box key={row.key} width="100%" minWidth={0}>
              <Text>{' '.repeat(row.indent * 2)}</Text>
              <Text color={row.color} bold={row.highlight}>
                {row.glyph}{' '}
              </Text>
              <Box flexGrow={1} flexShrink={1} minWidth={0}>
                <Text bold={row.highlight} wrap="truncate-end">
                  {row.title}
                </Text>
              </Box>
            </Box>
          ))}
      {overflow > 0
        ? <Text color="gray">+{overflow} more step{overflow === 1 ? '' : 's'}</Text>
        : null}
      {risks.length > 0
        ? (
            <Box flexDirection="column" marginTop={1} width="100%" minWidth={0}>
              <Text color="red" bold>Risks</Text>
              {risks.slice(0, 3).map((risk, i) => (
                <Text key={`r-${i}`} color={risk.severity === 'high' ? 'red' : 'yellow'} wrap="truncate-end">
                  [{risk.severity}] {risk.text}
                </Text>
              ))}
            </Box>
          )
        : null}
    </Box>
  )
}
