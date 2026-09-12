import type { ActivityItem } from '@sepilotd/api-client'
import { Box, Text } from 'ink'
import { colors } from '../theme.js'
import type { ToolCallState } from '../types.js'
import { ToolCallView } from './ToolCall.js'
import { buildToolActivitySummary } from '../utils/tool-activity-summary.js'
import { clearLineTail } from '../utils/terminal-line.js'

interface ToolActivityPanelProps {
  activities?: ActivityItem[]
  toolCalls: ToolCallState[]
  height?: number
  maxVisibleItems?: number
  runActive?: boolean
}

function activityStatusPresentation(
  status: ActivityItem['status'],
): {
  label: string
  color: string
} {
  switch (status) {
    case 'running':
      return { label: '[run ]', color: colors.warning }
    case 'pending':
      return { label: '[wait]', color: colors.pending }
    case 'success':
      return { label: '[ ok ]', color: colors.success }
    case 'error':
      return { label: '[err ]', color: colors.error }
    default:
      return { label: '[note]', color: colors.info }
  }
}

export function ToolActivityPanel({
  activities = [],
  toolCalls,
  height,
  maxVisibleItems = 3,
  runActive = false,
}: ToolActivityPanelProps) {
  const showingActivities = activities.length > 0
  const linesPerItem = showingActivities ? 2 : 1
  const summaryLine = buildToolActivitySummary({
    activities,
    toolCalls,
    runActive,
  })
  const showSummaryLine = summaryLine.length > 0 && (height == null || height >= 5)
  const headerRows = showSummaryLine ? 2 : 1
  const visibleLimit = height == null
    ? maxVisibleItems
    : Math.max(1, Math.min(maxVisibleItems, Math.floor((height - headerRows) / linesPerItem) || 1))
  const tailActivities = activities.slice(-visibleLimit)
  const latestProgressActivity = [...activities]
    .reverse()
    .find((activity) => activity.label === 'Progress update')
  const visibleActivities = (
    latestProgressActivity
    && visibleLimit === 1
    && !runActive
  )
    ? [latestProgressActivity]
    : (
        latestProgressActivity
        && visibleLimit > 1
        && !tailActivities.some((activity) => activity.id === latestProgressActivity.id)
      )
        ? activities
            .filter((activity) => (
              activity.id === latestProgressActivity.id
              || tailActivities.slice(-(visibleLimit - 1)).some((candidate) => candidate.id === activity.id)
            ))
            .slice(-visibleLimit)
        : tailActivities
  const visibleToolCalls = toolCalls.slice(-visibleLimit)
  const omitted = showingActivities
    ? Math.max(0, activities.length - visibleActivities.length)
    : Math.max(0, toolCalls.length - visibleToolCalls.length)

  if (
    (showingActivities ? visibleActivities.length === 0 : visibleToolCalls.length === 0)
    || (height != null && height < headerRows + 1)
  ) {
    return null
  }

  return (
    <Box
      flexDirection="column"
      height={height}
      borderStyle="single"
      borderColor={runActive ? colors.warning : colors.border}
      borderTop
      borderBottom={false}
      borderLeft={false}
      borderRight={false}
      marginBottom={1}
      overflow="hidden"
      width="100%"
      minWidth={0}
    >
      <Text color={runActive ? colors.warning : colors.dimText} bold={runActive}>
        {clearLineTail(`Activity${
          omitted > 0
            ? ` · showing last ${showingActivities ? visibleActivities.length : visibleToolCalls.length} of ${showingActivities ? activities.length : toolCalls.length}`
            : ''
        }`)}
      </Text>
      {showSummaryLine ? (
        <Text color={colors.dimText} wrap="truncate-end">
          {clearLineTail(summaryLine)}
        </Text>
      ) : null}
      {showingActivities
        ? visibleActivities.map((activity) => {
            const status = activityStatusPresentation(activity.status)
            return (
              <Box key={activity.id} flexDirection="column" marginY={0} width="100%" minWidth={0}>
                <Text color={status.color} wrap="wrap">
                  {clearLineTail(`${status.label} ${activity.label}${activity.meta ? ` ${activity.meta}` : ''}`)}
                </Text>
                <Text color={colors.dimText} wrap="wrap">
                  {clearLineTail(activity.detail)}
                </Text>
              </Box>
            )
          })
        : visibleToolCalls.map((tool) => (
            <ToolCallView
              key={tool.id}
              tool={tool}
              indent={0}
              compact
              softenSupersededError
            />
          ))}
    </Box>
  )
}
