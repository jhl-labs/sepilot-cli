import { Box, Text, useStdout } from 'ink'
import stringWidth from 'string-width'
import { colors } from '../theme.js'
import type { StatusBarPresentationState } from '../utils/status-bar-state.js'
import { clearLineTail } from '../utils/terminal-line.js'

interface StatusBarProps {
  presentation: StatusBarPresentationState
  width?: number
}

const DOT = '\u00B7'
const SEPARATOR = `  ${DOT}  `
const FIT_SAFETY_COLUMNS = 2

interface StatusSegment {
  key: string
  text: string
  dropPriority?: number
}

function measureSegments(segments: StatusSegment[]): number {
  return segments.reduce((total, segment, index) => (
    total
    + stringWidth(segment.text)
    + (index > 0 ? stringWidth(SEPARATOR) : 0)
  ), 0)
}

function fitSegments(
  segments: StatusSegment[],
  availableWidth?: number,
): StatusSegment[] {
  if (availableWidth == null || availableWidth <= 0) {
    return segments
  }

  let visible = segments
  for (const priority of [1, 2, 3]) {
    if (measureSegments(visible) <= Math.max(0, availableWidth - FIT_SAFETY_COLUMNS)) {
      break
    }
    visible = visible.filter((segment) => segment.dropPriority !== priority)
  }

  return visible
}

function rightStatusWidth(
  text: string,
  barWidth: number | undefined,
  coreWidth: number,
): number {
  const textWidth = stringWidth(text)
  if (barWidth == null) {
    return textWidth
  }

  return Math.max(
    0,
    Math.min(
      textWidth,
      32,
      barWidth - coreWidth - stringWidth(SEPARATOR),
    ),
  )
}

function padToWidth(text: string, width: number | string): string {
  if (typeof width !== 'number' || width <= 0) {
    return text
  }
  const used = stringWidth(text)
  if (used >= width) {
    return text
  }
  return text + ' '.repeat(width - used)
}

export function StatusBar({
  presentation,
  width,
}: StatusBarProps) {
  const { stdout } = useStdout()
  const barWidth = width ?? stdout.columns
  const segments: StatusSegment[] = [
    {
      key: 'session',
      text: `session ${presentation.sessionLabel}`,
    },
    {
      key: 'mode',
      text: presentation.modeLabel,
    },
    ...(presentation.modelLabel
      ? [{
          key: 'model',
          text: presentation.modelLabel,
          dropPriority: 1,
        }]
      : []),
    ...(presentation.showPlanBadge
      ? [{
          key: 'plan',
          text: '[plan]',
          dropPriority: 3,
        }]
      : []),
    {
      key: 'autonomy',
      text: presentation.autonomyLabel,
    },
    ...(presentation.contextPercent != null
      ? [{
          key: 'context',
          text: `ctx ${presentation.contextEstimated ? '~' : ''}${presentation.contextPercent}%`,
          dropPriority: 2,
        }]
      : []),
    {
      key: 'cost',
      text: presentation.costLabel,
      dropPriority: 2,
    },
    ...(presentation.maxOutputLabel
      ? [{
          key: 'max-output',
          text: presentation.maxOutputLabel,
          dropPriority: 1,
        }]
      : []),
    ...(presentation.phaseLabel
      ? [{
          key: 'phase',
          text: presentation.phaseLabel,
          dropPriority: 2,
        }]
      : []),
    ...(presentation.criteriaLabel
      ? [{
          key: 'criteria',
          text: presentation.criteriaLabel,
          dropPriority: 3,
        }]
      : []),
    ...(presentation.todosLabel
      ? [{
          key: 'todos',
          text: presentation.todosLabel,
          dropPriority: 3,
        }]
      : []),
    ...(presentation.phaseLabel || presentation.criteriaLabel || presentation.todosLabel
      ? [{
          key: 'state-hint',
          text: '/state for details',
          dropPriority: 1,
        }]
      : []),
  ]
  const coreWidth = measureSegments(segments.filter((segment) => segment.dropPriority == null))
  const rightWidth = presentation.rightStatus
    ? rightStatusWidth(presentation.rightStatus.text, barWidth, coreWidth)
    : 0
  const showRightStatus = Boolean(
    presentation.rightStatus
    && (barWidth == null || rightWidth >= Math.min(8, stringWidth(presentation.rightStatus.text))),
  )
  const leftAvailableWidth = barWidth == null
    ? undefined
    : Math.max(0, barWidth - (showRightStatus ? rightWidth + 2 : 0))
  const visibleSegments = fitSegments(segments, leftAvailableWidth)
  const leftText = padToWidth(
    visibleSegments.map((segment) => segment.text).join(SEPARATOR),
    leftAvailableWidth ?? '100%',
  )
  const leftWidth = leftAvailableWidth ?? '100%'

  return (
    <Box
      width={barWidth ?? '100%'}
      justifyContent={showRightStatus ? 'space-between' : 'flex-start'}
      height={1}
      overflow="hidden"
    >
      <Box width={leftWidth} flexShrink={0} minWidth={0} overflow="hidden">
        <Text color={colors.muted} wrap="end">{showRightStatus ? leftText : clearLineTail(leftText)}</Text>
      </Box>
      {showRightStatus && presentation.rightStatus && (
        <Box flexShrink={0} width={rightWidth} marginLeft={2} overflow="hidden">
          <Text
            color={presentation.rightStatus.color}
            bold={presentation.rightStatus.bold}
            wrap="truncate-end"
          >
            {clearLineTail(presentation.rightStatus.text)}
          </Text>
        </Box>
      )}
    </Box>
  )
}
