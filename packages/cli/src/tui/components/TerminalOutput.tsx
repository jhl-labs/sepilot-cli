import { Box, Text } from 'ink'
import { colors } from '../theme.js'
import {
  previewTextLines,
  splitTerminalOutput,
  terminalCommandLabel,
  toolExecutionPostureMeta,
  truncatePreviewItems,
} from '../utils/tooling.js'
import type { ToolCallState } from '../types.js'

interface TerminalOutputProps {
  tool: ToolCallState
  maxLines?: number
}

interface TerminalPreviewLine {
  key: string
  text: string
  color?: string
}

const COMMAND_PREVIEW_CHARS = 320

function boundedCommandText(command: string): string {
  return command.length <= COMMAND_PREVIEW_CHARS
    ? command
    : `${command.slice(0, COMMAND_PREVIEW_CHARS).trimEnd()}… (+${command.length - COMMAND_PREVIEW_CHARS} chars)`
}

export function TerminalOutput({ tool, maxLines }: TerminalOutputProps) {
  const { stdout, stderr } = splitTerminalOutput(tool.output)
  const cwd = typeof tool.input.cwd === 'string' ? tool.input.cwd : undefined
  const posture = toolExecutionPostureMeta(tool.meta)
  const timeoutMs = typeof tool.input.timeoutMs === 'number'
    ? tool.input.timeoutMs
    : undefined
  const previewLines: TerminalPreviewLine[] = stdout
    ? previewTextLines(stdout).map((line, index) => ({
        key: `stdout-${index}`,
        text: line || ' ',
      }))
    : [{
        key: 'stdout-empty',
        text: 'No stdout',
        color: colors.dimText,
      }]

  if (stderr) {
    previewLines.push({
      key: 'stderr-header',
      text: 'stderr',
      color: colors.error,
    })
    previewLines.push(
      ...previewTextLines(stderr).map((line, index) => ({
        key: `stderr-${index}`,
        text: line || ' ',
        color: colors.error,
      })),
    )
  }

  const visiblePreview = maxLines == null
    ? { visible: previewLines, omitted: 0 }
    : truncatePreviewItems(previewLines, maxLines)

  return (
    <Box flexDirection="column">
      <Text color={colors.info}>
        $ {boundedCommandText(terminalCommandLabel(tool.input))}
      </Text>
      {cwd && (
        <Text color={colors.dimText} wrap="truncate-end">cwd {cwd}</Text>
      )}
      {posture && (
        <Text color={colors.dimText} wrap="truncate-end">{posture}</Text>
      )}
      {timeoutMs && (
        <Text color={colors.dimText} wrap="truncate-end">timeout {timeoutMs}ms</Text>
      )}
      {visiblePreview.visible.map((line) => (
        <Text key={line.key} color={line.color} wrap="truncate-end">
          {line.text}
        </Text>
      ))}
      {visiblePreview.omitted > 0 && (
        <Text color={colors.dimText} wrap="truncate-end">
          ... {visiblePreview.omitted} more output line{visiblePreview.omitted === 1 ? '' : 's'}
        </Text>
      )}
      <Text
        color={
          tool.status === 'success'
            ? colors.success
            : tool.status === 'pending'
              ? colors.pending
            : tool.status === 'running'
              ? colors.info
              : colors.error
        }
        wrap="truncate-end"
      >
        {tool.status === 'success'
          ? 'exit 0'
          : tool.status === 'pending'
            ? 'awaiting approval'
            : tool.status === 'running'
              ? 'running'
              : 'command failed'}
      </Text>
    </Box>
  )
}
