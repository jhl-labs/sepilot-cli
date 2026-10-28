import React from 'react'
import { Box, Text } from 'ink'
import { colors } from '../theme.js'
import type { ToolCallState } from '../types.js'
import { DiffView, UnifiedDiffView } from './DiffView.js'
import { TerminalOutput } from './TerminalOutput.js'
import {
  previewTextLines,
  rowToolMeta,
  summarizeToolInput,
  toolDisplayName,
  toolFilePath,
  truncatePreviewItems,
} from '../utils/tooling.js'
import { terminalFileLink } from '../utils/file-link.js'

interface ToolCallProps {
  tool: ToolCallState
  indent?: number
  compact?: boolean
  softenSupersededError?: boolean
  forceExpanded?: boolean
  maxRows?: number
}

const DEFAULT_TOOL_PREVIEW_LINES = 12
const SUPERSEDED_ERROR_META = 'Earlier attempt failed'
const TOOL_OUTPUT_PREVIEW_CHARS = 2000

function toolStatusPresentation(
  status: ToolCallState['status'],
  options?: {
    superseded?: boolean
    softenSupersededError?: boolean
  },
): {
  label: string
  color: string
} {
  if (
    status === 'error'
    && options?.superseded
    && options?.softenSupersededError
  ) {
    return { label: '[past]', color: colors.dimText }
  }

  switch (status) {
    case 'running':
      return { label: '[run ]', color: colors.warning }
    case 'pending':
      return { label: '[wait]', color: colors.pending }
    case 'success':
      return { label: '[ ok ]', color: colors.success }
    case 'error':
      return { label: '[err ]', color: colors.error }
  }
}

function buildToolHeader(
  tool: ToolCallState,
  softenSupersededError = false,
  collapsed = tool.collapsed,
  compact = false,
): string {
  const toggleLabel = collapsed || compact ? '▸' : '▾'
  const status = toolStatusPresentation(tool.status, {
    superseded: tool.superseded,
    softenSupersededError,
  })
  const parts = [toggleLabel, status.label, toolDisplayName(tool.name)]
  if (tool.repeatCount && tool.repeatCount > 1) parts.push(`×${tool.repeatCount}`)

  const meta = (
    softenSupersededError
    && tool.status === 'error'
    && tool.superseded
  )
    ? SUPERSEDED_ERROR_META
    : rowToolMeta(tool.meta, tool.status)

  const summary = linkedToolInputSummary(tool)
  if (summary) parts.push(summary)

  if (meta) {
    parts.push(meta)
  }

  return parts.join(' ')
}

function linkedToolInputSummary(tool: ToolCallState): string {
  const summary = summarizeToolInput(tool.input, tool.name)
  const path = toolFilePath(tool.input, tool.name)
  return path ? summary.replace(path, terminalFileLink(path)) : summary
}

function previewToolOutput(output: string): string {
  if (output.length <= TOOL_OUTPUT_PREVIEW_CHARS) {
    return output
  }

  return `${output.slice(0, TOOL_OUTPUT_PREVIEW_CHARS).trimEnd()}\n[tool output preview truncated in UI; full output is stored in the session]`
}

function FileReadPreview({ path, output, input }: { path?: string; output: string; input: Record<string, unknown> }) {
  const lines = previewTextLines(output)
  const preview = truncatePreviewItems(lines, DEFAULT_TOOL_PREVIEW_LINES)
  const offset = typeof input.offset === 'number' && input.offset > 0 ? input.offset : 1
  const numbered = input.lineNumbers !== false && lines.every((line, index) => {
    const match = /^\s*(\d+)\t/u.exec(line)
    return match && Number(match[1]) === offset + index
  })
  const lineNumberWidth = String(offset + Math.max(0, preview.visible.length - 1)).length

  return (
    <Box flexDirection="column">
      {path ? (
        <Text wrap="truncate-end">
          <Text color={colors.info} underline>file {terminalFileLink(path)}</Text>
          <Text color={colors.dimText}> · Ctrl+O full file</Text>
        </Text>
      ) : null}
      {preview.visible.map((line, index) => (
        <Text key={`${index}-${line.slice(0, 24)}`} wrap="truncate-end">
          <Text color={colors.dimText}>{String(offset + index).padStart(lineNumberWidth)} │ </Text>
          {(numbered ? line.replace(/^\s*\d+\t/u, '') : line) || ' '}
        </Text>
      ))}
      {preview.omitted > 0 ? (
        <Text color={colors.dimText}>
          … {preview.omitted} more line{preview.omitted === 1 ? '' : 's'} · Ctrl+O full file · Esc close
        </Text>
      ) : null}
    </Box>
  )
}

export const ToolCallView = React.memo(function ToolCallView({
  tool,
  indent = 2,
  compact = false,
  softenSupersededError = false,
  forceExpanded = false,
  maxRows,
}: ToolCallProps) {
  const status = toolStatusPresentation(tool.status, {
    superseded: tool.superseded,
    softenSupersededError,
  })
  const forceExpandedError = (
    tool.status === 'error'
    && !(softenSupersededError && tool.superseded)
  )
  const renderCompact = compact && !forceExpanded && tool.status !== 'error' && tool.status !== 'pending'
  const collapsed = tool.collapsed && !forceExpandedError && !forceExpanded
  const editOldText = tool.name === 'fs.edit' && typeof tool.input.oldText === 'string'
    ? tool.input.oldText
    : undefined
  const editNewText = tool.name === 'fs.edit' && typeof tool.input.newText === 'string'
    ? tool.input.newText
    : undefined
  const editDiff = tool.name === 'fs.edit' && typeof tool.editDiff === 'string'
    ? tool.editDiff
    : undefined
  const hasEditPreview = editOldText !== undefined && editNewText !== undefined
  const header = [buildToolHeader(tool, softenSupersededError, collapsed, renderCompact),
    renderCompact && tool.status === 'success' && (editDiff !== undefined || hasEditPreview) ? '· Ctrl+O changes' : '',
  ].filter(Boolean).join(' ')
  const showDetails = (!renderCompact && !collapsed)
    || (renderCompact && tool.status !== 'success' && (editDiff !== undefined || hasEditPreview))
  const path = toolFilePath(tool.input, tool.name) ?? undefined

  if (maxRows !== undefined) {
    const outputRows = Math.max(0, maxRows - 1)
    const output = tool.output || (tool.status === 'pending' ? tool.meta || 'Approval required' : '')
    const lines = output ? previewTextLines(previewToolOutput(output)) : []
    return (
      <Box flexDirection="column" marginLeft={indent} flexShrink={0}>
        <Text color={status.color} wrap="truncate-end">{buildToolHeader(tool, softenSupersededError, false, true)}</Text>
        {outputRows > 0 && lines.length > 0 ? (
          <Text color={tool.status === 'error' ? colors.error : colors.dimText} wrap="truncate-end">
            {lines.slice(-outputRows).join('\n')}
          </Text>
        ) : null}
      </Box>
    )
  }

  return (
    <Box flexDirection="column" marginLeft={indent} marginY={0}>
      <Text color={status.color} wrap={tool.status === 'error' ? 'wrap' : 'truncate-end'}>{header}</Text>
      {showDetails && (
        <Box flexDirection="column" marginLeft={4}>
          {editDiff !== undefined ? (
            <UnifiedDiffView diff={editDiff} />
          ) : hasEditPreview ? (
            <>
              <DiffView
                path={typeof tool.input.path === 'string' ? tool.input.path : undefined}
                previousContent={editOldText}
                nextContent={editNewText}
              />
              {tool.status === 'error' && tool.output ? (
                <Text color={colors.error} wrap="wrap">{previewToolOutput(tool.output)}</Text>
              ) : null}
            </>
          ) : tool.name === 'fs.write' && typeof tool.input.content === 'string' ? (
            <>
              {tool.status === 'error' && tool.output ? (
                <Text color={colors.error} wrap="wrap">{previewToolOutput(tool.output)}</Text>
              ) : null}
              <DiffView
                path={path}
                previousContent={tool.previousContent}
                nextContent={tool.input.content}
                maxLines={DEFAULT_TOOL_PREVIEW_LINES}
              />
            </>
          ) : tool.name === 'fs.read' && tool.output ? (
            tool.status === 'error' ? (
              <Text color={colors.error} wrap="wrap">{previewToolOutput(tool.output)}</Text>
            ) : (
              <FileReadPreview path={path} output={tool.output} input={tool.input} />
            )
          ) : tool.name === 'terminal.run' ? (
            <TerminalOutput tool={tool} maxLines={DEFAULT_TOOL_PREVIEW_LINES} />
          ) : (
            <>
              {tool.output && (
                <Text
                  color={tool.status === 'error' ? colors.error : colors.muted}
                  wrap={tool.status === 'error' ? 'wrap' : 'truncate-end'}
                >
                  {previewToolOutput(tool.output)}
                </Text>
              )}
            </>
          )}
        </Box>
      )}
    </Box>
  )
})
