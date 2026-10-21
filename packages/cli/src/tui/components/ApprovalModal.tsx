import { useEffect, useRef, useState, type ReactElement } from 'react'
import { Box, Text, useInput } from 'ink'
import type { ApprovalRequest } from '../types.js'
import { colors, symbols } from '../theme.js'
import { ControlSafeTextInput } from './ControlSafeTextInput.js'
import { DiffView, UnifiedDiffView } from './DiffView.js'
import { TerminalOutput } from './TerminalOutput.js'
import {
  buildStoredToolCallState,
  formatToolInput,
  previewTextLines,
  terminalCommandLabel,
  truncatePreviewItems,
} from '../utils/tooling.js'
import { isReturnKey } from '../utils/key.js'

interface ApprovalModalProps {
  approval: ApprovalRequest
  height?: number
  inputActive?: boolean
  compact?: boolean
  onResolve?: (approved: boolean, scope: ApprovalActionScope, note?: string, interrupt?: boolean) => void
  onCommentModeChange?: (active: boolean) => void
  onShowFull?: (text: string) => void
}

interface ModalRow {
  key: string
  priority: number
  element: ReactElement
}

type ApprovalActionScope = 'once' | 'session' | 'always' | 'run'

interface ApprovalAction {
  id: string
  shortcut: string
  label: string
  approved: boolean
  scope: ApprovalActionScope
}

const APPROVAL_ACTIONS: ApprovalAction[] = [
  { id: 'once', shortcut: 'y', label: 'once', approved: true, scope: 'once' },
  { id: 'session', shortcut: 's', label: 'session', approved: true, scope: 'session' },
  { id: 'run', shortcut: 'r', label: 'run', approved: true, scope: 'run' },
  { id: 'always', shortcut: 'a', label: 'always', approved: true, scope: 'always' },
  { id: 'deny', shortcut: 'n', label: 'deny', approved: false, scope: 'once' },
]

const COMMENT_SHORTCUT = 'm'
const VIEW_FULL_SHORTCUT = 'v'
const UNKNOWN_KEY_HINT_MS = 2500
const KNOWN_SHORTCUTS = new Set([
  ...APPROVAL_ACTIONS.map((action) => action.shortcut),
  COMMENT_SHORTCUT,
  VIEW_FULL_SHORTCUT,
])
const COMPACT_COMMAND_LINES = 8
const COMPACT_DIFF_LINES = 20
const COMPACT_INPUT_LINES = 10

export function approvalReviewText(approval: ApprovalRequest): string {
  if (approval.toolName === 'terminal.run') {
    const input = approval.input
    const executable = typeof input.executable === 'string' ? input.executable : ''
    const args = Array.isArray(input.args) ? input.args.map(String) : []
    const shellScriptIndex = ['sh', 'bash', 'zsh'].includes(executable.split('/').pop() ?? '')
      ? args.findIndex((arg) => arg === '-c' || arg === '-lc')
      : -1
    const header = typeof input.cwd === 'string' && input.cwd ? [`# cwd: ${input.cwd}`] : []
    if (shellScriptIndex >= 0 && args[shellScriptIndex + 1] !== undefined) {
      const script = args[shellScriptIndex + 1]!
      const lines = script.split('\n').flatMap((line) => line.split(/;\s+(?=\S)/).map((part, index, all) => index < all.length - 1 ? `${part};` : part))
      return [...header, `$ ${executable} ${args.slice(0, shellScriptIndex + 1).join(' ')}`, ...lines.map((line) => `  ${line}`)].join('\n')
    }
    return [...header, `$ ${terminalCommandLabel(input)}`].join('\n')
  }
  if (approval.previewDiff) return approval.previewDiff
  return formatToolInput(approval.input)
}

function boundedLines(text: string, limit: number): { lines: string[]; hidden: number } {
  const lines = text.split('\n')
  return lines.length <= limit ? { lines, hidden: 0 } : { lines: lines.slice(0, limit), hidden: lines.length - limit }
}

export function ApprovalModal({
  approval,
  height,
  inputActive = true,
  compact = false,
  onResolve,
  onCommentModeChange,
  onShowFull,
}: ApprovalModalProps) {
  const [selectedActionIndex, setSelectedActionIndex] = useState(0)
  const [commentMode, setCommentMode] = useState(false)
  const [commentDraft, setCommentDraft] = useState('')
  const [unknownKeyHint, setUnknownKeyHint] = useState<string | null>(null)
  const hintTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const clampedActionIndex = Math.max(0, Math.min(selectedActionIndex, APPROVAL_ACTIONS.length - 1))

  const setCommentModeNotified = (active: boolean) => {
    setCommentMode(active)
    onCommentModeChange?.(active)
  }

  useEffect(() => {
    setSelectedActionIndex(0)
    setCommentDraft('')
    setUnknownKeyHint(null)
    setCommentModeNotified(false)
  }, [approval.requestId])

  useEffect(
    () => () => {
      if (hintTimerRef.current) clearTimeout(hintTimerRef.current)
      onCommentModeChange?.(false)
    },
    [],
  )

  const flashUnknownKey = (input: string) => {
    const shown = input.trim() ? `'${input}'` : 'that key'
    setUnknownKeyHint(
      `${shown} is not a shortcut — y approve · n deny · m message · s/r/a scopes`,
    )
    if (hintTimerRef.current) clearTimeout(hintTimerRef.current)
    hintTimerRef.current = setTimeout(() => setUnknownKeyHint(null), UNKNOWN_KEY_HINT_MS)
  }

  useInput(
    (input, key) => {
      if (!onResolve) return
      const cancelApproval = key.escape
        || (key.ctrl && input.toLowerCase() === 'c')
        || input === '\u0003'
      if (commentMode) {
        if (key.escape) {
          setCommentDraft('')
          setCommentModeNotified(false)
        } else if (cancelApproval) {
          setCommentDraft('')
          setCommentModeNotified(false)
          onResolve(false, 'once', undefined, true)
        }
        return
      }

      if (cancelApproval) {
        onResolve(false, 'once', undefined, true)
        return
      }

      if (key.upArrow || key.leftArrow) {
        setUnknownKeyHint(null)
        setSelectedActionIndex(Math.max(0, clampedActionIndex - 1))
        return
      }

      if (key.downArrow || key.rightArrow) {
        setUnknownKeyHint(null)
        setSelectedActionIndex(Math.min(APPROVAL_ACTIONS.length - 1, clampedActionIndex + 1))
        return
      }

      const normalized = input.toLowerCase()
      if (normalized === VIEW_FULL_SHORTCUT) {
        setUnknownKeyHint(null)
        onShowFull?.(approvalReviewText(approval))
        return
      }
      if (normalized === COMMENT_SHORTCUT) {
        setUnknownKeyHint(null)
        setCommentModeNotified(true)
        return
      }

      const shortcutAction = APPROVAL_ACTIONS.find((action) => normalized === action.shortcut)
      if (shortcutAction) {
        onResolve(shortcutAction.approved, shortcutAction.scope)
        return
      }

      if (isReturnKey(input, key)) {
        const action = APPROVAL_ACTIONS[clampedActionIndex]
        onResolve(action.approved, action.scope)
        return
      }

      if (input && !key.ctrl && !key.meta && !KNOWN_SHORTCUTS.has(normalized)) {
        flashUnknownKey(input)
      }
    },
    { isActive: inputActive && Boolean(onResolve) },
  )

  const submitComment = (value: string) => {
    const note = value.trim()
    setCommentDraft('')
    setCommentModeNotified(false)
    if (!note) return
    onResolve?.(false, 'once', note)
  }

  const tool = buildStoredToolCallState({
    id: approval.toolCallId,
    name: approval.toolName,
    input: approval.input,
    status: 'pending',
    meta:
      approval.state === 'stale'
        ? approval.resumeAvailable
          ? 'Saved checkpoint available for resume'
          : 'Fresh recovery run will be required'
        : 'Run is paused for confirmation',
  })

  const path = typeof approval.input.path === 'string' ? approval.input.path : undefined
  const content = typeof approval.input.content === 'string' ? approval.input.content : undefined
  const oldText = typeof approval.input.oldText === 'string' ? approval.input.oldText : undefined
  const newText = typeof approval.input.newText === 'string' ? approval.input.newText : undefined
  const metaBits = [
    `request ${approval.requestId.slice(0, 8)}`,
    approval.state,
    approval.resumeAvailable ? 'resume available' : 'no checkpoint',
  ]
  const serializedInput = formatToolInput(approval.input)

  const availableRows = height == null ? Number.POSITIVE_INFINITY : Math.max(1, height - 2)

  const structuralRows: ModalRow[] = []
  structuralRows.push({
    key: 'action',
    priority: 1,
    element: commentMode ? (
      <Box key="action" minWidth={0} width="100%">
        <Text color={colors.primary} bold>
          message{' '}
        </Text>
        <ControlSafeTextInput
          value={commentDraft}
          placeholder="tell the agent what to do instead…"
          focus={inputActive}
          onChange={setCommentDraft}
          onSubmit={submitComment}
        />
      </Box>
    ) : (
      <Box key="action" gap={1} minWidth={0} width="100%">
        {APPROVAL_ACTIONS.map((action, index) => {
          const selected = index === clampedActionIndex
          return (
            <Text
              key={action.id}
              color={selected ? colors.text : colors.dimText}
              backgroundColor={selected ? colors.primary : undefined}
              bold={selected}
            >
              {selected ? '>' : ' '} {action.shortcut} {action.label}{' '}
            </Text>
          )
        })}
        <Text color={colors.dimText}> {COMMENT_SHORTCUT} message</Text>
      </Box>
    ),
  })
  structuralRows.push({
    key: 'action-hint',
    priority: 2,
    element: commentMode ? (
      <Text key="action-hint" color={colors.dimText}>
        Enter send (deny + tell the agent) · Esc cancel
      </Text>
    ) : unknownKeyHint ? (
      <Text key="action-hint" color={colors.warning}>
        {unknownKeyHint}
      </Text>
    ) : (
      <Text key="action-hint" color={colors.dimText}>
        ↑/↓/←/→ move Enter select y/s/r/a/n shortcuts m message Esc/Ctrl+C stop the run
      </Text>
    ),
  })
  if (approval.repeatCount && approval.repeatCount > 1) {
    structuralRows.push({
      key: 'repeat',
      priority: 2,
      element: (
        <Text key="repeat" color={colors.warning}>
          Same command requested again in this run ({formatOrdinal(approval.repeatCount)} time) —
          consider [r] run scope
        </Text>
      ),
    })
  }
  if (approval.phase) {
    structuralRows.push({
      key: 'phase',
      priority: 2,
      element: (
        <Text key="phase" color={colors.dimText}>
          requested during {approval.phase}
        </Text>
      ),
    })
  }
  if (approval.suggestedRule) {
    structuralRows.push({
      key: 'rule',
      priority: 2,
      element: (
        <Text key="rule" color={colors.dimText}>
          remembered as {approval.suggestedRule.tool}: {approval.suggestedRule.pattern}
        </Text>
      ),
    })
  }
  if (approval.expiresAt) {
    structuralRows.push({
      key: 'expires',
      priority: 2,
      element: (
        <Text key="expires" color={colors.dimText}>
          expires {approval.expiresAt}
        </Text>
      ),
    })
  }
  if (approval.context) {
    structuralRows.push({
      key: 'context',
      priority: 2,
      element: (
        <Text key="context" color={colors.dimText} wrap="truncate-end">
          why: {approval.context.replace(/\s+/g, ' ')}
        </Text>
      ),
    })
  }
  structuralRows.push({
    key: 'tool',
    priority: 3,
    element: (
      <Text key="tool">
        {approval.toolName} ({approval.state})
      </Text>
    ),
  })
  structuralRows.push({
    key: 'meta',
    priority: 4,
    element: (
      <Text key="meta" color={colors.dimText}>
        {metaBits.join(` ${symbols.separator} `)}
      </Text>
    ),
  })
  structuralRows.push({
    key: 'header',
    priority: 5,
    element: (
      <Text key="header" color={colors.warning} bold>
        Approval Required
      </Text>
    ),
  })

  if (compact) {
    const actionRow = structuralRows.find((row) => row.key === 'action')!
    const actionHintRow = structuralRows.find((row) => row.key === 'action-hint')!
    const reviewText = approvalReviewText(approval)
    const commandPreview = approval.toolName === 'terminal.run' ? boundedLines(reviewText, COMPACT_COMMAND_LINES) : null
    const inputPreview = !commandPreview && !approval.previewDiff ? boundedLines(reviewText, COMPACT_INPUT_LINES) : null
    const diffLineCount = approval.previewDiff ? approval.previewDiff.split('\n').length : 0
    const compactDiff = approval.previewDiff ? (
      <UnifiedDiffView diff={approval.previewDiff} maxLines={COMPACT_DIFF_LINES} />
    ) : approval.toolName === 'fs.edit' && oldText !== undefined && newText !== undefined ? (
      <DiffView path={path} previousContent={oldText} nextContent={newText} maxLines={COMPACT_DIFF_LINES} />
    ) : null
    const hiddenLines = commandPreview?.hidden ?? inputPreview?.hidden ?? Math.max(0, diffLineCount - COMPACT_DIFF_LINES)

    return (
      <Box
        flexDirection="column"
        borderStyle="round"
        borderColor={colors.warning}
        paddingX={1}
        overflow="hidden"
        width="100%"
        minWidth={0}
      >
        <Text color={colors.warning} bold>
          Approval required {symbols.separator} {approval.toolName}
          {approval.state === 'stale' ? ` ${symbols.separator} stale` : ''}
        </Text>
        {approval.context ? (
          <Text color={colors.dimText} wrap="wrap">
            why: {approval.context.replace(/\s+/g, ' ')}
          </Text>
        ) : null}
        {compactDiff ?? (
          <Box flexDirection="column">
            {(commandPreview ?? inputPreview)!.lines.map((line, index) => (
              <Text key={index} color={approval.toolName === 'terminal.run' ? colors.info : colors.text} wrap="wrap">
                {line}
              </Text>
            ))}
          </Box>
        )}
        {hiddenLines > 0 ? (
          <Text color={colors.warning}>
            +{hiddenLines} more lines {symbols.separator} press v to print the full {approval.toolName === 'terminal.run' ? 'command' : approval.previewDiff ? 'diff' : 'input'} above
          </Text>
        ) : null}
        {approval.state === 'parked' ? (
          <Text color={colors.dimText}>Still waiting for your decision · choose an option to continue</Text>
        ) : approval.state === 'stale' ? (
          <Text color={colors.dimText}>
            {approval.resumeAvailable
              ? 'Saved checkpoint available for resume'
              : 'Approval starts a fresh recovery run'}
          </Text>
        ) : null}
        {approval.repeatCount && approval.repeatCount > 1 ? (
          <Text color={colors.warning}>
            Repeated {formatOrdinal(approval.repeatCount)} time {symbols.separator} [r] approves this run
          </Text>
        ) : null}
        {actionRow.element}
        {commentMode || unknownKeyHint ? (
          actionHintRow.element
        ) : (
          <Text color={colors.dimText}>v full {approval.toolName === 'terminal.run' ? 'command' : 'change'} {symbols.separator} n deny, keep going {symbols.separator} Esc stop the run</Text>
        )}
      </Box>
    )
  }

  const previewLines =
    availableRows === Number.POSITIVE_INFINITY
      ? undefined
      : Math.max(0, availableRows - structuralRows.length)

  const textPreview =
    previewLines == null
      ? { visible: previewTextLines(serializedInput), omitted: 0 }
      : truncatePreviewItems(previewTextLines(serializedInput), previewLines)

  const previewChildren: ReactElement[] = []
  if (previewLines === 0) {
    previewChildren.push(
      <Text key="preview-hidden" color={colors.dimText}>
        Preview hidden to fit the current terminal height.
      </Text>,
    )
  } else if (approval.toolName === 'terminal.run') {
    previewChildren.push(
      <TerminalOutput key="preview-terminal" tool={tool} maxLines={previewLines} />,
    )
  } else if (approval.previewDiff) {
    previewChildren.push(
      <UnifiedDiffView
        key="preview-unified-diff"
        diff={approval.previewDiff}
        maxLines={previewLines}
      />,
    )
  } else if (approval.toolName === 'fs.edit' && oldText !== undefined && newText !== undefined) {
    previewChildren.push(
      <DiffView
        key="preview-edit-diff"
        path={path}
        previousContent={oldText}
        nextContent={newText}
        maxLines={previewLines}
      />,
    )
  } else if (approval.toolName === 'fs.write' && content) {
    previewChildren.push(
      <DiffView
        key="preview-diff"
        path={path}
        previousContent={undefined}
        nextContent={content}
        maxLines={previewLines}
      />,
    )
  } else {
    textPreview.visible.forEach((line, index) => {
      previewChildren.push(
        <Text key={`preview-line-${index}`} color={colors.dimText}>
          {line}
        </Text>,
      )
    })
    if (textPreview.omitted > 0) {
      previewChildren.push(
        <Text key="preview-omitted" color={colors.dimText}>
          ... {textPreview.omitted} more input line
          {textPreview.omitted === 1 ? '' : 's'}
        </Text>,
      )
    }
  }

  const readingOrder: ModalRow[] = [
    structuralRows.find((row) => row.key === 'header')!,
    structuralRows.find((row) => row.key === 'tool')!,
    ...structuralRows.filter((row) => row.key === 'context'),
    structuralRows.find((row) => row.key === 'meta')!,
    ...structuralRows.filter((row) => row.key === 'phase'),
    ...structuralRows.filter((row) => row.key === 'repeat'),
    ...structuralRows.filter((row) => row.key === 'expires'),
    ...previewChildren.map((element, index) => ({
      key: `preview-${index}`,
      priority: 6 + index,
      element,
    })),
    structuralRows.find((row) => row.key === 'action')!,
    structuralRows.find((row) => row.key === 'action-hint')!,
  ]

  const visibleRows =
    availableRows === Number.POSITIVE_INFINITY
      ? readingOrder
      : selectByPriority(readingOrder, availableRows)

  return (
    <Box
      flexDirection="column"
      height={height}
      borderStyle="round"
      borderColor={colors.warning}
      paddingX={1}
      marginY={1}
      overflow="hidden"
      width="100%"
      minWidth={0}
    >
      {visibleRows.map((row) => row.element)}
    </Box>
  )
}

function formatOrdinal(value: number): string {
  const mod100 = value % 100
  if (mod100 >= 11 && mod100 <= 13) return `${value}th`
  switch (value % 10) {
    case 1:
      return `${value}st`
    case 2:
      return `${value}nd`
    case 3:
      return `${value}rd`
    default:
      return `${value}th`
  }
}

function selectByPriority(rows: ModalRow[], limit: number): ModalRow[] {
  if (rows.length <= limit) {
    return rows
  }
  const sortedByPriority = [...rows]
    .map((row, order) => ({ row, order }))
    .sort((a, b) => a.row.priority - b.row.priority)
    .slice(0, limit)
    .sort((a, b) => a.order - b.order)
  return sortedByPriority.map((entry) => entry.row)
}
