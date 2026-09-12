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
  summarizeToolInput,
  terminalCommandLabel,
  truncatePreviewItems,
} from '../utils/tooling.js'
import { isReturnKey } from '../utils/key.js'

interface ApprovalModalProps {
  approval: ApprovalRequest
  height?: number
  inputActive?: boolean
  compact?: boolean
  onResolve?: (approved: boolean, scope: ApprovalActionScope, note?: string) => void
  /**
   * Fired when the inline comment editor opens/closes so the host can
   * suspend its global y/s/a/n/Esc keybindings while the user is typing
   * free text (otherwise typing "yes" would approve mid-sentence).
   */
  onCommentModeChange?: (active: boolean) => void
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
const UNKNOWN_KEY_HINT_MS = 2500
const KNOWN_SHORTCUTS = new Set([
  ...APPROVAL_ACTIONS.map((action) => action.shortcut),
  COMMENT_SHORTCUT,
])

export function ApprovalModal({
  approval,
  height,
  inputActive = true,
  compact = false,
  onResolve,
  onCommentModeChange,
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
        // Esc backs out of the draft so a mistaken press does not discard
        // written feedback. Ctrl+C is the unconditional escape hatch for the
        // whole HITL prompt, including while the comment editor is open.
        if (key.escape) {
          setCommentDraft('')
          setCommentModeNotified(false)
        } else if (cancelApproval) {
          setCommentDraft('')
          setCommentModeNotified(false)
          onResolve(false, 'once')
        }
        return
      }

      if (cancelApproval) {
        onResolve(false, 'once')
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

      // Any other printable key: tell the user instead of silently ignoring.
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
    // Free-text feedback: deny this call and hand the note to the agent
    // so it can revise the approach ([approval:needs-changes] semantics).
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

  // availableRows accounts for the top+bottom borders of the surrounding Box.
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
        ↑/↓/←/→ move Enter select y/s/r/a/n shortcuts m message Esc/Ctrl+C deny
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
        {approval.toolName} {approval.state === 'stale' ? '(stale)' : '(live)'}
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
    const compactPreview = approval.toolName === 'terminal.run'
      ? `$ ${terminalCommandLabel(approval.input)}`
      : summarizeToolInput(approval.input, approval.toolName)
    const compactDiff = approval.previewDiff ? (
      <UnifiedDiffView diff={approval.previewDiff} maxLines={10} />
    ) : approval.toolName === 'fs.edit' && oldText !== undefined && newText !== undefined ? (
      <DiffView path={path} previousContent={oldText} nextContent={newText} maxLines={10} />
    ) : null

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
          <Text
            color={approval.toolName === 'terminal.run' ? colors.info : colors.dimText}
            wrap="truncate-end"
          >
            {compactPreview}
          </Text>
        )}
        {approval.state === 'stale' ? (
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
          <Text color={colors.dimText}>Esc/Ctrl+C deny</Text>
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
