import { useEffect, useMemo, useState } from 'react'
import { Box, Text, useInput } from 'ink'
import { colors } from '../theme.js'
import { calculateVisibleWindow } from '../utils/layout.js'
import { buildCommands, buildRecommendedCommandItems } from '../utils/command-autocomplete.js'

interface HelpModalProps {
  height?: number
  onClose: () => void
  context?: HelpModalContext | null
}

interface HelpEntry {
  keys: string
  label: string
}

interface HelpSection {
  title: string
  entries: HelpEntry[]
}

interface FlatLine {
  type: 'header' | 'entry'
  text: string
  keys?: string
}

export interface HelpModalContext {
  sessionId?: string | null
  projectName?: string | null
  provider?: string | null
  model?: string | null
  mode?: string | null
  autonomyLabel?: string | null
  contextPercent?: number | null
  providerCount?: number
  activeProgressLabel?: string | null
  pendingApprovalToolName?: string | null
  hasPendingApproval?: boolean
  isStreaming?: boolean
}

const STATIC_HELP_SECTIONS: ReadonlyArray<HelpSection> = [
  {
    title: 'Keyboard',
    entries: [
      { keys: 'Enter', label: 'send message' },
      { keys: 'Ctrl+J', label: 'newline within message' },
      { keys: 'Ctrl+←/→ or Alt+←/→', label: 'move cursor by word (also Alt+B / Alt+F)' },
      { keys: 'Ctrl+W or Alt+Backspace', label: 'delete the word before the cursor' },
      { keys: 'Alt+D', label: 'delete the word after the cursor' },
      { keys: 'Ctrl+U / Ctrl+K', label: 'delete to line start / line end' },
      { keys: 'Tab', label: 'autocomplete slash command, @path attachment, or @skill: selector' },
      { keys: '↑ / ↓', label: 'browse input history (when input is empty)' },
      { keys: 'PgUp / PgDn', label: 'scroll chat transcript' },
      { keys: 'Ctrl+P / Ctrl+N', label: 'page through older/newer transcript content' },
      { keys: 'Ctrl+Shift+← / Ctrl+Shift+→', label: 'switch older/newer recent sessions' },
      { keys: 'Ctrl+C', label: 'cancel current run / clear input · press twice to exit' },
      { keys: 'Ctrl+D', label: 'exit when the composer is empty' },
      { keys: 'Ctrl+L', label: 'start a new session' },
      { keys: 'Ctrl+S', label: 'open the session picker' },
      { keys: 'Ctrl+F', label: 'open the file picker for attachments' },
      { keys: 'Ctrl+T', label: 'open the model + provider picker' },
      { keys: 'Ctrl+G', label: 'open the agent mode picker' },
      { keys: 'Shift+Tab', label: 'cycle agent modes inline' },
      { keys: 'Ctrl+A', label: 'cycle autonomy (readonly → supervised → autonomous)' },
      { keys: 'Ctrl+Y', label: 'copy the latest assistant code block' },
      { keys: 'Ctrl+B', label: 'toggle the Activity panel' },
      {
        keys: 'Ctrl+O',
        label: 'print full transcript to terminal scrollback for copy (Esc to return)',
      },
      { keys: '↑ / ↓ + Enter', label: 'choose and submit pending tool approvals' },
      { keys: 'y / s / a / n', label: 'approval shortcuts: once / session / always / deny' },
      { keys: 'Esc', label: 'close pickers, palettes, and dialogs' },
    ],
  },
  {
    title: 'Session Picker',
    entries: [
      { keys: 'Ctrl+S', label: 'open the session picker from the main composer' },
      { keys: 'Tab', label: 'toggle between session search and action bar' },
      { keys: '← / →', label: 'choose session maintenance action while actions are focused' },
      { keys: 'Enter', label: 'load the selected session or run the highlighted action' },
      { keys: 'Delete action', label: 'opens a Cancel / Delete confirmation row' },
    ],
  },
  {
    title: 'Composer Shortcuts',
    entries: [
      { keys: '!<command>', label: 'run a local shell command in the workspace' },
      { keys: '!!', label: 'rerun the last local shell command' },
      { keys: '?', label: 'open this keyboard and command help' },
      { keys: '#<memory>', label: 'save a manual memory entry' },
      { keys: '#? <query>', label: 'search semantic memory' },
      { keys: '# current', label: 'show the memory panel status' },
      { keys: '@path', label: 'attach a file or add a folder summary reference' },
      { keys: '@skill:<query>', label: 'select an installed skill and convert the turn to /run' },
      { keys: '/hooks', label: 'manage outbound hook delivery, replay, and dead letters' },
    ],
  },
]

function shortId(value: string | null | undefined): string {
  return value ? value.slice(0, 8) : 'none'
}

function normalizeRunLabel(context?: HelpModalContext | null): string {
  if (!context) {
    return 'idle'
  }

  if (context.hasPendingApproval) {
    return context.pendingApprovalToolName
      ? `approval pending on ${context.pendingApprovalToolName}`
      : 'approval pending'
  }

  if (context.activeProgressLabel?.trim()) {
    return context.activeProgressLabel.trim()
  }

  return context.isStreaming ? 'running' : 'idle'
}

function buildRuntimeHelpSections(context?: HelpModalContext | null): HelpSection[] {
  if (!context) {
    return []
  }

  const currentSessionEntries: HelpEntry[] = [
    {
      keys: 'Session',
      label: context.sessionId ? shortId(context.sessionId) : 'none active',
    },
    {
      keys: 'Stack',
      label: [
        context.provider && context.model
          ? `${context.provider}/${context.model}`
          : 'provider/model not selected',
        context.mode ? `mode ${context.mode}` : null,
        context.autonomyLabel ?? null,
      ]
        .filter(Boolean)
        .join(' · '),
    },
    {
      keys: 'Run',
      label: normalizeRunLabel(context),
    },
    {
      keys: 'Context',
      label:
        context.contextPercent != null
          ? `${context.contextPercent}% of model window`
          : 'context window unknown',
    },
  ]

  if (context.projectName) {
    currentSessionEntries.push({
      keys: 'Project',
      label: context.projectName,
    })
  }

  if ((context.providerCount ?? 0) <= 0) {
    currentSessionEntries.push({
      keys: 'Providers',
      label: 'none configured',
    })
  }

  const suggestedEntries = buildRecommendedCommandItems({
    currentProjectName: context.projectName,
    currentSessionId: context.sessionId,
    providerCount: context.providerCount,
    contextPercent: context.contextPercent,
    hasPendingApproval: context.hasPendingApproval,
    isStreaming: context.isStreaming,
  }).map((item) => ({
    keys: item.insertValue,
    label: item.description,
  }))

  return [
    {
      title: 'Current Session',
      entries: currentSessionEntries,
    },
    {
      title: 'Suggested Next',
      entries: suggestedEntries,
    },
  ]
}

export function buildHelpSections(context?: HelpModalContext | null): ReadonlyArray<HelpSection> {
  return [
    ...buildRuntimeHelpSections(context),
    STATIC_HELP_SECTIONS[0],
    STATIC_HELP_SECTIONS[1],
    {
      title: 'Slash Commands',
      entries: buildCommands().map((command) => ({
        keys: command.args ? `${command.name} ${command.args}` : command.name,
        label: command.description,
      })),
    },
    STATIC_HELP_SECTIONS[2],
  ]
}

export function buildHelpLines(
  sections: ReadonlyArray<HelpSection> = buildHelpSections(),
): FlatLine[] {
  return sections.flatMap((section, index) => {
    const lines: FlatLine[] = []
    if (index > 0) {
      lines.push({ type: 'header', text: '' })
    }
    lines.push({ type: 'header', text: section.title })
    for (const entry of section.entries) {
      lines.push({ type: 'entry', text: entry.label, keys: entry.keys })
    }
    return lines
  })
}

export function HelpModal({ height, onClose, context }: HelpModalProps) {
  const flatLines = useMemo(() => buildHelpLines(buildHelpSections(context)), [context])
  const keysColumnWidth = useMemo(
    () =>
      Math.max(
        0,
        ...flatLines
          .filter((line) => line.type === 'entry')
          .map((line) => (line.keys ?? '').length),
      ),
    [flatLines],
  )
  const reservedRows = 4 // border + title + footer hint + divider
  const visibleRows = Math.max(1, (height ?? flatLines.length + reservedRows) - reservedRows)
  const [scrollOffset, setScrollOffset] = useState(0)
  const maxOffset = Math.max(0, flatLines.length - visibleRows)

  useEffect(() => {
    if (scrollOffset > maxOffset) {
      setScrollOffset(maxOffset)
    }
  }, [maxOffset, scrollOffset])

  useInput((input, key) => {
    if (key.escape || input === 'q') {
      onClose()
      return
    }
    if (key.upArrow) {
      setScrollOffset((current) => Math.max(0, current - 1))
      return
    }
    if (key.downArrow) {
      setScrollOffset((current) => Math.min(maxOffset, current + 1))
      return
    }
    if (key.pageUp) {
      setScrollOffset((current) => Math.max(0, current - visibleRows))
      return
    }
    if (key.pageDown) {
      setScrollOffset((current) => Math.min(maxOffset, current + visibleRows))
      return
    }
    if (input === 'g') {
      setScrollOffset(0)
      return
    }
    if (input === 'G') {
      setScrollOffset(maxOffset)
    }
  })

  const window = useMemo(
    () => calculateVisibleWindow(flatLines.length, scrollOffset, visibleRows),
    [flatLines.length, scrollOffset, visibleRows],
  )
  const visibleLines = flatLines.slice(window.start, window.start + visibleRows)
  const showScrollHint = flatLines.length > visibleRows
  const totalLines = flatLines.length
  const lastVisible = Math.min(totalLines, window.start + visibleRows)

  return (
    <Box
      flexDirection="column"
      height={height}
      borderStyle="round"
      borderColor={colors.primary}
      paddingX={1}
      marginY={1}
      overflow="hidden"
      width="100%"
      minWidth={0}
    >
      <Text color={colors.primary} bold>
        Help · keyboard shortcuts and slash commands
      </Text>
      {visibleLines.map((line, index) => {
        if (line.type === 'header') {
          if (!line.text) {
            return <Text key={`spacer-${window.start + index}`}> </Text>
          }
          return (
            <Text key={`header-${window.start + index}`} color={colors.info} bold>
              {line.text}
            </Text>
          )
        }
        const padded = (line.keys ?? '').padEnd(keysColumnWidth)
        return (
          <Box key={`entry-${window.start + index}`} width="100%" minWidth={0}>
            <Text color={colors.text} bold>
              {padded}
            </Text>
            <Box flexGrow={1} flexShrink={1} minWidth={0}>
              <Text color={colors.muted} wrap="truncate-end">{`  ${line.text}`}</Text>
            </Box>
          </Box>
        )
      })}
      <Text color={colors.dimText}>
        {showScrollHint
          ? `↑/↓ scroll  PgUp/PgDn or Ctrl+P/N page  g/G top/bottom  Esc close  ·  ${window.start + 1}-${lastVisible}/${totalLines}`
          : 'Esc close'}
      </Text>
    </Box>
  )
}
