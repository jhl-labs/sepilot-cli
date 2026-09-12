import { useState, useCallback, useEffect, useRef } from 'react'
import { useInputHistory } from '../hooks/useInputHistory.js'
import { Box, Text, useInput, useStdin } from 'ink'
import chalk from 'chalk'
import { colors } from '../theme.js'
import {
  applyHangulInputChange,
  composeHangul,
  createHangulInputState,
  stripBracketedPasteDelimiters,
} from '../utils/hangul.js'
import { isReturnKey } from '../utils/key.js'
import { wrapPlainText } from '../utils/transcript.js'
import {
  deleteToLineEnd,
  deleteToLineStart,
  deleteWordLeft,
  deleteWordRight,
  wordLeftOffset,
  wordRightOffset,
} from '../utils/input-editing.js'
import {
  clampGraphemeOffset,
  countGraphemes,
  graphemeAt,
  nextGraphemeOffset,
  previousGraphemeOffset,
} from '../utils/graphemes.js'
import { deriveComposerInputPresentation } from '../utils/composer-mode.js'

interface InputBoxProps {
  value: string
  onChange: (value: string) => void
  onSubmit: (text: string, options?: { displayText?: string }) => void
  width: number
  maxEditorRows?: number
  isStreaming: boolean
  streamLabel?: string | null
  /** Messages queued while the agent is streaming; shows a chip when > 0. */
  queuedCount?: number
  blockedReason?: string | null
  blockedHint?: string | null
  paletteOpen?: boolean
  attachmentAutocompleteOpen?: boolean
  skillAutocompleteOpen?: boolean
  pendingAttachments?: string[]
  denialFollowup?: { toolName: string } | null
  exitHint?: string | null
  onPalettePrev?: () => void
  onPaletteNext?: () => void
  onPaletteAutocomplete?: () => void
  onPaletteAccept?: () => boolean
  onAttachmentPrev?: () => void
  onAttachmentNext?: () => void
  onAttachmentAutocomplete?: () => void
  onSkillPrev?: () => void
  onSkillNext?: () => void
  onSkillAutocomplete?: () => void
  onToggleMode?: () => void
  onExit?: () => void
}

interface InputBoxHeightOptions {
  value: string
  width: number
  maxEditorRows?: number
  isStreaming?: boolean
  streamLabel?: string | null
  blockedReason?: string | null
  blockedHint?: string | null
  paletteOpen?: boolean
  attachmentAutocompleteOpen?: boolean
  skillAutocompleteOpen?: boolean
  pendingAttachments?: string[]
  denialFollowup?: { toolName: string } | null
  exitHint?: string | null
}

const PROMPT_COLUMNS = 2
const INPUT_BOX_BORDER_ROWS = 1
const DEFAULT_MAX_EDITOR_ROWS = 6
const BRACKETED_PASTE_START = '\x1b[200~'
const BRACKETED_PASTE_END = '\x1b[201~'
const INK_TRIMMED_BRACKETED_PASTE_START = '[200~'
const INK_TRIMMED_BRACKETED_PASTE_END = '[201~'

interface SubmissionText {
  text: string
  displayText?: string
}

const MEMORY_CONTROL_SHORTCUTS = new Set([
  'audit',
  'audits',
  'backlog',
  'backlogs',
  'cleanup',
  'close',
  'current',
  'file',
  'files',
  'health',
  'hide',
  'info',
  'lifecycle',
  'maintain',
  'maintenance',
  'maintenance-status',
  'markdown',
  'off',
  'open-loop',
  'open-loops',
  'openloops',
])

function firstToken(value: string): string {
  return value.trim().split(/\s+/, 1)[0]?.toLowerCase() ?? ''
}

function formatCount(value: number): string {
  return value.toLocaleString('en-US')
}

function formatPastePlaceholder(index: number, content: string): string {
  return `[Paste #${index} - ${formatCount(countGraphemes(content))} chars]`
}

function countOccurrences(input: string, needle: string): number {
  return needle ? input.split(needle).length - 1 : 0
}

function summarizeInputCharacterCount(
  input: string,
  pasteContents: ReadonlyMap<string, string>,
): { characters: number; pasteCount: number } {
  let resolved = input
  let pasteCount = 0

  for (const [placeholder, content] of pasteContents) {
    const occurrences = countOccurrences(resolved, placeholder)
    if (occurrences === 0) continue
    pasteCount += occurrences
    resolved = resolved.split(placeholder).join(content)
  }

  return {
    characters: countGraphemes(resolved),
    pasteCount,
  }
}

function formatInputCharacterCount(
  input: string,
  pasteContents: ReadonlyMap<string, string>,
): string {
  const { characters, pasteCount } = summarizeInputCharacterCount(input, pasteContents)
  const pasteSuffix =
    pasteCount > 0 ? ` (${formatCount(pasteCount)} paste${pasteCount === 1 ? '' : 's'})` : ''
  return `${formatCount(characters)} chars${pasteSuffix}`
}

function findNextPasteStart(
  input: string,
  fromIndex: number,
): { index: number; length: number } | null {
  const full = input.indexOf(BRACKETED_PASTE_START, fromIndex)
  const trimmed = input.indexOf(INK_TRIMMED_BRACKETED_PASTE_START, fromIndex)

  if (full < 0 && trimmed < 0) return null
  if (full >= 0 && (trimmed < 0 || full <= trimmed)) {
    return { index: full, length: BRACKETED_PASTE_START.length }
  }
  return { index: trimmed, length: INK_TRIMMED_BRACKETED_PASTE_START.length }
}

function findPasteEnd(input: string, fromIndex: number): { index: number; length: number } | null {
  const full = input.indexOf(BRACKETED_PASTE_END, fromIndex)
  const trimmed = input.indexOf(INK_TRIMMED_BRACKETED_PASTE_END, fromIndex)

  if (full < 0 && trimmed < 0) return null
  if (full >= 0 && (trimmed < 0 || full <= trimmed)) {
    return { index: full, length: BRACKETED_PASTE_END.length }
  }
  return { index: trimmed, length: INK_TRIMMED_BRACKETED_PASTE_END.length }
}

function summarizeBracketedPastes(
  input: string,
  createPlaceholder: (content: string) => string,
): string | null {
  let cursor = 0
  let output = ''
  let found = false

  while (cursor < input.length) {
    const start = findNextPasteStart(input, cursor)
    if (!start) {
      output += input.slice(cursor)
      break
    }

    found = true
    output += input.slice(cursor, start.index)
    const contentStart = start.index + start.length
    const end = findPasteEnd(input, contentStart)
    const contentEnd = end?.index ?? input.length
    const pastedContent = input.slice(contentStart, contentEnd)
    output += createPlaceholder(pastedContent)
    cursor = end ? end.index + end.length : input.length
  }

  return found ? output : null
}

function resolveShortcutSubmission(text: string, displayText?: string): SubmissionText {
  if (text === '?') {
    return {
      text: '/help',
      displayText: displayText ?? text,
    }
  }

  if (text === '!!') {
    return { text, displayText }
  }

  if (!text.startsWith('#')) {
    return displayText && displayText !== text ? { text, displayText } : { text }
  }

  const content = text.slice(1).trim()
  if (!content || content === '?') {
    return {
      text: '/memory current',
      displayText: displayText ?? text,
    }
  }

  if (content.startsWith('?')) {
    const query = content.slice(1).trim()
    return {
      text: query ? `/memory ${query}` : '/memory current',
      displayText: displayText ?? text,
    }
  }

  if (MEMORY_CONTROL_SHORTCUTS.has(firstToken(content))) {
    return {
      text: `/memory ${content}`,
      displayText: displayText ?? text,
    }
  }

  return {
    text: content ? `/remember ${content}` : '/remember',
    displayText: displayText ?? text,
  }
}

function memoryShortcutFooter(value: string): string {
  const content = value.slice(1).trim()
  if (!content || content === '?' || ['current', 'info'].includes(firstToken(content))) {
    return 'Enter:show memory status'
  }
  if (content.startsWith('?')) return 'Enter:search memory'
  if (MEMORY_CONTROL_SHORTCUTS.has(firstToken(content))) return 'Enter:memory command'
  return 'Enter:save memory'
}

function calculateEditorRows(
  text: string,
  width: number,
  maxEditorRows = DEFAULT_MAX_EDITOR_ROWS,
): number {
  const editorWidth = Math.max(20, width - PROMPT_COLUMNS)
  const wrappedRows = wrapPlainText(text || ' ', editorWidth).length
  return Math.max(1, Math.min(maxEditorRows, wrappedRows))
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function renderEditableText(value: string, placeholder: string, cursorOffset: number): string {
  if (!value) {
    return placeholder
      ? `${chalk.inverse(placeholder[0])}${chalk.grey(placeholder.slice(1))}`
      : chalk.inverse(' ')
  }

  const safeOffset = clamp(cursorOffset, 0, value.length)
  const graphemeOffset = clampGraphemeOffset(value, safeOffset)
  const before = value.slice(0, graphemeOffset)
  const current = graphemeAt(value, graphemeOffset)
  const after = value.slice(graphemeOffset + current.length)

  return current ? `${before}${chalk.inverse(current)}${after}` : `${value}${chalk.inverse(' ')}`
}

function hasInlineControl(input: string): boolean {
  for (const ch of input) {
    if (ch === '\x7f' || ch === '\b' || ch < ' ') return true
  }
  return false
}

export function calculateInputBoxHeight({
  value,
  width,
  maxEditorRows = DEFAULT_MAX_EDITOR_ROWS,
  isStreaming = false,
  blockedReason = null,
  blockedHint = null,
  paletteOpen = false,
  attachmentAutocompleteOpen = false,
  skillAutocompleteOpen = false,
  pendingAttachments = [],
  denialFollowup = null,
  exitHint = null,
}: InputBoxHeightOptions): number {
  const editorText = blockedReason ?? value
  const isBlocked = Boolean(blockedReason)
  const showStatusRow = shouldShowStatusRow({
    isBlocked,
    pendingAttachments,
    denialFollowup,
  })
  const footerHint = footerHintForState({
    isBlocked,
    blockedHint,
    paletteOpen,
    attachmentAutocompleteOpen,
    skillAutocompleteOpen,
    exitHint,
    value,
  })
  const showFooterRow = shouldShowFooterRow({
    footerHint,
    isBlocked,
    isStreaming,
    value,
  })
  return (
    INPUT_BOX_BORDER_ROWS +
    (showStatusRow ? 1 : 0) +
    calculateEditorRows(editorText, width, maxEditorRows) +
    (showFooterRow ? 1 : 0)
  )
}

function shouldShowStatusRow({
  isBlocked,
  pendingAttachments,
  denialFollowup,
}: {
  isBlocked: boolean
  pendingAttachments: string[]
  denialFollowup: { toolName: string } | null
}): boolean {
  return (!isBlocked && pendingAttachments.length > 0) || Boolean(denialFollowup)
}

function footerHintForState({
  isBlocked,
  blockedHint,
  paletteOpen,
  attachmentAutocompleteOpen,
  skillAutocompleteOpen,
  exitHint,
  value,
}: {
  isBlocked: boolean
  blockedHint?: string | null
  paletteOpen?: boolean
  attachmentAutocompleteOpen?: boolean
  skillAutocompleteOpen?: boolean
  exitHint?: string | null
  value?: string
}): string | null {
  if (exitHint) return exitHint
  if (isBlocked) return blockedHint ?? 'Esc:back'
  if (paletteOpen) return 'Tab:complete ↑↓:select'
  if (attachmentAutocompleteOpen) return 'Tab/Enter:file ↑↓:select Ctrl+R:reindex'
  if (skillAutocompleteOpen) return 'Tab/Enter:skill ↑↓:select'
  if (value?.trim() === '?') return 'Enter:show help'
  if (value?.trim() === '!!') return 'SHELL · Enter:rerun · Esc:chat'
  if (value?.startsWith('!')) return 'SHELL · Enter:run · Esc:chat'
  if (value?.startsWith('#')) return memoryShortcutFooter(value)
  return null
}

function shouldShowFooterRow({
  footerHint,
  isBlocked,
  isStreaming,
  value,
}: {
  footerHint: string | null
  isBlocked: boolean
  isStreaming: boolean
  value: string
}): boolean {
  return Boolean(footerHint) || isStreaming || (!isBlocked && value.length > 0)
}

export function InputBox({
  value,
  onChange,
  onSubmit,
  width,
  maxEditorRows = DEFAULT_MAX_EDITOR_ROWS,
  isStreaming,
  streamLabel = null,
  queuedCount = 0,
  blockedReason = null,
  blockedHint = null,
  paletteOpen = false,
  attachmentAutocompleteOpen = false,
  skillAutocompleteOpen = false,
  pendingAttachments = [],
  denialFollowup = null,
  exitHint = null,
  onPalettePrev,
  onPaletteNext,
  onPaletteAutocomplete,
  onPaletteAccept,
  onAttachmentPrev,
  onAttachmentNext,
  onAttachmentAutocomplete,
  onSkillPrev,
  onSkillNext,
  onSkillAutocomplete,
  onToggleMode,
  onExit,
}: InputBoxProps) {
  const { history, append } = useInputHistory()
  const historyIndexRef = useRef(-1)
  const isBlocked = Boolean(blockedReason)
  const valueRef = useRef(value)
  const cursorOffsetRef = useRef(value.length)
  const hangulInputStateRef = useRef(createHangulInputState())
  const pendingLocalValuesRef = useRef(new Set<string>())
  const pasteCounterRef = useRef(0)
  const pasteContentsRef = useRef(new Map<string, string>())
  const lastReturnSubmitAtRef = useRef(0)
  const lastExitAtRef = useRef(0)
  const { stdin } = useStdin()
  const [displayValue, setDisplayValue] = useState(value)
  const [cursorOffset, setCursorOffset] = useState(value.length)

  useEffect(() => {
    if (valueRef.current === value) {
      pendingLocalValuesRef.current.delete(value)
      setDisplayValue(value)
      return
    }

    if (pendingLocalValuesRef.current.delete(value)) {
      return
    }

    const previousValue = valueRef.current
    const previousCursorOffset = cursorOffsetRef.current
    valueRef.current = value
    if (!value) {
      pasteCounterRef.current = 0
      pasteContentsRef.current.clear()
    }
    setDisplayValue(value)
    hangulInputStateRef.current = createHangulInputState()
    const nextOffset =
      previousCursorOffset >= previousValue.length
        ? value.length
        : clampGraphemeOffset(value, previousCursorOffset)
    cursorOffsetRef.current = nextOffset
    setCursorOffset(nextOffset)
  }, [value])

  const applyValue = useCallback(
    (nextValue: string, nextCursorOffset = nextValue.length): void => {
      valueRef.current = nextValue
      pendingLocalValuesRef.current.add(nextValue)
      setDisplayValue(nextValue)
      const safeOffset = clampGraphemeOffset(nextValue, nextCursorOffset)
      cursorOffsetRef.current = safeOffset
      setCursorOffset(safeOffset)
      onChange(nextValue)
    },
    [onChange],
  )

  const resetComposition = useCallback((): void => {
    hangulInputStateRef.current = createHangulInputState()
  }, [])

  const resetHistoryNavigation = useCallback((): void => {
    historyIndexRef.current = -1
  }, [])

  const createPasteSummary = useCallback((content: string): string => {
    const index = pasteCounterRef.current + 1
    pasteCounterRef.current = index
    const placeholder = formatPastePlaceholder(index, content)
    pasteContentsRef.current.set(placeholder, content)
    return placeholder
  }, [])

  const resolvePasteSummaries = useCallback(
    (text: string): { text: string; displayText?: string } => {
      if (pasteContentsRef.current.size === 0) {
        return { text }
      }

      let resolved = text
      let changed = false
      for (const [placeholder, content] of pasteContentsRef.current) {
        if (!resolved.includes(placeholder)) continue
        resolved = resolved.split(placeholder).join(content)
        changed = true
      }

      return changed ? { text: resolved, displayText: text } : { text }
    },
    [],
  )

  const clearPasteSummaries = useCallback((): void => {
    pasteCounterRef.current = 0
    pasteContentsRef.current.clear()
  }, [])

  const requestExit = useCallback((): void => {
    const now = Date.now()
    if (now - lastExitAtRef.current < 30) {
      return
    }
    lastExitAtRef.current = now
    onExit?.()
  }, [onExit])

  const insertPlainText = useCallback(
    (text: string): void => {
      if (!text) return
      resetHistoryNavigation()
      resetComposition()
      const current = valueRef.current
      const cursor = cursorOffsetRef.current
      const normalized = text.normalize('NFC')
      applyValue(
        `${current.slice(0, cursor)}${normalized}${current.slice(cursor)}`,
        cursor + normalized.length,
      )
    },
    [applyValue, resetComposition, resetHistoryNavigation],
  )

  const applyInsertedInput = useCallback(
    (input: string): void => {
      const summarizedPaste = summarizeBracketedPastes(input, createPasteSummary)
      if (summarizedPaste !== null) {
        insertPlainText(summarizedPaste)
        return
      }

      const insertedInput = stripBracketedPasteDelimiters(input)
      if (!insertedInput) return
      resetHistoryNavigation()

      let nextValue = valueRef.current
      let nextCursorOffset = cursorOffsetRef.current
      let nextHangulState = hangulInputStateRef.current
      let changed = false

      if (!hasInlineControl(insertedInput)) {
        const beforeCursor = nextValue.slice(0, nextCursorOffset)
        const afterCursor = nextValue.slice(nextCursorOffset)
        const applied = applyHangulInputChange(
          beforeCursor,
          `${beforeCursor}${insertedInput}`,
          nextHangulState,
        )
        hangulInputStateRef.current = applied.state
        applyValue(`${applied.value}${afterCursor}`, applied.value.length)
        return
      }

      const resetNextComposition = (): void => {
        nextHangulState = createHangulInputState()
      }

      for (const ch of insertedInput) {
        if (ch === '\x7f' || ch === '\b') {
          if (nextCursorOffset > 0) {
            const previousOffset = previousGraphemeOffset(nextValue, nextCursorOffset)
            nextValue = `${nextValue.slice(0, previousOffset)}${nextValue.slice(nextCursorOffset)}`
            nextCursorOffset = previousOffset
            changed = true
          }
          resetNextComposition()
          continue
        }

        if ((ch < ' ' && ch !== '\n') || ch === '\x7f') {
          resetNextComposition()
          continue
        }

        const beforeCursor = nextValue.slice(0, nextCursorOffset)
        const afterCursor = nextValue.slice(nextCursorOffset)
        const applied = applyHangulInputChange(
          beforeCursor,
          `${beforeCursor}${ch}`,
          nextHangulState,
        )
        nextValue = `${applied.value}${afterCursor}`
        nextHangulState = applied.state
        nextCursorOffset = applied.value.length
        changed = true
      }

      if (!changed) {
        hangulInputStateRef.current = nextHangulState
        return
      }

      hangulInputStateRef.current = nextHangulState
      applyValue(nextValue, nextCursorOffset)
    },
    [applyValue, createPasteSummary, insertPlainText, resetHistoryNavigation],
  )

  const handleSubmit = useCallback(
    (text: string) => {
      const now = Date.now()
      if (now - lastReturnSubmitAtRef.current < 30) {
        return
      }
      lastReturnSubmitAtRef.current = now

      if (paletteOpen && onPaletteAccept?.()) {
        return
      }
      if (attachmentAutocompleteOpen && !paletteOpen) {
        onAttachmentAutocomplete?.()
        return
      }
      if (skillAutocompleteOpen && !paletteOpen && !attachmentAutocompleteOpen) {
        onSkillAutocomplete?.()
        return
      }
      const composedText = composeHangul(stripBracketedPasteDelimiters(valueRef.current || text))
      const resolved = resolvePasteSummaries(composedText)
      const trimmed = resolved.text.trim()
      if (!trimmed || isBlocked) return
      const displayText = resolved.displayText?.trim()
      const submitted = resolveShortcutSubmission(trimmed, displayText)
      append(submitted.displayText ?? submitted.text)
      resetHistoryNavigation()
      resetComposition()
      clearPasteSummaries()
      applyValue('')
      onSubmit(
        submitted.text,
        submitted.displayText && submitted.displayText !== submitted.text
          ? { displayText: submitted.displayText }
          : undefined,
      )
    },
    [
      append,
      applyValue,
      attachmentAutocompleteOpen,
      clearPasteSummaries,
      isBlocked,
      isStreaming,
      onAttachmentAutocomplete,
      onPaletteAccept,
      onSkillAutocomplete,
      onSubmit,
      paletteOpen,
      resetComposition,
      resetHistoryNavigation,
      resolvePasteSummaries,
      skillAutocompleteOpen,
    ],
  )

  useEffect(() => {
    const source = stdin as unknown as {
      on?: (event: 'data', listener: (chunk: Buffer | string) => void) => void
      off?: (event: 'data', listener: (chunk: Buffer | string) => void) => void
    } | null
    if (!source?.on || !source.off) return

    const handleData = (chunk: Buffer | string): void => {
      const text = typeof chunk === 'string' ? chunk : chunk.toString('utf8')
      if (text === '\x04' && !valueRef.current && !isBlocked) {
        requestExit()
        return
      }
      const lineTail = text.replace(/[\r\n]+$/g, '')
      if (lineTail.includes('\r') || lineTail.includes('\n')) {
        return
      }
      if (
        (text.endsWith('\r') || text.endsWith('\n')) &&
        !text.includes(BRACKETED_PASTE_START) &&
        !text.includes(BRACKETED_PASTE_END) &&
        text.length < 500
      ) {
        setTimeout(() => {
          const current = valueRef.current
          const submitText =
            lineTail && !current.endsWith(lineTail) ? `${current}${lineTail}` : current
          handleSubmit(submitText)
        }, 0)
      }
    }

    source.on('data', handleData)
    return () => {
      source.off?.('data', handleData)
    }
  }, [handleSubmit, isBlocked, requestExit, stdin])

  useInput((input, key) => {
    if (isBlocked) {
      return
    }

    // Ctrl+J for newline
    if (input === 'j' && key.ctrl) {
      const current = valueRef.current
      const cursor = cursorOffsetRef.current
      const nextValue = `${current.slice(0, cursor)}\n${current.slice(cursor)}`
      resetHistoryNavigation()
      resetComposition()
      applyValue(nextValue, cursor + 1)
      return
    }

    if (isReturnKey(input, key)) {
      handleSubmit(valueRef.current)
      return
    }

    if (input === 'd' && key.ctrl && !valueRef.current) {
      requestExit()
      return
    }

    // A leading `!` is a real input mode, not merely a visual prefix. Escape
    // leaves shell mode without submitting anything and returns to chat.
    if (key.escape && valueRef.current.startsWith('!')) {
      resetHistoryNavigation()
      resetComposition()
      clearPasteSummaries()
      applyValue('')
      return
    }

    // While actively browsing input history (historyIndexRef.current >= 0)
    // arrow up/down keep navigating history even when the recalled entry
    // opens the command palette (e.g. `/health`) or attachment autocomplete
    // (e.g. `@src/foo.ts`); otherwise the user gets stuck on the first
    // recalled slash/@ command and cannot reach older entries.
    const historyNavActive = historyIndexRef.current >= 0

    if (paletteOpen && !historyNavActive && key.upArrow) {
      onPalettePrev?.()
      return
    }

    if (paletteOpen && !historyNavActive && key.downArrow) {
      onPaletteNext?.()
      return
    }

    if (paletteOpen && input === '\t') {
      onPaletteAutocomplete?.()
      return
    }

    if (!paletteOpen && attachmentAutocompleteOpen && !historyNavActive && key.upArrow) {
      onAttachmentPrev?.()
      return
    }

    if (!paletteOpen && attachmentAutocompleteOpen && !historyNavActive && key.downArrow) {
      onAttachmentNext?.()
      return
    }

    if (!paletteOpen && attachmentAutocompleteOpen && input === '\t') {
      onAttachmentAutocomplete?.()
      return
    }

    if (
      !paletteOpen &&
      !attachmentAutocompleteOpen &&
      skillAutocompleteOpen &&
      !historyNavActive &&
      key.upArrow
    ) {
      onSkillPrev?.()
      return
    }

    if (
      !paletteOpen &&
      !attachmentAutocompleteOpen &&
      skillAutocompleteOpen &&
      !historyNavActive &&
      key.downArrow
    ) {
      onSkillNext?.()
      return
    }

    if (!paletteOpen && !attachmentAutocompleteOpen && skillAutocompleteOpen && input === '\t') {
      onSkillAutocomplete?.()
      return
    }

    if (
      input === '\t' &&
      !paletteOpen &&
      !attachmentAutocompleteOpen &&
      !skillAutocompleteOpen &&
      !valueRef.current
    ) {
      onToggleMode?.()
      return
    }

    // Readline-style word/line editing. These MUST precede the single-char
    // arrow and backspace handlers below, otherwise a modified arrow/backspace
    // (Ctrl+←, Alt+Backspace, ...) would be consumed as a plain cursor move or
    // one-char delete. Alt/Option surfaces as key.meta in ink.
    const moveCursorTo = (nextOffset: number): void => {
      resetComposition()
      const minimumOffset = valueRef.current.startsWith('!') ? 1 : 0
      const safeOffset = Math.max(
        minimumOffset,
        clampGraphemeOffset(valueRef.current, nextOffset),
      )
      cursorOffsetRef.current = safeOffset
      setCursorOffset(safeOffset)
    }

    // Word-wise motion: Ctrl/Alt+←→ and emacs Alt+B / Alt+F.
    if ((key.leftArrow && (key.ctrl || key.meta)) || (key.meta && !key.ctrl && input === 'b')) {
      moveCursorTo(wordLeftOffset(valueRef.current, cursorOffsetRef.current))
      return
    }
    if ((key.rightArrow && (key.ctrl || key.meta)) || (key.meta && !key.ctrl && input === 'f')) {
      moveCursorTo(wordRightOffset(valueRef.current, cursorOffsetRef.current))
      return
    }

    // Delete word left: Ctrl+W, Alt/Ctrl+Backspace. Delete word right: Alt+D.
    if (
      (input === 'w' && key.ctrl) ||
      ((key.backspace || key.delete) && (key.meta || key.ctrl))
    ) {
      resetHistoryNavigation()
      resetComposition()
      const shellMode = valueRef.current.startsWith('!')
      const source = shellMode ? valueRef.current.slice(1) : valueRef.current
      const sourceCursor = shellMode
        ? Math.max(0, cursorOffsetRef.current - 1)
        : cursorOffsetRef.current
      const edit = deleteWordLeft(source, sourceCursor)
      applyValue(shellMode ? `!${edit.value}` : edit.value, edit.cursor + (shellMode ? 1 : 0))
      return
    }
    if (key.meta && !key.ctrl && input === 'd') {
      resetHistoryNavigation()
      resetComposition()
      const shellMode = valueRef.current.startsWith('!')
      const source = shellMode ? valueRef.current.slice(1) : valueRef.current
      const sourceCursor = shellMode
        ? Math.max(0, cursorOffsetRef.current - 1)
        : cursorOffsetRef.current
      const edit = deleteWordRight(source, sourceCursor)
      applyValue(shellMode ? `!${edit.value}` : edit.value, edit.cursor + (shellMode ? 1 : 0))
      return
    }

    // Kill to line start (Ctrl+U) / line end (Ctrl+K).
    if (input === 'u' && key.ctrl) {
      resetHistoryNavigation()
      resetComposition()
      const shellMode = valueRef.current.startsWith('!')
      const source = shellMode ? valueRef.current.slice(1) : valueRef.current
      const sourceCursor = shellMode
        ? Math.max(0, cursorOffsetRef.current - 1)
        : cursorOffsetRef.current
      const edit = deleteToLineStart(source, sourceCursor)
      applyValue(shellMode ? `!${edit.value}` : edit.value, edit.cursor + (shellMode ? 1 : 0))
      return
    }
    if (input === 'k' && key.ctrl) {
      resetHistoryNavigation()
      resetComposition()
      const shellMode = valueRef.current.startsWith('!')
      const source = shellMode ? valueRef.current.slice(1) : valueRef.current
      const sourceCursor = shellMode
        ? Math.max(0, cursorOffsetRef.current - 1)
        : cursorOffsetRef.current
      const edit = deleteToLineEnd(source, sourceCursor)
      applyValue(shellMode ? `!${edit.value}` : edit.value, edit.cursor + (shellMode ? 1 : 0))
      return
    }

    if (key.leftArrow) {
      resetComposition()
      const nextOffset = Math.max(
        valueRef.current.startsWith('!') ? 1 : 0,
        previousGraphemeOffset(valueRef.current, cursorOffsetRef.current),
      )
      cursorOffsetRef.current = nextOffset
      setCursorOffset(nextOffset)
      return
    }

    if (key.rightArrow) {
      resetComposition()
      const nextOffset = nextGraphemeOffset(valueRef.current, cursorOffsetRef.current)
      cursorOffsetRef.current = nextOffset
      setCursorOffset(nextOffset)
      return
    }

    // Ink 5 reports the DEL byte (0x7f), emitted by Backspace in most Linux
    // terminals, as `key.delete`; treat both flags as backward deletion.
    if (key.backspace || key.delete) {
      const cursor = cursorOffsetRef.current
      if (cursor > 0) {
        resetHistoryNavigation()
        resetComposition()
        const current = valueRef.current
        const previousOffset = previousGraphemeOffset(current, cursor)
        applyValue(`${current.slice(0, previousOffset)}${current.slice(cursor)}`, previousOffset)
      }
      return
    }

    // Arrow up starts history navigation only from an empty composer. Once
    // browsing, the recalled text itself is non-empty, so keep arrows bound
    // to history until the user edits the value or returns to the draft.
    if (key.upArrow && (!valueRef.current || historyIndexRef.current >= 0)) {
      if (history.length > 0) {
        const currentIndex = historyIndexRef.current
        const newIndex = currentIndex < history.length - 1 ? currentIndex + 1 : currentIndex
        historyIndexRef.current = newIndex
        resetComposition()
        applyValue(history[history.length - 1 - newIndex])
      }
      return
    }
    if (key.downArrow && historyIndexRef.current >= 0) {
      if (historyIndexRef.current > 0) {
        const newIndex = historyIndexRef.current - 1
        historyIndexRef.current = newIndex
        resetComposition()
        applyValue(history[history.length - 1 - newIndex])
      } else {
        resetHistoryNavigation()
        resetComposition()
        applyValue('')
      }
      return
    }

    if (key.ctrl || key.meta || key.escape || key.upArrow || key.downArrow || key.tab) {
      return
    }

    if (input) {
      applyInsertedInput(input)
    }
  })

  const showAttachmentsRow = !isBlocked && pendingAttachments.length > 0
  const showStatusRow = shouldShowStatusRow({
    isBlocked,
    pendingAttachments,
    denialFollowup,
  })
  const inputPresentation = deriveComposerInputPresentation(displayValue, cursorOffset)
  const shellMode = inputPresentation.mode === 'shell'
  const editorValue = inputPresentation.editorValue
  const editorRows = calculateEditorRows(
    blockedReason ?? editorValue,
    width,
    maxEditorRows,
  )
  const footerHint = footerHintForState({
    isBlocked,
    blockedHint,
    paletteOpen,
    attachmentAutocompleteOpen,
    skillAutocompleteOpen,
    exitHint,
    value: displayValue,
  })
  const showFooterRow = shouldShowFooterRow({
    footerHint,
    isBlocked,
    isStreaming,
    value: displayValue,
  })
  const containerHeight =
    INPUT_BOX_BORDER_ROWS + (showStatusRow ? 1 : 0) + editorRows + (showFooterRow ? 1 : 0)
  const placeholder = denialFollowup
    ? `Tell the agent how to proceed instead of ${denialFollowup.toolName}...`
    : shellMode
      ? 'Type a local shell command...'
      : 'Type a message...'
  const renderedValue = renderEditableText(
    editorValue,
    placeholder,
    inputPresentation.editorCursorOffset,
  )

  return (
    <Box
      borderStyle="single"
      borderColor={isStreaming || shellMode ? colors.warning : colors.inputBorder}
      borderTop
      borderBottom={false}
      borderLeft={false}
      borderRight={false}
      flexDirection="column"
      height={containerHeight}
      overflow="hidden"
    >
      {showStatusRow ? (
        <Box height={1}>
          {showAttachmentsRow ? (
            <Text color={colors.dimText} wrap="truncate-end">
              files: {pendingAttachments.join(', ')}
            </Text>
          ) : denialFollowup ? (
            <Text color={colors.warning} wrap="truncate-end">
              {denialFollowup.toolName} denied · type guidance or Enter to skip
            </Text>
          ) : null}
        </Box>
      ) : null}
      <Box height={editorRows}>
        <Box flexGrow={1} flexShrink={1} minWidth={0} overflow="hidden">
          <Text color={shellMode ? colors.warning : colors.primary} bold>
            {shellMode ? '$ ' : '> '}
          </Text>
          {isBlocked ? (
            <Text color={colors.warning} wrap="wrap">
              {blockedReason}
            </Text>
          ) : (
            <Text>{renderedValue}</Text>
          )}
        </Box>
      </Box>
      {showFooterRow ? (
        <Box
          height={1}
          justifyContent={footerHint || isStreaming ? 'space-between' : 'flex-end'}
        >
          {footerHint ? (
            <Text
              color={exitHint ? colors.warning : colors.dimText}
              bold={Boolean(exitHint)}
              wrap="truncate-end"
            >
              {footerHint}
            </Text>
          ) : isStreaming ? (
            <Text color={colors.dimText} wrap="truncate-end">
              {streamLabel ?? 'in progress...'}
            </Text>
          ) : null}
          {queuedCount > 0 ? (
            <Text color={colors.warning} wrap="truncate-end">
              queued ({queuedCount})
            </Text>
          ) : !exitHint && !isBlocked && !isStreaming && editorValue.length > 0 ? (
            <Text color={colors.dimText} wrap="truncate-end">
              {formatInputCharacterCount(editorValue, pasteContentsRef.current)}
            </Text>
          ) : null}
        </Box>
      ) : null}
    </Box>
  )
}
