import { useEffect, useRef, useState } from 'react'
import { Text, useInput } from 'ink'
import chalk from 'chalk'
import {
  applyHangulInputChunk,
  composeHangul,
  createHangulInputState,
  replayInlineInputControls,
  stripBracketedPasteDelimiters,
} from '../utils/hangul.js'
import {
  clampGraphemeOffset,
  graphemeBoundaries,
  nextGraphemeOffset,
  previousGraphemeOffset,
  splitGraphemes,
} from '../utils/graphemes.js'

interface ControlSafeTextInputProps {
  value: string
  placeholder?: string
  focus?: boolean
  mask?: string
  showCursor?: boolean
  onChange: (value: string) => void
  onSubmit?: (value: string) => void
}

function renderPlaceholder(placeholder: string): string {
  const [first = '', ...rest] = splitGraphemes(placeholder)
  return first
    ? `${chalk.inverse(first)}${chalk.grey(rest.join(''))}`
    : chalk.inverse(' ')
}

function renderValue(value: string, cursorOffset: number, mask?: string): string {
  const graphemes = splitGraphemes(value)
  const boundaries = graphemeBoundaries(value)
  const safeOffset = clampGraphemeOffset(value, cursorOffset)
  const cursorIndex = Math.max(0, boundaries.indexOf(safeOffset))
  const visible = mask ? graphemes.map(() => mask) : graphemes

  if (cursorIndex >= visible.length) {
    return `${visible.join('')}${chalk.inverse(' ')}`
  }

  return `${visible.slice(0, cursorIndex).join('')}${chalk.inverse(visible[cursorIndex]!)}${visible.slice(cursorIndex + 1).join('')}`
}

export function ControlSafeTextInput({
  value: originalValue,
  placeholder = '',
  focus = true,
  mask,
  showCursor = true,
  onChange,
  onSubmit,
}: ControlSafeTextInputProps) {
  const [cursorOffset, setCursorOffset] = useState(originalValue.length)
  const valueRef = useRef(originalValue)
  const cursorOffsetRef = useRef(originalValue.length)
  const hangulInputStateRef = useRef(createHangulInputState())

  useEffect(() => {
    if (valueRef.current !== originalValue) {
      valueRef.current = originalValue
      hangulInputStateRef.current = createHangulInputState()
    }
    const safeOffset = clampGraphemeOffset(originalValue, cursorOffsetRef.current)
    cursorOffsetRef.current = safeOffset
    setCursorOffset(safeOffset)
  }, [originalValue])

  const applyValue = (nextValue: string, nextOffset: number, resetComposition = true): void => {
    const safeOffset = clampGraphemeOffset(nextValue, nextOffset)
    valueRef.current = nextValue
    cursorOffsetRef.current = safeOffset
    if (resetComposition) hangulInputStateRef.current = createHangulInputState()
    setCursorOffset(safeOffset)
    if (nextValue !== originalValue) onChange(nextValue)
  }

  const value = mask
    ? mask.repeat(splitGraphemes(originalValue).length)
    : originalValue
  const renderedPlaceholder = renderPlaceholder(placeholder)
  const renderedValue = showCursor && focus
    ? originalValue
      ? renderValue(originalValue, cursorOffset, mask)
      : chalk.inverse(' ')
    : value

  useInput((input, key) => {
    if (
      key.upArrow
      || key.downArrow
      || key.tab
      || (key.shift && key.tab)
      || key.escape
    ) {
      return
    }

    if (key.ctrl) {
      if (input === 'u' && valueRef.current.length > 0) {
        applyValue('', 0)
      }
      return
    }

    if (key.return) {
      onSubmit?.(composeHangul(replayInlineInputControls(valueRef.current)).normalize('NFC'))
      return
    }

    const current = valueRef.current
    const cursor = cursorOffsetRef.current

    if (key.leftArrow) {
      if (showCursor) {
        applyValue(current, previousGraphemeOffset(current, cursor))
      }
      return
    }

    if (key.rightArrow) {
      if (showCursor) {
        applyValue(current, nextGraphemeOffset(current, cursor))
      }
      return
    }

    // Ink 5 calls the DEL byte used by Linux Backspace `delete`.
    if (key.backspace || key.delete) {
      if (cursor > 0) {
        const previousOffset = previousGraphemeOffset(current, cursor)
        applyValue(
          `${current.slice(0, previousOffset)}${current.slice(cursor)}`,
          previousOffset,
        )
      }
      return
    }

    const inserted = stripBracketedPasteDelimiters(input)
    if (!inserted) return
    const applied = applyHangulInputChunk(
      current,
      cursor,
      inserted,
      hangulInputStateRef.current,
    )
    valueRef.current = applied.value
    cursorOffsetRef.current = applied.cursorOffset
    hangulInputStateRef.current = applied.state
    setCursorOffset(applied.cursorOffset)
    if (applied.value !== originalValue) onChange(applied.value)
  }, { isActive: focus })

  return (
    <Text>
      {placeholder && value.length === 0 ? renderedPlaceholder : renderedValue}
    </Text>
  )
}
