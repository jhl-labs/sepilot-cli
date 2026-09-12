// Readline-style word/line editing primitives for the composer.
//
// Ink's `useInput` only reports single-character cursor moves and single-char
// backspace/delete. Every mature terminal agent (opencode, claude-code, codex)
// also supports word-wise motion (Alt/Ctrl+←→, Alt+B/F), word deletion
// (Ctrl+W, Alt+Backspace, Alt+D) and line kills (Ctrl+U/K). These pure helpers
// implement that logic on a plain `(value, cursor)` pair so they can be unit
// tested in isolation and reused by the InputBox key handlers.
//
// Word boundary convention matches a shell's "unix word": a run of
// non-whitespace characters, with any adjacent whitespace consumed first. This
// keeps Ctrl+W muscle memory identical to bash/readline and stays predictable
// across punctuation instead of splitting on every symbol.

import { clampGraphemeOffset } from './graphemes.js'

export interface EditResult {
  value: string
  cursor: number
}

function clampOffset(value: string, offset: number): number {
  return clampGraphemeOffset(value, offset)
}

function isSpace(ch: string | undefined): boolean {
  return ch === ' ' || ch === '\t'
}

/**
 * Offset one word to the left of `cursor`: skip whitespace immediately to the
 * left, then skip the run of non-whitespace characters. Stops at line starts
 * so word motion never jumps across a newline in a single press.
 */
export function wordLeftOffset(value: string, cursor: number): number {
  let index = clampOffset(value, cursor)
  while (index > 0 && value[index - 1] !== '\n' && isSpace(value[index - 1])) {
    index -= 1
  }
  while (index > 0 && value[index - 1] !== '\n' && !isSpace(value[index - 1])) {
    index -= 1
  }
  return index
}

/**
 * Offset one word to the right of `cursor`: skip whitespace immediately to the
 * right, then skip the run of non-whitespace characters. Stops at line ends.
 */
export function wordRightOffset(value: string, cursor: number): number {
  const length = value.length
  let index = clampOffset(value, cursor)
  while (index < length && value[index] !== '\n' && isSpace(value[index])) {
    index += 1
  }
  while (index < length && value[index] !== '\n' && !isSpace(value[index])) {
    index += 1
  }
  return index
}

/** Start of the current line: just after the previous newline, or 0. */
export function lineStartOffset(value: string, cursor: number): number {
  const index = clampOffset(value, cursor)
  const previousNewline = value.lastIndexOf('\n', index - 1)
  return previousNewline < 0 ? 0 : previousNewline + 1
}

/** End of the current line: the next newline, or the value length. */
export function lineEndOffset(value: string, cursor: number): number {
  const index = clampOffset(value, cursor)
  const nextNewline = value.indexOf('\n', index)
  return nextNewline < 0 ? value.length : nextNewline
}

function removeRange(value: string, from: number, to: number): EditResult {
  const start = Math.min(from, to)
  const end = Math.max(from, to)
  return {
    value: `${value.slice(0, start)}${value.slice(end)}`,
    cursor: start,
  }
}

/** Delete the word to the left of the cursor (Ctrl+W / Alt+Backspace). */
export function deleteWordLeft(value: string, cursor: number): EditResult {
  const target = wordLeftOffset(value, cursor)
  if (target === cursor) return { value, cursor }
  return removeRange(value, target, cursor)
}

/** Delete the word to the right of the cursor (Alt+D). */
export function deleteWordRight(value: string, cursor: number): EditResult {
  const target = wordRightOffset(value, cursor)
  if (target === cursor) return { value, cursor }
  return removeRange(value, cursor, target)
}

/** Kill from the cursor back to the start of the line (Ctrl+U). */
export function deleteToLineStart(value: string, cursor: number): EditResult {
  const start = lineStartOffset(value, cursor)
  if (start === cursor) return { value, cursor }
  return removeRange(value, start, cursor)
}

/** Kill from the cursor to the end of the line (Ctrl+K). */
export function deleteToLineEnd(value: string, cursor: number): EditResult {
  const end = lineEndOffset(value, cursor)
  if (end === cursor) return { value, cursor }
  return removeRange(value, cursor, end)
}
