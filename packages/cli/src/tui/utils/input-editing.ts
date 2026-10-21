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

export function lineStartOffset(value: string, cursor: number): number {
  const index = clampOffset(value, cursor)
  const previousNewline = value.lastIndexOf('\n', index - 1)
  return previousNewline < 0 ? 0 : previousNewline + 1
}

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

export function deleteWordLeft(value: string, cursor: number): EditResult {
  const target = wordLeftOffset(value, cursor)
  if (target === cursor) return { value, cursor }
  return removeRange(value, target, cursor)
}

export function deleteWordRight(value: string, cursor: number): EditResult {
  const target = wordRightOffset(value, cursor)
  if (target === cursor) return { value, cursor }
  return removeRange(value, cursor, target)
}

export function deleteToLineStart(value: string, cursor: number): EditResult {
  const start = lineStartOffset(value, cursor)
  if (start === cursor) return { value, cursor }
  return removeRange(value, start, cursor)
}

export function deleteToLineEnd(value: string, cursor: number): EditResult {
  const end = lineEndOffset(value, cursor)
  if (end === cursor) return { value, cursor }
  return removeRange(value, cursor, end)
}
