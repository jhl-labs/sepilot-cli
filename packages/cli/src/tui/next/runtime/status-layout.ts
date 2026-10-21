import stringWidth from 'string-width'
import { splitGraphemes } from '../../utils/graphemes.js'

export interface StatusSegment {
  id: string
  text: string
  compactText?: string
  priority: number
  minWidth?: number
}

export const STATUS_SEPARATOR = ' · '

export function truncateStatusText(value: string, width: number): string {
  const available = Math.max(0, Math.floor(width))
  const text = value.replace(/\s+/gu, ' ').trim()
  if (stringWidth(text) <= available) return text
  if (available === 0) return ''
  let result = ''
  for (const grapheme of splitGraphemes(text)) {
    if (stringWidth(result + grapheme) + 1 > available) break
    result += grapheme
  }
  return `${result}…`
}

export function fitStatusSegments<T extends StatusSegment>(segments: T[], width: number): T[] {
  const available = Math.max(0, Math.floor(width))
  const clean = segments.filter(({ text }) => text).map((segment) => ({
    ...segment,
    text: segment.text.replace(/\s+/gu, ' ').trim(),
  }))
  if (stringWidth(clean.map(({ text }) => text).join(STATUS_SEPARATOR)) <= available) return clean
  const compact = clean.map((segment) => ({ ...segment, text: segment.compactText ?? segment.text }))
  const selected = new Map<string, T>()
  let used = 0
  for (const segment of [...compact].sort((a, b) => b.priority - a.priority)) {
    const remaining = available - used - (selected.size ? stringWidth(STATUS_SEPARATOR) : 0)
    if (remaining <= 0) continue
    const size = stringWidth(segment.text)
    if (size > remaining && (segment.minWidth == null || remaining < Math.min(size, segment.minWidth))) continue
    const text = truncateStatusText(segment.text, remaining)
    selected.set(segment.id, { ...segment, text })
    used += stringWidth(text) + (selected.size > 1 ? stringWidth(STATUS_SEPARATOR) : 0)
  }
  return compact.flatMap(({ id }) => selected.has(id) ? [selected.get(id)!] : [])
}
