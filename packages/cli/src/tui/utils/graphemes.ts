// Terminal input offsets are UTF-16 string offsets because callers edit with
// String#slice. User-visible cursor movement must nevertheless operate on
// extended grapheme clusters: one Korean syllable, an NFD syllable, or a ZWJ
// emoji is one editable character even when it spans several code units.

const segmenter = typeof Intl.Segmenter === 'function'
  ? new Intl.Segmenter('ko', { granularity: 'grapheme' })
  : null

const MARK = /^\p{Mark}$/u
const REGIONAL_INDICATOR = /^\p{Regional_Indicator}$/u

function isVariationSelector(ch: string): boolean {
  const codePoint = ch.codePointAt(0) ?? 0
  return (codePoint >= 0xfe00 && codePoint <= 0xfe0f)
    || (codePoint >= 0xe0100 && codePoint <= 0xe01ef)
}

function isEmojiModifier(ch: string): boolean {
  const codePoint = ch.codePointAt(0) ?? 0
  return codePoint >= 0x1f3fb && codePoint <= 0x1f3ff
}

function fallbackSegments(value: string): string[] {
  const segments: string[] = []
  let regionalIndicators = 0

  for (const ch of value) {
    const previous = segments.at(-1)
    const joinsPrevious = Boolean(previous)
      && (MARK.test(ch)
        || isVariationSelector(ch)
        || isEmojiModifier(ch)
        || previous!.endsWith('\u200d')
        || ch === '\u200d')

    if (joinsPrevious) {
      segments[segments.length - 1] = `${previous}${ch}`
      regionalIndicators = 0
      continue
    }

    if (REGIONAL_INDICATOR.test(ch)) {
      if (regionalIndicators % 2 === 1 && previous) {
        segments[segments.length - 1] = `${previous}${ch}`
      } else {
        segments.push(ch)
      }
      regionalIndicators += 1
      continue
    }

    regionalIndicators = 0
    segments.push(ch)
  }

  return segments
}

export function splitGraphemes(value: string): string[] {
  if (!value) return []
  if (!segmenter) return fallbackSegments(value)
  return Array.from(segmenter.segment(value), ({ segment }) => segment)
}

export function graphemeBoundaries(value: string): number[] {
  const boundaries = [0]
  let offset = 0
  for (const grapheme of splitGraphemes(value)) {
    offset += grapheme.length
    boundaries.push(offset)
  }
  return boundaries
}

/** Clamp to the grapheme boundary at or before the requested UTF-16 offset. */
export function clampGraphemeOffset(value: string, offset: number): number {
  const requested = Math.max(0, Math.min(value.length, Math.trunc(offset)))
  let result = 0
  for (const boundary of graphemeBoundaries(value)) {
    if (boundary > requested) break
    result = boundary
  }
  return result
}

export function previousGraphemeOffset(value: string, offset: number): number {
  const requested = Math.max(0, Math.min(value.length, Math.trunc(offset)))
  let previous = 0
  for (const boundary of graphemeBoundaries(value)) {
    if (boundary >= requested) return previous
    previous = boundary
  }
  return previous
}

export function nextGraphemeOffset(value: string, offset: number): number {
  const requested = Math.max(0, Math.min(value.length, Math.trunc(offset)))
  for (const boundary of graphemeBoundaries(value)) {
    if (boundary > requested) return boundary
  }
  return value.length
}

export function graphemeAt(value: string, offset: number): string {
  const start = clampGraphemeOffset(value, offset)
  if (start >= value.length) return ''
  return value.slice(start, nextGraphemeOffset(value, start))
}

export function countGraphemes(value: string): number {
  return splitGraphemes(value).length
}
