import {
  clampGraphemeOffset,
  previousGraphemeOffset,
} from './graphemes.js'

const CHOSEONG_COMPAT: Record<string, number> = {
  'ㄱ': 0, 'ㄲ': 1, 'ㄴ': 2, 'ㄷ': 3, 'ㄸ': 4, 'ㄹ': 5, 'ㅁ': 6,
  'ㅂ': 7, 'ㅃ': 8, 'ㅅ': 9, 'ㅆ': 10, 'ㅇ': 11, 'ㅈ': 12, 'ㅉ': 13,
  'ㅊ': 14, 'ㅋ': 15, 'ㅌ': 16, 'ㅍ': 17, 'ㅎ': 18,
}

const JUNGSEONG_COMPAT: Record<string, number> = {
  'ㅏ': 0, 'ㅐ': 1, 'ㅑ': 2, 'ㅒ': 3, 'ㅓ': 4, 'ㅔ': 5, 'ㅕ': 6,
  'ㅖ': 7, 'ㅗ': 8, 'ㅘ': 9, 'ㅙ': 10, 'ㅚ': 11, 'ㅛ': 12, 'ㅜ': 13,
  'ㅝ': 14, 'ㅞ': 15, 'ㅟ': 16, 'ㅠ': 17, 'ㅡ': 18, 'ㅢ': 19, 'ㅣ': 20,
}

const JONGSEONG_COMPAT: Record<string, number> = {
  'ㄱ': 1, 'ㄲ': 2, 'ㄳ': 3, 'ㄴ': 4, 'ㄵ': 5, 'ㄶ': 6, 'ㄷ': 7,
  'ㄹ': 8, 'ㄺ': 9, 'ㄻ': 10, 'ㄼ': 11, 'ㄽ': 12, 'ㄾ': 13, 'ㄿ': 14,
  'ㅀ': 15, 'ㅁ': 16, 'ㅂ': 17, 'ㅄ': 18, 'ㅅ': 19, 'ㅆ': 20, 'ㅇ': 21,
  'ㅈ': 22, 'ㅊ': 23, 'ㅋ': 24, 'ㅌ': 25, 'ㅍ': 26, 'ㅎ': 27,
}

const T_TO_L: Record<number, number | undefined> = {
  1: 0, 2: 1, 4: 2, 7: 3, 8: 5, 16: 6, 17: 7, 19: 9, 20: 10,
  21: 11, 22: 12, 23: 14, 24: 15, 25: 16, 26: 17, 27: 18,
}

const COMPAT_OF_L = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
const COMPAT_OF_V = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']

const COMPOUND_VOWELS: Record<number, Record<number, number | undefined> | undefined> = {
  8: { 0: 9, 1: 10, 20: 11 },
  13: { 4: 14, 5: 15, 20: 16 },
  18: { 20: 19 },
}

const COMPOUND_FINALS: Record<number, Record<number, number | undefined> | undefined> = {
  1: { 19: 3 },
  4: { 22: 5, 27: 6 },
  8: { 1: 9, 16: 10, 17: 11, 19: 12, 25: 13, 26: 14, 27: 15 },
  17: { 19: 18 },
}

const SPLIT_COMPOUND_FINALS: Record<number, { first: number; secondL: number } | undefined> = {
  3: { first: 1, secondL: 9 },
  5: { first: 4, secondL: 12 },
  6: { first: 4, secondL: 18 },
  9: { first: 8, secondL: 0 },
  10: { first: 8, secondL: 6 },
  11: { first: 8, secondL: 7 },
  12: { first: 8, secondL: 9 },
  13: { first: 8, secondL: 16 },
  14: { first: 8, secondL: 17 },
  15: { first: 8, secondL: 18 },
  18: { first: 17, secondL: 9 },
}

const SYLLABLE_BASE = 0xAC00
const SYLLABLE_END = 0xD7A3
const CHOSEONG_BASE = 0x1100
const CHOSEONG_END = 0x1112
const JUNGSEONG_BASE = 0x1161
const JUNGSEONG_END = 0x1175
const JONGSEONG_BASE = 0x11A8
const JONGSEONG_END = 0x11C2
const BRACKETED_PASTE_START = '\x1b[200~'
const BRACKETED_PASTE_END = '\x1b[201~'

function isPrecomposedSyllable(ch: string): boolean {
  const code = ch.codePointAt(0) ?? 0
  return code >= SYLLABLE_BASE && code <= SYLLABLE_END
}

type State =
  | { kind: 'empty' }
  | { kind: 'L'; L: number }
  | { kind: 'V'; V: number }
  | { kind: 'LV'; L: number; V: number }
  | { kind: 'LVT'; L: number; V: number; T: number }

function composeSyllable(L: number, V: number, T = 0): string {
  return String.fromCharCode(SYLLABLE_BASE + L * 588 + V * 28 + T)
}

function decomposeSyllable(ch: string): Exclude<State, { kind: 'empty' | 'L' | 'V' }> {
  const code = (ch.codePointAt(0) ?? SYLLABLE_BASE) - SYLLABLE_BASE
  const L = Math.floor(code / 588)
  const V = Math.floor((code % 588) / 28)
  const T = code % 28
  return T > 0 ? { kind: 'LVT', L, V, T } : { kind: 'LV', L, V }
}

function getL(ch: string): number | undefined {
  const compat = CHOSEONG_COMPAT[ch]
  if (compat !== undefined) return compat

  const code = ch.codePointAt(0) ?? 0
  if (code >= CHOSEONG_BASE && code <= CHOSEONG_END) {
    return code - CHOSEONG_BASE
  }
  return undefined
}

function getV(ch: string): number | undefined {
  const compat = JUNGSEONG_COMPAT[ch]
  if (compat !== undefined) return compat

  const code = ch.codePointAt(0) ?? 0
  if (code >= JUNGSEONG_BASE && code <= JUNGSEONG_END) {
    return code - JUNGSEONG_BASE
  }
  return undefined
}

function getT(ch: string): number | undefined {
  const compat = JONGSEONG_COMPAT[ch]
  if (compat !== undefined) return compat

  const code = ch.codePointAt(0) ?? 0
  if (code >= JONGSEONG_BASE && code <= JONGSEONG_END) {
    return code - JONGSEONG_BASE + 1
  }
  return undefined
}

function combineVowels(left: number, right: number): number | undefined {
  return COMPOUND_VOWELS[left]?.[right]
}

function combineFinals(left: number, right: number): number | undefined {
  return COMPOUND_FINALS[left]?.[right]
}

function flush(state: State, out: string[]): void {
  switch (state.kind) {
    case 'empty':
      return
    case 'L':
      out.push(COMPAT_OF_L[state.L])
      return
    case 'V':
      out.push(COMPAT_OF_V[state.V])
      return
    case 'LV':
      out.push(composeSyllable(state.L, state.V))
      return
    case 'LVT':
      out.push(composeSyllable(state.L, state.V, state.T))
      return
  }
}

function renderState(state: Exclude<State, { kind: 'empty' }>): string {
  const out: string[] = []
  flush(state, out)
  return out.join('')
}

function containsHangul(value: string): boolean {
  return /[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]/u.test(value)
}

function commonPrefixLength(left: string, right: string): number {
  const leftChars = Array.from(left)
  const rightChars = Array.from(right)
  const max = Math.min(leftChars.length, rightChars.length)
  let index = 0
  while (index < max && leftChars[index] === rightChars[index]) {
    index += 1
  }
  return index
}

function looksLikeLineSnapshot(
  previousValue: string,
  inserted: string,
  state: HangulInputState,
): boolean {
  if (!state.active || !previousValue || inserted.length <= 1) return false
  if (!containsHangul(previousValue) || !containsHangul(inserted)) return false
  if (inserted.startsWith(previousValue)) return true

  const previousComposed = composeHangul(previousValue)
  const insertedComposed = composeHangul(inserted)
  return insertedComposed.length >= previousComposed.length
    && commonPrefixLength(previousComposed, insertedComposed) > 0
}

function shouldReplaceActiveWithSyllable(
  active: State,
  syllable: Exclude<State, { kind: 'empty' | 'L' | 'V' }>,
): boolean {
  if (active.kind === 'L') {
    return active.L === syllable.L
  }
  if (active.kind === 'LV') {
    return active.L === syllable.L && active.V === syllable.V
  }
  if (active.kind === 'LVT') {
    return active.L === syllable.L && active.V === syllable.V
  }
  return false
}

function reduceActiveWithChar(
  committed: string[],
  active: State,
  ch: string,
): State {
  const L = getL(ch)
  const V = getV(ch)
  const T = getT(ch)

  if (isPrecomposedSyllable(ch)) {
    const syllable = decomposeSyllable(ch)
    if (shouldReplaceActiveWithSyllable(active, syllable)) {
      return syllable
    }
    flush(active, committed)
    committed.push(ch)
    return { kind: 'empty' }
  }

  switch (active.kind) {
    case 'empty':
      if (L !== undefined) return { kind: 'L', L }
      if (V !== undefined) return { kind: 'V', V }
      committed.push(ch)
      return { kind: 'empty' }
    case 'V':
      if (V !== undefined) {
        const combined = combineVowels(active.V, V)
        if (combined !== undefined) return { kind: 'V', V: combined }
        flush(active, committed)
        return { kind: 'V', V }
      }
      if (L !== undefined) {
        flush(active, committed)
        return { kind: 'L', L }
      }
      flush(active, committed)
      committed.push(ch)
      return { kind: 'empty' }
    case 'L':
      if (V !== undefined) return { kind: 'LV', L: active.L, V }
      if (L !== undefined) {
        flush(active, committed)
        return { kind: 'L', L }
      }
      flush(active, committed)
      committed.push(ch)
      return { kind: 'empty' }
    case 'LV':
      if (V !== undefined) {
        const combined = combineVowels(active.V, V)
        if (combined !== undefined) return { kind: 'LV', L: active.L, V: combined }
        flush(active, committed)
        return { kind: 'V', V }
      }
      if (T !== undefined) return { kind: 'LVT', L: active.L, V: active.V, T }
      if (L !== undefined) {
        flush(active, committed)
        return { kind: 'L', L }
      }
      flush(active, committed)
      committed.push(ch)
      return { kind: 'empty' }
    case 'LVT':
      if (V !== undefined) {
        const split = SPLIT_COMPOUND_FINALS[active.T]
        if (split) {
          committed.push(composeSyllable(active.L, active.V, split.first))
          return { kind: 'LV', L: split.secondL, V }
        }
        const demoted: number | undefined = T_TO_L[active.T]
        if (demoted !== undefined) {
          committed.push(composeSyllable(active.L, active.V))
          return { kind: 'LV', L: demoted, V }
        }
        flush(active, committed)
        return { kind: 'V', V }
      }
      if (L !== undefined) {
        if (T !== undefined) {
          const combined = combineFinals(active.T, T)
          if (combined !== undefined) {
            return { kind: 'LVT', L: active.L, V: active.V, T: combined }
          }
        }
        flush(active, committed)
        return { kind: 'L', L }
      }
      flush(active, committed)
      committed.push(ch)
      return { kind: 'empty' }
  }
}

export interface HangulInputState {
  active: Exclude<State, { kind: 'empty' }> | null
}

export function createHangulInputState(): HangulInputState {
  return { active: null }
}

export function stripBracketedPasteDelimiters(input: string): string {
  let next = input
    .replaceAll(BRACKETED_PASTE_START, '')
    .replaceAll(BRACKETED_PASTE_END, '')

  if (next.startsWith('[200~') || next.startsWith('[201~')) {
    next = next.slice(5)
  }

  return next
}

export function applyHangulInputChange(
  previousValue: string,
  nextRawValue: string,
  state: HangulInputState,
): { value: string; state: HangulInputState } {
  if (!nextRawValue.startsWith(previousValue)) {
    return {
      value: composeHangul(nextRawValue),
      state: createHangulInputState(),
    }
  }

  const inserted = nextRawValue.slice(previousValue.length)
  if (!inserted) {
    return { value: previousValue, state }
  }

  if (looksLikeLineSnapshot(previousValue, inserted, state)) {
    return applyHangulInputChange('', inserted, createHangulInputState())
  }

  const activeText = state.active ? renderState(state.active) : ''
  const hasActiveTail = Boolean(activeText) && previousValue.endsWith(activeText)
  const committed = hasActiveTail
    ? [previousValue.slice(0, -activeText.length)]
    : [previousValue]
  let active: State = hasActiveTail && state.active ? state.active : { kind: 'empty' }

  for (const ch of inserted.normalize('NFC')) {
    active = reduceActiveWithChar(committed, active, ch)
  }

  const nextActive = active.kind === 'empty' ? null : active
  return {
    value: `${committed.join('')}${nextActive ? renderState(nextActive) : ''}`,
    state: { active: nextActive },
  }
}

export interface HangulInputChunkResult {
  value: string
  cursorOffset: number
  state: HangulInputState
}

function hasInlineInputControl(input: string): boolean {
  for (const ch of input) {
    if (ch === '\x7f' || ch === '\b' || ch < ' ') return true
  }
  return false
}

/**
 * Apply one Ink input callback, including IME replacement chunks such as
 * `초<DEL>최근`. Ink only sets `key.delete` when DEL is the whole callback;
 * when an IME batches it with replacement text we must replay the control
 * byte ourselves instead of persisting it as prompt content.
 */
export function applyHangulInputChunk(
  value: string,
  cursorOffset: number,
  input: string,
  state: HangulInputState,
): HangulInputChunkResult {
  let nextValue = value
  let nextCursorOffset = clampGraphemeOffset(value, cursorOffset)
  let nextState = state

  if (!hasInlineInputControl(input)) {
    const beforeCursor = nextValue.slice(0, nextCursorOffset)
    const afterCursor = nextValue.slice(nextCursorOffset)
    const applied = applyHangulInputChange(
      beforeCursor,
      `${beforeCursor}${input}`,
      nextState,
    )
    return {
      value: `${applied.value}${afterCursor}`,
      cursorOffset: applied.value.length,
      state: applied.state,
    }
  }

  for (const ch of input) {
    if (ch === '\x7f' || ch === '\b') {
      if (nextCursorOffset > 0) {
        const previousOffset = previousGraphemeOffset(nextValue, nextCursorOffset)
        nextValue = `${nextValue.slice(0, previousOffset)}${nextValue.slice(nextCursorOffset)}`
        nextCursorOffset = previousOffset
      }
      nextState = createHangulInputState()
      continue
    }

    if (ch < ' ' && ch !== '\n') {
      nextState = createHangulInputState()
      continue
    }

    const beforeCursor = nextValue.slice(0, nextCursorOffset)
    const afterCursor = nextValue.slice(nextCursorOffset)
    const applied = applyHangulInputChange(
      beforeCursor,
      `${beforeCursor}${ch}`,
      nextState,
    )
    nextValue = `${applied.value}${afterCursor}`
    nextCursorOffset = applied.value.length
    nextState = applied.state
  }

  return {
    value: nextValue,
    cursorOffset: nextCursorOffset,
    state: nextState,
  }
}

/** Replay embedded terminal editing controls in already-buffered input. */
export function replayInlineInputControls(input: string): string {
  let value = ''
  for (const ch of input) {
    if (ch === '\x7f' || ch === '\b') {
      value = value.slice(0, previousGraphemeOffset(value, value.length))
      continue
    }
    if (ch < ' ' && ch !== '\n' && ch !== '\t') continue
    value += ch
  }
  return value
}

export function composeHangul(input: string | null | undefined): string {
  if (input == null || input === '') return ''

  try {
    const normalized = input.normalize('NFC')
    const out: string[] = []
    let state: State = { kind: 'empty' }

    for (const ch of normalized) {
      const L = getL(ch)
      const V = getV(ch)
      const T = getT(ch)

      if (isPrecomposedSyllable(ch)) {
        const syllable = decomposeSyllable(ch)
        if (state.kind === 'L' && state.L === syllable.L) {
          state = syllable
        } else {
          flush(state, out)
          state = syllable
        }
        continue
      }

      switch (state.kind) {
        case 'empty':
          if (L !== undefined) state = { kind: 'L', L }
          else if (V !== undefined) state = { kind: 'V', V }
          else out.push(ch)
          break
        case 'V':
          if (V !== undefined) {
            const combined = combineVowels(state.V, V)
            if (combined !== undefined) state = { kind: 'V', V: combined }
            else {
              flush(state, out)
              state = { kind: 'V', V }
            }
          } else if (L !== undefined) {
            flush(state, out)
            state = { kind: 'L', L }
          } else {
            flush(state, out)
            out.push(ch)
            state = { kind: 'empty' }
          }
          break
        case 'L':
          if (V !== undefined) state = { kind: 'LV', L: state.L, V }
          else if (L !== undefined) {
            flush(state, out)
            state = { kind: 'L', L }
          } else {
            flush(state, out)
            out.push(ch)
            state = { kind: 'empty' }
          }
          break
        case 'LV':
          if (V !== undefined) {
            const combined = combineVowels(state.V, V)
            if (combined !== undefined) state = { kind: 'LV', L: state.L, V: combined }
            else {
              flush(state, out)
              state = { kind: 'V', V }
            }
          } else if (T !== undefined) {
            state = { kind: 'LVT', L: state.L, V: state.V, T }
          } else if (L !== undefined) {
            flush(state, out)
            state = { kind: 'L', L }
          } else {
            flush(state, out)
            out.push(ch)
            state = { kind: 'empty' }
          }
          break
        case 'LVT':
          if (V !== undefined) {
            const split: { first: number; secondL: number } | undefined =
              SPLIT_COMPOUND_FINALS[state.T]
            if (split) {
              out.push(composeSyllable(state.L, state.V, split.first))
              state = { kind: 'LV', L: split.secondL, V }
              break
            }
            const demoted: number | undefined = T_TO_L[state.T]
            if (demoted !== undefined) {
              out.push(composeSyllable(state.L, state.V))
              state = { kind: 'LV', L: demoted, V }
            } else {
              flush(state, out)
              state = { kind: 'V', V }
            }
          } else if (L !== undefined) {
            const combined: number | undefined =
              T === undefined ? undefined : combineFinals(state.T, T)
            if (combined !== undefined) {
              state = { kind: 'LVT', L: state.L, V: state.V, T: combined }
            } else {
              flush(state, out)
              state = { kind: 'L', L }
            }
          } else {
            flush(state, out)
            out.push(ch)
            state = { kind: 'empty' }
          }
          break
      }
    }

    flush(state, out)
    return out.join('')
  } catch (err) {
    console.error('composeHangul failed; returning input as-is', err)
    return input
  }
}
