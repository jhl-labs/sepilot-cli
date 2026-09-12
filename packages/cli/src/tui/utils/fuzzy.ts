export interface FuzzyResult {
  score: number
  indices: number[]
}

const SCORE_MATCH = 16
const SCORE_GAP_LEADING = -5
const SCORE_GAP_TRAILING = -3
const SCORE_GAP_INNER = -4
const BONUS_BOUNDARY = 8
const BONUS_CAMEL = 12
const BONUS_CONSECUTIVE = 12
const BONUS_FIRST_CHAR = 8
const WORD_SEPARATORS = new Set(['/', '\\', '.', '-', '_', ' '])

function isLower(ch: string): boolean {
  return ch >= 'a' && ch <= 'z'
}

function isUpper(ch: string): boolean {
  return ch >= 'A' && ch <= 'Z'
}

function isBoundary(prev: string | undefined, current: string): boolean {
  if (prev === undefined) return true
  if (WORD_SEPARATORS.has(prev)) return true
  if (isLower(prev) && isUpper(current)) return true
  return false
}

function charBonus(i: number, target: string): number {
  let bonus = SCORE_MATCH
  if (i === 0) bonus += BONUS_FIRST_CHAR
  // camelCase is treated as a stronger boundary; the two are mutually exclusive
  // to avoid double-counting a single position.
  if (i > 0 && isLower(target[i - 1]) && isUpper(target[i])) {
    bonus += BONUS_CAMEL
  } else if (isBoundary(target[i - 1], target[i])) {
    bonus += BONUS_BOUNDARY
  }
  return bonus
}

export function fuzzyScore(query: string, target: string): FuzzyResult | null {
  if (query.length === 0) return { score: 0, indices: [] }
  if (query.length > target.length) return null

  const queryLower = query.toLowerCase()
  const targetLower = target.toLowerCase()
  const m = query.length
  const n = target.length

  const NEG_INF = -Infinity
  // dp[qi][ti] = best cumulative score matching first qi+1 query chars, last match at ti
  const dp: number[][] = Array.from({ length: m }, () => Array(n).fill(NEG_INF))
  const prevPos: number[][] = Array.from({ length: m }, () => Array(n).fill(-1))

  // Fill first query character
  for (let ti = 0; ti < n; ti += 1) {
    if (targetLower[ti] !== queryLower[0]) continue
    dp[0][ti] = charBonus(ti, target) + ti * SCORE_GAP_LEADING
  }

  // Fill remaining query characters
  for (let qi = 1; qi < m; qi += 1) {
    for (let ti = qi; ti < n; ti += 1) {
      if (targetLower[ti] !== queryLower[qi]) continue
      const matchBonus = charBonus(ti, target)
      for (let prevTi = qi - 1; prevTi < ti; prevTi += 1) {
        if (dp[qi - 1][prevTi] === NEG_INF) continue
        const innerGap = (ti - prevTi - 1) * SCORE_GAP_INNER
        const consecutive = prevTi === ti - 1 ? BONUS_CONSECUTIVE : 0
        const candidate = dp[qi - 1][prevTi] + matchBonus + innerGap + consecutive
        if (candidate > dp[qi][ti]) {
          dp[qi][ti] = candidate
          prevPos[qi][ti] = prevTi
        }
      }
    }
  }

  // Find best ending position with trailing gap penalty
  let bestScore = NEG_INF
  let bestTi = -1
  for (let ti = m - 1; ti < n; ti += 1) {
    if (dp[m - 1][ti] === NEG_INF) continue
    const trailingGap = (n - 1 - ti) * SCORE_GAP_TRAILING
    const total = dp[m - 1][ti] + trailingGap
    if (total > bestScore) {
      bestScore = total
      bestTi = ti
    }
  }

  if (bestTi === -1) return null

  // Backtrack to collect matched indices
  const indices: number[] = Array(m)
  let ti = bestTi
  for (let qi = m - 1; qi >= 0; qi -= 1) {
    indices[qi] = ti
    if (qi > 0) ti = prevPos[qi][ti]
  }

  return { score: bestScore, indices }
}
