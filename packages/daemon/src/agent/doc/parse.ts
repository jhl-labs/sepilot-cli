// Lightweight markdown heading outline. doc.outline tool과
// doc.replace_section / doc.insert_after_section의 range 계산용.
//
// 의도적으로 의존성 없음. headings만 추적하면 충분 (코드 블록 fenced 안의 # 제외).

import type { DocOutlineEntry } from '@sepilotd/core'

const HEADING_RE = /^(#{1,6})\s+(.+?)\s*$/
const FENCE_RE = /^(```|~~~)/

export class AmbiguousSectionError extends Error {
  constructor(
    readonly selector: string,
    readonly candidates: DocOutlineEntry[],
  ) {
    super(`ambiguous section: ${selector}`)
    this.name = 'AmbiguousSectionError'
  }
}

export function parseOutline(content: string): DocOutlineEntry[] {
  const lines = content.split('\n')
  const headings: Array<{
    level: number
    title: string
    lineStart: number
  }> = []

  let inFence = false
  let offset = 0
  const lineStarts: number[] = []
  for (let i = 0; i < lines.length; i++) {
    lineStarts.push(offset)
    offset += lines[i].length + 1 // +1 newline (마지막 newline은 정확치 않지만 outline 용도엔 OK)
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (FENCE_RE.test(line)) {
      inFence = !inFence
      continue
    }
    if (inFence) continue
    const m = HEADING_RE.exec(line)
    if (!m) continue
    headings.push({
      level: m[1].length,
      title: m[2],
      lineStart: lineStarts[i],
    })
  }

  const entries: DocOutlineEntry[] = []
  for (let i = 0; i < headings.length; i++) {
    const h = headings[i]
    let end = content.length
    for (let j = i + 1; j < headings.length; j++) {
      // 다음 같은 또는 상위 레벨 heading 직전까지가 section 끝
      if (headings[j].level <= h.level) {
        end = headings[j].lineStart
        break
      }
    }
    entries.push({
      level: h.level,
      title: h.title,
      start: h.lineStart,
      end,
      index: i,
    })
  }
  return entries
}

/** title 매칭(완전일치 → case-insensitive) 또는 index로 outline entry 찾기 */
export function findSection(
  outline: DocOutlineEntry[],
  selector: string | number,
): DocOutlineEntry | null {
  if (typeof selector === 'number') {
    return outline[selector] ?? null
  }
  const exact = outline.filter((e) => e.title === selector)
  if (exact.length === 1) return exact[0]
  if (exact.length > 1) throw new AmbiguousSectionError(selector, exact)
  const lower = selector.toLowerCase()
  const insensitive = outline.filter((e) => e.title.toLowerCase() === lower)
  if (insensitive.length === 1) return insensitive[0]
  if (insensitive.length > 1) {
    throw new AmbiguousSectionError(selector, insensitive)
  }
  return null
}
