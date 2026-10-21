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
    offset += lines[i].length + 1
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
