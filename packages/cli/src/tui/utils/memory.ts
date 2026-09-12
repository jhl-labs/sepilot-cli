export function compactMemoryContent(
  content: string,
  maxLength = 72,
): string {
  const normalized = content.replace(/\s+/g, ' ').trim()
  if (normalized.length <= maxLength) {
    return normalized
  }
  return `${normalized.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`
}

export function formatMemoryScore(score?: number): string {
  if (score == null || !Number.isFinite(score)) {
    return '--'
  }

  const clamped = Math.max(0, Math.min(1, score))
  return `${Math.round(clamped * 100)}%`
}

export function summarizeMemoryTags(
  tags: string[],
  limit = 2,
): string {
  if (tags.length === 0) {
    return ''
  }

  const visible = tags.slice(0, Math.max(0, limit)).map((tag) => `#${tag}`)
  const remaining = tags.length - visible.length
  return remaining > 0
    ? `${visible.join(' ')} +${remaining}`
    : visible.join(' ')
}

export const TUI_OPEN_LOOP_QUEUE_SECTION = 'Open Loop Queue'
const TUI_DAILY_BACKLOG_SECTION = 'Backlog'
const TUI_DAILY_REFLECTION_SECTION = 'Reflection Ledger'

export interface TuiFileMemorySnapshotLike {
  sections: Array<{ title: string; content: string }>
  todayNote: string
}

export function buildTuiMemoryBacklogSnapshot(
  snapshot: TuiFileMemorySnapshotLike,
) {
  return {
    openLoopQueue: splitMemoryDisplayLines(findMemorySection(snapshot, TUI_OPEN_LOOP_QUEUE_SECTION)),
    todayBacklog: splitMemoryDisplayLines(extractMarkdownSection(snapshot.todayNote, TUI_DAILY_BACKLOG_SECTION)),
    todayReflections: splitMemoryDisplayLines(extractMarkdownSection(snapshot.todayNote, TUI_DAILY_REFLECTION_SECTION)),
  }
}

export function findMemorySection(
  snapshot: Pick<TuiFileMemorySnapshotLike, 'sections'>,
  title: string,
): string {
  return snapshot.sections.find((section) => section.title === title)?.content ?? ''
}

export function extractMarkdownSection(markdown: string, title: string): string {
  const lines = markdown.split(/\r?\n/)
  const start = lines.findIndex((line) => line.trim() === `## ${title}`)
  if (start < 0) return ''
  const collected: string[] = []
  for (const line of lines.slice(start + 1)) {
    if (/^##\s+\S/.test(line)) break
    collected.push(line)
  }
  return collected.join('\n').trim()
}

function splitMemoryDisplayLines(content: string): string[] {
  return content.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
}

export function appendUniqueMemoryLine(content: string, line: string): string {
  const lines = splitMemoryDisplayLines(content)
  if (lines.includes(line)) return lines.join('\n')
  return [...lines, line].join('\n')
}

export function removeMatchingMemoryLines(
  content: string,
  query: string,
): { removed: string[]; remaining: string } {
  const normalizedQuery = query.trim().toLowerCase()
  const removed: string[] = []
  const remaining: string[] = []
  for (const line of splitMemoryDisplayLines(content)) {
    if (normalizedQuery && line.toLowerCase().includes(normalizedQuery)) {
      removed.push(line)
    } else {
      remaining.push(line)
    }
  }
  return { removed, remaining: remaining.join('\n') }
}

export function formatManualMemoryBacklogLine(content: string): string {
  return `- ${new Date().toISOString()} source=manual: ${content} | status=needs_follow_up | next=Resume this backlog item when relevant.`
}
