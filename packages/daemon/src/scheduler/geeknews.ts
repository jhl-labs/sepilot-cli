import type { ScheduledJob } from './job-store.js'

export const GEEKNEWS_INSIGHTS_SOURCE_TYPE = 'geeknews.insights'
export const GEEKNEWS_FEED_URL = 'https://news.hada.io/rss/news'

const DEFAULT_MAX_ITEMS = 8
const MAX_ITEMS_LIMIT = 20
const SEEN_ID_LIMIT = 500
const FETCH_TIMEOUT_MS = 20_000

export interface GeekNewsItem {
  id: string
  title: string
  url: string
  publishedAt: string | null
  content: string
}

interface GeekNewsSourceMetadata {
  type: typeof GEEKNEWS_INSIGHTS_SOURCE_TYPE
  feedUrl?: string
  maxItems?: number
  seenItemIds?: unknown
  lastCheckedAt?: string
  lastDeliveredAt?: string
  lastDeliveredCount?: number
  lastSkippedCount?: number
}

interface GeekNewsRunDeps {
  runAgent: (instruction: string) => Promise<string | void>
  onStatus?: (message: string) => Promise<void> | void
  fetch?: typeof fetch
  now?: () => Date
  signal?: AbortSignal
}

export interface GeekNewsRunResult {
  output?: string
  metadata: Record<string, unknown> | null
  hasNewItems: boolean
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function sourceFromJob(job: ScheduledJob): GeekNewsSourceMetadata | null {
  const source = job.metadata?.source
  if (!isRecord(source) || source.type !== GEEKNEWS_INSIGHTS_SOURCE_TYPE) return null
  return source as unknown as GeekNewsSourceMetadata
}

export function isGeekNewsInsightsJob(job: ScheduledJob): boolean {
  return Boolean(sourceFromJob(job))
}

function itemId(item: GeekNewsItem): string {
  return item.id || item.url
}

function normalizeMaxItems(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return DEFAULT_MAX_ITEMS
  return Math.max(1, Math.min(MAX_ITEMS_LIMIT, Math.round(value)))
}

function normalizeSeenIds(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  const out: string[] = []
  const seen = new Set<string>()
  for (const item of value) {
    if (typeof item !== 'string') continue
    const trimmed = item.trim()
    if (!trimmed || seen.has(trimmed)) continue
    seen.add(trimmed)
    out.push(trimmed)
  }
  return out.slice(0, SEEN_ID_LIMIT)
}

function mergeSeenIds(currentFeedIds: string[], existingIds: string[]): string[] {
  const next: string[] = []
  const seen = new Set<string>()
  for (const id of [...currentFeedIds, ...existingIds]) {
    if (!id || seen.has(id)) continue
    seen.add(id)
    next.push(id)
    if (next.length >= SEEN_ID_LIMIT) break
  }
  return next
}

function withGeekNewsSource(
  metadata: Record<string, unknown> | null,
  source: GeekNewsSourceMetadata,
): Record<string, unknown> {
  return {
    ...(metadata ?? {}),
    source,
  }
}

function readTag(xml: string, tag: string): string {
  const match = new RegExp(`<${tag}\\b[^>]*>([\\s\\S]*?)<\\/${tag}>`, 'i').exec(xml)
  return match ? decodeXml(match[1]) : ''
}

function readAlternateLink(xml: string): string {
  const alternate = /<link\b(?=[^>]*\brel=['"]alternate['"])[^>]*\bhref=['"]([^'"]+)['"][^>]*\/?>/i.exec(xml)
  if (alternate?.[1]) return decodeXml(alternate[1])
  const anyLink = /<link\b[^>]*\bhref=['"]([^'"]+)['"][^>]*\/?>/i.exec(xml)
  return anyLink?.[1] ? decodeXml(anyLink[1]) : ''
}

function decodeXml(value: string): string {
  return value
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x([0-9a-f]+);/gi, (_, hex: string) =>
      String.fromCodePoint(Number.parseInt(hex, 16)),
    )
    .replace(/&#(\d+);/g, (_, num: string) =>
      String.fromCodePoint(Number.parseInt(num, 10)),
    )
    .trim()
}

function htmlToText(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, '')
    .replace(/<style[\s\S]*?<\/style>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

export function parseGeekNewsAtomFeed(xml: string): GeekNewsItem[] {
  const entries = xml.match(/<entry\b[\s\S]*?<\/entry>/gi) ?? []
  const items: GeekNewsItem[] = []
  const seen = new Set<string>()

  for (const entry of entries) {
    const title = readTag(entry, 'title')
    const id = readTag(entry, 'id')
    const url = readAlternateLink(entry) || id
    const publishedAt = readTag(entry, 'published') || readTag(entry, 'updated') || null
    const content = htmlToText(readTag(entry, 'content') || readTag(entry, 'summary'))
    const stableId = id || url
    if (!title || !url || !stableId || seen.has(stableId)) continue
    seen.add(stableId)
    items.push({
      id: stableId,
      title,
      url,
      publishedAt,
      content,
    })
  }

  return items
}

async function fetchGeekNewsItems(
  fetchFn: typeof fetch,
  shutdownSignal?: AbortSignal,
): Promise<GeekNewsItem[]> {
  const controller = new AbortController()
  const abortForShutdown = (): void => controller.abort(shutdownSignal?.reason)
  shutdownSignal?.throwIfAborted()
  shutdownSignal?.addEventListener('abort', abortForShutdown, { once: true })
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS)
  try {
    const response = await fetchFn(GEEKNEWS_FEED_URL, {
      signal: controller.signal,
      headers: { 'user-agent': 'sepilotd-geeknews-insights/1' },
    })
    if (!response.ok) {
      throw new Error(`GeekNews feed fetch failed (${response.status})`)
    }
    const xml = await response.text()
    const items = parseGeekNewsAtomFeed(xml)
    if (items.length === 0) {
      throw new Error('GeekNews feed did not contain any Atom entries')
    }
    return items
  } finally {
    clearTimeout(timeout)
    shutdownSignal?.removeEventListener('abort', abortForShutdown)
  }
}

function buildInsightInstruction(items: GeekNewsItem[], checkedAt: string): string {
  const lines = items.map((item, index) => [
    `### ${index + 1}. ${item.title}`,
    `- URL: ${item.url}`,
    item.publishedAt ? `- Published: ${item.publishedAt}` : '',
    item.content ? `- Feed excerpt: ${item.content.slice(0, 900)}` : '',
  ].filter(Boolean).join('\n'))

  return [
    'GeekNews RSS에서 이번 실행에 처음 발견된 최신 글만 정리한다.',
    `수집 시각: ${checkedAt}`,
    '',
    '규칙:',
    '- 아래 제공된 항목만 사용하고 웹 추가 조회는 하지 않는다.',
    '- 한국어로 작성한다.',
    '- 각 글마다 제목 링크, 핵심 요약 1문장, 실무/제품/개발 관점 인사이트 1-2개를 쓴다.',
    '- 마지막에 "오늘 볼 흐름" 섹션으로 공통 트렌드나 추적할 질문 2-3개를 정리한다.',
    '- 중복 전달 방지는 런타임이 처리했으니 이전 글을 다시 언급하지 않는다.',
    '',
    lines.join('\n\n'),
  ].join('\n')
}

async function emitStatus(
  deps: GeekNewsRunDeps,
  message: string,
): Promise<void> {
  deps.signal?.throwIfAborted()
  try {
    await deps.onStatus?.(message)
  } catch {
    // Chat progress is best-effort. The scheduler run itself should not fail
    // because a desktop session disappeared while the background job was running.
  }
  deps.signal?.throwIfAborted()
}

export async function runGeekNewsInsightsJob(
  job: ScheduledJob,
  deps: GeekNewsRunDeps,
): Promise<GeekNewsRunResult | undefined> {
  const source = sourceFromJob(job)
  if (!source) return undefined
  deps.signal?.throwIfAborted()

  await emitStatus(
    deps,
    `[예약 작업 진행] ${job.name}\nGeekNews RSS를 확인하고 있습니다.`,
  )

  const now = deps.now?.() ?? new Date()
  const checkedAt = now.toISOString()
  const maxItems = normalizeMaxItems(source.maxItems)
  const existingSeenIds = normalizeSeenIds(source.seenItemIds)
  const existingSeen = new Set(existingSeenIds)
  const items = await fetchGeekNewsItems(deps.fetch ?? fetch, deps.signal)
  deps.signal?.throwIfAborted()
  const currentFeedIds = items.map(itemId)
  const newItems = items
    .filter((item) => !existingSeen.has(itemId(item)))
    .slice(0, maxItems)
  const nextSeenItemIds = mergeSeenIds(currentFeedIds, existingSeenIds)
  const skippedCount = Math.max(0, items.filter((item) => !existingSeen.has(itemId(item))).length - newItems.length)
  const baseSource: GeekNewsSourceMetadata = {
    ...source,
    type: GEEKNEWS_INSIGHTS_SOURCE_TYPE,
    feedUrl: GEEKNEWS_FEED_URL,
    maxItems,
    seenItemIds: nextSeenItemIds,
    lastCheckedAt: checkedAt,
    lastSkippedCount: skippedCount,
  }

  if (newItems.length === 0) {
    await emitStatus(
      deps,
      `[예약 작업 완료] ${job.name}\nGeekNews RSS를 확인했지만 새 글이 없어 이번 실행에서는 새 인사이트를 만들지 않았습니다.`,
    )
    return {
      metadata: withGeekNewsSource(job.metadata, {
        ...baseSource,
        lastDeliveredCount: 0,
      }),
      hasNewItems: false,
    }
  }

  await emitStatus(
    deps,
    `[예약 작업 진행] ${job.name}\n새 GeekNews 글 ${newItems.length}개를 발견했습니다. 인사이트를 생성합니다.`,
  )

  const output = await deps.runAgent(buildInsightInstruction(newItems, checkedAt))
  deps.signal?.throwIfAborted()
  return {
    output: typeof output === 'string' ? output : undefined,
    metadata: withGeekNewsSource(job.metadata, {
      ...baseSource,
      lastDeliveredAt: checkedAt,
      lastDeliveredCount: newItems.length,
    }),
    hasNewItems: true,
  }
}
