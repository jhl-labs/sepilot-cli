import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { homedir } from 'node:os'
import chalk from 'chalk'
import { output, getOutputFormat } from '../output/formatter.js'

interface TraceEntry {
  timestamp?: string
  event?: string
  source?: string
  mode?: string
  graphId?: string
  node?: string
  sessionId?: string
  provider?: string
  model?: string
  iteration?: number
  sequence?: number
  runId?: string
  turnId?: string
  requestId?: string
  toolCallId?: string
  executionId?: string
  status?: string
  durationMs?: number
  usage?: { inputTokens?: number; outputTokens?: number; estimatedCost?: number }
  output?: string
  error?: string
  request?: {
    model?: string
    messages?: Array<{ role?: string; content?: unknown; toolCalls?: unknown }>
    systemPrompt?: string
    tools?: Array<{ name?: string }>
    temperature?: number
    maxTokens?: number
  }
  response?: {
    message?: { role?: string; content?: unknown; toolCalls?: unknown }
    thinking?: string
    usage?: { inputTokens?: number; outputTokens?: number }
    finishReason?: string
  }
  data?: Record<string, unknown>
  meta?: Record<string, unknown>
  [k: string]: unknown
}

export interface TraceCommandOptions {
  last?: string
  session?: string
  event?: string
  since?: string
  until?: string
  full?: boolean
  grep?: string
  file?: string
  stats?: boolean
}

function defaultTracePath(): string {
  const dataDir = process.env.SEPILOTD_DATA_DIR?.trim()
  return join(dataDir || join(homedir(), '.sepilotd'), 'logs', 'agent-trace.jsonl')
}

async function readTraceFiles(path: string): Promise<string> {
  const parts: string[] = []
  for (const candidate of [`${path}.3`, `${path}.2`, `${path}.1`, path]) {
    try {
      parts.push(await readFile(candidate, 'utf-8'))
    } catch {
      // Rotation files are optional. The caller handles the all-missing case.
    }
  }
  if (parts.length === 0) {
    throw new Error('trace-not-found')
  }
  return parts.join('\n')
}

function parseEntries(content: string): TraceEntry[] {
  const entries: TraceEntry[] = []
  for (const line of content.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed) continue
    try {
      entries.push(JSON.parse(trimmed) as TraceEntry)
    } catch {
      // Skip malformed lines rather than aborting — a partial last
      // write or a manual edit shouldn't make the whole trace unreadable.
    }
  }
  return entries
}

function tsMs(entry: TraceEntry): number {
  const t = entry.timestamp ? Date.parse(entry.timestamp) : NaN
  return Number.isFinite(t) ? t : 0
}

function shortId(id: string | undefined): string {
  if (!id) return '—'
  return id.length > 8 ? id.slice(0, 8) : id
}

function previewContent(content: unknown, limit = 120): string {
  if (typeof content === 'string') {
    const oneLine = content.replace(/\s+/g, ' ').trim()
    return oneLine.length > limit ? `${oneLine.slice(0, limit - 1)}…` : oneLine
  }
  if (Array.isArray(content)) {
    const text = content
      .map((p) =>
        p && typeof p === 'object' && 'text' in p
          ? String((p as { text: unknown }).text ?? '')
          : `[${(p as { type?: string })?.type ?? 'part'}]`,
      )
      .join(' ')
    return previewContent(text, limit)
  }
  if (content == null) return ''
  return previewContent(JSON.stringify(content), limit)
}

function tokenLabel(usage: { inputTokens?: number; outputTokens?: number } | undefined): string {
  if (!usage) return ''
  const i = usage.inputTokens ?? 0
  const o = usage.outputTokens ?? 0
  if (i === 0 && o === 0) return ''
  return chalk.gray(`(${i}↑/${o}↓)`)
}

function applyFilters(entries: TraceEntry[], opts: TraceCommandOptions): TraceEntry[] {
  let result = entries
  if (opts.session) {
    const s = opts.session
    result = result.filter((e) => e.sessionId === s || (e.sessionId ?? '').startsWith(s))
  }
  if (opts.event) {
    const ev = opts.event.toLowerCase()
    result = result.filter((e) => (e.event ?? '').toLowerCase() === ev)
  }
  if (opts.since) {
    const cutoff = Date.parse(opts.since)
    if (Number.isFinite(cutoff)) result = result.filter((e) => tsMs(e) >= cutoff)
  }
  if (opts.until) {
    const cutoff = Date.parse(opts.until)
    if (Number.isFinite(cutoff)) result = result.filter((e) => tsMs(e) <= cutoff)
  }
  if (opts.grep) {
    const needle = opts.grep.toLowerCase()
    result = result.filter((e) => JSON.stringify(e).toLowerCase().includes(needle))
  }
  return result
}

function printStats(entries: TraceEntry[]): void {
  if (entries.length === 0) {
    console.log(chalk.gray('No trace entries match the filters.'))
    return
  }
  const byEvent = new Map<string, number>()
  const bySession = new Map<string, number>()
  const byModel = new Map<string, number>()
  let errors = 0
  let firstTs = Infinity
  let lastTs = -Infinity
  let inTokens = 0
  let outTokens = 0
  const hasRunUsage = entries.some((entry) =>
    entry.event === 'agent.run'
    && entry.usage
    && (
      typeof entry.usage.inputTokens === 'number'
      || typeof entry.usage.outputTokens === 'number'
    ),
  )
  for (const e of entries) {
    byEvent.set(e.event ?? 'unknown', (byEvent.get(e.event ?? 'unknown') ?? 0) + 1)
    if (e.sessionId) bySession.set(e.sessionId, (bySession.get(e.sessionId) ?? 0) + 1)
    if (e.event === 'llm.call' && e.model) {
      byModel.set(e.model, (byModel.get(e.model) ?? 0) + 1)
    }
    if (e.error || e.status === 'error' || e.status === 'failed') errors++
    const t = tsMs(e)
    if (t > 0) {
      firstTs = Math.min(firstTs, t)
      lastTs = Math.max(lastTs, t)
    }
    const shouldCountUsage = hasRunUsage
      ? e.event === 'agent.run'
      : e.event === 'llm.call'
    if (shouldCountUsage) {
      const u = e.usage ?? e.response?.usage
      inTokens += typeof u?.inputTokens === 'number' ? u.inputTokens : 0
      outTokens += typeof u?.outputTokens === 'number' ? u.outputTokens : 0
    }
  }
  console.log(chalk.bold(`Trace stats — ${entries.length} entries`))
  if (Number.isFinite(firstTs) && Number.isFinite(lastTs)) {
    console.log(`  range: ${chalk.gray(new Date(firstTs).toISOString())} → ${chalk.gray(new Date(lastTs).toISOString())}`)
  }
  console.log(`  errors: ${errors > 0 ? chalk.red(String(errors)) : chalk.gray('0')}`)
  if (inTokens || outTokens) console.log(`  tokens: ${chalk.gray(`${inTokens}↑ / ${outTokens}↓`)}`)
  console.log('  by event:')
  for (const [k, v] of [...byEvent.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`    ${k.padEnd(14)} ${v}`)
  }
  if (byModel.size > 0) {
    console.log('  by model:')
    for (const [k, v] of [...byModel.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10)) {
      console.log(`    ${k.padEnd(28)} ${v}`)
    }
  }
  console.log(`  sessions: ${bySession.size}`)
  for (const [k, v] of [...bySession.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10)) {
    console.log(`    ${shortId(k)}…  ${v}`)
  }
}

function printEntry(entry: TraceEntry, full: boolean): void {
  const ts = chalk.gray(entry.timestamp ?? '')
  const ev = entry.event ?? '?'
  const sess = chalk.cyan(shortId(entry.sessionId))
  const route = [entry.source, entry.mode, entry.node].filter(Boolean).join('/')

  if (ev === 'llm.call') {
    const model = chalk.magenta(entry.model ?? entry.request?.model ?? '?')
    const iter = entry.iteration != null ? chalk.gray(`#${entry.iteration}`) : ''
    const finish = entry.response?.finishReason ? chalk.gray(`→${entry.response.finishReason}`) : ''
    const toks = tokenLabel(entry.response?.usage)
    const err = entry.error ? chalk.red(` ERROR: ${previewContent(entry.error, 200)}`) : ''
    console.log(`${ts} ${chalk.blue('llm.call')} ${sess} ${model} ${iter} ${chalk.gray(route)} ${finish} ${toks}${err}`)
    if (full) {
      const msgs = entry.request?.messages ?? []
      for (const m of msgs) {
        const role = (m.role ?? '?').padEnd(9)
        const tc = m.toolCalls ? chalk.yellow(' [toolCalls]') : ''
        console.log(`  ${chalk.gray(role)} ${previewContent(m.content, 200)}${tc}`)
      }
      if (entry.response?.message) {
        const rm = entry.response.message
        const tc = rm.toolCalls ? chalk.yellow(' [toolCalls]') : ''
        console.log(`  ${chalk.green('← assistant')} ${previewContent(rm.content, 300)}${tc}`)
      }
      if (entry.response?.thinking) {
        console.log(`  ${chalk.gray('← thinking')} ${previewContent(entry.response.thinking, 200)}`)
      }
    }
    return
  }

  if (ev === 'agent.run') {
    const status = entry.status === 'error' || entry.status === 'failed'
      ? chalk.red(entry.status)
      : entry.status === 'completed' || entry.status === 'done'
        ? chalk.green(entry.status ?? '')
        : chalk.yellow(entry.status ?? '')
    const model = entry.model ? chalk.magenta(entry.model) : ''
    const toks = tokenLabel(entry.usage)
    const err = entry.error ? chalk.red(` ERROR: ${previewContent(entry.error, 200)}`) : ''
    const out = entry.output ? chalk.gray(` ${previewContent(entry.output, 120)}`) : ''
    console.log(`${ts} ${chalk.blue('agent.run')} ${sess} ${status} ${chalk.gray(route)} ${model} ${toks}${err}${out}`)
    return
  }

  // Unknown event type — dump compactly.
  const tool = typeof entry.data?.tool === 'string' ? entry.data.tool : ''
  const state = entry.data?.state && typeof entry.data.state === 'object'
    ? entry.data.state as Record<string, unknown>
    : undefined
  const next = typeof entry.data?.nextEdge === 'string' ? ` → ${entry.data.nextEdge}` : ''
  const duration = entry.durationMs != null ? ` ${entry.durationMs}ms` : ''
  const execution = entry.executionId ? ` exec=${shortId(entry.executionId)}` : ''
  const detail = tool
    ? tool
    : state
      ? `tools=${state.toolHistory ?? 0} output=${state.outputChars ?? 0}`
      : previewContent(entry.data, full ? 2_000 : 300)
  console.log(`${ts} ${chalk.blue(ev)} ${sess} ${chalk.gray(route)} ${entry.status ?? ''}${next}${duration}${execution} ${chalk.gray(detail)}`.trimEnd())
  if (full) {
    console.log(chalk.gray(JSON.stringify(entry.data ?? {}, null, 2)))
  }
}

export async function traceCommand(options: TraceCommandOptions): Promise<void> {
  const path = options.file ?? defaultTracePath()
  let content: string
  try {
    content = await readTraceFiles(path)
  } catch {
    console.error(chalk.red(`Trace file not found: ${path}`))
    console.error(chalk.gray('Start the daemon with SEPILOT_DEBUG=1, run an agent turn, then retry.'))
    return
  }

  const all = parseEntries(content)
  const filtered = applyFilters(all, options)

  if (options.stats) {
    if (getOutputFormat() === 'json') {
      output({ total: filtered.length, entries: filtered })
      return
    }
    printStats(filtered)
    return
  }

  const last = Math.max(1, parseInt(options.last ?? '20', 10) || 20)
  const slice = filtered.slice(-last)

  if (getOutputFormat() === 'json') {
    output(slice)
    return
  }

  if (slice.length === 0) {
    console.log(chalk.gray('No trace entries match the filters.'))
    return
  }
  console.log(chalk.gray(`Showing ${slice.length} of ${filtered.length} matching entries (${all.length} total) — ${path}\n`))
  for (const entry of slice) {
    printEntry(entry, Boolean(options.full))
  }
}
