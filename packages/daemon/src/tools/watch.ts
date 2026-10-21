import { randomUUID } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from 'node:fs'
import { dirname, isAbsolute, join, relative, resolve } from 'node:path'
import { SessionInbox, type SessionInboxPublishInput } from '../server/runtime/session-inbox.js'
import type { ManagedProcessRegistry } from './process.js'
import type { ToolDefinitionRuntime, ToolExecutionContext, ToolResult } from './registry.js'
import { createLogger } from '../logger.js'

const log = createLogger('watch')

const POLL_MS = 500
const FILE_POLL_MS = 1_000
const DEFAULT_TIMEOUT_S = 3_600
const MAX_TIMEOUT_S = 24 * 3_600
const MAX_WATCHES_PER_SESSION = 20
const MAX_WATCHES_TOTAL = 200
const MAX_FIRES_PER_WATCH = 20
const MIN_REPEAT_SPACING_MS = 10_000
const MAX_MATCH_LINES = 20
const MIN_WAKEUP_S = 10
const MAX_WAKEUP_S = 7 * 24 * 3_600
const MAX_WAKEUPS_PER_SESSION = 20
const MAX_TEXT = 1_000

export type WatchKind = 'process_output' | 'file'

export interface WatchSummary {
  id: string
  kind: WatchKind | 'wakeup'
  sessionId: string
  label: string
  createdAt: string
  expiresAt: string
  fires: number
  target: string
}

interface ActiveWatch extends WatchSummary {
  timer: ReturnType<typeof setInterval>
  stop(): void
}

interface StoredWakeup {
  id: string
  sessionId: string
  reason: string
  prompt: string
  createdAt: string
  fireAt: string
}

function clampSeconds(value: unknown, fallback: number, min: number, max: number): number {
  const number = typeof value === 'number' && Number.isFinite(value) ? value : fallback
  return Math.max(min, Math.min(max, Math.trunc(number)))
}

function text(value: unknown, max = MAX_TEXT): string {
  return typeof value === 'string' ? value.trim().slice(0, max) : ''
}

function error(output: string, code: string, startedAt: number): ToolResult {
  return { status: 'error', code, output, durationMs: Date.now() - startedAt }
}

export class WatchService {
  private readonly watches = new Map<string, ActiveWatch>()
  private readonly wakeupTimers = new Map<string, ReturnType<typeof setTimeout>>()
  private wakeups: StoredWakeup[] = []

  constructor(private readonly deps: {
    processes?: ManagedProcessRegistry
    inbox?: () => SessionInbox
    dataDir?: string
    now?: () => number
  } = {}) {
    this.wakeups = this.loadWakeups()
    for (const wakeup of this.wakeups) this.armWakeup(wakeup)
  }

  private now(): number { return this.deps.now?.() ?? Date.now() }

  private publish(input: SessionInboxPublishInput): void {
    (this.deps.inbox?.() ?? new SessionInbox()).tryPublish(input)
  }

  list(sessionId: string): WatchSummary[] {
    const watches = [...this.watches.values()]
      .filter(watch => watch.sessionId === sessionId)
      .map(({ timer: _timer, stop: _stop, ...summary }) => summary)
    const wakeups = this.wakeups
      .filter(wakeup => wakeup.sessionId === sessionId)
      .map((wakeup): WatchSummary => ({
        id: wakeup.id, kind: 'wakeup', sessionId, label: wakeup.reason,
        createdAt: wakeup.createdAt, expiresAt: wakeup.fireAt, fires: 0, target: 'this conversation',
      }))
    return [...watches, ...wakeups]
  }

  stop(sessionId: string, id: string): boolean {
    const watch = this.watches.get(id)
    if (watch && watch.sessionId === sessionId) {
      watch.stop()
      return true
    }
    const wakeup = this.wakeups.find(entry => entry.id === id && entry.sessionId === sessionId)
    if (!wakeup) return false
    this.removeWakeup(id)
    return true
  }

  dispose(): void {
    for (const watch of this.watches.values()) clearInterval(watch.timer)
    this.watches.clear()
    for (const timer of this.wakeupTimers.values()) clearTimeout(timer)
    this.wakeupTimers.clear()
  }

  admit(sessionId: string): string | null {
    if (this.watches.size >= MAX_WATCHES_TOTAL) return 'The daemon is already running the maximum number of watches. Stop an existing watch first.'
    const own = [...this.watches.values()].filter(watch => watch.sessionId === sessionId).length
    if (own >= MAX_WATCHES_PER_SESSION) return `This conversation already has ${MAX_WATCHES_PER_SESSION} active watches. Stop one with monitor.watches first.`
    return null
  }

  private register(watch: Omit<ActiveWatch, 'stop'>, cleanup?: () => void): WatchSummary {
    const active: ActiveWatch = {
      ...watch,
      stop: () => {
        clearInterval(watch.timer)
        cleanup?.()
        this.watches.delete(watch.id)
      },
    }
    watch.timer.unref?.()
    this.watches.set(watch.id, active)
    const { timer: _timer, ...summary } = watch
    return summary
  }

  watchProcessOutput(input: {
    sessionId: string
    processId: string
    pattern?: RegExp
    label: string
    timeoutSeconds: number
    repeat: boolean
  }): WatchSummary {
    const processes = this.deps.processes
    if (!processes) throw new Error('Managed processes are not available on this daemon.')
    if (processes.ownerOf(input.processId) !== input.sessionId) {
      throw new Error(`Process ${input.processId} was not started by this conversation.`)
    }
    const id = `watch-${randomUUID().slice(0, 8)}`
    const createdAt = this.now()
    const expiresAt = createdAt + input.timeoutSeconds * 1_000
    let offsets = { stdout: 0, stderr: 0 }
    const initial = processes.outputSince(input.processId, offsets)
    if (initial) offsets = initial.next
    let fires = 0
    let lastFire = 0
    let partial = ''
    let buffered: string[] = []
    const fire = (title: string, lines: string[], final: boolean) => {
      fires += 1
      lastFire = this.now()
      const watch = this.watches.get(id)
      if (watch) watch.fires = fires
      this.publish({
        sessionId: input.sessionId,
        source: 'monitor',
        sourceId: id,
        eventKey: `${id}:${fires}`,
        title: `${input.label}: ${title}`.slice(0, 200),
        body: [
          `Watch ${id} on process ${input.processId}${input.pattern ? ` for /${input.pattern.source}/${input.pattern.flags}` : ''}: ${title}.`,
          lines.length ? `Matching output:\n${lines.slice(-MAX_MATCH_LINES).join('\n')}` : '',
          final ? 'The watch has ended.' : `The watch keeps running (${fires}/${MAX_FIRES_PER_WATCH} notifications used). Stop it with monitor.watches when no longer needed.`,
        ].filter(Boolean).join('\n\n'),
        wake: true,
      })
    }
    const timer = setInterval(() => {
      const watch = this.watches.get(id)
      if (!watch) return
      const chunk = processes.outputSince(input.processId, offsets)
      if (!chunk) {
        fire('the process is no longer tracked', [], true)
        watch.stop()
        return
      }
      offsets = chunk.next
      const combined = partial + chunk.stdout + chunk.stderr
      const lines = combined.split(/\r?\n/)
      partial = chunk.exited ? '' : lines.pop() ?? ''
      if (input.pattern) {
        for (const line of lines) {
          input.pattern.lastIndex = 0
          if (input.pattern.test(line)) buffered.push(line.slice(0, 500))
        }
        if (buffered.length && (input.repeat ? this.now() - lastFire >= MIN_REPEAT_SPACING_MS || fires === 0 : true)) {
          const matched = buffered
          buffered = []
          const final = !input.repeat || fires + 1 >= MAX_FIRES_PER_WATCH
          fire(`output matched (${matched.length} line${matched.length === 1 ? '' : 's'})`, matched, final)
          if (final) { watch.stop(); return }
        }
      }
      if (chunk.exited) {
        if (buffered.length) fire(`output matched (${buffered.length} lines)`, buffered, false)
        fire(input.pattern ? 'the process exited before the watch was satisfied' : 'the process exited', lines.filter(Boolean).slice(-MAX_MATCH_LINES), true)
        watch.stop()
        return
      }
      if (this.now() >= expiresAt) {
        fire(`timed out after ${input.timeoutSeconds}s without ${input.pattern ? 'a further match' : 'exiting'}`, [], true)
        watch.stop()
      }
    }, POLL_MS)
    return this.register({
      id, kind: 'process_output', sessionId: input.sessionId, label: input.label,
      createdAt: new Date(createdAt).toISOString(), expiresAt: new Date(expiresAt).toISOString(),
      fires: 0, target: `process ${input.processId}`, timer,
    })
  }

  watchFile(input: {
    sessionId: string
    path: string
    pattern?: RegExp
    label: string
    timeoutSeconds: number
  }): WatchSummary {
    const id = `watch-${randomUUID().slice(0, 8)}`
    const createdAt = this.now()
    const expiresAt = createdAt + input.timeoutSeconds * 1_000
    const signature = () => {
      try {
        const stat = statSync(input.path)
        return `${stat.size}:${stat.mtimeMs}`
      } catch { return 'missing' }
    }
    let baseline = signature()
    const finish = (title: string, detail: string) => {
      this.publish({
        sessionId: input.sessionId,
        source: 'monitor',
        sourceId: id,
        eventKey: `${id}:1`,
        title: `${input.label}: ${title}`.slice(0, 200),
        body: `Watch ${id} on ${input.path}: ${title}.${detail ? `\n\n${detail}` : ''}\n\nThe watch has ended.`,
        wake: true,
      })
      this.watches.get(id)?.stop()
    }
    const timer = setInterval(() => {
      if (!this.watches.has(id)) return
      const current = signature()
      if (current !== baseline) {
        baseline = current
        if (!input.pattern) {
          finish(current === 'missing' ? 'the file was removed' : 'the file changed', '')
          return
        }
        if (current !== 'missing') {
          try {
            const lines = readFileSync(input.path, 'utf8').split(/\r?\n/)
            const matched = lines.filter(line => { input.pattern!.lastIndex = 0; return input.pattern!.test(line) })
            if (matched.length) {
              finish(`content matched (${matched.length} line${matched.length === 1 ? '' : 's'})`, `Matching lines:\n${matched.slice(-MAX_MATCH_LINES).map(line => line.slice(0, 500)).join('\n')}`)
              return
            }
          } catch (readError) {
            log.warn('file watch read failed', { id, error: String(readError) })
          }
        }
      }
      if (this.now() >= expiresAt) finish(`timed out after ${input.timeoutSeconds}s without the expected change`, '')
    }, FILE_POLL_MS)
    return this.register({
      id, kind: 'file', sessionId: input.sessionId, label: input.label,
      createdAt: new Date(createdAt).toISOString(), expiresAt: new Date(expiresAt).toISOString(),
      fires: 0, target: input.path, timer,
    })
  }

  scheduleWakeup(input: { sessionId: string; delaySeconds: number; reason: string; prompt: string }): WatchSummary {
    const own = this.wakeups.filter(wakeup => wakeup.sessionId === input.sessionId).length
    if (own >= MAX_WAKEUPS_PER_SESSION) throw new Error(`This conversation already has ${MAX_WAKEUPS_PER_SESSION} pending wakeups.`)
    const createdAt = this.now()
    const wakeup: StoredWakeup = {
      id: `wakeup-${randomUUID().slice(0, 8)}`,
      sessionId: input.sessionId,
      reason: input.reason,
      prompt: input.prompt,
      createdAt: new Date(createdAt).toISOString(),
      fireAt: new Date(createdAt + input.delaySeconds * 1_000).toISOString(),
    }
    this.wakeups.push(wakeup)
    this.saveWakeups()
    this.armWakeup(wakeup)
    return {
      id: wakeup.id, kind: 'wakeup', sessionId: wakeup.sessionId, label: wakeup.reason,
      createdAt: wakeup.createdAt, expiresAt: wakeup.fireAt, fires: 0, target: 'this conversation',
    }
  }

  private armWakeup(wakeup: StoredWakeup): void {
    const delay = Math.max(0, Date.parse(wakeup.fireAt) - this.now())
    const timer = setTimeout(() => {
      this.wakeupTimers.delete(wakeup.id)
      this.publish({
        sessionId: wakeup.sessionId,
        source: 'wakeup',
        sourceId: wakeup.id,
        eventKey: wakeup.id,
        title: `Scheduled wakeup: ${wakeup.reason}`.slice(0, 200),
        body: [
          `You scheduled this wakeup at ${wakeup.createdAt} for ${wakeup.fireAt}.`,
          wakeup.prompt ? `What you asked yourself to do now:\n${wakeup.prompt}` : '',
        ].filter(Boolean).join('\n\n'),
        wake: true,
      })
      this.removeWakeup(wakeup.id)
    }, delay)
    timer.unref?.()
    this.wakeupTimers.set(wakeup.id, timer)
  }

  private removeWakeup(id: string): void {
    const timer = this.wakeupTimers.get(id)
    if (timer) clearTimeout(timer)
    this.wakeupTimers.delete(id)
    this.wakeups = this.wakeups.filter(wakeup => wakeup.id !== id)
    this.saveWakeups()
  }

  private wakeupFile(): string | null {
    return this.deps.dataDir ? join(this.deps.dataDir, 'session-wakeups.json') : null
  }

  private loadWakeups(): StoredWakeup[] {
    const file = this.wakeupFile()
    if (!file || !existsSync(file)) return []
    try {
      const parsed = JSON.parse(readFileSync(file, 'utf8')) as unknown
      if (!Array.isArray(parsed)) return []
      return parsed.filter((entry): entry is StoredWakeup => Boolean(entry)
        && typeof entry.id === 'string' && typeof entry.sessionId === 'string'
        && typeof entry.fireAt === 'string' && Number.isFinite(Date.parse(entry.fireAt)))
        .map(entry => ({ ...entry, reason: text(entry.reason, 200), prompt: text(entry.prompt) }))
    } catch (loadError) {
      log.warn('stored wakeups unreadable; starting empty', { error: String(loadError) })
      return []
    }
  }

  private saveWakeups(): void {
    const file = this.wakeupFile()
    if (!file) return
    try {
      mkdirSync(dirname(file), { recursive: true })
      const temp = `${file}.${process.pid}.tmp`
      writeFileSync(temp, JSON.stringify(this.wakeups, null, 2), { mode: 0o600 })
      renameSync(temp, file)
    } catch (saveError) {
      log.warn('failed to persist wakeups; they will not survive a restart', { error: String(saveError) })
    }
  }
}

function compilePattern(raw: unknown): RegExp | undefined | string {
  const source = text(raw, 500)
  if (!source) return undefined
  try { return new RegExp(source, 'i') }
  catch (patternError) { return `pattern is not a valid regular expression: ${patternError instanceof Error ? patternError.message : String(patternError)}` }
}

function resolveWatchedPath(raw: string, context: ToolExecutionContext): string | { error: string } {
  const base = context.cwd ?? process.cwd()
  const path = isAbsolute(raw) ? resolve(raw) : resolve(base, raw)
  if (context.workspaceRoot) {
    const root = resolve(context.workspaceRoot)
    const rel = relative(root, path)
    if (rel.startsWith('..') || isAbsolute(rel)) return { error: `path must stay inside the workspace ${root}` }
  }
  return path
}

export function createWatchTools(service: WatchService): ToolDefinitionRuntime[] {
  const watchTool: ToolDefinitionRuntime = {
    name: 'monitor.watch',
    description: [
      'Wait for a condition without spending turns: the daemon watches in the background and this conversation is notified (and woken if idle) when the condition holds, the watch times out, or the target goes away.',
      "kind 'process_output': watch a background process started in this conversation (process.start or terminal.run with runInBackground) for output lines matching `pattern` (case-insensitive regex); omit pattern to wait for its exit. Set repeat=true to be notified on every later match (spaced, at most 20).",
      "kind 'file': watch a file for any change, or for content matching `pattern`.",
      'For any other condition (CI status, an HTTP endpoint, a queue), start a policy-checked background command that loops until the condition holds and prints a line, e.g. terminal.run runInBackground with bash -c \'until <check>; do sleep 30; done; echo READY\', then watch its output for READY.',
      'Do not poll with repeated tool calls or sleep; end your turn after starting the watch.',
    ].join(' '),
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        kind: { type: 'string', enum: ['process_output', 'file'] },
        processId: { type: 'string', description: "Managed process id (kind 'process_output')." },
        path: { type: 'string', description: "File path (kind 'file'), relative to the session cwd." },
        pattern: { type: 'string', description: 'Case-insensitive regular expression a line must match. Omit to watch for exit (process) or any change (file).' },
        label: { type: 'string', description: 'Short description shown in the notification, e.g. "dev server ready".' },
        timeoutSeconds: { type: 'number', description: `Give up and notify after this many seconds. Default ${DEFAULT_TIMEOUT_S}, max ${MAX_TIMEOUT_S}.` },
        repeat: { type: 'boolean', description: "kind 'process_output' with a pattern: keep watching after the first match. Default false." },
      },
      required: ['kind', 'label'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!context?.sessionId) return error('monitor.watch needs a conversation to notify.', 'WATCH_SESSION_REQUIRED_PERMANENT', startedAt)
      const label = text(input.label, 120) || 'Watch'
      const pattern = compilePattern(input.pattern)
      if (typeof pattern === 'string') return error(pattern, 'INVALID_INPUT_PERMANENT', startedAt)
      const timeoutSeconds = clampSeconds(input.timeoutSeconds, DEFAULT_TIMEOUT_S, 5, MAX_TIMEOUT_S)
      const refusal = service.admit(context.sessionId)
      if (refusal) return error(refusal, 'WATCH_LIMIT_USER', startedAt)
      try {
        let summary: WatchSummary
        if (input.kind === 'process_output') {
          const processId = text(input.processId, 200)
          if (!processId) return error("kind 'process_output' requires processId.", 'INVALID_INPUT_PERMANENT', startedAt)
          summary = service.watchProcessOutput({ sessionId: context.sessionId, processId, pattern, label, timeoutSeconds, repeat: input.repeat === true && Boolean(pattern) })
        } else if (input.kind === 'file') {
          const raw = text(input.path, 1_000)
          if (!raw) return error("kind 'file' requires path.", 'INVALID_INPUT_PERMANENT', startedAt)
          const path = resolveWatchedPath(raw, context)
          if (typeof path !== 'string') return error(path.error, 'WORKSPACE_BOUNDARY_PERMANENT', startedAt)
          summary = service.watchFile({ sessionId: context.sessionId, path, pattern, label, timeoutSeconds })
        } else {
          return error("kind must be 'process_output' or 'file'.", 'INVALID_INPUT_PERMANENT', startedAt)
        }
        return {
          status: 'success',
          output: `${JSON.stringify(summary, null, 2)}\n\nWatching in the background. End your turn; this conversation is notified when the watch fires or ends.`,
          durationMs: Date.now() - startedAt,
        }
      } catch (watchError) {
        return error(watchError instanceof Error ? watchError.message : String(watchError), 'WATCH_TARGET_PERMANENT', startedAt)
      }
    },
  }

  const watchesTool: ToolDefinitionRuntime = {
    name: 'monitor.watches',
    description: 'List this conversation\'s active watches and scheduled wakeups, or stop one by id.',
    resumeSafety: 'replay-safe',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        action: { type: 'string', enum: ['list', 'stop'], description: 'Default list.' },
        id: { type: 'string', description: 'Watch or wakeup id to stop.' },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!context?.sessionId) return error('monitor.watches needs a conversation.', 'WATCH_SESSION_REQUIRED_PERMANENT', startedAt)
      if (input.action === 'stop') {
        const id = text(input.id, 100)
        if (!id) return error('action stop requires id.', 'INVALID_INPUT_PERMANENT', startedAt)
        return service.stop(context.sessionId, id)
          ? { status: 'success', output: `Stopped ${id}.`, durationMs: Date.now() - startedAt }
          : error(`No active watch or wakeup ${id} in this conversation.`, 'WATCH_NOT_FOUND_PERMANENT', startedAt)
      }
      const items = service.list(context.sessionId)
      return {
        status: 'success',
        output: items.length ? JSON.stringify(items, null, 2) : 'No active watches or wakeups in this conversation.',
        durationMs: Date.now() - startedAt,
      }
    },
  }

  const wakeupTool: ToolDefinitionRuntime = {
    name: 'session.schedule_wakeup',
    description: `Resume this same conversation after a delay, with its full context, instead of sleeping or polling. When it fires, the conversation receives your prompt and continues (it is woken if idle). Use for "check back in N minutes" or paced loops. ${MIN_WAKEUP_S}s to ${MAX_WAKEUP_S / 86_400} days; survives a daemon restart. For recurring or calendar-time jobs that should run on their own, use schedule_create instead.`,
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        delaySeconds: { type: 'number', description: `Seconds from now (${MIN_WAKEUP_S}-${MAX_WAKEUP_S}).` },
        reason: { type: 'string', description: 'One short line on what you are waiting for.' },
        prompt: { type: 'string', description: 'What to do when woken, written for your future self.' },
      },
      required: ['delaySeconds', 'reason', 'prompt'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!context?.sessionId) return error('session.schedule_wakeup needs a conversation to resume.', 'WATCH_SESSION_REQUIRED_PERMANENT', startedAt)
      const reason = text(input.reason, 200)
      const prompt = text(input.prompt)
      if (!reason || !prompt) return error('reason and prompt are required.', 'INVALID_INPUT_PERMANENT', startedAt)
      try {
        const summary = service.scheduleWakeup({
          sessionId: context.sessionId,
          delaySeconds: clampSeconds(input.delaySeconds, 300, MIN_WAKEUP_S, MAX_WAKEUP_S),
          reason,
          prompt,
        })
        return {
          status: 'success',
          output: `${JSON.stringify(summary, null, 2)}\n\nWakeup scheduled. End your turn now; this conversation resumes at ${summary.expiresAt}.`,
          durationMs: Date.now() - startedAt,
        }
      } catch (wakeupError) {
        return error(wakeupError instanceof Error ? wakeupError.message : String(wakeupError), 'WATCH_LIMIT_USER', startedAt)
      }
    },
  }

  return [watchTool, watchesTool, wakeupTool]
}
