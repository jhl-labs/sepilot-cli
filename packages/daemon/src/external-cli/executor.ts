import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { StringDecoder } from 'node:string_decoder'
import { isAbsolute } from 'node:path'
import { AutonomyLevel } from '@sepilotd/core'

export const EXTERNAL_CLI_BACKENDS = ['claude', 'codex', 'opencode'] as const
export type ExternalCliBackend = typeof EXTERNAL_CLI_BACKENDS[number]

export function isExternalCliBackend(value: unknown): value is ExternalCliBackend {
  return typeof value === 'string' && (EXTERNAL_CLI_BACKENDS as readonly string[]).includes(value)
}

export function externalCliFromEngine(engine: string | undefined): ExternalCliBackend | null {
  if (!engine?.startsWith('external:')) return null
  const cli = engine.slice('external:'.length)
  return isExternalCliBackend(cli) ? cli : null
}

export type ExternalCliPermission = 'read' | 'edit' | 'full'

export const EXTERNAL_CLI_PERMISSIONS: readonly ExternalCliPermission[] = ['read', 'edit', 'full']

export function isExternalCliPermission(value: unknown): value is ExternalCliPermission {
  return typeof value === 'string' && (EXTERNAL_CLI_PERMISSIONS as readonly string[]).includes(value)
}

export function externalCliPermissionCeiling(env: NodeJS.ProcessEnv = process.env): ExternalCliPermission | undefined {
  const raw = env.SEPILOTD_EXTERNAL_CLI_MAX_PERMISSION?.trim()
  if (!raw) return undefined
  return isExternalCliPermission(raw) ? raw : 'read'
}

export function capPermission(derived: ExternalCliPermission, bound: ExternalCliPermission | undefined): ExternalCliPermission {
  if (!bound) return derived
  return EXTERNAL_CLI_PERMISSIONS.indexOf(bound) < EXTERNAL_CLI_PERMISSIONS.indexOf(derived) ? bound : derived
}

export function permissionForAutonomy(autonomy: AutonomyLevel | undefined): ExternalCliPermission {
  switch (autonomy) {
    case AutonomyLevel.Autonomous:
      return 'full'
    case AutonomyLevel.AcceptEdits:
    case AutonomyLevel.WorkspaceWrite:
      return 'edit'
    case AutonomyLevel.ReadOnly:
    case AutonomyLevel.Supervised:
    default:
      return 'read'
  }
}

export interface ExternalCliRunInput {
  backend: ExternalCliBackend
  message: string
  cwd: string
  permission: ExternalCliPermission
  model?: string
  threadId?: string
  timeoutMs?: number
  idleTimeoutMs?: number
  env?: NodeJS.ProcessEnv
}

export type ExternalCliFailureKind = 'stalled' | 'timeout' | 'incomplete' | 'cancelled' | 'failed'

export class ExternalCliRunError extends Error {
  constructor(message: string, readonly kind: ExternalCliFailureKind, readonly threadId?: string) {
    super(message)
    this.name = 'ExternalCliRunError'
  }
}

export function isResumableFailure(kind: ExternalCliFailureKind | undefined): boolean {
  return kind === 'stalled' || kind === 'timeout' || kind === 'incomplete'
}

export type ExternalCliEvent =
  | { type: 'text'; text: string }
  | { type: 'progress'; text: string }
  | { type: 'usage'; inputTokens: number; outputTokens: number }

export interface ExternalCliRunResult {
  output: string
  threadId?: string
  usage: { inputTokens: number; outputTokens: number }
  durationMs: number
}

export interface ExternalCliRunOptions {
  signal?: AbortSignal
  onEvent?: (event: ExternalCliEvent) => void
  spawnFn?: typeof spawn
  log?: (event: string, fields: Record<string, unknown>) => void
}

const DEFAULT_TIMEOUT_MS = 20 * 60_000
const DEFAULT_IDLE_TIMEOUT_MS = 10 * 60_000

export function externalCliIdleTimeout(input: Pick<ExternalCliRunInput, 'idleTimeoutMs'>, env: NodeJS.ProcessEnv = process.env): number {
  if (input.idleTimeoutMs !== undefined) return Math.max(0, input.idleTimeoutMs)
  const raw = Number(env.SEPILOTD_EXTERNAL_CLI_IDLE_TIMEOUT_MS)
  return env.SEPILOTD_EXTERNAL_CLI_IDLE_TIMEOUT_MS?.trim() && Number.isFinite(raw) && raw >= 0 ? raw : DEFAULT_IDLE_TIMEOUT_MS
}
const THREAD_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
const MODEL_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$/
const MAX_BUFFER_BYTES = 24 * 1024 * 1024

type Row = Record<string, unknown>
const row = (value: unknown): Row => (value && typeof value === 'object' ? (value as Row) : {})
const str = (value: unknown): string => (typeof value === 'string' ? value : '')

export function resolveExternalCliCommand(backend: ExternalCliBackend, env: NodeJS.ProcessEnv = process.env): string {
  const override = env[`SEPILOTD_EXTERNAL_CLI_${backend.toUpperCase()}_COMMAND`]?.trim()
  return override || backend
}

export function buildExternalCliCommand(input: ExternalCliRunInput, env: NodeJS.ProcessEnv = process.env): string[] {
  const command = resolveExternalCliCommand(input.backend, env)
  const model = input.model ? ['--model', input.model] : []
  switch (input.backend) {
    case 'claude': {
      const permissionMode = input.permission === 'full' ? 'bypassPermissions' : input.permission === 'edit' ? 'acceptEdits' : 'dontAsk'
      return [
        command, '--print', '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose',
        '--permission-mode', permissionMode,
        ...(input.permission === 'full' ? ['--dangerously-skip-permissions'] : []),
        ...model,
        ...(input.threadId ? ['--resume', input.threadId] : []),
      ]
    }
    case 'codex':
      return [command, 'app-server']
    case 'opencode':
      return [
        command, 'run', '--format', 'json',
        ...(input.permission === 'read' ? ['--agent', 'plan'] : []),
        ...model,
        ...(input.threadId ? ['--session', input.threadId] : []),
      ]
  }
}

function codexSandbox(permission: ExternalCliPermission): { sandbox: string; sandboxPolicy: Row } {
  if (permission === 'full') return { sandbox: 'danger-full-access', sandboxPolicy: { type: 'dangerFullAccess' } }
  if (permission === 'edit') return { sandbox: 'workspace-write', sandboxPolicy: { type: 'workspaceWrite' } }
  return { sandbox: 'read-only', sandboxPolicy: { type: 'readOnly' } }
}

export async function runExternalCli(input: ExternalCliRunInput, options: ExternalCliRunOptions = {}): Promise<ExternalCliRunResult> {
  if (!isExternalCliBackend(input.backend)) throw new Error(`EXTERNAL_CLI_UNKNOWN_BACKEND: ${String(input.backend)}`)
  if (!input.message.trim()) throw new Error('EXTERNAL_CLI_INVALID_REQUEST: message is required')
  if (!isAbsolute(input.cwd)) throw new Error('EXTERNAL_CLI_INVALID_REQUEST: cwd must be absolute')
  if (input.threadId !== undefined && !THREAD_ID_PATTERN.test(input.threadId)) throw new Error('EXTERNAL_CLI_INVALID_REQUEST: threadId must be an identifier')
  if (input.model !== undefined && !MODEL_PATTERN.test(input.model)) throw new Error('EXTERNAL_CLI_INVALID_REQUEST: model must be an identifier')
  if (options.signal?.aborted) throw new ExternalCliRunError('external CLI run cancelled', 'cancelled', input.threadId)

  const env = input.env ?? process.env
  const [file, ...args] = buildExternalCliCommand(input, env)
  if (input.permission === 'full') {
    options.log?.('external_cli.full_permission', { backend: input.backend, cwd: input.cwd })
  }
  const spawnFn = options.spawnFn ?? spawn
  const startedAt = Date.now()
  const timeoutMs = Math.max(1, input.timeoutMs ?? DEFAULT_TIMEOUT_MS)
  const idleTimeoutMs = externalCliIdleTimeout(input, env)
  const texts: string[] = []
  const usage = { inputTokens: 0, outputTokens: 0 }
  let threadId = input.threadId
  const emit = (event: ExternalCliEvent) => {
    if (event.type === 'text') texts.push(event.text)
    if (event.type === 'usage') { usage.inputTokens += event.inputTokens; usage.outputTokens += event.outputTokens }
    options.onEvent?.(event)
  }
  const text = (value: unknown) => { if (typeof value === 'string' && value) emit({ type: 'text', text: value }) }

  await new Promise<void>((resolve, reject) => {
    let child: ChildProcessWithoutNullStreams
    try {
      child = spawnFn(file, args, { cwd: input.cwd, env, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true }) as ChildProcessWithoutNullStreams
    } catch (error) {
      reject(describeSpawnError(input.backend, file, error))
      return
    }
    let finished = false, sawCompletion = false, buffer = '', stderr = ''
    let lastStepReason = ''
    const decoder = new StringDecoder('utf8')
    const seen = new Set<string>()
    const fail = (message: string, kind: ExternalCliFailureKind) => finish(new ExternalCliRunError(message, kind, threadId))
    const finish = (error?: Error) => {
      if (finished) return
      finished = true
      clearTimeout(timer)
      if (idle) clearTimeout(idle)
      options.signal?.removeEventListener('abort', abort)
      try { child.stdin.end() } catch {}
      child.kill()
      if (child.exitCode === null && child.signalCode === null) {
        const force = setTimeout(() => child.kill('SIGKILL'), 1000)
        force.unref()
        child.once('exit', () => clearTimeout(force))
      }
      if (error) reject(error instanceof ExternalCliRunError ? error : new ExternalCliRunError(error.message, 'failed', threadId))
      else resolve()
    }
    const abort = () => fail('external CLI run cancelled', 'cancelled')
    const timer = setTimeout(() => fail(`external CLI run timed out after ${timeoutMs}ms`, 'timeout'), timeoutMs)
    let idle: NodeJS.Timeout | undefined
    const heard = () => {
      if (!idleTimeoutMs || finished) return
      if (idle) clearTimeout(idle)
      idle = setTimeout(() => {
        options.log?.('external_cli.stalled', { backend: input.backend, idleMs: idleTimeoutMs, threadId })
        fail(`${input.backend} produced no output for ${idleTimeoutMs}ms (stalled: network or model server); the session can be resumed`, 'stalled')
      }, idleTimeoutMs)
    }
    heard()
    options.signal?.addEventListener('abort', abort, { once: true })
    const send = (value: unknown) => child.stdin.write(`${JSON.stringify(value)}\n`)

    const parse = (line: string) => {
      if (!line.trim() || finished) return
      if (!line.trimStart().startsWith('{')) { emit({ type: 'progress', text: line.slice(0, 1000) }); return }
      const event = row(JSON.parse(line))
      if (input.backend === 'codex') {
        if (event.error) throw new Error(str(row(event.error).message) || 'Codex request failed')
        const sandbox = codexSandbox(input.permission)
        if (event.id === 1) {
          send({ method: 'initialized' })
          send({ id: 2, method: threadId ? 'thread/resume' : 'thread/start', params: {
            ...(threadId ? { threadId } : {}), cwd: input.cwd, ...(input.model ? { model: input.model } : {}),
            approvalPolicy: 'never', sandbox: sandbox.sandbox, serviceName: 'sepilotd-external-cli',
          } })
        } else if (event.id === 2) {
          const id = str(row(row(event.result).thread).id)
          if (!id) throw new Error('Codex returned no thread id')
          threadId = id
          send({ id: 3, method: 'turn/start', params: {
            threadId: id, ...(input.model ? { model: input.model } : {}),
            approvalPolicy: 'never', sandboxPolicy: sandbox.sandboxPolicy,
            input: [{ type: 'text', text: input.message, text_elements: [] }],
          } })
        }
        const params = row(event.params)
        if (params.threadId && params.threadId !== threadId) return
        if (event.method === 'thread/tokenUsage/updated') {
          const last = row(row(params.tokenUsage).last)
          emit({ type: 'usage', inputTokens: Number(last.inputTokens) || 0, outputTokens: Number(last.outputTokens) || 0 })
        }
        if (event.method === 'item/completed') {
          const item = row(params.item)
          if (typeof item.id === 'string') { if (seen.has(item.id)) return; seen.add(item.id) }
          if (item.type === 'agentMessage') text(item.text)
        }
        if (event.method === 'item/started') {
          const item = row(params.item)
          const target = str(item.command) || str(row(item).path)
          emit({ type: 'progress', text: `${str(item.type)}${target ? ` ${target}` : ''}`.slice(0, 200) })
        }
        if (event.method === 'turn/completed') {
          const turn = row(params.turn)
          if (turn.status !== 'completed') throw new Error(str(row(turn.error).message) || `Codex turn ${str(turn.status)}`)
          sawCompletion = true
          finish()
        }
        if (event.id !== undefined && typeof event.method === 'string') {
          send({ id: event.id, error: { code: -32601, message: 'sepilotd external CLI runs are non-interactive' } })
        }
      } else if (input.backend === 'claude') {
        if (typeof event.session_id === 'string') threadId = event.session_id
        if (event.type === 'assistant') {
          const content = row(event.message).content
          if (Array.isArray(content)) {
            for (const part of content) {
              const p = row(part)
              if (p.type === 'text') text(p.text)
              if (p.type === 'tool_use') {
                const args = row(p.input)
                const target = str(args.file_path) || str(args.command) || str(args.pattern) || str(args.path)
                emit({ type: 'progress', text: `${str(p.name)}${target ? ` ${target}` : ''}`.slice(0, 200) })
              }
            }
          }
        }
        if (event.type === 'result') {
          const u = row(event.usage)
          emit({
            type: 'usage',
            inputTokens: (Number(u.input_tokens) || 0) + (Number(u.cache_read_input_tokens) || 0) + (Number(u.cache_creation_input_tokens) || 0),
            outputTokens: Number(u.output_tokens) || 0,
          })
          if (event.is_error || event.subtype !== 'success') {
            throw new Error(Array.isArray(event.errors) ? event.errors.join('\n') : str(event.result) || 'Claude turn failed')
          }
          sawCompletion = true
          finish()
        }
      } else {
        if (typeof event.sessionID === 'string') threadId = event.sessionID
        if (event.type === 'error') throw new Error(str(row(row(event.error).data).message) || 'OpenCode turn failed')
        if (event.type === 'text') text(row(event.part).text)
        if (event.type === 'tool_use' || event.type === 'tool') {
          const part = row(event.part)
          emit({ type: 'progress', text: `${str(part.tool) || str(part.name) || 'tool'}`.slice(0, 200) })
        }
        if (event.type === 'step_start') { lastStepReason = ''; sawCompletion = false }
        if (event.type === 'step_finish') {
          const part = row(event.part), tokens = row(part.tokens), cache = row(tokens.cache)
          emit({
            type: 'usage',
            inputTokens: (Number(tokens.input) || 0) + (Number(cache.read) || 0) + (Number(cache.write) || 0),
            outputTokens: (Number(tokens.output) || 0) + (Number(tokens.reasoning) || 0),
          })
          lastStepReason = str(part.reason)
          sawCompletion = lastStepReason === 'stop'
        }
      }
    }

    child.on('error', (error) => finish(describeSpawnError(input.backend, file, error)))
    child.stdin.on('error', (error) => finish(error))
    child.stderr.on('data', (data: Buffer) => { heard(); stderr = (stderr + data.toString()).slice(-4096) })
    child.stdout.on('data', (data: Buffer) => {
      if (finished) return
      heard()
      buffer += decoder.write(data)
      if (buffer.length > MAX_BUFFER_BYTES) { fail('external CLI event exceeded size limit', 'failed'); return }
      try {
        let newline: number
        while (!finished && (newline = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, newline)
          buffer = buffer.slice(newline + 1)
          parse(line)
        }
      } catch (error) { finish(error instanceof Error ? error : new Error(String(error))) }
    })
    child.on('close', (code) => {
      if (finished) return
      try { parse(buffer + decoder.end()) } catch (error) { finish(error as Error); return }
      if (finished) return
      if (code === 0 && sawCompletion) { finish(); return }
      const midTask = input.backend === 'opencode' && code === 0 && lastStepReason
      fail(midTask
        ? `opencode exited while its turn was still in progress (last step ended with "${lastStepReason}"); the session can be resumed.${stderr ? `\n${stderr}` : ''}`
        : `${input.backend} exited (${code ?? 'unknown'}) before a successful turn completed.${stderr ? `\n${stderr}` : ''}`, midTask ? 'incomplete' : 'failed')
    })

    if (options.signal?.aborted) { abort(); return }
    if (input.backend === 'codex') {
      send({ id: 1, method: 'initialize', params: { clientInfo: { name: 'sepilotd', version: '1.0.0' } } })
    } else if (input.backend === 'claude') {
      child.stdin.end(`${JSON.stringify({ type: 'user', message: { role: 'user', content: [{ type: 'text', text: input.message }] } })}\n`)
    } else {
      child.stdin.end(input.message)
    }
  })

  return { output: texts.join('').trim(), threadId, usage, durationMs: Date.now() - startedAt }
}

function describeSpawnError(backend: ExternalCliBackend, file: string, error: unknown): Error {
  const message = error instanceof Error ? error.message : String(error)
  if (/ENOENT/.test(message)) {
    return new Error(`${backend} CLI not found (${file}). Install it on PATH for the daemon user, or set SEPILOTD_EXTERNAL_CLI_${backend.toUpperCase()}_COMMAND.`)
  }
  return error instanceof Error ? error : new Error(message)
}
