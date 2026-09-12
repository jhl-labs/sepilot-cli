import { spawn as defaultSpawn } from 'node:child_process'
import type { Readable, Writable } from 'node:stream'
import {
  createJsonRpcFramer,
  type JsonRpcCodec,
  type JsonRpcFraming,
  type JsonRpcMessage,
} from './rpc.js'

export interface AcpChildProcess {
  stdout: Pick<Readable, 'on'> | null
  stderr?: Pick<Readable, 'on'> | null
  stdin: Pick<Writable, 'write'> | null
  on(event: 'exit', cb: (code: number | null, signal?: NodeJS.Signals | null) => void): void
  on(event: 'error', cb: (error: Error) => void): void
  kill(signal?: NodeJS.Signals): void
}

export interface AcpSpawnOptions {
  cwd?: string
  env?: NodeJS.ProcessEnv
}

export interface AcpSpawner {
  (command: string, args: readonly string[], options: AcpSpawnOptions): AcpChildProcess
}

export interface AcpClientSpec {
  command: string
  args?: readonly string[]
  cwd?: string
  env?: NodeJS.ProcessEnv
  framing?: JsonRpcFraming
}

export interface AcpClientOptions {
  spawner?: AcpSpawner
  requestTimeoutMs?: number
  onNotification?: (message: JsonRpcMessage) => void
  onClientRequest?: (message: JsonRpcMessage) => Promise<unknown> | unknown
  onStderr?: (chunk: Buffer) => void
  onExit?: (error: Error) => void
}

interface PendingRequest {
  method: string
  resolve: (value: unknown) => void
  reject: (reason?: unknown) => void
  timer: ReturnType<typeof setTimeout> | null
}

const DEFAULT_REQUEST_TIMEOUT_MS = 30_000
const ACP_PROTOCOL_VERSION = 1

function requestKey(id: string | number | null | undefined): string {
  return String(id)
}

function isRequest(message: JsonRpcMessage): boolean {
  return Boolean(message.method) && Object.hasOwn(message, 'id')
}

export class AcpStdioClient {
  private readonly framer: JsonRpcCodec
  private readonly spawner: AcpSpawner
  private readonly pending = new Map<string, PendingRequest>()
  private proc: AcpChildProcess | null = null
  private nextId = 1
  private stopped = false

  constructor(
    private readonly spec: AcpClientSpec,
    private readonly options: AcpClientOptions = {},
  ) {
    this.framer = createJsonRpcFramer(spec.framing)
    this.spawner = options.spawner ?? ((command, args, spawnOptions) => defaultSpawn(command, [...args], {
      cwd: spawnOptions.cwd,
      env: spawnOptions.env,
      stdio: ['pipe', 'pipe', 'pipe'],
    }) as unknown as AcpChildProcess)
  }

  start(): void {
    if (this.proc) throw new Error('ACP client already started')
    this.stopped = false
    const proc = this.spawner(this.spec.command, this.spec.args ?? [], {
      cwd: this.spec.cwd,
      env: this.spec.env,
    })
    this.proc = proc

    proc.stdout?.on('data', (chunk: Buffer) => {
      for (const message of this.framer.push(chunk)) {
        void this.handle(message)
      }
    })
    proc.stderr?.on('data', (chunk: Buffer) => {
      this.options.onStderr?.(chunk)
    })
    proc.on('error', (error) => {
      this.proc = null
      this.rejectPending(error)
      this.options.onExit?.(error)
    })
    proc.on('exit', (code, signal) => {
      this.proc = null
      const error = new Error(`ACP agent exited (code=${code ?? 'null'}, signal=${signal ?? 'null'})`)
      this.rejectPending(error)
      this.options.onExit?.(error)
    })
  }

  isRunning(): boolean {
    return this.proc !== null
  }

  stop(signal: NodeJS.Signals = 'SIGTERM'): void {
    this.stopped = true
    this.rejectPending(new Error('ACP client stopped'))
    const proc = this.proc
    if (proc) {
      const forceKill = setTimeout(() => proc.kill('SIGKILL'), 5000)
      forceKill.unref()
      proc.on('exit', () => clearTimeout(forceKill))
      proc.kill(signal)
    }
    this.proc = null
  }

  async initialize(): Promise<unknown> {
    return this.request('initialize', {
      protocolVersion: ACP_PROTOCOL_VERSION,
      clientInfo: { name: 'sepilotd', title: 'sepilotd', version: '0.0.0' },
      clientCapabilities: {
        fs: { readTextFile: false, writeTextFile: false },
        terminal: false,
      },
    })
  }

  async newSession(params: { cwd: string; mcpServers?: unknown[] }): Promise<{ sessionId: string }> {
    const result = await this.request('session/new', {
      cwd: params.cwd,
      mcpServers: params.mcpServers ?? [],
    })
    if (!result || typeof result !== 'object') {
      throw new Error('ACP session/new returned a non-object result')
    }
    const sessionId = (result as { sessionId?: unknown }).sessionId
    if (typeof sessionId !== 'string' || sessionId.trim().length === 0) {
      throw new Error('ACP session/new did not return sessionId')
    }
    return { sessionId }
  }

  prompt(sessionId: string, prompt: string): Promise<unknown> {
    return this.request('session/prompt', {
      sessionId,
      prompt: [{ type: 'text', text: prompt }],
    })
  }

  cancel(sessionId: string): void {
    this.notify('session/cancel', { sessionId })
  }

  request(method: string, params?: unknown, timeoutMs = this.options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS): Promise<unknown> {
    if (!this.proc?.stdin) throw new Error('ACP client is not started')
    const id = this.nextId++
    const key = requestKey(id)
    const promise = new Promise<unknown>((resolve, reject) => {
      const timer = timeoutMs > 0
        ? setTimeout(() => {
            this.pending.delete(key)
            reject(new Error(`ACP request '${method}' timed out after ${timeoutMs}ms`))
          }, timeoutMs)
        : null
      this.pending.set(key, { method, resolve, reject, timer })
    })
    this.proc.stdin.write(this.framer.frame({ jsonrpc: '2.0', id, method, params }))
    return promise
  }

  notify(method: string, params?: unknown): void {
    if (!this.proc?.stdin) return
    this.proc.stdin.write(this.framer.frame({ jsonrpc: '2.0', method, params }))
  }

  private async handle(message: JsonRpcMessage): Promise<void> {
    if (this.stopped) return
    if (Object.hasOwn(message, 'id') && !message.method) {
      const key = requestKey(message.id)
      const pending = this.pending.get(key)
      if (!pending) return
      this.pending.delete(key)
      if (pending.timer) clearTimeout(pending.timer)
      if (message.error) {
        pending.reject(new Error(message.error.message || `ACP request '${pending.method}' failed`))
      } else {
        pending.resolve(message.result)
      }
      return
    }

    if (isRequest(message)) {
      await this.respondToClientRequest(message)
      return
    }

    this.options.onNotification?.(message)
  }

  private async respondToClientRequest(message: JsonRpcMessage): Promise<void> {
    if (!this.proc?.stdin) return
    try {
      const result = this.options.onClientRequest
        ? await this.options.onClientRequest(message)
        : this.defaultClientRequestResponse(message)
      this.proc.stdin.write(this.framer.frame({
        jsonrpc: '2.0',
        id: message.id,
        result,
      }))
    } catch (error) {
      this.proc.stdin.write(this.framer.frame({
        jsonrpc: '2.0',
        id: message.id,
        error: {
          code: (error as { code?: number }).code ?? -32601,
          message: error instanceof Error ? error.message : String(error),
        },
      }))
    }
  }

  private defaultClientRequestResponse(message: JsonRpcMessage): unknown {
    if (message.method === 'session/request_permission') {
      return denyPermissionResult(message.params)
    }
    const error = new Error(`method not found: ${String(message.method)}`) as Error & { code?: number }
    error.code = -32601
    throw error
  }

  private rejectPending(error: Error): void {
    for (const [key, pending] of this.pending) {
      if (pending.timer) clearTimeout(pending.timer)
      pending.reject(error)
      this.pending.delete(key)
    }
  }
}

export function denyPermissionResult(params: unknown): { outcome: { outcome: 'selected'; optionId: string } } | { outcome: { outcome: 'cancelled' } } {
  const options = params && typeof params === 'object'
    ? (params as { options?: unknown }).options
    : undefined
  if (Array.isArray(options)) {
    const reject = options.find((option) => {
      if (!option || typeof option !== 'object') return false
      const record = option as Record<string, unknown>
      return record.kind === 'reject_once' && typeof record.optionId === 'string'
    }) as { optionId?: string } | undefined
    if (reject?.optionId) {
      return { outcome: { outcome: 'selected', optionId: reject.optionId } }
    }
  }
  return { outcome: { outcome: 'cancelled' } }
}
