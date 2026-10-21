import { spawn as defaultSpawn } from 'node:child_process'
import { JsonRpcFramer, type JsonRpcMessage } from '../acp/rpc.js'
import { DiagnosticCollector, type LspDiagnostic, type LspDiagnosticSeverity } from './diagnostics.js'
import type { LspServerSpec } from './registry.js'

export interface LspChildProcess {
  stdout: { on(event: 'data', cb: (chunk: Buffer) => void): void } | null
  stdin: { write(data: Buffer | string): void } | null
  on(event: 'exit', cb: (code: number | null) => void): void
  kill(signal?: NodeJS.Signals): void
}

export interface LspSpawner {
  (bin: string, args: readonly string[]): LspChildProcess
}

interface LspRange {
  start: { line: number; character: number }
  end: { line: number; character: number }
}

interface LspPublishDiagnosticsParams {
  uri: string
  diagnostics: Array<{
    message: string
    range: LspRange
    severity?: number
    source?: string
    code?: string | number
  }>
}

export interface LspLocation {
  uri: string
  range: LspRange
}

export interface LspDocumentSymbol {
  name: string
  kind: number
  range: LspRange
  selectionRange: LspRange
  children?: LspDocumentSymbol[]
}

export interface LspWorkspaceSymbol {
  name: string
  kind: number
  location: LspLocation
  containerName?: string
}

export interface LspHover {
  contents: unknown
  range?: LspRange
}

interface PendingRequest {
  resolve: (value: unknown) => void
  reject: (reason?: unknown) => void
  timer: ReturnType<typeof setTimeout> | null
}

const DEFAULT_RPC_TIMEOUT_MS = 5_000

function severityFromNumber(n?: number): LspDiagnosticSeverity | undefined {
  switch (n) {
    case 1: return 'error'
    case 2: return 'warning'
    case 3: return 'info'
    case 4: return 'hint'
    default: return undefined
  }
}

export class LspClient {
  private readonly framer = new JsonRpcFramer()
  private readonly spawner: LspSpawner
  private proc: LspChildProcess | null = null
  private nextId = 1
  private readonly pending = new Map<number, PendingRequest>()
  private exitCallback: ((code: number | null) => void) | null = null
  private initialized = false
  private readonly openDocs = new Map<string, { languageId: string; text: string }>()
  readonly diagnostics = new DiagnosticCollector()

  constructor(spawner?: LspSpawner) {
    this.spawner = spawner ?? ((bin, args) => defaultSpawn(bin, [...args], {
      stdio: ['pipe', 'pipe', 'inherit'],
    }) as unknown as LspChildProcess)
  }

  start(spec: LspServerSpec, rootUri: string): void {
    if (this.proc) throw new Error('LspClient already started')
    const proc = this.spawner(spec.bin, spec.args)
    this.proc = proc
    proc.stdout?.on('data', (chunk: Buffer) => {
      for (const msg of this.framer.push(chunk)) {
        this.handle(msg)
      }
    })
    proc.on('exit', (code) => {
      for (const [id, req] of this.pending) {
        if (req.timer) clearTimeout(req.timer)
        req.reject(new Error(`LSP server exited (code=${code ?? 'null'})`))
        this.pending.delete(id)
      }
      this.proc = null
      this.initialized = false
      this.openDocs.clear()
      const cb = this.exitCallback
      if (cb) {
        try {
          cb(code)
        } catch {
        }
      }
    })
    void this.request(
      'initialize',
      {
        processId: typeof process !== 'undefined' ? process.pid : null,
        rootUri,
        capabilities: {
          textDocument: {
            synchronization: { didSave: true, dynamicRegistration: false },
            publishDiagnostics: { relatedInformation: false },
          },
        },
      },
      0,
    )
      .then(() => {
        this.initialized = true
        this.notify('initialized', {})
        for (const [uri, doc] of this.openDocs) {
          this.sendDidOpen(uri, doc.languageId, doc.text)
        }
      })
      .catch(() => {
      })
  }

  ensureOpen(uri: string, languageId: string, text: string): boolean {
    if (this.openDocs.has(uri)) return false
    this.openDocs.set(uri, { languageId, text })
    if (this.initialized) {
      this.sendDidOpen(uri, languageId, text)
    }
    return true
  }

  private sendDidOpen(uri: string, languageId: string, text: string): void {
    this.notify('textDocument/didOpen', {
      textDocument: { uri, languageId, version: 1, text },
    })
  }

  private notify(method: string, params: Record<string, unknown>): void {
    this.send({ jsonrpc: '2.0', method, params })
  }

  onExit(cb: (code: number | null) => void): void {
    this.exitCallback = cb
  }

  isRunning(): boolean {
    return this.proc !== null
  }

  stop(signal: NodeJS.Signals = 'SIGTERM'): void {
    for (const [id, req] of this.pending) {
      if (req.timer) clearTimeout(req.timer)
      req.reject(new Error('LspClient stopped'))
      this.pending.delete(id)
    }
    this.proc?.kill(signal)
    this.proc = null
    this.initialized = false
    this.openDocs.clear()
  }

  async request<T = unknown>(
    method: string,
    params: Record<string, unknown>,
    timeoutMs: number = DEFAULT_RPC_TIMEOUT_MS,
  ): Promise<T> {
    const id = this.nextId++
    const promise = new Promise<T>((resolve, reject) => {
      const timer = timeoutMs > 0
        ? setTimeout(() => {
            this.pending.delete(id)
            reject(new Error(`LSP request '${method}' timed out after ${timeoutMs}ms`))
          }, timeoutMs)
        : null
      this.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
        timer,
      })
    })
    this.send({ jsonrpc: '2.0', id, method, params })
    return promise
  }

  documentSymbol(uri: string): Promise<LspDocumentSymbol[]> {
    return this.request<LspDocumentSymbol[]>('textDocument/documentSymbol', {
      textDocument: { uri },
    })
  }

  references(uri: string, line: number, character: number): Promise<LspLocation[]> {
    return this.request<LspLocation[]>('textDocument/references', {
      textDocument: { uri },
      position: { line, character },
      context: { includeDeclaration: false },
    })
  }

  async definition(uri: string, line: number, character: number): Promise<LspLocation[]> {
    const result = await this.request<LspLocation | LspLocation[] | null>(
      'textDocument/definition',
      { textDocument: { uri }, position: { line, character } },
    )
    if (!result) return []
    return Array.isArray(result) ? result : [result]
  }

  hover(uri: string, line: number, character: number): Promise<LspHover | null> {
    return this.request<LspHover | null>('textDocument/hover', {
      textDocument: { uri },
      position: { line, character },
    })
  }

  async workspaceSymbol(query: string): Promise<LspWorkspaceSymbol[]> {
    const result = await this.request<LspWorkspaceSymbol[] | null>('workspace/symbol', {
      query,
    })
    return result ?? []
  }

  private handle(msg: JsonRpcMessage): void {
    if (typeof msg.id === 'number' && this.pending.has(msg.id)) {
      const pending = this.pending.get(msg.id)!
      this.pending.delete(msg.id)
      if (pending.timer) clearTimeout(pending.timer)
      const errorBag = (msg as { error?: { message?: string } }).error
      if (errorBag) {
        pending.reject(new Error(errorBag.message ?? `LSP error for request ${msg.id}`))
      } else {
        pending.resolve((msg as { result?: unknown }).result)
      }
      return
    }
    if (msg.method !== 'textDocument/publishDiagnostics') return
    const params = msg.params as LspPublishDiagnosticsParams | undefined
    if (!params?.uri || !Array.isArray(params.diagnostics)) return
    const mapped: LspDiagnostic[] = params.diagnostics.map((d) => ({
      message: d.message,
      line: d.range.start.line + 1,
      severity: severityFromNumber(d.severity),
      source: d.source,
      code: d.code,
    }))
    this.diagnostics.update(params.uri, mapped)
  }

  private send(msg: JsonRpcMessage): void {
    const stdin = this.proc?.stdin
    if (!stdin) return
    stdin.write(this.framer.frame(msg))
  }
}
