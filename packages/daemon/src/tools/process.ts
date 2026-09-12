import {
  type ChildProcessByStdio,
  execFile,
  spawn,
} from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { readdirSync, readFileSync } from 'node:fs'
import { access, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import type { Readable } from 'node:stream'
import { StringDecoder } from 'node:string_decoder'
import { promisify } from 'node:util'
import xtermHeadless from '@xterm/headless'
import type { Terminal as HeadlessTerminal } from '@xterm/headless'
import type { IHookRegistry, ToolExecutionPosture } from '@sepilotd/core'
import { getAbortError, throwIfAborted } from '../abort.js'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

const execFileAsync = promisify(execFile)
const { Terminal } = xtermHeadless

const DEFAULT_MAX_MANAGED_PROCESSES = 16
const DEFAULT_MANAGED_PROCESS_MAX_OUTPUT_CHARS = 1_000_000
const DEFAULT_MANAGED_PROCESS_TTL_MS = 15 * 60_000
const MAX_MANAGED_PROCESS_TTL_MS = 24 * 60 * 60_000
const STOP_ALL_GRACE_MS = 2_000
const STOP_KILL_SETTLE_MS = 1_000
const FOLLOW_POLL_MS = 100
const DEFAULT_PTY_COLUMNS = 120
const DEFAULT_PTY_ROWS = 40
const MIN_PTY_COLUMNS = 20
const MAX_PTY_COLUMNS = 300
const MIN_PTY_ROWS = 5
const MAX_PTY_ROWS = 120
const PROCESS_HOOK_OUTPUT_CHARS = 64_000
const MAX_MANAGED_LOOPBACK_PORTS = 8
const SANDBOX_LOOPBACK_SOCKET_DIR = '/run/sepilotd/loopback'

const MANAGED_LOOPBACK_HOST_WRAPPER = String.raw`
import json, signal, socket, sys, threading, time

config = json.loads(sys.argv[1])
stop = threading.Event()
listeners = []

def copy_stream(source, target):
    try:
        while True:
            chunk = source.recv(65536)
            if not chunk:
                break
            target.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            target.shutdown(socket.SHUT_WR)
        except OSError:
            pass

def handle(client, path):
    try:
        initial = client.recv(65536)
    except OSError:
        initial = b''
    if not initial:
        client.close()
        return
    upstream = None
    deadline = time.monotonic() + 10.0
    while not stop.is_set() and time.monotonic() < deadline:
        candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            candidate.connect(path)
            upstream = candidate
            break
        except OSError:
            candidate.close()
            time.sleep(0.05)
    if upstream is None:
        client.close()
        return
    try:
        upstream.sendall(initial)
    except OSError:
        client.close()
        upstream.close()
        return
    left = threading.Thread(target=copy_stream, args=(client, upstream), daemon=True)
    right = threading.Thread(target=copy_stream, args=(upstream, client), daemon=True)
    left.start()
    right.start()
    left.join()
    right.join()
    client.close()
    upstream.close()

def serve(listener, path):
    listener.settimeout(0.2)
    while not stop.is_set():
        try:
            client, _ = listener.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        threading.Thread(target=handle, args=(client, path), daemon=True).start()

try:
    for connection in config['connections']:
        bindings = [(socket.AF_INET, ('127.0.0.1', connection['port']))]
        if socket.has_ipv6:
            bindings.append((socket.AF_INET6, ('::1', connection['port'])))
        for family, address in bindings:
            listener = socket.socket(family, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            try:
                listener.bind(address)
            except OSError:
                listener.close()
                if family == socket.AF_INET6:
                    continue
                raise
            listener.listen(64)
            listeners.append(listener)
            threading.Thread(
                target=serve,
                args=(listener, connection['socketPath']),
                daemon=True,
            ).start()
except OSError as error:
    print(json.dumps({'error': str(error), 'errno': error.errno}), file=sys.stderr, flush=True)
    sys.exit(1)

print('READY', flush=True)

def terminate(_signum, _frame):
    stop.set()

signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)
while not stop.wait(0.2):
    pass
for listener in listeners:
    try:
        listener.close()
    except OSError:
        pass
`

const MANAGED_LOOPBACK_SERVER_WRAPPER = String.raw`
import json, os, signal, socket, subprocess, sys, threading, time

config = json.loads(sys.argv[1])
command_index = sys.argv.index('--') + 1
command = sys.argv[command_index:]
stop = threading.Event()
listeners = []

def copy_stream(source, target):
    try:
        while True:
            chunk = source.recv(65536)
            if not chunk:
                break
            target.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            target.shutdown(socket.SHUT_WR)
        except OSError:
            pass

def handle(client, port):
    try:
        initial = client.recv(65536)
    except OSError:
        initial = b''
    if not initial:
        client.close()
        return
    upstream = None
    deadline = time.monotonic() + 10.0
    while not stop.is_set() and time.monotonic() < deadline:
        try:
            upstream = socket.create_connection(('127.0.0.1', port), timeout=0.5)
            upstream.settimeout(None)
            break
        except OSError:
            time.sleep(0.05)
    if upstream is None:
        client.close()
        return
    try:
        upstream.sendall(initial)
    except OSError:
        client.close()
        upstream.close()
        return
    left = threading.Thread(target=copy_stream, args=(client, upstream), daemon=True)
    right = threading.Thread(target=copy_stream, args=(upstream, client), daemon=True)
    left.start()
    right.start()
    left.join()
    right.join()
    client.close()
    upstream.close()

def serve(port):
    path = os.path.join(config['socketDir'], str(port) + '.sock')
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    os.chmod(path, 0o600)
    listener.listen(64)
    listener.settimeout(0.2)
    listeners.append((listener, path))
    while not stop.is_set():
        try:
            client, _ = listener.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        threading.Thread(target=handle, args=(client, port), daemon=True).start()

for port in config['ports']:
    threading.Thread(target=serve, args=(port,), daemon=True).start()

child = subprocess.Popen(command)

def terminate(_signum, _frame):
    if child.poll() is None:
        child.terminate()

signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)
exit_code = child.wait()
stop.set()
for listener, path in listeners:
    try:
        listener.close()
    except OSError:
        pass
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
sys.exit(exit_code)
`

export interface ManagedLoopbackConnection {
  port: number
  socketPath: string
}

interface ManagedLoopbackBridge {
  socketDirectory: string
  connections: ManagedLoopbackConnection[]
  close(): Promise<void>
}

function normalizeManagedLoopbackPorts(rawNetwork: unknown): number[] | null {
  if (rawNetwork == null || rawNetwork === 'none') return null
  if (rawNetwork === 'loopback') return []
  if (!rawNetwork || typeof rawNetwork !== 'object' || Array.isArray(rawNetwork)) {
    throw Object.assign(new Error("network must be 'none' or { mode: 'loopback', ports: [...] }"), {
      code: 'INVALID_NETWORK_CAPABILITY_PERMANENT',
    })
  }
  const network = rawNetwork as Record<string, unknown>
  if (network.mode === 'none') return null
  if (network.mode !== 'loopback') {
    throw Object.assign(new Error("only the managed 'loopback' network capability is supported"), {
      code: 'INVALID_NETWORK_CAPABILITY_PERMANENT',
    })
  }
  const rawPorts = Array.isArray(network.ports) ? network.ports : []
  const ports = [...new Set(rawPorts.map((value) => (
    typeof value === 'string' && /^\d+$/.test(value) ? Number(value) : value
  )).filter((value): value is number => (
    typeof value === 'number'
    && Number.isInteger(value)
    && value >= 1
    && value <= 65_535
  )))]
  if (ports.length === 0) {
    throw Object.assign(new Error('loopback network mode requires at least one TCP port'), {
      code: 'INVALID_NETWORK_CAPABILITY_PERMANENT',
    })
  }
  if (ports.length > MAX_MANAGED_LOOPBACK_PORTS || ports.length !== rawPorts.length) {
    throw Object.assign(
      new Error(`loopback ports must be ${MAX_MANAGED_LOOPBACK_PORTS} or fewer unique TCP ports from 1 to 65535`),
      { code: 'INVALID_NETWORK_CAPABILITY_PERMANENT' },
    )
  }
  return ports
}

const MANAGED_PROCESS_START_FIELDS = new Set([
  'executable',
  'args',
  'cwd',
  'env',
  'ttlMs',
  'lifetime',
  'pty',
  'network',
])

function invalidManagedProcessStartShape(output: string): ToolResult {
  return {
    output,
    status: 'error',
    durationMs: 0,
    code: 'PROCESS_START_INPUT_INVALID_PERMANENT',
  }
}

function validateManagedProcessStartShape(input: Record<string, unknown>): ToolResult | null {
  const unsupported = Object.keys(input).filter((key) => !MANAGED_PROCESS_START_FIELDS.has(key))
  if (unsupported.length > 0) {
    return invalidManagedProcessStartShape(
      `process.start received unsupported field(s): ${unsupported.sort().join(', ')}. `
      + `Allowed fields are: ${[...MANAGED_PROCESS_START_FIELDS].join(', ')}. `
      + "For managed localhost access, use network: { mode: 'loopback', ports: [PORT] }.",
    )
  }

  const network = input.network
  if (network && typeof network === 'object' && !Array.isArray(network)) {
    const unsupportedNetwork = Object.keys(network)
      .filter((key) => key !== 'mode' && key !== 'ports')
    if (unsupportedNetwork.length > 0) {
      return invalidManagedProcessStartShape(
        `process.start network received unsupported field(s): ${unsupportedNetwork.sort().join(', ')}. `
        + "Allowed network fields are: mode, ports.",
      )
    }
  }

  const pty = input.pty
  if (pty && typeof pty === 'object' && !Array.isArray(pty)) {
    const unsupportedPty = Object.keys(pty)
      .filter((key) => key !== 'columns' && key !== 'rows')
    if (unsupportedPty.length > 0) {
      return invalidManagedProcessStartShape(
        `process.start pty received unsupported field(s): ${unsupportedPty.sort().join(', ')}. `
        + 'Allowed pty fields are: columns, rows.',
      )
    }
  }

  return null
}

async function createManagedLoopbackBridge(ports: number[]): Promise<ManagedLoopbackBridge> {
  const socketDirectory = await mkdtemp(join(tmpdir(), 'sepilot-loopback-'))
  const connections = ports.map((port) => ({
    port,
    socketPath: join(socketDirectory, `${port}.sock`),
  }))
  const bridgeProcess = spawn(
    '/usr/bin/python3',
    ['-c', MANAGED_LOOPBACK_HOST_WRAPPER, JSON.stringify({ connections })],
    { stdio: ['ignore', 'pipe', 'pipe'] },
  )
  let stderr = ''
  bridgeProcess.stderr.on('data', (chunk: Buffer | string) => {
    stderr += chunk.toString()
  })
  let closed = false

  const close = async () => {
    if (closed) return
    closed = true
    if (bridgeProcess.exitCode === null && bridgeProcess.signalCode === null) {
      bridgeProcess.kill('SIGTERM')
      await Promise.race([
        new Promise<void>((resolvePromise) => bridgeProcess.once('close', () => resolvePromise())),
        delay(2_000),
      ])
      if (bridgeProcess.exitCode === null && bridgeProcess.signalCode === null) {
        bridgeProcess.kill('SIGKILL')
      }
    }
    await rm(socketDirectory, { recursive: true, force: true })
  }

  try {
    await new Promise<void>((resolvePromise, reject) => {
      let stdout = ''
      const timer = setTimeout(() => reject(new Error('localhost bridge startup timed out')), 5_000)
      const cleanup = () => {
        clearTimeout(timer)
        bridgeProcess.stdout.off('data', onData)
        bridgeProcess.off('close', onClose)
        bridgeProcess.off('error', onError)
      }
      const onData = (chunk: Buffer | string) => {
        stdout += chunk.toString()
        if (!stdout.includes('READY')) return
        cleanup()
        resolvePromise()
      }
      const onClose = () => {
        cleanup()
        reject(new Error(stderr.trim() || 'localhost bridge exited before becoming ready'))
      }
      const onError = (error: Error) => {
        cleanup()
        reject(error)
      }
      bridgeProcess.stdout.on('data', onData)
      bridgeProcess.once('close', onClose)
      bridgeProcess.once('error', onError)
    })
  } catch (error) {
    await close()
    const code = (error as NodeJS.ErrnoException).code
    const addressInUse = code === 'EADDRINUSE' || /errno[^\d]*98|address already in use/iu.test(stderr)
    throw Object.assign(
      new Error(
        addressInUse
          ? `A localhost service is already using one of the requested ports: ${ports.join(', ')}`
          : `Could not expose managed loopback ports ${ports.join(', ')}: ${error instanceof Error ? error.message : String(error)}`,
      ),
      { code: addressInUse ? 'LOOPBACK_PORT_IN_USE_PERMANENT' : 'LOOPBACK_BRIDGE_TRANSIENT' },
    )
  }

  return { socketDirectory, connections, close }
}

async function waitForManagedLoopbackSockets(
  bridge: ManagedLoopbackBridge,
  record: ManagedProcessRecord,
  timeoutMs = 2_000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs
  let socketsReadySince: number | undefined
  while (Date.now() < deadline) {
    if (record.status === 'exited') {
      await record.exitPromise
      const stdout = record.stdout.read(0).chunk.trim()
      const stderr = record.stderr.read(0).chunk.trim()
      const diagnostic = (stderr || stdout || 'no process output').slice(0, 4_000)
      throw Object.assign(
        new Error([
          `Managed process ${record.id} exited before exposing its declared loopback port.`,
          `Command: ${record.command}`,
          `Exit code: ${record.exitCode ?? 'unknown'}`,
          diagnostic,
        ].join('\n')),
        { code: 'MANAGED_PROCESS_START_FAILED_PERMANENT' },
      )
    }
    const ready = await Promise.all(
      bridge.connections.map((connection) => access(connection.socketPath).then(
        () => true,
        () => false,
      )),
    )
    if (ready.every(Boolean)) {
      socketsReadySince ??= Date.now()
      // The wrapper creates its Unix sockets immediately before spawning the
      // requested executable. Require a short stable window so an exec failure
      // cannot race the socket check and be misreported as a running server.
      if (Date.now() - socketsReadySince >= 100) return
    } else {
      socketsReadySince = undefined
    }
    await delay(25)
  }
  throw Object.assign(
    new Error('Managed localhost bridge did not create its session sockets in time'),
    { code: 'LOOPBACK_BRIDGE_TRANSIENT' },
  )
}

function resolveMaxManagedProcesses(): number {
  const raw = Number(process.env.SEPILOTD_MAX_MANAGED_PROCESSES)
  return Number.isFinite(raw) && raw >= 1 ? Math.floor(raw) : DEFAULT_MAX_MANAGED_PROCESSES
}

function resolveManagedProcessMaxOutputChars(): number {
  const raw = Number(process.env.SEPILOTD_MANAGED_PROCESS_MAX_OUTPUT_CHARS)
  return Number.isFinite(raw) && raw >= 1_000
    ? Math.floor(raw)
    : DEFAULT_MANAGED_PROCESS_MAX_OUTPUT_CHARS
}

export function resolveDefaultManagedProcessTtlMs(): number {
  const raw = Number(process.env.SEPILOTD_MANAGED_PROCESS_TTL_MS)
  if (Number.isFinite(raw) && raw === 0) return 0
  return Number.isFinite(raw) && raw >= 100
    ? Math.min(MAX_MANAGED_PROCESS_TTL_MS, Math.floor(raw))
    : DEFAULT_MANAGED_PROCESS_TTL_MS
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms)
    timer.unref?.()
  })
}

export class ManagedProcessLimitError extends Error {
  readonly code = 'MANAGED_PROCESS_LIMIT_PERMANENT'
  constructor(limit: number) {
    super(
      `Refusing to start another managed process: the ${limit}-session limit is reached. Stop an existing session with process.stop before starting a new one.`,
    )
    this.name = 'ManagedProcessLimitError'
  }
}

/**
 * Bounded, multibyte-safe accumulator for a managed process stream. A
 * StringDecoder keeps UTF-8 codepoints intact across chunk boundaries (raw
 * `chunk.toString()` corrupts multibyte sequences split at a byte boundary),
 * and the retained window is capped so a chatty child cannot grow the buffer
 * without limit. Absolute offsets are preserved across eviction so incremental
 * readers stay consistent; a reader that falls behind the evicted window gets a
 * truncation marker instead of silently shifted data.
 */
export class BoundedProcessOutput {
  private readonly decoder = new StringDecoder('utf8')
  private text = ''
  private droppedChars = 0

  constructor(private readonly maxChars: number) {}

  append(chunk: Buffer | string): void {
    const piece = typeof chunk === 'string' ? chunk : this.decoder.write(chunk)
    if (!piece) return
    this.text += piece
    if (this.text.length > this.maxChars) {
      const overflow = this.text.length - this.maxChars
      this.text = this.text.slice(overflow)
      this.droppedChars += overflow
    }
  }

  get totalLength(): number {
    return this.droppedChars + this.text.length
  }

  read(fromOffset: number): { chunk: string; nextOffset: number } {
    const total = this.totalLength
    const clampedFrom = Math.max(0, Math.min(fromOffset, total))
    if (clampedFrom < this.droppedChars) {
      const marker = `[process output truncated: ${this.droppedChars} earlier characters were dropped to bound memory]\n`
      return { chunk: marker + this.text, nextOffset: total }
    }
    return { chunk: this.text.slice(clampedFrom - this.droppedChars), nextOffset: total }
  }
}

export interface ManagedProcessPtyOptions {
  columns: number
  rows: number
}

/**
 * Headless VT screen used for bounded observation of full-screen terminal
 * programs. Raw output remains available for audit/debugging while callers get
 * the current visible screen without having to interpret cursor-control bytes.
 */
export class ManagedProcessPtyScreen {
  private readonly terminal: HeadlessTerminal
  private pendingWrites = 0
  private flushWaiters: Array<() => void> = []

  constructor(readonly options: ManagedProcessPtyOptions) {
    this.terminal = new Terminal({
      cols: options.columns,
      rows: options.rows,
      allowProposedApi: true,
      scrollback: options.rows * 4,
      logLevel: 'off',
    })
  }

  append(chunk: Buffer | string): void {
    this.pendingWrites += 1
    this.terminal.write(
      typeof chunk === 'string' ? chunk : new Uint8Array(chunk),
      () => {
        this.pendingWrites -= 1
        if (this.pendingWrites === 0) {
          for (const resolve of this.flushWaiters.splice(0)) resolve()
        }
      },
    )
  }

  async flush(): Promise<void> {
    if (this.pendingWrites === 0) return
    await new Promise<void>((resolve) => this.flushWaiters.push(resolve))
  }

  snapshot(): string {
    const buffer = this.terminal.buffer.active
    const start = buffer.viewportY
    const lines: string[] = []
    for (let index = 0; index < this.options.rows; index += 1) {
      lines.push(buffer.getLine(start + index)?.translateToString(true) ?? '')
    }
    while (lines.length > 0 && !lines[0]!.trim()) lines.shift()
    while (lines.length > 0 && !lines[lines.length - 1]!.trim()) lines.pop()
    return lines.join('\n')
  }

  dispose(): void {
    this.terminal.dispose()
    for (const resolve of this.flushWaiters.splice(0)) resolve()
  }
}

type ManagedProcessStatus = 'running' | 'exited'
type ManagedProcessLifetime = 'bounded' | 'session'

interface ManagedProcessRecord {
  id: string
  child: ChildProcessByStdio<null, Readable, Readable>
  pid: number
  command: string
  cwd?: string
  executionPosture?: ToolExecutionPosture
  startedAt: string
  lifetime: ManagedProcessLifetime
  processGroupId?: number
  expiresAt?: string
  terminationReason?: 'requested' | 'ttl' | 'daemon-shutdown'
  status: ManagedProcessStatus
  exitCode: number | null
  signal: NodeJS.Signals | null
  stdout: BoundedProcessOutput
  stderr: BoundedProcessOutput
  pty?: ManagedProcessPtyScreen
  ownerSessionId?: string
  ownerExecutionId?: string
  /** Strict workspace capability that may rediscover this process across turns. */
  ownerWorkspaceRoot?: string
  loopbackBridge?: ManagedLoopbackBridge
  hookChain: Promise<void>
  exitPromise: Promise<void>
  finalizeExit: () => void
  ttlTimer?: ReturnType<typeof setTimeout>
}

function normalizeTtlMs(rawValue: unknown, fallback: number): number {
  if (typeof rawValue !== 'number' || !Number.isFinite(rawValue)) {
    return fallback
  }
  if (rawValue === 0) return 0
  return Math.max(100, Math.min(MAX_MANAGED_PROCESS_TTL_MS, Math.trunc(rawValue)))
}

function normalizeArgs(rawArgs: unknown): string[] {
  return Array.isArray(rawArgs)
    ? rawArgs.filter((arg): arg is string => typeof arg === 'string')
    : []
}

function normalizeEnv(
  rawEnv: unknown,
): Record<string, string> | undefined {
  if (!rawEnv || typeof rawEnv !== 'object') {
    return undefined
  }
  const entries = Object.entries(rawEnv)
    .filter((entry): entry is [string, string] => typeof entry[1] === 'string')
  return entries.length > 0
    ? Object.fromEntries(entries)
    : undefined
}

function normalizePtyOptions(rawValue: unknown): ManagedProcessPtyOptions | undefined {
  if (rawValue !== true && (!rawValue || typeof rawValue !== 'object' || Array.isArray(rawValue))) {
    return undefined
  }
  const input = rawValue === true ? {} : rawValue as Record<string, unknown>
  const boundedInteger = (value: unknown, fallback: number, min: number, max: number) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) return fallback
    return Math.max(min, Math.min(max, Math.trunc(value)))
  }
  return {
    columns: boundedInteger(input.columns, DEFAULT_PTY_COLUMNS, MIN_PTY_COLUMNS, MAX_PTY_COLUMNS),
    rows: boundedInteger(input.rows, DEFAULT_PTY_ROWS, MIN_PTY_ROWS, MAX_PTY_ROWS),
  }
}

function quotePosixShellArg(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`
}

function preparePtyCommand(
  executable: string,
  args: string[],
  pty: ManagedProcessPtyOptions | undefined,
): { executable: string; args: string[]; env?: Record<string, string> } {
  if (!pty) return { executable, args }
  if (process.platform !== 'linux') {
    throw Object.assign(
      new Error('Headless PTY background execution is currently available on Linux hosts only.'),
      { code: 'PTY_UNAVAILABLE_PERMANENT' },
    )
  }
  const command = [executable, ...args].map(quotePosixShellArg).join(' ')
  const script = `stty cols ${pty.columns} rows ${pty.rows}; exec ${command}`
  return {
    executable: '/usr/bin/script',
    args: ['-q', '-e', '-f', '-c', script, '/dev/null'],
    env: {
      TERM: 'xterm-256color',
      COLUMNS: String(pty.columns),
      LINES: String(pty.rows),
    },
  }
}

function parseOffset(rawValue: unknown): number {
  if (typeof rawValue !== 'number' || !Number.isFinite(rawValue)) {
    return 0
  }
  return Math.max(0, Math.trunc(rawValue))
}

function parseLimit(rawValue: unknown, fallback: number, max: number): number {
  if (typeof rawValue !== 'number' || !Number.isFinite(rawValue)) {
    return fallback
  }
  return Math.max(1, Math.min(max, Math.trunc(rawValue)))
}

function parseSignal(rawSignal: unknown): NodeJS.Signals {
  return typeof rawSignal === 'string' && rawSignal.trim()
    ? rawSignal.trim() as NodeJS.Signals
    : 'SIGTERM'
}

function snapshotManagedProcess(record: ManagedProcessRecord) {
  return {
    id: record.id,
    pid: record.pid,
    command: record.command,
    cwd: record.cwd,
    executionPosture: record.executionPosture,
    startedAt: record.startedAt,
    lifetime: record.lifetime,
    processGroupId: record.processGroupId,
    expiresAt: record.expiresAt,
    terminationReason: record.terminationReason,
    status: record.status,
    exitCode: record.exitCode,
    signal: record.signal,
    ...(record.loopbackBridge ? {
      loopback: {
        mode: 'managed' as const,
        hosts: ['127.0.0.1', '::1'],
        ports: record.loopbackBridge.connections.map((connection) => connection.port),
      },
    } : {}),
    ...(record.pty ? {
      terminal: {
        mode: 'pty' as const,
        columns: record.pty.options.columns,
        rows: record.pty.options.rows,
      },
    } : {}),
  }
}

/**
 * A POSIX process group can remain addressable while it contains only zombie
 * descendants waiting for the host/container init process to reap them.
 * kill(-pgid, 0) reports that group as present even though no member can run
 * or handle another signal. Linux /proc lets us distinguish that settled
 * state without treating a genuinely live descendant as exited.
 *
 * null means the inspection was not authoritative, so callers must retain the
 * conservative kill(0) result.
 */
function linuxProcessGroupHasLiveMembers(processGroupId: number): boolean | null {
  if (process.platform !== 'linux') return null

  let processEntries: string[]
  try {
    processEntries = readdirSync('/proc')
  } catch {
    return null
  }

  let inspectionIncomplete = false
  for (const entry of processEntries) {
    if (!/^\d+$/.test(entry)) continue
    try {
      const stat = readFileSync(`/proc/${entry}/stat`, 'utf8')
      const commandEnd = stat.lastIndexOf(')')
      if (commandEnd < 0) {
        inspectionIncomplete = true
        continue
      }
      const fields = stat.slice(commandEnd + 1).trim().split(/\s+/)
      const state = fields[0]
      const memberProcessGroupId = Number(fields[2])
      if (memberProcessGroupId !== processGroupId) continue
      if (!['Z', 'X', 'x'].includes(state)) return true
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code
      if (code !== 'ENOENT' && code !== 'ESRCH') inspectionIncomplete = true
    }
  }
  return inspectionIncomplete ? null : false
}

function isManagedTargetAlive(record: ManagedProcessRecord): boolean {
  if (record.processGroupId && process.platform !== 'win32') {
    try {
      process.kill(-record.processGroupId, 0)
      if (record.terminationReason) {
        const hasLiveMembers = linuxProcessGroupHasLiveMembers(record.processGroupId)
        if (hasLiveMembers !== null) return hasLiveMembers
      }
      return true
    } catch (error) {
      return (error as NodeJS.ErrnoException).code === 'EPERM'
    }
  }
  return record.status === 'running'
}

function signalManagedTarget(record: ManagedProcessRecord, signal: NodeJS.Signals): void {
  if (record.processGroupId && process.platform !== 'win32') {
    try {
      process.kill(-record.processGroupId, signal)
      return
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ESRCH') throw error
    }
  }
  record.child.kill(signal)
}

async function waitForManagedTargetExit(
  record: ManagedProcessRecord,
  timeoutMs: number,
): Promise<boolean> {
  const deadline = Date.now() + Math.max(0, timeoutMs)
  while (Date.now() <= deadline) {
    if (!isManagedTargetAlive(record)) return true
    await delay(Math.min(FOLLOW_POLL_MS, Math.max(1, deadline - Date.now())))
  }
  return !isManagedTargetAlive(record)
}

function readManagedProcessOutput(
  record: ManagedProcessRecord,
  stdoutOffset: number,
  stderrOffset: number,
) {
  const stdout = record.stdout.read(stdoutOffset)
  const stderr = record.stderr.read(stderrOffset)
  return {
    stdout: stdout.chunk,
    stderr: stderr.chunk,
    nextStdoutOffset: stdout.nextOffset,
    nextStderrOffset: stderr.nextOffset,
  }
}

function presentManagedProcessOutput<T extends {
  process: ReturnType<typeof snapshotManagedProcess>
  stdout: string
}>(payload: T, includeRawOutput: boolean): T | (Omit<T, 'stdout'> & {
  stdout: ''
  rawOutput: { suppressed: true; stdoutChars: number }
}) {
  if (!payload.process.terminal || includeRawOutput) return payload
  return {
    ...payload,
    stdout: '',
    rawOutput: {
      suppressed: true,
      stdoutChars: payload.stdout.length,
    },
  }
}

export class ManagedProcessRegistry {
  private readonly records = new Map<string, ManagedProcessRecord>()
  private readonly maxProcesses = resolveMaxManagedProcesses()
  private readonly maxOutputChars = resolveManagedProcessMaxOutputChars()
  private readonly defaultTtlMs = resolveDefaultManagedProcessTtlMs()
  private hookRegistry?: IHookRegistry

  setHookRegistry(hookRegistry: IHookRegistry | undefined): void {
    this.hookRegistry = hookRegistry
  }

  private async emitProcessHook(
    event: 'post:process:start' | 'post:process:exit',
    record: ManagedProcessRecord,
  ): Promise<void> {
    if (!this.hookRegistry) return
    try {
      if (event === 'post:process:exit' && record.pty) await record.pty.flush()
      const stdout = event === 'post:process:exit'
        ? record.stdout.read(
            Math.max(0, record.stdout.totalLength - PROCESS_HOOK_OUTPUT_CHARS),
          ).chunk
        : ''
      const stderr = event === 'post:process:exit'
        ? record.stderr.read(
            Math.max(0, record.stderr.totalLength - PROCESS_HOOK_OUTPUT_CHARS),
          ).chunk
        : ''
      await this.hookRegistry.trigger({
        event,
        data: {
          tool: event === 'post:process:start' ? 'process.start' : 'process.exit',
          process: snapshotManagedProcess(record),
          ...(record.ownerSessionId ? { sessionId: record.ownerSessionId } : {}),
          ...(record.ownerExecutionId ? { executionId: record.ownerExecutionId } : {}),
          ...(event === 'post:process:exit' ? {
            output: {
              stdout,
              stderr,
              ...(record.pty ? { screen: record.pty.snapshot() } : {}),
            },
          } : {}),
        },
      })
    } catch {
      // Background lifecycle hooks are observational and must never break or
      // retain a managed child when a handler fails.
    }
  }

  private queueProcessHook(
    event: 'post:process:start' | 'post:process:exit',
    record: ManagedProcessRecord,
  ): void {
    // Serialize lifecycle events per child. A fast process can exit while a
    // slow webhook is still delivering its start event; consumers must never
    // observe exit before start.
    record.hookChain = record.hookChain
      .then(() => this.emitProcessHook(event, record))
      .catch(() => undefined)
  }

  countRunning(): number {
    let count = 0
    for (const record of this.records.values()) {
      if (record.status === 'running') count += 1
    }
    return count
  }

  start(input: {
    executable: string
    args?: string[]
    cwd?: string
    env?: Record<string, string>
    ttlMs?: number
    /** Operator-facing command when the actual child is a sandbox launcher. */
    displayCommand?: string
    executionPosture?: ToolExecutionPosture
    pty?: ManagedProcessPtyOptions
    ownerSessionId?: string
    ownerExecutionId?: string
    ownerWorkspaceRoot?: string
    loopbackBridge?: ManagedLoopbackBridge
  }): ManagedProcessRecord {
    if (this.countRunning() >= this.maxProcesses) {
      throw new ManagedProcessLimitError(this.maxProcesses)
    }
    const args = input.args ?? []
    const startedAt = new Date().toISOString()
    const isolatedProcessGroup = process.platform !== 'win32'
    const ttlMs = normalizeTtlMs(input.ttlMs, this.defaultTtlMs)
    const prepared = preparePtyCommand(input.executable, args, input.pty)
    const child = spawn(prepared.executable, prepared.args, {
      cwd: input.cwd,
      env: input.env || prepared.env
        ? { ...process.env, ...input.env, ...prepared.env }
        : process.env,
      detached: isolatedProcessGroup,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    const id = randomUUID()

    let resolveExit: (() => void) | undefined
    const exitPromise = new Promise<void>((resolve) => {
      resolveExit = resolve
    })

    const record: ManagedProcessRecord = {
      id,
      child,
      pid: child.pid ?? 0,
      command: input.displayCommand ?? [input.executable, ...args].join(' '),
      cwd: input.cwd,
      executionPosture: input.executionPosture,
      startedAt,
      lifetime: ttlMs === 0 ? 'session' : 'bounded',
      ...(isolatedProcessGroup && child.pid ? { processGroupId: child.pid } : {}),
      ...(ttlMs > 0 ? { expiresAt: new Date(Date.now() + ttlMs).toISOString() } : {}),
      status: 'running',
      exitCode: null,
      signal: null,
      stdout: new BoundedProcessOutput(this.maxOutputChars),
      stderr: new BoundedProcessOutput(this.maxOutputChars),
      ...(input.pty ? { pty: new ManagedProcessPtyScreen(input.pty) } : {}),
      ...(input.ownerSessionId ? { ownerSessionId: input.ownerSessionId } : {}),
      ...(input.ownerExecutionId ? { ownerExecutionId: input.ownerExecutionId } : {}),
      ...(input.ownerWorkspaceRoot ? { ownerWorkspaceRoot: resolve(input.ownerWorkspaceRoot) } : {}),
      ...(input.loopbackBridge ? { loopbackBridge: input.loopbackBridge } : {}),
      hookChain: Promise.resolve(),
      exitPromise,
      finalizeExit: () => undefined,
    }
    record.finalizeExit = () => {
      if (record.status === 'exited') return
      record.status = 'exited'
      if (record.ttlTimer) clearTimeout(record.ttlTimer)
      resolveExit?.()
      void record.loopbackBridge?.close()
      this.queueProcessHook('post:process:exit', record)
    }

    child.stdout.on('data', (chunk: Buffer | string) => {
      record.stdout.append(chunk)
      record.pty?.append(chunk)
    })
    child.stderr.on('data', (chunk: Buffer | string) => {
      record.stderr.append(chunk)
    })
    child.on('error', (error) => {
      record.stderr.append(`${error.message}\n`)
      // spawn() reports setup failures asynchronously and does not emit exit
      // when no child was created. Close the registry session immediately so
      // callers do not wait until the TTL for a process that never existed.
      if (!record.pid) record.finalizeExit()
    })
    child.on('close', (code, signal) => {
      record.exitCode = code
      record.signal = signal
      if (!record.processGroupId || !isManagedTargetAlive(record)) {
        record.finalizeExit()
        return
      }
      // A shell wrapper can exit while a child remains in the isolated process
      // group. Keep the session live (and its TTL armed) until the whole group
      // is gone instead of losing ownership of an orphaned daemon.
      void waitForManagedTargetExit(record, ttlMs > 0 ? ttlMs : MAX_MANAGED_PROCESS_TTL_MS)
        .then((exited) => { if (exited) record.finalizeExit() })
    })

    this.records.set(id, record)
    this.queueProcessHook('post:process:start', record)
    if (ttlMs > 0) {
      record.ttlTimer = setTimeout(() => {
        void this.stopAndWait(id, 'SIGTERM', STOP_ALL_GRACE_MS, 'ttl')
      }, ttlMs)
      record.ttlTimer.unref?.()
    }
    return record
  }

  /**
   * Terminate every still-running managed child and drop all records. Invoked
   * from lifecycle teardown so a daemon restart does not leave orphaned
   * background processes behind. Running children get SIGTERM, then any that
   * have not exited within a short grace window are escalated to SIGKILL.
   */
  async stopAll(): Promise<void> {
    const running = [...this.records.values()].filter(
      (record) => record.status === 'running',
    )
    await Promise.all(running.map((record) => (
      this.stopAndWait(record.id, 'SIGTERM', STOP_ALL_GRACE_MS, 'daemon-shutdown')
        .catch(() => null)
    )))
    await Promise.all([...this.records.values()].map((record) => record.hookChain))
    for (const record of this.records.values()) record.pty?.dispose()
    this.records.clear()
  }

  get(id: string): ManagedProcessRecord | null {
    return this.records.get(id) ?? null
  }

  list(): ReturnType<typeof snapshotManagedProcess>[] {
    return [...this.records.values()].map(snapshotManagedProcess)
  }

  isAccessible(id: string, workspaceRoot?: string): boolean {
    const record = this.get(id)
    return Boolean(record && (!workspaceRoot || record.ownerWorkspaceRoot === resolve(workspaceRoot)))
  }

  loopbackConnectionsForSession(sessionId: string | undefined): ManagedLoopbackConnection[] {
    return this.loopbackConnectionsForScope(sessionId)
  }

  /**
   * Return loopback bridges available to the current execution capability.
   *
   * Exact session ownership remains sufficient. A live process may also be
   * reused by a later turn (including a cowork child) when both executions
   * carry the same strict workspace root. This is the capability boundary
   * promised by lifetime='session' + process.sessions; it does not expose a
   * bridge to another workspace or to an unscoped caller.
   */
  loopbackConnectionsForScope(
    sessionId: string | undefined,
    workspaceRoot?: string,
  ): ManagedLoopbackConnection[] {
    const normalizedWorkspaceRoot = workspaceRoot ? resolve(workspaceRoot) : undefined
    if (!sessionId && !normalizedWorkspaceRoot) return []
    return [...this.records.values()]
      .filter((record) => (
        record.status === 'running'
        && record.loopbackBridge
        && (
          (sessionId != null && record.ownerSessionId === sessionId)
          || (
            normalizedWorkspaceRoot != null
            && record.ownerWorkspaceRoot === normalizedWorkspaceRoot
          )
        )
      ))
      .flatMap((record) => record.loopbackBridge?.connections ?? [])
  }

  ownsLoopbackUrl(
    sessionId: string | undefined,
    rawUrl: string,
    workspaceRoot?: string,
  ): boolean {
    return this.loopbackConnectionForUrl(sessionId, rawUrl, workspaceRoot) != null
  }

  loopbackConnectionForUrl(
    sessionId: string | undefined,
    rawUrl: string,
    workspaceRoot?: string,
  ): ManagedLoopbackConnection | null {
    if (!sessionId && !workspaceRoot) return null
    try {
      const url = new URL(rawUrl)
      const hostname = url.hostname.replace(/^\[(.*)\]$/, '$1').toLowerCase()
      if (!['localhost', '127.0.0.1', '::1'].includes(hostname)) return null
      const port = url.port
        ? Number(url.port)
        : url.protocol === 'https:'
          ? 443
          : url.protocol === 'http:'
            ? 80
            : NaN
      return this.loopbackConnectionsForScope(sessionId, workspaceRoot)
        .find((connection) => connection.port === port) ?? null
    } catch {
      return null
    }
  }

  async flushPty(id: string): Promise<void> {
    await this.get(id)?.pty?.flush()
  }

  read(
    id: string,
    stdoutOffset = 0,
    stderrOffset = 0,
  ): {
    process: ReturnType<typeof snapshotManagedProcess>
    stdout: string
    stderr: string
    nextStdoutOffset: number
    nextStderrOffset: number
    screen?: string
  } | null {
    const record = this.get(id)
    if (!record) {
      return null
    }
    return {
      process: snapshotManagedProcess(record),
      ...readManagedProcessOutput(record, stdoutOffset, stderrOffset),
      ...(record.pty ? { screen: record.pty.snapshot() } : {}),
    }
  }

  async wait(
    id: string,
    timeoutMs = 30_000,
    stdoutOffset = 0,
    stderrOffset = 0,
    signal?: AbortSignal,
  ): Promise<{
    process: ReturnType<typeof snapshotManagedProcess>
    stdout: string
    stderr: string
    nextStdoutOffset: number
    nextStderrOffset: number
    timedOut: boolean
    screen?: string
  } | null> {
    throwIfAborted(signal, 'Managed process wait aborted')
    const record = this.get(id)
    if (!record) {
      return null
    }

    let timedOut = false
    if (record.status === 'running') {
      if (timeoutMs <= 0) {
        timedOut = true
      } else {
        await new Promise<void>((resolve, reject) => {
          const cleanup = () => {
            clearTimeout(timer)
            signal?.removeEventListener('abort', onAbort)
          }
          const onAbort = () => {
            cleanup()
            reject(getAbortError(signal, 'Managed process wait aborted'))
          }
          const timer = setTimeout(() => {
            timedOut = true
            cleanup()
            resolve()
          }, timeoutMs)
          signal?.addEventListener('abort', onAbort, { once: true })
          record.exitPromise.then(() => {
            cleanup()
            resolve()
          })
        })
      }
    }

    throwIfAborted(signal, 'Managed process wait aborted')
    await record.pty?.flush()

    return {
      process: snapshotManagedProcess(record),
      ...readManagedProcessOutput(record, stdoutOffset, stderrOffset),
      ...(record.pty ? { screen: record.pty.snapshot() } : {}),
      timedOut,
    }
  }

  async follow(
    id: string,
    timeoutMs = 10_000,
    stdoutOffset = 0,
    stderrOffset = 0,
    signal?: AbortSignal,
    returnOnOutput?: boolean,
  ): Promise<{
    process: ReturnType<typeof snapshotManagedProcess>
    stdout: string
    stderr: string
    nextStdoutOffset: number
    nextStderrOffset: number
    timedOut: boolean
    screen?: string
  } | null> {
    const record = this.get(id)
    if (!record) return null
    const deadline = Date.now() + Math.max(0, timeoutMs)
    let timedOut = false
    const shouldReturnOnOutput = returnOnOutput ?? !record.pty

    while (record.status === 'running') {
      throwIfAborted(signal, 'Managed process log follow aborted')
      const output = readManagedProcessOutput(record, stdoutOffset, stderrOffset)
      if (shouldReturnOnOutput && (output.stdout || output.stderr)) {
        await record.pty?.flush()
        return {
          process: snapshotManagedProcess(record),
          ...output,
          ...(record.pty ? { screen: record.pty.snapshot() } : {}),
          timedOut,
        }
      }
      if (Date.now() >= deadline) {
        timedOut = true
        break
      }
      await Promise.race([
        record.exitPromise,
        delay(Math.min(FOLLOW_POLL_MS, Math.max(1, deadline - Date.now()))),
      ])
    }

    await record.pty?.flush()

    return {
      process: snapshotManagedProcess(record),
      ...readManagedProcessOutput(record, stdoutOffset, stderrOffset),
      ...(record.pty ? { screen: record.pty.snapshot() } : {}),
      timedOut,
    }
  }

  stop(
    id: string,
    signal: NodeJS.Signals = 'SIGTERM',
    reason: ManagedProcessRecord['terminationReason'] = 'requested',
  ): ReturnType<typeof snapshotManagedProcess> | null {
    const record = this.get(id)
    if (!record) {
      return null
    }
    if (record.status === 'running') {
      if (record.ttlTimer) {
        clearTimeout(record.ttlTimer)
        record.ttlTimer = undefined
      }
      record.terminationReason = reason
      signalManagedTarget(record, signal)
    }
    return snapshotManagedProcess(record)
  }

  async stopAndWait(
    id: string,
    signal: NodeJS.Signals = 'SIGTERM',
    graceMs = STOP_ALL_GRACE_MS,
    reason: ManagedProcessRecord['terminationReason'] = 'requested',
  ): Promise<ReturnType<typeof snapshotManagedProcess> | null> {
    const record = this.get(id)
    if (!record) return null
    if (record.status !== 'running') return snapshotManagedProcess(record)

    if (record.ttlTimer) {
      clearTimeout(record.ttlTimer)
      record.ttlTimer = undefined
    }
    record.terminationReason = reason
    signalManagedTarget(record, signal)
    let exited = await waitForManagedTargetExit(
      record,
      signal === 'SIGKILL' ? STOP_KILL_SETTLE_MS : Math.max(0, graceMs),
    )
    if (!exited && signal !== 'SIGKILL') {
      signalManagedTarget(record, 'SIGKILL')
      exited = await waitForManagedTargetExit(record, STOP_KILL_SETTLE_MS)
    }
    if (exited && record.status === 'running') {
      // Group liveness can settle just before Node emits the leader's close
      // event. Give that event one polling interval to retain its exit code or
      // signal before force-finalizing a wrapper whose close event occurred
      // earlier while descendants were still alive.
      await Promise.race([record.exitPromise, delay(FOLLOW_POLL_MS)])
    }
    if (exited && record.status === 'running') {
      // Process-group liveness is authoritative.  Waiting only for the
      // leader's exit event left a short race where callers saw "running"
      // after every descendant was gone on busy hosts.
      record.finalizeExit()
    }
    return snapshotManagedProcess(record)
  }
}

export interface ManagedProcessSandboxLauncher {
  prepareManagedProcess(spec: {
    executable: string
    args: string[]
    cwd?: string
    workspaceRoot: string
    cwdBoundary: ToolExecutionPosture['filesystem']['boundary']
    pty?: ManagedProcessPtyOptions
    loopbackBridge?: {
      socketDirectory: string
      sandboxSocketDirectory: string
      ports: number[]
      wrapperScript: string
    }
  }): Promise<{
    executable: string
    args: string[]
    cwd: string
    executionPosture: ToolExecutionPosture
  }>
}

export function createProcessListTool(): ToolDefinitionRuntime {
  return {
    name: 'process.list',
    description: 'List local processes in a structured JSON format.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'processes' },
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Optional substring filter for the command.' },
        limit: { type: 'number', description: 'Maximum number of processes to return. Defaults to 50.' },
      },
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const query = typeof input.query === 'string' ? input.query.trim().toLowerCase() : ''
      const limit = parseLimit(input.limit, 50, 500)
      try {
        throwIfAborted(context?.signal, 'Process listing aborted')
        const { stdout } = await execFileAsync(
          'ps',
          ['-eo', 'pid=,ppid=,stat=,etime=,command='],
          {
            signal: context?.signal,
            maxBuffer: 10 * 1024 * 1024,
          },
        )
        const rows = stdout
          .split('\n')
          .map((line) => line.trim())
          .filter(Boolean)
          .map((line) => {
            const match = line.match(/^(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(.+)$/)
            if (!match) {
              return null
            }
            return {
              pid: Number(match[1]),
              ppid: Number(match[2]),
              state: match[3],
              elapsed: match[4],
              command: match[5],
            }
          })
          .filter((row): row is NonNullable<typeof row> => row !== null)
          .filter((row) => (query ? row.command.toLowerCase().includes(query) : true))
          .slice(0, limit)
        return {
          output: JSON.stringify(rows, null, 2),
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return {
          output: message,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
    },
  }
}

export function createManagedProcessSessionsTool(
  registry: ManagedProcessRegistry,
): ToolDefinitionRuntime {
  return {
    name: 'process.sessions',
    description: 'List managed background process sessions started through process.start. Use this to rediscover a session id in a later agent turn, then read or follow its logs. Session-lifetime processes remain listed until they exit, are stopped, or the daemon shuts down; bounded processes also expose expiresAt.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'managed-processes' },
    inputSchema: {
      type: 'object',
      properties: {},
    },
    async execute(_input, context): Promise<ToolResult> {
      const start = Date.now()
      return {
        output: JSON.stringify(registry.list().filter(record => registry.isAccessible(record.id, context?.workspaceRoot)), null, 2),
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }
}

export function createManagedProcessStartTool(
  registry: ManagedProcessRegistry,
  options: {
    sandboxMode?: 'local' | 'docker' | 'bubblewrap'
    strictWorkspaceLauncher?: ManagedProcessSandboxLauncher
    strictWorkspaceWriteLauncher?: ManagedProcessSandboxLauncher
  } = {},
): ToolDefinitionRuntime {
  return {
    name: 'process.start',
    unavailableReason: (context) => context.workspaceRoot?.trim()
      && !options.strictWorkspaceLauncher && !options.strictWorkspaceWriteLauncher
      ? 'No strict-workspace background process launcher is configured on this host.'
      : undefined,
    description: "Spawn a sandboxed background process and track it as a managed session. Installed developer toolchains use the same read-only sandbox projection as terminal.run, so invoke the intended runtime directly without rediscovering host paths. The default lifetime is bounded and temporary. For a user-requested development server that should stay available across later agent turns while this daemon remains running, set lifetime to 'session'; retrieve it later with process.sessions and read logs with process.read/process.follow. For a workspace-local server that browser.navigate, webfetch, or terminal.run must reach, set network to { mode: 'loopback', ports: [PORT] }. Managed loopback is agent-local localhost access only: it does not expose 0.0.0.0 or a LAN port on the host. Use service.start when the user explicitly needs host/LAN exposure or survival across daemon restarts.",
    resumeSafety: 'replay-risky',
    validateInput: (input) => validateManagedProcessStartShape(input),
    inputSchema: {
      type: 'object',
      properties: {
        executable: { type: 'string', description: 'Executable to spawn.' },
        args: { type: 'array', items: { type: 'string' }, description: 'Command arguments.' },
        cwd: { type: 'string', description: 'Working directory for the child process. Defaults to the active session cwd.' },
        env: {
          type: 'object',
          description: 'Optional environment variable overrides for non-workspace local execution. Rejected for strict-workspace sandbox launches so values cannot leak through launcher arguments.',
          additionalProperties: { type: 'string' },
        },
        ttlMs: {
          type: 'number',
          description: "Automatic lifetime in milliseconds for lifetime='bounded'. Defaults to 15 minutes. Legacy ttlMs=0 is equivalent to lifetime='session'. Do not combine a positive ttlMs with lifetime='session'.",
        },
        lifetime: {
          type: 'string',
          enum: ['bounded', 'session'],
          description: "Lifecycle scope. 'bounded' is temporary and uses ttlMs/default TTL. 'session' disables TTL and survives agent turns, but is still stopped when the daemon shuts down. Defaults to 'bounded'.",
        },
        pty: {
          description: 'Allocate a headless terminal for full-screen/TUI output. Use true for defaults or provide bounded columns/rows. No interactive stdin is available.',
          anyOf: [
            { type: 'boolean' },
            {
              type: 'object',
              properties: {
                columns: { type: 'number', description: `Terminal width (${MIN_PTY_COLUMNS}-${MAX_PTY_COLUMNS}; default ${DEFAULT_PTY_COLUMNS}).` },
                rows: { type: 'number', description: `Terminal height (${MIN_PTY_ROWS}-${MAX_PTY_ROWS}; default ${DEFAULT_PTY_ROWS}).` },
              },
              additionalProperties: false,
            },
          ],
        },
        network: {
          description: "Optional managed network capability. Omit or use 'none' for full network isolation. Use { mode: 'loopback', ports: [PORT] } only for a local development/test server that must be inspected from browser or HTTP tools.",
          anyOf: [
            { type: 'string', enum: ['none'] },
            {
              type: 'object',
              properties: {
                mode: { type: 'string', enum: ['none', 'loopback'] },
                ports: {
                  type: 'array',
                  items: { type: 'number' },
                  description: `TCP ports to expose on localhost (${MAX_MANAGED_LOOPBACK_PORTS} maximum).`,
                },
              },
              required: ['mode'],
              additionalProperties: false,
            },
          ],
        },
      },
      required: ['executable'],
      additionalProperties: false,
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const invalidShape = validateManagedProcessStartShape(input)
      if (invalidShape) return invalidShape
      const executable = typeof input.executable === 'string'
        ? input.executable.trim()
        : ''
      if (!executable) {
        return {
          output: 'executable is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      try {
        const args = normalizeArgs(input.args)
        const requestedCwd = typeof input.cwd === 'string' && input.cwd.trim()
          ? input.cwd
          : context?.cwd
        const env = normalizeEnv(input.env)
        const pty = normalizePtyOptions(input.pty)
        const loopbackPorts = normalizeManagedLoopbackPorts(input.network)
        const lifetime = typeof input.lifetime === 'string' ? input.lifetime.trim() : ''
        if (lifetime && lifetime !== 'bounded' && lifetime !== 'session') {
          return {
            output: "lifetime must be 'bounded' or 'session'",
            status: 'error',
            durationMs: Date.now() - start,
            code: 'PROCESS_LIFETIME_INVALID_PERMANENT',
          }
        }
        const explicitTtlMs = typeof input.ttlMs === 'number' ? input.ttlMs : undefined
        if (lifetime === 'session' && explicitTtlMs !== undefined && explicitTtlMs !== 0) {
          return {
            output: "lifetime='session' cannot be combined with a positive ttlMs",
            status: 'error',
            durationMs: Date.now() - start,
            code: 'PROCESS_LIFETIME_CONFLICT_PERMANENT',
          }
        }
        if (lifetime === 'bounded' && explicitTtlMs === 0) {
          return {
            output: "lifetime='bounded' requires a positive ttlMs or an omitted ttlMs",
            status: 'error',
            durationMs: Date.now() - start,
            code: 'PROCESS_LIFETIME_CONFLICT_PERMANENT',
          }
        }
        const managedTtlMs = lifetime === 'session' ? 0 : explicitTtlMs
        if (context?.workspaceRoot) {
          const workspaceWriteAuthorized = Boolean(
            context.delegatedAgentPolicy
            && context.delegatedAgentPolicy.autonomy !== 'readonly',
          )
          const strictWorkspaceLauncher = workspaceWriteAuthorized
            ? options.strictWorkspaceWriteLauncher
            : options.strictWorkspaceLauncher
          if (!strictWorkspaceLauncher) {
            return {
              output: 'Strict-workspace background execution requires an active managed-process sandbox launcher. The command was not run on the host.',
              status: 'error',
              durationMs: Date.now() - start,
              code: 'SANDBOX_UNAVAILABLE',
            }
          }
          if (env && Object.keys(env).length > 0) {
            return {
              output: 'Strict-workspace background execution does not accept env overrides because bubblewrap command-line environment values would be visible to other host processes. Configure non-secret runtime inputs through workspace files, or use a supervised non-workspace process.',
              status: 'error',
              durationMs: Date.now() - start,
              code: 'SANDBOX_ENV_UNSUPPORTED_PERMANENT',
            }
          }
          const loopbackBridge = loopbackPorts
            ? await createManagedLoopbackBridge(loopbackPorts)
            : undefined
          let prepared
          try {
            prepared = await strictWorkspaceLauncher.prepareManagedProcess({
              executable,
              args,
              cwd: requestedCwd,
              workspaceRoot: context.workspaceRoot,
              cwdBoundary: 'strict_workspace',
              pty,
              ...(loopbackBridge ? {
                loopbackBridge: {
                  socketDirectory: loopbackBridge.socketDirectory,
                  sandboxSocketDirectory: SANDBOX_LOOPBACK_SOCKET_DIR,
                  ports: loopbackPorts!,
                  wrapperScript: MANAGED_LOOPBACK_SERVER_WRAPPER,
                },
              } : {}),
            })
          } catch (error) {
            await loopbackBridge?.close()
            throw error
          }
          let record
          try {
            record = registry.start({
              executable: prepared.executable,
              args: prepared.args,
              cwd: prepared.cwd,
              ttlMs: managedTtlMs,
              displayCommand: [executable, ...args].join(' '),
              executionPosture: prepared.executionPosture,
              pty,
              ownerSessionId: context.sessionId,
              ownerExecutionId: context.executionId,
              ownerWorkspaceRoot: context.workspaceRoot,
              loopbackBridge,
            })
          } catch (error) {
            await loopbackBridge?.close()
            throw error
          }
          if (loopbackBridge) {
            try {
              await waitForManagedLoopbackSockets(loopbackBridge, record)
            } catch (error) {
              await registry.stopAndWait(record.id, 'SIGTERM', STOP_ALL_GRACE_MS, 'requested')
                .catch(() => null)
              throw error
            }
          }
          return {
            output: JSON.stringify(snapshotManagedProcess(record), null, 2),
            status: 'success',
            durationMs: Date.now() - start,
            executionPosture: prepared.executionPosture,
          }
        }

        if (options.sandboxMode === 'docker' || options.sandboxMode === 'bubblewrap') {
          return {
            output:
              `process.start is disabled while security.sandbox=${options.sandboxMode} unless the request carries a strict workspace backed by the managed-process sandbox launcher.`,
            status: 'error',
            durationMs: Date.now() - start,
            code: 'SANDBOX_REQUIRED_PERMANENT',
          }
        }
        if (loopbackPorts) {
          return {
            output: 'Managed loopback exposure requires a strict-workspace sandbox; the command was not run on the host.',
            status: 'error',
            durationMs: Date.now() - start,
            code: 'SANDBOX_REQUIRED_PERMANENT',
          }
        }
        const record = registry.start({
          executable,
          args,
          cwd: requestedCwd,
          env,
          ttlMs: managedTtlMs,
          pty,
          ownerSessionId: context?.sessionId,
          ownerExecutionId: context?.executionId,
        })
        return {
          output: JSON.stringify(snapshotManagedProcess(record), null, 2),
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return {
          output: message,
          status: 'error',
          durationMs: Date.now() - start,
          ...((error instanceof ManagedProcessLimitError || (error && typeof error === 'object' && 'code' in error))
            ? { code: String((error as { code: unknown }).code) }
            : {}),
        }
      }
    },
  }
}

export function createManagedProcessFollowTool(
  registry: ManagedProcessRegistry,
): ToolDefinitionRuntime {
  return {
    name: 'process.follow',
    description: 'Wait for incremental output, process exit, or a bounded observation window. Log sessions return on new output; PTY sessions wait for the full timeout by default and then return the latest screen.',
    resumeSafety: 'replay-safe',
    scheduling: {
      mode: 'parallel-safe',
      resource: 'managed-processes',
      key: (input) => typeof input.id === 'string' ? input.id : null,
    },
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Managed process session id.' },
        timeoutMs: { type: 'number', description: 'Maximum wait for new output or exit. Defaults to 10000ms; capped at 30000ms.' },
        stdoutOffset: { type: 'number', description: 'Previously consumed stdout character offset.' },
        stderrOffset: { type: 'number', description: 'Previously consumed stderr character offset.' },
        includeRawOutput: { type: 'boolean', description: 'For PTY sessions, also return raw ANSI stdout. Defaults to false because the clean screen snapshot is smaller and easier to consume.' },
        returnOnOutput: { type: 'boolean', description: 'Return as soon as new output arrives. Defaults to true for log sessions and false for continuously updating PTY screens.' },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (id && !registry.isAccessible(id, context?.workspaceRoot)) {
        return { output: `managed process '${id}' not found`, status: 'error', durationMs: Date.now() - start }
      }
      if (!id) {
        return { output: 'id is required', status: 'error', durationMs: Date.now() - start }
      }
      const payload = await registry.follow(
        id,
        parseLimit(input.timeoutMs, 10_000, 30_000),
        parseOffset(input.stdoutOffset),
        parseOffset(input.stderrOffset),
        context?.signal,
        typeof input.returnOnOutput === 'boolean' ? input.returnOnOutput : undefined,
      )
      if (!payload) {
        return {
          output: `managed process '${id}' not found`,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      return {
        output: JSON.stringify(
          presentManagedProcessOutput(payload, input.includeRawOutput === true),
          null,
          2,
        ),
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }
}

export function createManagedProcessReadTool(
  registry: ManagedProcessRegistry,
): ToolDefinitionRuntime {
  return {
    name: 'process.read',
    description: 'Read incremental stdout and stderr from a managed background process session.',
    resumeSafety: 'replay-safe',
    scheduling: {
      mode: 'parallel-safe',
      resource: 'managed-processes',
      key: (input) => typeof input.id === 'string' ? input.id : null,
    },
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Managed process session id.' },
        stdoutOffset: { type: 'number', description: 'Previously consumed stdout character offset.' },
        stderrOffset: { type: 'number', description: 'Previously consumed stderr character offset.' },
        includeRawOutput: { type: 'boolean', description: 'For PTY sessions, also return raw ANSI stdout. Defaults to false because the clean screen snapshot is smaller and easier to consume.' },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (id && !registry.isAccessible(id, context?.workspaceRoot)) {
        return { output: `managed process '${id}' not found`, status: 'error', durationMs: Date.now() - start }
      }
      if (!id) {
        return {
          output: 'id is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      await registry.flushPty(id)
      const payload = registry.read(
        id,
        parseOffset(input.stdoutOffset),
        parseOffset(input.stderrOffset),
      )
      if (!payload) {
        return {
          output: `managed process '${id}' not found`,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      return {
        output: JSON.stringify(
          presentManagedProcessOutput(payload, input.includeRawOutput === true),
          null,
          2,
        ),
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }
}

export function createManagedProcessWaitTool(
  registry: ManagedProcessRegistry,
): ToolDefinitionRuntime {
  return {
    name: 'process.wait',
    description: 'Wait for a managed process session to exit and return any unread output.',
    resumeSafety: 'replay-safe',
    scheduling: {
      mode: 'parallel-safe',
      resource: 'managed-processes',
      key: (input) => typeof input.id === 'string' ? input.id : null,
    },
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Managed process session id.' },
        timeoutMs: { type: 'number', description: 'Maximum wait time in milliseconds. Defaults to 30000.' },
        stdoutOffset: { type: 'number', description: 'Previously consumed stdout character offset.' },
        stderrOffset: { type: 'number', description: 'Previously consumed stderr character offset.' },
        includeRawOutput: { type: 'boolean', description: 'For PTY sessions, also return raw ANSI stdout. Defaults to false because the clean screen snapshot is smaller and easier to consume.' },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (id && !registry.isAccessible(id, context?.workspaceRoot)) {
        return { output: `managed process '${id}' not found`, status: 'error', durationMs: Date.now() - start }
      }
      if (!id) {
        return {
          output: 'id is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      const payload = await registry.wait(
        id,
        parseLimit(input.timeoutMs, 30_000, 300_000),
        parseOffset(input.stdoutOffset),
        parseOffset(input.stderrOffset),
        context?.signal,
      )
      if (!payload) {
        return {
          output: `managed process '${id}' not found`,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      return {
        output: JSON.stringify(
          presentManagedProcessOutput(payload, input.includeRawOutput === true),
          null,
          2,
        ),
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }
}

export function createManagedProcessStopTool(
  registry: ManagedProcessRegistry,
): ToolDefinitionRuntime {
  return {
    name: 'process.stop',
    description: 'Stop a managed background process session with a signal.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string', description: 'Managed process session id.' },
        signal: { type: 'string', description: 'Signal such as SIGTERM, SIGINT, or SIGKILL.' },
        graceMs: { type: 'number', description: 'Wait before escalating a non-SIGKILL stop to SIGKILL. Defaults to 2000ms; capped at 30000ms.' },
      },
      required: ['id'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      const id = typeof input.id === 'string' ? input.id.trim() : ''
      if (id && !registry.isAccessible(id, context?.workspaceRoot)) {
        return { output: `managed process '${id}' not found`, status: 'error', durationMs: Date.now() - start }
      }
      if (!id) {
        return {
          output: 'id is required',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      const record = await registry.stopAndWait(
        id,
        parseSignal(input.signal),
        parseLimit(input.graceMs, STOP_ALL_GRACE_MS, 30_000),
      )
      if (!record) {
        return {
          output: `managed process '${id}' not found`,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
      return {
        output: JSON.stringify(record, null, 2),
        status: 'success',
        durationMs: Date.now() - start,
      }
    },
  }
}

export function createProcessSignalTool(
  registry?: ManagedProcessRegistry,
): ToolDefinitionRuntime {
  return {
    name: 'process.signal',
    description:
      'Send a signal to a process previously started by the agent via `process.start`. PIDs not owned by the agent are refused so a runaway tool cannot kill arbitrary host processes (sshd, daemon, other users).',
    inputSchema: {
      type: 'object',
      properties: {
        pid: { type: 'number', description: 'Target process id (must be one started via process.start).' },
        signal: { type: 'string', description: 'Signal name such as SIGTERM, SIGINT, SIGKILL, or 0.' },
      },
      required: ['pid'],
    },
    async execute(input): Promise<ToolResult> {
      const start = Date.now()
      const pid = typeof input.pid === 'number' ? Math.trunc(input.pid) : Number.NaN
      if (!Number.isFinite(pid) || pid <= 0) {
        return {
          output: 'pid must be a positive integer',
          status: 'error',
          durationMs: Date.now() - start,
        }
      }

      // PID ownership check: only allow signalling processes the agent
      // itself started via the ManagedProcessRegistry AND that are still
      // running. Without the liveness half, an exited record keeps a dead
      // PID alive as an owned target; once the OS recycles that PID onto an
      // unrelated process, the agent could signal (and kill) it. Requiring a
      // running record closes the PID-reuse bypass while still letting the
      // agent stop its own live children.
      if (registry) {
        const owned = registry
          .list()
          .some((record) => record.pid === pid && record.status === 'running')
        if (!owned) {
          return {
            output:
              'refusing to signal a pid that was not started by this agent via process.start',
            status: 'error',
            durationMs: Date.now() - start,
            code: 'UNOWNED_PID_PERMANENT',
          }
        }
      }

      const rawSignal = typeof input.signal === 'string' ? input.signal.trim() : 'SIGTERM'
      const signal = rawSignal === '0' ? 0 : rawSignal

      try {
        process.kill(pid, signal as NodeJS.Signals | 0)
        return {
          output: signal === 0
            ? `Process ${pid} is reachable`
            : `Sent ${rawSignal} to process ${pid}`,
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return {
          output: message,
          status: 'error',
          durationMs: Date.now() - start,
        }
      }
    },
  }
}
