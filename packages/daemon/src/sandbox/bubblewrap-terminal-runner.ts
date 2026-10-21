import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { constants } from 'node:fs'
import { access, mkdir, readFile, realpath, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { basename, delimiter, dirname, isAbsolute, join, resolve } from 'node:path'
import type { ToolExecutionPosture } from '@sepilotd/core'
import { getAbortError, isAbortError, throwIfAborted } from '../abort.js'
import { createLogger } from '../logger.js'
import type {
  TerminalRunner,
  TerminalRunnerResult,
  TerminalRunSpec,
} from '../tools/terminal.js'
import {
  defaultSandboxToolchainProjection,
  type SandboxToolchainProjection,
} from './toolchain-projection.js'

const log = createLogger('bwrap-runner')

const MAX_OUTPUT_BYTES = 10 * 1024 * 1024

const SYSTEM_RO_PATHS = [
  '/usr',
  '/bin',
  '/sbin',
  '/lib',
  '/lib64',
  '/etc/alternatives',
  '/etc/hosts',
  '/etc/nsswitch.conf',
  '/etc/resolv.conf',
  '/etc/ssl',
]

const NETWORK_RESOLV_CONF_CANDIDATES = [
  '/run/systemd/resolve/resolv.conf',
  '/run/NetworkManager/no-stub-resolv.conf',
] as const

function hasNonLoopbackNameserver(content: string): boolean {
  return content.split(/\r?\n/u).some((line) => {
    const match = /^\s*nameserver\s+(\S+)/u.exec(line)
    if (!match) return false
    const address = match[1]!.replace(/^\[|\]$/gu, '').toLowerCase()
    return address !== '::1'
      && address !== 'localhost'
      && !address.startsWith('127.')
  })
}

export async function resolveNetworkResolvConfPath(
  candidates: readonly string[] = NETWORK_RESOLV_CONF_CANDIDATES,
): Promise<string | null> {
  for (const candidate of candidates) {
    try {
      const content = await readFile(candidate, 'utf8')
      if (hasNonLoopbackNameserver(content)) return candidate
    } catch {
    }
  }
  return null
}

const SANDBOX_RUNTIME_DIR = '/run/sepilotd'
const SANDBOX_KUBECTL_PATH = `${SANDBOX_RUNTIME_DIR}/bin/kubectl`
const SANDBOX_KUBECONFIG_PATH = `${SANDBOX_RUNTIME_DIR}/kubeconfig`
const SANDBOX_TEA_PATH = `${SANDBOX_RUNTIME_DIR}/bin/tea`
const SANDBOX_TEA_CONFIG_DIR = `${SANDBOX_RUNTIME_DIR}/tea`
const SANDBOX_TEA_CONFIG_PATH = `${SANDBOX_TEA_CONFIG_DIR}/config.yml`
const SANDBOX_CACHE_DIR = `${SANDBOX_RUNTIME_DIR}/cache`
const MANAGED_LOOPBACK_CLIENT_WRAPPER = String.raw`
import json, signal, socket, subprocess, sys, threading

connections = json.loads(sys.argv[1])
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

def handle(client, path):
    upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        upstream.connect(path)
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

for connection in connections:
    bindings = [(socket.AF_INET, ('127.0.0.1', connection['port']))]
    if socket.has_ipv6:
        bindings.append((socket.AF_INET6, ('::1', connection['port'])))
    for family, address in bindings:
        listener = socket.socket(family, socket.SOCK_STREAM)
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
        threading.Thread(target=serve, args=(listener, connection['socketPath']), daemon=True).start()

child = subprocess.Popen(command)

def terminate(_signum, _frame):
    if child.poll() is None:
        child.terminate()

signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)
exit_code = child.wait()
stop.set()
for listener in listeners:
    try:
        listener.close()
    except OSError:
        pass
sys.exit(exit_code)
`

export interface SandboxReadOnlyBind {
  source: string
  target: string
}

export interface SandboxResourceLimits {
  pidsMax: number
  memoryMaxBytes: number
  tmpfsSizeBytes: number
}

export const DEFAULT_SANDBOX_RESOURCE_LIMITS: SandboxResourceLimits = {
  pidsMax: 512,
  memoryMaxBytes: 4 * 1024 * 1024 * 1024,
  tmpfsSizeBytes: 256 * 1024 * 1024,
}

export interface SandboxLauncherCapabilities {
  cgroup: boolean
  prlimit: boolean
}

function resolveResourceLimits(partial?: Partial<SandboxResourceLimits>): SandboxResourceLimits {
  return { ...DEFAULT_SANDBOX_RESOURCE_LIMITS, ...(partial ?? {}) }
}

const bwrapSizeSupport = new Map<string, Promise<boolean>>()

function bwrapSupportsSizeOption(bwrapPath: string): Promise<boolean> {
  const cached = bwrapSizeSupport.get(bwrapPath)
  if (cached) return cached
  const probe = new Promise<boolean>((resolve) => {
    let child
    try {
      child = spawn(bwrapPath, ['--help'], { stdio: ['ignore', 'pipe', 'pipe'] })
    } catch {
      resolve(false)
      return
    }
    let out = ''
    child.stdout?.on('data', (c: Buffer) => {
      out += c.toString('utf8')
    })
    child.stderr?.on('data', (c: Buffer) => {
      out += c.toString('utf8')
    })
    child.on('error', () => resolve(false))
    child.on('close', () => resolve(/--size\b/.test(out)))
  })
  bwrapSizeSupport.set(bwrapPath, probe)
  return probe
}

const binaryAvailability = new Map<string, Promise<boolean>>()

function probeBinaryRuns(bin: string, args: string[]): Promise<boolean> {
  const key = `${bin} ${args.join(' ')}`
  const cached = binaryAvailability.get(key)
  if (cached) return cached
  const probe = new Promise<boolean>((resolve) => {
    let child
    try {
      child = spawn(bin, args, { stdio: 'ignore' })
    } catch {
      resolve(false)
      return
    }
    child.on('error', () => resolve(false))
    child.on('close', (code) => resolve(code === 0))
  })
  binaryAvailability.set(key, probe)
  return probe
}

async function resolveLauncherCapabilities(): Promise<SandboxLauncherCapabilities> {
  const [cgroup, prlimit] = await Promise.all([
    probeBinaryRuns('systemd-run', ['--user', '--scope', '--quiet', '--', 'true']),
    probeBinaryRuns('prlimit', ['--as=4294967296', '--', 'true']),
  ])
  return { cgroup, prlimit }
}

export function isSandboxSetupFailure(stderr: string): boolean {
  return (
    /prlimit: failed to execute/i.test(stderr) ||
    /Failed to (?:start|allocate) .*scope/i.test(stderr) ||
    /bwrap: (?:Can't|Unknown option|Creating new namespace failed|No permissions|setting up|Unexpected)/i.test(
      stderr,
    )
  )
}

export function isSandboxTargetExecutableMissing(stderr: string): boolean {
  return /bwrap:\s+execvp\s+[^:\r\n]+:\s+No such file or directory/i.test(stderr)
}

export interface BubblewrapTerminalRunnerConfig {
  networkMode?: 'none' | 'host'
  readOnlyWorkspace?: boolean
  bwrapPath?: string
  processIsolation?: 'namespace' | 'container-boundary'
  resourceLimits?: Partial<SandboxResourceLimits>
  projectHostToolchains?: boolean
  toolchainProjection?: SandboxToolchainProjection
  cacheRoot?: string
}

export interface BubblewrapArgvInput {
  workspace: string
  cwd?: string
  executable: string
  args: string[]
  networkMode: 'none' | 'host'
  readOnlyWorkspace: boolean
  processIsolation?: 'namespace' | 'container-boundary'
  tmpfsSizeBytes?: number
  readOnlyBinds?: SandboxReadOnlyBind[]
  readWriteBinds?: SandboxReadOnlyBind[]
  capabilityEnv?: Record<string, string>
  sandboxPath?: string
  sandboxExecutable?: string
  resolvConfPath?: string
}

function sandboxCacheEnv(): Record<string, string> {
  return {
    GOMODCACHE: `${SANDBOX_CACHE_DIR}/go-mod`,
    GOCACHE: `${SANDBOX_CACHE_DIR}/go-build`,
    NPM_CONFIG_CACHE: `${SANDBOX_CACHE_DIR}/npm`,
    npm_config_store_dir: `${SANDBOX_CACHE_DIR}/pnpm-store`,
    PIP_CACHE_DIR: `${SANDBOX_CACHE_DIR}/pip`,
    CARGO_HOME: `${SANDBOX_CACHE_DIR}/cargo`,
    GRADLE_USER_HOME: `${SANDBOX_CACHE_DIR}/gradle`,
    XDG_CACHE_HOME: `${SANDBOX_CACHE_DIR}/xdg`,
  }
}

export function buildBubblewrapArgv(input: BubblewrapArgvInput): string[] {
  const argv: string[] = [
    '--die-with-parent',
    '--unshare-ipc',
    '--unshare-uts',
    '--unshare-user-try',
    '--new-session',
    '--clearenv',
    '--setenv', 'CI', '1',
    '--setenv', 'HOME', '/tmp',
    '--setenv', 'PATH', input.sandboxPath ?? '/usr/local/bin:/usr/bin:/bin',
    '--setenv', 'PYTHONPYCACHEPREFIX', '/tmp/sepilot-pycache',
    '--setenv', 'SEPILOTD_SANDBOX', 'bubblewrap',
    '--dev', '/dev',
  ]

  if ((input.processIsolation ?? 'namespace') === 'namespace') {
    argv.splice(1, 0, '--unshare-pid')
    argv.push('--proc', '/proc')
  } else {
    argv.push('--ro-bind', '/proc', '/proc')
  }

  if (input.tmpfsSizeBytes && input.tmpfsSizeBytes > 0) {
    argv.push('--size', String(input.tmpfsSizeBytes))
  }
  argv.push('--tmpfs', '/tmp')

  for (const path of SYSTEM_RO_PATHS) {
    argv.push('--ro-bind-try', path, path)
  }

  if (input.networkMode === 'host' && input.resolvConfPath) {
    argv.push('--ro-bind', input.resolvConfPath, '/etc/resolv.conf')
  }

  if (input.readOnlyBinds?.length || input.readWriteBinds?.length) {
    argv.push(
      '--dir', '/run',
      '--dir', SANDBOX_RUNTIME_DIR,
      '--dir', `${SANDBOX_RUNTIME_DIR}/bin`,
      '--dir', SANDBOX_TEA_CONFIG_DIR,
      '--dir', `${SANDBOX_RUNTIME_DIR}/toolchains`,
      '--dir', `${SANDBOX_RUNTIME_DIR}/toolchains/bin`,
    )
  }
  if (input.readOnlyBinds?.length) {
    for (const binding of input.readOnlyBinds) {
      argv.push('--ro-bind', binding.source, binding.target)
    }
  }
  if (input.readWriteBinds?.length) {
    for (const binding of input.readWriteBinds) {
      argv.push('--dir', binding.target)
      argv.push('--bind', binding.source, binding.target)
    }
  }
  for (const [key, value] of Object.entries(input.capabilityEnv ?? {})) {
    argv.push('--setenv', key, value)
  }
  if (input.readOnlyWorkspace) {
    argv.push('--ro-bind', input.workspace, input.workspace)
  } else {
    argv.push('--bind', input.workspace, input.workspace)
  }
  argv.push('--chdir', input.cwd ?? input.workspace)

  if (input.networkMode === 'none') {
    argv.push('--unshare-net')
  }

  argv.push('--', input.sandboxExecutable ?? input.executable, ...input.args)
  return argv
}

async function executablePathOnHost(executable: string): Promise<string | null> {
  const candidates = isAbsolute(executable)
    ? [executable]
    : (process.env.PATH ?? '')
        .split(delimiter)
        .filter(Boolean)
        .map((entry) => resolve(entry, executable))
  for (const candidate of candidates) {
    try {
      await access(candidate, constants.X_OK)
      return await realpath(candidate)
    } catch {
    }
  }
  return null
}

async function kubeconfigPathOnHost(): Promise<string | null> {
  const configured = (process.env.KUBECONFIG ?? '')
    .split(delimiter)
    .map((entry) => entry.trim())
    .filter(Boolean)
  const candidates = configured.length > 0
    ? configured
    : [join(homedir(), '.kube', 'config')]
  for (const candidate of candidates) {
    try {
      const resolved = isAbsolute(candidate) ? candidate : resolve(candidate)
      if ((await stat(resolved)).isFile()) return await realpath(resolved)
    } catch {
    }
  }
  return null
}

async function teaExecutablePathOnHost(): Promise<string | null> {
  const candidates = [
    '/usr/bin/tea',
    '/usr/local/bin/tea',
    '/snap/bin/tea',
    join(homedir(), '.local', 'bin', 'tea'),
  ]
  for (const candidate of candidates) {
    try {
      await access(candidate, constants.X_OK)
      return await realpath(candidate)
    } catch {
    }
  }
  return null
}

async function teaConfigPathOnHost(): Promise<string | null> {
  const configuredRoot = process.env.XDG_CONFIG_HOME?.trim()
  const candidates = [
    ...(configuredRoot ? [join(configuredRoot, 'tea', 'config.yml')] : []),
    join(homedir(), '.config', 'tea', 'config.yml'),
  ]
  for (const candidate of candidates) {
    try {
      const resolved = isAbsolute(candidate) ? candidate : resolve(candidate)
      if ((await stat(resolved)).isFile()) return await realpath(resolved)
    } catch {
    }
  }
  return null
}

function teaArgsWithJsonOutput(args: string[]): string[] {
  const hasOutput = args.some((arg) => (
    arg === '--output'
    || arg === '-o'
    || arg.startsWith('--output=')
    || arg.startsWith('-o=')
  ))
  return hasOutput ? args : [...args, '--output', 'json']
}

interface PreparedCapabilityInvocation {
  executable: string
  args: string[]
  bindings: Pick<
    BubblewrapArgvInput,
    'readOnlyBinds' | 'capabilityEnv' | 'sandboxExecutable'
  >
}

async function prepareCapabilityInvocation(
  spec: TerminalRunSpec,
): Promise<PreparedCapabilityInvocation> {
  if (spec.capability === 'managed-loopback') {
    const connections = spec.managedLoopbackConnections ?? []
    const directories = [...new Set(connections.map((connection) => dirname(connection.socketPath)))]
    const targetByDirectory = new Map(
      directories.map((directory, index) => [directory, `${SANDBOX_RUNTIME_DIR}/loopback-${index}`]),
    )
    const mappedConnections = connections.map((connection) => ({
      port: connection.port,
      socketPath: join(
        targetByDirectory.get(dirname(connection.socketPath))!,
        basename(connection.socketPath),
      ),
    }))
    return {
      executable: '/usr/bin/python3',
      args: [
        '-c',
        MANAGED_LOOPBACK_CLIENT_WRAPPER,
        JSON.stringify(mappedConnections),
        '--',
        spec.executable,
        ...spec.args,
      ],
      bindings: {
        readOnlyBinds: directories.map((directory) => ({
          source: directory,
          target: targetByDirectory.get(directory)!,
        })),
      },
    }
  }

  if (spec.capability === 'gitea-actions-readonly') {
    const executable = await teaExecutablePathOnHost()
    if (!executable) return { executable: spec.executable, args: spec.args, bindings: {} }
    const config = await teaConfigPathOnHost()
    return {
      executable: spec.executable,
      args: teaArgsWithJsonOutput(spec.args),
      bindings: {
        sandboxExecutable: SANDBOX_TEA_PATH,
        readOnlyBinds: [
          { source: executable, target: SANDBOX_TEA_PATH },
          ...(config ? [{ source: config, target: SANDBOX_TEA_CONFIG_PATH }] : []),
        ],
        capabilityEnv: {
          TZ: 'UTC',
          ...(config ? { XDG_CONFIG_HOME: SANDBOX_RUNTIME_DIR } : {}),
        },
      },
    }
  }

  if (spec.capability !== 'kubectl-readonly') {
    return { executable: spec.executable, args: spec.args, bindings: {} }
  }

  const executable = await executablePathOnHost(spec.executable)
  if (!executable) return { executable: spec.executable, args: spec.args, bindings: {} }
  const kubeconfig = await kubeconfigPathOnHost()
  return {
    executable: spec.executable,
    args: spec.args,
    bindings: {
      sandboxExecutable: SANDBOX_KUBECTL_PATH,
      readOnlyBinds: [
        { source: executable, target: SANDBOX_KUBECTL_PATH },
        ...(kubeconfig
          ? [{ source: kubeconfig, target: SANDBOX_KUBECONFIG_PATH }]
          : []),
      ],
      ...(kubeconfig
        ? { capabilityEnv: { KUBECONFIG: SANDBOX_KUBECONFIG_PATH } }
        : {}),
    },
  }
}

export function buildBubblewrapCommand(
  input: BubblewrapArgvInput,
  limits: SandboxResourceLimits,
  bwrapPath: string,
  caps: SandboxLauncherCapabilities,
): { command: string; argv: string[] } {
  const bwrapArgv = buildBubblewrapArgv({ ...input, tmpfsSizeBytes: limits.tmpfsSizeBytes })

  let command = bwrapPath
  let argv = bwrapArgv
  if (caps.prlimit && limits.memoryMaxBytes > 0) {
    argv = [`--as=${limits.memoryMaxBytes}`, '--', command, ...argv]
    command = 'prlimit'
  }

  if (caps.cgroup && (limits.pidsMax > 0 || limits.memoryMaxBytes > 0)) {
    const scopeFlags = ['--user', '--scope', '--quiet']
    if (limits.pidsMax > 0) scopeFlags.push('-p', `TasksMax=${limits.pidsMax}`)
    if (limits.memoryMaxBytes > 0) scopeFlags.push('-p', `MemoryMax=${limits.memoryMaxBytes}`)
    argv = [...scopeFlags, '--', command, ...argv]
    command = 'systemd-run'
  }

  return { command, argv }
}

export function buildBubblewrapExecutionPosture(input: {
  cwd: string
  boundary: ToolExecutionPosture['filesystem']['boundary']
  active: boolean
  networkMode: 'none' | 'host'
  readOnlyWorkspace?: boolean
  fallbackReason?: string
  capability?: TerminalRunSpec['capability']
}): ToolExecutionPosture {
  return {
    sandbox: {
      requested: true,
      active: input.active,
      mode: input.active ? 'bubblewrap' : 'host',
      ...(input.fallbackReason ? { fallbackReason: input.fallbackReason } : {}),
    },
    filesystem: {
      cwd: input.cwd,
      boundary: input.boundary,
      isolated: input.active,
      readOnly: input.active && input.readOnlyWorkspace === true,
      note: input.active
        ? input.capability === 'gitea-actions-readonly'
          ? 'bubblewrap binds the workspace read-only plus the fixed tea binary and one tea config file for bounded Actions run list/view; host home, SSH keys, other credentials, and daemon state are not mounted.'
          : input.capability === 'kubectl-readonly'
          ? 'bubblewrap binds the workspace read-only plus the resolved kubectl binary and one kubeconfig file; host home, SSH keys, and daemon state are not mounted.'
          : input.capability === 'workspace-network-write'
            ? 'bubblewrap binds only the active workspace read-write and exposes host networking after explicit policy approval; host home, SSH keys, credentials, and daemon state are not mounted.'
          : input.capability === 'managed-loopback'
            ? `bubblewrap binds the active workspace ${input.readOnlyWorkspace === true ? 'read-only' : 'read-write'} and exposes only daemon-owned Unix socket bridges for localhost services available to this session or strict workspace; external network, host home, credentials, and daemon state remain unavailable.`
          : 'bubblewrap binds only the workspace (system dirs read-only, private /tmp); host home, SSH keys, credentials, and daemon state are not mounted.'
        : 'bubblewrap sandbox was requested but did not become active; command ran on the host.',
    },
    network: {
      isolated: input.active && (
        input.networkMode === 'none' || input.capability === 'managed-loopback'
      ),
      mode: input.active
        ? input.capability === 'managed-loopback'
          ? 'managed-loopback'
          : input.networkMode
        : 'host',
    },
  }
}

export class BubblewrapTerminalRunner implements TerminalRunner {
  private readonly bwrapPath: string
  private readonly networkMode: 'none' | 'host'
  private readonly readOnlyWorkspace: boolean
  private readonly processIsolation: 'namespace' | 'container-boundary'
  private readonly resourceLimits: SandboxResourceLimits
  private readonly toolchainProjection: Promise<SandboxToolchainProjection>
  private readonly cacheRoot?: string

  constructor(config: BubblewrapTerminalRunnerConfig = {}) {
    this.bwrapPath = config.bwrapPath ?? 'bwrap'
    this.networkMode = config.networkMode ?? 'none'
    this.readOnlyWorkspace = config.readOnlyWorkspace ?? false
    this.processIsolation = config.processIsolation ?? 'namespace'
    this.resourceLimits = resolveResourceLimits(config.resourceLimits)
    this.cacheRoot = config.cacheRoot ? resolve(config.cacheRoot) : undefined
    this.toolchainProjection = config.toolchainProjection
      ? Promise.resolve(config.toolchainProjection)
      : config.projectHostToolchains === false
        ? Promise.resolve({
            path: '/usr/local/bin:/usr/bin:/bin',
            readOnlyBinds: [],
            env: {},
          })
        : defaultSandboxToolchainProjection()
  }

  private async workspaceCacheBinding(workspace: string): Promise<{
    binding: SandboxReadOnlyBind
    env: Record<string, string>
  } | null> {
    if (!this.cacheRoot) return null
    let canonicalWorkspace = resolve(workspace)
    try {
      canonicalWorkspace = await realpath(canonicalWorkspace)
    } catch {
    }
    const workspaceKey = createHash('sha256')
      .update(canonicalWorkspace)
      .digest('hex')
      .slice(0, 32)
    const source = resolve(this.cacheRoot, workspaceKey)
    await mkdir(source, { recursive: true, mode: 0o700 })
    return {
      binding: { source, target: SANDBOX_CACHE_DIR },
      env: sandboxCacheEnv(),
    }
  }

  async prepareManagedProcess(spec: {
    executable: string
    args: string[]
    cwd?: string
    workspaceRoot: string
    cwdBoundary: ToolExecutionPosture['filesystem']['boundary']
    pty?: { columns: number; rows: number }
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
  }> {
    const available = await probeBinaryRuns(this.bwrapPath, ['--version'])
    if (!available) {
      throw new Error(`bubblewrap is unavailable at '${this.bwrapPath}'`)
    }
    const workspace = spec.workspaceRoot
    const cwd = spec.cwd ?? workspace
    const sizeSupported = this.resourceLimits.tmpfsSizeBytes > 0
      ? await bwrapSupportsSizeOption(this.bwrapPath)
      : false
    const effectiveLimits: SandboxResourceLimits = {
      ...this.resourceLimits,
      tmpfsSizeBytes: sizeSupported ? this.resourceLimits.tmpfsSizeBytes : 0,
    }
    const caps = await resolveLauncherCapabilities()
    const toolchainProjection = await this.toolchainProjection
    const workspaceCache = await this.workspaceCacheBinding(workspace)
    const managedExecutable = spec.loopbackBridge ? '/usr/bin/python3' : spec.executable
    const managedArgs = spec.loopbackBridge
      ? [
          '-c',
          spec.loopbackBridge.wrapperScript,
          JSON.stringify({
            ports: spec.loopbackBridge.ports,
            socketDir: spec.loopbackBridge.sandboxSocketDirectory,
          }),
          '--',
          spec.executable,
          ...spec.args,
        ]
      : spec.args
    const prepared = buildBubblewrapCommand(
      {
        workspace,
        cwd,
        executable: managedExecutable,
        args: managedArgs,
        networkMode: this.networkMode,
        readOnlyWorkspace: this.readOnlyWorkspace,
        processIsolation: this.processIsolation,
        resolvConfPath: this.networkMode === 'host'
          ? (await resolveNetworkResolvConfPath()) ?? undefined
          : undefined,
        sandboxPath: toolchainProjection.path,
        readOnlyBinds: toolchainProjection.readOnlyBinds,
        readWriteBinds: workspaceCache ? [workspaceCache.binding] : undefined,
        capabilityEnv: {
          ...toolchainProjection.env,
          ...(workspaceCache?.env ?? {}),
        },
        ...(spec.loopbackBridge ? {
          readWriteBinds: [
            ...(workspaceCache ? [workspaceCache.binding] : []),
            {
              source: spec.loopbackBridge.socketDirectory,
              target: spec.loopbackBridge.sandboxSocketDirectory,
            },
          ],
        } : {}),
        ...(spec.pty ? {
          capabilityEnv: {
            ...toolchainProjection.env,
            ...(workspaceCache?.env ?? {}),
            TERM: 'xterm-256color',
            COLUMNS: String(spec.pty.columns),
            LINES: String(spec.pty.rows),
          },
        } : {}),
      },
      effectiveLimits,
      this.bwrapPath,
      caps,
    )
    return {
      executable: prepared.command,
      args: prepared.argv,
      cwd: workspace,
      executionPosture: buildBubblewrapExecutionPosture({
        cwd,
        boundary: spec.cwdBoundary,
        active: true,
        networkMode: this.networkMode,
        readOnlyWorkspace: this.readOnlyWorkspace,
        ...(spec.loopbackBridge ? { capability: 'managed-loopback' } : {}),
      }),
    }
  }

  async run(spec: TerminalRunSpec): Promise<TerminalRunnerResult> {
    throwIfAborted(spec.signal, `Command ${spec.executable} aborted`)
    const workspace = spec.workspaceRoot ?? spec.cwd ?? process.cwd()
    const cwd = spec.cwd ?? workspace
    const sizeSupported =
      this.resourceLimits.tmpfsSizeBytes > 0
        ? await bwrapSupportsSizeOption(this.bwrapPath)
        : false
    const effectiveLimits: SandboxResourceLimits = {
      ...this.resourceLimits,
      tmpfsSizeBytes: sizeSupported ? this.resourceLimits.tmpfsSizeBytes : 0,
    }
    const caps = await resolveLauncherCapabilities()
    const capabilityInvocation = await prepareCapabilityInvocation(spec)
    const toolchainProjection = await this.toolchainProjection
    const workspaceCache = await this.workspaceCacheBinding(workspace)
    const resolvConfPath = this.networkMode === 'host'
      ? (await resolveNetworkResolvConfPath()) ?? undefined
      : undefined
    const { command, argv } = buildBubblewrapCommand(
      {
        workspace,
        cwd,
        executable: capabilityInvocation.executable,
        args: capabilityInvocation.args,
        networkMode: this.networkMode,
        readOnlyWorkspace: this.readOnlyWorkspace,
        processIsolation: this.processIsolation,
        resolvConfPath,
        sandboxPath: toolchainProjection.path,
        readOnlyBinds: [
          ...toolchainProjection.readOnlyBinds,
          ...(capabilityInvocation.bindings.readOnlyBinds ?? []),
        ],
        readWriteBinds: workspaceCache ? [workspaceCache.binding] : undefined,
        capabilityEnv: {
          ...toolchainProjection.env,
          ...(workspaceCache?.env ?? {}),
          ...(capabilityInvocation.bindings.capabilityEnv ?? {}),
        },
        ...(capabilityInvocation.bindings.sandboxExecutable
          ? { sandboxExecutable: capabilityInvocation.bindings.sandboxExecutable }
          : {}),
      },
      effectiveLimits,
      this.bwrapPath,
      caps,
    )

    const start = Date.now()
    return new Promise<TerminalRunnerResult>((resolvePromise) => {
      let settled = false
      const settle = (value: TerminalRunnerResult) => {
        if (settled) return
        settled = true
        clearTimeout(timer)
        if (spec.signal) spec.signal.removeEventListener('abort', onAbort)
        resolvePromise(value)
      }

      const fallback = (reason: string, code: string): TerminalRunnerResult => ({
        stdout: '',
        stderr: reason,
        status: 'error',
        durationMs: Date.now() - start,
        executionPosture: buildBubblewrapExecutionPosture({
          cwd,
          boundary: spec.cwdBoundary,
          active: false,
          networkMode: this.networkMode,
          readOnlyWorkspace: this.readOnlyWorkspace,
          fallbackReason: reason,
          capability: spec.capability,
        }),
        code,
      })

      let child
      try {
        child = spawn(command, argv, { stdio: ['ignore', 'pipe', 'pipe'] })
      } catch (err) {
        log.warn('bubblewrap spawn failed', {
          error: err instanceof Error ? err.message : String(err),
        })
        settle(fallback(`bubblewrap could not start: ${err instanceof Error ? err.message : String(err)}`, 'SANDBOX_UNAVAILABLE'))
        return
      }

      const onAbort = () => {
        child.kill('SIGKILL')
        settle({
          stdout,
          stderr: stderr || 'aborted',
          status: 'error',
          durationMs: Date.now() - start,
          executionPosture: activePosture(),
          code: 'ABORTED',
        })
      }
      if (spec.signal) spec.signal.addEventListener('abort', onAbort, { once: true })

      const timer = setTimeout(() => {
        child.kill('SIGKILL')
        settle({
          stdout,
          stderr: stderr || `timed out after ${spec.timeoutMs}ms`,
          status: 'error',
          durationMs: Date.now() - start,
          executionPosture: activePosture(),
          code: 'TIMEOUT',
        })
      }, spec.timeoutMs)

      const activePosture = (): ToolExecutionPosture =>
        buildBubblewrapExecutionPosture({
          cwd,
          boundary: spec.cwdBoundary,
          active: true,
          networkMode: this.networkMode,
          readOnlyWorkspace: this.readOnlyWorkspace,
          capability: spec.capability,
        })

      let stdout = ''
      let stderr = ''
      child.stdout.on('data', (chunk: Buffer) => {
        if (stdout.length < MAX_OUTPUT_BYTES) stdout += chunk.toString('utf8')
      })
      child.stderr.on('data', (chunk: Buffer) => {
        if (stderr.length < MAX_OUTPUT_BYTES) stderr += chunk.toString('utf8')
      })
      child.on('error', (err) => {
        settle(fallback(`bubblewrap not available: ${err.message}`, 'SANDBOX_UNAVAILABLE'))
      })
      child.on('close', (exitCode) => {
        if (isAbortError(undefined) || spec.signal?.aborted) return
        if (exitCode !== 0 && isSandboxSetupFailure(stderr)) {
          settle(fallback(`bubblewrap sandbox setup failed: ${stderr.trim()}`, 'SANDBOX_UNAVAILABLE'))
          return
        }
        const targetExecutableMissing = exitCode !== 0
          && isSandboxTargetExecutableMissing(stderr)
        settle({
          stdout,
          stderr,
          status: exitCode === 0 ? 'success' : 'error',
          exitCode,
          durationMs: Date.now() - start,
          executionPosture: activePosture(),
          ...(exitCode !== 0
            ? {
                code: targetExecutableMissing
                  ? 'EXECUTABLE_NOT_FOUND_PERMANENT'
                  : `EXIT_${exitCode}`,
              }
            : {}),
        })
      })
    }).catch((err) => {
      if (isAbortError(err) || spec.signal?.aborted) {
        throw getAbortError(spec.signal, `Command ${spec.executable} aborted`)
      }
      throw err
    })
  }
}
