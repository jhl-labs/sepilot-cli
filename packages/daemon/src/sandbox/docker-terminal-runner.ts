import Docker from 'dockerode'
import { stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import {
  isAbsolute,
  join,
  parse,
  relative,
  resolve,
  sep,
} from 'node:path'
import type { ToolExecutionPosture } from '@sepilotd/core'
import { getAbortError, isAbortError, throwIfAborted } from '../abort.js'
import type {
  TerminalRunner,
  TerminalRunnerResult,
  TerminalRunSpec,
} from '../tools/terminal.js'

export interface DockerTerminalRunnerConfig {
  image?: string
  cpuLimit?: string
  memoryLimit?: string
  networkMode?: 'none' | 'bridge'
  mountMode?: 'rw' | 'ro'
  readOnlyRootfs?: boolean
  noNewPrivileges?: boolean
  capDrop?: string[]
  pidsLimit?: number
}

const DEFAULT_IMAGE = 'node:22-slim'
const DEFAULT_MEMORY_LIMIT = '512m'
const DEFAULT_CPU_LIMIT = '1.0'
const DEFAULT_PIDS_LIMIT = 100
const WORKSPACE_TARGET = '/workspace'

// Identifying label stamped on every sandbox container so a boot-time reaper can
// find and remove orphans left by a crashed daemon (bwrap avoids this with
// --die-with-parent; Docker containers outlive the daemon process).
export const SANDBOX_CONTAINER_LABEL = 'com.sepilot.sandbox'
export const SANDBOX_CONTAINER_LABEL_VALUE = '1'

/** Minimal dockerode surface the reaper needs (so it can be unit-tested). */
export interface DockerReapApi {
  listContainers(options: {
    all?: boolean
    filters?: { label?: string[] }
  }): Promise<Array<{ Id: string }>>
  getContainer(id: string): { remove(options: { force: boolean }): Promise<unknown> }
}

/**
 * Remove every container tagged with the sandbox label. Called at daemon boot to
 * reap orphans from a previous (possibly crashed) run. Best-effort: an
 * unreachable Docker daemon or a container that vanished mid-sweep is ignored.
 * Returns the number of containers removed.
 */
export async function reapOrphanedSandboxContainers(docker: DockerReapApi): Promise<number> {
  let containers: Array<{ Id: string }>
  try {
    containers = await docker.listContainers({
      all: true,
      filters: { label: [`${SANDBOX_CONTAINER_LABEL}=${SANDBOX_CONTAINER_LABEL_VALUE}`] },
    })
  } catch {
    return 0
  }
  let removed = 0
  for (const info of containers) {
    try {
      await docker.getContainer(info.Id).remove({ force: true })
      removed += 1
    } catch {
      /* already gone or not removable — skip */
    }
  }
  return removed
}

export function buildDockerTerminalExecutionPosture(input: {
  cwd: string
  boundary: ToolExecutionPosture['filesystem']['boundary']
  active: boolean
  networkMode: 'none' | 'bridge'
  readOnlyWorkspace?: boolean
  fallbackReason?: string
}): ToolExecutionPosture {
  return {
    sandbox: {
      requested: true,
      active: input.active,
      mode: input.active ? 'docker' : 'host',
      ...(input.fallbackReason ? { fallbackReason: input.fallbackReason } : {}),
    },
    filesystem: {
      cwd: input.cwd,
      boundary: input.boundary,
      isolated: input.active,
      readOnly: input.active && input.readOnlyWorkspace === true,
      note: input.active
        ? 'Docker sandbox bind-mounts the workspace at /workspace and does not mount host home, daemon state, credential stores, or the Docker socket.'
        : 'Docker sandbox was requested but did not become active.',
    },
    network: {
      isolated: input.active && input.networkMode === 'none',
      mode: input.active ? input.networkMode : 'host',
    },
  }
}

interface WorkspaceMount {
  source: string
  workingDir: string
}

export class DockerTerminalRunner implements TerminalRunner {
  private readonly docker: Docker
  private readonly config: Required<Omit<DockerTerminalRunnerConfig, 'capDrop'>> & {
    capDrop: string[]
  }

  constructor(config: DockerTerminalRunnerConfig = {}) {
    this.docker = new Docker()
    this.config = {
      image: config.image?.trim() || DEFAULT_IMAGE,
      cpuLimit: config.cpuLimit?.trim() || DEFAULT_CPU_LIMIT,
      memoryLimit: config.memoryLimit?.trim() || DEFAULT_MEMORY_LIMIT,
      networkMode: config.networkMode ?? 'none',
      mountMode: config.mountMode ?? 'rw',
      readOnlyRootfs: config.readOnlyRootfs ?? true,
      noNewPrivileges: config.noNewPrivileges ?? true,
      capDrop: config.capDrop?.length ? [...config.capDrop] : ['ALL'],
      pidsLimit: config.pidsLimit ?? DEFAULT_PIDS_LIMIT,
    }
  }

  /**
   * Reap sandbox containers orphaned by a previous daemon run. Best-effort;
   * intended to be called once at startup.
   */
  reapOrphans(): Promise<number> {
    return reapOrphanedSandboxContainers(this.docker as unknown as DockerReapApi)
  }

  async run(spec: TerminalRunSpec): Promise<TerminalRunnerResult> {
    const start = Date.now()
    const hostCwd = resolve(spec.cwd ?? process.cwd())
    const unavailablePosture = (reason: string) =>
      buildDockerTerminalExecutionPosture({
        cwd: hostCwd,
        boundary: spec.cwdBoundary,
        active: false,
        networkMode: this.config.networkMode,
        readOnlyWorkspace: this.config.mountMode === 'ro',
        fallbackReason: reason,
      })

    try {
      throwIfAborted(spec.signal, `Command ${spec.executable} aborted`)
      const mount = await resolveWorkspaceMount(hostCwd)
      const executionPosture = buildDockerTerminalExecutionPosture({
        cwd: mount.source,
        boundary: spec.cwdBoundary,
        active: true,
        networkMode: this.config.networkMode,
        readOnlyWorkspace: this.config.mountMode === 'ro',
      })
      const container = await this.docker.createContainer({
        Image: this.config.image,
        Cmd: [spec.executable, ...spec.args],
        WorkingDir: mount.workingDir,
        User: hostUser(),
        Env: [
          'CI=1',
          'HOME=/tmp',
          'SEPILOTD_SANDBOX=docker',
        ],
        AttachStdout: true,
        AttachStderr: true,
        Tty: false,
        Labels: { [SANDBOX_CONTAINER_LABEL]: SANDBOX_CONTAINER_LABEL_VALUE },
        HostConfig: {
          Memory: parseMemory(this.config.memoryLimit),
          NanoCpus: parseCpu(this.config.cpuLimit),
          NetworkMode: this.config.networkMode,
          ReadonlyRootfs: this.config.readOnlyRootfs,
          PidsLimit: this.config.pidsLimit,
          AutoRemove: false,
          Tmpfs: { '/tmp': 'rw,nosuid,nodev,size=256m' },
          CapDrop: this.config.capDrop,
          SecurityOpt: this.config.noNewPrivileges
            ? ['no-new-privileges:true']
            : undefined,
          Mounts: [
            {
              Type: 'bind',
              Source: mount.source,
              Target: WORKSPACE_TARGET,
              ReadOnly: this.config.mountMode === 'ro',
            },
          ],
        },
      })

      let abortListener: (() => void) | undefined
      try {
        await container.start()
        const waitPromise = container.wait()
        const timeoutPromise = new Promise<never>((_, reject) => {
          const timer = setTimeout(() => {
            reject(new SandboxTerminalRunError(
              `Docker sandbox command timed out after ${spec.timeoutMs}ms`,
              'TIMEOUT_TRANSIENT',
              executionPosture,
              true,
            ))
          }, spec.timeoutMs)
          waitPromise.finally(() => clearTimeout(timer)).catch(() => undefined)
        })
        const abortPromise = new Promise<never>((_, reject) => {
          if (!spec.signal) return
          abortListener = () => {
            reject(getAbortError(spec.signal, `Command ${spec.executable} aborted`))
          }
          spec.signal.addEventListener('abort', abortListener, { once: true })
        })
        const waitResult = await Promise.race([
          waitPromise,
          timeoutPromise,
          abortPromise,
        ])
        const logs = await container.logs({
          stdout: true,
          stderr: true,
          follow: false,
        })
        const output = demuxDockerLogs(Buffer.isBuffer(logs) ? logs : Buffer.from(logs))
        const exitCode = waitResult.StatusCode ?? 0
        return {
          stdout: output.stdout,
          stderr: output.stderr,
          status: exitCode === 0 ? 'success' : 'error',
          exitCode,
          code: exitCode === 0 ? undefined : 'EXIT_NONZERO_PERMANENT',
          durationMs: Date.now() - start,
          executionPosture,
        }
      } catch (error) {
        await container.stop().catch(() => undefined)
        if (isAbortError(error) || spec.signal?.aborted) {
          throw getAbortError(spec.signal, `Command ${spec.executable} aborted`)
        }
        if (error instanceof SandboxTerminalRunError) {
          return {
            stdout: '',
            stderr: error.message,
            status: 'error',
            code: error.code,
            durationMs: Date.now() - start,
            executionPosture: error.executionPosture,
          }
        }
        throw error
      } finally {
        if (abortListener && spec.signal) {
          spec.signal.removeEventListener('abort', abortListener)
        }
        await container.remove({ force: true }).catch(() => undefined)
      }
    } catch (error) {
      if (isAbortError(error) || spec.signal?.aborted) {
        throw getAbortError(spec.signal, `Command ${spec.executable} aborted`)
      }
      const message = error instanceof Error ? error.message : String(error)
      const code = classifyDockerError(message)
      return {
        stdout: '',
        stderr: message,
        status: 'error',
        code,
        durationMs: Date.now() - start,
        executionPosture: unavailablePosture(message),
      }
    }
  }
}

class SandboxTerminalRunError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly executionPosture: ToolExecutionPosture,
    readonly killed = false,
  ) {
    super(message)
    this.name = 'SandboxTerminalRunError'
  }
}

async function resolveWorkspaceMount(cwd: string): Promise<WorkspaceMount> {
  const source = await findGitWorkspaceRoot(cwd) ?? cwd
  const resolvedSource = resolve(source)
  if (isUnsafeWorkspaceMount(resolvedSource)) {
    throw new Error(`Refusing to bind-mount unsafe sandbox workspace: ${resolvedSource}`)
  }

  const rel = relative(resolvedSource, cwd)
  const workingDir = rel && !rel.startsWith('..') && !isAbsolute(rel)
    ? toContainerPath(join(WORKSPACE_TARGET, rel))
    : WORKSPACE_TARGET
  return { source: resolvedSource, workingDir }
}

async function findGitWorkspaceRoot(start: string): Promise<string | null> {
  let current = resolve(start)
  while (true) {
    if (await pathExists(join(current, '.git'))) {
      return current
    }
    const parent = resolve(current, '..')
    if (parent === current) {
      return null
    }
    current = parent
  }
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path)
    return true
  } catch {
    return false
  }
}

function isUnsafeWorkspaceMount(path: string): boolean {
  const root = parse(path).root
  return path === root || path === homedir()
}

function toContainerPath(path: string): string {
  return path.split(sep).join('/')
}

function hostUser(): string | undefined {
  const getuid = process.getuid
  const getgid = process.getgid
  return typeof getuid === 'function' && typeof getgid === 'function'
    ? `${getuid()}:${getgid()}`
    : undefined
}

function parseMemory(mem: string): number {
  const match = mem.match(/^(\d+)(m|g)$/i)
  if (!match) return 512 * 1024 * 1024
  const [, num, unit] = match
  return (
    parseInt(num!, 10)
    * (unit!.toLowerCase() === 'g' ? 1024 * 1024 * 1024 : 1024 * 1024)
  )
}

function parseCpu(cpu: string): number {
  const value = Number.parseFloat(cpu)
  return Number.isFinite(value) && value > 0
    ? Math.floor(value * 1e9)
    : 1_000_000_000
}

function classifyDockerError(message: string): string {
  if (/unsafe sandbox workspace/i.test(message)) {
    return 'SANDBOX_UNSAFE_WORKSPACE_PERMANENT'
  }
  return /no such image|pull access denied|not found/i.test(message)
    ? 'SANDBOX_IMAGE_UNAVAILABLE_PERMANENT'
    : 'SANDBOX_UNAVAILABLE_PERMANENT'
}

function demuxDockerLogs(buffer: Buffer): { stdout: string; stderr: string } {
  let offset = 0
  const stdout: Buffer[] = []
  const stderr: Buffer[] = []

  while (offset + 8 <= buffer.length) {
    const streamType = buffer[offset]
    const length = buffer.readUInt32BE(offset + 4)
    const start = offset + 8
    const end = start + length
    if (length < 0 || end > buffer.length) {
      break
    }
    const chunk = buffer.subarray(start, end)
    if (streamType === 2) {
      stderr.push(chunk)
    } else {
      stdout.push(chunk)
    }
    offset = end
  }

  if (offset === 0 || offset < buffer.length) {
    return { stdout: buffer.toString('utf-8'), stderr: '' }
  }

  return {
    stdout: Buffer.concat(stdout).toString('utf-8'),
    stderr: Buffer.concat(stderr).toString('utf-8'),
  }
}
