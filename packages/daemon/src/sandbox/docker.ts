import Docker from 'dockerode'
import { mkdir } from 'node:fs/promises'
import { basename, dirname, resolve } from 'node:path'
import type { ISandbox, RunOptions, RunResult } from '@sepilotd/core'

export interface DockerSandboxConfig {
  image?: string
  cpuLimit?: string // e.g. "1.0"
  memoryLimit?: string // e.g. "512m"
  networkMode?: 'none' | 'bridge' | 'host'
  readOnlyRootfs?: boolean
  pidsLimit?: number
}

export class DockerSandbox implements ISandbox {
  private docker: Docker
  private config: DockerSandboxConfig
  private containerId: string | null = null

  constructor(config?: DockerSandboxConfig) {
    this.docker = new Docker()
    this.config = config ?? {}
    // `networkMode: 'host'` shares the host network namespace with the
    // container, which means a workload run inside the "sandbox" can still
    // hit `127.0.0.1:17600` (the daemon itself) and any other loopback
    // service. That defeats the network-isolation purpose of the sandbox
    // — refuse it outright unless the operator explicitly acks it. The
    // legitimate cases are 'none' (default) and 'bridge'.
    if (this.config.networkMode === 'host') {
      const ack = process.env.SEPILOTD_SANDBOX_ALLOW_HOST_NETWORK
      if (ack !== 'YES_I_UNDERSTAND') {
        throw new Error(
          'DockerSandbox networkMode "host" shares the host network namespace and '
          + 'defeats sandbox isolation. Use "none" (default) or "bridge", or set '
          + 'SEPILOTD_SANDBOX_ALLOW_HOST_NETWORK=YES_I_UNDERSTAND to opt in.',
        )
      }
    }
  }

  async run(command: string, options?: RunOptions): Promise<RunResult> {
    const start = Date.now()
    const image = this.config.image ?? 'node:22-slim'
    const timeoutMs = options?.timeoutMs ?? 300000

    try {
      // Create container
      const container = await this.docker.createContainer({
        Image: image,
        Cmd: ['sh', '-c', command],
        WorkingDir: options?.cwd ?? '/workspace',
        Env: options?.env
          ? Object.entries(options.env).map(([k, v]) => `${k}=${v}`)
          : undefined,
        HostConfig: {
          Memory: this.parseMemory(this.config.memoryLimit ?? '512m'),
          NanoCpus: this.parseCpu(this.config.cpuLimit ?? '1.0'),
          NetworkMode: this.config.networkMode ?? 'none',
          ReadonlyRootfs: this.config.readOnlyRootfs ?? true,
          PidsLimit: this.config.pidsLimit ?? 100,
          AutoRemove: true,
          Tmpfs: { '/tmp': 'rw,noexec,nosuid,size=100m' },
        },
      })

      this.containerId = container.id

      // Start and wait
      await container.start()

      const waitPromise = container.wait()
      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('Container timeout')), timeoutMs),
      )

      const { StatusCode } = await Promise.race([waitPromise, timeoutPromise])

      // Get logs
      const logs = await container.logs({
        stdout: true,
        stderr: true,
        follow: false,
      })
      const logStr = logs.toString('utf-8')

      this.containerId = null
      return {
        stdout: logStr,
        stderr: '',
        exitCode: StatusCode,
        durationMs: Date.now() - start,
      }
    } catch (err: unknown) {
      // Cleanup on error
      if (this.containerId) {
        try {
          const c = this.docker.getContainer(this.containerId)
          await c.stop().catch(() => {})
          await c.remove().catch(() => {})
        } catch {
          // ignore cleanup errors
        }
        this.containerId = null
      }
      const message = err instanceof Error ? err.message : String(err)
      return {
        stdout: '',
        stderr: message,
        exitCode: 1,
        durationMs: Date.now() - start,
      }
    }
  }

  async writeFile(path: string, content: string): Promise<void> {
    const resolvedPath = resolve(path)
    const parentDir = dirname(resolvedPath)
    await mkdir(parentDir, { recursive: true })
    const result = await this.runEphemeralCommand({
      command: `cat > ${this.escapeShell(basename(resolvedPath))}`,
      cwd: parentDir,
      binds: [`${parentDir}:${parentDir}:rw`],
      stdin: content,
    })
    if (result.exitCode !== 0) {
      throw new Error(result.logs || `Failed to write ${resolvedPath}`)
    }
  }

  async readFile(path: string): Promise<string> {
    const resolvedPath = resolve(path)
    const parentDir = dirname(resolvedPath)
    const result = await this.runEphemeralCommand({
      command: `cat ${this.escapeShell(basename(resolvedPath))}`,
      cwd: parentDir,
      binds: [`${parentDir}:${parentDir}:ro`],
    })
    if (result.exitCode !== 0) {
      throw new Error(result.logs || `Failed to read ${resolvedPath}`)
    }
    return result.logs
  }

  async cleanup(): Promise<void> {
    if (this.containerId) {
      try {
        const container = this.docker.getContainer(this.containerId)
        await container.stop().catch(() => {})
        await container.remove().catch(() => {})
      } catch {
        // ignore cleanup errors
      }
      this.containerId = null
    }
  }

  /** Check if Docker is available */
  static async isAvailable(): Promise<boolean> {
    try {
      const docker = new Docker()
      await docker.ping()
      return true
    } catch {
      return false
    }
  }

  private parseMemory(mem: string): number {
    const match = mem.match(/^(\d+)(m|g)$/i)
    if (!match) return 512 * 1024 * 1024
    const [, num, unit] = match
    return (
      parseInt(num!) *
      (unit!.toLowerCase() === 'g' ? 1024 * 1024 * 1024 : 1024 * 1024)
    )
  }

  private parseCpu(cpu: string): number {
    return Math.floor(parseFloat(cpu) * 1e9)
  }

  private async runEphemeralCommand(params: {
    command: string
    cwd: string
    binds?: string[]
    stdin?: string
  }): Promise<{ exitCode: number; logs: string }> {
    const image = this.config.image ?? 'node:22-slim'
    const container = await this.docker.createContainer({
      Image: image,
      Cmd: ['sh', '-c', params.command],
      WorkingDir: params.cwd,
      AttachStdout: true,
      AttachStderr: true,
      AttachStdin: params.stdin !== undefined,
      OpenStdin: params.stdin !== undefined,
      StdinOnce: params.stdin !== undefined,
      Tty: true,
      HostConfig: {
        Memory: this.parseMemory(this.config.memoryLimit ?? '512m'),
        NanoCpus: this.parseCpu(this.config.cpuLimit ?? '1.0'),
        NetworkMode: this.config.networkMode ?? 'none',
        ReadonlyRootfs: this.config.readOnlyRootfs ?? true,
        PidsLimit: this.config.pidsLimit ?? 100,
        AutoRemove: false,
        Tmpfs: { '/tmp': 'rw,noexec,nosuid,size=100m' },
        Binds: params.binds,
      },
    })

    try {
      let ioStream:
        | { end(chunk?: string): void }
        | undefined
      if (params.stdin !== undefined) {
        ioStream = await container.attach({
          stream: true,
          stdin: true,
          stdout: true,
          stderr: true,
        }) as unknown as { end(chunk?: string): void }
      }

      await container.start()
      if (ioStream) {
        ioStream.end(params.stdin)
      }
      const waitResult = await container.wait()
      const logs = await container.logs({
        stdout: true,
        stderr: true,
        follow: false,
      })
      return {
        exitCode: waitResult.StatusCode ?? 0,
        logs: logs.toString('utf-8'),
      }
    } finally {
      await container.remove({ force: true }).catch(() => {})
    }
  }

  private escapeShell(value: string): string {
    return `'${value.replace(/'/g, `'\\''`)}'`
  }
}
