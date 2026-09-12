import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import type { ISandbox, RunOptions, RunResult } from '@sepilotd/core'

const execAsync = promisify(execFile)

export interface SshSandboxConfig {
  host: string
  port?: number
  user?: string
  identityFile?: string
  connectTimeout?: number
}

export class SshSandbox implements ISandbox {
  private config: SshSandboxConfig

  constructor(config: SshSandboxConfig) {
    this.config = config
  }

  async run(command: string, options?: RunOptions): Promise<RunResult> {
    const start = Date.now()
    const sshArgs = this.buildSshArgs()
    sshArgs.push(command)

    try {
      const { stdout, stderr } = await execAsync('ssh', sshArgs, {
        timeout: options?.timeoutMs ?? 300000,
        env: options?.env ? { ...process.env, ...options.env } : undefined,
        maxBuffer: 10 * 1024 * 1024,
      })
      return { stdout, stderr, exitCode: 0, durationMs: Date.now() - start }
    } catch (err) {
      const execErr = err as { stdout?: string; stderr?: string; message?: string; code?: number }
      return {
        stdout: execErr.stdout ?? '',
        stderr: execErr.stderr ?? execErr.message ?? String(err),
        exitCode: execErr.code ?? 1,
        durationMs: Date.now() - start,
      }
    }
  }

  async writeFile(path: string, content: string): Promise<void> {
    const sshArgs = this.buildSshArgs()
    sshArgs.push(`cat > ${this.escapeShell(path)}`)

    await new Promise<void>((resolve, reject) => {
      const child = execFile('ssh', sshArgs, (err) => {
        if (err) reject(err)
        else resolve()
      })
      child.stdin!.write(content)
      child.stdin!.end()
    })
  }

  async readFile(path: string): Promise<string> {
    const result = await this.run(`cat ${this.escapeShell(path)}`)
    if (result.exitCode !== 0) throw new Error(result.stderr)
    return result.stdout
  }

  async cleanup(): Promise<void> {
    // Nothing to clean up for SSH
  }

  /** Test SSH connectivity */
  async testConnection(): Promise<boolean> {
    try {
      const result = await this.run('echo ok', { timeoutMs: 10000 })
      return result.stdout.trim() === 'ok'
    } catch {
      return false
    }
  }

  private buildSshArgs(): string[] {
    const args = ['-o', 'BatchMode=yes', '-o', `ConnectTimeout=${this.config.connectTimeout ?? 10}`]
    if (this.config.port) args.push('-p', String(this.config.port))
    if (this.config.identityFile) args.push('-i', this.config.identityFile)
    const userHost = this.config.user ? `${this.config.user}@${this.config.host}` : this.config.host
    args.push(userHost)
    return args
  }

  private escapeShell(str: string): string {
    return `'${str.replace(/'/g, "'\\''")}'`
  }
}
