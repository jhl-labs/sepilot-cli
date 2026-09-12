import { createRequire } from 'node:module'
import { StringDecoder } from 'node:string_decoder'
import { accessSync, constants, statSync } from 'node:fs'
import { delimiter, isAbsolute, join } from 'node:path'
import type * as NodePty from 'node-pty'

export interface AgentPty {
  pid: number
  write(data: string): void
  resize(cols: number, rows: number): void
  kill(): void
}
export interface AgentPtyOptions {
  command: string
  args?: string[]
  cwd: string
  cols: number
  rows: number
  onData(data: string): void
  onExit(exitCode: number): void
}

/** Keep CLI configuration, but never forward daemon control credentials or injected Node code. */
export function terminalEnvironment(source: NodeJS.ProcessEnv): Record<string, string> {
  return Object.fromEntries(
    Object.entries(source).filter(
      ([key, value]) =>
        value !== undefined && !/^(SEPILOT|SEPilot|NODE_OPTIONS$|NODE_PATH$)/i.test(key),
    ),
  ) as Record<string, string>
}

export function resolveAgentExecutable(command: string, env: Record<string, string>): string {
  const extensions = process.platform === 'win32' ? ['.exe', '.cmd', '.bat', '.com', ''] : ['']
  for (const directory of (env.PATH ?? env.Path ?? '').split(delimiter)) {
    // Do not accidentally execute a repository-local file from an empty/relative PATH entry.
    if (!isAbsolute(directory)) continue
    for (const extension of extensions) {
      const file = join(directory, command + extension)
      try {
        accessSync(file, constants.X_OK)
        if (statSync(file).isFile()) return file
      } catch {
        /* try next */
      }
    }
  }
  throw new Error(
    `${command} 실행 파일을 daemon의 PATH에서 찾지 못했습니다. 해당 호스트에 CLI를 설치하고 daemon을 다시 시작하세요.`,
  )
}

interface BunTerminal {
  write(data: string): void
  resize(cols: number, rows: number): void
  close(): void
}
interface BunProcess {
  pid: number
  exitCode: number | null
  kill(signal: string): void
}
interface BunPtyRuntime {
  Terminal: new (options: {
    cols: number
    rows: number
    name: string
    data(terminal: BunTerminal, bytes: Uint8Array): void
  }) => BunTerminal
  spawn(
    command: string[],
    options: {
      cwd: string
      env: Record<string, string>
      terminal: BunTerminal
      onExit(proc: BunProcess, code: number | null): void
    },
  ): BunProcess
}
const nativeRequire = createRequire(import.meta.url)
export function spawnAgentPty(options: AgentPtyOptions): AgentPty {
  const env = {
    ...terminalEnvironment(process.env),
    TERM: 'xterm-256color',
    COLORTERM: 'truecolor',
  }
  const executable = resolveAgentExecutable(options.command, env)
  // Standalone macOS/Linux binaries use Bun's built-in PTY, without native addon assets.
  const bun = (globalThis as unknown as { Bun?: BunPtyRuntime }).Bun
  if (bun?.Terminal && process.platform !== 'win32') {
    const decoder = new StringDecoder('utf8')
    const terminal = new bun.Terminal({
      cols: options.cols,
      rows: options.rows,
      name: 'xterm-256color',
      data: (_terminal, bytes) => {
        const data = decoder.write(Buffer.from(bytes))
        if (data) options.onData(data)
      },
    })
    let proc: ReturnType<typeof bun.spawn>
    let forceKill: ReturnType<typeof setTimeout> | undefined
    try {
      proc = bun.spawn([executable, ...(options.args ?? [])], {
        cwd: options.cwd,
        env,
        terminal,
        onExit: (_proc, code) => {
          if (forceKill) clearTimeout(forceKill)
          const tail = decoder.end()
          if (tail) options.onData(tail)
          terminal.close()
          options.onExit(code ?? 1)
        },
      })
    } catch (error) {
      terminal.close()
      throw error
    }
    return {
      pid: proc.pid,
      write: (data) => {
        terminal.write(data)
      },
      resize: (cols, rows) => terminal.resize(cols, rows),
      kill: () => {
        if (proc.exitCode !== null) return
        proc.kill('SIGTERM')
        forceKill ??= setTimeout(() => {
          proc.kill('SIGKILL')
          terminal.close()
        }, 3000)
        forceKill.unref()
      },
    }
  }
  const pty = nativeRequire('node-pty') as typeof NodePty
  // Windows npm shims require cmd.exe. Only a resolved allowlisted executable is passed.
  const shim = process.platform === 'win32' && /\.(cmd|bat)$/i.test(executable)
  const child = pty.spawn(
    shim ? (process.env.COMSPEC ?? 'cmd.exe') : executable,
    shim ? ['/d', '/s', '/c', `"${executable}"`] : (options.args ?? []),
    {
      name: 'xterm-256color',
      cwd: options.cwd,
      env,
      cols: options.cols,
      rows: options.rows,
    },
  )
  let exited = false
  let forceKill: ReturnType<typeof setTimeout> | undefined
  child.onData(options.onData)
  child.onExit(({ exitCode }) => {
    exited = true
    if (forceKill) clearTimeout(forceKill)
    options.onExit(exitCode)
  })
  return {
    pid: child.pid,
    write: (data) => child.write(data),
    resize: (cols, rows) => child.resize(cols, rows),
    kill: () => {
      if (exited) return
      child.kill()
      forceKill ??= setTimeout(() => {
        if (!exited) child.kill('SIGKILL')
      }, 3000)
      forceKill.unref()
    },
  }
}
