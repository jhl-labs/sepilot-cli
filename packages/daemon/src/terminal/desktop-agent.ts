import { sshAgentLaunch } from './ssh-launch.js'
import { randomUUID } from 'node:crypto'
import xterm from '@xterm/headless'
import { SerializeAddon } from '@xterm/addon-serialize'
import type {
  DesktopAgentEvent,
  DesktopAgentId,
  DesktopAgentSession,
  DesktopAgentSshTarget,
} from '@sepilotd/api-client'
import { spawnAgentPty, type AgentPty, type AgentPtyOptions } from './pty.js'

export const DESKTOP_AGENT_PRESETS = {
  claude: { command: 'claude' },
  codex: { command: 'codex' },
  gemini: { command: 'gemini' },
  opencode: { command: 'opencode' },
} satisfies Record<DesktopAgentId, { command: string }>
interface LiveSession {
  info: DesktopAgentSession
  pty?: AgentPty
  terminal: InstanceType<typeof xterm.Terminal>
  serializer: SerializeAddon
  seq: number
  tail: Promise<void>
  pendingBytes: number
  listeners: Set<(event: DesktopAgentEvent) => void>
}
/** sepilot-multi's persistent PTY + terminal grid model, with ordered snapshot/stream handoff. */
export class DesktopAgentSessions {
  private sessions = new Map<string, LiveSession>()
  constructor(private readonly spawn: (options: AgentPtyOptions) => AgentPty = spawnAgentPty) {}
  list(): DesktopAgentSession[] {
    return [...this.sessions.values()].map((live) => ({ ...live.info }))
  }
  private require(id: string): LiveSession {
    const live = this.sessions.get(id)
    if (!live)
      throw new Error(
        '터미널 세션을 찾지 못했습니다. daemon이 재시작되었다면 새 세션을 시작하세요.',
      )
    return live
  }
  async screen(id: string) {
    const live = this.require(id)
    await live.tail
    const buffer = live.terminal.buffer.active
    const lines: string[] = []
    for (let i = Math.max(0, buffer.length - 500); i < buffer.length; i++) {
      lines.push(buffer.getLine(i)?.translateToString(true) ?? '')
    }
    return { text: lines.join('\n'), cols: live.info.cols, rows: live.info.rows, seq: live.seq }
  }
  get(id: string): DesktopAgentSession {
    return { ...this.require(id).info }
  }
  private emit(live: LiveSession, event: Omit<DesktopAgentEvent, 'seq'>): void {
    const next = { ...event, seq: ++live.seq }
    for (const listener of live.listeners) {
      try {
        listener(next)
      } catch {
        live.listeners.delete(listener)
      }
    }
  }
  create(
    agent: DesktopAgentId,
    cwd: string,
    cols = 120,
    rows = 36,
    ssh?: DesktopAgentSshTarget,
  ): DesktopAgentSession {
    if (!Object.hasOwn(DESKTOP_AGENT_PRESETS, agent)) throw new Error('지원하지 않는 CLI입니다.')
    if (this.sessions.size >= 16)
      throw new Error('터미널 세션은 최대 16개입니다. 기존 세션을 종료하세요.')
    this.validateSize(cols, rows)
    const launch = ssh
      ? sshAgentLaunch(DESKTOP_AGENT_PRESETS[agent].command, cwd, ssh)
      : { command: DESKTOP_AGENT_PRESETS[agent].command, cwd }
    const terminal = new xterm.Terminal({ cols, rows, scrollback: 1000, allowProposedApi: true })
    const addon = new SerializeAddon()
    terminal.loadAddon(addon)
    const id = randomUUID()
    const live: LiveSession = {
      info: {
        id,
        ...(ssh ? { ssh: { ...ssh } } : {}),
        agent,
        cwd,
        command: DESKTOP_AGENT_PRESETS[agent].command,
        pid: 0,
        state: 'running',
        cols,
        rows,
        createdAt: new Date().toISOString(),
      },
      terminal,
      serializer: addon,
      seq: 0,
      tail: Promise.resolve(),
      pendingBytes: 0,
      listeners: new Set(),
    }
    this.sessions.set(id, live)
    terminal.onData((data) => {
      if (live.listeners.size === 0 && live.info.state === 'running') live.pty?.write(data)
    })
    try {
      live.pty = this.spawn({
        ...launch,
        cols,
        rows,
        onData: (data) => {
          if (!this.sessions.has(id)) return
          live.pendingBytes += data.length
          if (live.pendingBytes > 4 * 1024 * 1024) {
            live.pty?.kill()
            return
          }
          live.tail = live.tail.then(
            () =>
              new Promise<void>((resolve) => {
                terminal.write(data, () => {
                  live.pendingBytes -= data.length
                  this.emit(live, { type: 'output', data })
                  resolve()
                })
              }),
          )
        },
        onExit: (exitCode) => {
          if (!this.sessions.has(id)) return
          live.tail = live.tail.then(() => {
            live.info.state = 'exited'
            live.info.exitCode = exitCode
            this.emit(live, { type: 'exit', exitCode })
          })
        },
      })
      live.info.pid = live.pty.pid
      return this.get(id)
    } catch (error) {
      this.sessions.delete(id)
      terminal.dispose()
      throw error
    }
  }
  private validateSize(cols: number, rows: number): void {
    if (
      !Number.isInteger(cols) ||
      !Number.isInteger(rows) ||
      cols < 2 ||
      cols > 400 ||
      rows < 2 ||
      rows > 160
    )
      throw new Error('잘못된 터미널 크기입니다.')
  }
  write(id: string, data: string): void {
    const live = this.require(id)
    if (live.info.state !== 'running') throw new Error('CLI 프로세스가 종료되었습니다.')
    if (!data || Buffer.byteLength(data) > 65536)
      throw new Error('입력은 1~65536바이트여야 합니다.')
    live.pty!.write(data)
  }
  resize(id: string, cols: number, rows: number): Promise<void> {
    this.validateSize(cols, rows)
    const live = this.require(id)
    const resized = live.tail.then(() => {
      if (live.info.state !== 'running' || !this.sessions.has(id)) return
      live.pty!.resize(cols, rows)
      live.terminal.resize(cols, rows)
      live.info.cols = cols
      live.info.rows = rows
      this.emit(live, { type: 'resize', cols, rows })
    })
    live.tail = resized.catch(() => {})
    return resized
  }
  async subscribe(id: string, listener: (event: DesktopAgentEvent) => void): Promise<() => void> {
    const live = this.require(id)
    if (live.listeners.size >= 8) throw new Error('이 세션의 연결 수가 너무 많습니다.')
    // Snapshot and subscription share the same output queue; no missing or duplicated tail.
    const ready = live.tail.then(() => {
      if (!this.sessions.has(id)) throw new Error('세션이 종료되었습니다.')
      if (live.listeners.size >= 8) throw new Error('이 세션의 연결 수가 너무 많습니다.')
      listener({
        type: 'snapshot',
        seq: live.seq,
        data: '\x1bc' + live.serializer.serialize(),
        cols: live.info.cols,
        rows: live.info.rows,
      })
      if (live.info.state === 'exited')
        listener({ type: 'exit', seq: live.seq, exitCode: live.info.exitCode })
      live.listeners.add(listener)
    })
    live.tail = ready.catch(() => {})
    await ready
    return () => {
      live.listeners.delete(listener)
    }
  }
  close(id: string): void {
    const live = this.require(id)
    live.pty?.kill()
    this.sessions.delete(id)
    live.tail = live.tail.then(() => {
      this.emit(live, { type: 'exit', exitCode: live.info.exitCode ?? -1 })
      live.listeners.clear()
      live.terminal.dispose()
    })
  }
  closeAll(): void {
    for (const id of this.sessions.keys()) this.close(id)
  }
}
