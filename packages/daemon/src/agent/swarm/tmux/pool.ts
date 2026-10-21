import { execFile, execFileSync } from 'node:child_process'
import { promisify } from 'node:util'
import { cleanAnsi, extractIncremental, extractIncrementalSince } from './capture.js'
import { resolveTmuxKeyName } from './keymap.js'

const run = promisify(execFile)
const INCREMENTAL_CAPTURE_LINES = 5000

export interface CreateOpts {
  name: string
  cwd: string
  env?: Record<string, string>
  paneWidth?: number
  paneHeight?: number
  command?: string[]
}

const NAME_ALLOWED = /^[A-Za-z0-9_-]+$/

function assertValidSessionName(name: string): void {
  if (!NAME_ALLOWED.test(name)) {
    throw new Error(
      `invalid tmux session name: ${JSON.stringify(name)} (allowed: [A-Za-z0-9_-])`,
    )
  }
}

export class TmuxSessionPool {
  private cursors = new Map<string, number>()
  private lastCaptures = new Map<string, string>()
  private locks = new Map<string, Promise<unknown>>()

  private async withLock<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const prev = this.locks.get(name) ?? Promise.resolve()
    let release!: () => void
    const next = new Promise<void>((r) => {
      release = r
    })
    this.locks.set(
      name,
      prev.then(() => next),
    )
    await prev
    try {
      return await fn()
    } finally {
      release()
    }
  }

  async create(opts: CreateOpts): Promise<string> {
    assertValidSessionName(opts.name)
    const name = opts.name
    const baseArgs = [
      'new-session',
      '-d',
      '-s',
      name,
      '-c',
      opts.cwd,
      '-x',
      String(opts.paneWidth ?? 200),
      '-y',
      String(opts.paneHeight ?? 50),
    ]
    const allArgs = opts.command?.length
      ? [...baseArgs, ...opts.command]
      : baseArgs
    await run('tmux', allArgs, { cwd: opts.cwd, env: { ...process.env, ...opts.env } })
    this.cursors.set(name, 0)
    this.lastCaptures.set(name, '')
    return name
  }

  async destroy(name: string): Promise<void> {
    try {
      await run('tmux', ['kill-session', '-t', name])
    } catch {
    }
    this.cursors.delete(name)
    this.lastCaptures.delete(name)
    this.locks.delete(name)
  }

  isAlive(name: string): boolean {
    try {
      execFileSync('tmux', ['has-session', '-t', name], { stdio: 'ignore' })
      return true
    } catch {
      return false
    }
  }

  async sendKeys(
    name: string,
    text: string,
    enter = true,
    options: { updateCursor?: boolean } = {},
  ): Promise<void> {
    return this.withLock(name, async () => {
      if (options.updateCursor ?? true) {
        const current = await this.captureRaw(name, INCREMENTAL_CAPTURE_LINES)
        const cleaned = cleanAnsi(current)
        this.cursors.set(name, cleaned.length)
        this.lastCaptures.set(name, cleaned)
      }
      if (text.length > 0) {
        await run('tmux', ['send-keys', '-t', name, '-l', '--', text])
      }
      if (enter) {
        await run('tmux', ['send-keys', '-t', name, 'Enter'])
      }
    })
  }

  async sendNamedKeys(name: string, keyName: string): Promise<void> {
    const resolved = resolveTmuxKeyName(keyName)
    if (!resolved) {
      throw new Error(`unsupported tmux key: ${keyName}`)
    }
    return this.withLock(name, async () => {
      await run('tmux', ['send-keys', '-t', name, resolved])
    })
  }

  async captureRaw(name: string, lines = 200): Promise<string> {
    const { stdout } = await run('tmux', [
      'capture-pane',
      '-t',
      name,
      '-p',
      '-e',
      '-S',
      `-${lines}`,
    ])
    return stdout
  }

  async captureClean(name: string, lines = 200): Promise<string> {
    return cleanAnsi(await this.captureRaw(name, lines))
  }

  async readNewOutput(name: string): Promise<string> {
    return this.withLock(name, async () => {
      const cleaned = await this.captureClean(name, INCREMENTAL_CAPTURE_LINES)
      const previous = this.lastCaptures.get(name)
      const slice = previous !== undefined
        ? extractIncrementalSince(previous, cleaned)
        : extractIncremental(cleaned, this.cursors.get(name) ?? 0)
      this.cursors.set(name, slice.nextCursor)
      this.lastCaptures.set(name, cleaned)
      return slice.text
    })
  }

  async resize(name: string, cols: number, rows: number): Promise<void> {
    await run('tmux', [
      'resize-window',
      '-t',
      name,
      '-x',
      String(cols),
      '-y',
      String(rows),
    ])
  }
}
