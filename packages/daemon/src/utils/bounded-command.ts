import { spawn } from 'node:child_process'

/** A short-lived owned process: neither descendants nor output may outlive its budget. */
export async function runBoundedCommand(input: {
  command: string
  args?: string[]
  shell?: boolean
  cwd?: string
  env?: NodeJS.ProcessEnv
  stdin?: string
  signal?: AbortSignal
  timeoutMs: number
  maxOutputBytes: number
}): Promise<{ exitCode: number | null; signal: NodeJS.Signals | null; stdout: string; stderr: string }> {
  input.signal?.throwIfAborted()
  return await new Promise((resolve, reject) => {
    const child = spawn(input.command, input.args ?? [], {
      cwd: input.cwd,
      env: input.env,
      shell: input.shell ?? false,
      detached: process.platform !== 'win32',
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    let settled = false
    let outputBytes = 0
    const stdout: Buffer[] = []
    const stderr: Buffer[] = []
    let terminationSent = false
    // Each launch owns a POSIX process group. Killing just a shell/ssh PID
    // abandons grandchildren (and can leave inherited stdout pipes open).
    const killTree = () => {
      if (!child.pid || terminationSent) return
      terminationSent = true
      if (process.platform === 'win32') {
        const killer = spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], {
          windowsHide: true, stdio: 'ignore',
        })
        killer.once('error', () => { child.kill('SIGKILL') })
        killer.unref()
      } else {
        try { process.kill(-child.pid, 'SIGKILL') } catch { /* already exited */ }
      }
    }
    const cleanup = () => {
      if (timer) clearTimeout(timer)
      input.signal?.removeEventListener('abort', abort)
    }
    const fail = (error: unknown) => {
      if (settled) return
      settled = true
      cleanup()
      killTree()
      reject(error)
    }
    const abort = () => fail(input.signal?.reason ?? new Error('command aborted'))
    const capture = (target: Buffer[], chunk: Buffer) => {
      if (settled) return
      outputBytes += chunk.length
      if (outputBytes > input.maxOutputBytes) {
        fail(new Error(`command output exceeded ${input.maxOutputBytes} bytes`))
        return
      }
      target.push(chunk)
    }
    child.stdout.on('data', (chunk: Buffer) => capture(stdout, chunk))
    child.stderr.on('data', (chunk: Buffer) => capture(stderr, chunk))
    child.once('error', fail)
    // Even a successful shell must not leave an unowned background child.
    child.once('exit', killTree)
    child.once('close', (exitCode, signal) => {
      if (settled) return
      settled = true
      cleanup()
      resolve({ exitCode, signal, stdout: Buffer.concat(stdout).toString('utf8'), stderr: Buffer.concat(stderr).toString('utf8') })
    })
    child.stdin.on('error', () => { /* early exit / EPIPE */ })
    child.stdin.end(input.stdin)
    input.signal?.addEventListener('abort', abort, { once: true })
    const timer = setTimeout(() => fail(new Error(`command timed out after ${input.timeoutMs}ms`)), input.timeoutMs)
    timer.unref?.()
    if (input.signal?.aborted) abort()
  })
}
