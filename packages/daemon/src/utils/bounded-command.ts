import { spawn } from 'node:child_process'

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
        try { process.kill(-child.pid, 'SIGKILL') } catch {}
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
    child.once('exit', killTree)
    child.once('close', (exitCode, signal) => {
      if (settled) return
      settled = true
      cleanup()
      resolve({ exitCode, signal, stdout: Buffer.concat(stdout).toString('utf8'), stderr: Buffer.concat(stderr).toString('utf8') })
    })
    child.stdin.on('error', () => {})
    child.stdin.end(input.stdin)
    input.signal?.addEventListener('abort', abort, { once: true })
    const timer = setTimeout(() => fail(new Error(`command timed out after ${input.timeoutMs}ms`)), input.timeoutMs)
    timer.unref?.()
    if (input.signal?.aborted) abort()
  })
}
