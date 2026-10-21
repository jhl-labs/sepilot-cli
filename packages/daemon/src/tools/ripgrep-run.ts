import { spawn } from 'node:child_process'

const MAX_STDERR_CHARS = 8_192

export interface RipgrepCaptureResult {
  stdout: string
  truncated: boolean
  code: number | null
}

export function runRipgrepCapture(options: {
  args: string[]
  cwd: string
  maxBytes: number
  signal?: AbortSignal
}): Promise<RipgrepCaptureResult> {
  return new Promise((resolve, reject) => {
    const child = spawn('rg', options.args, { cwd: options.cwd, signal: options.signal })
    const chunks: Buffer[] = []
    let captured = 0
    let truncated = false
    let stderr = ''
    let settled = false

    child.stdout.on('data', (chunk: Buffer) => {
      if (truncated) return
      const remaining = options.maxBytes - captured
      if (chunk.byteLength <= remaining) {
        chunks.push(chunk)
        captured += chunk.byteLength
        return
      }
      if (remaining > 0) {
        chunks.push(chunk.subarray(0, remaining))
        captured += remaining
      }
      truncated = true
      child.kill('SIGTERM')
    })

    child.stderr.on('data', (chunk: Buffer) => {
      if (stderr.length >= MAX_STDERR_CHARS) return
      stderr = (stderr + chunk.toString('utf8')).slice(0, MAX_STDERR_CHARS)
    })

    child.on('error', (error) => {
      if (settled) return
      settled = true
      reject(error)
    })

    child.on('close', (code, termSignal) => {
      if (settled) return
      settled = true
      const stdout = Buffer.concat(chunks, captured).toString('utf8')
      if (!truncated && code !== 0 && code !== 1 && code !== null) {
        reject(new Error(
          stderr.trim()
          || `rg exited with code ${code}${termSignal ? ` signal ${termSignal}` : ''}`,
        ))
        return
      }
      resolve({ stdout, truncated, code: code ?? null })
    })
  })
}
