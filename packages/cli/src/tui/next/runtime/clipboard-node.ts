import { spawn } from 'node:child_process'
import type { ClipboardDeps } from './clipboard.js'

export function createNodeClipboardDeps(
  stdout: Pick<NodeJS.WriteStream, 'write'>,
): ClipboardDeps {
  return {
    platform: process.platform,
    env: process.env,
    writeStdout: (data) => {
      stdout.write(data)
    },
    spawn: spawnClipboardCommand,
  }
}

export function spawnClipboardCommand(
  command: string,
  args: string[],
  input: string,
): Promise<boolean> {
  return new Promise((resolve) => {
    let settled = false
    const finish = (ok: boolean) => {
      if (settled) return
      settled = true
      resolve(ok)
    }

    let child: ReturnType<typeof spawn>
    try {
      child = spawn(command, args, { stdio: ['pipe', 'ignore', 'ignore'] })
    } catch {
      finish(false)
      return
    }

    child.once('error', () => finish(false))
    child.once('close', (code) => finish(code === 0))
    child.stdin?.once('error', () => finish(false))
    child.stdin?.end(input)
  })
}
