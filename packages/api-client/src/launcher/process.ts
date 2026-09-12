import { fork, spawn, type ChildProcess } from 'node:child_process'
import { dirname } from 'node:path'
import type { DaemonInvocation, DaemonLaunchOptions, DaemonProcess } from './types.js'

export function launchDaemon(
  invocation: DaemonInvocation,
  options: DaemonLaunchOptions = {},
): DaemonProcess {
  // Next.js globally augments NodeJS.ProcessEnv with a required NODE_ENV even
  // though Node's child-process API accepts an environment without that key.
  // A least-privilege launcher must not synthesize NODE_ENV (or any ambient
  // value) into an `inheritEnv: false` snapshot merely to satisfy that UI-only
  // type declaration. Keep runtime behavior exact and narrow only at the Node
  // process boundary.
  const childEnv = (options.inheritEnv === false
    ? { ...options.env }
    : { ...process.env, ...options.env }) as NodeJS.ProcessEnv
  let child: ChildProcess

  if (invocation.kind === 'module' && options.detached) {
    const spawnOpts: Parameters<typeof spawn>[2] = {
      cwd: dirname(invocation.args[0]),
      stdio: options.stdio ?? 'ignore',
      env: {
        ...childEnv,
        ELECTRON_RUN_AS_NODE: '1',
      },
      detached: true,
    }
    if (process.platform === 'win32') {
      spawnOpts.windowsHide = true
    }
    child = spawn(invocation.command, invocation.args, spawnOpts)
    child.unref()
  } else if (invocation.kind === 'module') {
    const forkOpts: Parameters<typeof fork>[2] = {
      cwd: dirname(invocation.args[0]),
      stdio: options.stdio ?? 'ignore',
      execArgv: [],
      env: {
        ...childEnv,
        ELECTRON_RUN_AS_NODE: '1',
      },
    }
    child = fork(invocation.args[0], [], {
      ...forkOpts,
    })
  } else {
    const spawnOpts: Parameters<typeof spawn>[2] = {
      stdio: options.stdio ?? 'ignore',
      env: childEnv,
    }
    if (options.detached) {
      spawnOpts.detached = true
    }
    if (process.platform === 'win32') {
      spawnOpts.windowsHide = true
    }
    child = spawn(invocation.command, invocation.args, spawnOpts)
    if (options.detached) {
      child.unref()
    }
  }

  return {
    pid: child.pid ?? -1,
    isRunning: () => child.exitCode === null && child.signalCode === null,
    stop() {
      try {
        child.kill(process.platform === 'win32' ? undefined : 'SIGTERM')
      } catch {
        // Process may already be gone.
      }
    },
  }
}

export async function waitForDaemonReady(
  healthUrl: string,
  options: { timeoutMs?: number; intervalMs?: number; isProcessAlive?: () => boolean } = {},
): Promise<boolean> {
  const intervalMs = options.intervalMs ?? 500
  const timeoutMs = options.timeoutMs ?? 10_000
  const maxAttempts = Math.ceil(timeoutMs / intervalMs)

  for (let i = 0; i < maxAttempts; i++) {
    if (options.isProcessAlive?.() === false) return false
    await new Promise((resolve) => setTimeout(resolve, intervalMs))
    if (options.isProcessAlive?.() === false) return false
    try {
      const response = await fetch(healthUrl, {
        signal: AbortSignal.timeout(2_000),
      })
      if (response.ok) return true
    } catch {
      // Not ready yet.
    }
  }

  return false
}
