import {
  resolveDaemonInvocation,
  launchDaemon,
  tryAcquireSpawnLock,
  waitForDaemonReady,
} from '@sepilotd/api-client/node'
import { homedir } from 'node:os'
import { join } from 'node:path'
import chalk from 'chalk'
import type { DaemonClient } from './http.js'
import {
  LOCAL_DAEMON_START_ERROR,
  resolveLocalDaemonListenTarget,
} from './token.js'
import { startEmbeddedDaemonIfAvailable } from './embedded-daemon.js'
import { getStandaloneDaemon, resolveStandaloneDaemonInvocation } from './standalone-daemon.js'
import {
  persistDaemonForwardEnvNames,
  prepareDaemonEnvironment,
} from './daemon-forward-env.js'

const HEALTH_PROBE_TIMEOUT_MS = 2_000
const DEFAULT_DAEMON_READY_TIMEOUT_MS = 60_000
const MIN_DAEMON_READY_TIMEOUT_MS = 5_000
const MAX_DAEMON_READY_TIMEOUT_MS = 300_000

interface EnsureDaemonOptions {
  url?: string
  autoStart?: boolean
  quiet?: boolean
  preferEmbedded?: boolean
  signal?: AbortSignal
}

export interface EnsureDaemonResult {
  started: boolean
  pid?: number
  stop?: () => void
}

const existingDaemon: EnsureDaemonResult = { started: false }

function createAbortError(): Error {
  const error = new Error('Daemon startup aborted')
  error.name = 'AbortError'
  return error
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw createAbortError()
}

async function raceAbort<T>(promise: Promise<T>, signal: AbortSignal | undefined): Promise<T> {
  if (!signal) return promise
  throwIfAborted(signal)
  return Promise.race([
    promise,
    new Promise<T>((_, reject) => {
      const onAbort = () => reject(createAbortError())
      signal.addEventListener('abort', onAbort, { once: true })
      promise.finally(() => signal.removeEventListener('abort', onAbort)).catch(() => undefined)
    }),
  ])
}

export { buildDaemonChildEnv } from './daemon-forward-env.js'

export function resolveDaemonReadyTimeoutMs(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.SEPILOT_DAEMON_READY_MS?.trim()
  if (!raw) return DEFAULT_DAEMON_READY_TIMEOUT_MS
  const parsed = Number(raw)
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
    return DEFAULT_DAEMON_READY_TIMEOUT_MS
  }
  return Math.max(
    MIN_DAEMON_READY_TIMEOUT_MS,
    Math.min(MAX_DAEMON_READY_TIMEOUT_MS, parsed),
  )
}

async function waitForLaunchedDaemon(
  proc: ReturnType<typeof launchDaemon>,
  healthUrl: string,
  timeoutMs: number,
  signal: AbortSignal | undefined,
): Promise<void> {
  let ready: boolean
  try {
    ready = await raceAbort(
      waitForDaemonReady(healthUrl, { timeoutMs, isProcessAlive: proc.isRunning }),
      signal,
    )
  } catch (error) {
    proc.stop()
    throw error
  }
  if (ready) return
  proc.stop()
  throw new Error(
    `sepilotd was launched but did not become ready${proc.isRunning?.() === false ? ' because the child exited' : ` within ${timeoutMs / 1000}s`}. `
      + `Check ${join(process.env.SEPILOTD_DATA_DIR?.trim() || join(homedir(), '.sepilotd'), 'logs', 'daemon.log')}. `
      + 'Run sepilot __daemon in the same environment to inspect early startup errors.',
  )
}

export async function ensureDaemon(
  client: DaemonClient,
  options: EnsureDaemonOptions = {},
): Promise<EnsureDaemonResult> {
  throwIfAborted(options.signal)
  try {
    await raceAbort(
      client.health({ signal: AbortSignal.timeout(HEALTH_PROBE_TIMEOUT_MS) }),
      options.signal,
    )
    return existingDaemon
  } catch (error) {
    if (isAbortError(error)) throw error
  }
  throwIfAborted(options.signal)

  const envDisable = process.env.SEPILOT_NO_AUTO_START === '1'
  const autoStart = options.autoStart !== false && !envDisable
  if (!autoStart) {
    throw new Error(
      'Cannot connect to sepilotd. Start it with: sepilot daemon start',
    )
  }

  const listenTarget = resolveLocalDaemonListenTarget(options.url)
  if (!listenTarget) throw new Error(LOCAL_DAEMON_START_ERROR)

  const dataDir = process.env.SEPILOTD_DATA_DIR?.trim() || join(homedir(), '.sepilotd')
  const lock = await raceAbort(
    tryAcquireSpawnLock({ lockPath: join(dataDir, '.spawn.lock') }),
    options.signal,
  )
  const healthUrl = `${listenTarget.baseUrl}/api/v1/health`
  const readyTimeoutMs = resolveDaemonReadyTimeoutMs()

  if (!lock) {
    const ready = await raceAbort(
      waitForDaemonReady(healthUrl, { timeoutMs: readyTimeoutMs }),
      options.signal,
    )
    if (!ready) {
      throw new Error(
        'Another sepilotd spawn is in progress but the daemon never became ready.',
      )
    }
    return existingDaemon
  }

  try {
    throwIfAborted(options.signal)
    try {
      await raceAbort(
        client.health({ signal: AbortSignal.timeout(HEALTH_PROBE_TIMEOUT_MS) }),
        options.signal,
      )
      return existingDaemon
    } catch (error) {
      if (isAbortError(error)) throw error
    }

    const preferEmbedded = options.preferEmbedded === true
      || process.env.SEPILOTD_CLI_EMBEDDED_DAEMON === '1'
    if (preferEmbedded && process.env.SEPILOTD_NO_EMBEDDED_DAEMON !== '1') {
      if (!options.quiet) {
        process.stderr.write(chalk.gray('Starting embedded sepilotd...\n'))
      }
      const factory = getStandaloneDaemon()?.embeddedFactory
      if (factory) {
        await raceAbort(
          factory({
            dataDir,
            host: listenTarget.host,
            port: listenTarget.port,
            autoApproveCliFlag: process.argv.includes('--yes-to-everything'),
          }),
          options.signal,
        )
        return existingDaemon
      }
      const embeddedStarted = await raceAbort(
        startEmbeddedDaemonIfAvailable({ baseUrl: listenTarget.baseUrl, dataDir }),
        options.signal,
      )
      if (embeddedStarted) return existingDaemon
    }

    const standaloneInvocation = resolveStandaloneDaemonInvocation()
    if (standaloneInvocation) {
      const childEnvOverrides: Record<string, string> = {
        SEPILOTD_HOST: listenTarget.host,
        SEPILOTD_PORT: String(listenTarget.port),
      }
      if (process.env.SEPILOTD_DATA_DIR?.trim()) {
        childEnvOverrides.SEPILOTD_DATA_DIR = process.env.SEPILOTD_DATA_DIR.trim()
      }
      const prepared = await prepareDaemonEnvironment({
        dataDir,
        env: process.env,
        overrides: childEnvOverrides,
      })
      if (!options.quiet) {
        process.stderr.write(chalk.gray('Starting sepilotd in background...\n'))
      }
      const proc = launchDaemon(
        { ...standaloneInvocation, kind: 'binary' },
        { detached: true, stdio: 'ignore', env: prepared.env, inheritEnv: false },
      )
      await waitForLaunchedDaemon(proc, healthUrl, readyTimeoutMs, options.signal)
      await persistDaemonForwardEnvNames(prepared.forwardNames, dataDir)
      return { started: true, pid: proc.pid, stop: proc.stop }
    }

    const invocation = resolveDaemonInvocation({
      moduleSearchRoots: [import.meta.dirname],
      execPath: process.execPath,
    })
    if (!invocation) {
      throw new Error(
        'Cannot find the sepilotd daemon binary. Build or install @sepilotd/daemon, '
          + 'or set SEPILOTD_BIN to point at a known sepilotd entry.',
      )
    }

    if (!options.quiet) {
      process.stderr.write(chalk.gray('Starting sepilotd in background...\n'))
    }

    const prepared = await prepareDaemonEnvironment({
      dataDir,
      env: process.env,
      overrides: {
        SEPILOTD_HOST: listenTarget.host,
        SEPILOTD_PORT: String(listenTarget.port),
      },
    })
    const proc = launchDaemon(invocation, {
      detached: true,
      stdio: 'ignore',
      env: prepared.env,
      inheritEnv: false,
    })

    await waitForLaunchedDaemon(proc, healthUrl, readyTimeoutMs, options.signal)
    await persistDaemonForwardEnvNames(prepared.forwardNames, dataDir)
    return { started: true, pid: proc.pid, stop: proc.stop }
  } finally {
    await lock.release()
  }
}
