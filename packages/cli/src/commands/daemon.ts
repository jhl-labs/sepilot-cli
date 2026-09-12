import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import chalk from 'chalk'
import {
  resolveDaemonInvocation,
  launchDaemon,
  waitForDaemonReady,
  type DaemonInvocation,
} from '@sepilotd/api-client/node'
import { DaemonClient } from '../client/http.js'
import {
  LOCAL_DAEMON_START_ERROR,
  resolveDaemonDataDir,
  resolveLocalDaemonListenTarget,
} from '../client/token.js'
import { resolveStandaloneDaemonInvocation } from '../client/standalone-daemon.js'
import {
  persistDaemonForwardEnvNames,
  prepareDaemonEnvironment,
  type PreparedDaemonEnvironment,
} from '../client/daemon-forward-env.js'
import { detectCliLocale } from '../utils/locale.js'

const DAEMON_COPY = {
  en: {
    alreadyRunning: 'sepilotd is already running.',
    standaloneForegroundHint:
      'In the standalone build, run `sepilot __daemon` to run the daemon in the foreground.',
    daemonMissing: 'Cannot find sepilotd binary. Build or install @sepilotd/daemon first.',
    startingForeground: 'Starting sepilotd in foreground...',
    startingBackground: 'Starting sepilotd in background...',
    startedReady: (pid: number | undefined, version: string) =>
      `sepilotd started (PID ${pid}, v${version})`,
    startedNotReady: (pid: number | undefined) =>
      `sepilotd started (PID ${pid}) but not responding yet. Check logs.`,
    stopped: (pid: number) => `sepilotd stopped (PID ${pid})`,
    stoppedByRequest: 'sepilotd shutdown requested.',
    stillRunning: 'Process still running. Use kill -9 if needed.',
    noRunningDaemon: 'No running daemon found (no PID file).',
    unknown: 'unknown',
  },
  ko: {
    alreadyRunning: 'sepilotd가 이미 실행 중입니다.',
    standaloneForegroundHint:
      'standalone 빌드에서는 foreground 실행에 `sepilot __daemon`을 사용하세요.',
    daemonMissing:
      'sepilotd 바이너리를 찾을 수 없습니다. 먼저 @sepilotd/daemon을 빌드하거나 설치하세요.',
    startingForeground: 'sepilotd를 foreground에서 시작합니다...',
    startingBackground: 'sepilotd를 background에서 시작합니다...',
    startedReady: (pid: number | undefined, version: string) =>
      `sepilotd가 시작되었습니다 (PID ${pid}, v${version})`,
    startedNotReady: (pid: number | undefined) =>
      `sepilotd가 시작되었지만 (PID ${pid}) 아직 응답하지 않습니다. 로그를 확인하세요.`,
    stopped: (pid: number) => `sepilotd가 중지되었습니다 (PID ${pid})`,
    stoppedByRequest: 'sepilotd 종료를 요청했습니다.',
    stillRunning: '프로세스가 아직 실행 중입니다. 필요하면 kill -9를 사용하세요.',
    noRunningDaemon: '실행 중인 daemon을 찾을 수 없습니다 (PID 파일 없음).',
    unknown: '알 수 없음',
  },
} as const

const STOP_WAIT_ATTEMPTS = 100
const STOP_WAIT_INTERVAL_MS = 300

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function waitForProcessExit(pid: number): Promise<boolean> {
  for (let i = 0; i < STOP_WAIT_ATTEMPTS; i++) {
    await delay(STOP_WAIT_INTERVAL_MS)
    try {
      process.kill(pid, 0)
    } catch {
      return true
    }
  }
  return false
}

async function waitForDaemonUnreachable(client: DaemonClient): Promise<boolean> {
  for (let i = 0; i < STOP_WAIT_ATTEMPTS; i++) {
    try {
      await client.health()
    } catch {
      return true
    }
    await delay(STOP_WAIT_INTERVAL_MS)
  }
  return false
}

export interface DaemonStartOptions {
  url?: string
  foreground?: boolean
  forwardEnv?: string[]
  clearForwardEnv?: boolean
}

async function startCommandWithEnvironment(
  options: DaemonStartOptions,
  preparedEnvironment?: PreparedDaemonEnvironment,
): Promise<void> {
  const copy = DAEMON_COPY[detectCliLocale()] ?? DAEMON_COPY.en
  const client = new DaemonClient(options.url)

  // Check if already running
  try {
    await client.health()
    console.log(chalk.yellow(copy.alreadyRunning))
    return
  } catch {
    /* not running — good */
  }

  const listenTarget = resolveLocalDaemonListenTarget(options.url)
  if (!listenTarget) throw new Error(LOCAL_DAEMON_START_ERROR)

  // In the standalone single-file build there is no separate `sepilotd` entry;
  // re-enter ourselves with `__daemon`. Falls back to the module/PATH resolver
  // for the npm-installed CLI.
  const standalone = resolveStandaloneDaemonInvocation()
  if (options.foreground && standalone) {
    // `launchDaemon` returns a handle, not a promise that resolves on the
    // child's exit, so we can't truly "run in the foreground and wait" from
    // here without importing @sepilotd/daemon (forbidden by the architecture
    // boundary). Point the user at the dedicated subcommand instead.
    console.log(chalk.gray(copy.standaloneForegroundHint))
    return
  }
  const daemon: DaemonInvocation | null =
    (standalone && { ...standalone, kind: 'binary' as const }) ??
    resolveDaemonInvocation({
      moduleSearchRoots: [import.meta.dirname],
      execPath: process.execPath,
    })
  if (!daemon) {
    console.error(chalk.red(copy.daemonMissing))
    process.exit(1)
  }

  const dataDir = resolveDaemonDataDir()
  const listenOverrides = {
    SEPILOTD_HOST: listenTarget.host,
    SEPILOTD_PORT: String(listenTarget.port),
  }
  const prepared = preparedEnvironment
    ? {
        ...preparedEnvironment,
        env: { ...preparedEnvironment.env, ...listenOverrides },
      }
    : await prepareDaemonEnvironment({
      requestedForwardNames: options.forwardEnv,
      clearForwardEnv: options.clearForwardEnv,
      dataDir,
      overrides: listenOverrides,
    })

  if (options.foreground) {
    console.log(chalk.gray(copy.startingForeground))
    launchDaemon(daemon, {
      stdio: 'inherit',
      env: prepared.env,
      inheritEnv: false,
    })
    await persistDaemonForwardEnvNames(prepared.forwardNames, dataDir)
    return
  }

  // Background mode
  console.log(chalk.gray(copy.startingBackground))
  const proc = launchDaemon(daemon, {
    detached: true,
    env: prepared.env,
    inheritEnv: false,
  })
  await persistDaemonForwardEnvNames(prepared.forwardNames, dataDir)

  const healthUrl = `${listenTarget.baseUrl}/api/v1/health`
  const ready = await waitForDaemonReady(healthUrl, { timeoutMs: 5_000, isProcessAlive: proc.isRunning })

  if (ready) {
    const health = await client.health().catch(() => undefined)
    const version = health?.version ?? copy.unknown
    console.log(chalk.green(copy.startedReady(proc.pid, version)))
  } else {
    throw new Error(`${copy.startedNotReady(proc.pid)} Logs: ${join(dataDir, 'logs', 'daemon.log')}. Run sepilot __daemon in the same environment to inspect early startup errors.`)
  }
}

export async function startCommand(options: DaemonStartOptions): Promise<void> {
  // Commander appends its Command instance after declared positional
  // arguments. Keep that framework-owned callback context outside the
  // internal prepared-environment slot so only validated lifecycle state can
  // bypass prepareDaemonEnvironment().
  await startCommandWithEnvironment(options)
}

async function stopDaemon(_options: { url?: string }): Promise<boolean> {
  const copy = DAEMON_COPY[detectCliLocale()] ?? DAEMON_COPY.en
  const client = new DaemonClient(_options.url)
  // Honor SEPILOTD_DATA_DIR so `sepilot stop` finds the daemon that the
  // matching `sepilot start` / standalone-binary `__daemon` child actually
  // launched (which writes <dataDir>/sepilotd.pid). With the env unset this is
  // exactly ~/.sepilotd/sepilotd.pid as before.
  const dataDir = resolveDaemonDataDir()

  // Try graceful shutdown via PID file
  try {
    const pidStr = await readFile(join(dataDir, 'sepilotd.pid'), 'utf-8')
    const pid = parseInt(pidStr.trim(), 10)
    process.kill(pid, 'SIGTERM')
    console.log(chalk.green(copy.stopped(pid)))

    const stopped = await waitForProcessExit(pid)
    if (!stopped) console.log(chalk.yellow(copy.stillRunning))
    return stopped
  } catch {
    /* Fall back to the daemon's own shutdown endpoint below. */
  }

  try {
    await client.systemShutdown()
    console.log(chalk.green(copy.stoppedByRequest))
    const stopped = await waitForDaemonUnreachable(client)
    if (!stopped) console.log(chalk.yellow(copy.stillRunning))
    return stopped
  } catch {
    console.log(chalk.gray(copy.noRunningDaemon))
    return true
  }
}

export async function stopCommand(options: { url?: string }): Promise<void> {
  await stopDaemon(options)
}

export async function restartCommand(options: DaemonStartOptions) {
  // Validate both the local lifecycle target and named forwarding before
  // stopping the current daemon. Invalid reconfiguration must not become an
  // outage, and a remote/operator-owned target must never trigger local
  // process lifecycle actions.
  const listenTarget = resolveLocalDaemonListenTarget(options.url)
  if (!listenTarget) throw new Error(LOCAL_DAEMON_START_ERROR)
  const prepared = await prepareDaemonEnvironment({
    requestedForwardNames: options.forwardEnv,
    clearForwardEnv: options.clearForwardEnv,
    dataDir: resolveDaemonDataDir(),
    overrides: {
      SEPILOTD_HOST: listenTarget.host,
      SEPILOTD_PORT: String(listenTarget.port),
    },
  })
  const stopped = await stopDaemon(options)
  if (!stopped) {
    throw new Error('Cannot restart while the previous sepilotd process is still running.')
  }
  await new Promise((r) => setTimeout(r, 1000))
  await startCommandWithEnvironment(options, prepared)
}
