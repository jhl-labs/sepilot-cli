import { execFile, execFileSync } from 'node:child_process'
import { promisify } from 'node:util'
import chalk from 'chalk'
import type { SwarmAgentName } from '@sepilotd/api-client'
import { resolveDaemonInvocation } from '@sepilotd/api-client/node'
import { resolveStandaloneDaemonInvocation } from '../client/standalone-daemon.js'
import { detectCliLocale } from '../utils/locale.js'

const SWARM_AGENT_NAMES: readonly SwarmAgentName[] = ['claude', 'codex', 'gemini', 'opencode']

const execAsync = promisify(execFile)

const DOCTOR_COPY = {
  en: {
    daemonMissing: 'Cannot find sepilotd binary. Build or install @sepilotd/daemon first.',
    swarmHeader: 'Swarm Environment',
    tmuxInstalledSuffix: (path?: string) => `tmux installed${path ? ` (${path})` : ''}`,
    tmuxMissing: 'tmux not installed — swarm runs unavailable',
    agentAvailableSuffix: (name: string, path?: string) => `${name} agent available${path ? ` (${path})` : ''}`,
    agentMissingSuffix: (name: string) => `${name} agent not installed`,
  },
  ko: {
    daemonMissing: 'sepilotd 바이너리를 찾을 수 없습니다. 먼저 @sepilotd/daemon을 빌드하거나 설치하세요.',
    swarmHeader: 'Swarm 환경',
    tmuxInstalledSuffix: (path?: string) => `tmux 설치됨${path ? ` (${path})` : ''}`,
    tmuxMissing: 'tmux가 설치되지 않음 — swarm 실행을 사용할 수 없습니다',
    agentAvailableSuffix: (name: string, path?: string) => `${name} 에이전트 사용 가능${path ? ` (${path})` : ''}`,
    agentMissingSuffix: (name: string) => `${name} 에이전트가 설치되지 않음`,
  },
} as const

export async function doctorCommand() {
  const locale = detectCliLocale()
  const copy = DOCTOR_COPY[locale] ?? DOCTOR_COPY.en
  // The bundled CLI embeds the daemon and re-enters itself through __daemon;
  // it intentionally has no separate sepilotd binary on PATH. Keep doctor on
  // the same invocation resolver as start/init so a healthy standalone package
  // does not diagnose its own packaging layout as missing.
  const daemon = resolveStandaloneDaemonInvocation()
    ?? resolveDaemonInvocation({ moduleSearchRoots: [import.meta.dirname] })
  if (!daemon) {
    console.error(chalk.red(copy.daemonMissing))
    process.exit(1)
  }

  try {
    const { stdout, stderr } = await execAsync(daemon.command, [...daemon.args, 'doctor'])
    if (stdout) process.stdout.write(stdout)
    if (stderr) process.stderr.write(stderr)
  } catch (err) {
    const execErr = err as { stdout?: Buffer | string; stderr?: Buffer | string }
    if (execErr.stdout) process.stdout.write(execErr.stdout)
    if (execErr.stderr) process.stderr.write(execErr.stderr)
  }

  await printSwarmSection()
}

export interface SwarmCheckResult {
  tmux: { ok: boolean; path?: string }
  agents: Record<SwarmAgentName, { ok: boolean; path?: string }>
}

function defaultWhich(cmd: string): string | null {
  try {
    const out = execFileSync('which', [cmd], { encoding: 'utf-8' }).trim()
    return out || null
  } catch {
    return null
  }
}

export async function checkSwarmEnvironment(
  deps?: { which?: (cmd: string) => string | null },
): Promise<SwarmCheckResult> {
  const w = deps?.which ?? defaultWhich
  const tmuxPath = w('tmux')
  const result: SwarmCheckResult = {
    tmux: { ok: Boolean(tmuxPath), path: tmuxPath ?? undefined },
    agents: Object.fromEntries(
      SWARM_AGENT_NAMES.map((name) => [name, { ok: false }]),
    ) as SwarmCheckResult['agents'],
  }
  for (const a of SWARM_AGENT_NAMES) {
    const p = w(a)
    if (p) result.agents[a] = { ok: true, path: p }
  }
  return result
}

async function printSwarmSection(): Promise<void> {
  const locale = detectCliLocale()
  const copy = DOCTOR_COPY[locale] ?? DOCTOR_COPY.en
  const swarmCheck = await checkSwarmEnvironment()
  console.log(`\n${copy.swarmHeader}`)
  console.log('================================')
  const tmuxIcon = swarmCheck.tmux.ok ? '[PASS]' : '[WARN]'
  const tmuxMsg = swarmCheck.tmux.ok
    ? copy.tmuxInstalledSuffix(swarmCheck.tmux.path)
    : copy.tmuxMissing
  console.log(`${tmuxIcon} ${tmuxMsg}`)

  for (const name of SWARM_AGENT_NAMES) {
    const info = swarmCheck.agents[name]
    const icon = info.ok ? '[PASS]' : '[WARN]'
    const msg = info.ok
      ? copy.agentAvailableSuffix(name, info.path)
      : copy.agentMissingSuffix(name)
    console.log(`${icon} ${msg}`)
  }
}
