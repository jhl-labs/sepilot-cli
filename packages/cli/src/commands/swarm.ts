import { StringDecoder } from 'node:string_decoder'
import chalk from 'chalk'
import {
  createSwarmClient,
  type CreateRunInput,
  type DriveAgentResult,
  type SwarmAgentName,
  type SwarmClient,
  type SwarmEvent,
  type SwarmRun,
  type SwarmRunStatus,
} from '@sepilotd/api-client'
import { loadDaemonToken, resolveDaemonBaseUrl } from '../client/token.js'

const VALID_AGENTS: readonly SwarmAgentName[] = ['claude', 'codex', 'gemini', 'opencode']
type SwarmAgentView = SwarmRun['agents'][number]

export interface SwarmRunOptions {
  agent?: string
  cwd?: string
  worktree?: string
  autonomy?: 'readonly' | 'accept-edits' | 'workspace-write' | 'supervised' | 'autonomous'
  autoApproveAgents?: boolean
  noSupervisor?: boolean
  supervisor?: boolean
  detach?: boolean
  attach?: boolean
  url?: string
}

export interface RunSwarmStartDeps {
  client: SwarmClient
  goal: string
  options: SwarmRunOptions
  printer?: (line: string) => void
}

function parseWarmPool(raw: string | undefined): SwarmAgentName[] {
  if (!raw) return ['claude']
  const parts = raw.split(',').map((s) => s.trim()).filter(Boolean)
  if (parts.length === 0) return ['claude']
  for (const p of parts) {
    if (!VALID_AGENTS.includes(p as SwarmAgentName)) {
      throw new Error(
        `unknown agent: ${p}; valid options: ${VALID_AGENTS.join(', ')}`,
      )
    }
  }
  return parts as SwarmAgentName[]
}

/**
 * Pure async core of `sepilot swarm run`. Tests inject a stub `client`
 * and `printer` so we can assert wire shape without standing up the
 * daemon transport.
 */
export async function runSwarmStart(
  deps: RunSwarmStartDeps,
): Promise<{ runId: string }> {
  const print = deps.printer ?? ((s: string) => console.log(s))
  const input: CreateRunInput = {
    goal: deps.goal,
    cwd: deps.options.cwd,
    worktree: deps.options.worktree,
    autonomy: deps.options.autonomy,
    autoApproveAgents: deps.options.autoApproveAgents ?? true,
    noSupervisor: deps.options.noSupervisor
      ?? (deps.options.supervisor === false ? true : undefined),
    warmPool: parseWarmPool(deps.options.agent),
  }
  const { runId } = await deps.client.createRun(input)
  print(`${chalk.green('swarm started')} ${runId}`)
  return { runId }
}

function buildClient(url?: string): SwarmClient {
  const baseUrl = resolveDaemonBaseUrl(url) ?? 'http://127.0.0.1:17600'
  const token = loadDaemonToken()
  return createSwarmClient({ baseUrl, token })
}

function isStartupBlocker(state: string | undefined): boolean {
  return state === 'trust_required' || state === 'tool_permission_required' || state === 'failed'
}

function formatAgentStartupState(agent: SwarmAgentView): string {
  const state = agent.startupEvidence?.lifecycleState
  if (!state) return ''
  const runtime = agent.runtime ?? agent.startupEvidence?.runtime
  const text = runtime ? `${runtime}/${state}` : state
  return isStartupBlocker(state) ? chalk.yellow(text) : chalk.gray(text)
}

function formatBlockedAgents(agents: SwarmAgentView[]): string | null {
  const blocked = agents.filter((agent) => isStartupBlocker(agent.startupEvidence?.lifecycleState))
  if (!blocked.length) return null
  return `blocked: ${blocked.map((agent) =>
    `${agent.handle}:${agent.startupEvidence?.lifecycleState}${
      agent.startupEvidence?.recoveryHint?.scenario
        ? `(${agent.startupEvidence.recoveryHint.scenario})`
        : ''
    }`).join(', ')}`
}

function truncate(text: string, max = 140): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`
}

export async function swarmRunCommand(
  goal: string,
  options: SwarmRunOptions,
): Promise<void> {
  const client = buildClient(options.url)
  const { runId } = await runSwarmStart({ client, goal, options })
  if (options.attach) {
    await swarmAttachCommand(runId, { url: options.url })
  }
}

function isInteractiveTty(): boolean {
  return Boolean(process.stdin.isTTY && process.stdout.isTTY)
}

export interface SwarmAttachInputState {
  prefixed: boolean
  decoder: StringDecoder
}

export type SwarmAttachInputAction =
  | { type: 'send'; keys: string; enter: boolean }
  | { type: 'detach' }
  | { type: 'cancel' }
  | { type: 'list-agents' }

export function createSwarmAttachInputState(): SwarmAttachInputState {
  return { prefixed: false, decoder: new StringDecoder('utf8') }
}

export function parseSwarmAttachInputChunk(
  chunk: Buffer,
  state: SwarmAttachInputState,
): SwarmAttachInputAction[] {
  const actions: SwarmAttachInputAction[] = []
  let literal: number[] = []
  const flushLiteral = (): void => {
    if (literal.length === 0) return
    const keys = state.decoder.write(Buffer.from(literal))
    literal = []
    if (keys) actions.push({ type: 'send', keys, enter: false })
  }

  for (const byte of chunk) {
    if (state.prefixed) {
      state.prefixed = false
      if (byte === 0x64 || byte === 0x44) {
        flushLiteral()
        actions.push({ type: 'detach' })
        continue
      }
      if (byte === 0x6b || byte === 0x4b) {
        flushLiteral()
        actions.push({ type: 'cancel' })
        continue
      }
      if (byte === 0x6c || byte === 0x4c) {
        flushLiteral()
        actions.push({ type: 'list-agents' })
        continue
      }
      literal.push(0x02, byte)
      continue
    }
    if (byte === 0x02) {
      flushLiteral()
      state.prefixed = true
      continue
    }
    if (byte === 0x0d || byte === 0x0a) {
      flushLiteral()
      actions.push({ type: 'send', keys: '', enter: true })
      continue
    }
    literal.push(byte)
  }
  flushLiteral()
  return actions
}

export interface SwarmAttachOptions {
  url?: string
  agent?: string
}

/**
 * `sepilot swarm attach <runId>` — raw tmux mirror + key passthrough.
 *
 * Renders the active (or `--agent`-selected) pane by capturing the tmux
 * session and writing diffs to stdout. In an interactive TTY, stdin is put
 * in raw mode and every byte is forwarded to the daemon via sendKeys.
 *
 * Escape prefix is Ctrl-B (0x02), inspired by tmux. Recognised chords:
 *   - Ctrl-B d → detach (exit 0)
 *   - Ctrl-B k → cancel the run, then exit 0
 *   - Ctrl-B l → print agent list to stderr
 *
 * The SSE stream is still consumed so run-ended and pane.snapshot events
 * terminate or refresh the mirror promptly between poll ticks.
 */
export async function swarmAttachCommand(
  runId: string,
  options: SwarmAttachOptions,
): Promise<void> {
  const client = buildClient(options.url)
  const run = await client.get(runId)
  let handleId = options.agent
  if (!handleId) handleId = run.activeHandle
  if (!handleId) {
    console.error(chalk.red('no active handle; pass --agent <handle>'))
    return
  }
  const handle = run.agents.find((a) => a.handle === handleId || a.role === handleId)
  if (!handle) {
    console.error(chalk.red(`unknown handle: ${handleId}`))
    return
  }
  const targetHandle = handle.handle

  const readonly = !isInteractiveTty()
  let leaseHeld = false
  let renewTimer: ReturnType<typeof setTimeout> | undefined
  const clearLeaseRenewal = (): void => {
    if (!renewTimer) return
    clearTimeout(renewTimer)
    renewTimer = undefined
  }
  const scheduleLeaseRenewal = (expiresAt: number | undefined): void => {
    clearLeaseRenewal()
    if (!leaseHeld) return
    const delay = expiresAt
      ? Math.max(1000, Math.min(60_000, expiresAt - Date.now() - 10_000))
      : 60_000
    renewTimer = setTimeout(() => {
      void client
        .attachAgent(runId, targetHandle, { renew: true })
        .then((res) => scheduleLeaseRenewal(res.lease?.expiresAt))
        .catch(() => scheduleLeaseRenewal(Date.now() + 20_000))
    }, delay)
    renewTimer.unref?.()
  }
  const releaseInteractiveLease = async (): Promise<void> => {
    if (!leaseHeld) return
    leaseHeld = false
    clearLeaseRenewal()
    await client.detachAgent(runId, targetHandle).catch(() => undefined)
  }

  if (!readonly) {
    const attached = await client.attachAgent(runId, targetHandle)
    leaseHeld = true
    scheduleLeaseRenewal(attached.lease?.expiresAt)
    process.stdin.setRawMode?.(true)
    process.stdin.resume()
  }

  const ac = new AbortController()
  let pollTimer: ReturnType<typeof setInterval> | undefined
  let stopped = false
  const stopRendering = (releaseLease = true): void => {
    if (stopped) return
    stopped = true
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = undefined
    }
    clearLeaseRenewal()
    ac.abort()
    if (!readonly) {
      process.stdin.setRawMode?.(false)
      process.stdin.pause()
    }
    if (releaseLease) void releaseInteractiveLease()
  }
  process.on('SIGINT', stopRendering)

  let lastSnapshot = ''
  const captureLines = (): number => Math.max(1, Math.min(5000, process.stdout.rows ?? 200))
  const renderSnapshot = (text: string): void => {
    if (text === lastSnapshot) return
    const diff = text.startsWith(lastSnapshot)
      ? text.slice(lastSnapshot.length)
      : text
    if (diff.length > 0) process.stdout.write(diff)
    lastSnapshot = text
  }

  await client
    .captureAgent(runId, targetHandle, { lines: captureLines(), raw: true })
    .then((pane) => renderSnapshot(pane.text))
    .catch(() => undefined)

  if (!readonly) {
    let polling = false
    pollTimer = setInterval(() => {
      if (stopped || polling) return
      polling = true
      void client
        .captureAgent(runId, targetHandle, { lines: captureLines(), raw: true })
        .then((pane) => renderSnapshot(pane.text))
        .catch(() => undefined)
        .finally(() => {
          polling = false
        })
    }, 500)
    pollTimer.unref?.()
  }

  ;(async () => {
    try {
      for await (const ev of client.streamEvents(runId, { signal: ac.signal })) {
        if (ev.type === 'pane.snapshot' && ev.handle === targetHandle) {
          renderSnapshot(ev.text)
        }
        if (ev.type === 'run.ended') stopRendering()
      }
    } catch {
      /* abort */
    }
  })().catch(() => undefined)

  if (!readonly) {
    const inputState = createSwarmAttachInputState()
    process.stdin.on('data', async (chunk: Buffer) => {
      for (const action of parseSwarmAttachInputChunk(chunk, inputState)) {
        if (action.type === 'detach') {
          stopRendering(false)
          await releaseInteractiveLease()
          process.exit(0)
        }
        if (action.type === 'cancel') {
          await client.cancel(runId).catch(() => undefined)
          stopRendering(false)
          await releaseInteractiveLease()
          process.exit(0)
        }
        if (action.type === 'list-agents') {
          const snap = await client.get(runId)
          process.stderr.write(
            '\nagents:\n'
              + snap.agents.map((a) => `  ${a.handle} ${a.agent} ${a.status}`).join('\n')
              + '\n',
          )
          continue
        }
        await client
          .sendKeys(runId, targetHandle, { keys: action.keys, enter: action.enter })
          .catch(() => undefined)
      }
    })

    process.stdout.on('resize', () => {
      const cols = process.stdout.columns ?? 80
      const rows = process.stdout.rows ?? 24
      void client
        .sendKeys(runId, targetHandle, { resize: { cols, rows } })
        .catch(() => undefined)
    })
  }
}

export interface SwarmListDeps {
  client: Pick<SwarmClient, 'list'>
  status?: SwarmRunStatus
  printer?: (s: string) => void
}

export async function runSwarmList(deps: SwarmListDeps): Promise<void> {
  const print = deps.printer ?? ((s: string) => console.log(s))
  const runs = await deps.client.list(deps.status)
  if (!runs.length) {
    print(chalk.gray('(no runs)'))
    return
  }
  for (const r of runs) {
    print(`${r.id}  ${chalk.cyan(r.status)}  ${r.goal}`)
  }
}

export interface SwarmStatusDeps {
  client: Pick<SwarmClient, 'get'>
  runId: string
  printer?: (s: string) => void
}

export async function runSwarmStatus(deps: SwarmStatusDeps): Promise<void> {
  const print = deps.printer ?? ((s: string) => console.log(s))
  const run = await deps.client.get(deps.runId)
  print(`id: ${run.id}`)
  print(`status: ${run.status}`)
  print(`goal: ${run.goal}`)
  print(
    `worktree: ${run.worktree.path}${run.worktree.createdByDaemon ? ' (daemon-created)' : ''}`,
  )
  print(`active: ${run.activeHandle ?? '-'}`)
  print(`agents: ${run.agents.length}`)
  const blocked = formatBlockedAgents(run.agents)
  if (blocked) print(blocked)
}

export interface SwarmAgentsDeps {
  client: Pick<SwarmClient, 'get'>
  runId: string
  printer?: (s: string) => void
}

export async function runSwarmAgents(deps: SwarmAgentsDeps): Promise<void> {
  const print = deps.printer ?? ((s: string) => console.log(s))
  const run = await deps.client.get(deps.runId)
  for (const a of run.agents) {
    const isActive = run.activeHandle === a.handle ? chalk.green('●') : ' '
    const startup = formatAgentStartupState(a)
    print(
      `${isActive} ${a.handle}  ${chalk.cyan(a.agent)}  ${chalk.gray(a.status)}  ${startup}  ${a.role ?? ''}`,
    )
  }
}

export interface SwarmKillDeps {
  client: Pick<SwarmClient, 'cancel'>
  runId: string
  printer?: (s: string) => void
}

export async function runSwarmKill(deps: SwarmKillDeps): Promise<void> {
  const print = deps.printer ?? ((s: string) => console.log(s))
  await deps.client.cancel(deps.runId)
  print(`${chalk.yellow('cancelled')} ${deps.runId}`)
}

export async function swarmListCommand(
  options: { url?: string; status?: SwarmRunStatus },
): Promise<void> {
  const client = buildClient(options.url)
  await runSwarmList({ client, status: options.status })
}

export async function swarmStatusCommand(
  runId: string,
  options: { url?: string },
): Promise<void> {
  const client = buildClient(options.url)
  await runSwarmStatus({ client, runId })
}

export async function swarmAgentsCommand(
  runId: string,
  options: { url?: string },
): Promise<void> {
  const client = buildClient(options.url)
  await runSwarmAgents({ client, runId })
}

export async function swarmKillCommand(
  runId: string,
  options: { url?: string },
): Promise<void> {
  const client = buildClient(options.url)
  await runSwarmKill({ client, runId })
}

export interface SwarmDriveOptions {
  url?: string
  maxTurns?: string
  timeoutSec?: string
  continuePrompt?: string
  followup?: string[]
  stopPattern?: string[]
  watch?: boolean
}

export async function swarmDriveCommand(
  runId: string,
  handle: string,
  prompt: string,
  options: SwarmDriveOptions,
): Promise<void> {
  const client = buildClient(options.url)
  const watch = options.watch ? watchSwarmDrive(client, runId, handle, Date.now()) : undefined
  let result: DriveAgentResult
  try {
    result = await client.driveAgent(runId, handle, {
      prompt,
      followups: options.followup,
      continue_prompt: options.continuePrompt,
      stop_patterns: options.stopPattern,
      max_turns: options.maxTurns ? Number(options.maxTurns) : undefined,
      timeout_sec: options.timeoutSec ? Number(options.timeoutSec) : undefined,
    })
  } finally {
    await watch?.stop()
  }
  const status = result.status === 'done' ? chalk.green(result.status) : chalk.yellow(result.status)
  console.log(`${status} ${runId}/${handle} turns=${result.turns.length}`)
  const last = result.turns[result.turns.length - 1]
  if (last?.reason) console.log(chalk.gray(`reason: ${last.reason}`))
  const requiredAction = result.required_action ?? last?.required_action
  if (requiredAction) {
    console.log(chalk.yellow(`required_action: ${requiredAction.type} (${requiredAction.action})`))
    console.log(chalk.gray(`suggested_input: ${requiredAction.suggested_input}`))
  }
  if (result.lastOutput.trim()) console.log(result.lastOutput.trim())
}

function watchSwarmDrive(
  client: Pick<SwarmClient, 'streamEvents'>,
  runId: string,
  handle: string,
  since: number,
): { stop: () => Promise<void> } {
  const ac = new AbortController()
  const done = (async () => {
    try {
      for await (const ev of client.streamEvents(runId, { since, signal: ac.signal })) {
        if (!isDriveWatchEvent(ev, handle)) continue
        console.log(fmt(ev))
      }
    } catch {
      /* stream abort or transient watch failure; drive result remains authoritative */
    }
  })()
  return {
    async stop() {
      ac.abort()
      await done.catch(() => undefined)
    },
  }
}

function isDriveWatchEvent(ev: SwarmEvent, handle: string): boolean {
  if (ev.type === 'supervisor.message') return true
  if (ev.type === 'pane.snapshot') return ev.handle === handle
  if (ev.type === 'agent.status') return ev.handle === handle
  if (ev.type === 'agent.startup') return ev.handle === handle
  if (ev.type === 'agent.recovery') return ev.handle === handle
  if (ev.type === 'tool.call' || ev.type === 'tool.result') return ev.tool === 'swarm.drive'
  return false
}

function fmt(ev: SwarmEvent): string {
  const t = new Date(ev.ts).toISOString().slice(11, 19)
  switch (ev.type) {
    case 'run.started': return `${t}  ${chalk.bold('[run]')}      started — goal: "${ev.goal}"`
    case 'run.ended': return `${t}  ${chalk.bold('[run]')}      ${ev.status}`
    case 'agent.spawned': return `${t}  ${chalk.bold('[agent]')}    ${ev.agent.agent} (${ev.agent.handle}) spawned`
    case 'agent.startup': {
      const state = ev.evidence.lifecycleState
      const label = isStartupBlocker(state) ? chalk.yellow('[blocker]') : chalk.bold('[startup]')
      const preview = ev.evidence.lastOutputPreview ? ` — ${truncate(ev.evidence.lastOutputPreview)}` : ''
      return `${t}  ${label}  ${ev.handle} ${ev.evidence.runtime}/${state}${preview}`
    }
    case 'agent.recovery': {
      const replacement = ev.replacementHandle ? ` → ${ev.replacementHandle}` : ''
      return `${t}  ${chalk.yellow('[recover]')}  ${ev.handle} ${ev.scenario}/${ev.action} ${ev.status}${replacement}`
    }
    case 'agent.killed': return `${t}  ${chalk.bold('[agent]')}    killed ${ev.handle}`
    case 'agent.status': return `${t}  ${chalk.bold('[agent]')}    ${ev.handle} → ${ev.status}`
    case 'agent.active': return `${t}  ${chalk.green('[active]')}   → ${ev.handle}`
    case 'tool.call': return `${t}  ${chalk.bold('[tool]')}     ${ev.tool}  ${JSON.stringify(ev.input)}`
    case 'tool.result': return `${t}  ${chalk.bold('[tool]')}     ${ev.tool} → ${ev.outputPreview}`
    case 'pane.snapshot': return `${t}  ${chalk.bold('[pane]')}     ${ev.handle} (${ev.text.length} chars)`
    case 'supervisor.message': return `${t}  ${chalk.magenta('[super]')}    ${ev.text}`
    default: return `${t}  [event] ${JSON.stringify(ev)}`
  }
}

function passes(ev: SwarmEvent, filter: string | undefined): boolean {
  if (!filter) return true
  switch (filter) {
    case 'tool': return ev.type === 'tool.call' || ev.type === 'tool.result'
    case 'pane': return ev.type === 'pane.snapshot'
    case 'agent': return ev.type.startsWith('agent.')
    default: return true
  }
}

export interface SwarmLogsDeps {
  client: Pick<SwarmClient, 'events' | 'streamEvents'>
  runId: string
  options: { follow?: boolean; since?: number; filter?: string; json?: boolean }
  printer?: (s: string) => void
}

export async function runSwarmLogs(deps: SwarmLogsDeps): Promise<void> {
  const print = deps.printer ?? ((s) => console.log(s))
  const print1 = (e: SwarmEvent): void => {
    if (!passes(e, deps.options.filter)) return
    print(deps.options.json ? JSON.stringify(e) : fmt(e))
  }
  if (!deps.options.follow) {
    const evs = await deps.client.events(deps.runId, deps.options.since)
    for (const e of evs) print1(e)
    return
  }
  for await (const e of deps.client.streamEvents(deps.runId, { since: deps.options.since })) {
    print1(e)
    if (e.type === 'run.ended') break
  }
}

export async function swarmLogsCommand(
  runId: string,
  options: { url?: string; follow?: boolean; since?: string; filter?: string; json?: boolean },
): Promise<void> {
  const client = buildClient(options.url)
  await runSwarmLogs({
    client,
    runId,
    options: {
      follow: options.follow,
      since: options.since ? Number(options.since) : undefined,
      filter: options.filter,
      json: options.json,
    },
  })
}
