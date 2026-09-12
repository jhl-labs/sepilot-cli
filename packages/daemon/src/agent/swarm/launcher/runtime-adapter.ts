import { randomUUID } from 'node:crypto'
import { setTimeout as sleep } from 'node:timers/promises'
import type { SwarmAgentHandle, SwarmAgentName } from '@sepilotd/core'
import { A2AHttpClient } from '../../../a2a/client.js'
import { AcpStdioClient, type AcpClientSpec } from '../../../acp/client.js'
import {
  isExternalAcpAgentName,
  resolveExternalAcpAgentPreset,
} from '../../../acp/external-agent.js'
import type { JsonRpcMessage } from '../../../acp/rpc.js'
import type { LaunchOpts, AgentLauncher } from './agent-launcher.js'
import { getAgentConfig } from '../config/agents.js'
import { formatStartupCommand, redactStartupEvidenceText } from '../run/startup-evidence.js'
import type { TmuxSessionPool } from '../tmux/pool.js'
import { resolveTmuxKeyName } from '../tmux/keymap.js'

export interface SwarmAgentIdleResult {
  status: 'idle' | 'timeout'
  output: string
  snapshot: string
}

export interface SwarmAgentRuntimeAdapter {
  readonly name: string
  launch(opts: LaunchOpts): Promise<SwarmAgentHandle>
  stop(handle: SwarmAgentHandle): Promise<void>
  sendPrompt(handle: SwarmAgentHandle, prompt: string): Promise<void>
  sendKeys(handle: SwarmAgentHandle, keys: string, enter?: boolean): Promise<void>
  sendNamedKeys(handle: SwarmAgentHandle, keyName: string): Promise<void>
  capture(handle: SwarmAgentHandle, lines?: number, options?: { raw?: boolean }): Promise<string>
  readNewOutput(handle: SwarmAgentHandle): Promise<string>
  resize(handle: SwarmAgentHandle, cols: number, rows: number): Promise<void>
  waitForIdle?(handle: SwarmAgentHandle, options: {
    timeoutMs: number
    signal?: AbortSignal
  }): Promise<SwarmAgentIdleResult> | undefined
}

type AcpClientLike = Pick<
  AcpStdioClient,
  'start' | 'initialize' | 'newSession' | 'prompt' | 'cancel' | 'stop'
>
type AcpClientOptions = ConstructorParameters<typeof AcpStdioClient>[1]
type AcpClientFactory = (spec: AcpClientSpec, options: AcpClientOptions) => AcpClientLike

interface TextRuntimeState {
  transcript: string[]
  cursor: number
  inputBuffer: string
}

function appendTranscript(state: TextRuntimeState, text: string): void {
  const trimmed = text.trim()
  if (!trimmed) return
  if (state.transcript[state.transcript.length - 1] === trimmed) return
  state.transcript.push(trimmed)
}

function transcriptText(state: TextRuntimeState): string {
  return state.transcript.join('\n\n')
}

function readNewTranscript(state: TextRuntimeState): string {
  const text = transcriptText(state)
  const next = text.slice(state.cursor)
  state.cursor = text.length
  return next.trimStart()
}

async function sendBufferedTextInput(
  state: TextRuntimeState,
  keys: string,
  enter: boolean,
  sendPrompt: (prompt: string) => Promise<void>,
): Promise<void> {
  state.inputBuffer += keys
  if (!enter) return
  const prompt = state.inputBuffer.trim()
  state.inputBuffer = ''
  if (prompt) await sendPrompt(prompt)
}

function waitForTextRuntimeIdle(state: TextRuntimeState): SwarmAgentIdleResult {
  const snapshot = transcriptText(state)
  const output = readNewTranscript(state)
  return { status: 'idle', output, snapshot }
}

function extractRecordText(value: unknown): string {
  if (typeof value === 'string') return value
  if (!value || typeof value !== 'object') return ''
  const record = value as Record<string, unknown>
  if (typeof record.text === 'string') return record.text
  if (typeof record.content === 'string') return record.content
  if (record.content && typeof record.content === 'object' && !Array.isArray(record.content)) {
    return extractRecordText(record.content)
  }
  if (Array.isArray(record.content)) {
    return record.content.map(extractRecordText).filter(Boolean).join('')
  }
  return ''
}

function extractAcpNotificationText(message: JsonRpcMessage): string {
  if (message.method !== 'session/update') return ''
  const params = message.params
  if (!params || typeof params !== 'object') return ''
  const update = (params as { update?: unknown }).update
  if (!update || typeof update !== 'object') return ''
  const record = update as Record<string, unknown>
  const text = extractRecordText(record.content)
  if (text) return text
  const title = typeof record.title === 'string' ? record.title : ''
  const status = typeof record.status === 'string' ? record.status : ''
  return title && status ? `[${status}] ${title}` : title
}

function envKeyForAgent(agent: SwarmAgentName, suffix: string): string {
  return `SEPILOTD_SWARM_${agent.toUpperCase()}_${suffix}`
}

function parseTimeoutMs(value: string | undefined, fallback: number): number {
  const parsed = value ? Number(value) : NaN
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

export class TmuxSwarmAgentRuntimeAdapter implements SwarmAgentRuntimeAdapter {
  readonly name = 'tmux'

  constructor(
    private readonly pool: TmuxSessionPool,
    private readonly launcher: AgentLauncher,
  ) {}

  launch(opts: LaunchOpts): Promise<SwarmAgentHandle> {
    return this.launcher.launch(opts)
  }

  stop(handle: SwarmAgentHandle): Promise<void> {
    return this.launcher.stop(handle)
  }

  async sendPrompt(handle: SwarmAgentHandle, prompt: string): Promise<void> {
    const sendCfg = getAgentConfig(handle.agent)
    const submitEnters = sendCfg.submitEnters ?? 1
    await this.pool.sendKeys(handle.tmuxSessionName, prompt, submitEnters > 0)
    for (let i = 1; i < submitEnters; i++) {
      await sleep(150)
      await this.pool.sendKeys(handle.tmuxSessionName, '', true, { updateCursor: false })
    }
  }

  sendKeys(handle: SwarmAgentHandle, keys: string, enter = false): Promise<void> {
    return this.pool.sendKeys(handle.tmuxSessionName, keys, enter)
  }

  sendNamedKeys(handle: SwarmAgentHandle, keyName: string): Promise<void> {
    return this.pool.sendNamedKeys(handle.tmuxSessionName, keyName)
  }

  capture(handle: SwarmAgentHandle, lines = 200, options?: { raw?: boolean }): Promise<string> {
    return options?.raw
      ? this.pool.captureRaw(handle.tmuxSessionName, lines)
      : this.pool.captureClean(handle.tmuxSessionName, lines)
  }

  readNewOutput(handle: SwarmAgentHandle): Promise<string> {
    return this.pool.readNewOutput(handle.tmuxSessionName)
  }

  resize(handle: SwarmAgentHandle, cols: number, rows: number): Promise<void> {
    return this.pool.resize(handle.tmuxSessionName, cols, rows)
  }
}

export interface AcpSwarmAgentRuntimeAdapterOptions {
  env?: NodeJS.ProcessEnv
  clientFactory?: AcpClientFactory
  timeoutMs?: number
}

interface AcpRuntimeState extends TextRuntimeState {
  client: AcpClientLike
  externalSessionId: string
}

export class AcpSwarmAgentRuntimeAdapter implements SwarmAgentRuntimeAdapter {
  readonly name = 'acp'
  private readonly env: NodeJS.ProcessEnv
  private readonly clientFactory: AcpClientFactory
  private readonly timeoutMs: number
  private readonly states = new Map<string, AcpRuntimeState>()

  constructor(options: AcpSwarmAgentRuntimeAdapterOptions = {}) {
    this.env = options.env ?? process.env
    this.clientFactory = options.clientFactory ?? ((spec, clientOptions) =>
      new AcpStdioClient(spec, clientOptions))
    this.timeoutMs = options.timeoutMs
      ?? parseTimeoutMs(this.env.SEPILOTD_SWARM_ACP_TIMEOUT_MS, 120_000)
  }

  supports(agent: SwarmAgentName): boolean {
    return isExternalAcpAgentName(agent)
  }

  async launch(opts: LaunchOpts): Promise<SwarmAgentHandle> {
    const startedAt = Date.now()
    if (!isExternalAcpAgentName(opts.agent)) {
      throw new Error(`ACP swarm runtime does not support agent: ${opts.agent}`)
    }
    const resolved = resolveExternalAcpAgentPreset(opts.agent, this.env)
    const state: AcpRuntimeState = {
      client: null as unknown as AcpClientLike,
      externalSessionId: '',
      transcript: [],
      cursor: 0,
      inputBuffer: '',
    }
    const client = this.clientFactory({
      command: resolved.command,
      args: resolved.args,
      framing: resolved.framing,
      cwd: opts.cwd,
      env: this.env,
    }, {
      requestTimeoutMs: this.timeoutMs,
      onNotification: (message) => {
        appendTranscript(state, extractAcpNotificationText(message))
      },
    })
    state.client = client

    try {
      client.start()
      await client.initialize()
      const session = await client.newSession({ cwd: opts.cwd })
      state.externalSessionId = session.sessionId
    } catch (error) {
      client.stop()
      throw error
    }

    const readyAt = Date.now()
    const handle: SwarmAgentHandle = {
      handle: opts.handle,
      agent: opts.agent,
      role: opts.role,
      tmuxSessionName: `acp:${opts.runId}:${opts.handle}:${state.externalSessionId}`,
      cwd: opts.cwd,
      status: 'idle',
      spawnedAt: readyAt,
      runtime: 'acp',
      startupEvidence: {
        handle: opts.handle,
        agent: opts.agent,
        runtime: 'acp',
        lifecycleState: 'ready_for_prompt',
        cwd: opts.cwd,
        paneCommand: formatStartupCommand([resolved.command, ...resolved.args]),
        startedAt,
        readyAt,
        promptAccepted: false,
        trustPromptDetected: false,
        toolPermissionPromptDetected: false,
        transportHealthy: true,
        elapsedMs: Math.max(0, readyAt - startedAt),
      },
    }
    this.states.set(handle.handle, state)
    return handle
  }

  async stop(handle: SwarmAgentHandle): Promise<void> {
    const state = this.requireState(handle)
    try {
      if (state.externalSessionId) state.client.cancel(state.externalSessionId)
    } finally {
      state.client.stop()
      this.states.delete(handle.handle)
    }
  }

  async sendPrompt(handle: SwarmAgentHandle, prompt: string): Promise<void> {
    const state = this.requireState(handle)
    appendTranscript(state, `> ${prompt}`)
    try {
      const result = await state.client.prompt(state.externalSessionId, prompt)
      appendTranscript(state, extractRecordText(result))
    } catch (error) {
      appendTranscript(state, `[acp error] ${error instanceof Error ? error.message : String(error)}`)
      throw error
    }
  }

  async sendKeys(handle: SwarmAgentHandle, keys: string, enter = false): Promise<void> {
    const state = this.requireState(handle)
    await sendBufferedTextInput(state, keys, enter, (prompt) => this.sendPrompt(handle, prompt))
  }

  async sendNamedKeys(handle: SwarmAgentHandle, keyName: string): Promise<void> {
    const resolved = resolveTmuxKeyName(keyName)
    if (resolved === 'C-c' || resolved === 'Escape') {
      const state = this.requireState(handle)
      state.client.cancel(state.externalSessionId)
      appendTranscript(state, `[${resolved}] cancelled`)
      return
    }
    throw new Error(`ACP swarm runtime does not support key: ${keyName}`)
  }

  async capture(handle: SwarmAgentHandle): Promise<string> {
    return transcriptText(this.requireState(handle))
  }

  async readNewOutput(handle: SwarmAgentHandle): Promise<string> {
    return readNewTranscript(this.requireState(handle))
  }

  async resize(_handle: SwarmAgentHandle, _cols: number, _rows: number): Promise<void> {
    // ACP stdio agents are not pane-backed; resize is intentionally a no-op.
  }

  async waitForIdle(
    handle: SwarmAgentHandle,
    _options?: { timeoutMs: number; signal?: AbortSignal },
  ): Promise<SwarmAgentIdleResult> {
    return waitForTextRuntimeIdle(this.requireState(handle))
  }

  private requireState(handle: SwarmAgentHandle): AcpRuntimeState {
    const state = this.states.get(handle.handle)
    if (!state) throw new Error(`ACP swarm handle is not active: ${handle.handle}`)
    return state
  }
}

export interface A2ASwarmAgentRuntimeAdapterOptions {
  env?: NodeJS.ProcessEnv
  client?: A2AHttpClient
  timeoutMs?: number
}

interface A2ARuntimeState extends TextRuntimeState {
  agentCardUrl: string
  headers?: Record<string, string>
  contextId: string
}

function validateA2AAgentCardUrl(raw: string, label: string): string {
  let parsed: URL
  try {
    parsed = new URL(raw)
  } catch {
    throw new Error(`${label} must be an absolute URL`)
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`${label} must use http or https`)
  }
  if (parsed.username || parsed.password) {
    throw new Error(`${label} must not contain credentials`)
  }
  return parsed.toString()
}

function a2aAgentCardUrlForAgent(env: NodeJS.ProcessEnv, agent: SwarmAgentName): string | null {
  const specific = env[envKeyForAgent(agent, 'A2A_AGENT_CARD_URL')]
    ?? env[envKeyForAgent(agent, 'A2A_URL')]
  const global = env.SEPILOTD_SWARM_A2A_AGENT_CARD_URL ?? env.SEPILOTD_SWARM_A2A_URL
  const raw = specific ?? global
  return raw?.trim() ? validateA2AAgentCardUrl(raw.trim(), `A2A URL for ${agent}`) : null
}

function a2aHeadersForAgent(env: NodeJS.ProcessEnv, agent: SwarmAgentName): Record<string, string> | undefined {
  const authorization = env[envKeyForAgent(agent, 'A2A_AUTHORIZATION')]
    ?? env.SEPILOTD_SWARM_A2A_AUTHORIZATION
  return authorization?.trim() ? { authorization: authorization.trim() } : undefined
}

export class A2ASwarmAgentRuntimeAdapter implements SwarmAgentRuntimeAdapter {
  readonly name = 'a2a'
  private readonly env: NodeJS.ProcessEnv
  private readonly client: A2AHttpClient
  private readonly timeoutMs: number
  private readonly states = new Map<string, A2ARuntimeState>()

  constructor(options: A2ASwarmAgentRuntimeAdapterOptions = {}) {
    this.env = options.env ?? process.env
    this.client = options.client ?? new A2AHttpClient()
    this.timeoutMs = options.timeoutMs
      ?? parseTimeoutMs(this.env.SEPILOTD_SWARM_A2A_TIMEOUT_MS, 120_000)
  }

  supports(agent: SwarmAgentName): boolean {
    return Boolean(a2aAgentCardUrlForAgent(this.env, agent))
  }

  async launch(opts: LaunchOpts): Promise<SwarmAgentHandle> {
    const startedAt = Date.now()
    const agentCardUrl = a2aAgentCardUrlForAgent(this.env, opts.agent)
    if (!agentCardUrl) {
      throw new Error(
        `A2A swarm runtime for ${opts.agent} requires ${envKeyForAgent(opts.agent, 'A2A_AGENT_CARD_URL')} or SEPILOTD_SWARM_A2A_AGENT_CARD_URL`,
      )
    }
    const state: A2ARuntimeState = {
      agentCardUrl,
      headers: a2aHeadersForAgent(this.env, opts.agent),
      contextId: randomUUID(),
      transcript: [],
      cursor: 0,
      inputBuffer: '',
    }
    await this.client.getAgentCard(agentCardUrl)
    const readyAt = Date.now()
    const handle: SwarmAgentHandle = {
      handle: opts.handle,
      agent: opts.agent,
      role: opts.role,
      tmuxSessionName: `a2a:${opts.runId}:${opts.handle}:${state.contextId}`,
      cwd: opts.cwd,
      status: 'idle',
      spawnedAt: readyAt,
      runtime: 'a2a',
      startupEvidence: {
        handle: opts.handle,
        agent: opts.agent,
        runtime: 'a2a',
        lifecycleState: 'ready_for_prompt',
        cwd: opts.cwd,
        paneCommand: redactStartupEvidenceText(agentCardUrl),
        startedAt,
        readyAt,
        promptAccepted: false,
        trustPromptDetected: false,
        toolPermissionPromptDetected: false,
        transportHealthy: true,
        elapsedMs: Math.max(0, readyAt - startedAt),
      },
    }
    this.states.set(handle.handle, state)
    return handle
  }

  async stop(handle: SwarmAgentHandle): Promise<void> {
    this.states.delete(handle.handle)
  }

  async sendPrompt(handle: SwarmAgentHandle, prompt: string): Promise<void> {
    const state = this.requireState(handle)
    appendTranscript(state, `> ${prompt}`)
    try {
      const result = await this.client.sendMessage({
        agentCardUrl: state.agentCardUrl,
        message: prompt,
        contextId: state.contextId,
        headers: state.headers,
        timeoutMs: this.timeoutMs,
      })
      const task = result.result.task
      if (task?.contextId) state.contextId = task.contextId
      appendTranscript(state, result.output || JSON.stringify(result.result, null, 2))
    } catch (error) {
      appendTranscript(state, `[a2a error] ${error instanceof Error ? error.message : String(error)}`)
      throw error
    }
  }

  async sendKeys(handle: SwarmAgentHandle, keys: string, enter = false): Promise<void> {
    const state = this.requireState(handle)
    await sendBufferedTextInput(state, keys, enter, (prompt) => this.sendPrompt(handle, prompt))
  }

  async sendNamedKeys(_handle: SwarmAgentHandle, keyName: string): Promise<void> {
    const resolved = resolveTmuxKeyName(keyName)
    if (resolved === 'C-c' || resolved === 'Escape') {
      throw new Error('A2A swarm runtime does not support interrupt keys yet')
    }
    throw new Error(`A2A swarm runtime does not support key: ${keyName}`)
  }

  async capture(handle: SwarmAgentHandle): Promise<string> {
    return transcriptText(this.requireState(handle))
  }

  async readNewOutput(handle: SwarmAgentHandle): Promise<string> {
    return readNewTranscript(this.requireState(handle))
  }

  async resize(_handle: SwarmAgentHandle, _cols: number, _rows: number): Promise<void> {
    // Remote A2A agents are not pane-backed; resize is intentionally a no-op.
  }

  async waitForIdle(
    handle: SwarmAgentHandle,
    _options?: { timeoutMs: number; signal?: AbortSignal },
  ): Promise<SwarmAgentIdleResult> {
    return waitForTextRuntimeIdle(this.requireState(handle))
  }

  private requireState(handle: SwarmAgentHandle): A2ARuntimeState {
    const state = this.states.get(handle.handle)
    if (!state) throw new Error(`A2A swarm handle is not active: ${handle.handle}`)
    return state
  }
}

type SwarmRuntimeName = 'tmux' | 'acp' | 'a2a'

function parseRuntimeName(value: string | undefined): SwarmRuntimeName | null {
  const normalized = value?.trim().toLowerCase()
  if (!normalized) return null
  if (normalized === 'tmux' || normalized === 'acp' || normalized === 'a2a') return normalized
  throw new Error(`Unsupported swarm runtime: ${value}`)
}

function runtimeNameForAgent(env: NodeJS.ProcessEnv, agent: SwarmAgentName): SwarmRuntimeName {
  return parseRuntimeName(env[envKeyForAgent(agent, 'RUNTIME')])
    ?? parseRuntimeName(env.SEPILOTD_SWARM_AGENT_RUNTIME)
    ?? 'tmux'
}

export interface MultiplexSwarmAgentRuntimeAdapterOptions {
  env?: NodeJS.ProcessEnv
  tmux: TmuxSwarmAgentRuntimeAdapter
  acp?: AcpSwarmAgentRuntimeAdapter
  a2a?: A2ASwarmAgentRuntimeAdapter
}

export class MultiplexSwarmAgentRuntimeAdapter implements SwarmAgentRuntimeAdapter {
  readonly name = 'multiplex'
  private readonly env: NodeJS.ProcessEnv
  private readonly acp: AcpSwarmAgentRuntimeAdapter
  private readonly a2a: A2ASwarmAgentRuntimeAdapter
  private readonly handles = new Map<string, SwarmAgentRuntimeAdapter>()

  constructor(private readonly options: MultiplexSwarmAgentRuntimeAdapterOptions) {
    this.env = options.env ?? process.env
    this.acp = options.acp ?? new AcpSwarmAgentRuntimeAdapter({ env: this.env })
    this.a2a = options.a2a ?? new A2ASwarmAgentRuntimeAdapter({ env: this.env })
  }

  async launch(opts: LaunchOpts): Promise<SwarmAgentHandle> {
    const runtime = this.runtimeForLaunch(opts.agent)
    const handle = await runtime.launch(opts)
    this.handles.set(handle.handle, runtime)
    return handle
  }

  async stop(handle: SwarmAgentHandle): Promise<void> {
    try {
      await this.runtimeForHandle(handle).stop(handle)
    } finally {
      this.handles.delete(handle.handle)
    }
  }

  sendPrompt(handle: SwarmAgentHandle, prompt: string): Promise<void> {
    return this.runtimeForHandle(handle).sendPrompt(handle, prompt)
  }

  sendKeys(handle: SwarmAgentHandle, keys: string, enter = false): Promise<void> {
    return this.runtimeForHandle(handle).sendKeys(handle, keys, enter)
  }

  sendNamedKeys(handle: SwarmAgentHandle, keyName: string): Promise<void> {
    return this.runtimeForHandle(handle).sendNamedKeys(handle, keyName)
  }

  capture(handle: SwarmAgentHandle, lines?: number, options?: { raw?: boolean }): Promise<string> {
    return this.runtimeForHandle(handle).capture(handle, lines, options)
  }

  readNewOutput(handle: SwarmAgentHandle): Promise<string> {
    return this.runtimeForHandle(handle).readNewOutput(handle)
  }

  resize(handle: SwarmAgentHandle, cols: number, rows: number): Promise<void> {
    return this.runtimeForHandle(handle).resize(handle, cols, rows)
  }

  waitForIdle(handle: SwarmAgentHandle, options: { timeoutMs: number; signal?: AbortSignal }): Promise<SwarmAgentIdleResult> | undefined {
    return this.runtimeForHandle(handle).waitForIdle?.(handle, options)
  }

  private runtimeForLaunch(agent: SwarmAgentName): SwarmAgentRuntimeAdapter {
    const name = runtimeNameForAgent(this.env, agent)
    if (name === 'tmux') return this.options.tmux
    if (name === 'acp') {
      if (!this.acp.supports(agent)) {
        throw new Error(`ACP swarm runtime does not support agent: ${agent}`)
      }
      return this.acp
    }
    if (!this.a2a.supports(agent)) {
      throw new Error(
        `A2A swarm runtime for ${agent} is selected but no A2A Agent Card URL is configured`,
      )
    }
    return this.a2a
  }

  private runtimeForHandle(handle: SwarmAgentHandle): SwarmAgentRuntimeAdapter {
    const runtime = this.handles.get(handle.handle)
    if (runtime) return runtime
    if (handle.tmuxSessionName.startsWith('acp:')) return this.acp
    if (handle.tmuxSessionName.startsWith('a2a:')) return this.a2a
    return this.options.tmux
  }
}

export function createDefaultSwarmAgentRuntimeAdapter(
  pool: TmuxSessionPool,
  launcher: AgentLauncher,
  options: {
    env?: NodeJS.ProcessEnv
    acpClientFactory?: AcpClientFactory
    a2aClient?: A2AHttpClient
  } = {},
): SwarmAgentRuntimeAdapter {
  const env = options.env ?? process.env
  const tmux = new TmuxSwarmAgentRuntimeAdapter(pool, launcher)
  return new MultiplexSwarmAgentRuntimeAdapter({
    env,
    tmux,
    acp: new AcpSwarmAgentRuntimeAdapter({ env, clientFactory: options.acpClientFactory }),
    a2a: new A2ASwarmAgentRuntimeAdapter({ env, client: options.a2aClient }),
  })
}
