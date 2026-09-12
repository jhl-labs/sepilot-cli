import { randomUUID } from 'node:crypto'
import { resolve } from 'node:path'
import type { AgentRunContract, ISessionStore } from '@sepilotd/core'
import { AcpStdioClient, type AcpClientSpec, type AcpStdioClient as AcpClient } from './client.js'
import type { JsonRpcFraming, JsonRpcMessage } from './rpc.js'
import { triggerDreamingSessionEnd, type DreamingEngine } from '../memory/dreaming.js'
import { formatRunContractForPrompt } from '../agent/task-contract.js'

export const EXTERNAL_ACP_AGENT_NAMES = ['opencode', 'codex'] as const
export type ExternalAcpAgentName = typeof EXTERNAL_ACP_AGENT_NAMES[number]

export function isExternalAcpAgentName(value: unknown): value is ExternalAcpAgentName {
  return typeof value === 'string'
    && (EXTERNAL_ACP_AGENT_NAMES as readonly string[]).includes(value)
}

export interface ExternalAcpDispatchInput {
  prompt: string
  agent?: ExternalAcpAgentName
  cwd?: string
  sessionId?: string
  parentSessionId?: string
  timeoutMs?: number
  scopeTags?: string[]
  runContract?: AgentRunContract
}

export interface ExternalAcpDispatchResult {
  output: string
  sessionId: string
  externalSessionId: string
  agent: string
  stopReason: string
  status: 'completed' | 'failed'
  events: ExternalAcpEvent[]
  error?: string
}

export interface ExternalAcpEvent {
  type: string
  text?: string
  title?: string
  status?: string
}

export interface ExternalAcpAgentDispatcherDeps {
  sessions: ISessionStore
  deviceName: string
  dreaming?: DreamingEngine
  clientFactory?: (spec: AcpClientSpec, options: { timeoutMs: number; onNotification: (message: JsonRpcMessage) => void }) => AcpClient
}

const DEFAULT_TIMEOUT_MS = 120_000

interface ExternalAcpAgentPreset {
  command: string
  args: string[]
  framing: JsonRpcFraming
  commandEnv?: string
  argsEnv?: string
}

const AGENT_PRESETS: Record<ExternalAcpAgentName, ExternalAcpAgentPreset> = {
  opencode: {
    command: 'opencode',
    // OpenCode 1.18+ emits newline-delimited JSON on its stdio ACP transport.
    // Keep external-agent launches aligned with the proven LLM provider path;
    // the legacy Content-Length reader waits forever for a header that never arrives.
    args: ['--pure', 'acp'],
    framing: 'ndjson',
  },
  codex: {
    // Codex CLI does not currently expose a native `codex acp` command.
    // Use a stdio ACP adapter by default, while allowing operators to point
    // at another adapter such as `acp-adapter --adapter codex`.
    command: 'codex-acp',
    args: [],
    framing: 'ndjson',
    commandEnv: 'SEPILOTD_CODEX_ACP_COMMAND',
    argsEnv: 'SEPILOTD_CODEX_ACP_ARGS',
  },
}

function parseEnvArgs(raw: string | undefined): string[] | undefined {
  const trimmed = raw?.trim()
  if (!trimmed) return undefined
  return trimmed.split(/\s+/).filter(Boolean)
}

export function resolveExternalAcpAgentPreset(
  agent: ExternalAcpAgentName,
  env: NodeJS.ProcessEnv = process.env,
): { command: string; args: string[]; framing: JsonRpcFraming } {
  return resolvePreset(AGENT_PRESETS[agent], env)
}

function resolvePreset(
  preset: ExternalAcpAgentPreset,
  env: NodeJS.ProcessEnv,
): { command: string; args: string[]; framing: JsonRpcFraming } {
  const command = preset.commandEnv && env[preset.commandEnv]?.trim()
    ? env[preset.commandEnv]!.trim()
    : preset.command
  const args = preset.argsEnv
    ? parseEnvArgs(env[preset.argsEnv]) ?? preset.args
    : preset.args
  return { command, args, framing: preset.framing }
}

function externalAgentErrorHint(agent: ExternalAcpAgentName): string {
  if (agent === 'codex') {
    return 'Install @agentclientprotocol/codex-acp on PATH, or set SEPILOTD_CODEX_ACP_COMMAND and SEPILOTD_CODEX_ACP_ARGS.'
  }
  return 'Install opencode on PATH, or choose another supported external ACP agent preset.'
}

function formatExternalAgentError(agent: ExternalAcpAgentName, message: string): string {
  if (/ENOENT|not found|command not found/i.test(message)) {
    return `${message}. ${externalAgentErrorHint(agent)}`
  }
  return message
}

function formatExternalAcpPrompt(prompt: string, runContract: AgentRunContract | undefined): string {
  const contractPrompt = formatRunContractForPrompt(runContract)
  if (!contractPrompt) return prompt
  return [
    contractPrompt,
    '[User task]',
    prompt,
  ].join('\n\n')
}

export class ExternalAcpAgentDispatcher {
  constructor(private readonly deps: ExternalAcpAgentDispatcherDeps) {}

  async dispatch(input: ExternalAcpDispatchInput): Promise<ExternalAcpDispatchResult> {
    const prompt = input.prompt.trim()
    if (!prompt) throw new Error('EXTERNAL_ACP_INVALID_REQUEST: prompt is required')

    const agent = input.agent ?? 'opencode'
    const preset = AGENT_PRESETS[agent]
    if (!preset) throw new Error(`EXTERNAL_ACP_UNKNOWN_AGENT: ${agent}`)
    const resolvedPreset = resolveExternalAcpAgentPreset(agent, process.env)

    const cwd = resolve(input.cwd ?? process.cwd())
    const sessionId = input.sessionId ?? randomUUID()
    const timeoutMs = Math.max(1, input.timeoutMs ?? DEFAULT_TIMEOUT_MS)
    const events: ExternalAcpEvent[] = []
    const assistantText: string[] = []
    const client = this.createClient({
      command: resolvedPreset.command,
      args: resolvedPreset.args,
      framing: resolvedPreset.framing,
      cwd,
      env: process.env,
    }, {
      timeoutMs,
      onNotification: (message) => {
        const event = mapAcpNotification(message)
        if (!event) return
        events.push(event)
        if (event.type === 'agent_message_chunk' && event.text) {
          assistantText.push(event.text)
        }
      },
    })

    const createdAt = new Date().toISOString()
    let existing = await this.deps.sessions.get?.(sessionId)
    if (!existing) {
      existing = await this.deps.sessions.create({
        id: sessionId,
        title: `external acp: ${prompt.slice(0, 40)}`,
        createdAt,
        updatedAt: createdAt,
        provider: 'external-acp',
        model: agent,
        device: this.deps.deviceName,
        status: 'active',
        cwd,
        tags: input.parentSessionId
          ? ['external-acp', `acp:${agent}`, `parent:${input.parentSessionId}`]
          : ['external-acp', `acp:${agent}`],
      })
    } else if (this.deps.sessions.updateMeta && existing.cwd !== cwd) {
      existing = await this.deps.sessions.updateMeta(sessionId, { cwd })
    }

    await this.deps.sessions.appendEvent(sessionId, {
      type: 'user_message',
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      content: prompt,
    })

    const startedAt = Date.now()
    let externalSessionId = ''
    try {
      client.start()
      await client.initialize()
      const session = await client.newSession({ cwd })
      externalSessionId = session.sessionId
      if (input.runContract) {
        await this.deps.sessions.appendEvent(sessionId, {
          type: 'run_contract',
          id: randomUUID(),
          timestamp: new Date().toISOString(),
          contract: input.runContract,
        })
      }
      const promptResult = await client.prompt(
        externalSessionId,
        formatExternalAcpPrompt(prompt, input.runContract),
      )
      const stopReason = extractStopReason(promptResult)
      const output = assistantText.join('').trim() || extractText(promptResult).trim()
      const finalOutput = output || `[external ACP agent stopped: ${stopReason}]`
      await this.deps.sessions.appendEvent(sessionId, {
        type: 'assistant_message',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        content: finalOutput,
      })
      await this.deps.sessions.appendEvent(sessionId, {
        type: 'session_end',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        totalTokens: { input: 0, output: 0 },
        totalCost: 0,
        duration_ms: Date.now() - startedAt,
      })
      triggerDreamingSessionEnd(this.deps.dreaming, sessionId, 'external-acp', input.scopeTags)
      return {
        output: finalOutput,
        sessionId,
        externalSessionId,
        agent,
        stopReason,
        status: 'completed',
        events,
      }
    } catch (error) {
      cancelExternalSession(client, externalSessionId)
      const rawMessage = error instanceof Error ? error.message : String(error)
      const message = formatExternalAgentError(agent, rawMessage)
      const partialOutput = assistantText.join('').trim()
      const errorOutput = partialOutput
        ? `${partialOutput}\n\n[external ACP agent error] ${message}`
        : `[external ACP agent error] ${message}`
      await this.deps.sessions.appendEvent(sessionId, {
        type: 'assistant_message',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        content: errorOutput,
      })
      await this.deps.sessions.appendEvent(sessionId, {
        type: 'session_end',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        totalTokens: { input: 0, output: 0 },
        totalCost: 0,
        duration_ms: Date.now() - startedAt,
      })
      triggerDreamingSessionEnd(this.deps.dreaming, sessionId, 'external-acp', input.scopeTags)
      return {
        output: errorOutput,
        sessionId,
        externalSessionId,
        agent,
        stopReason: 'error',
        status: 'failed',
        events,
        error: message,
      }
    } finally {
      client.stop()
    }
  }

  private createClient(
    spec: AcpClientSpec,
    options: { timeoutMs: number; onNotification: (message: JsonRpcMessage) => void },
  ): AcpClient {
    return this.deps.clientFactory
      ? this.deps.clientFactory(spec, options)
      : new AcpStdioClient(spec, {
          requestTimeoutMs: options.timeoutMs,
          onNotification: options.onNotification,
        })
  }
}

function cancelExternalSession(client: AcpClient, externalSessionId: string): void {
  if (!externalSessionId) return
  try {
    client.cancel(externalSessionId)
  } catch {
    /* Best-effort cleanup; the process is still stopped in the caller's finally block. */
  }
}

function extractStopReason(result: unknown): string {
  if (result && typeof result === 'object') {
    const stopReason = (result as { stopReason?: unknown }).stopReason
    if (typeof stopReason === 'string' && stopReason.trim()) return stopReason
  }
  return 'end_turn'
}

function extractText(value: unknown): string {
  if (typeof value === 'string') return value
  if (!value || typeof value !== 'object') return ''
  const record = value as Record<string, unknown>
  if (typeof record.text === 'string') return record.text
  if (typeof record.content === 'string') return record.content
  if (Array.isArray(record.content)) return record.content.map(extractText).join('')
  return ''
}

function mapAcpNotification(message: JsonRpcMessage): ExternalAcpEvent | null {
  if (message.method !== 'session/update') return null
  const params = message.params
  if (!params || typeof params !== 'object') return null
  const update = (params as { update?: unknown }).update
  if (!update || typeof update !== 'object') return null
  const record = update as Record<string, unknown>
  const type = typeof record.sessionUpdate === 'string'
    ? record.sessionUpdate
    : 'unknown'
  const event: ExternalAcpEvent = { type }
  const text = extractText(record.content)
  if (text) event.text = text
  if (typeof record.title === 'string') event.title = record.title
  if (typeof record.status === 'string') event.status = record.status
  return event
}
