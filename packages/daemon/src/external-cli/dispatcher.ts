import { randomUUID } from 'node:crypto'
import { resolve } from 'node:path'
import type { AgentRunContract, AutonomyLevel, ISessionStore } from '@sepilotd/core'
import { formatRunContractForPrompt } from '../agent/task-contract.js'
import type { UsageTracker } from '../memory/usage-tracker.js'
import {
  capPermission,
  externalCliPermissionCeiling,
  permissionForAutonomy,
  runExternalCli,
  type ExternalCliBackend,
  type ExternalCliEvent,
  type ExternalCliPermission,
  type ExternalCliRunOptions,
} from './executor.js'

export interface ExternalCliDispatchInput {
  backend: ExternalCliBackend
  prompt: string
  cwd?: string
  model?: string
  threadId?: string
  timeoutMs?: number
  autonomy?: AutonomyLevel
  maxPermission?: ExternalCliPermission
  parentSessionId?: string
  scopeTags?: string[]
  runContract?: AgentRunContract
  signal?: AbortSignal
  onEvent?: (event: ExternalCliEvent) => void
  sessionId?: string
}

export interface ExternalCliDispatchResult {
  output: string
  sessionId: string
  threadId?: string
  backend: ExternalCliBackend
  permission: ExternalCliPermission
  status: 'completed' | 'failed'
  usage: { inputTokens: number; outputTokens: number }
  durationMs: number
  error?: string
}

export interface ExternalCliDispatcherDeps {
  sessions: ISessionStore
  deviceName: string
  usageTracker?: UsageTracker
  run?: typeof runExternalCli
  runOptions?: Pick<ExternalCliRunOptions, 'spawnFn' | 'log'>
  env?: NodeJS.ProcessEnv
}

export class ExternalCliDispatcher {
  constructor(private readonly deps: ExternalCliDispatcherDeps) {}

  async dispatch(input: ExternalCliDispatchInput): Promise<ExternalCliDispatchResult> {
    const prompt = input.prompt.trim()
    if (!prompt) throw new Error('EXTERNAL_CLI_INVALID_REQUEST: prompt is required')
    const cwd = resolve(input.cwd ?? process.cwd())
    const permission = capPermission(capPermission(permissionForAutonomy(input.autonomy), externalCliPermissionCeiling(this.deps.env)), input.maxPermission)
    const sessionId = input.sessionId ?? randomUUID()
    const createdAt = new Date().toISOString()
    const startedAt = Date.now()
    await this.deps.sessions.create({
      id: sessionId,
      title: `${input.backend}: ${prompt.slice(0, 40)}`,
      createdAt,
      updatedAt: createdAt,
      provider: 'external-cli',
      model: input.model ? `${input.backend}/${input.model}` : input.backend,
      device: this.deps.deviceName,
      status: 'active',
      cwd,
      tags: [
        'external-cli',
        `cli:${input.backend}`,
        `permission:${permission}`,
        ...(input.parentSessionId ? [`parent:${input.parentSessionId}`] : []),
      ],
    })
    if (input.runContract) {
      await this.deps.sessions.appendEvent(sessionId, {
        type: 'run_contract', id: randomUUID(), timestamp: new Date().toISOString(), contract: input.runContract,
      })
    }
    const message = formatPrompt(prompt, input.runContract)
    await this.deps.sessions.appendEvent(sessionId, {
      type: 'user_message', id: randomUUID(), timestamp: new Date().toISOString(), content: message,
    })

    const run = this.deps.run ?? runExternalCli
    let output = ''
    let threadId = input.threadId
    let usage = { inputTokens: 0, outputTokens: 0 }
    let status: ExternalCliDispatchResult['status'] = 'completed'
    let error: string | undefined
    const partial: string[] = []
    try {
      const result = await run(
        { backend: input.backend, message, cwd, permission, model: input.model, threadId: input.threadId, timeoutMs: input.timeoutMs },
        {
          signal: input.signal,
          ...this.deps.runOptions,
          onEvent: (event) => {
            if (event.type === 'text') partial.push(event.text)
            input.onEvent?.(event)
          },
        },
      )
      output = result.output || `[${input.backend} returned no text]`
      threadId = result.threadId
      usage = result.usage
    } catch (caught) {
      status = 'failed'
      error = caught instanceof Error ? caught.message : String(caught)
      const soFar = partial.join('').trim()
      output = soFar ? `${soFar}\n\n[${input.backend} error] ${error}` : `[${input.backend} error] ${error}`
    }

    const durationMs = Date.now() - startedAt
    await this.deps.sessions.appendEvent(sessionId, {
      type: 'assistant_message', id: randomUUID(), timestamp: new Date().toISOString(), content: output,
    })
    await this.deps.sessions.appendEvent(sessionId, {
      type: 'session_end', id: randomUUID(), timestamp: new Date().toISOString(),
      totalTokens: { input: usage.inputTokens, output: usage.outputTokens }, totalCost: 0, duration_ms: durationMs,
    })
    if (usage.inputTokens || usage.outputTokens) {
      this.deps.usageTracker?.record({
        sessionId: input.parentSessionId ?? sessionId,
        provider: 'external-cli',
        model: input.model ? `${input.backend}/${input.model}` : input.backend,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        costKnown: false,
        durationMs,
        toolName: 'external_cli.run',
      })
    }
    return { output, sessionId, threadId, backend: input.backend, permission, status, usage, durationMs, error }
  }
}

function formatPrompt(prompt: string, runContract: AgentRunContract | undefined): string {
  const contract = formatRunContractForPrompt(runContract)
  return contract ? [contract, '[Task]', prompt].join('\n\n') : prompt
}
