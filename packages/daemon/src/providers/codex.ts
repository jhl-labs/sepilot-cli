import { execFile as defaultExecFile } from 'node:child_process'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type {
  ChatRequest,
  ChatResponse,
  ILLMProvider,
  LLMRequestOptions,
  Message,
  ModelCatalogAuthority,
  ModelCatalogRefreshResult,
  ModelInfo,
  StreamChunk,
  TokenUsage,
} from '@sepilotd/core'
import { AcpStdioClient, type AcpClientSpec } from '../acp/client.js'
import type { JsonRpcMessage } from '../acp/rpc.js'
import { extractContent, toApiError } from './utils.js'

const DEFAULT_CONTEXT_WINDOW = 192_000
const DEFAULT_MAX_OUTPUT_TOKENS = 128_000
const DEFAULT_TIMEOUT_MS = 120_000
const MODEL_LIST_TIMEOUT_MS = 15_000

interface CodexAppServerClient {
  start(): void
  stop(): void
  request(method: string, params?: unknown, timeoutMs?: number): Promise<unknown>
  notify(method: string, params?: unknown): void
}

export interface CodexCatalogModel {
  id: string
  name: string
  contextWindow: number
  maxOutputTokens: number
  vision: boolean
}

interface CodexProviderOptions {
  models?: string[]
  defaultContextWindow?: number
  defaultMaxOutputTokens?: number
  env?: NodeJS.ProcessEnv
  command?: string
  clientFactory?: (
    spec: AcpClientSpec,
    options: {
      timeoutMs: number
      onNotification: (message: JsonRpcMessage) => void
      onClientRequest: (message: JsonRpcMessage) => unknown
    },
  ) => CodexAppServerClient
  modelLister?: () => Promise<CodexCatalogModel[]>
}

interface RawCodexCatalogModel {
  slug?: unknown
  display_name?: unknown
  visibility?: unknown
  context_window?: unknown
  max_context_window?: unknown
  max_output_tokens?: unknown
  input_modalities?: unknown
}

function codexCommand(env: NodeJS.ProcessEnv): string {
  return env.SEPILOTD_CODEX_COMMAND?.trim() || 'codex'
}

function positiveInteger(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0
    ? value
    : fallback
}

function uniqueModels(models: string[]): string[] {
  return Array.from(new Set(models.map((model) => model.trim()).filter(Boolean))).sort()
}

function toModelInfo(model: CodexCatalogModel): ModelInfo {
  return {
    id: model.id,
    name: model.name,
    contextWindow: model.contextWindow,
    maxOutputTokens: Math.min(model.maxOutputTokens, model.contextWindow),
    capabilities: {
      vision: model.vision,
      toolUse: false,
      streaming: false,
      embedding: false,
      thinking: true,
    },
  }
}

function configuredModel(
  id: string,
  contextWindow: number,
  maxOutputTokens: number,
): CodexCatalogModel {
  return {
    id,
    name: id === 'default' ? 'Codex default' : id,
    contextWindow,
    maxOutputTokens,
    vision: false,
  }
}

function messageForPrompt(message: Message): string {
  const content = extractContent(message)
  const toolCalls = message.toolCalls?.length
    ? `\nTool calls: ${JSON.stringify(message.toolCalls)}`
    : ''
  const toolName = message.role === 'tool'
    ? ` ${message.name ?? message.toolCallId ?? 'unknown'}`
    : ''
  return `[${message.role}${toolName}]\n${content}${toolCalls}`
}

export function formatCodexProviderPrompt(request: ChatRequest): string {
  return [
    '[Transport boundary]',
    'Act only as the language model for the transcript below. Do not use tools, read files, run commands, browse, access MCP servers, delegate work, or modify state. Return only the next assistant response based on the supplied transcript.',
    request.systemPrompt ? `[system]\n${request.systemPrompt}` : '',
    ...request.messages.map(messageForPrompt),
    '[assistant]',
  ].filter(Boolean).join('\n\n')
}

function parseThreadId(result: unknown): string {
  const thread = result && typeof result === 'object'
    ? (result as { thread?: unknown }).thread
    : undefined
  const id = thread && typeof thread === 'object'
    ? (thread as { id?: unknown }).id
    : undefined
  if (typeof id !== 'string' || id.length === 0) {
    throw new Error('Codex app-server thread/start did not return a thread id')
  }
  return id
}

function parseTurnId(result: unknown): string {
  const turn = result && typeof result === 'object'
    ? (result as { turn?: unknown }).turn
    : undefined
  const id = turn && typeof turn === 'object'
    ? (turn as { id?: unknown }).id
    : undefined
  if (typeof id !== 'string' || id.length === 0) {
    throw new Error('Codex app-server turn/start did not return a turn id')
  }
  return id
}

function notificationParams(message: JsonRpcMessage): Record<string, unknown> | null {
  return message.params && typeof message.params === 'object'
    ? message.params as Record<string, unknown>
    : null
}

function tokenUsageFromNotification(message: JsonRpcMessage): TokenUsage | null {
  if (message.method !== 'thread/tokenUsage/updated') return null
  const params = notificationParams(message)
  const tokenUsage = params?.tokenUsage
  const last = tokenUsage && typeof tokenUsage === 'object'
    ? (tokenUsage as { last?: unknown }).last
    : undefined
  if (!last || typeof last !== 'object') return null
  const usage = last as Record<string, unknown>
  return {
    inputTokens: positiveInteger(usage.inputTokens, 0),
    outputTokens: positiveInteger(usage.outputTokens, 0),
    thinkingTokens: positiveInteger(usage.reasoningOutputTokens, 0),
    cacheReadTokens: positiveInteger(usage.cachedInputTokens, 0),
    cacheCreationTokens: positiveInteger(usage.cacheWriteInputTokens, 0),
  }
}

function completedAgentMessage(message: JsonRpcMessage): { text: string; final: boolean } | null {
  if (message.method !== 'item/completed') return null
  const item = notificationParams(message)?.item
  if (!item || typeof item !== 'object') return null
  const record = item as Record<string, unknown>
  if (record.type !== 'agentMessage' || typeof record.text !== 'string') return null
  return { text: record.text, final: record.phase === 'final_answer' }
}

function toolItemType(message: JsonRpcMessage): string | null {
  if (message.method !== 'item/started') return null
  const item = notificationParams(message)?.item
  if (!item || typeof item !== 'object') return null
  const type = (item as { type?: unknown }).type
  if (typeof type !== 'string') return null
  return new Set([
    'commandExecution',
    'fileChange',
    'mcpToolCall',
    'dynamicToolCall',
    'collabToolCall',
    'webSearch',
  ]).has(type) ? type : null
}

function denyCodexClientRequest(message: JsonRpcMessage): unknown {
  if (message.method === 'item/commandExecution/requestApproval') return { decision: 'cancel' }
  if (message.method === 'item/fileChange/requestApproval') return { decision: 'cancel' }
  if (message.method === 'item/permissions/requestApproval') return { permissions: {} }
  if (message.method === 'tool/requestUserInput') return { answers: {} }
  if (message.method === 'mcpServer/elicitation/request') return { action: 'cancel', content: null }
  const error = new Error(`method not found: ${String(message.method)}`) as Error & { code?: number }
  error.code = -32601
  throw error
}

function turnResult(message: JsonRpcMessage): { status: string; error?: string } | null {
  if (message.method !== 'turn/completed') return null
  const turn = notificationParams(message)?.turn
  if (!turn || typeof turn !== 'object') return null
  const record = turn as Record<string, unknown>
  const error = record.error && typeof record.error === 'object'
    ? (record.error as { message?: unknown }).message
    : undefined
  return {
    status: typeof record.status === 'string' ? record.status : 'failed',
    ...(typeof error === 'string' ? { error } : {}),
  }
}

export class CodexProvider implements ILLMProvider {
  readonly id = 'codex'
  readonly name = 'Codex'
  readonly models: ModelInfo[]
  private readonly env: NodeJS.ProcessEnv
  private readonly command: string
  private readonly clientFactory?: CodexProviderOptions['clientFactory']
  private readonly modelLister?: CodexProviderOptions['modelLister']
  private readonly contextWindow: number
  private readonly maxOutputTokens: number
  private catalogAuthority: ModelCatalogAuthority = 'configured'

  constructor(options: CodexProviderOptions = {}) {
    this.env = options.env ?? process.env
    this.command = options.command ?? codexCommand(this.env)
    this.clientFactory = options.clientFactory
    this.modelLister = options.modelLister
    this.contextWindow = options.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW
    this.maxOutputTokens = options.defaultMaxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS
    const configured = uniqueModels(options.models ?? [])
    this.models = (configured.length > 0 ? configured : ['default'])
      .map((id) => toModelInfo(configuredModel(id, this.contextWindow, this.maxOutputTokens)))
  }

  get modelCatalogAuthority(): ModelCatalogAuthority {
    return this.catalogAuthority
  }

  static async readModelCatalog(
    options: { command?: string; env?: NodeJS.ProcessEnv } = {},
  ): Promise<CodexCatalogModel[]> {
    const env = options.env ?? process.env
    const command = options.command ?? codexCommand(env)
    const result = await new Promise<{ stdout: string }>((resolve, reject) => {
      defaultExecFile(command, ['debug', 'models'], {
        env,
        encoding: 'utf8',
        timeout: MODEL_LIST_TIMEOUT_MS,
        maxBuffer: 16 * 1024 * 1024,
      }, (error, stdout) => {
        if (error) reject(error)
        else resolve({ stdout })
      })
    })
    const parsed = JSON.parse(result.stdout) as { models?: unknown }
    if (!Array.isArray(parsed.models)) throw new Error('Codex did not return a model catalog')
    const models = parsed.models.flatMap((value): CodexCatalogModel[] => {
      if (!value || typeof value !== 'object') return []
      const model = value as RawCodexCatalogModel
      if (typeof model.slug !== 'string' || model.visibility !== 'list') return []
      const contextWindow = positiveInteger(
        model.context_window,
        positiveInteger(model.max_context_window, DEFAULT_CONTEXT_WINDOW),
      )
      return [{
        id: model.slug,
        name: typeof model.display_name === 'string' ? model.display_name : model.slug,
        contextWindow,
        maxOutputTokens: positiveInteger(model.max_output_tokens, DEFAULT_MAX_OUTPUT_TOKENS),
        vision: Array.isArray(model.input_modalities) && model.input_modalities.includes('image'),
      }]
    })
    if (models.length === 0) throw new Error('Codex did not advertise any selectable models')
    return models
  }

  static async listAvailableModels(): Promise<string[]> {
    return (await CodexProvider.readModelCatalog()).map((model) => model.id)
  }

  async refreshModelCatalog(): Promise<ModelCatalogRefreshResult> {
    const discovered = await (
      this.modelLister?.() ?? CodexProvider.readModelCatalog({
        command: this.command,
        env: this.env,
      })
    )
    const catalog = [configuredModel('default', this.contextWindow, this.maxOutputTokens), ...discovered]
    const deduplicated = Array.from(new Map(catalog.map((model) => [model.id, model])).values())
    const previousIds = this.models.map((model) => model.id)
    const modelIds = deduplicated.map((model) => model.id)
    const previousSet = new Set(previousIds)
    const nextSet = new Set(modelIds)
    const added = modelIds.filter((id) => !previousSet.has(id))
    const removed = previousIds.filter((id) => !nextSet.has(id))
    this.models.splice(0, this.models.length, ...deduplicated.map(toModelInfo))
    this.catalogAuthority = 'endpoint'
    return { modelIds, added, removed }
  }

  async chat(request: ChatRequest, options?: LLMRequestOptions): Promise<ChatResponse> {
    const timeoutMs = Math.max(1, request.timeoutMs ?? DEFAULT_TIMEOUT_MS)
    const workspace = await mkdtemp(join(tmpdir(), 'sepilot-codex-provider-'))
    let threadId = ''
    let turnId = ''
    let finalText = ''
    let fallbackText = ''
    let usage: TokenUsage = { inputTokens: 0, outputTokens: 0 }
    let toolViolation = ''
    let completeTurn!: (result: { status: string; error?: string }) => void
    const completion = new Promise<{ status: string; error?: string }>((resolve) => {
      completeTurn = resolve
    })
    const client = this.createClient({
      command: this.command,
      args: ['app-server', '--listen', 'stdio://'],
      cwd: workspace,
      env: this.env,
      framing: 'ndjson',
    }, {
      timeoutMs,
      onClientRequest: denyCodexClientRequest,
      onNotification: (message) => {
        const completed = completedAgentMessage(message)
        if (completed) {
          fallbackText = completed.text
          if (completed.final) finalText = completed.text
        }
        const nextUsage = tokenUsageFromNotification(message)
        if (nextUsage) usage = nextUsage
        const forbiddenTool = toolItemType(message)
        if (forbiddenTool && !toolViolation) {
          toolViolation = forbiddenTool
          if (threadId && turnId) {
            void client.request('turn/interrupt', { threadId, turnId }, timeoutMs).catch(() => {})
          }
        }
        const result = turnResult(message)
        if (result) completeTurn(result)
      },
    })
    const abort = () => {
      if (threadId && turnId) {
        void client.request('turn/interrupt', { threadId, turnId }, timeoutMs).catch(() => {})
      }
      client.stop()
    }
    let completionTimer: ReturnType<typeof setTimeout> | undefined

    try {
      options?.signal?.throwIfAborted()
      options?.signal?.addEventListener('abort', abort, { once: true })
      client.start()
      await client.request('initialize', {
        clientInfo: { name: 'sepilotd', title: 'sepilotd', version: '0.0.0' },
      }, timeoutMs)
      client.notify('initialized', {})
      threadId = parseThreadId(await client.request('thread/start', {
        ...(request.model === 'default' ? {} : { model: request.model }),
        cwd: workspace,
        approvalPolicy: 'never',
        sandbox: 'read-only',
        ephemeral: true,
        serviceName: 'sepilotd-provider',
        developerInstructions: 'Act only as a language model transport. Never use tools, commands, files, browsing, MCP servers, plugins, skills, or subagents. Produce only the requested assistant response.',
      }, timeoutMs))
      turnId = parseTurnId(await client.request('turn/start', {
        threadId,
        input: [{ type: 'text', text: formatCodexProviderPrompt(request) }],
        ...(request.model === 'default' ? {} : { model: request.model }),
        approvalPolicy: 'never',
        sandboxPolicy: {
          type: 'readOnly',
        },
      }, timeoutMs))
      const timedCompletion = new Promise<never>((_, reject) => {
        completionTimer = setTimeout(
          () => reject(new Error(`Codex turn timed out after ${timeoutMs}ms`)),
          timeoutMs,
        )
      })
      const result = await Promise.race([completion, timedCompletion])
      if (toolViolation) {
        throw new Error(`Codex provider attempted to use a forbidden tool: ${toolViolation}`)
      }
      if (result.status !== 'completed') {
        throw new Error(result.error ?? `Codex turn ended with status: ${result.status}`)
      }
      const content = (finalText || fallbackText).trim()
      if (!content) throw new Error('Codex app-server returned an empty assistant response')
      return {
        message: { role: 'assistant', content },
        usage,
        finishReason: 'stop',
      }
    } finally {
      if (completionTimer) clearTimeout(completionTimer)
      options?.signal?.removeEventListener('abort', abort)
      client.stop()
      await rm(workspace, { recursive: true, force: true, maxRetries: 3 })
    }
  }

  async *stream(request: ChatRequest, options?: LLMRequestOptions): AsyncIterable<StreamChunk> {
    try {
      const response = await this.chat(request, options)
      const content = extractContent(response.message)
      if (content) yield { type: 'text', text: content }
      yield { type: 'usage', usage: response.usage }
      yield { type: 'done', finishReason: response.finishReason }
    } catch (error) {
      yield { type: 'error', error: toApiError(error) }
    }
  }

  private createClient(
    spec: AcpClientSpec,
    options: {
      timeoutMs: number
      onNotification: (message: JsonRpcMessage) => void
      onClientRequest: (message: JsonRpcMessage) => unknown
    },
  ): CodexAppServerClient {
    return this.clientFactory
      ? this.clientFactory(spec, options)
      : new AcpStdioClient(spec, {
          requestTimeoutMs: options.timeoutMs,
          onNotification: options.onNotification,
          onClientRequest: options.onClientRequest,
        })
  }
}
