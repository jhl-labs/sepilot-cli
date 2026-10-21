import { execFile as defaultExecFile } from 'node:child_process'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type {
  ContentPart,
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
import { extractContent, ProviderError, toApiError } from './utils.js'
import { codexToolOutputSchema, parseCodexToolResponse, withCodexToolInstructions } from './codex-tool-transport.js'

const DEFAULT_CONTEXT_WINDOW = 192_000
const DEFAULT_MAX_OUTPUT_TOKENS = 128_000
const DEFAULT_TIMEOUT_MS = 120_000
const MODEL_LIST_TIMEOUT_MS = 15_000

class CodexHostProtocolError extends Error {
  constructor(message: string, readonly usage: TokenUsage) {
    super(message)
    this.name = 'CodexHostProtocolError'
  }
}

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
  structuredToolCalling?: boolean
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

function toModelInfo(model: CodexCatalogModel, structuredToolCalling = false): ModelInfo {
  return {
    id: model.id,
    name: model.name,
    contextWindow: model.contextWindow,
    maxOutputTokens: Math.min(model.maxOutputTokens, model.contextWindow),
    capabilities: {
      vision: model.vision,
      toolUse: structuredToolCalling,
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

function messageForPrompt(message: Message, nextImage: () => number): string {
  const content = typeof message.content === 'string' ? message.content : message.content.map(part => part.type === 'text' ? part.text : part.type === 'image' ? `[Image attachment ${nextImage()}]` : '').join('\n')
  const toolCalls = message.toolCalls?.length
    ? `\nTool calls: ${JSON.stringify(message.toolCalls)}`
    : ''
  const toolName = message.role === 'tool'
    ? ` ${message.name ?? message.toolCallId ?? 'unknown'}`
    : ''
  return `[${message.role}${toolName}]\n${content}${toolCalls}`
}

const CODEX_DISABLED_BUILTIN_TOOL_FEATURES = [
  'shell_tool',
  'unified_exec',
  'code_mode_host',
  'apps',
  'plugins',
  'remote_plugin',
  'browser_use',
  'browser_use_external',
  'in_app_browser',
  'computer_use',
  'image_generation',
  'view_image',
  'multi_agent',
  'multi_agent_v2',
  'skill_search',
  'tool_suggest',
  'sleep_tool',
  'goals',
  'memories',
] as const

export const CODEX_APP_SERVER_ARGS: readonly string[] = [
  'app-server',
  '--listen',
  'stdio://',
  ...CODEX_DISABLED_BUILTIN_TOOL_FEATURES.flatMap((feature) => ['-c', `features.${feature}=false`]),
  '-c',
  'web_search="disabled"',
]

export const CODEX_TRANSPORT_BOUNDARY = [
  "You are the language model behind a host application. Codex's own execution environment is disabled for this session: do not invoke Codex's shell, code mode, file, patch, web, MCP, plugin, skill, or subagent tools. The Codex sandbox and approval settings apply only to those disabled Codex tools.",
  "If a Codex tool answers that it is disabled or unavailable, that is expected and says nothing about the host's tools or environment: never report it as a host failure or blocker. Do that work — a shell or git command included — with the host tool that runs commands.",
  'The host application has its own tools and executes them itself, under its own permissions. When the transcript defines a tool-call format, every tool it lists is available to you, including tools that read, create, or edit files. To use one, write the call in exactly that format as your response. That is ordinary response text, not Codex tool use, and the host returns the result in a later message. Do not claim that a listed host tool is unavailable.',
  'When the transcript defines no tool-call format, answer in text.',
  'Return only the next assistant response for the transcript below.',
].join(' ')

const CODEX_DEVELOPER_INSTRUCTIONS = [
  "Codex's own tools (shell commands, code mode, file access, patches, browsing, MCP servers, plugins, skills, subagents) are disabled and must not be invoked. The read-only sandbox applies only to them. A disabled-tool answer from Codex is expected and is not a host failure; run commands through the host's tools instead.",
  'The user message is a transcript from a host application that executes its own tools. When it defines a tool-call format, request the tools it lists, file writes included, by writing calls in that format as your response text; that is not Codex tool use. Otherwise answer in text.',
].join(' ')

export function formatCodexProviderPrompt(request: ChatRequest): string {
  let imageNumber = 0
  return [
    '[Transport boundary]',
    CODEX_TRANSPORT_BOUNDARY,
    request.systemPrompt ? `[system]\n${request.systemPrompt}` : '',
    ...request.messages.map(message => messageForPrompt(message, () => ++imageNumber)),
    '[assistant]',
  ].filter(Boolean).join('\n\n')
}

export function formatCodexProviderInput(request: ChatRequest): Array<{ type: 'text'; text: string } | { type: 'image'; url: string }> {
  const images = request.messages.flatMap(message => typeof message.content === 'string' ? [] : message.content.filter((part): part is Extract<ContentPart, { type: 'image' }> => part.type === 'image'))
  if (images.length > 16) throw new Error('Codex image input exceeds the 16 attachment limit')
  const input: Array<{ type: 'text'; text: string } | { type: 'image'; url: string }> = [{ type: 'text', text: formatCodexProviderPrompt(request) }]
  let bytes = 0
  for (const image of images) {
    const source = image.source
    const url = source.type === 'base64' ? `data:${source.mediaType};base64,${source.data}` : source.data
    if (!/^data:image\/[a-z0-9.+-]+;base64,[A-Za-z0-9+/]+={0,2}$/i.test(url)) {
      const remote = new URL(url)
      if (!['https:', 'http:'].includes(remote.protocol) || remote.username || remote.password) throw new Error('Codex image input requires an image data URL or credential-free HTTP(S) URL')
    }
    bytes += Buffer.byteLength(url)
    if (bytes > 20 * 1024 * 1024) throw new Error('Codex image input exceeds the 20 MiB encoded payload limit')
    input.push({ type: 'image', url })
  }
  return input
}

function assertNoNativeToolCalls(request: ChatRequest): void {
  if (!request.tools?.length || request.toolChoice === 'none') return
  throw new ProviderError({
    code: 'INVALID_REQUEST',
    message: `Codex provider does not support native tool calling (${request.tools.length} tool definition(s) received for model ${request.model}); use the prompt-react tool transport`,
  })
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
  private readonly structuredToolCalling: boolean
  private catalogAuthority: ModelCatalogAuthority = 'configured'

  constructor(options: CodexProviderOptions = {}) {
    this.env = options.env ?? process.env
    this.command = options.command ?? codexCommand(this.env)
    this.clientFactory = options.clientFactory
    this.modelLister = options.modelLister
    this.contextWindow = options.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW
    this.maxOutputTokens = options.defaultMaxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS
    this.structuredToolCalling = options.structuredToolCalling ?? this.env.SEPILOTD_CODEX_STRUCTURED_TOOLS === '1'
    const configured = uniqueModels(options.models ?? [])
    this.models = (configured.length > 0 ? configured : ['default'])
      .map((id) => toModelInfo(configuredModel(id, this.contextWindow, this.maxOutputTokens), this.structuredToolCalling))
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
    this.models.splice(0, this.models.length, ...deduplicated.map(model => toModelInfo(model, this.structuredToolCalling)))
    this.catalogAuthority = 'endpoint'
    return { modelIds, added, removed }
  }

  async chat(request: ChatRequest, options?: LLMRequestOptions): Promise<ChatResponse> {
    try {
      return await this.chatOnce(request, options)
    } catch (error) {
      if (!(error instanceof CodexHostProtocolError) || options?.signal?.aborted) throw error
      const retry = await this.chatOnce({
        ...request,
        systemPrompt: [request.systemPrompt, `[Host response correction] The previous response was rejected by the host: ${error.message}. Respond again with exactly one JSON object with toolCalls and answer; every argumentsJson must be a valid JSON object string.`].filter(Boolean).join('\n'),
      }, options)
      return { ...retry, usage: { ...retry.usage, inputTokens: retry.usage.inputTokens + error.usage.inputTokens, outputTokens: retry.usage.outputTokens + error.usage.outputTokens } }
    }
  }

  private async chatOnce(request: ChatRequest, options?: LLMRequestOptions): Promise<ChatResponse> {
    if (!this.structuredToolCalling) assertNoNativeToolCalls(request)
    const structured = this.structuredToolCalling && Boolean(request.tools?.length) && request.toolChoice !== 'none'
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
      args: [...CODEX_APP_SERVER_ARGS],
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
      completeTurn({ status: 'cancelled' })
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
        baseInstructions: CODEX_TRANSPORT_BOUNDARY,
        developerInstructions: CODEX_DEVELOPER_INSTRUCTIONS,
      }, timeoutMs))
      turnId = parseTurnId(await client.request('turn/start', {
        threadId,
        input: formatCodexProviderInput(structured ? withCodexToolInstructions(request) : request),
        ...(structured ? { outputSchema: codexToolOutputSchema(request) } : {}),
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
      options?.signal?.throwIfAborted()
      if (toolViolation) {
        throw new Error(`Codex provider attempted to use a forbidden tool: ${toolViolation}`)
      }
      if (result.status !== 'completed') {
        throw new Error(result.error ?? `Codex turn ended with status: ${result.status}`)
      }
      const content = (finalText || fallbackText).trim()
      if (!content) throw new Error('Codex app-server returned an empty assistant response')
      let message: Message
      try {
        message = structured ? parseCodexToolResponse(content, request) : { role: 'assistant' as const, content }
      } catch (error) {
        throw new CodexHostProtocolError((error as Error).message, usage)
      }
      return {
        message,
        usage,
        finishReason: message.toolCalls?.length ? 'tool_use' : 'stop',
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
      for (const toolCall of response.message.toolCalls ?? []) {
        yield { type: 'tool_call_start', toolCall: { id: toolCall.id, name: toolCall.name } }
        yield { type: 'tool_call_delta', toolCallId: toolCall.id, delta: JSON.stringify(toolCall.arguments) }
        yield { type: 'tool_call_end', toolCallId: toolCall.id }
      }
      yield { type: 'usage', usage: response.usage }
      yield { type: 'done', finishReason: response.finishReason }
    } catch (error) {
      yield { type: 'error', error: error instanceof ProviderError ? error.apiError : toApiError(error) }
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
