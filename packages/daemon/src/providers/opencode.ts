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
} from '@sepilotd/core'
import { AcpStdioClient, type AcpClientSpec } from '../acp/client.js'
import type { JsonRpcMessage } from '../acp/rpc.js'
import { extractContent, toApiError } from './utils.js'

const DEFAULT_CONTEXT_WINDOW = 192_000
const DEFAULT_MAX_OUTPUT_TOKENS = 128_000
const DEFAULT_TIMEOUT_MS = 120_000
const MODEL_LIST_TIMEOUT_MS = 10_000

interface OpencodeAcpClient {
  start(): void
  stop(): void
  initialize(): Promise<unknown>
  request(method: string, params?: unknown, timeoutMs?: number): Promise<unknown>
  prompt(sessionId: string, prompt: string): Promise<unknown>
  cancel(sessionId: string): void
}

interface OpencodeProviderOptions {
  models?: string[]
  defaultContextWindow?: number
  defaultMaxOutputTokens?: number
  env?: NodeJS.ProcessEnv
  command?: string
  clientFactory?: (
    spec: AcpClientSpec,
    options: { timeoutMs: number; onNotification: (message: JsonRpcMessage) => void },
  ) => OpencodeAcpClient
  modelLister?: () => Promise<string[]>
}

interface AcpConfigOption {
  id?: unknown
  category?: unknown
  currentValue?: unknown
  options?: Array<{ value?: unknown }>
}

function parseEnvArgs(raw: string | undefined): string[] {
  return raw?.trim().split(/\s+/).filter(Boolean) ?? []
}

function opencodeCommand(env: NodeJS.ProcessEnv): string {
  return env.SEPILOTD_OPENCODE_COMMAND?.trim() || 'opencode'
}

function opencodeAcpArgs(env: NodeJS.ProcessEnv): string[] {
  const configured = parseEnvArgs(env.SEPILOTD_OPENCODE_ACP_ARGS)
  return configured.length > 0 ? configured : ['--pure', 'acp']
}

function opencodeProviderEnv(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  return {
    ...env,
    OPENCODE_PERMISSION: JSON.stringify({ '*': 'deny' }),
    OPENCODE_DISABLE_DEFAULT_PLUGINS: 'true',
    OPENCODE_DISABLE_CLAUDE_CODE: 'true',
    OPENCODE_DISABLE_LSP_DOWNLOAD: 'true',
  }
}

function modelInfo(
  id: string,
  contextWindow = DEFAULT_CONTEXT_WINDOW,
  maxOutputTokens = DEFAULT_MAX_OUTPUT_TOKENS,
): ModelInfo {
  return {
    id,
    name: id === 'default' ? 'OpenCode default' : id,
    contextWindow,
    maxOutputTokens: Math.min(maxOutputTokens, contextWindow),
    capabilities: {
      vision: false,
      toolUse: false,
      streaming: false,
      embedding: false,
      thinking: true,
    },
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

export function formatOpencodeProviderPrompt(request: ChatRequest): string {
  return [
    '[Transport boundary]',
    'Act only as the language model for the transcript below. Do not use OpenCode tools, read files, run commands, access the network, or modify any state. Return only the next assistant response based on the supplied transcript.',
    request.systemPrompt ? `[system]\n${request.systemPrompt}` : '',
    ...request.messages.map(messageForPrompt),
    '[assistant]',
  ].filter(Boolean).join('\n\n')
}

function extractAcpText(value: unknown): string {
  if (typeof value === 'string') return value
  if (!value || typeof value !== 'object') return ''
  const record = value as Record<string, unknown>
  if (typeof record.text === 'string') return record.text
  if (typeof record.content === 'string') return record.content
  if (Array.isArray(record.content)) return record.content.map(extractAcpText).join('')
  return ''
}

function notificationText(message: JsonRpcMessage, expectedType: string): string {
  if (message.method !== 'session/update') return ''
  const params = message.params
  if (!params || typeof params !== 'object') return ''
  const update = (params as { update?: unknown }).update
  if (!update || typeof update !== 'object') return ''
  const record = update as Record<string, unknown>
  if (record.sessionUpdate !== expectedType) return ''
  return extractAcpText(record.content)
}

function parseSession(result: unknown): { sessionId: string; configOptions: AcpConfigOption[] } {
  if (!result || typeof result !== 'object') {
    throw new Error('OpenCode ACP session/new returned a non-object result')
  }
  const record = result as { sessionId?: unknown; configOptions?: unknown }
  if (typeof record.sessionId !== 'string' || record.sessionId.trim().length === 0) {
    throw new Error('OpenCode ACP session/new did not return sessionId')
  }
  return {
    sessionId: record.sessionId,
    configOptions: Array.isArray(record.configOptions)
      ? record.configOptions.filter((value): value is AcpConfigOption => (
          Boolean(value) && typeof value === 'object' && !Array.isArray(value)
        ))
      : [],
  }
}

function modelConfigOption(options: AcpConfigOption[]): AcpConfigOption | undefined {
  return options.find((option) => option.category === 'model')
    ?? options.find((option) => option.id === 'model')
}

function mapStopReason(result: unknown): ChatResponse['finishReason'] {
  const reason = result && typeof result === 'object'
    ? (result as { stopReason?: unknown }).stopReason
    : undefined
  if (reason === 'max_tokens' || reason === 'length') return 'length'
  if (reason === 'content_filter') return 'content_filter'
  return 'stop'
}

function uniqueModels(models: string[]): string[] {
  return Array.from(new Set(models.map((model) => model.trim()).filter(Boolean))).sort()
}

export class OpencodeProvider implements ILLMProvider {
  readonly id = 'opencode'
  readonly name = 'Opencode'
  readonly models: ModelInfo[]
  private readonly env: NodeJS.ProcessEnv
  private readonly command: string
  private readonly clientFactory?: OpencodeProviderOptions['clientFactory']
  private readonly modelLister?: OpencodeProviderOptions['modelLister']
  private readonly contextWindow: number
  private readonly maxOutputTokens: number
  private catalogAuthority: ModelCatalogAuthority = 'configured'

  constructor(options: OpencodeProviderOptions = {}) {
    this.env = options.env ?? process.env
    this.command = options.command ?? opencodeCommand(this.env)
    this.clientFactory = options.clientFactory
    this.modelLister = options.modelLister
    this.contextWindow = options.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW
    this.maxOutputTokens = options.defaultMaxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS
    const configuredModels = uniqueModels(options.models ?? [])
    this.models = (configuredModels.length > 0 ? configuredModels : ['default'])
      .map((id) => modelInfo(id, this.contextWindow, this.maxOutputTokens))
  }

  get modelCatalogAuthority(): ModelCatalogAuthority {
    return this.catalogAuthority
  }

  static async listAvailableModels(
    options: { command?: string; env?: NodeJS.ProcessEnv } = {},
  ): Promise<string[]> {
    const env = options.env ?? process.env
    const command = options.command ?? opencodeCommand(env)
    const result = await new Promise<{ stdout: string }>((resolve, reject) => {
      defaultExecFile(command, ['models'], {
        env,
        encoding: 'utf8',
        timeout: MODEL_LIST_TIMEOUT_MS,
        maxBuffer: 4 * 1024 * 1024,
      }, (error, stdout) => {
        if (error) reject(error)
        else resolve({ stdout })
      })
    })
    const models = uniqueModels(
      result.stdout
        .replace(/\u001B\[[0-?]*[ -/]*[@-~]/g, '')
        .split(/\r?\n/),
    )
    if (models.length === 0) {
      throw new Error('OpenCode did not report any configured models')
    }
    return models
  }

  async refreshModelCatalog(): Promise<ModelCatalogRefreshResult> {
    const discovered = uniqueModels(await (
      this.modelLister?.() ?? OpencodeProvider.listAvailableModels({
        command: this.command,
        env: this.env,
      })
    ))
    const previousIds = this.models.map((model) => model.id)
    const modelIds = discovered.includes('default') ? discovered : ['default', ...discovered]
    const nextSet = new Set(modelIds)
    const previousSet = new Set(previousIds)
    const added = modelIds.filter((id) => !previousSet.has(id))
    const removed = previousIds.filter((id) => !nextSet.has(id))
    const previousById = new Map(this.models.map((model) => [model.id, model] as const))
    this.models.splice(0, this.models.length, ...modelIds.map((id) => (
      previousById.get(id) ?? modelInfo(id, this.contextWindow, this.maxOutputTokens)
    )))
    this.catalogAuthority = 'endpoint'
    return { modelIds, added, removed }
  }

  async chat(request: ChatRequest, options?: LLMRequestOptions): Promise<ChatResponse> {
    const timeoutMs = Math.max(1, request.timeoutMs ?? DEFAULT_TIMEOUT_MS)
    const workspace = await mkdtemp(join(tmpdir(), 'sepilot-opencode-provider-'))
    const assistantText: string[] = []
    const thinkingText: string[] = []
    let sessionId = ''
    const client = this.createClient({
      command: this.command,
      args: opencodeAcpArgs(this.env),
      cwd: workspace,
      env: opencodeProviderEnv(this.env),
      framing: 'ndjson',
    }, {
      timeoutMs,
      onNotification: (message) => {
        const text = notificationText(message, 'agent_message_chunk')
        if (text) assistantText.push(text)
        const thought = notificationText(message, 'agent_thought_chunk')
        if (thought) thinkingText.push(thought)
      },
    })
    const abort = () => {
      if (sessionId) client.cancel(sessionId)
      client.stop()
    }

    try {
      options?.signal?.throwIfAborted()
      options?.signal?.addEventListener('abort', abort, { once: true })
      client.start()
      await client.initialize()
      const session = parseSession(await client.request('session/new', {
        cwd: workspace,
        mcpServers: [],
      }, timeoutMs))
      sessionId = session.sessionId
      if (request.model !== 'default') {
        const option = modelConfigOption(session.configOptions)
        if (!option || typeof option.id !== 'string') {
          throw new Error('OpenCode ACP did not expose a model configuration option')
        }
        const available = option.options?.flatMap((value) => (
          typeof value.value === 'string' ? [value.value] : []
        )) ?? []
        if (available.length > 0 && !available.includes(request.model)) {
          throw new Error(`OpenCode model is not available: ${request.model}`)
        }
        await client.request('session/set_config_option', {
          sessionId,
          configId: option.id,
          value: request.model,
        }, timeoutMs)
      }
      const result = await client.prompt(sessionId, formatOpencodeProviderPrompt(request))
      const content = assistantText.join('').trim() || extractAcpText(result).trim()
      if (!content) throw new Error('OpenCode ACP returned an empty assistant response')
      return {
        message: { role: 'assistant', content },
        ...(thinkingText.length > 0 ? { thinking: thinkingText.join('') } : {}),
        usage: { inputTokens: 0, outputTokens: 0 },
        finishReason: mapStopReason(result),
      }
    } finally {
      options?.signal?.removeEventListener('abort', abort)
      client.stop()
      await rm(workspace, { recursive: true, force: true, maxRetries: 3 })
    }
  }

  async *stream(request: ChatRequest, options?: LLMRequestOptions): AsyncIterable<StreamChunk> {
    try {
      const response = await this.chat(request, options)
      if (response.thinking) yield { type: 'thinking', text: response.thinking }
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
    options: { timeoutMs: number; onNotification: (message: JsonRpcMessage) => void },
  ): OpencodeAcpClient {
    return this.clientFactory
      ? this.clientFactory(spec, options)
      : new AcpStdioClient(spec, {
          requestTimeoutMs: options.timeoutMs,
          onNotification: options.onNotification,
        })
  }
}
