import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import type { RequestOptions } from '@modelcontextprotocol/sdk/shared/protocol.js'
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js'
import {
  CreateMessageRequestSchema,
  ElicitRequestSchema,
  ListRootsRequestSchema,
  LoggingMessageNotificationSchema,
  ResourceUpdatedNotificationSchema,
} from '@modelcontextprotocol/sdk/types.js'
import type {
  ClientCapabilities,
  CompleteRequest,
  CreateMessageRequest,
  CreateMessageResult,
  ElicitRequest,
  ElicitResult,
  LoggingLevel,
  Prompt as SdkPrompt,
  Resource as SdkResource,
  ResourceTemplate as SdkResourceTemplate,
  Root,
  Tool as SdkTool,
} from '@modelcontextprotocol/sdk/types.js'
import type { ChatResponse, ContentPart, ILLMProvider, Message } from '@sepilotd/core'
import type { ToolDefinitionRuntime, ToolExecutionContext, ToolResult } from '../tools/registry.js'
import { DAEMON_VERSION } from '../version.js'
import { createMcpTransport } from './transport.js'
import type { McpServerConfig } from './transport.js'
import type { SecretVault } from '../security/secret-vault.js'
import {
  DEFAULT_MCP_CLIENT_CONFIG,
  type McpClientConfig,
} from '../config/schema.js'

export type { McpServerConfig } from './transport.js'

export interface PromptDescriptor {
  name: string
  description: string
  arguments: Array<{ name: string; description?: string; required?: boolean }>
  title?: string
  icons?: McpIcon[]
  _meta?: Record<string, unknown>
}

export interface McpAnnotations {
  audience?: Array<'user' | 'assistant'>
  priority?: number
  lastModified?: string
}

export interface McpIcon {
  src: string
  mimeType?: string
  sizes?: string[]
  theme?: 'light' | 'dark'
}

export interface ResourceDescriptor {
  uri: string
  name?: string
  title?: string
  description?: string
  mimeType?: string
  size?: number
  annotations?: McpAnnotations
  icons?: McpIcon[]
  _meta?: Record<string, unknown>
}

export interface ResourceTemplateDescriptor {
  uriTemplate: string
  name?: string
  title?: string
  description?: string
  mimeType?: string
  annotations?: McpAnnotations
  icons?: McpIcon[]
  _meta?: Record<string, unknown>
}

export interface ResourceReadResult {
  contents: Array<{ uri: string; text?: string; blob?: string; mimeType?: string; _meta?: Record<string, unknown> }>
}

export type McpCompletionRef =
  | { type: 'ref/prompt'; name: string }
  | { type: 'ref/resource'; uri: string }

export interface McpCompletionInput {
  ref: McpCompletionRef
  argument: { name: string; value: string }
  context?: { arguments?: Record<string, string> }
}

export interface McpCompletionResult {
  completion: {
    values: string[]
    total?: number
    hasMore?: boolean
  }
  _meta?: Record<string, unknown>
}

export type McpLoggingLevel = LoggingLevel

export interface McpLogMessage {
  level: McpLoggingLevel
  logger?: string
  data: unknown
  timestamp: string
}

export interface McpResourceUpdate {
  uri: string
  timestamp: string
}

export interface McpClientHooks {
  onDisconnect?: (name: string) => void
  recordCall?: (tool: string, durationMs: number, status: 'success' | 'error') => void
  onToolsChanged?: (name: string, error: Error | null, tools: ToolDefinitionRuntime[] | null) => void
  onPromptsChanged?: (name: string, error: Error | null, prompts: PromptDescriptor[] | null) => void
  onResourcesChanged?: (name: string, error: Error | null, resources: ResourceDescriptor[] | null) => void
  onResourceUpdated?: (name: string, update: McpResourceUpdate) => void
  onLogMessage?: (name: string, message: McpLogMessage) => void
}

export interface McpSamplingProviderRegistry {
  get(id: string): ILLMProvider | undefined
  getDefault(): ILLMProvider | undefined
}

export interface McpClientFeatureDeps {
  getProviderRegistry?: () => McpSamplingProviderRegistry | undefined
}

const MAX_INLINE_IMAGE_BYTES = 10 * 1024 * 1024
const MAX_NOTIFICATION_HISTORY = 200
const DEFAULT_MCP_TOOL_TIMEOUT_MS = 60_000
const MIN_MCP_TOOL_TIMEOUT_MS = 100
const MAX_MCP_TOOL_TIMEOUT_MS = 600_000

interface McpContentBlock {
  type: string
  text?: string
  data?: string
  mimeType?: string
  [key: string]: unknown
}

function base64ByteLength(data: string): number {
  const normalized = data.replace(/\s/g, '')
  const padding = normalized.endsWith('==') ? 2 : normalized.endsWith('=') ? 1 : 0
  return Math.max(0, Math.floor((normalized.length * 3) / 4) - padding)
}

function previewMcpContentBlock(block: McpContentBlock): string {
  if (block.type === 'text') {
    return block.text ?? ''
  }
  if (block.type === 'image') {
    const bytes = typeof block.data === 'string' ? base64ByteLength(block.data) : 0
    const mimeType = block.mimeType ?? 'image/png'
    const size = bytes > 0 ? ` ${bytes} bytes` : ''
    const omitted = bytes > MAX_INLINE_IMAGE_BYTES ? ' omitted from model context: too large' : ''
    return `[image:${mimeType}${size}${omitted}]`
  }

  return JSON.stringify(block, (_key, value) => {
    if (typeof value === 'string' && value.length > 200) {
      return `[${value.length} chars omitted]`
    }
    return value
  })
}

function contentPartsFromMcpBlocks(blocks: McpContentBlock[]): ContentPart[] | undefined {
  const parts: ContentPart[] = []
  for (const block of blocks) {
    if (block.type !== 'image' || typeof block.data !== 'string') {
      continue
    }
    const bytes = base64ByteLength(block.data)
    if (bytes > MAX_INLINE_IMAGE_BYTES) {
      continue
    }
    parts.push({
      type: 'image',
      source: {
        type: 'base64',
        mediaType: block.mimeType ?? 'image/png',
        data: block.data,
      },
    })
  }
  return parts.length > 0 ? parts : undefined
}

function omitUndefined<T extends Record<string, unknown>>(value: T): T {
  for (const key of Object.keys(value)) {
    if (value[key] === undefined) {
      delete value[key]
    }
  }
  return value
}

function normalizeMcpClientConfig(config?: McpClientConfig): McpClientConfig {
  const next = config ?? DEFAULT_MCP_CLIENT_CONFIG
  return {
    roots: {
      enabled: next.roots.enabled,
      listChanged: next.roots.listChanged,
      entries: next.roots.entries.map((root) => ({ ...root })),
    },
    sampling: {
      enabled: next.sampling.enabled,
      provider: next.sampling.provider,
      model: next.sampling.model,
      maxTokens: next.sampling.maxTokens,
      temperature: next.sampling.temperature,
    },
    elicitation: {
      enabled: next.elicitation.enabled,
      mode: next.elicitation.mode,
      applyDefaults: next.elicitation.applyDefaults,
    },
  }
}

function buildClientCapabilities(config: McpClientConfig): ClientCapabilities {
  const capabilities: ClientCapabilities = {}

  if (config.roots.enabled) {
    capabilities.roots = { listChanged: config.roots.listChanged }
  }

  if (config.sampling.enabled) {
    capabilities.sampling = {}
  }

  if (config.elicitation.enabled) {
    capabilities.elicitation = {
      form: { applyDefaults: config.elicitation.applyDefaults },
    }
  }

  return capabilities
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function resolveMcpToolTimeoutMs(config: { timeoutMs?: number }): number {
  const timeoutMs = config.timeoutMs
  if (
    typeof timeoutMs !== 'number' ||
    !Number.isFinite(timeoutMs) ||
    timeoutMs < MIN_MCP_TOOL_TIMEOUT_MS
  ) {
    return DEFAULT_MCP_TOOL_TIMEOUT_MS
  }
  return Math.min(Math.trunc(timeoutMs), MAX_MCP_TOOL_TIMEOUT_MS)
}

function buildMcpToolRequestOptions(
  config: { timeoutMs?: number },
  context?: ToolExecutionContext,
): RequestOptions {
  return omitUndefined({
    signal: context?.signal,
    timeout: resolveMcpToolTimeoutMs(config),
  })
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value) ?? String(value)
  } catch {
    return String(value)
  }
}

function mcpSamplingContentToText(content: unknown): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content.map(mcpSamplingContentToText).filter(Boolean).join('\n')
  }
  if (!isRecord(content)) return String(content ?? '')

  switch (content.type) {
    case 'text':
      return typeof content.text === 'string' ? content.text : ''
    case 'image':
      return `[image:${typeof content.mimeType === 'string' ? content.mimeType : 'image'}]`
    case 'audio':
      return `[audio:${typeof content.mimeType === 'string' ? content.mimeType : 'audio'}]`
    case 'tool_use':
      return `[tool_use:${String(content.name ?? '')}] ${safeJson(content.input ?? {})}`
    case 'tool_result':
      return mcpSamplingContentToText(content.content)
    case 'resource': {
      const resource = content.resource
      if (isRecord(resource) && typeof resource.text === 'string') return resource.text
      if (isRecord(resource) && typeof resource.uri === 'string') return `[resource:${resource.uri}]`
      return '[resource]'
    }
    case 'resource_link':
      return `[resource_link:${String(content.uri ?? '')}]`
    default:
      return safeJson(content)
  }
}

function responseContentToText(content: Message['content']): string {
  if (typeof content === 'string') return content
  return content.map((part) => {
    if (part.type === 'text') return part.text
    if (part.type === 'image') return `[image:${part.source.mediaType}]`
    if (part.type === 'document') return `[document:${part.source.mediaType}]`
    return safeJson(part)
  }).join('\n')
}

function mapSamplingStopReason(reason: ChatResponse['finishReason']): CreateMessageResult['stopReason'] {
  if (reason === 'length') return 'maxTokens'
  if (reason === 'stop') return 'endTurn'
  return reason
}

function defaultsFromElicitationSchema(schema: unknown): Record<string, string | number | boolean | string[]> {
  if (!isRecord(schema) || !isRecord(schema.properties)) return {}
  const content: Record<string, string | number | boolean | string[]> = {}
  for (const [key, property] of Object.entries(schema.properties)) {
    if (!isRecord(property) || !('default' in property)) continue
    const value = property.default
    if (
      typeof value === 'string'
      || typeof value === 'number'
      || typeof value === 'boolean'
      || (Array.isArray(value) && value.every((item) => typeof item === 'string'))
    ) {
      content[key] = value
    }
  }
  return content
}

function mapPrompt(prompt: SdkPrompt): PromptDescriptor {
  return omitUndefined({
    name: prompt.name,
    title: prompt.title,
    description: prompt.description ?? '',
    arguments: (prompt.arguments ?? []).map((arg) =>
      omitUndefined({
        name: arg.name,
        description: arg.description,
        required: arg.required,
      }),
    ),
    icons: prompt.icons,
    _meta: prompt._meta,
  })
}

function mapResource(resource: SdkResource): ResourceDescriptor {
  return omitUndefined({
    uri: resource.uri,
    name: resource.name,
    title: resource.title,
    description: resource.description,
    mimeType: resource.mimeType,
    size: resource.size,
    annotations: resource.annotations,
    icons: resource.icons,
    _meta: resource._meta,
  })
}

function mapResourceTemplate(template: SdkResourceTemplate): ResourceTemplateDescriptor {
  return omitUndefined({
    uriTemplate: template.uriTemplate,
    name: template.name,
    title: template.title,
    description: template.description,
    mimeType: template.mimeType,
    annotations: template.annotations,
    icons: template.icons,
    _meta: template._meta,
  })
}

export class McpClient {
  private client: Client
  private transport: Transport | null = null
  private config: McpServerConfig
  private connected = false
  private hooks: McpClientHooks
  private vault: SecretVault | null
  private clientFeatures: McpClientConfig
  private featureDeps: McpClientFeatureDeps
  private resourceSubscriptions = new Set<string>()
  private resourceUpdates: McpResourceUpdate[] = []
  private logMessages: McpLogMessage[] = []
  private loggingLevel: McpLoggingLevel | null = null

  constructor(
    config: McpServerConfig,
    hooks: McpClientHooks = {},
    vault?: SecretVault | null,
    clientFeatures?: McpClientConfig,
    featureDeps: McpClientFeatureDeps = {},
  ) {
    this.config = config
    this.hooks = hooks
    this.vault = vault ?? null
    this.clientFeatures = normalizeMcpClientConfig(clientFeatures)
    this.featureDeps = featureDeps
    this.client = new Client(
      { name: 'sepilotd', version: DAEMON_VERSION },
      {
        capabilities: buildClientCapabilities(this.clientFeatures),
        listChanged: {
          tools: {
            onChanged: (error, tools) => {
              this.hooks.onToolsChanged?.(
                this.config.name,
                error,
                tools ? tools.map((tool) => this.wrapMcpTool(tool as SdkTool)) : null,
              )
            },
          },
          prompts: {
            onChanged: (error, prompts) => {
              this.hooks.onPromptsChanged?.(
                this.config.name,
                error,
                prompts ? prompts.map(mapPrompt) : null,
              )
            },
          },
          resources: {
            onChanged: (error, resources) => {
              this.hooks.onResourcesChanged?.(
                this.config.name,
                error,
                resources ? resources.map(mapResource) : null,
              )
            },
          },
        },
      },
    )
    this.registerClientRequestHandlers()
    this.client.setNotificationHandler(ResourceUpdatedNotificationSchema, (notification) => {
      const update = {
        uri: notification.params.uri,
        timestamp: new Date().toISOString(),
      }
      this.resourceUpdates.push(update)
      this.resourceUpdates = this.resourceUpdates.slice(-MAX_NOTIFICATION_HISTORY)
      this.hooks.onResourceUpdated?.(this.config.name, update)
    })
    this.client.setNotificationHandler(LoggingMessageNotificationSchema, (notification) => {
      const message = {
        level: notification.params.level,
        logger: notification.params.logger,
        data: notification.params.data,
        timestamp: new Date().toISOString(),
      }
      this.logMessages.push(message)
      this.logMessages = this.logMessages.slice(-MAX_NOTIFICATION_HISTORY)
      this.hooks.onLogMessage?.(this.config.name, message)
    })
  }

  private registerClientRequestHandlers(): void {
    if (this.clientFeatures.roots.enabled) {
      this.client.setRequestHandler(ListRootsRequestSchema, async () => this.handleListRootsRequest())
    }

    if (this.clientFeatures.sampling.enabled) {
      this.client.setRequestHandler(CreateMessageRequestSchema, async (request, extra) => (
        this.handleCreateMessageRequest(request as CreateMessageRequest, extra.signal)
      ))
    }

    if (this.clientFeatures.elicitation.enabled) {
      this.client.setRequestHandler(ElicitRequestSchema, async (request) => (
        this.handleElicitRequest(request as ElicitRequest)
      ))
    }
  }

  private handleListRootsRequest(): { roots: Root[] } {
    if (!this.clientFeatures.roots.enabled) return { roots: [] }
    return {
      roots: this.clientFeatures.roots.entries.map((root) => omitUndefined({
        uri: root.uri,
        name: root.name,
        _meta: root._meta,
      })),
    }
  }

  private resolveSamplingProvider(): { provider: ILLMProvider; model: string } {
    const registry = this.featureDeps.getProviderRegistry?.()
    if (!registry) {
      throw new Error('MCP sampling is enabled but no provider registry is available')
    }

    const provider = this.clientFeatures.sampling.provider
      ? registry.get(this.clientFeatures.sampling.provider)
      : registry.getDefault()
    if (!provider) {
      throw new Error('MCP sampling provider is not configured')
    }

    const model = this.clientFeatures.sampling.model ?? provider.models[0]?.id
    if (!model) {
      throw new Error(`MCP sampling provider ${provider.id} has no available model`)
    }

    return { provider, model }
  }

  private async handleCreateMessageRequest(
    request: CreateMessageRequest,
    signal: AbortSignal,
  ): Promise<CreateMessageResult> {
    if (!this.clientFeatures.sampling.enabled) {
      throw new Error('MCP sampling is disabled')
    }
    if (request.params.tools?.length || request.params.toolChoice) {
      throw new Error('MCP sampling tool use is not enabled')
    }

    const { provider, model } = this.resolveSamplingProvider()
    const messages: Message[] = request.params.messages.map((message) => ({
      role: message.role,
      content: mcpSamplingContentToText(message.content),
    }))
    const maxTokens = Math.min(
      request.params.maxTokens,
      this.clientFeatures.sampling.maxTokens,
    )
    const response = await provider.chat({
      model,
      messages,
      systemPrompt: request.params.systemPrompt,
      temperature: this.clientFeatures.sampling.temperature ?? request.params.temperature,
      maxTokens,
      stopSequences: request.params.stopSequences,
    }, { signal })

    return {
      model,
      role: response.message.role === 'user' ? 'user' : 'assistant',
      content: {
        type: 'text',
        text: responseContentToText(response.message.content),
      },
      stopReason: mapSamplingStopReason(response.finishReason),
    }
  }

  private handleElicitRequest(request: ElicitRequest): ElicitResult {
    if (!this.clientFeatures.elicitation.enabled) {
      return { action: 'decline' }
    }
    if (request.params.mode === 'url') {
      return { action: 'decline' }
    }
    if (this.clientFeatures.elicitation.mode !== 'accept-defaults') {
      return { action: 'decline' }
    }
    return {
      action: 'accept',
      content: defaultsFromElicitationSchema(request.params.requestedSchema),
    }
  }

  getClientFeatureSnapshot(): McpClientConfig {
    return normalizeMcpClientConfig(this.clientFeatures)
  }

  getClientCapabilitiesSnapshot(): ClientCapabilities {
    return buildClientCapabilities(this.clientFeatures)
  }

  private wireTransportHooks(): void {
    if (!this.transport) return
    const t = this.transport as Transport & { onclose?: () => void; onerror?: (err: unknown) => void }
    t.onclose = () => {
      if (!this.connected) return
      this.connected = false
      this.hooks.onDisconnect?.(this.config.name)
    }
    t.onerror = () => {
      if (!this.connected) return
      this.connected = false
      this.hooks.onDisconnect?.(this.config.name)
    }
  }

  async connect(): Promise<void> {
    this.transport = createMcpTransport(this.config, this.vault)
    await this.client.connect(this.transport)
    this.connected = true
    this.wireTransportHooks()
  }

  async disconnect(): Promise<void> {
    if (this.transport) {
      this.connected = false
      await this.client.close()
      this.transport = null
    }
  }

  isConnected(): boolean {
    return this.connected
  }

  private hasCapability(name: 'tools' | 'prompts' | 'resources' | 'completions' | 'logging'): boolean {
    const caps = this.client.getServerCapabilities?.()
    return Boolean(caps && (caps as Record<string, unknown>)[name])
  }

  private supportsResourceSubscriptions(): boolean {
    const caps = this.client.getServerCapabilities?.() as { resources?: { subscribe?: boolean } } | undefined
    return Boolean(caps?.resources?.subscribe)
  }

  async discoverTools(): Promise<ToolDefinitionRuntime[]> {
    if (!this.connected) throw new Error('MCP client not connected')
    if (!this.hasCapability('tools')) return []
    const { tools } = await this.client.listTools()
    return tools.map((tool) => this.wrapMcpTool(tool))
  }

  async discoverPrompts(): Promise<PromptDescriptor[]> {
    if (!this.connected) throw new Error('MCP client not connected')
    if (!this.hasCapability('prompts')) return []
    try {
      const { prompts } = await this.client.listPrompts()
      return prompts.map(mapPrompt)
    } catch {
      return []
    }
  }

  async discoverResources(): Promise<ResourceDescriptor[]> {
    if (!this.connected) throw new Error('MCP client not connected')
    if (!this.hasCapability('resources')) return []
    try {
      const { resources } = await this.client.listResources()
      return resources.map(mapResource)
    } catch {
      return []
    }
  }

  async discoverResourceTemplates(): Promise<ResourceTemplateDescriptor[]> {
    if (!this.connected) throw new Error('MCP client not connected')
    if (!this.hasCapability('resources')) return []
    try {
      const { resourceTemplates } = await this.client.listResourceTemplates()
      return resourceTemplates.map(mapResourceTemplate)
    } catch {
      return []
    }
  }

  async getPrompt(
    name: string,
    args: Record<string, string> = {},
  ): Promise<{ messages: Array<{ role: string; content: unknown }> }> {
    if (!this.connected) throw new Error('MCP client not connected')
    const res = await this.client.getPrompt({ name, arguments: args })
    return { messages: res.messages as Array<{ role: string; content: unknown }> }
  }

  async readResource(uri: string): Promise<ResourceReadResult> {
    if (!this.connected) throw new Error('MCP client not connected')
    const res = await this.client.readResource({ uri })
    return {
      contents: res.contents as Array<{
        uri: string
        text?: string
        blob?: string
        mimeType?: string
        _meta?: Record<string, unknown>
      }>,
    }
  }

  async subscribeResource(uri: string): Promise<{ uri: string; subscribed: true }> {
    if (!this.connected) throw new Error('MCP client not connected')
    if (!this.supportsResourceSubscriptions()) {
      throw new Error('MCP server does not advertise resource subscriptions')
    }
    await this.client.subscribeResource({ uri })
    this.resourceSubscriptions.add(uri)
    return { uri, subscribed: true }
  }

  async unsubscribeResource(uri: string): Promise<{ uri: string; subscribed: false }> {
    if (!this.connected) throw new Error('MCP client not connected')
    if (!this.supportsResourceSubscriptions()) {
      throw new Error('MCP server does not advertise resource subscriptions')
    }
    await this.client.unsubscribeResource({ uri })
    this.resourceSubscriptions.delete(uri)
    return { uri, subscribed: false }
  }

  listResourceSubscriptions(): string[] {
    return Array.from(this.resourceSubscriptions).sort()
  }

  listResourceUpdates(limit = 50): McpResourceUpdate[] {
    return this.resourceUpdates.slice(-Math.max(0, limit))
  }

  async complete(input: McpCompletionInput): Promise<McpCompletionResult> {
    if (!this.connected) throw new Error('MCP client not connected')
    if (!this.hasCapability('completions')) {
      throw new Error('MCP server does not advertise completions')
    }
    const res = await this.client.complete(input as CompleteRequest['params'])
    return {
      completion: res.completion,
      _meta: res._meta,
    }
  }

  async setLoggingLevel(level: McpLoggingLevel): Promise<{ level: McpLoggingLevel }> {
    if (!this.connected) throw new Error('MCP client not connected')
    if (!this.hasCapability('logging')) {
      throw new Error('MCP server does not advertise logging')
    }
    await this.client.setLoggingLevel(level)
    this.loggingLevel = level
    return { level }
  }

  getLoggingState(): { level: McpLoggingLevel | null; messages: McpLogMessage[] } {
    return {
      level: this.loggingLevel,
      messages: [...this.logMessages],
    }
  }

  private wrapMcpTool(mcpTool: {
    name: string
    description?: string
    inputSchema: Record<string, unknown>
  }): ToolDefinitionRuntime {
    const prefix = `mcp.${this.config.name}`
    return {
      name: `${prefix}.${mcpTool.name}`,
      description: mcpTool.description ?? `MCP tool: ${mcpTool.name}`,
      inputSchema: mcpTool.inputSchema ?? { type: 'object', properties: {} },
      execute: async (
        input: Record<string, unknown>,
        context?: ToolExecutionContext,
      ): Promise<ToolResult> => {
        const start = Date.now()
        try {
          const result = await this.client.callTool(
            { name: mcpTool.name, arguments: input },
            undefined,
            buildMcpToolRequestOptions(this.config, context),
          )
          const content = result.content as McpContentBlock[]
          const output = content
            .map(previewMcpContentBlock)
            .join('\n')
          const contentParts = contentPartsFromMcpBlocks(content)
          const status: 'success' | 'error' = result.isError ? 'error' : 'success'
          const durationMs = Date.now() - start
          this.hooks.recordCall?.(mcpTool.name, durationMs, status)
          return {
            output: output ?? '',
            status,
            durationMs,
            ...(contentParts ? { contentParts } : {}),
          }
        } catch (err: unknown) {
          const durationMs = Date.now() - start
          this.hooks.recordCall?.(mcpTool.name, durationMs, 'error')
          const message = err instanceof Error ? err.message : String(err)
          return { output: message, status: 'error', durationMs }
        }
      },
    }
  }
}

export const __testables = {
  buildClientCapabilities,
  contentPartsFromMcpBlocks,
  defaultsFromElicitationSchema,
  resolveMcpToolTimeoutMs,
  mapPrompt,
  mapResource,
  mapResourceTemplate,
  mapSamplingStopReason,
  mcpSamplingContentToText,
  normalizeMcpClientConfig,
  previewMcpContentBlock,
  responseContentToText,
  safeJson,
}
