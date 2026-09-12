import Anthropic from '@anthropic-ai/sdk'
import type {
  ILLMProvider,
  LLMRequestOptions,
  ChatRequest,
  ChatResponse,
  StreamChunk,
  ModelInfo,
  Message,
  ToolDefinition,
  ToolCall,
  TokenUsage,
} from '@sepilotd/core'
import { ThinkingLevel } from '@sepilotd/core'
import { extractContent, toApiError, wrapError } from './utils.js'
import { normalizeToolSchema } from './schema-normalize.js'
import { partitionSystemMessages } from './system-messages.js'

export interface AnthropicProviderConfig {
  apiKey: string
  models: string[]
  baseUrl?: string
  headers?: Record<string, string>
}

export function toAnthropicUserContent(
  msg: Message,
): string | Anthropic.ContentBlockParam[] {
  if (typeof msg.content === 'string') return msg.content
  const blocks: Anthropic.ContentBlockParam[] = []
  for (const part of msg.content) {
    if (part.type === 'text') {
      blocks.push({ type: 'text', text: part.text })
    } else if (part.type === 'image') {
      blocks.push({
        type: 'image',
        source:
          part.source.type === 'url'
            ? { type: 'url', url: part.source.data }
            : {
                type: 'base64',
                media_type:
                  part.source.mediaType as Anthropic.Base64ImageSource['media_type'],
                data: part.source.data,
              },
      })
    } else if (part.type === 'document') {
      blocks.push({
        type: 'document',
        source:
          part.source.type === 'url'
            ? { type: 'url', url: part.source.data }
            : {
                type: 'base64',
                media_type:
                  part.source.mediaType as 'application/pdf',
                data: part.source.data,
              },
      })
    }
  }
  return blocks.length > 0 ? blocks : extractContent(msg)
}

function mapMessages(
  messages: Message[],
): Anthropic.MessageParam[] {
  const result: Anthropic.MessageParam[] = []

  for (const msg of messages) {
    if (msg.role === 'system') continue // handled separately

    if (msg.role === 'user') {
      result.push({ role: 'user', content: toAnthropicUserContent(msg) })
    } else if (msg.role === 'assistant') {
      const content: Anthropic.ContentBlockParam[] = []
      const text = extractContent(msg)
      if (text) {
        content.push({ type: 'text', text })
      }
      if (msg.toolCalls?.length) {
        for (const tc of msg.toolCalls) {
          content.push({
            type: 'tool_use',
            id: tc.id,
            name: tc.name,
            input: tc.arguments,
          })
        }
      }
      result.push({ role: 'assistant', content })
    } else if (msg.role === 'tool') {
      result.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: msg.toolCallId ?? '',
            content: extractContent(msg),
          },
        ],
      })
    }
  }

  return result
}

function mapTools(
  tools: ToolDefinition[],
): Anthropic.Tool[] {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: normalizeToolSchema(t.inputSchema, 'anthropic') as Anthropic.Tool.InputSchema,
  }))
}

function mapFinishReason(
  reason: string,
): ChatResponse['finishReason'] {
  switch (reason) {
    case 'end_turn':
      return 'stop'
    case 'max_tokens':
      return 'length'
    case 'tool_use':
      return 'tool_use'
    default:
      return 'stop'
  }
}

function getThinkingBudget(
  level: ThinkingLevel | undefined,
): number | undefined {
  switch (level) {
    case ThinkingLevel.Low:
      return 1024
    case ThinkingLevel.Medium:
      return 4096
    case ThinkingLevel.High:
      return 16384
    case ThinkingLevel.Max:
      return 32768
    default:
      return undefined
  }
}

function markLastMessageCacheBreakpoint(messages: Anthropic.MessageParam[]): void {
  const lastMessage = messages[messages.length - 1]
  if (!lastMessage) return

  if (typeof lastMessage.content === 'string') {
    lastMessage.content = [
      {
        type: 'text',
        text: lastMessage.content,
        cache_control: { type: 'ephemeral' },
      },
    ]
    return
  }

  const lastBlockIndex = lastMessage.content.length - 1
  if (lastBlockIndex < 0) return

  const lastBlock = lastMessage.content[lastBlockIndex] as Anthropic.ContentBlockParam & {
    cache_control?: Anthropic.CacheControlEphemeral | null
  }
  lastMessage.content[lastBlockIndex] = {
    ...lastBlock,
    cache_control: { type: 'ephemeral' },
  } as Anthropic.ContentBlockParam
}

export function buildAnthropicParams(
  request: ChatRequest,
  defaultMaxTokens?: number,
): Anthropic.MessageCreateParamsNonStreaming {
  const normalized = partitionSystemMessages(request.messages, request.systemPrompt)
  const params: Anthropic.MessageCreateParamsNonStreaming = {
    model: request.model,
    messages: mapMessages(normalized.messages),
    // Default to the model's output cap (not a hard 4096 floor) so large
    // tool-call / write turns are not truncated. An explicit request.maxTokens
    // always wins; 4096 remains only as a last resort when no model info.
    max_tokens: request.maxTokens ?? defaultMaxTokens ?? 4096,
  }

  if (normalized.systemText) {
    // The system slot holds only the stable prefix: partitionSystemMessages
    // moves per-turn context into the latest user message, so this cache
    // breakpoint (and the history behind it) survives across turns.
    params.system = [
      {
        type: 'text' as const,
        text: normalized.systemText,
        cache_control: { type: 'ephemeral' as const },
      },
    ]
  }

  if (request.tools?.length) {
    params.tools = mapTools(request.tools)
    if (request.toolChoice === 'required') {
      params.tool_choice = { type: 'any' }
    } else if (request.toolChoice === 'auto') {
      params.tool_choice = { type: 'auto' }
    } else if (request.toolChoice === 'none') {
      params.tool_choice = { type: 'none' }
    }
    // Cache tool definitions (they rarely change within a session)
    if (params.tools.length > 0) {
      const lastTool = params.tools[params.tools.length - 1] as typeof params.tools[number] & {
        cache_control?: { type: 'ephemeral' }
      }
      lastTool.cache_control = { type: 'ephemeral' }
    }
  }
  if (request.temperature !== undefined) {
    params.temperature = request.temperature
  }
  if (request.stopSequences?.length) {
    params.stop_sequences = request.stopSequences
  }

  const thinkingBudget = getThinkingBudget(request.thinkingLevel)
  if (thinkingBudget) {
    params.thinking = {
      type: 'enabled',
      budget_tokens: thinkingBudget,
    }
    // Anthropic requires temperature=1 when thinking is enabled
    delete params.temperature
  }

  markLastMessageCacheBreakpoint(params.messages)

  return params
}

export class AnthropicProvider implements ILLMProvider {
  readonly id = 'anthropic'
  readonly name = 'Anthropic'
  readonly models: ModelInfo[]
  private client: Anthropic

  constructor(config: AnthropicProviderConfig) {
    this.client = new Anthropic({
      apiKey: config.apiKey,
      baseURL: config.baseUrl,
      defaultHeaders: config.headers,
    })
    this.models = config.models.map((m) => ({
      id: m,
      name: m,
      contextWindow: 200_000,
      maxOutputTokens: 8192,
      capabilities: {
        vision: true,
        toolUse: true,
        streaming: true,
        embedding: false,
        thinking: m.includes('claude'),
      },
    }))
  }

  private resolveMaxOutputTokens(modelId: string): number | undefined {
    return (this.models.find((m) => m.id === modelId) ?? this.models[0])?.maxOutputTokens
  }

  async chat(request: ChatRequest, options?: LLMRequestOptions): Promise<ChatResponse> {
    try {
      const params = buildAnthropicParams(request, this.resolveMaxOutputTokens(request.model))
      const response = await this.client.messages.create(params, {
        signal: options?.signal,
      })

      let text = ''
      let thinking = ''
      const toolCalls: ToolCall[] = []

      for (const block of response.content) {
        if (block.type === 'text') {
          text += block.text
        } else if (block.type === 'thinking') {
          thinking += block.thinking
        } else if (block.type === 'tool_use') {
          toolCalls.push({
            id: block.id,
            name: block.name,
            arguments: block.input as Record<string, unknown>,
          })
        }
      }

      return {
        message: {
          role: 'assistant',
          content: text,
          toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        },
        thinking: thinking || undefined,
        usage: {
          inputTokens: response.usage.input_tokens,
          outputTokens: response.usage.output_tokens,
          cacheReadTokens: response.usage.cache_read_input_tokens ?? 0,
          cacheCreationTokens: response.usage.cache_creation_input_tokens ?? 0,
        },
        finishReason: mapFinishReason(response.stop_reason ?? 'end_turn'),
      }
    } catch (err) {
      throw wrapError(err)
    }
  }

  async *stream(request: ChatRequest, options?: LLMRequestOptions): AsyncIterable<StreamChunk> {
    try {
      const params: Anthropic.MessageCreateParamsStreaming = { ...buildAnthropicParams(request, this.resolveMaxOutputTokens(request.model)), stream: true }
      const stream = this.client.messages.stream(params, {
        signal: options?.signal,
      })

      let currentToolCallId = ''

      for await (const event of stream) {
        if (event.type === 'message_start') {
          const startEvent = event as Anthropic.RawMessageStartEvent
          const usage = startEvent.message.usage
          if (usage) {
            const chunkUsage: TokenUsage = {
              inputTokens: usage.input_tokens ?? 0,
              outputTokens: 0,
            }
            if (usage.cache_read_input_tokens != null) {
              chunkUsage.cacheReadTokens = usage.cache_read_input_tokens
            }
            if (usage.cache_creation_input_tokens != null) {
              chunkUsage.cacheCreationTokens = usage.cache_creation_input_tokens
            }
            yield {
              type: 'usage',
              usage: chunkUsage,
            }
          }
        } else if (event.type === 'content_block_start') {
          const block = event.content_block
          if (block.type === 'tool_use') {
            currentToolCallId = block.id
            yield {
              type: 'tool_call_start',
              toolCall: {
                id: block.id,
                name: block.name,
              },
            }
          }
        } else if (event.type === 'content_block_delta') {
          const delta = event.delta
          if (delta.type === 'text_delta') {
            yield { type: 'text', text: delta.text }
          } else if (delta.type === 'thinking_delta') {
            yield { type: 'thinking', text: delta.thinking }
          } else if (delta.type === 'input_json_delta') {
            yield {
              type: 'tool_call_delta',
              toolCallId: currentToolCallId,
              delta: delta.partial_json,
            }
          }
        } else if (event.type === 'content_block_stop') {
          // We don't know block type here; tool_call_end is best-effort
        } else if (event.type === 'message_delta') {
          const deltaEvent = event as Anthropic.RawMessageDeltaEvent
          if (deltaEvent.usage) {
            yield {
              type: 'usage',
              usage: {
                inputTokens: 0,
                outputTokens: deltaEvent.usage.output_tokens ?? 0,
              },
            }
          }
          yield {
            type: 'done',
            finishReason: mapFinishReason(deltaEvent.delta.stop_reason ?? 'end_turn'),
          }
        }
      }
    } catch (err) {
      yield { type: 'error', error: toApiError(err) }
    }
  }
}
