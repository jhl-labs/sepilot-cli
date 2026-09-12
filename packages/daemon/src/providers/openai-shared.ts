import type OpenAI from 'openai'
import type {
  ChatRequest,
  ChatResponse,
  LLMRequestOptions,
  Message,
  ModelInfo,
  StreamChunk,
  ToolCall,
  ToolDefinition,
} from '@sepilotd/core'
import { extractContent, toApiError, wrapError } from './utils.js'
import { normalizeToolSchema } from './schema-normalize.js'
import { partitionSystemMessages } from './system-messages.js'

type NonStreamingParams = OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming
type StreamingParams = OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming
type ChatCompletionChunk = OpenAI.Chat.Completions.ChatCompletionChunk
type StreamDelta = NonNullable<ChatCompletionChunk['choices'][number]['delta']>
type OpenAIThinkingControl = NonNullable<ModelInfo['capabilities']['thinkingControl']>

interface OpenAIResponseOptions {
  filterThinkTags?: boolean
  thinkingControl?: OpenAIThinkingControl
}

interface ThinkTagState {
  insideBlock: boolean
  buffer: string
  closeToken: '</think>' | '</thinking>'
}

export function toOpenAIUserContent(msg: Message): string | OpenAI.Chat.Completions.ChatCompletionContentPart[] {
  if (typeof msg.content === 'string') return msg.content
  const parts: OpenAI.Chat.Completions.ChatCompletionContentPart[] = []
  for (const part of msg.content) {
    if (part.type === 'text') {
      parts.push({ type: 'text', text: part.text })
    } else if (part.type === 'image') {
      const url =
        part.source.type === 'url' ? part.source.data : `data:${part.source.mediaType};base64,${part.source.data}`
      parts.push({ type: 'image_url', image_url: { url } })
    }
  }
  return parts.length > 0 ? parts : extractContent(msg)
}

export function mapOpenAIChatMessages(
  messages: Message[],
  systemPrompt?: string,
): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
  const result: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = []
  const normalized = partitionSystemMessages(messages, systemPrompt)

  if (normalized.systemText) {
    result.push({ role: 'system', content: normalized.systemText })
  }

  for (const msg of normalized.messages) {
    if (msg.role === 'user') {
      result.push({ role: 'user', content: toOpenAIUserContent(msg) })
    } else if (msg.role === 'assistant') {
      const param: OpenAI.Chat.Completions.ChatCompletionAssistantMessageParam = {
        role: 'assistant',
        content: extractContent(msg),
      }
      if (msg.toolCalls?.length) {
        param.tool_calls = msg.toolCalls.map((tc) => ({
          id: tc.id,
          type: 'function' as const,
          function: {
            name: tc.name,
            arguments: JSON.stringify(tc.arguments),
          },
        }))
      }
      result.push(param)
    } else if (msg.role === 'tool') {
      result.push({
        role: 'tool',
        tool_call_id: msg.toolCallId ?? '',
        content: extractContent(msg),
      })
    }
  }

  return result
}

function mapOpenAIChatTools(tools: ToolDefinition[]): OpenAI.Chat.Completions.ChatCompletionTool[] {
  return tools.map((tool) => ({
    type: 'function' as const,
    function: {
      name: tool.name,
      description: tool.description,
      parameters: normalizeToolSchema(tool.inputSchema, 'openai') as Record<string, unknown>,
    },
  }))
}

function mapOpenAIFinishReason(reason: string | null): ChatResponse['finishReason'] {
  switch (reason) {
    case 'stop':
      return 'stop'
    case 'length':
      return 'length'
    case 'tool_calls':
      return 'tool_use'
    case 'content_filter':
      return 'content_filter'
    default:
      return 'stop'
  }
}

function extractOpenAIReasoningContent(value: unknown): string {
  if (!value || typeof value !== 'object') {
    return ''
  }

  const record = value as Record<string, unknown>
  for (const key of ['reasoning_content', 'reasoning', 'thinking', 'reasoning_text']) {
    const text = record[key]
    if (typeof text === 'string' && text.length > 0) {
      return text
    }
  }

  return ''
}

function addOptionalChatParams(
  params: NonStreamingParams | StreamingParams,
  request: ChatRequest,
  thinkingControl?: OpenAIThinkingControl,
) {
  if (request.tools?.length) {
    params.tools = mapOpenAIChatTools(request.tools)
    if (request.toolChoice) {
      params.tool_choice = request.toolChoice
    }
  }
  if (request.stopSequences?.length) {
    params.stop = request.stopSequences
  }
  if (!request.thinkingLevel || !thinkingControl) return

  const compatibleParams = params as unknown as Record<string, unknown>
  if (thinkingControl === 'chat-template-kwargs') {
    compatibleParams.chat_template_kwargs = {
      enable_thinking: request.thinkingLevel !== 'off',
    }
    return
  }
  compatibleParams.reasoning_effort = request.thinkingLevel === 'off'
    ? 'none'
    : request.thinkingLevel === 'max'
      ? 'high'
      : request.thinkingLevel
}

function createNonStreamingParams(
  request: ChatRequest,
  thinkingControl?: OpenAIThinkingControl,
): NonStreamingParams {
  const params: NonStreamingParams = {
    model: request.model,
    messages: mapOpenAIChatMessages(request.messages, request.systemPrompt),
    stream: false,
    temperature: request.temperature,
    max_tokens: request.maxTokens,
  }
  addOptionalChatParams(params, request, thinkingControl)
  return params
}

function createStreamingParams(
  request: ChatRequest,
  thinkingControl?: OpenAIThinkingControl,
): StreamingParams {
  const params: StreamingParams = {
    model: request.model,
    messages: mapOpenAIChatMessages(request.messages, request.systemPrompt),
    stream: true,
    stream_options: { include_usage: true },
    temperature: request.temperature,
    max_tokens: request.maxTokens,
  }
  addOptionalChatParams(params, request, thinkingControl)
  return params
}

function mapToolCalls(
  toolCalls: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[] | undefined,
): ToolCall[] | undefined {
  return toolCalls
    ?.filter((tc): tc is Extract<typeof tc, { type: 'function' }> => tc.type === 'function')
    .map((tc) => ({
      id: tc.id,
      name: tc.function.name,
      arguments: JSON.parse(tc.function.arguments || '{}'),
    }))
}

/**
 * Split inline `<think>…</think>` / `<thinking>…</thinking>` reasoning out of
 * a complete (non-streamed) content string, reusing the same tokenizer the
 * streaming path uses. Returns the visible answer text and the extracted
 * reasoning so DeepSeek/Qwen/gpt-oss reasoning does not leak into the answer.
 */
export function stripInlineThinkTags(content: string): { content: string; thinking: string } {
  const state = createThinkTagState()
  let text = ''
  const thinkingParts: string[] = []
  for (const chunk of splitThinkingTags(content, state)) {
    if (chunk.type === 'text') text += chunk.text
    else if (chunk.type === 'thinking' && chunk.text) thinkingParts.push(chunk.text)
  }
  // An unterminated block (opened `<think>` with no close) stays buffered —
  // treat its contents as reasoning rather than dropping it or leaking it.
  if (state.insideBlock && state.buffer.trim()) {
    thinkingParts.push(state.buffer.trim())
  }
  return { content: text, thinking: thinkingParts.join('\n') }
}

export async function createOpenAIChatResponse(
  client: OpenAI,
  request: ChatRequest,
  options?: LLMRequestOptions,
  responseOptions: OpenAIResponseOptions = {},
): Promise<ChatResponse> {
  try {
    const response = await client.chat.completions.create(
      createNonStreamingParams(request, responseOptions.thinkingControl), {
      signal: options?.signal,
      },
    )
    const choice = response.choices[0]
    const reasoning = extractOpenAIReasoningContent(choice.message)

    let content = choice.message.content ?? ''
    let inlineThinking = ''
    if (responseOptions.filterThinkTags && content) {
      const split = stripInlineThinkTags(content)
      content = split.content
      inlineThinking = split.thinking
    }

    const thinking = [reasoning, inlineThinking].filter((t) => t && t.length > 0).join('\n')

    return {
      message: {
        role: 'assistant',
        content,
        toolCalls: mapToolCalls(choice.message.tool_calls),
      },
      thinking: thinking || undefined,
      usage: {
        inputTokens: response.usage?.prompt_tokens ?? 0,
        outputTokens: response.usage?.completion_tokens ?? 0,
      },
      finishReason: mapOpenAIFinishReason(choice.finish_reason),
    }
  } catch (err) {
    throw wrapError(err)
  }
}

function resolveEmbeddingSubBatchSize(): number {
  const raw = Number(process.env.SEPILOTD_EMBED_SUB_BATCH)
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : 96
}

export async function createOpenAIEmbeddings(
  client: OpenAI,
  texts: string[],
  model = 'text-embedding-3-small',
): Promise<number[][]> {
  if (texts.length === 0) return []
  // Sub-batch large inputs so one oversized request cannot exceed provider
  // input limits and fail an entire backfill batch at once.
  const subBatchSize = resolveEmbeddingSubBatchSize()
  try {
    if (texts.length <= subBatchSize) {
      const response = await client.embeddings.create({
        model,
        input: texts,
        encoding_format: 'float',
      })
      return response.data.sort((a, b) => a.index - b.index).map((item) => item.embedding)
    }
    const out: number[][] = []
    for (let start = 0; start < texts.length; start += subBatchSize) {
      const chunk = texts.slice(start, start + subBatchSize)
      const response = await client.embeddings.create({
        model,
        input: chunk,
        encoding_format: 'float',
      })
      for (const item of response.data.sort((a, b) => a.index - b.index)) {
        out.push(item.embedding)
      }
    }
    return out
  } catch (err) {
    throw wrapError(err)
  }
}

function createThinkTagState(): ThinkTagState {
  return {
    insideBlock: false,
    buffer: '',
    closeToken: '</think>',
  }
}

function findNextThinkOpenToken(content: string): {
  index: number
  openToken: '<think>' | '<thinking>'
  closeToken: '</think>' | '</thinking>'
} | null {
  const openThink = content.indexOf('<think>')
  const openThinking = content.indexOf('<thinking>')
  const candidates: Array<{
    index: number
    openToken: '<think>' | '<thinking>'
    closeToken: '</think>' | '</thinking>'
  }> = []

  if (openThink >= 0) {
    candidates.push({ index: openThink, openToken: '<think>', closeToken: '</think>' })
  }
  if (openThinking >= 0) {
    candidates.push({
      index: openThinking,
      openToken: '<thinking>',
      closeToken: '</thinking>',
    })
  }

  candidates.sort((left, right) => left.index - right.index)
  return candidates[0] ?? null
}

function* splitThinkingTags(content: string, state: ThinkTagState): Iterable<StreamChunk> {
  let remaining = content

  while (remaining.length > 0) {
    if (state.insideBlock) {
      const closeIndex = remaining.indexOf(state.closeToken)
      if (closeIndex === -1) {
        state.buffer += remaining
        return
      }

      state.buffer += remaining.slice(0, closeIndex)
      if (state.buffer.trim()) {
        yield { type: 'thinking', text: state.buffer.trim() }
      }
      state.buffer = ''
      state.insideBlock = false
      remaining = remaining.slice(closeIndex + state.closeToken.length)
      continue
    }

    const open = findNextThinkOpenToken(remaining)
    if (!open) {
      yield { type: 'text', text: remaining }
      return
    }

    const before = remaining.slice(0, open.index)
    if (before) {
      yield { type: 'text', text: before }
    }
    state.insideBlock = true
    state.closeToken = open.closeToken
    remaining = remaining.slice(open.index + open.openToken.length)
  }
}

function* mapContentDelta(content: string, thinkTagState: ThinkTagState | null): Iterable<StreamChunk> {
  if (!thinkTagState) {
    yield { type: 'text', text: content }
    return
  }

  yield* splitThinkingTags(content, thinkTagState)
}

function* mapToolCallDeltas(
  delta: StreamDelta,
  toolCallBuffers: Map<number, { id: string; name: string }>,
): Iterable<StreamChunk> {
  if (!delta.tool_calls) {
    return
  }

  for (const toolCall of delta.tool_calls) {
    if (toolCall.id) {
      toolCallBuffers.set(toolCall.index, {
        id: toolCall.id,
        name: toolCall.function?.name ?? '',
      })
      yield {
        type: 'tool_call_start',
        toolCall: {
          id: toolCall.id,
          name: toolCall.function?.name ?? '',
        },
      }
    }

    if (toolCall.function?.arguments) {
      const buffered = toolCallBuffers.get(toolCall.index)
      if (buffered) {
        yield {
          type: 'tool_call_delta',
          toolCallId: buffered.id,
          delta: toolCall.function.arguments,
        }
      }
    }
  }
}

export async function* streamOpenAIChatResponse(
  client: OpenAI,
  request: ChatRequest,
  options?: LLMRequestOptions,
  streamOptions: OpenAIResponseOptions = {},
): AsyncIterable<StreamChunk> {
  try {
    const stream = await client.chat.completions.create(
      createStreamingParams(request, streamOptions.thinkingControl), {
      signal: options?.signal,
      },
    )

    const toolCallBuffers = new Map<number, { id: string; name: string }>()
    const thinkTagState = streamOptions.filterThinkTags ? createThinkTagState() : null
    let usageEmitted = false
    let pendingDone: Extract<StreamChunk, { type: 'done' }> | undefined

    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta
      const reasoning = extractOpenAIReasoningContent(delta)
      if (reasoning) {
        yield { type: 'thinking', text: reasoning }
      }

      if (delta?.content) {
        yield* mapContentDelta(delta.content, thinkTagState)
      }

      if (delta) {
        yield* mapToolCallDeltas(delta, toolCallBuffers)
      }

      const finishReason = chunk.choices?.[0]?.finish_reason
      const usageChunk: StreamChunk | null =
        chunk.usage && !usageEmitted
          ? {
              type: 'usage',
              usage: {
                inputTokens: chunk.usage.prompt_tokens ?? 0,
                outputTokens: chunk.usage.completion_tokens ?? 0,
              },
            }
          : null

      if (finishReason) {
        for (const [, buffered] of toolCallBuffers) {
          yield { type: 'tool_call_end', toolCallId: buffered.id }
        }
        toolCallBuffers.clear()

        if (usageChunk) {
          usageEmitted = true
          yield usageChunk
        }
        // finish_reason ends content, not the wire stream. OpenAI-compatible
        // servers can send usage in a subsequent empty-choices frame. Our
        // guarded consumer stops at done, so it must be the final adapter event.
        pendingDone = { type: 'done', finishReason: mapOpenAIFinishReason(finishReason) }
      } else if (usageChunk) {
        // Usage often arrives (with stream_options.include_usage) in its own
        // trailing chunk that has an empty `choices` array and therefore no
        // finish_reason. Emit it here too, or it is silently dropped for every
        // OpenAI-compatible provider.
        usageEmitted = true
        yield usageChunk
      }
      if (pendingDone && usageEmitted) {
        yield pendingDone
        return
      }
    }
    // Some endpoints omit usage even when requested. Finish at actual EOF,
    // without inventing a zero-token usage event.
    if (pendingDone) yield pendingDone
  } catch (err) {
    yield { type: 'error', error: toApiError(err) }
  }
}
