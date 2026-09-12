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
} from '@sepilotd/core'
import { extractContent, toApiError, wrapError } from './utils.js'
import { normalizeToolSchema } from './schema-normalize.js'
import { partitionSystemMessages } from './system-messages.js'
import { randomUUID } from 'node:crypto'

export interface GeminiProviderConfig {
  apiKey: string
  models: string[]
  baseUrl?: string
  headers?: Record<string, string>
}

const MODEL_DEFAULTS: Record<string, Partial<ModelInfo>> = {
  'gemini-2.5-pro': {
    contextWindow: 1_048_576,
    maxOutputTokens: 65_536,
    capabilities: { vision: true, toolUse: true, streaming: true, embedding: false, thinking: true },
  },
  'gemini-2.5-flash': {
    contextWindow: 1_048_576,
    maxOutputTokens: 65_536,
    capabilities: { vision: true, toolUse: true, streaming: true, embedding: false, thinking: true },
  },
  'gemini-2.0-flash': {
    contextWindow: 1_048_576,
    maxOutputTokens: 8_192,
    capabilities: { vision: true, toolUse: true, streaming: true, embedding: false, thinking: false },
  },
}

interface GeminiContent {
  role: 'user' | 'model'
  parts: GeminiPart[]
}

type GeminiPart =
  | { text: string }
  | { inlineData: { mimeType: string; data: string } }
  | { fileData: { mimeType: string; fileUri: string } }
  | { functionCall: { name: string; args: Record<string, unknown> } }
  | { functionResponse: { name: string; response: { result: string } } }

interface GeminiTool {
  functionDeclarations: Array<{
    name: string
    description: string
    parameters: Record<string, unknown>
  }>
}

interface GeminiResponse {
  candidates?: Array<{
    content?: { parts?: GeminiPart[] }
    finishReason?: string
  }>
  usageMetadata?: {
    promptTokenCount?: number
    candidatesTokenCount?: number
    thoughtsTokenCount?: number
  }
  /** Top-level API error surfaced mid-stream by the SSE endpoint. */
  error?: { code?: number; message?: string; status?: string }
  /** Prompt-level safety block (request blocked before any candidate). */
  promptFeedback?: { blockReason?: string }
}

export function toGeminiUserParts(msg: Message): GeminiPart[] {
  if (typeof msg.content === 'string') return [{ text: msg.content }]

  const parts: GeminiPart[] = []
  for (const part of msg.content) {
    if (part.type === 'text') {
      parts.push({ text: part.text })
    } else if (part.type === 'image' || part.type === 'document') {
      if (part.source.type === 'base64') {
        parts.push({
          inlineData: {
            mimeType: part.source.mediaType,
            data: part.source.data,
          },
        })
      } else {
        parts.push({
          fileData: {
            mimeType: part.source.mediaType,
            fileUri: part.source.data,
          },
        })
      }
    }
  }

  return parts.length > 0 ? parts : [{ text: extractContent(msg) }]
}

export function mapMessages(messages: Message[], systemPrompt?: string): { contents: GeminiContent[]; systemInstruction?: { parts: GeminiPart[] } } {
  const contents: GeminiContent[] = []
  const normalized = partitionSystemMessages(messages, systemPrompt)

  for (const msg of normalized.messages) {
    if (msg.role === 'user') {
      contents.push({
        role: 'user',
        parts: toGeminiUserParts(msg),
      })
    } else if (msg.role === 'assistant') {
      const parts: GeminiPart[] = []
      const text = extractContent(msg)
      if (text) parts.push({ text })
      if (msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          parts.push({
            functionCall: { name: tc.name, args: tc.arguments },
          })
        }
      }
      if (parts.length > 0) {
        contents.push({ role: 'model', parts })
      }
    } else if (msg.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [{
          functionResponse: {
            // Gemini correlates a tool result to its call by function name,
            // not by an opaque id. Use the original function name captured on
            // the tool message; fall back to the id only for legacy messages
            // that predate the `name` field.
            name: msg.name ?? msg.toolCallId ?? 'unknown',
            response: { result: extractContent(msg) },
          },
        }],
      })
    }
  }

  const systemInstruction = normalized.systemText
    ? { parts: [{ text: normalized.systemText }] }
    : undefined

  return { contents, systemInstruction }
}

function mapTools(tools: ToolDefinition[]): GeminiTool[] {
  return [{
    functionDeclarations: tools.map((t) => ({
      name: t.name,
      description: t.description,
      parameters: normalizeToolSchema(t.inputSchema, 'gemini') as Record<string, unknown>,
    })),
  }]
}

// Gemini finishReasons that mean the content was withheld by a policy/safety
// layer rather than a natural stop. Surface these as content_filter so the
// caller does not treat a blocked/recited response as a clean answer.
const GEMINI_CONTENT_FILTER_REASONS = new Set([
  'SAFETY',
  'RECITATION',
  'BLOCKLIST',
  'PROHIBITED_CONTENT',
  'SPII',
  'IMAGE_SAFETY',
])

// Gemini finishReasons that mean the turn failed outright (not just filtered)
// and cannot be presented as any kind of completion.
const GEMINI_ERROR_REASONS = new Set(['MALFORMED_FUNCTION_CALL', 'OTHER', 'UNEXPECTED_TOOL_CALL'])

function mapFinishReason(reason?: string): ChatResponse['finishReason'] {
  switch (reason) {
    case 'STOP':
      return 'stop'
    case 'MAX_TOKENS':
      return 'length'
    default:
      return reason && GEMINI_CONTENT_FILTER_REASONS.has(reason) ? 'content_filter' : 'stop'
  }
}

/**
 * Returns an error message when a Gemini finishReason means the turn failed
 * outright (malformed tool call, unexpected tool call, generic OTHER) so the
 * stream can surface an error instead of a phantom clean stop.
 */
function geminiFinishReasonError(reason?: string): string | null {
  if (reason && GEMINI_ERROR_REASONS.has(reason)) {
    return `Gemini stopped with a non-recoverable finish reason: ${reason}`
  }
  return null
}

export class GeminiProvider implements ILLMProvider {
  readonly id = 'gemini'
  readonly name = 'Google Gemini'
  readonly models: ModelInfo[]
  private apiKey: string
  private baseUrl: string
  private headers: Record<string, string>

  constructor(config: GeminiProviderConfig) {
    this.apiKey = config.apiKey
    this.baseUrl = config.baseUrl ?? 'https://generativelanguage.googleapis.com'
    this.headers = { ...(config.headers ?? {}) }
    this.models = config.models.map((m) => {
      const defaults = MODEL_DEFAULTS[m]
      return {
        id: m,
        name: m,
        contextWindow: defaults?.contextWindow ?? 1_048_576,
        maxOutputTokens: defaults?.maxOutputTokens ?? 8_192,
        capabilities: defaults?.capabilities ?? {
          vision: true,
          toolUse: true,
          streaming: true,
          embedding: false,
          thinking: false,
        },
      }
    })
  }

  private async request(
    model: string,
    body: Record<string, unknown>,
    stream: boolean,
    signal?: AbortSignal,
  ): Promise<Response> {
    const action = stream ? 'streamGenerateContent' : 'generateContent'
    const url = new URL(`${this.baseUrl}/v1beta/models/${model}:${action}`)
    url.searchParams.set('key', this.apiKey)
    if (stream) {
      url.searchParams.set('alt', 'sse')
    }
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.headers,
      },
      body: JSON.stringify(body),
      signal,
    })
    if (!response.ok) {
      const text = await response.text()
      // Preserve the HTTP status so toApiError/circuit-breaker can classify
      // this (rate-limit / auth / context-length) instead of collapsing it to
      // a generic PROVIDER_ERROR parsed from the message string.
      const error = new Error(`Gemini API error ${response.status}: ${text}`) as Error & {
        status: number
      }
      error.status = response.status
      throw error
    }
    return response
  }

  async chat(request: ChatRequest, options?: LLMRequestOptions): Promise<ChatResponse> {
    try {
      const { contents, systemInstruction } = mapMessages(request.messages, request.systemPrompt)

      const body: Record<string, unknown> = {
        contents,
        generationConfig: {
          temperature: request.temperature,
          maxOutputTokens: request.maxTokens,
        },
      }

      if (systemInstruction) body.systemInstruction = systemInstruction
      if (request.tools?.length) {
        body.tools = mapTools(request.tools)
        if (request.toolChoice) {
          body.toolConfig = {
            functionCallingConfig: {
              mode: request.toolChoice === 'required'
                ? 'ANY'
                : request.toolChoice === 'none'
                  ? 'NONE'
                  : 'AUTO',
            },
          }
        }
      }
      if (request.stopSequences?.length) {
        (body.generationConfig as Record<string, unknown>).stopSequences = request.stopSequences
      }

      const response = await this.request(request.model, body, false, options?.signal)
      const data = await response.json() as GeminiResponse

      const candidate = data.candidates?.[0]
      const parts = candidate?.content?.parts ?? []

      let text = ''
      const toolCalls: ToolCall[] = []

      for (const part of parts) {
        if ('text' in part) {
          text += part.text
        } else if ('functionCall' in part) {
          toolCalls.push({
            id: randomUUID(),
            name: part.functionCall.name,
            arguments: part.functionCall.args,
          })
        }
      }

      const finishReason = toolCalls.length > 0 ? 'tool_use' as const : mapFinishReason(candidate?.finishReason)

      return {
        message: {
          role: 'assistant',
          content: text,
          toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        },
        usage: {
          inputTokens: data.usageMetadata?.promptTokenCount ?? 0,
          outputTokens: data.usageMetadata?.candidatesTokenCount ?? 0,
          thinkingTokens: data.usageMetadata?.thoughtsTokenCount,
        },
        finishReason,
      }
    } catch (err) {
      throw wrapError(err)
    }
  }

  async *stream(request: ChatRequest, options?: LLMRequestOptions): AsyncIterable<StreamChunk> {
    try {
      const { contents, systemInstruction } = mapMessages(request.messages, request.systemPrompt)

      const body: Record<string, unknown> = {
        contents,
        generationConfig: {
          temperature: request.temperature,
          maxOutputTokens: request.maxTokens,
        },
      }

      if (systemInstruction) body.systemInstruction = systemInstruction
      if (request.tools?.length) {
        body.tools = mapTools(request.tools)
        if (request.toolChoice) {
          body.toolConfig = {
            functionCallingConfig: {
              mode: request.toolChoice === 'required'
                ? 'ANY'
                : request.toolChoice === 'none'
                  ? 'NONE'
                  : 'AUTO',
            },
          }
        }
      }
      if (request.stopSequences?.length) {
        (body.generationConfig as Record<string, unknown>).stopSequences = request.stopSequences
      }

      const response = await this.request(request.model, body, true, options?.signal)
      const reader = response.body?.getReader()
      if (!reader) {
        yield { type: 'error', error: { code: 'PROVIDER_ERROR', message: 'No response body' } }
        return
      }

      const decoder = new TextDecoder()
      let buffer = ''
      let sawFinish = false

      outer: while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const jsonStr = line.slice(6).trim()
          if (!jsonStr || jsonStr === '[DONE]') continue

          let data: GeminiResponse
          try {
            data = JSON.parse(jsonStr) as GeminiResponse
          } catch (parseErr) {
            // A complete `data:` SSE line that does not parse means the stream
            // is corrupt — surface it instead of silently swallowing so the
            // turn does not end as a phantom clean stop.
            yield {
              type: 'error',
              error: {
                code: 'PROVIDER_ERROR',
                message: `Gemini stream returned malformed JSON: ${
                  parseErr instanceof Error ? parseErr.message : String(parseErr)
                }`,
              },
            }
            return
          }

          // Top-level API error delivered mid-stream (quota, invalid arg, …).
          if (data.error) {
            yield {
              type: 'error',
              error: {
                code: 'PROVIDER_ERROR',
                message:
                  data.error.message ??
                  `Gemini stream error (${data.error.status ?? data.error.code ?? 'unknown'})`,
              },
            }
            return
          }

          // Prompt-level safety block: no candidate is produced, so nothing
          // else would ever be emitted for this turn.
          if (data.promptFeedback?.blockReason) {
            yield {
              type: 'error',
              error: {
                code: 'PROVIDER_ERROR',
                message: `Gemini blocked the prompt: ${data.promptFeedback.blockReason}`,
              },
            }
            return
          }

          const parts = data.candidates?.[0]?.content?.parts ?? []

          for (const part of parts) {
            if ('text' in part) {
              yield { type: 'text', text: part.text }
            } else if ('functionCall' in part) {
              const id = randomUUID()
              yield {
                type: 'tool_call_start',
                toolCall: { id, name: part.functionCall.name },
              }
              yield {
                type: 'tool_call_delta',
                toolCallId: id,
                delta: JSON.stringify(part.functionCall.args),
              }
              yield { type: 'tool_call_end', toolCallId: id }
            }
          }

          const finishReason = data.candidates?.[0]?.finishReason
          if (finishReason && finishReason !== 'FINISH_REASON_UNSPECIFIED') {
            if (data.usageMetadata) {
              yield {
                type: 'usage',
                usage: {
                  inputTokens: data.usageMetadata.promptTokenCount ?? 0,
                  outputTokens: data.usageMetadata.candidatesTokenCount ?? 0,
                  thinkingTokens: data.usageMetadata.thoughtsTokenCount,
                },
              }
            }
            const errorMessage = geminiFinishReasonError(finishReason)
            if (errorMessage) {
              yield { type: 'error', error: { code: 'PROVIDER_ERROR', message: errorMessage } }
              return
            }
            sawFinish = true
            // A turn that produced a functionCall is a tool_use turn even
            // though Gemini still reports STOP as its finishReason.
            const hasToolCall = parts.some((p) => 'functionCall' in p)
            yield {
              type: 'done',
              finishReason: hasToolCall ? 'tool_use' : mapFinishReason(finishReason),
            }
            break outer
          }
        }
      }

      if (!sawFinish) {
        // Reader ended without any terminal finishReason — the response was
        // truncated or dropped. Do not present this as a clean stop.
        yield {
          type: 'error',
          error: {
            code: 'PROVIDER_ERROR',
            message: 'Gemini stream ended without a finish reason (truncated or dropped connection)',
          },
        }
      }
    } catch (err) {
      yield { type: 'error', error: toApiError(err) }
    }
  }
}
