import OpenAI from 'openai'
import type {
  ILLMProvider,
  LLMRequestOptions,
  ChatRequest,
  ChatResponse,
  StreamChunk,
  ModelInfo,
} from '@sepilotd/core'
import {
  createOpenAIChatResponse,
  createOpenAIEmbeddings,
  streamOpenAIChatResponse,
} from './openai-shared.js'
import { getProviderSdkFetch } from './http-timeout.js'

export { toOpenAIUserContent } from './openai-shared.js'

export interface OpenAIProviderConfig {
  apiKey: string
  models: string[]
  baseUrl?: string
  headers?: Record<string, string>
}

export class OpenAIProvider implements ILLMProvider {
  readonly id = 'openai'
  readonly name = 'OpenAI'
  readonly models: ModelInfo[]
  private client: OpenAI

  constructor(config: OpenAIProviderConfig) {
    this.client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseUrl,
      defaultHeaders: config.headers,
      // Keep SDK response bodies on the daemon's dispatcher-compatible path,
      // including the compiled Bun EOF adapter used by local providers.
      fetch: getProviderSdkFetch(),
      // The daemon's guarded provider layer owns bounded retries and circuit
      // accounting. Disable the SDK's hidden retry loop so every retry follows
      // the same observable policy.
      maxRetries: 0,
    })
    this.models = config.models.map((m) => ({
      id: m,
      name: m,
      contextWindow: 128_000,
      maxOutputTokens: 16_384,
      capabilities: {
        vision: true,
        toolUse: true,
        streaming: true,
        embedding: true,
        thinking: false,
      },
    }))
  }

  async chat(request: ChatRequest, options?: LLMRequestOptions): Promise<ChatResponse> {
    return createOpenAIChatResponse(this.client, request, options, {
      thinkingControl: this.models.find((model) => model.id === request.model)
        ?.capabilities.thinkingControl,
    })
  }

  async embed(texts: string[], model = 'text-embedding-3-small'): Promise<number[][]> {
    return createOpenAIEmbeddings(this.client, texts, model)
  }

  async *stream(request: ChatRequest, options?: LLMRequestOptions): AsyncIterable<StreamChunk> {
    yield* streamOpenAIChatResponse(this.client, request, options, {
      thinkingControl: this.models.find((model) => model.id === request.model)
        ?.capabilities.thinkingControl,
    })
  }
}
