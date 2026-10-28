import OpenAI from 'openai'
import type {
  ILLMProvider,
  LLMRequestOptions,
  ChatRequest,
  ChatResponse,
  StreamChunk,
  ModelInfo,
  PromptCacheConfig,
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
  promptCache?: PromptCacheConfig
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
      fetch: getProviderSdkFetch(),
      maxRetries: 0,
    })
    this.models = config.models.map((m) => ({
      id: m,
      name: m,
      promptCache: config.promptCache,
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
    const model = this.models.find((candidate) => candidate.id === request.model)
    return createOpenAIChatResponse(this.client, request, options, {
      promptCache: model?.promptCache,
      thinkingControl: model?.capabilities.thinkingControl,
      reasoningEffortOffValue: model?.compatibility?.reasoningEffortOffValue,
      reasoningEffortAliases: model?.compatibility?.reasoningEffortAliases,
    })
  }

  async embed(texts: string[], model = 'text-embedding-3-small'): Promise<number[][]> {
    return createOpenAIEmbeddings(this.client, texts, model)
  }

  async *stream(request: ChatRequest, options?: LLMRequestOptions): AsyncIterable<StreamChunk> {
    const model = this.models.find((candidate) => candidate.id === request.model)
    yield* streamOpenAIChatResponse(this.client, request, options, {
      promptCache: model?.promptCache,
      thinkingControl: model?.capabilities.thinkingControl,
      reasoningEffortOffValue: model?.compatibility?.reasoningEffortOffValue,
      reasoningEffortAliases: model?.compatibility?.reasoningEffortAliases,
    })
  }
}
