import OpenAI from 'openai'
import type {
  ILLMProvider,
  LLMRequestOptions,
  ChatRequest,
  ChatResponse,
  StreamChunk,
  ModelInfo,
  ModelCatalogAuthority,
  ModelCatalogRefreshResult,
} from '@sepilotd/core'
import {
  createOpenAIChatResponse,
  createOpenAIEmbeddings,
  streamOpenAIChatResponse,
} from './openai-shared.js'
import { extractContent, toApiError } from './utils.js'
import { getProviderSdkFetch } from './http-timeout.js'

export { toOpenAIUserContent } from './openai-shared.js'

export interface OpenAICompatibleConfig {
  id: string
  name: string
  apiKey: string
  baseUrl: string
  headers?: Record<string, string>
  models: string[]
  defaultContextWindow?: number
  defaultMaxOutputTokens?: number
  capabilities?: Partial<ModelInfo['capabilities']>
}

interface ProviderModelDefaults {
  contextWindow: number
  maxOutputTokens: number
  capabilities: ModelInfo['capabilities']
  modelOverrides?: Record<string, Partial<ModelInfo>>
}

const OPENAI_COMPATIBLE_FALLBACK_CONTEXT_WINDOW = 192_000
const OPENAI_COMPATIBLE_FALLBACK_MAX_OUTPUT_TOKENS = 128_000

interface OpenAICompatibleModelRecord {
  id?: unknown
  context_window?: unknown
  context_length?: unknown
  max_context_length?: unknown
  max_model_len?: unknown
  max_sequence_length?: unknown
  max_seq_len?: unknown
  max_output_tokens?: unknown
  max_completion_tokens?: unknown
  output_token_limit?: unknown
  top_provider?: {
    context_length?: unknown
    max_completion_tokens?: unknown
  }
}

function positiveInteger(value: unknown): number | undefined {
  const parsed = typeof value === 'number'
    ? value
    : typeof value === 'string' && value.trim().length > 0
      ? Number(value)
      : Number.NaN
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : undefined
}

function firstPositiveInteger(...values: unknown[]): number | undefined {
  for (const value of values) {
    const parsed = positiveInteger(value)
    if (parsed !== undefined) return parsed
  }
  return undefined
}

export function extractOpenAICompatibleModelLimits(
  model: OpenAICompatibleModelRecord,
): Partial<Pick<ModelInfo, 'contextWindow' | 'maxOutputTokens'>> | undefined {
  const contextWindow = firstPositiveInteger(
    model.context_window,
    model.context_length,
    model.max_context_length,
    model.max_model_len,
    model.max_sequence_length,
    model.max_seq_len,
    model.top_provider?.context_length,
  )
  const maxOutputTokens = firstPositiveInteger(
    model.max_output_tokens,
    model.max_completion_tokens,
    model.output_token_limit,
    model.top_provider?.max_completion_tokens,
  )
  if (contextWindow === undefined && maxOutputTokens === undefined) return undefined
  return {
    ...(contextWindow !== undefined ? { contextWindow } : {}),
    ...(maxOutputTokens !== undefined ? { maxOutputTokens } : {}),
  }
}

const PROVIDER_DEFAULTS: Record<string, ProviderModelDefaults> = {
  groq: {
    contextWindow: 131_072,
    maxOutputTokens: 8_192,
    capabilities: {
      vision: false,
      toolUse: true,
      streaming: true,
      embedding: true,
      thinking: false,
    },
    modelOverrides: {
      'llama-3.3-70b-versatile': { contextWindow: 131_072, maxOutputTokens: 32_768 },
      'llama-3.1-8b-instant': { contextWindow: 131_072, maxOutputTokens: 8_192 },
      'mixtral-8x7b-32768': { contextWindow: 32_768, maxOutputTokens: 8_192 },
      'gemma2-9b-it': { contextWindow: 8_192, maxOutputTokens: 4_096 },
    },
  },
  together: {
    contextWindow: 131_072,
    maxOutputTokens: 128_000,
    capabilities: {
      vision: false,
      toolUse: true,
      streaming: true,
      embedding: true,
      thinking: false,
    },
    modelOverrides: {
      'meta-llama/Llama-3.3-70B-Instruct-Turbo': { contextWindow: 131_072 },
      'deepseek-ai/DeepSeek-R1': { contextWindow: 163_840 },
      'Qwen/Qwen2.5-72B-Instruct-Turbo': { contextWindow: 131_072 },
    },
  },
  deepseek: {
    contextWindow: 65_536,
    maxOutputTokens: 8_192,
    capabilities: {
      vision: false,
      toolUse: true,
      streaming: true,
      embedding: true,
      thinking: false,
    },
    modelOverrides: {
      'deepseek-chat': { contextWindow: 65_536, maxOutputTokens: 8_192 },
      'deepseek-reasoner': {
        contextWindow: 65_536,
        maxOutputTokens: 8_192,
        capabilities: {
          vision: false,
          toolUse: false,
          streaming: true,
          embedding: false,
          thinking: true,
        },
      },
    },
  },
  openrouter: {
    // OpenRouter fronts 200+ models with wildly different limits. These
    // are broad fallbacks; per-model context/maxTokens/capabilities should
    // be set via provider config modelOverrides when it matters
    // (e.g. a 1M-context Gemini route, or a vision-capable model).
    contextWindow: 128_000,
    maxOutputTokens: 128_000,
    capabilities: {
      vision: false,
      toolUse: true,
      streaming: true,
      embedding: false,
      thinking: false,
    },
  },
}

const PROVIDER_BASE_URLS: Record<string, string> = {
  groq: 'https://api.groq.com/openai/v1',
  together: 'https://api.together.xyz/v1',
  deepseek: 'https://api.deepseek.com',
  openrouter: 'https://openrouter.ai/api/v1',
}

/**
 * OpenRouter uses optional ranking headers (`HTTP-Referer`, `X-Title`)
 * to attribute traffic. We set neutral defaults that carry no host /
 * user / IP information; an operator can override them via the provider
 * config's `headers` map if they want their own attribution.
 */
const OPENROUTER_DEFAULT_HEADERS: Record<string, string> = {
  'HTTP-Referer': 'https://github.com/jhl-labs/sepilot-cli',
  'X-Title': 'sepilotd',
}

/**
 * Provider for OpenAI-compatible APIs (Groq, Together AI, DeepSeek, etc.)
 * Reuses the OpenAI SDK with a different baseURL.
 */
export class OpenAICompatibleProvider implements ILLMProvider {
  readonly id: string
  readonly name: string
  readonly models: ModelInfo[]
  private client: OpenAI
  private readonly config: OpenAICompatibleConfig
  private catalogAuthority: ModelCatalogAuthority = 'configured'

  constructor(config: OpenAICompatibleConfig) {
    this.config = config
    this.id = config.id
    this.name = config.name

    const baseUrl = config.baseUrl || PROVIDER_BASE_URLS[config.id] || config.baseUrl
    this.client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: baseUrl,
      defaultHeaders: config.headers,
      // Compiled Bun must consume provider bodies through the daemon's
      // undici.request adapter; its native/undici fetch bridges can receive
      // all bytes yet fail to deliver body EOF to the OpenAI SDK.
      fetch: getProviderSdkFetch(),
      // Retry ownership lives in guardedProviderChat/guardedProviderStream.
      // Leaving the SDK's independent retry loop enabled multiplies attempts
      // (and honours long Retry-After values) underneath a single silent graph
      // node, so a local 429 can look like a frozen agent with an idle GPU.
      maxRetries: 0,
    })

    const defaults = PROVIDER_DEFAULTS[config.id]
    this.models = config.models.map((m) => {
      const override = defaults?.modelOverrides?.[m]
      return {
        id: m,
        name: m,
        contextWindow:
          override?.contextWindow ??
          config.defaultContextWindow ??
          defaults?.contextWindow ??
          OPENAI_COMPATIBLE_FALLBACK_CONTEXT_WINDOW,
        maxOutputTokens:
          override?.maxOutputTokens ??
          config.defaultMaxOutputTokens ??
          defaults?.maxOutputTokens ??
          OPENAI_COMPATIBLE_FALLBACK_MAX_OUTPUT_TOKENS,
        capabilities: {
          ...(defaults?.capabilities ?? {
            vision: false,
            toolUse: true,
            // Streaming is part of the OpenAI-compatible chat-completions
            // contract and preserves incremental progress for long local
            // reasoning/tool responses. The guarded stream layer treats the
            // protocol's `done` chunk as terminal, so a runtime no longer has
            // to wait for a second iterator EOF. Operators can still declare
            // `streaming: false` for endpoints that only implement chat().
            streaming: true,
            embedding: true,
            // A custom OpenAI-compatible endpoint does not expose a reliable
            // capability catalog. Treat hidden reasoning as supported until
            // the operator says otherwise so short orchestration calls leave
            // enough output headroom for reasoning + the requested JSON.
            // This only raises a max-token ceiling; ordinary models still stop
            // as soon as their short answer is complete.
            thinking: true,
            // OpenAI-compatible is a wire protocol, not evidence that native
            // tools are broken. Tool-only assistant turns with empty visible
            // prose are valid and common, so automatic adaptive switching is
            // opt-in through the model compatibility profile/capability. Real
            // transport rejection still has its separate guarded recovery.
            adaptivePromptReact: false,
          }),
          ...config.capabilities,
          ...override?.capabilities,
        },
      }
    })
  }

  get modelCatalogAuthority(): ModelCatalogAuthority {
    return this.catalogAuthority
  }

  async refreshModelCatalog(): Promise<ModelCatalogRefreshResult> {
    const page = await this.client.models.list()
    const records = typeof (page as { getPaginatedItems?: unknown }).getPaginatedItems === 'function'
      ? await (page as unknown as { getPaginatedItems(): Promise<unknown[]> }).getPaginatedItems()
      : (page as unknown as { data?: unknown[] }).data ?? []
    const discovered = new Map<string, OpenAICompatibleModelRecord>()
    for (const candidate of records) {
      if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) continue
      const record = candidate as OpenAICompatibleModelRecord
      if (typeof record.id !== 'string' || record.id.trim().length === 0) continue
      discovered.set(record.id.trim(), record)
    }

    const previousById = new Map(this.models.map((model) => [model.id, model] as const))
    const defaults = PROVIDER_DEFAULTS[this.config.id]
    const buildModel = (id: string, record: OpenAICompatibleModelRecord): ModelInfo => {
      const previous = previousById.get(id)
      const providerOverride = defaults?.modelOverrides?.[id]
      const advertised = extractOpenAICompatibleModelLimits(record)
      const contextWindow =
        this.config.defaultContextWindow ??
        advertised?.contextWindow ??
        providerOverride?.contextWindow ??
        previous?.contextWindow ??
        defaults?.contextWindow ??
        OPENAI_COMPATIBLE_FALLBACK_CONTEXT_WINDOW
      const maxOutputTokens = Math.min(
        this.config.defaultMaxOutputTokens ??
        advertised?.maxOutputTokens ??
        providerOverride?.maxOutputTokens ??
        previous?.maxOutputTokens ??
        defaults?.maxOutputTokens ??
        OPENAI_COMPATIBLE_FALLBACK_MAX_OUTPUT_TOKENS,
        contextWindow,
      )
      return {
        id,
        name: previous?.name ?? id,
        contextWindow,
        maxOutputTokens,
        capabilities: previous?.capabilities ?? {
          ...(defaults?.capabilities ?? {
            vision: false,
            toolUse: true,
            streaming: true,
            embedding: true,
            thinking: true,
            adaptivePromptReact: false,
          }),
          ...this.config.capabilities,
          ...providerOverride?.capabilities,
        },
        ...(previous?.compatibility ? { compatibility: previous.compatibility } : {}),
      }
    }

    const modelIds = Array.from(discovered.keys()).sort()
    const discoveredSet = new Set(modelIds)
    const added = modelIds.filter((id) => !previousById.has(id))
    const removed = this.models
      .map((model) => model.id)
      .filter((id) => !discoveredSet.has(id))
    const nextModels = modelIds.map((id) => buildModel(id, discovered.get(id)!))
    this.models.splice(0, this.models.length, ...nextModels)
    this.catalogAuthority = 'endpoint'
    return { modelIds, added, removed }
  }

  /** Create a Groq provider */
  static groq(
    apiKey: string,
    models: string[],
    options?: { baseUrl?: string; headers?: Record<string, string> },
  ): OpenAICompatibleProvider {
    return new OpenAICompatibleProvider({
      id: 'groq',
      name: 'Groq',
      apiKey,
      baseUrl: options?.baseUrl ?? PROVIDER_BASE_URLS.groq,
      headers: options?.headers,
      models,
    })
  }

  /** Create a Together AI provider */
  static together(
    apiKey: string,
    models: string[],
    options?: { baseUrl?: string; headers?: Record<string, string> },
  ): OpenAICompatibleProvider {
    return new OpenAICompatibleProvider({
      id: 'together',
      name: 'Together AI',
      apiKey,
      baseUrl: options?.baseUrl ?? PROVIDER_BASE_URLS.together,
      headers: options?.headers,
      models,
    })
  }

  /** Create a DeepSeek provider */
  static deepseek(
    apiKey: string,
    models: string[],
    options?: { baseUrl?: string; headers?: Record<string, string> },
  ): OpenAICompatibleProvider {
    return new OpenAICompatibleProvider({
      id: 'deepseek',
      name: 'DeepSeek',
      apiKey,
      baseUrl: options?.baseUrl ?? PROVIDER_BASE_URLS.deepseek,
      headers: options?.headers,
      models,
    })
  }

  /**
   * Create an OpenRouter provider — a single OpenAI-compatible endpoint
   * that fronts 200+ models from Anthropic, OpenAI, Google, Meta, Mistral,
   * DeepSeek, Qwen and many more. Model ids are namespaced
   * (`anthropic/claude-3.7-sonnet`, `openai/gpt-4o`, …).
   */
  static openrouter(
    apiKey: string,
    models: string[],
    options?: {
      baseUrl?: string
      headers?: Record<string, string>
      defaultContextWindow?: number
      defaultMaxOutputTokens?: number
      capabilities?: Partial<ModelInfo['capabilities']>
    },
  ): OpenAICompatibleProvider {
    return new OpenAICompatibleProvider({
      id: 'openrouter',
      name: 'OpenRouter',
      apiKey,
      baseUrl: options?.baseUrl ?? PROVIDER_BASE_URLS.openrouter,
      headers: { ...OPENROUTER_DEFAULT_HEADERS, ...options?.headers },
      models,
      defaultContextWindow: options?.defaultContextWindow,
      defaultMaxOutputTokens: options?.defaultMaxOutputTokens,
      capabilities: options?.capabilities,
    })
  }

  async chat(request: ChatRequest, options?: LLMRequestOptions): Promise<ChatResponse> {
    return createOpenAIChatResponse(this.client, request, options, {
      filterThinkTags: true,
      thinkingControl: this.models.find((model) => model.id === request.model)
        ?.capabilities.thinkingControl,
    })
  }

  async embed(texts: string[], model = 'text-embedding-3-small'): Promise<number[][]> {
    return createOpenAIEmbeddings(this.client, texts, model)
  }

  async *stream(request: ChatRequest, options?: LLMRequestOptions): AsyncIterable<StreamChunk> {
    const model = this.models.find((candidate) => candidate.id === request.model)
    if (model?.capabilities.streaming === false) {
      try {
        const response = await this.chat(request, options)
        if (response.thinking) {
          yield { type: 'thinking', text: response.thinking }
        }
        const content = extractContent(response.message)
        if (content) {
          yield { type: 'text', text: content }
        }
        for (const toolCall of response.message.toolCalls ?? []) {
          yield {
            type: 'tool_call_start',
            toolCall: { id: toolCall.id, name: toolCall.name },
          }
          yield {
            type: 'tool_call_delta',
            toolCallId: toolCall.id,
            delta: JSON.stringify(toolCall.arguments),
          }
          yield { type: 'tool_call_end', toolCallId: toolCall.id }
        }
        yield { type: 'usage', usage: response.usage }
        yield { type: 'done', finishReason: response.finishReason }
      } catch (error) {
        yield { type: 'error', error: toApiError(error) }
      }
      return
    }

    yield* streamOpenAIChatResponse(this.client, request, options, {
      filterThinkTags: true,
      thinkingControl: model?.capabilities.thinkingControl,
    })
  }
}
