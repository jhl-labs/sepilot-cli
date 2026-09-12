import { Ollama } from 'ollama'
import { randomUUID } from 'node:crypto'
import type {
  ILLMProvider,
  LLMRequestOptions,
  ChatRequest,
  ChatResponse,
  StreamChunk,
  ModelInfo,
  ModelCatalogAuthority,
  ModelCatalogRefreshResult,
  Message,
  ToolCall,
} from '@sepilotd/core'
import {
  getAbortError,
  throwIfAborted,
} from '../abort.js'
import { getProviderSdkFetch } from './http-timeout.js'
import { toApiError, wrapError } from './utils.js'
import { normalizeToolSchema } from './schema-normalize.js'
import { partitionSystemMessages } from './system-messages.js'

export interface OllamaProviderConfig {
  baseUrl: string
  headers?: HeadersInit
  models: string[]
}

interface OllamaToolCall {
  function?: { name?: string; arguments?: Record<string, unknown> }
}

interface OllamaChatResponse {
  message?: {
    content?: string
    thinking?: string
    tool_calls?: OllamaToolCall[]
  }
  prompt_eval_count?: number
  eval_count?: number
  done?: boolean
  done_reason?: string
}

interface OllamaChatChunk {
  message?: {
    content?: string
    thinking?: string
    tool_calls?: Array<{ function: { name: string; arguments: Record<string, unknown> } }>
  }
  done?: boolean
  done_reason?: string
  prompt_eval_count?: number
  eval_count?: number
}

/**
 * Ollama sends `done: true` on every terminal chunk regardless of *why* it
 * stopped, and only `done_reason` distinguishes a natural stop from a
 * num_predict/context truncation (`'length'`). Prefer `done_reason` and treat
 * a truncation as `length` so the loop knows the answer is incomplete; fall
 * back to the coarse `done` flag for older Ollama builds that omit it.
 */
function mapOllamaFinishReason(
  doneReason: string | undefined,
  done: boolean | undefined,
  hasToolCalls: boolean,
): ChatResponse['finishReason'] {
  if (hasToolCalls) return 'tool_use'
  if (doneReason === 'length') return 'length'
  if (doneReason === 'stop') return 'stop'
  return done ? 'stop' : 'length'
}

function mapRole(role: Message['role']): string {
  return role // ollama uses same role names
}

/**
 * Ollama's chat API splits multimodal turns into a plain text `content`
 * field plus a separate `images: string[]` array of base64 payloads.
 * `extractContent` alone would have dropped the image entirely (it
 * filters to text parts only), so we walk the ContentPart[] and keep
 * both halves. URL-source images are not pre-fetched here — Ollama
 * accepts only base64 today, so a remote URL is simply skipped.
 */
export function toOllamaMessage(msg: Message): {
  role: string
  content: string
  images?: string[]
} {
  if (typeof msg.content === 'string') {
    return { role: mapRole(msg.role), content: msg.content }
  }
  const textParts: string[] = []
  const images: string[] = []
  for (const part of msg.content) {
    if (part.type === 'text') {
      textParts.push(part.text)
    } else if (part.type === 'image' && part.source.type === 'base64') {
      images.push(part.source.data)
    }
    // Documents and URL-source images are silently dropped: Ollama has
    // no first-class equivalent.
  }
  return {
    role: mapRole(msg.role),
    content: textParts.join('\n'),
    ...(images.length > 0 ? { images } : {}),
  }
}

// Ollama doesn't expose per-model capability flags in /api/tags, so we infer
// from the model id as a fallback. Names like 'nomic-embed-text',
// 'qwen3-embedding:8b', 'bge-m3', 'all-minilm', 'mxbai-embed-large' are
// embedding-only — they accept tokens, return vectors, can't follow chat
// instructions or call tools. Vision-input names include the classic 'vl' /
// 'vision' / 'image' / 'multimodal' markers plus common model families that
// do not carry those tokens (llava, bakllava, moondream, minicpm-v,
// granite*vision). The default is a chat-capable text model
// (toolUse=true, embedding=false). When SEPILOTD_OLLAMA_CAPABILITY_PROBE is
// set, `/api/show` capabilities take priority over these regexes (structural,
// not a per-model name branch).
const EMBEDDING_ONLY_PATTERN =
  /(?:^|[-_/:])(embed(?:ding)?|bge[-_].*m3|nomic[-_]embed|all[-_]minilm|mxbai|gte|arctic[-_]embed|snowflake[-_]arctic[-_]embed)(?:[-_:]|$)/i
const VISION_PATTERN =
  /(?:^|[-_/:])(vl|vision|image|multimodal|llava|bakllava|moondream|minicpm[-_]v|granite[-_.].*vision)(?:[-_:]|$)/i
const OLLAMA_CONTEXT_WINDOW = 128_000
const OLLAMA_DEFAULT_MAX_OUTPUT_TOKENS = 32_768
const OLLAMA_CAPABILITY_PROBE_TIMEOUT_MS = 10_000
const OLLAMA_CAPABILITY_PROBE_MAX_CONCURRENCY = 4

export interface OllamaCapabilityProbeOptions {
  timeoutMs?: number
  maxConcurrency?: number
}

interface OllamaModelCapabilityProbe {
  capabilities: string[] | null
  permanentlyUnavailable: boolean
}

/**
 * On by default: the name-regex fallback guesses `tools` for every non-embedding
 * model, and guessing wrong is not a soft failure. Sending a `tools` array to a
 * model that does not advertise them makes Ollama answer 500, which surfaces as
 * an opaque "Internal Server Error" only after the whole turn has been built —
 * observed at ~108s on a cloud model, with nothing pointing at tool support.
 *
 * The probe itself is cheap and non-fatal: one `/api/show` per model, issued
 * with bounded concurrency and a shared deadline. Any model whose probe fails
 * keeps its last-known or regex-derived defaults.
 * Set SEPILOTD_OLLAMA_CAPABILITY_PROBE=0 to fall back to pure name matching.
 */
export function isOllamaCapabilityProbeEnabled(): boolean {
  const raw = process.env.SEPILOTD_OLLAMA_CAPABILITY_PROBE?.trim().toLowerCase()
  return raw !== '0' && raw !== 'false' && raw !== 'off'
}

/**
 * Map Ollama's `/api/show` `capabilities` array (e.g.
 * `['completion','tools','vision','thinking','embedding']`) onto our
 * ModelInfo capability flags. When the probe reports capabilities we trust it
 * over the name regex: a model that does not advertise `tools` must not be
 * sent a `tools` array (that hard-errors the run on non-tool models like
 * gemma2/phi3), and a model advertising `vision` gets vision even without a
 * telltale name.
 */
export function ollamaCapabilitiesFromProbe(capabilities: string[]): {
  vision: boolean
  toolUse: boolean
  embedding: boolean
  thinking: boolean
} {
  const set = new Set(capabilities.map((c) => c.toLowerCase()))
  return {
    vision: set.has('vision'),
    toolUse: set.has('tools'),
    embedding: set.has('embedding'),
    thinking: set.has('thinking'),
  }
}

function buildModelInfo(id: string, probed?: string[]): ModelInfo {
  const isEmbedding = probed
    ? probed.map((c) => c.toLowerCase()).includes('embedding')
    : EMBEDDING_ONLY_PATTERN.test(id)
  const probedCaps = probed ? ollamaCapabilitiesFromProbe(probed) : null
  const hasVision = probedCaps ? probedCaps.vision : VISION_PATTERN.test(id) && !isEmbedding
  const hasToolUse = probedCaps ? probedCaps.toolUse : !isEmbedding
  const hasThinking = probedCaps ? probedCaps.thinking : false
  return {
    id,
    name: id,
    contextWindow: OLLAMA_CONTEXT_WINDOW,
    maxOutputTokens: isEmbedding ? 0 : OLLAMA_DEFAULT_MAX_OUTPUT_TOKENS,
    capabilities: {
      vision: hasVision,
      toolUse: hasToolUse,
      streaming: !isEmbedding,
      embedding: isEmbedding,
      thinking: hasThinking,
    },
  }
}

/**
 * Query Ollama `/api/show` for a model's advertised capabilities. Returns the
 * `capabilities` string array, or `null` when the endpoint is unavailable or
 * the build predates the field (older Ollama). Never throws.
 */
export async function fetchOllamaModelCapabilities(
  baseUrl: string,
  model: string,
  headers?: HeadersInit,
  signal?: AbortSignal,
): Promise<string[] | null> {
  return (await probeOllamaModelCapabilities(baseUrl, model, headers, signal)).capabilities
}

async function probeOllamaModelCapabilities(
  baseUrl: string,
  model: string,
  headers?: HeadersInit,
  signal?: AbortSignal,
): Promise<OllamaModelCapabilityProbe> {
  try {
    const url = new URL('/api/show', baseUrl).toString()
    const requestHeaders = new Headers(headers)
    if (!requestHeaders.has('content-type')) {
      requestHeaders.set('content-type', 'application/json')
    }
    const res = await getProviderSdkFetch()(url, {
      method: 'POST',
      headers: requestHeaders,
      body: JSON.stringify({ model }),
      signal,
    })
    if (!res.ok) {
      await res.body?.cancel().catch(() => undefined)
      // `/api/tags` can retain a remotely retired model after the model
      // endpoint has made its permanent unavailability authoritative. Only
      // HTTP 410 is strong enough to remove that row: 404 can mean an older
      // `/api/show` surface, while auth, timeout, and 5xx failures do not prove
      // that a listed model is gone.
      return { capabilities: null, permanentlyUnavailable: res.status === 410 }
    }
    const data = (await res.json()) as { capabilities?: unknown }
    if (!Array.isArray(data.capabilities)) {
      return { capabilities: null, permanentlyUnavailable: false }
    }
    return {
      capabilities: data.capabilities.filter((c): c is string => typeof c === 'string'),
      permanentlyUnavailable: false,
    }
  } catch {
    return { capabilities: null, permanentlyUnavailable: false }
  }
}

function mergeProbedModelInfo(
  previous: ModelInfo | undefined,
  id: string,
  caps: string[],
): ModelInfo {
  const probed = buildModelInfo(id, caps)
  if (!previous) return probed
  let maxOutputTokens = previous.maxOutputTokens
  if (probed.capabilities.embedding) {
    maxOutputTokens = 0
  } else if (previous.capabilities.embedding) {
    maxOutputTokens = probed.maxOutputTokens
  }
  return {
    ...previous,
    maxOutputTokens,
    capabilities: probed.capabilities,
  }
}

export class OllamaProvider implements ILLMProvider {
  readonly id = 'ollama'
  readonly name = 'Ollama'
  readonly models: ModelInfo[]
  readonly baseUrl: string
  readonly headers?: HeadersInit
  private catalogAuthority: ModelCatalogAuthority = 'configured'
  private client: Ollama
  private readonly providerFetch: typeof globalThis.fetch

  constructor(config: OllamaProviderConfig) {
    this.baseUrl = config.baseUrl
    this.headers = config.headers
    this.providerFetch = getProviderSdkFetch()
    this.client = new Ollama({
      host: config.baseUrl,
      headers: config.headers,
      fetch: this.providerFetch,
    })
    this.models = config.models.map((m) => buildModelInfo(m))
  }

  /**
   * Ollama's SDK only attaches its internal AbortController to streaming
   * requests. Non-streaming `chat()` calls otherwise reach `fetch()` without
   * a signal, so a caller-side deadline can return while Ollama keeps
   * generating. Use a request-scoped client whose fetch implementation adds
   * the caller signal while preserving any signal supplied by the SDK.
   */
  private clientForSignal(signal?: AbortSignal): Ollama {
    if (!signal) return this.client

    return new Ollama({
      host: this.baseUrl,
      headers: this.headers,
      fetch: (input, init) => {
        const sdkSignal = init?.signal
        const requestSignal = sdkSignal && sdkSignal !== signal
          ? AbortSignal.any([sdkSignal, signal])
          : signal
        return this.providerFetch(input, { ...init, signal: requestSignal })
      },
    })
  }

  static async listAvailableModels(
    baseUrl: string,
    headers?: HeadersInit,
  ): Promise<string[]> {
    const url = new URL('/api/tags', baseUrl).toString()
    const res = await getProviderSdkFetch()(url, { headers })
    if (!res.ok) {
      throw new Error(`Ollama /api/tags responded ${res.status}`)
    }
    const data = (await res.json()) as { models?: Array<{ name?: string }> }
    return (data.models ?? [])
      .map((m) => m.name)
      .filter((name): name is string => typeof name === 'string' && name.length > 0)
  }

  get modelCatalogAuthority(): ModelCatalogAuthority {
    return this.catalogAuthority
  }

  async refreshModelCatalog(): Promise<ModelCatalogRefreshResult> {
    const discovered = Array.from(new Set(
      (await OllamaProvider.listAvailableModels(this.baseUrl, this.headers))
        .map((id) => id.trim())
        .filter((id) => id.length > 0),
    ))
    const previousById = new Map(this.models.map((model) => [model.id, model] as const))
    const probe = await this.probeModelCapabilities(discovered)
    const available = discovered.filter((id) => !probe.permanentlyUnavailable.has(id))
    const discoveredSet = new Set(available)
    const added = available.filter((id) => !previousById.has(id))
    const removed = this.models
      .map((model) => model.id)
      .filter((id) => !discoveredSet.has(id))
    const nextModels = available.map((id) => {
      const previous = previousById.get(id)
      const probed = probe.capabilities.get(id)
      return probed === undefined
        ? previous ?? buildModelInfo(id)
        : mergeProbedModelInfo(previous, id, probed)
    })

    this.models.splice(0, this.models.length, ...nextModels)
    this.catalogAuthority = 'endpoint'

    return { modelIds: available, added, removed }
  }

  addDiscoveredModels(modelIds: string[]): string[] {
    const existing = new Set(this.models.map((m) => m.id))
    const added: string[] = []
    for (const id of modelIds) {
      if (existing.has(id)) continue
      this.models.push(buildModelInfo(id))
      existing.add(id)
      added.push(id)
    }
    return added
  }

  /**
   * When SEPILOTD_OLLAMA_CAPABILITY_PROBE is enabled, replace the name-regex
   * capability guesses with the model's real `/api/show` capabilities. Best
   * effort and non-fatal: any retained model whose probe fails keeps its last
   * known metadata, while a newly discovered model keeps conservative
   * name-derived defaults. No-op when the env gate is off.
   */
  async probeCapabilities(options: OllamaCapabilityProbeOptions = {}): Promise<void> {
    const probedCapabilities = (await this.probeModelCapabilities(
      this.models.map((model) => model.id),
      options,
    )).capabilities
    for (let index = 0; index < this.models.length; index += 1) {
      const info = this.models[index]
      if (!info) continue
      const probed = probedCapabilities.get(info.id)
      if (probed !== undefined) {
        this.models[index] = mergeProbedModelInfo(info, info.id, probed)
      }
    }
  }

  private async probeModelCapabilities(
    modelIds: string[],
    options: OllamaCapabilityProbeOptions = {},
  ): Promise<{
    capabilities: Map<string, string[]>
    permanentlyUnavailable: Set<string>
  }> {
    const capabilities = new Map<string, string[]>()
    const permanentlyUnavailable = new Set<string>()
    if (!isOllamaCapabilityProbeEnabled() || modelIds.length === 0) {
      return { capabilities, permanentlyUnavailable }
    }

    const requestedTimeout = options.timeoutMs ?? OLLAMA_CAPABILITY_PROBE_TIMEOUT_MS
    const timeoutMs = Number.isFinite(requestedTimeout)
      ? Math.max(1, Math.trunc(requestedTimeout))
      : OLLAMA_CAPABILITY_PROBE_TIMEOUT_MS
    const requestedConcurrency = options.maxConcurrency
      ?? OLLAMA_CAPABILITY_PROBE_MAX_CONCURRENCY
    const maxConcurrency = Number.isFinite(requestedConcurrency)
      ? Math.max(1, Math.min(modelIds.length, Math.trunc(requestedConcurrency)))
      : Math.min(modelIds.length, OLLAMA_CAPABILITY_PROBE_MAX_CONCURRENCY)
    const signal = AbortSignal.timeout(timeoutMs)
    let nextIndex = 0

    const worker = async (): Promise<void> => {
      while (!signal.aborted) {
        const index = nextIndex
        nextIndex += 1
        const modelId = modelIds[index]
        if (!modelId) return
        const probed = await probeOllamaModelCapabilities(
          this.baseUrl,
          modelId,
          this.headers,
          signal,
        )
        if (probed.permanentlyUnavailable) {
          permanentlyUnavailable.add(modelId)
        } else if (probed.capabilities !== null) {
          capabilities.set(modelId, probed.capabilities)
        }
      }
    }

    await Promise.all(Array.from({ length: maxConcurrency }, () => worker()))
    return { capabilities, permanentlyUnavailable }
  }

  async chat(request: ChatRequest, options?: LLMRequestOptions): Promise<ChatResponse> {
    throwIfAborted(options?.signal, 'Ollama request aborted')
    try {
      const client = this.clientForSignal(options?.signal)
      const normalized = partitionSystemMessages(request.messages, request.systemPrompt)
      const messages = normalized.messages.map((m) => toOllamaMessage(m))

      if (normalized.systemText) {
        messages.unshift({ role: 'system', content: normalized.systemText })
      }

      // Map tool definitions to Ollama format
      const tools = request.tools?.map(t => ({
        type: 'function' as const,
        function: {
          name: t.name,
          description: t.description,
          parameters: normalizeToolSchema(t.inputSchema, 'ollama') as Record<string, unknown>,
        },
      }))

      const response = (await client.chat({
        model: request.model,
        messages,
        stream: false,
        tools: tools?.length ? tools : undefined,
        options: {
          temperature: request.temperature,
          num_predict: request.maxTokens,
        },
      } as Parameters<typeof this.client.chat>[0])) as OllamaChatResponse

      const toolCalls: ToolCall[] = (response.message?.tool_calls ?? []).map((tc) => ({
        id: randomUUID(),
        name: tc.function?.name ?? '',
        arguments: (tc.function?.arguments ?? {}) as Record<string, unknown>,
      }))

      return {
        message: {
          role: 'assistant',
          content: response.message?.content ?? '',
          toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        },
        thinking: response.message?.thinking || undefined,
        usage: {
          inputTokens: response.prompt_eval_count ?? 0,
          outputTokens: response.eval_count ?? 0,
        },
        finishReason: mapOllamaFinishReason(response.done_reason, response.done, toolCalls.length > 0),
      }
    } catch (err) {
      if (options?.signal?.aborted) {
        throw getAbortError(options.signal, 'Ollama request aborted')
      }
      throw wrapError(err)
    }
  }

  async embed(texts: string[], model: string): Promise<number[][]> {
    try {
      const response = await this.client.embed({
        model,
        input: texts,
      })

      return response.embeddings ?? []
    } catch (err) {
      throw wrapError(err)
    }
  }

  async *stream(request: ChatRequest, options?: LLMRequestOptions): AsyncIterable<StreamChunk> {
    throwIfAborted(options?.signal, 'Ollama stream aborted')
    try {
      const normalized = partitionSystemMessages(request.messages, request.systemPrompt)
      const messages = normalized.messages.map((m) => toOllamaMessage(m))

      if (normalized.systemText) {
        messages.unshift({ role: 'system', content: normalized.systemText })
      }

      const tools = request.tools?.map(t => ({
        type: 'function' as const,
        function: { name: t.name, description: t.description, parameters: normalizeToolSchema(t.inputSchema, 'ollama') as Record<string, unknown> },
      }))

      const streamResult = await this.client.chat({
        model: request.model,
        messages,
        stream: true,
        tools: tools?.length ? tools : undefined,
        options: {
          temperature: request.temperature,
          num_predict: request.maxTokens,
        },
      } as Parameters<typeof this.client.chat>[0])
      const response = streamResult as unknown as AsyncIterable<OllamaChatChunk>

      const onAbort = () => {
        this.client.abort()
      }
      options?.signal?.addEventListener('abort', onAbort, { once: true })

      try {
        for await (const chunk of response) {
          // Ollama exposes reasoning models' live progress through
          // `message.thinking`. Propagate it even when capability discovery
          // did not advertise thinking: the response field is authoritative,
          // and dropping it makes an active upstream stream look silent to the
          // first-token watchdog until content or a tool call appears.
          if (chunk.message?.thinking) {
            yield { type: 'thinking', text: chunk.message.thinking }
          }

          if (chunk.message?.content) {
            yield { type: 'text', text: chunk.message.content }
          }

          const chunkToolCalls = chunk.message?.tool_calls

          if (chunkToolCalls?.length) {
            for (const tc of chunkToolCalls) {
              const id = randomUUID()
              yield {
                type: 'tool_call_start' as const,
                toolCall: { id, name: tc.function?.name ?? '' },
              }
              yield {
                type: 'tool_call_delta' as const,
                toolCallId: id,
                delta: JSON.stringify(tc.function?.arguments ?? {}),
              }
              yield { type: 'tool_call_end' as const, toolCallId: id }
            }
          }

          if (chunk.done) {
            yield {
              type: 'usage',
              usage: {
                inputTokens: chunk.prompt_eval_count ?? 0,
                outputTokens: chunk.eval_count ?? 0,
              },
            }
            const hasToolCalls = Boolean(chunkToolCalls?.length)
            yield {
              type: 'done',
              finishReason: mapOllamaFinishReason(chunk.done_reason, chunk.done, hasToolCalls),
            }
            break
          }
        }
      } finally {
        options?.signal?.removeEventListener('abort', onAbort)
      }
    } catch (err) {
      if (options?.signal?.aborted) {
        yield { type: 'error', error: toApiError(getAbortError(options.signal, 'Ollama stream aborted')) }
        return
      }
      yield { type: 'error', error: toApiError(err) }
    }
  }
}
