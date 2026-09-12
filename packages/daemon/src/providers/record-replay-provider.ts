import { createHash } from 'node:crypto'
import { appendFile, mkdir, readFile, readdir, stat } from 'node:fs/promises'
import { dirname, extname, join } from 'node:path'
import type {
  ChatRequest,
  ChatResponse,
  ILLMProvider,
  LLMRequestOptions,
  Message,
  StreamChunk,
} from '@sepilotd/core'

export type RecordReplayMode = 'off' | 'record' | 'replay'

type CassetteKind = 'chat' | 'stream'

interface CassetteEntryBase {
  kind: CassetteKind
  hash: string
  providerId: string
  model: string
  recordedAt: string
}

interface ChatCassetteEntry extends CassetteEntryBase {
  kind: 'chat'
  response: ChatResponse
}

interface StreamCassetteEntry extends CassetteEntryBase {
  kind: 'stream'
  chunks: StreamChunk[]
}

type CassetteEntry = ChatCassetteEntry | StreamCassetteEntry

export interface RecordReplayOptions {
  mode: RecordReplayMode
  cassettePath?: string
}

function stableJsonValue(value: unknown): unknown {
  if (value === null || typeof value !== 'object') {
    return value
  }

  if (Array.isArray(value)) {
    return value.map(stableJsonValue)
  }

  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .filter(([, item]) => item !== undefined)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, item]) => [key, stableJsonValue(item)]),
  )
}

function normalizeMessage(message: Message): unknown {
  return stableJsonValue({
    role: message.role,
    content: message.content,
    toolCallId: message.toolCallId,
    toolCalls: message.toolCalls,
  })
}

function normalizeChatRequest(request: ChatRequest): unknown {
  return stableJsonValue({
    model: request.model,
    messages: request.messages.map(normalizeMessage),
    systemPrompt: request.systemPrompt,
    tools: request.tools,
    toolChoice: request.toolChoice,
    temperature: request.temperature,
    maxTokens: request.maxTokens,
    thinkingLevel: request.thinkingLevel,
    stopSequences: request.stopSequences,
    timeoutMs: request.timeoutMs,
  })
}

function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function cassetteKey(kind: CassetteKind, hash: string): string {
  return `${kind}:${hash}`
}

function isDisabled(value: string | undefined): boolean {
  return value == null || value.trim() === '' || value.trim() === '0'
}

function resolveCassettePath(value: string | undefined): string | undefined {
  if (isDisabled(value)) {
    return undefined
  }
  const trimmed = value!.trim()
  return trimmed === '1' || trimmed === 'memory' ? undefined : trimmed
}

function sanitizePathComponent(value: string): string {
  return value.replace(/[^A-Za-z0-9._-]/g, '_') || 'provider'
}

async function pathIsDirectory(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isDirectory()
  } catch {
    return extname(path) === ''
  }
}

async function resolveRecordFile(path: string, providerId: string): Promise<string> {
  if (await pathIsDirectory(path)) {
    await mkdir(path, { recursive: true })
    return join(path, `${sanitizePathComponent(providerId)}.jsonl`)
  }

  await mkdir(dirname(path), { recursive: true })
  return path
}

async function listReplayFiles(path: string): Promise<string[]> {
  if (await pathIsDirectory(path)) {
    let entries: string[]
    try {
      entries = await readdir(path)
    } catch {
      return []
    }
    return entries
      .filter((entry) => entry.endsWith('.jsonl'))
      .map((entry) => join(path, entry))
      .sort()
  }

  return [path]
}

export function hashChatRequest(request: ChatRequest): string {
  return createHash('sha256')
    .update(JSON.stringify(normalizeChatRequest(request)))
    .digest('hex')
}

export function resolveRecordReplayOptions(
  env: Record<string, string | undefined> = process.env,
): RecordReplayOptions {
  if (!isDisabled(env.SEPILOT_LLM_REPLAY)) {
    return {
      mode: 'replay',
      cassettePath: resolveCassettePath(env.SEPILOT_LLM_REPLAY),
    }
  }
  if (!isDisabled(env.SEPILOT_LLM_RECORD)) {
    return {
      mode: 'record',
      cassettePath: resolveCassettePath(env.SEPILOT_LLM_RECORD),
    }
  }
  return { mode: 'off' }
}

export class RecordReplayProvider implements ILLMProvider {
  readonly id: string
  readonly name: string
  readonly models: ILLMProvider['models']
  readonly embed?: ILLMProvider['embed']
  readonly countTokens?: ILLMProvider['countTokens']
  readonly refreshModelCatalog?: ILLMProvider['refreshModelCatalog']

  private readonly entries = new Map<string, CassetteEntry>()
  private loaded = false
  private recordFilePromise: Promise<string | undefined> | null = null

  constructor(
    providerId: string,
    private readonly provider: ILLMProvider,
    private readonly options: RecordReplayOptions,
  ) {
    this.id = providerId
    this.name = provider.name
    this.models = provider.models
    this.embed = provider.embed?.bind(provider)
    this.countTokens = provider.countTokens?.bind(provider)
    this.refreshModelCatalog = options.mode === 'replay'
      ? undefined
      : provider.refreshModelCatalog?.bind(provider)
  }

  get modelCatalogAuthority(): ILLMProvider['modelCatalogAuthority'] {
    return this.provider.modelCatalogAuthority
  }

  async chat(request: ChatRequest, options?: LLMRequestOptions): Promise<ChatResponse> {
    const hash = hashChatRequest(request)
    if (this.options.mode === 'replay') {
      await this.ensureLoaded()
      const entry = this.entries.get(cassetteKey('chat', hash))
      if (!entry || entry.kind !== 'chat') {
        throw new Error(`LLM replay miss for chat request ${hash} (${this.id}/${request.model})`)
      }
      return cloneJson(entry.response)
    }

    const response = await this.provider.chat(request, options)
    if (this.options.mode === 'record') {
      await this.record({
        kind: 'chat',
        hash,
        providerId: this.id,
        model: request.model,
        recordedAt: new Date().toISOString(),
        response: cloneJson(response),
      })
    }
    return response
  }

  async *stream(request: ChatRequest, options?: LLMRequestOptions): AsyncIterable<StreamChunk> {
    const hash = hashChatRequest(request)
    if (this.options.mode === 'replay') {
      await this.ensureLoaded()
      const entry = this.entries.get(cassetteKey('stream', hash))
      if (!entry || entry.kind !== 'stream') {
        throw new Error(`LLM replay miss for stream request ${hash} (${this.id}/${request.model})`)
      }
      for (const chunk of entry.chunks) {
        yield cloneJson(chunk)
      }
      return
    }

    const chunks: StreamChunk[] = []
    for await (const chunk of this.provider.stream(request, options)) {
      chunks.push(cloneJson(chunk))
      yield chunk
    }
    if (this.options.mode === 'record') {
      await this.record({
        kind: 'stream',
        hash,
        providerId: this.id,
        model: request.model,
        recordedAt: new Date().toISOString(),
        chunks,
      })
    }
  }

  private async ensureLoaded(): Promise<void> {
    if (this.loaded) {
      return
    }
    this.loaded = true
    if (!this.options.cassettePath) {
      return
    }

    for (const file of await listReplayFiles(this.options.cassettePath)) {
      let data: string
      try {
        data = await readFile(file, 'utf-8')
      } catch {
        continue
      }
      for (const line of data.split('\n')) {
        const trimmed = line.trim()
        if (!trimmed) {
          continue
        }
        const entry = JSON.parse(trimmed) as CassetteEntry
        this.entries.set(cassetteKey(entry.kind, entry.hash), entry)
      }
    }
  }

  private async record(entry: CassetteEntry): Promise<void> {
    await this.ensureLoaded()
    this.entries.set(cassetteKey(entry.kind, entry.hash), entry)
    const file = await this.resolveRecordFile()
    if (!file) {
      return
    }
    await appendFile(file, `${JSON.stringify(entry)}\n`, 'utf-8')
  }

  private async resolveRecordFile(): Promise<string | undefined> {
    if (!this.options.cassettePath) {
      return undefined
    }
    this.recordFilePromise ??= resolveRecordFile(this.options.cassettePath, this.id)
    return this.recordFilePromise
  }
}

export function wrapProviderForRecordReplay(
  providerId: string,
  provider: ILLMProvider,
  env: Record<string, string | undefined> = process.env,
): ILLMProvider {
  const options = resolveRecordReplayOptions(env)
  return options.mode === 'off' ? provider : new RecordReplayProvider(providerId, provider, options)
}

export const __testables = {
  normalizeChatRequest,
  sanitizePathComponent,
}
