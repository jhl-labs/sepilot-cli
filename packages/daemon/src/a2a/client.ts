import { randomUUID } from 'node:crypto'
import type { Dispatcher } from 'undici'
import { assertPublicUrl, createPinnedLookupDispatcher } from '../utils/ssrf-guard.js'
import {
  A2A_PROTOCOL_VERSION,
  type A2AAgentCard,
  type A2AJsonRpcResponse,
  type A2AMessage,
  type A2APart,
  type A2ASendMessageResponse,
} from './types.js'

export interface A2AClientOptions {
  fetcher?: typeof fetch
  headers?: Record<string, string>
  maxRedirects?: number
  timeoutMs?: number
}

export interface A2ASendInput {
  agentCardUrl: string
  message: string
  contextId?: string
  taskId?: string
  headers?: Record<string, string>
  timeoutMs?: number
}

const DEFAULT_MAX_REDIRECTS = 5
const MAX_REDIRECTS = 10
const DEFAULT_TIMEOUT_MS = 15_000
const MAX_TIMEOUT_MS = 120_000
const MAX_JSON_RESPONSE_BYTES = 2 * 1024 * 1024
const FORBIDDEN_CALLER_HEADERS = new Set([
  'connection',
  'content-length',
  'host',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
])

type FetchInitWithDispatcher = RequestInit & { dispatcher: Dispatcher }

interface ValidatedFetchResult {
  response: Response
  dispatcher: Dispatcher
  finalUrl: string
  callerHeadersAllowed: boolean
}

interface AgentCardDiscovery {
  card: A2AAgentCard
  origin: string
  callerHeadersAllowed: boolean
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function cardUrlFromInput(input: string): string {
  const url = new URL(input)
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('A2A Agent Card URL must use http or https')
  }
  if (url.username || url.password) {
    throw new Error('A2A Agent Card URL must not contain credentials')
  }
  if (url.pathname.endsWith('.json')) return url.toString()
  url.pathname = '/.well-known/agent-card.json'
  url.search = ''
  url.hash = ''
  return url.toString()
}

function mergeHeaders(
  ...sources: Array<Record<string, string> | undefined>
): Record<string, string> {
  const merged = new Headers()
  for (const source of sources) {
    for (const [name, value] of Object.entries(source ?? {})) {
      if (FORBIDDEN_CALLER_HEADERS.has(name.toLowerCase())) continue
      merged.set(name, value)
    }
  }
  return Object.fromEntries(merged.entries())
}

function redirectMethod(
  status: number,
  method: string,
  body: BodyInit | null | undefined,
  headers: Record<string, string>,
): { method: string; body: BodyInit | null | undefined; headers: Record<string, string> } {
  if (status !== 303 && !((status === 301 || status === 302) && method === 'POST')) {
    return { method, body, headers }
  }
  const nextHeaders = { ...headers }
  delete nextHeaders['content-length']
  delete nextHeaders['content-type']
  return { method: 'GET', body: undefined, headers: nextHeaders }
}

function isRedirectStatus(status: number): boolean {
  return status === 301 || status === 302 || status === 303 || status === 307 || status === 308
}

async function discardResponseBody(response: Response): Promise<void> {
  await response.body?.cancel().catch(() => undefined)
}

function normalizeTimeoutMs(value: number | undefined, fallback = DEFAULT_TIMEOUT_MS): number {
  if (value == null || !Number.isFinite(value) || value <= 0) return fallback
  return Math.min(MAX_TIMEOUT_MS, Math.max(1, Math.floor(value)))
}

function isJsonContentType(value: string | null): boolean {
  const mime = value?.split(';', 1)[0]?.trim().toLowerCase()
  return mime === 'application/json' || Boolean(mime?.endsWith('+json'))
}

async function readBoundedJson(response: Response): Promise<unknown> {
  if (!isJsonContentType(response.headers.get('content-type'))) {
    throw new Error('response content type is not JSON')
  }
  const contentLength = response.headers.get('content-length')
  if (contentLength != null) {
    if (!/^\d+$/u.test(contentLength)) throw new Error('response content length is invalid')
    const declaredBytes = Number(contentLength)
    if (!Number.isSafeInteger(declaredBytes) || declaredBytes > MAX_JSON_RESPONSE_BYTES) {
      throw new Error('response is too large')
    }
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('response body is empty')
  const chunks: Uint8Array[] = []
  let totalBytes = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (!value) continue
    totalBytes += value.byteLength
    if (totalBytes > MAX_JSON_RESPONSE_BYTES) {
      await reader.cancel().catch(() => undefined)
      throw new Error('response is too large')
    }
    chunks.push(value)
  }

  const bytes = new Uint8Array(totalBytes)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)) as unknown
}

function textMessage(
  text: string,
  options: { contextId?: string; taskId?: string } = {},
): A2AMessage {
  const part: A2APart = { text, mediaType: 'text/plain' }
  return {
    messageId: randomUUID(),
    contextId: options.contextId,
    taskId: options.taskId,
    role: 'ROLE_USER',
    parts: [part],
  }
}

function pickJsonRpcInterface(card: A2AAgentCard): { url: string; protocolVersion: string } {
  const found = card.supportedInterfaces?.find((item) => item.protocolBinding === 'JSONRPC')
  if (!found?.url) throw new Error('A2A Agent Card does not declare a JSONRPC interface')
  return {
    url: found.url,
    protocolVersion: found.protocolVersion || A2A_PROTOCOL_VERSION,
  }
}

function extractText(value: unknown): string {
  if (!isRecord(value)) return ''
  const task = isRecord(value.task) ? value.task : undefined
  const message = isRecord(value.message)
    ? value.message
    : isRecord(task?.status) && isRecord(task.status.message)
      ? task.status.message
      : undefined
  const parts = Array.isArray(message?.parts)
    ? message.parts
    : Array.isArray(task?.artifacts)
      ? task.artifacts.flatMap((artifact) =>
          isRecord(artifact) && Array.isArray(artifact.parts) ? artifact.parts : [],
        )
      : []
  return parts
    .map((part) => (isRecord(part) && typeof part.text === 'string' ? part.text : ''))
    .filter(Boolean)
    .join('\n')
}

export class A2AHttpClient {
  private readonly fetcher: typeof fetch
  private readonly maxRedirects: number
  private readonly timeoutMs: number

  constructor(private readonly options: A2AClientOptions = {}) {
    this.fetcher = options.fetcher ?? fetch
    this.maxRedirects =
      options.maxRedirects != null && Number.isFinite(options.maxRedirects)
        ? Math.min(MAX_REDIRECTS, Math.max(0, Math.floor(options.maxRedirects)))
        : DEFAULT_MAX_REDIRECTS
    this.timeoutMs = normalizeTimeoutMs(options.timeoutMs)
  }

  async getAgentCard(agentCardUrl: string): Promise<A2AAgentCard> {
    return this.withTimeout(
      undefined,
      async (signal) => (await this.discoverAgentCard(agentCardUrl, signal)).card,
    )
  }

  async sendMessage(input: A2ASendInput): Promise<{
    card: A2AAgentCard
    result: A2ASendMessageResponse
    output: string
  }> {
    return this.withTimeout(input.timeoutMs, async (signal) => {
      const discovery = await this.discoverAgentCard(input.agentCardUrl, signal)
      const target = pickJsonRpcInterface(discovery.card)
      const targetUrl = new URL(target.url)
      if (targetUrl.username || targetUrl.password) {
        throw new Error('A2A SendMessage refused: URL credentials are not allowed')
      }
      const result = await this.fetchJson<A2AJsonRpcResponse>({
        url: target.url,
        label: 'A2A SendMessage',
        method: 'POST',
        signal,
        requiredOrigin: discovery.origin,
        headers: mergeHeaders(
          discovery.callerHeadersAllowed ? this.options.headers : undefined,
          discovery.callerHeadersAllowed ? input.headers : undefined,
          {
            'content-type': 'application/json',
            accept: 'application/json',
            'a2a-version': target.protocolVersion || A2A_PROTOCOL_VERSION,
          },
        ),
        body: JSON.stringify({
          jsonrpc: '2.0',
          id: randomUUID(),
          method: 'SendMessage',
          params: {
            message: textMessage(input.message, {
              contextId: input.contextId,
              taskId: input.taskId,
            }),
            configuration: {
              acceptedOutputModes: ['text/plain'],
            },
          },
        }),
      })
      if (result.value.error) {
        throw new Error(`A2A ${result.value.error.code}: ${result.value.error.message}`)
      }
      const sendResult = result.value.result as A2ASendMessageResponse
      return {
        card: discovery.card,
        result: sendResult,
        output: extractText(sendResult),
      }
    })
  }

  private async discoverAgentCard(
    agentCardUrl: string,
    signal?: AbortSignal,
  ): Promise<AgentCardDiscovery> {
    const url = cardUrlFromInput(agentCardUrl)
    const result = await this.fetchJson<A2AAgentCard>({
      url,
      label: 'A2A Agent Card fetch',
      method: 'GET',
      signal,
      headers: mergeHeaders(this.options.headers, { accept: 'application/json' }),
      crossOriginHeaders: { accept: 'application/json' },
    })
    return {
      card: result.value,
      origin: new URL(result.finalUrl).origin,
      callerHeadersAllowed: result.callerHeadersAllowed,
    }
  }

  private async withTimeout<T>(
    requestedTimeoutMs: number | undefined,
    task: (signal: AbortSignal) => Promise<T>,
  ): Promise<T> {
    const controller = new AbortController()
    const timeoutMs = normalizeTimeoutMs(requestedTimeoutMs, this.timeoutMs)
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    timer.unref?.()
    try {
      return await task(controller.signal)
    } finally {
      clearTimeout(timer)
    }
  }

  private async fetchJson<T>(input: {
    url: string
    label: string
    method: string
    headers: Record<string, string>
    crossOriginHeaders?: Record<string, string>
    requiredOrigin?: string
    body?: BodyInit | null
    signal?: AbortSignal
  }): Promise<{ value: T; finalUrl: string; callerHeadersAllowed: boolean }> {
    const fetched = await this.fetchWithRedirects(input)
    try {
      if (!fetched.response.ok) {
        throw new Error(`${input.label} failed: HTTP ${fetched.response.status}`)
      }
      let value: unknown
      try {
        value = await readBoundedJson(fetched.response)
      } catch (error) {
        throw new Error(
          `${input.label} failed: ${error instanceof Error ? error.message : String(error)}`,
        )
      }
      return {
        value: value as T,
        finalUrl: fetched.finalUrl,
        callerHeadersAllowed: fetched.callerHeadersAllowed,
      }
    } finally {
      await discardResponseBody(fetched.response)
      await fetched.dispatcher.close().catch(() => undefined)
    }
  }

  private async fetchWithRedirects(input: {
    url: string
    label: string
    method: string
    headers: Record<string, string>
    crossOriginHeaders?: Record<string, string>
    requiredOrigin?: string
    body?: BodyInit | null
    signal?: AbortSignal
  }): Promise<ValidatedFetchResult> {
    let currentUrl = input.url
    let currentMethod = input.method.toUpperCase()
    let currentBody = input.body
    let currentHeaders = input.headers
    let callerHeadersAllowed = true

    for (let redirects = 0; ; redirects += 1) {
      let resolution
      try {
        resolution = await assertPublicUrl(currentUrl, input.signal)
      } catch (error) {
        throw new Error(
          `${input.label} refused: ${error instanceof Error ? error.message : String(error)}`,
        )
      }
      if (resolution.url.username || resolution.url.password) {
        throw new Error(`${input.label} refused: URL credentials are not allowed`)
      }
      if (input.requiredOrigin && resolution.url.origin !== input.requiredOrigin) {
        throw new Error(
          `${input.label} refused: JSONRPC interface must use the Agent Card origin`,
        )
      }

      const dispatcher = createPinnedLookupDispatcher(resolution)
      let response: Response
      try {
        response = await this.fetcher(resolution.url.toString(), {
          method: currentMethod,
          headers: currentHeaders,
          body: currentBody,
          signal: input.signal,
          redirect: 'manual',
          dispatcher,
        } as FetchInitWithDispatcher)
      } catch (error) {
        await dispatcher.close().catch(() => undefined)
        throw error
      }

      const location = isRedirectStatus(response.status) ? response.headers.get('location') : null
      if (!location) {
        return {
          response,
          dispatcher,
          finalUrl: resolution.url.toString(),
          callerHeadersAllowed,
        }
      }

      if (redirects >= this.maxRedirects) {
        await discardResponseBody(response)
        await dispatcher.close().catch(() => undefined)
        throw new Error(`${input.label} failed: too many redirects (${this.maxRedirects})`)
      }

      let nextUrl: URL
      try {
        nextUrl = new URL(location, resolution.url)
      } catch (error) {
        await discardResponseBody(response)
        await dispatcher.close().catch(() => undefined)
        throw new Error(
          `${input.label} refused: invalid redirect URL (${error instanceof Error ? error.message : String(error)})`,
        )
      }
      if (resolution.url.protocol === 'https:' && nextUrl.protocol !== 'https:') {
        await discardResponseBody(response)
        await dispatcher.close().catch(() => undefined)
        throw new Error(`${input.label} refused: HTTPS redirects must not downgrade to HTTP`)
      }
      if (nextUrl.origin !== resolution.url.origin) {
        if (!input.crossOriginHeaders) {
          await discardResponseBody(response)
          await dispatcher.close().catch(() => undefined)
          throw new Error(`${input.label} refused: cross-origin redirects are not allowed`)
        }
        // Caller/config headers have unknown provenance and may carry secrets
        // under arbitrary vendor names. Rebuild from a safe client-owned set.
        currentHeaders = input.crossOriginHeaders
        callerHeadersAllowed = false
      }
      const redirected = redirectMethod(response.status, currentMethod, currentBody, currentHeaders)
      currentUrl = nextUrl.toString()
      currentMethod = redirected.method
      currentBody = redirected.body
      currentHeaders = redirected.headers
      await discardResponseBody(response)
      await dispatcher.close().catch(() => undefined)
    }
  }
}
