import { randomUUID } from 'node:crypto'
import { appendFile, chmod, mkdir, open } from 'node:fs/promises'
import { dirname } from 'node:path'
import type {
  ChatRequest,
  ChatResponse,
  ContentPart,
  Message,
  TokenUsage,
  ToolDefinition,
} from '@sepilotd/core'
import {
  normalizeLogRotationOptions,
  rotateLogFileIfNeeded,
  type LogRotationOptions,
} from '../utils/log-rotation.js'
import {
  createTraceRedactionContext,
  isTraceRedactionEnabled,
  redactSecretKeys,
} from './trace-redaction.js'

const DEFAULT_MAX_STRING_LENGTH = 16_000
const DEFAULT_STRUCTURE_LIMITS = { maxArrayItems: 64, maxObjectKeys: 64, maxDepth: 6 }
const FULL_STRUCTURE_LIMITS = { maxArrayItems: 1024, maxObjectKeys: 1024, maxDepth: 32 }

function traceStructureLimits() {
  return process.env.SEPILOT_TRACE_FULL === '1' ? FULL_STRUCTURE_LIMITS : DEFAULT_STRUCTURE_LIMITS
}

const ROTATION_CHECK_INTERVAL = 8

const DEFAULT_TRACE_ROTATION: LogRotationOptions = {
  enabled: true,
  maxBytes: 50 * 1024 * 1024,
  maxFiles: 3,
}

let traceFilePath: string | null = null
let writeQueue: Promise<void> = Promise.resolve()
let rotationOptions: LogRotationOptions = { ...DEFAULT_TRACE_ROTATION }
let writesSinceCheck = 0
let traceSequence = 0
const responseIds = new WeakMap<ChatResponse, string>()

const SAFE_TRACE_IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/
const TRACE_IDENTIFIER_KEYS = [
  'callId',
  'responseId',
  'executionId',
  'requestId',
  'runId',
  'sessionId',
  'toolCallId',
  'turnId',
  'event',
  'source',
  'status',
  'node',
  'mode',
  'graphId',
] as const

const IDENTIFIER_KEY_SET: ReadonlySet<string> = new Set(TRACE_IDENTIFIER_KEYS)

export function isSepilotDebugEnabled(
  env: Record<string, string | undefined> = process.env,
): boolean {
  return env.SEPILOT_DEBUG === '1'
}

function traceStringLimit(): number {
  return process.env.SEPILOT_TRACE_FULL === '1'
    ? Number.MAX_SAFE_INTEGER
    : DEFAULT_MAX_STRING_LENGTH
}

function truncateString(value: string, limit = traceStringLimit()): string {
  return value.length > limit
    ? `${value.slice(0, limit - 1)}…`
    : value
}

function sanitizeValue(value: unknown, depth = 0): unknown {
  const limits = traceStructureLimits()
  if (depth >= limits.maxDepth) {
    return '[max-depth]'
  }

  if (
    value === null
    || typeof value === 'number'
    || typeof value === 'boolean'
  ) {
    return value
  }

  if (typeof value === 'string') {
    return truncateString(value)
  }

  if (typeof value === 'bigint' || typeof value === 'symbol') {
    return String(value)
  }

  if (Array.isArray(value)) {
    const items = value
      .slice(0, limits.maxArrayItems)
      .map((item) => sanitizeValue(item, depth + 1))
    if (value.length > limits.maxArrayItems) {
      items.push(`[+${value.length - limits.maxArrayItems} more items]`)
    }
    return items
  }

  if (typeof value === 'object') {
    const entries = Object.entries(value)
    const sanitized = Object.fromEntries(
      entries
        .slice(0, limits.maxObjectKeys)
        .map(([key, item]) => [key, sanitizeValue(item, depth + 1)]),
    )
    if (entries.length > limits.maxObjectKeys) {
      sanitized.__truncated_keys__ = entries.length - limits.maxObjectKeys
    }
    return sanitized
  }

  if (typeof value === 'undefined') {
    return null
  }

  return String(value)
}

function stripLiveViewImages(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripLiveViewImages)
  if (!value || typeof value !== 'object') return value
  const out: Record<string, unknown> = {}
  for (const [key, inner] of Object.entries(value as Record<string, unknown>)) {
    out[key] =
      key === 'image' && typeof inner === 'string' && inner.startsWith('data:image/')
        ? '[image]'
        : stripLiveViewImages(inner)
  }
  return out
}

export function redactTraceEntry(entry: Record<string, unknown>): Record<string, unknown> {
  if (!isTraceRedactionEnabled()) {
    return entry
  }
  return redactSecretKeys(
    stripLiveViewImages(entry),
    createTraceRedactionContext(),
    '',
    0,
    {
      maxStringLength: traceStringLimit(),
      ...traceStructureLimits(),
    },
  ) as Record<string, unknown>
}

function restoreSafeTraceIdentifiers(
  original: Record<string, unknown>,
  redacted: Record<string, unknown>,
): Record<string, unknown> {
  return restoreIdentifiersDeep(original, redacted, 0) as Record<string, unknown>
}

function restoreIdentifiersDeep(original: unknown, redacted: unknown, depth: number): unknown {
  if (depth > traceStructureLimits().maxDepth) return redacted

  if (Array.isArray(original) && Array.isArray(redacted)) {
    return redacted.map((item, index) =>
      index < original.length ? restoreIdentifiersDeep(original[index], item, depth + 1) : item,
    )
  }

  if (!isPlainRecord(original) || !isPlainRecord(redacted)) return redacted

  const restored: Record<string, unknown> = { ...redacted }
  for (const [key, value] of Object.entries(redacted)) {
    const source = original[key]
    if (source === undefined) continue
    if (
      typeof source === 'string'
      && IDENTIFIER_KEY_SET.has(key)
      && SAFE_TRACE_IDENTIFIER.test(source)
    ) {
      restored[key] = source
      continue
    }
    restored[key] = restoreIdentifiersDeep(source, value, depth + 1)
  }
  return restored
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function sanitizeBinarySource(
  source: { type: 'base64' | 'url'; mediaType: string; data: string },
): Record<string, unknown> {
  if (source.type === 'url') {
    return {
      type: source.type,
      mediaType: source.mediaType,
      data: truncateString(source.data, 2_000),
    }
  }

  return {
    type: source.type,
    mediaType: source.mediaType,
    omitted: true,
    encodedLength: source.data.length,
  }
}

function sanitizeContentPart(part: ContentPart): Record<string, unknown> {
  if (part.type === 'text') {
    return {
      type: part.type,
      text: truncateString(part.text),
    }
  }

  return {
    type: part.type,
    source: sanitizeBinarySource(part.source),
  }
}

function sanitizeMessageContent(
  content: Message['content'] | undefined,
): string | Record<string, unknown>[] {
  if (typeof content === 'string') {
    return truncateString(content)
  }

  if (!Array.isArray(content)) {
    return ''
  }

  return content.map((part) => sanitizeContentPart(part))
}

function sanitizeMessage(message: Message): Record<string, unknown> {
  return {
    role: message.role,
    content: sanitizeMessageContent(message.content),
    toolCallId: message.toolCallId,
    toolCalls: sanitizeValue(message.toolCalls),
  }
}

function sanitizeToolDefinitions(
  tools?: ToolDefinition[],
): Array<Record<string, unknown>> | undefined {
  if (!tools?.length) {
    return undefined
  }

  return tools.map((tool) => ({
    name: tool.name,
    description: truncateString(tool.description, 4_000),
    inputSchema: sanitizeValue(tool.inputSchema),
  }))
}

function sanitizeRequest(request: ChatRequest): Record<string, unknown> {
  return {
    model: request.model,
    messages: request.messages.map((message) => sanitizeMessage(message)),
    systemPrompt: request.systemPrompt
      ? truncateString(request.systemPrompt)
      : undefined,
    tools: sanitizeToolDefinitions(request.tools),
    temperature: request.temperature,
    maxTokens: request.maxTokens,
    thinkingLevel: request.thinkingLevel,
    stopSequences: sanitizeValue(request.stopSequences),
    timeoutMs: request.timeoutMs,
  }
}

function sanitizeResponse(response: ChatResponse): Record<string, unknown> {
  return {
    message: sanitizeMessage(response.message),
    thinking: response.thinking
      ? truncateString(response.thinking)
      : undefined,
    usage: sanitizeValue(response.usage),
    finishReason: response.finishReason,
    raw: sanitizeValue(response.raw),
  }
}

let pendingTraceWrites: Promise<void> = Promise.resolve()

function trackTraceWrite(write: Promise<void>): void {
  pendingTraceWrites = pendingTraceWrites.then(
    () => write.catch(() => {}),
    () => write.catch(() => {}),
  )
}

export async function flushAgentTraceWrites(): Promise<void> {
  await pendingTraceWrites
  await writeQueue
}

async function writeEntry(entry: Record<string, unknown>): Promise<void> {
  if (!traceFilePath || !isSepilotDebugEnabled()) {
    return
  }

  const original = {
    traceVersion: 2,
    timestamp: new Date().toISOString(),
    sequence: ++traceSequence,
    pid: process.pid,
    ...entry,
  }
  const line = JSON.stringify(restoreSafeTraceIdentifiers(
    original,
    redactTraceEntry(original),
  ))

  const path = traceFilePath
  const shouldCheckRotation =
    rotationOptions.enabled && ++writesSinceCheck >= ROTATION_CHECK_INTERVAL
  if (shouldCheckRotation) writesSinceCheck = 0

  writeQueue = writeQueue.then(async () => {
    try {
      if (shouldCheckRotation) {
        await rotateLogFileIfNeeded(path, rotationOptions).catch(() => {})
      }
      await appendFile(path, `${line}\n`, { encoding: 'utf-8', mode: 0o600 })
    } catch {
    }
  })
  await writeQueue
}

export async function setAgentTraceLogFile(
  path: string | null,
  options?: Partial<LogRotationOptions>,
): Promise<void> {
  if (!path || !isSepilotDebugEnabled()) {
    traceFilePath = null
    return
  }

  const directory = dirname(path)
  rotationOptions = normalizeLogRotationOptions(options, DEFAULT_TRACE_ROTATION)
  await mkdir(directory, { recursive: true, mode: 0o700 })
  await chmod(directory, 0o700).catch(() => {})
  const handle = await open(path, 'a', 0o600)
  await handle.close()
  await chmod(path, 0o600).catch(() => {})
  await Promise.all(
    Array.from({ length: rotationOptions.maxFiles }, (_, index) =>
      chmod(`${path}.${index + 1}`, 0o600).catch(() => {})),
  )
  traceFilePath = path
  writesSinceCheck = 0
  traceSequence = 0
}

export function setAgentTraceRotation(options: Partial<LogRotationOptions>): void {
  rotationOptions = normalizeLogRotationOptions(options, rotationOptions)
}

export function getAgentTraceRotation(): LogRotationOptions {
  return { ...rotationOptions }
}

export interface LlmTraceLogParams {
  source: string
  mode?: string
  graphId?: string
  node?: string
  sessionId?: string
  provider: string
  model: string
  iteration?: number
  request: ChatRequest
  response?: ChatResponse
  error?: string
  meta?: Record<string, unknown>
}

function chunkString(value: string): string[] {
  const out: string[] = []
  for (let i = 0; i < value.length && out.length < 60; i += 16_000) {
    out.push(value.slice(i, i + 16_000))
  }
  return out
}

export async function logLlmCallTrace(
  params: LlmTraceLogParams,
): Promise<void> {
  if (!isSepilotDebugEnabled()) {
    return
  }

  const callId = randomUUID()
  let responseId: string | undefined
  if (params.response) {
    responseId = responseIds.get(params.response) ?? randomUUID()
    responseIds.set(params.response, responseId)
  }
  await writeEntry({
    event: 'llm.call',
    callId,
    responseId,
    source: params.source,
    mode: params.mode,
    graphId: params.graphId,
    node: params.node,
    sessionId: params.sessionId,
    provider: params.provider,
    model: params.model,
    iteration: params.iteration,
    request: sanitizeRequest(params.request),
    response: params.response ? sanitizeResponse(params.response) : undefined,
    error: params.error,
    meta: params.meta ? sanitizeValue(params.meta) : undefined,
  })

  {
    const systemPrompt = params.request.systemPrompt ?? ''
    const messagesJson = JSON.stringify(params.request.messages ?? [])
    const systemPromptChunks = chunkString(systemPrompt)
    const messagesJsonChunks = chunkString(messagesJson)
    const truncatedChars =
      Math.max(0, systemPrompt.length - 60 * 16_000)
      + Math.max(0, messagesJson.length - 60 * 16_000)

    await writeEntry({
      event: 'prompt.rendered',
      source: params.source,
      sessionId: params.sessionId,
      runId: params.meta?.runId as string | undefined,
      turnId: params.meta?.turnId as string | undefined,
      iteration: params.iteration,
      callId,
      data: {
        systemPromptChunks,
        messagesJsonChunks,
        toolNames: (params.request.tools ?? []).map((tool) => tool.name),
        truncatedChars,
      },
    })
  }
}

export function logLlmCallTraceDetached(params: LlmTraceLogParams): void {
  trackTraceWrite(logLlmCallTrace(params))
}

export interface AgentRunTraceLogParams {
  source: string
  status: string
  mode?: string
  graphId?: string
  sessionId?: string
  provider?: string
  model?: string
  iteration?: number
  usage?: TokenUsage
  output?: string
  error?: string
  meta?: Record<string, unknown>
}

export async function logAgentRunTrace(
  params: AgentRunTraceLogParams,
): Promise<void> {
  if (!isSepilotDebugEnabled()) {
    return
  }

  await writeEntry({
    event: 'agent.run',
    source: params.source,
    status: params.status,
    mode: params.mode,
    graphId: params.graphId,
    sessionId: params.sessionId,
    provider: params.provider,
    model: params.model,
    iteration: params.iteration,
    usage: params.usage ? sanitizeValue(params.usage) : undefined,
    output: params.output ? truncateString(params.output) : undefined,
    error: params.error,
    meta: params.meta ? sanitizeValue(params.meta) : undefined,
  })
}

export interface AgentDebugTraceParams {
  event: string
  source: string
  sessionId?: string
  runId?: string
  turnId?: string
  requestId?: string
  toolCallId?: string
  executionId?: string
  mode?: string
  graphId?: string
  node?: string
  iteration?: number
  status?: string
  durationMs?: number
  data?: Record<string, unknown>
}

export async function logAgentDebugTrace(
  params: AgentDebugTraceParams,
): Promise<void> {
  if (!isSepilotDebugEnabled()) {
    return
  }

  await writeEntry({
    event: params.event,
    source: params.source,
    sessionId: params.sessionId,
    runId: params.runId,
    turnId: params.turnId,
    requestId: params.requestId,
    toolCallId: params.toolCallId,
    executionId: params.executionId,
    mode: params.mode,
    graphId: params.graphId,
    node: params.node,
    iteration: params.iteration,
    status: params.status,
    durationMs: params.durationMs,
    data: params.data ? sanitizeValue(params.data) as Record<string, unknown> : undefined,
  })
}
