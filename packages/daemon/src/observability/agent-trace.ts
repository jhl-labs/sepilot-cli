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
const MAX_ARRAY_ITEMS = 64
const MAX_OBJECT_KEYS = 64
const MAX_DEPTH = 6

// Trace entries are large (full LLM payloads), so checking every 8 writes
// keeps overshoot bounded without dominating throughput.
const ROTATION_CHECK_INTERVAL = 8

// Trace files default to a larger ceiling than daemon.log because each entry
// carries a full LLM request/response. Operators who care about disk usage
// can override via `logging.trace` in config.yaml or the Settings UI.
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
// Auxiliary wrappers and their graph callers may log the same received
// response. Preserve both trace entries but expose an identity for usage
// reconciliation; equal-looking replies from separate requests remain distinct.
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
  // Structural fields, never free text: event/status/node/mode/graphId are
  // fixed short identifiers emitted by call sites in this file (event type
  // names, node/graph ids, run-status enums), not user or model content.
  // Long structural names (e.g. 'supervisor.criterion-evidence-review') can
  // still match LONG_TOKEN_RE and get wholesale-redacted; restore them when
  // the original value is identifier-shaped.
  'event',
  'source',
  'status',
  'node',
  'mode',
  'graphId',
] as const

const IDENTIFIER_KEY_SET: ReadonlySet<string> = new Set(TRACE_IDENTIFIER_KEYS)

/**
 * Expensive request/response diagnostics are opt-in. Operational journals
 * remain available without this switch; agent-trace and TUI debug logs do not.
 */
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
  if (depth >= MAX_DEPTH) {
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
      .slice(0, MAX_ARRAY_ITEMS)
      .map((item) => sanitizeValue(item, depth + 1))
    if (value.length > MAX_ARRAY_ITEMS) {
      items.push(`[+${value.length - MAX_ARRAY_ITEMS} more items]`)
    }
    return items
  }

  if (typeof value === 'object') {
    const entries = Object.entries(value)
    const sanitized = Object.fromEntries(
      entries
        .slice(0, MAX_OBJECT_KEYS)
        .map(([key, item]) => [key, sanitizeValue(item, depth + 1)]),
    )
    if (entries.length > MAX_OBJECT_KEYS) {
      sanitized.__truncated_keys__ = entries.length - MAX_OBJECT_KEYS
    }
    return sanitized
  }

  if (typeof value === 'undefined') {
    return null
  }

  return String(value)
}

function redactTraceEntry(entry: Record<string, unknown>): Record<string, unknown> {
  if (!isTraceRedactionEnabled()) {
    return entry
  }
  return redactSecretKeys(
    entry,
    createTraceRedactionContext(),
    '',
    0,
    {
      maxStringLength: traceStringLimit(),
      maxDepth: MAX_DEPTH,
      maxArrayItems: MAX_ARRAY_ITEMS,
      maxObjectKeys: MAX_OBJECT_KEYS,
    },
  ) as Record<string, unknown>
}

function restoreSafeTraceIdentifiers(
  original: Record<string, unknown>,
  redacted: Record<string, unknown>,
): Record<string, unknown> {
  return restoreIdentifiersDeep(original, redacted, 0) as Record<string, unknown>
}

/**
 * Structural identifiers are structural wherever they sit. A trace entry is a
 * tree — `agent.event` carries the real payload under `data`, graph nodes nest
 * their own envelopes — so restoring only the top-level keys left every nested
 * copy of the same identifier wholesale-redacted, which is what broke
 * tool-call correlation. Walk the original and the redacted value in parallel
 * and restore an identifier-keyed string at any depth, provided the original
 * is identifier-shaped (`SAFE_TRACE_IDENTIFIER`: no whitespace, bounded
 * length) so free text can never be reinstated through this path.
 *
 * Bounded by MAX_DEPTH like the redactor itself; anything deeper keeps the
 * redacted value.
 */
function restoreIdentifiersDeep(original: unknown, redacted: unknown, depth: number): unknown {
  if (depth > MAX_DEPTH) return redacted

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

/** Await every trace write started so far (detached writes included). */
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
      // Trace logging must never throw into the agent execution path.
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
  // Guard before touching `params`: sanitizing a full request/response is the
  // expensive part, and callers on the default (non-debug) path must not pay
  // for it. writeEntry re-checks, but only after the work is already done.
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
    // Chunking happens on the raw text; per-string redaction/truncation is
    // applied uniformly by writeEntry's redactTraceEntry pass below (each
    // chunk is <= 16,000 chars, within the default trace string limit), so
    // this does not double-redact or truncate below the chunk boundary.
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

/**
 * Fire-and-forget variant for call sites that must not add the trace write to
 * a latency-sensitive path (e.g. a promise raced against a timeout). Failures
 * are swallowed; `flushAgentTraceWrites` awaits the outstanding writes.
 */
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

/** Write one correlated, redacted diagnostic event when SEPILOT_DEBUG=1. */
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
