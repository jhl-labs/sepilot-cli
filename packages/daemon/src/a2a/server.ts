import { randomUUID, createHash } from 'node:crypto'
import { rejectPrivateLiteralUrl } from '../utils/ssrf-guard.js'
import {
  A2A_PROTOCOL_VERSION,
  A2A_SUPPORTED_VERSIONS,
  type A2AAgentCard,
  type A2AArtifact,
  type A2AAuthenticationInfo,
  type A2AJsonRpcRequest,
  type A2AJsonRpcResponse,
  type A2AListPushNotificationConfigsRequest,
  type A2AListPushNotificationConfigsResponse,
  type A2AListTasksRequest,
  type A2AListTasksResponse,
  type A2AMessage,
  type A2APart,
  type A2APushNotificationConfig,
  type A2ASendMessageRequest,
  type A2ASendMessageResponse,
  type A2AStreamResponse,
  type A2ATaskArtifactUpdateEvent,
  type A2ATaskPushNotificationConfig,
  type A2ATask,
  type A2ATaskState,
  type A2ATaskStatusUpdateEvent,
} from './types.js'

export class A2AError extends Error {
  constructor(
    readonly code: number,
    message: string,
    readonly reason?: string,
    readonly metadata?: Record<string, string>,
  ) {
    super(message)
  }
}

export interface A2ADispatchInput {
  prompt: string
  taskId: string
  contextId: string
  metadata?: Record<string, unknown>
  signal?: AbortSignal
}

export interface A2ADispatchResult {
  output: string
  status: 'completed' | 'failed' | 'truncated'
  sessionId?: string
  error?: string
}

export type A2ADispatcher = (input: A2ADispatchInput) => Promise<A2ADispatchResult>
export type A2ATaskEventListener = (event: A2AStreamResponse) => void

export interface A2APushDelivery {
  deliver(config: A2ATaskPushNotificationConfig, event: A2AStreamResponse): Promise<void>
}

export interface A2ATaskStoreOptions {
  pushDelivery?: A2APushDelivery
  /** Hard cap on retained tasks. Oldest terminal tasks are evicted first. */
  maxTasks?: number
  /**
   * Terminal tasks older than this (ms) are evicted eagerly. Opt-in: when
   * unset, only the count cap bounds growth (age-based eviction can surprise
   * callers that resurrect old tasks by id, so it is not applied by default).
   */
  terminalRetentionMs?: number
}

const DEFAULT_A2A_MAX_TASKS = 2_000

const INPUT_MODES = ['text/plain', 'application/json']
const OUTPUT_MODES = ['text/plain', 'application/json']
const TERMINAL_STATES = new Set<A2ATaskState>([
  'TASK_STATE_COMPLETED',
  'TASK_STATE_FAILED',
  'TASK_STATE_CANCELED',
  'TASK_STATE_REJECTED',
])

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function nowIso(): string {
  return new Date().toISOString()
}

function a2aErrorData(reason: string, metadata?: Record<string, string>) {
  return [{
    '@type': 'type.googleapis.com/google.rpc.ErrorInfo',
    reason,
    domain: 'a2a-protocol.org',
    metadata: {
      timestamp: nowIso(),
      ...(metadata ?? {}),
    },
  }]
}

function jsonRpcError(
  id: string | number | null | undefined,
  error: A2AError,
): A2AJsonRpcResponse {
  return {
    jsonrpc: '2.0',
    id: id ?? null,
    error: {
      code: error.code,
      message: error.message,
      data: error.reason ? a2aErrorData(error.reason, error.metadata) : undefined,
    },
  }
}

function cloneTask(task: A2ATask): A2ATask {
  return structuredClone(task)
}

function stateFromStatus(status: A2ADispatchResult['status']): A2ATaskState {
  // A truncated run still produced output (it hit an iteration/length limit),
  // so it maps to COMPLETED (with a truncation flag recorded in metadata) —
  // not FAILED, which would discard the partial answer.
  return status === 'completed' || status === 'truncated'
    ? 'TASK_STATE_COMPLETED'
    : 'TASK_STATE_FAILED'
}

function trimHistory(task: A2ATask, historyLength?: number): A2ATask {
  const next = cloneTask(task)
  if (historyLength === 0) {
    delete next.history
  } else if (typeof historyLength === 'number' && historyLength > 0 && next.history) {
    next.history = next.history.slice(-historyLength)
  }
  return next
}

function withoutArtifacts(task: A2ATask): A2ATask {
  const next = cloneTask(task)
  delete next.artifacts
  return next
}

function textPart(text: string, mediaType = 'text/plain'): A2APart {
  return { text, mediaType }
}

function outputArtifact(text: string): A2AArtifact {
  return {
    artifactId: randomUUID(),
    name: 'response',
    parts: [textPart(text)],
  }
}

function isTerminalState(state: A2ATaskState): boolean {
  return TERMINAL_STATES.has(state)
}

function statusUpdateFromTask(task: A2ATask): A2ATaskStatusUpdateEvent {
  return {
    taskId: task.id,
    contextId: task.contextId ?? '',
    status: task.status,
    final: isTerminalState(task.status.state) ? true : undefined,
  }
}

function artifactUpdateFromTask(task: A2ATask, artifact: A2AArtifact): A2ATaskArtifactUpdateEvent {
  return {
    taskId: task.id,
    contextId: task.contextId ?? '',
    artifact,
    lastChunk: isTerminalState(task.status.state) ? true : undefined,
  }
}

function eventIsTerminal(event: A2AStreamResponse): boolean {
  return 'statusUpdate' in event && event.statusUpdate.final === true
}

function validatePart(part: unknown, path: string): A2APart {
  if (!isRecord(part)) {
    throw new A2AError(-32602, `${path} must be an object`, 'INVALID_PARAMETERS')
  }
  const variants = ['text', 'data', 'url', 'raw'].filter((key) =>
    Object.hasOwn(part, key),
  )
  if (variants.length !== 1) {
    throw new A2AError(-32602, `${path} must contain exactly one of text, data, url, raw`, 'INVALID_PARAMETERS')
  }
  const mediaType = typeof part.mediaType === 'string' ? part.mediaType : undefined
  if (variants[0] === 'url' || variants[0] === 'raw') {
    throw new A2AError(
      -32005,
      'Only text and structured data A2A parts are supported',
      'CONTENT_TYPE_NOT_SUPPORTED',
      { part: path },
    )
  }
  if (mediaType && !INPUT_MODES.includes(mediaType)) {
    throw new A2AError(
      -32005,
      `Unsupported A2A part media type: ${mediaType}`,
      'CONTENT_TYPE_NOT_SUPPORTED',
      { mediaType },
    )
  }
  if (variants[0] === 'text') {
    if (typeof part.text !== 'string') {
      throw new A2AError(-32602, `${path}.text must be a string`, 'INVALID_PARAMETERS')
    }
    return {
      text: part.text,
      mediaType,
      metadata: isRecord(part.metadata) ? part.metadata : undefined,
    }
  }
  return {
    data: part.data,
    mediaType: mediaType ?? 'application/json',
    metadata: isRecord(part.metadata) ? part.metadata : undefined,
  }
}

function validateMessage(value: unknown): A2AMessage {
  if (!isRecord(value)) {
    throw new A2AError(-32602, 'params.message must be an object', 'INVALID_PARAMETERS')
  }
  if (typeof value.messageId !== 'string' || !value.messageId.trim()) {
    throw new A2AError(-32602, 'message.messageId is required', 'INVALID_PARAMETERS')
  }
  if (value.role !== 'ROLE_USER') {
    throw new A2AError(-32602, 'message.role must be ROLE_USER for SendMessage', 'INVALID_PARAMETERS')
  }
  if (!Array.isArray(value.parts) || value.parts.length === 0) {
    throw new A2AError(-32602, 'message.parts must contain at least one Part', 'INVALID_PARAMETERS')
  }
  return {
    messageId: value.messageId,
    contextId: typeof value.contextId === 'string' ? value.contextId : undefined,
    taskId: typeof value.taskId === 'string' ? value.taskId : undefined,
    role: 'ROLE_USER',
    parts: value.parts.map((part, index) => validatePart(part, `message.parts[${index}]`)),
    metadata: isRecord(value.metadata) ? value.metadata : undefined,
    extensions: Array.isArray(value.extensions)
      ? value.extensions.filter((item): item is string => typeof item === 'string')
      : undefined,
    referenceTaskIds: Array.isArray(value.referenceTaskIds)
      ? value.referenceTaskIds.filter((item): item is string => typeof item === 'string')
      : undefined,
  }
}

function validateSendParams(params: unknown): A2ASendMessageRequest {
  if (!isRecord(params)) {
    throw new A2AError(-32602, 'params must be a SendMessageRequest object', 'INVALID_PARAMETERS')
  }
  const configuration = isRecord(params.configuration) ? params.configuration : undefined
  const acceptedOutputModes = Array.isArray(configuration?.acceptedOutputModes)
    ? configuration.acceptedOutputModes.filter((item): item is string => typeof item === 'string')
    : undefined
  if (acceptedOutputModes?.length && !acceptedOutputModes.some((mode) => OUTPUT_MODES.includes(mode))) {
    throw new A2AError(
      -32005,
      'No acceptedOutputModes value is supported by this agent',
      'CONTENT_TYPE_NOT_SUPPORTED',
      { acceptedOutputModes: acceptedOutputModes.join(',') },
    )
  }
  return {
    tenant: typeof params.tenant === 'string' ? params.tenant : undefined,
    message: validateMessage(params.message),
    metadata: isRecord(params.metadata) ? params.metadata : undefined,
    configuration: configuration
      ? {
          acceptedOutputModes,
          historyLength: typeof configuration.historyLength === 'number'
            ? Math.max(0, Math.floor(configuration.historyLength))
            : undefined,
          returnImmediately: configuration.returnImmediately === true,
          taskPushNotificationConfig: configuration.taskPushNotificationConfig,
        }
      : undefined,
  }
}

function validateAuthenticationInfo(value: unknown): A2AAuthenticationInfo | undefined {
  if (value === undefined) return undefined
  if (!isRecord(value)) {
    throw new A2AError(-32602, 'pushNotificationConfig.authentication must be an object', 'INVALID_PARAMETERS')
  }
  if (typeof value.scheme !== 'string' || !value.scheme.trim()) {
    throw new A2AError(-32602, 'pushNotificationConfig.authentication.scheme is required', 'INVALID_PARAMETERS')
  }
  return {
    scheme: value.scheme.trim(),
    credentials: typeof value.credentials === 'string' ? value.credentials : undefined,
  }
}

function validatePushUrl(value: unknown): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new A2AError(-32602, 'pushNotificationConfig.url is required', 'INVALID_PARAMETERS')
  }
  let parsed: URL
  try {
    parsed = new URL(value)
  } catch {
    throw new A2AError(-32602, 'pushNotificationConfig.url must be absolute', 'INVALID_PARAMETERS')
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new A2AError(-32602, 'pushNotificationConfig.url must use http or https', 'INVALID_PARAMETERS')
  }
  if (parsed.username || parsed.password) {
    throw new A2AError(-32602, 'pushNotificationConfig.url must not contain credentials', 'INVALID_PARAMETERS')
  }
  try {
    rejectPrivateLiteralUrl(parsed)
  } catch {
    throw new A2AError(
      -32602,
      'pushNotificationConfig.url must not target a private/loopback address',
      'INVALID_PARAMETERS',
    )
  }
  return parsed.toString()
}

function validatePushConfig(value: unknown): A2APushNotificationConfig {
  if (!isRecord(value)) {
    throw new A2AError(-32602, 'pushNotificationConfig must be an object', 'INVALID_PARAMETERS')
  }
  return {
    id: typeof value.id === 'string' && value.id.trim()
      ? value.id.trim()
      : randomUUID(),
    url: validatePushUrl(value.url),
    authentication: validateAuthenticationInfo(value.authentication),
    metadata: isRecord(value.metadata) ? value.metadata : undefined,
  }
}

function validateTaskPushConfig(value: unknown, taskIdHint?: string): A2ATaskPushNotificationConfig {
  if (!isRecord(value)) {
    throw new A2AError(-32602, 'params must be a TaskPushNotificationConfig object', 'INVALID_PARAMETERS')
  }
  const taskId = typeof value.taskId === 'string' && value.taskId.trim()
    ? value.taskId.trim()
    : taskIdHint
  if (!taskId) {
    throw new A2AError(-32602, 'taskId is required', 'INVALID_PARAMETERS')
  }
  const configValue = isRecord(value.pushNotificationConfig)
    ? value.pushNotificationConfig
    : isRecord(value.config)
      ? value.config
      : value
  return {
    taskId,
    pushNotificationConfig: validatePushConfig(configValue),
  }
}

function messageToPrompt(message: A2AMessage): string {
  return message.parts
    .map((part) => {
      if ('text' in part) return part.text
      if ('data' in part) return JSON.stringify(part.data)
      return ''
    })
    .filter(Boolean)
    .join('\n')
    .trim()
}

class A2AEventQueue implements AsyncIterable<A2AStreamResponse> {
  private readonly queue: A2AStreamResponse[] = []
  private notify: (() => void) | null = null
  private closed = false

  push(event: A2AStreamResponse): void {
    if (this.closed) return
    this.queue.push(event)
    this.notify?.()
    this.notify = null
  }

  close(): void {
    this.closed = true
    this.notify?.()
    this.notify = null
  }

  async *[Symbol.asyncIterator](): AsyncIterator<A2AStreamResponse> {
    while (!this.closed || this.queue.length > 0) {
      const next = this.queue.shift()
      if (next) {
        yield next
        if (eventIsTerminal(next)) return
        continue
      }
      await new Promise<void>((resolve) => {
        this.notify = resolve
      })
    }
  }
}

export class A2ATaskStore {
  private readonly tasks = new Map<string, A2ATask>()
  private readonly listeners = new Map<string, Set<A2ATaskEventListener>>()
  private readonly pushConfigs = new Map<string, Map<string, A2ATaskPushNotificationConfig>>()

  constructor(
    initialTasks: readonly A2ATask[] = [],
    private readonly options: A2ATaskStoreOptions = {},
  ) {
    for (const task of initialTasks) {
      this.tasks.set(task.id, cloneTask(task))
    }
  }

  save(task: A2ATask): A2ATask {
    this.tasks.set(task.id, cloneTask(task))
    const saved = cloneTask(task)
    this.emitTaskEvents(saved)
    this.evict(task.id)
    return saved
  }

  /**
   * Bound retained tasks by age and count. Terminal tasks past the retention
   * window are dropped; if still over the cap, the oldest terminal (then oldest
   * overall) tasks are evicted. The just-saved task id is never evicted, and
   * tasks with a live push-config are preferentially kept. Returns nothing.
   */
  protected evict(protectId?: string): void {
    const maxTasks = this.options.maxTasks ?? DEFAULT_A2A_MAX_TASKS
    const retentionMs = this.options.terminalRetentionMs
    const now = Date.now()

    if (typeof retentionMs === 'number' && retentionMs > 0) {
      for (const [id, task] of this.tasks) {
        if (id === protectId) continue
        if (!isTerminalState(task.status.state)) continue
        const ts = Date.parse(task.status.timestamp ?? '')
        if (Number.isFinite(ts) && now - ts > retentionMs) {
          this.tasks.delete(id)
        }
      }
    }

    if (this.tasks.size <= maxTasks) return
    const candidates = [...this.tasks.entries()]
      .filter(([id]) => id !== protectId)
      .map(([id, task]) => ({
        id,
        terminal: isTerminalState(task.status.state),
        ts: Date.parse(task.status.timestamp ?? '') || 0,
      }))
      // Evict terminal tasks before non-terminal, oldest first within each group.
      .sort((a, b) => (a.terminal === b.terminal ? a.ts - b.ts : a.terminal ? -1 : 1))

    let over = this.tasks.size - maxTasks
    for (const candidate of candidates) {
      if (over <= 0) break
      this.tasks.delete(candidate.id)
      over -= 1
    }
  }

  get(id: string): A2ATask | null {
    const task = this.tasks.get(id)
    return task ? cloneTask(task) : null
  }

  update(id: string, mutator: (task: A2ATask) => A2ATask): A2ATask | null {
    const current = this.tasks.get(id)
    if (!current) return null
    const next = mutator(cloneTask(current))
    this.tasks.set(id, cloneTask(next))
    const saved = cloneTask(next)
    this.emitTaskEvents(saved)
    return saved
  }

  protected snapshotTasks(): A2ATask[] {
    return Array.from(this.tasks.values()).map(cloneTask)
  }

  protected restorePushConfigs(configs: readonly A2ATaskPushNotificationConfig[]): void {
    this.pushConfigs.clear()
    for (const config of configs) {
      const byTask = this.pushConfigs.get(config.taskId) ?? new Map<string, A2ATaskPushNotificationConfig>()
      byTask.set(config.pushNotificationConfig.id, structuredClone(config))
      this.pushConfigs.set(config.taskId, byTask)
    }
  }

  protected snapshotPushConfigs(): A2ATaskPushNotificationConfig[] {
    return Array.from(this.pushConfigs.values())
      .flatMap((configs) => Array.from(configs.values()).map((config) => structuredClone(config)))
  }

  subscribe(taskId: string, listener: A2ATaskEventListener): () => void {
    const listeners = this.listeners.get(taskId) ?? new Set<A2ATaskEventListener>()
    listeners.add(listener)
    this.listeners.set(taskId, listeners)
    return () => {
      listeners.delete(listener)
      if (listeners.size === 0) this.listeners.delete(taskId)
    }
  }

  createPushNotificationConfig(input: A2ATaskPushNotificationConfig): A2ATaskPushNotificationConfig {
    if (!this.tasks.has(input.taskId)) {
      throw new A2AError(-32001, 'Task not found', 'TASK_NOT_FOUND', { taskId: input.taskId })
    }
    const config = structuredClone(input)
    const byTask = this.pushConfigs.get(config.taskId) ?? new Map<string, A2ATaskPushNotificationConfig>()
    byTask.set(config.pushNotificationConfig.id, config)
    this.pushConfigs.set(config.taskId, byTask)
    this.afterPushConfigMutation()
    return structuredClone(config)
  }

  getPushNotificationConfig(taskId: string, id: string): A2ATaskPushNotificationConfig | null {
    return structuredClone(this.pushConfigs.get(taskId)?.get(id) ?? null)
  }

  listPushNotificationConfigs(input: A2AListPushNotificationConfigsRequest): A2AListPushNotificationConfigsResponse {
    const pageSize = Math.max(1, Math.min(100, Math.floor(input.pageSize ?? 50)))
    const offset = Math.max(0, Number.parseInt(input.pageToken ?? '0', 10) || 0)
    const configs = Array.from(this.pushConfigs.get(input.taskId)?.values() ?? [])
      .map((config) => structuredClone(config))
    const nextOffset = offset + pageSize
    return {
      configs: configs.slice(offset, nextOffset),
      nextPageToken: nextOffset < configs.length ? String(nextOffset) : '',
    }
  }

  deletePushNotificationConfig(taskId: string, id: string): A2ATaskPushNotificationConfig | null {
    const byTask = this.pushConfigs.get(taskId)
    const existing = byTask?.get(id)
    if (!byTask || !existing) return null
    byTask.delete(id)
    if (byTask.size === 0) this.pushConfigs.delete(taskId)
    this.afterPushConfigMutation()
    return structuredClone(existing)
  }

  protected afterPushConfigMutation(): void {
    // Subclasses with persistence override this hook.
  }

  list(input: A2AListTasksRequest = {}): A2AListTasksResponse {
    const pageSize = Math.max(1, Math.min(100, Math.floor(input.pageSize ?? 50)))
    const offset = Math.max(0, Number.parseInt(input.pageToken ?? '0', 10) || 0)
    let tasks = this.snapshotTasks()
    if (input.contextId) tasks = tasks.filter((task) => task.contextId === input.contextId)
    if (input.status) tasks = tasks.filter((task) => task.status.state === input.status)
    if (input.statusTimestampAfter) {
      const min = Date.parse(input.statusTimestampAfter)
      if (Number.isFinite(min)) {
        tasks = tasks.filter((task) => Date.parse(task.status.timestamp ?? '') >= min)
      }
    }
    tasks.sort((a, b) => (b.status.timestamp ?? '').localeCompare(a.status.timestamp ?? ''))
    const totalSize = tasks.length
    const page = tasks
      .slice(offset, offset + pageSize)
      .map((task) => {
        const withHistory = trimHistory(task, input.historyLength)
        return input.includeArtifacts ? withHistory : withoutArtifacts(withHistory)
      })
    const nextOffset = offset + pageSize
    return {
      tasks: page,
      nextPageToken: nextOffset < totalSize ? String(nextOffset) : '',
      pageSize,
      totalSize,
    }
  }

  private emitTaskEvents(task: A2ATask): void {
    const events: A2AStreamResponse[] = [
      ...(task.artifacts ?? []).map((artifact): A2AStreamResponse => ({
        artifactUpdate: artifactUpdateFromTask(task, artifact),
      })),
      { statusUpdate: statusUpdateFromTask(task) },
    ]
    for (const event of events) {
      for (const listener of this.listeners.get(task.id) ?? []) {
        listener(event)
      }
      this.deliverPush(task.id, event)
    }
    if (isTerminalState(task.status.state)) {
      this.pushConfigs.delete(task.id)
      this.afterPushConfigMutation()
    }
  }

  private deliverPush(taskId: string, event: A2AStreamResponse): void {
    const delivery = this.options.pushDelivery
    if (!delivery) return
    for (const config of this.pushConfigs.get(taskId)?.values() ?? []) {
      void delivery.deliver(structuredClone(config), event).catch(() => {})
    }
  }
}

export interface A2AServiceOptions {
  taskStore: A2ATaskStore
  dispatch: A2ADispatcher
  /** Max concurrent in-flight A2A dispatches. Defaults to env / 4. */
  maxConcurrent?: number
}

const DEFAULT_A2A_MAX_CONCURRENT = 4

export function resolveA2AMaxConcurrent(
  env: NodeJS.ProcessEnv = process.env,
): number {
  const raw = Number(env.SEPILOTD_A2A_MAX_CONCURRENT)
  return Number.isFinite(raw) && raw >= 1 ? Math.floor(raw) : DEFAULT_A2A_MAX_CONCURRENT
}

export class A2AService {
  private readonly runningTaskAbortControllers = new Map<string, AbortController>()
  private readonly maxConcurrent: number
  private inFlight = 0

  constructor(private readonly options: A2AServiceOptions) {
    this.maxConcurrent = options.maxConcurrent ?? resolveA2AMaxConcurrent()
  }

  /** Reserve a dispatch slot; throws RESOURCE_EXHAUSTED when at capacity. */
  private acquireSlot(): void {
    if (this.inFlight >= this.maxConcurrent) {
      throw new A2AError(
        -32003,
        `A2A dispatch capacity exceeded (max ${this.maxConcurrent} concurrent)`,
        'RESOURCE_EXHAUSTED',
      )
    }
    this.inFlight += 1
  }

  private releaseSlot(): void {
    if (this.inFlight > 0) this.inFlight -= 1
  }

  async handleJsonRpc(
    body: unknown,
    options: { version: string },
  ): Promise<A2AJsonRpcResponse> {
    const request = isRecord(body) ? body as unknown as A2AJsonRpcRequest : null
    const id = request?.id ?? null
    try {
      if (!A2A_SUPPORTED_VERSIONS.has(options.version)) {
        throw new A2AError(
          -32009,
          `A2A protocol version not supported: ${options.version}`,
          'VERSION_NOT_SUPPORTED',
          { version: options.version },
        )
      }
      if (!request || request.jsonrpc !== '2.0' || typeof request.method !== 'string') {
        throw new A2AError(-32600, 'Request payload validation error', 'INVALID_REQUEST')
      }
      if (!Object.hasOwn(request, 'id')) {
        throw new A2AError(-32600, 'A2A JSON-RPC requests must include id', 'INVALID_REQUEST')
      }
      const result = await this.dispatchMethod(request.method, request.params)
      return { jsonrpc: '2.0', id, result }
    } catch (error) {
      if (error instanceof A2AError) return jsonRpcError(id, error)
      return jsonRpcError(id, new A2AError(-32603, error instanceof Error ? error.message : String(error)))
    }
  }

  async *handleJsonRpcStream(
    body: unknown,
    options: { version: string; signal?: AbortSignal },
  ): AsyncIterable<A2AJsonRpcResponse> {
    const request = isRecord(body) ? body as unknown as A2AJsonRpcRequest : null
    const id = request?.id ?? null
    try {
      if (!A2A_SUPPORTED_VERSIONS.has(options.version)) {
        throw new A2AError(
          -32009,
          `A2A protocol version not supported: ${options.version}`,
          'VERSION_NOT_SUPPORTED',
          { version: options.version },
        )
      }
      if (!request || request.jsonrpc !== '2.0' || typeof request.method !== 'string') {
        throw new A2AError(-32600, 'Request payload validation error', 'INVALID_REQUEST')
      }
      if (!Object.hasOwn(request, 'id')) {
        throw new A2AError(-32600, 'A2A JSON-RPC requests must include id', 'INVALID_REQUEST')
      }
      const stream = request.method === 'SendStreamingMessage'
        ? this.sendStreamingMessage(validateSendParams(request.params), options.signal)
        : request.method === 'SubscribeToTask'
          ? this.subscribeToTask(request.params)
          : null
      if (!stream) {
        throw new A2AError(-32601, 'Method not found')
      }
      for await (const event of stream) {
        yield { jsonrpc: '2.0', id, result: event }
      }
    } catch (error) {
      yield error instanceof A2AError
        ? jsonRpcError(id, error)
        : jsonRpcError(id, new A2AError(-32603, error instanceof Error ? error.message : String(error)))
    }
  }

  private async dispatchMethod(method: string, params: unknown): Promise<unknown> {
    switch (method) {
      case 'SendMessage':
      case 'message/send':
        return this.sendMessage(validateSendParams(params))
      case 'GetTask':
      case 'tasks/get':
        return this.getTask(params)
      case 'ListTasks':
      case 'tasks/list':
        return this.listTasks(params)
      case 'CancelTask':
      case 'tasks/cancel':
        return this.cancelTask(params)
      case 'CreateTaskPushNotificationConfig':
        return this.createPushNotificationConfig(params)
      case 'ListTaskPushNotificationConfigs':
        return this.listPushNotificationConfigs(params)
      case 'GetTaskPushNotificationConfig':
        return this.getPushNotificationConfig(params)
      case 'DeleteTaskPushNotificationConfig':
        return this.deletePushNotificationConfig(params)
      case 'SendStreamingMessage':
      case 'SubscribeToTask':
        throw new A2AError(-32004, `${method} must be requested as an A2A SSE stream`, 'UNSUPPORTED_OPERATION')
      case 'GetExtendedAgentCard':
        throw new A2AError(-32007, 'Extended Agent Card is not configured', 'EXTENDED_AGENT_CARD_NOT_CONFIGURED')
      default:
        throw new A2AError(-32601, 'Method not found')
    }
  }

  private async sendMessage(input: A2ASendMessageRequest): Promise<A2ASendMessageResponse> {
    // Cap concurrent dispatches before creating the task so an authenticated
    // caller cannot accumulate unbounded concurrent agent runs.
    this.acquireSlot()
    let prepared: { task: A2ATask; prompt: string }
    try {
      prepared = this.createTask(input)
    } catch (error) {
      this.releaseSlot()
      throw error
    }
    const run = () => this.runTask(prepared, input)

    if (input.configuration?.returnImmediately) {
      void run()
      return { task: trimHistory(prepared.task, input.configuration.historyLength) }
    }

    const finalTask = await run()
    if (!finalTask) throw new A2AError(-32001, 'Task not found', 'TASK_NOT_FOUND', { taskId: prepared.task.id })
    return { task: trimHistory(finalTask, input.configuration?.historyLength) }
  }

  private createTask(input: A2ASendMessageRequest): { task: A2ATask; prompt: string } {
    const prompt = messageToPrompt(input.message)
    if (!prompt) {
      throw new A2AError(-32602, 'message.parts did not contain prompt content', 'INVALID_PARAMETERS')
    }
    const contextId = input.message.contextId ?? randomUUID()
    // A caller-supplied taskId must not collide with an existing task: reusing
    // an arbitrary id would overwrite (hijack) a terminal task or a task owned
    // by another context, and concurrent runs of the same id would share an
    // abort controller and interleave saves. Fail closed.
    const providedTaskId = input.message.taskId
    if (providedTaskId !== undefined && this.options.taskStore.get(providedTaskId)) {
      throw new A2AError(
        -32602,
        'taskId already exists; A2A does not allow reusing or overwriting an existing task id',
        'INVALID_PARAMETERS',
        { taskId: providedTaskId },
      )
    }
    const taskId = providedTaskId ?? randomUUID()
    const task: A2ATask = {
      id: taskId,
      contextId,
      status: {
        state: 'TASK_STATE_WORKING',
        timestamp: nowIso(),
      },
      history: [{
        ...input.message,
        contextId,
        taskId,
      }],
      metadata: {
        ...(input.metadata ?? {}),
        tenant: input.tenant,
      },
    }
    const pushConfig = input.configuration?.taskPushNotificationConfig
      ? validateTaskPushConfig(input.configuration.taskPushNotificationConfig, taskId)
      : undefined
    if (pushConfig && pushConfig.taskId !== taskId) {
      throw new A2AError(
        -32602,
        'taskPushNotificationConfig.taskId must match the created task',
        'INVALID_PARAMETERS',
      )
    }
    this.options.taskStore.save(task)
    if (pushConfig) {
      this.options.taskStore.createPushNotificationConfig(pushConfig)
    }
    return { task, prompt }
  }

  private async runTask(
    prepared: { task: A2ATask; prompt: string },
    input: A2ASendMessageRequest,
  ): Promise<A2ATask | null> {
    const { task, prompt } = prepared
    // Guard against a second run reusing a taskId whose run is still in flight
    // (would share/replace the abort controller and interleave task saves).
    // The caller has already reserved a concurrency slot, so release it here.
    if (this.runningTaskAbortControllers.has(task.id)) {
      this.releaseSlot()
      throw new A2AError(
        -32602,
        'Task is already running',
        'INVALID_PARAMETERS',
        { taskId: task.id },
      )
    }
    const abortController = new AbortController()
    this.runningTaskAbortControllers.set(task.id, abortController)
    try {
      const result = await this.options.dispatch({
        prompt,
        taskId: task.id,
        contextId: task.contextId ?? '',
        metadata: input.metadata,
        signal: abortController.signal,
      })
      const current = this.options.taskStore.get(task.id)
      if (!current || current.status.state === 'TASK_STATE_CANCELED') return current
      const finalText = result.output || result.error || ''
      const agentMessage: A2AMessage = {
        messageId: randomUUID(),
        contextId: task.contextId,
        taskId: task.id,
        role: 'ROLE_AGENT',
        parts: [textPart(finalText)],
        metadata: result.sessionId ? { sepilotdSessionId: result.sessionId } : undefined,
      }
      return this.options.taskStore.save({
        ...current,
        status: {
          state: stateFromStatus(result.status),
          message: agentMessage,
          timestamp: nowIso(),
        },
        artifacts: finalText ? [outputArtifact(finalText)] : undefined,
        history: [...(current.history ?? []), agentMessage],
        metadata: {
          ...(current.metadata ?? {}),
          sepilotdSessionId: result.sessionId,
          error: result.error,
          truncated: result.status === 'truncated' ? true : undefined,
        },
      })
    } catch (error) {
      const current = this.options.taskStore.get(task.id)
      if (!current || current.status.state === 'TASK_STATE_CANCELED') {
        return current
      }
      return this.options.taskStore.update(task.id, (current) => {
        const message: A2AMessage = {
          messageId: randomUUID(),
          contextId: current.contextId,
          taskId: current.id,
          role: 'ROLE_AGENT',
          parts: [textPart(error instanceof Error ? error.message : String(error))],
        }
        return {
          ...current,
          status: {
            state: 'TASK_STATE_FAILED',
            timestamp: nowIso(),
            message,
          },
          history: [...(current.history ?? []), message],
        }
      })
    } finally {
      this.releaseSlot()
      if (this.runningTaskAbortControllers.get(task.id) === abortController) {
        this.runningTaskAbortControllers.delete(task.id)
      }
    }
  }

  private async *sendStreamingMessage(
    input: A2ASendMessageRequest,
    signal?: AbortSignal,
  ): AsyncIterable<A2AStreamResponse> {
    this.acquireSlot()
    let prepared: { task: A2ATask; prompt: string }
    try {
      prepared = this.createTask(input)
    } catch (error) {
      this.releaseSlot()
      throw error
    }
    const queue = new A2AEventQueue()
    const unsubscribe = this.options.taskStore.subscribe(prepared.task.id, (event) => queue.push(event))
    // Abort the underlying run and end the stream when the client disconnects.
    const onAbort = () => {
      this.runningTaskAbortControllers.get(prepared.task.id)?.abort()
      queue.close()
    }
    if (signal) {
      if (signal.aborted) onAbort()
      else signal.addEventListener('abort', onAbort, { once: true })
    }
    void this.runTask(prepared, input).finally(() => queue.close())
    try {
      yield { task: trimHistory(prepared.task, input.configuration?.historyLength) }
      for await (const event of queue) {
        yield event
      }
    } finally {
      signal?.removeEventListener('abort', onAbort)
      unsubscribe()
      queue.close()
    }
  }

  private async *subscribeToTask(params: unknown): AsyncIterable<A2AStreamResponse> {
    if (!isRecord(params) || typeof params.id !== 'string') {
      throw new A2AError(-32602, 'SubscribeToTask params.id is required', 'INVALID_PARAMETERS')
    }
    const queue = new A2AEventQueue()
    const unsubscribe = this.options.taskStore.subscribe(params.id, (event) => queue.push(event))
    try {
      const task = this.options.taskStore.get(params.id)
      if (!task) throw new A2AError(-32001, 'Task not found', 'TASK_NOT_FOUND', { taskId: params.id })
      if (isTerminalState(task.status.state)) {
        throw new A2AError(-32004, 'Cannot subscribe to a terminal task', 'UNSUPPORTED_OPERATION', { taskId: params.id })
      }
      yield { task: trimHistory(task, typeof params.historyLength === 'number' ? params.historyLength : undefined) }
      for await (const event of queue) {
        yield event
      }
    } finally {
      unsubscribe()
      queue.close()
    }
  }

  private getTask(params: unknown): A2ATask {
    if (!isRecord(params) || typeof params.id !== 'string') {
      throw new A2AError(-32602, 'GetTask params.id is required', 'INVALID_PARAMETERS')
    }
    const task = this.options.taskStore.get(params.id)
    if (!task) throw new A2AError(-32001, 'Task not found', 'TASK_NOT_FOUND', { taskId: params.id })
    return trimHistory(task, typeof params.historyLength === 'number' ? params.historyLength : undefined)
  }

  private listTasks(params: unknown): A2AListTasksResponse {
    if (params !== undefined && !isRecord(params)) {
      throw new A2AError(-32602, 'ListTasks params must be an object', 'INVALID_PARAMETERS')
    }
    return this.options.taskStore.list(params as A2AListTasksRequest | undefined)
  }

  private cancelTask(params: unknown): A2ATask {
    if (!isRecord(params) || typeof params.id !== 'string') {
      throw new A2AError(-32602, 'CancelTask params.id is required', 'INVALID_PARAMETERS')
    }
    const task = this.options.taskStore.get(params.id)
    if (!task) throw new A2AError(-32001, 'Task not found', 'TASK_NOT_FOUND', { taskId: params.id })
    if (TERMINAL_STATES.has(task.status.state)) {
      throw new A2AError(-32002, 'Task is not cancelable', 'TASK_NOT_CANCELABLE', { taskId: params.id })
    }
    const cancelled = this.options.taskStore.save({
      ...task,
      status: {
        state: 'TASK_STATE_CANCELED',
        timestamp: nowIso(),
      },
    })
    this.runningTaskAbortControllers.get(task.id)?.abort()
    return cancelled
  }

  private createPushNotificationConfig(params: unknown): A2ATaskPushNotificationConfig {
    const config = validateTaskPushConfig(params)
    return this.options.taskStore.createPushNotificationConfig(config)
  }

  private getPushNotificationConfig(params: unknown): A2ATaskPushNotificationConfig {
    if (!isRecord(params) || typeof params.taskId !== 'string' || typeof params.id !== 'string') {
      throw new A2AError(-32602, 'GetTaskPushNotificationConfig params.taskId and params.id are required', 'INVALID_PARAMETERS')
    }
    const found = this.options.taskStore.getPushNotificationConfig(params.taskId, params.id)
    if (!found) {
      throw new A2AError(-32001, 'Push notification config not found', 'TASK_NOT_FOUND', {
        taskId: params.taskId,
        configId: params.id,
      })
    }
    return found
  }

  private listPushNotificationConfigs(params: unknown): A2AListPushNotificationConfigsResponse {
    if (!isRecord(params) || typeof params.taskId !== 'string') {
      throw new A2AError(-32602, 'ListTaskPushNotificationConfigs params.taskId is required', 'INVALID_PARAMETERS')
    }
    if (!this.options.taskStore.get(params.taskId)) {
      throw new A2AError(-32001, 'Task not found', 'TASK_NOT_FOUND', { taskId: params.taskId })
    }
    return this.options.taskStore.listPushNotificationConfigs(params as unknown as A2AListPushNotificationConfigsRequest)
  }

  private deletePushNotificationConfig(params: unknown): A2ATaskPushNotificationConfig {
    if (!isRecord(params) || typeof params.taskId !== 'string' || typeof params.id !== 'string') {
      throw new A2AError(-32602, 'DeleteTaskPushNotificationConfig params.taskId and params.id are required', 'INVALID_PARAMETERS')
    }
    const deleted = this.options.taskStore.deletePushNotificationConfig(params.taskId, params.id)
    if (!deleted) {
      throw new A2AError(-32001, 'Push notification config not found', 'TASK_NOT_FOUND', {
        taskId: params.taskId,
        configId: params.id,
      })
    }
    return deleted
  }
}

export function buildA2AAgentCard(input: {
  baseUrl: string
  authRequired: boolean
  version?: string
}): A2AAgentCard {
  const rpcUrl = new URL('/api/v1/a2a', input.baseUrl).toString()
  const securitySchemes = input.authRequired
    ? {
        sepilotdBearer: {
          httpAuthSecurityScheme: {
            scheme: 'Bearer',
            bearerFormat: 'sepilotd daemon token',
            description: 'Use the local daemon bearer token for A2A JSON-RPC calls.',
          },
        },
      }
    : undefined
  return {
    name: 'sepilotd',
    description: 'Local sepilotd agent runtime exposed through the Agent2Agent protocol.',
    supportedInterfaces: [{
      url: rpcUrl,
      protocolBinding: 'JSONRPC',
      protocolVersion: A2A_PROTOCOL_VERSION,
    }],
    version: input.version ?? '1.0.0',
    capabilities: {
      streaming: true,
      pushNotifications: true,
      extendedAgentCard: false,
    },
    securitySchemes,
    securityRequirements: input.authRequired ? [{ sepilotdBearer: [] }] : undefined,
    defaultInputModes: INPUT_MODES,
    defaultOutputModes: OUTPUT_MODES,
    skills: [{
      id: 'sepilotd-agent',
      name: 'sepilotd Agent',
      description: 'Handle coding, automation, research, and operational tasks using the configured sepilotd runtime and tools.',
      tags: ['coding', 'automation', 'research', 'tools'],
      examples: [
        'Review this repository change and summarize risks.',
        'Investigate a failing test and propose a fix.',
      ],
      inputModes: INPUT_MODES,
      outputModes: OUTPUT_MODES,
      securityRequirements: input.authRequired ? [{ sepilotdBearer: [] }] : undefined,
    }],
  }
}

export function agentCardEtag(card: A2AAgentCard): string {
  const hash = createHash('sha256')
    .update(JSON.stringify(card))
    .digest('hex')
    .slice(0, 16)
  return `"a2a-${hash}"`
}
