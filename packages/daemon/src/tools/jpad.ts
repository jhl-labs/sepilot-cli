import { getAbortError, isAbortError, throwIfAborted } from '../abort.js'
import { getDispatcherCompatibleFetch } from '../providers/http-timeout.js'
import type {
  ToolDefinitionRuntime,
  ToolExecutionContext,
  ToolResult,
} from './registry.js'

const DEFAULT_JPAD_BASE_URL = 'https://jpad.euno.work/api/v1'
const DEFAULT_JPAD_TIMEOUT_MS = 30_000
const MAX_JPAD_CONTENT_BYTES = 1024 * 1024
const MAX_JPAD_RECEIPT_ID_CHARS = 512
const UNSAFE_RECEIPT_ID_PATTERN = /[\u0000-\u001f\u007f]/
const MAX_JPAD_READ_SESSIONS = 256
const MAX_JPAD_READS_PER_SESSION = 64

export const JPAD_PUBLICATION_METADATA_KEY = 'jpadPublication'

export interface JpadPublicationToolMetadata {
  schemaVersion: 1
  operation: 'create' | 'update'
  outcome: 'confirmed' | 'unconfirmed'
  pageId: string | null
  workspaceId: string | null
  httpStatus: number
  completedAt: number
}

export interface JpadToolOptions {
  env?: NodeJS.ProcessEnv
  fetchImpl?: typeof fetch
  timeoutMs?: number
}

class JpadRequestError extends Error {
  constructor(
    message: string,
    readonly code: string,
  ) {
    super(message)
    this.name = 'JpadRequestError'
  }
}

interface JpadApiResponse {
  output: string
  status: number
  etag: string | null
  data: unknown
}

function boundedReceiptId(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  if (
    !trimmed
    || trimmed.length > MAX_JPAD_RECEIPT_ID_CHARS
    || UNSAFE_RECEIPT_ID_PATTERN.test(trimmed)
  ) {
    return null
  }
  return trimmed
}

function responsePageId(data: unknown): string | null {
  if (!data || typeof data !== 'object' || Array.isArray(data)) return null
  return boundedReceiptId((data as Record<string, unknown>).id)
}

export function parseJpadPublicationToolMetadata(
  value: unknown,
): JpadPublicationToolMetadata | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const record = value as Record<string, unknown>
  if (record.schemaVersion !== 1) return null
  if (record.operation !== 'create' && record.operation !== 'update') return null
  if (record.outcome !== 'confirmed' && record.outcome !== 'unconfirmed') return null
  const pageId = record.pageId === null ? null : boundedReceiptId(record.pageId)
  if (record.pageId !== null && pageId === null) return null
  const workspaceId = record.workspaceId === null
    ? null
    : boundedReceiptId(record.workspaceId)
  if (record.workspaceId !== null && workspaceId === null) return null
  if (
    !Number.isInteger(record.httpStatus)
    || (record.httpStatus as number) < 200
    || (record.httpStatus as number) > 299
  ) {
    return null
  }
  if (!Number.isSafeInteger(record.completedAt) || (record.completedAt as number) < 0) return null
  if (record.outcome === 'confirmed' && pageId === null) return null
  return {
    schemaVersion: 1,
    operation: record.operation,
    outcome: record.outcome,
    pageId,
    workspaceId,
    httpStatus: record.httpStatus as number,
    completedAt: record.completedAt as number,
  }
}

function publicationMetadata(input: {
  operation: 'create' | 'update'
  response: JpadApiResponse
  workspaceId?: string
  requestedPageId?: string
}): JpadPublicationToolMetadata {
  const returnedPageId = responsePageId(input.response.data)
  const requestedPageId = boundedReceiptId(input.requestedPageId)
  const pageId = input.operation === 'update'
    ? requestedPageId
    : returnedPageId
  const responseMatchesRequest = input.operation !== 'update'
    || returnedPageId === null
    || returnedPageId === requestedPageId
  return {
    schemaVersion: 1,
    operation: input.operation,
    outcome: pageId !== null && responseMatchesRequest ? 'confirmed' : 'unconfirmed',
    pageId,
    workspaceId: boundedReceiptId(input.workspaceId),
    httpStatus: input.response.status,
    completedAt: Date.now(),
  }
}

function resultError(error: JpadRequestError, start: number): ToolResult {
  return {
    output: error.message,
    status: 'error',
    durationMs: Date.now() - start,
    code: error.code,
  }
}

function requiredString(value: unknown, label: string, max?: number): string {
  const text = typeof value === 'string' ? value.trim() : ''
  if (!text) throw new JpadRequestError(`${label} is required.`, 'INVALID_INPUT_PERMANENT')
  if (max && text.length > max) {
    throw new JpadRequestError(`${label} must be at most ${max} characters.`, 'INVALID_INPUT_PERMANENT')
  }
  return text
}

function markdownContent(value: unknown): string {
  if (typeof value !== 'string') {
    throw new JpadRequestError('content must be Markdown text.', 'INVALID_INPUT_PERMANENT')
  }
  if (Buffer.byteLength(value, 'utf8') > MAX_JPAD_CONTENT_BYTES) {
    throw new JpadRequestError('content exceeds JPAD\'s 1 MiB API limit.', 'INVALID_INPUT_PERMANENT')
  }
  return value
}

function requirePublishConfirmation(input: Record<string, unknown>): void {
  if (input.confirmPublish !== true) {
    throw new JpadRequestError(
      'JPAD publish requires confirmPublish=true after the user has confirmed the destination and final Markdown.',
      'PUBLISH_CONFIRMATION_REQUIRED_USER',
    )
  }
}

function jpadErrorForStatus(status: number): JpadRequestError {
  if (status === 401) {
    return new JpadRequestError(
      'JPAD rejected the personal API token. Configure JPAD_PERSONAL_API_TOKEN with the required scopes.',
      'JPAD_AUTH_PERMANENT',
    )
  }
  if (status === 403) {
    return new JpadRequestError(
      'JPAD denied this workspace or page operation. Check token scope and workspace role.',
      'JPAD_PERMISSION_USER',
    )
  }
  if (status === 409) {
    return new JpadRequestError(
      'JPAD page revision changed. Read the page again, reconcile the new content, and retry with its latest ETag.',
      'JPAD_REVISION_CONFLICT_USER',
    )
  }
  if (status === 423) {
    return new JpadRequestError(
      'JPAD page is locked by another user. Wait for the lock to clear before retrying.',
      'JPAD_PAGE_LOCKED_USER',
    )
  }
  if (status === 429 || status >= 500) {
    return new JpadRequestError(
      `JPAD is temporarily unavailable (HTTP ${status}).`,
      'JPAD_UNAVAILABLE_TRANSIENT',
    )
  }
  return new JpadRequestError(`JPAD request failed (HTTP ${status}).`, 'JPAD_REQUEST_PERMANENT')
}

export function createJpadTools(options: JpadToolOptions = {}): ToolDefinitionRuntime[] {
  const env = options.env ?? process.env
  const fetchImpl = options.fetchImpl ?? getDispatcherCompatibleFetch()
  const timeoutMs = Number.isFinite(options.timeoutMs)
    ? Math.max(1, Math.trunc(options.timeoutMs!))
    : DEFAULT_JPAD_TIMEOUT_MS
  const pageReadsBySession = new Map<string, Map<string, string>>()
  const configuredBaseUrl = () => (
    env.JPAD_BASE_URL?.trim() || DEFAULT_JPAD_BASE_URL
  ).replace(/\/+$/, '')

  function pageReadsForSession(sessionId: string): Map<string, string> {
    const existing = pageReadsBySession.get(sessionId)
    if (existing) {
      pageReadsBySession.delete(sessionId)
      pageReadsBySession.set(sessionId, existing)
      return existing
    }
    if (pageReadsBySession.size >= MAX_JPAD_READ_SESSIONS) {
      const oldestSessionId = pageReadsBySession.keys().next().value as string | undefined
      if (oldestSessionId) pageReadsBySession.delete(oldestSessionId)
    }
    const created = new Map<string, string>()
    pageReadsBySession.set(sessionId, created)
    return created
  }

  function recordPageRead(
    context: ToolExecutionContext | undefined,
    pageId: string,
    etag: string | null,
  ): void {
    if (!context?.sessionId) return
    const normalizedEtag = etag?.trim()
    if (!normalizedEtag) {
      const reads = pageReadsBySession.get(context.sessionId)
      reads?.delete(pageId)
      if (reads?.size === 0) pageReadsBySession.delete(context.sessionId)
      return
    }
    const reads = pageReadsForSession(context.sessionId)
    reads.delete(pageId)
    if (reads.size >= MAX_JPAD_READS_PER_SESSION) {
      const oldestPageId = reads.keys().next().value as string | undefined
      if (oldestPageId) reads.delete(oldestPageId)
    }
    reads.set(pageId, normalizedEtag)
  }

  function consumeFreshPageRead(
    context: ToolExecutionContext | undefined,
    pageId: string,
    ifMatch: string,
  ): void {
    const sessionId = context?.sessionId
    const reads = sessionId ? pageReadsBySession.get(sessionId) : undefined
    const observedEtag = reads?.get(pageId)
    if (!sessionId || !reads || observedEtag !== ifMatch) {
      throw new JpadRequestError(
        'JPAD update requires jpad.pages.get for this page in the same session and the exact returned ETag.',
        'JPAD_FRESH_READ_REQUIRED_USER',
      )
    }
    reads.delete(pageId)
    if (reads.size === 0) pageReadsBySession.delete(sessionId)
  }

  async function request(
    method: string,
    path: string,
    contextSignal?: AbortSignal,
    payload?: Record<string, unknown>,
    ifMatch?: string,
  ): Promise<JpadApiResponse> {
    const token = env.JPAD_PERSONAL_API_TOKEN?.trim()
    if (!token) {
      throw new JpadRequestError(
        'JPAD_PERSONAL_API_TOKEN is not configured in the daemon environment.',
        'JPAD_NOT_CONFIGURED_USER',
      )
    }

    const baseUrl = configuredBaseUrl()
    const controller = new AbortController()
    let timedOut = false
    const abortFromContext = () => controller.abort(contextSignal?.reason)
    if (contextSignal?.aborted) abortFromContext()
    else contextSignal?.addEventListener('abort', abortFromContext, { once: true })
    const timeout = setTimeout(() => {
      timedOut = true
      controller.abort(new Error(`JPAD request timed out after ${timeoutMs}ms`))
    }, timeoutMs)
    timeout.unref?.()
    let removeControllerAbortListener = () => {}

    try {
      throwIfAborted(controller.signal, 'JPAD request aborted')
      const requestPromise = (async (): Promise<JpadApiResponse> => {
        const headers = new Headers({
          accept: 'application/json',
          authorization: `Bearer ${token}`,
          // JPAD's front door permits its documented Agent API clients while
          // rejecting generic automation user agents.
          'user-agent': 'curl/8.5.0 sepilotd-jpad-agent-api-v1',
        })
        let body: string | undefined
        if (payload) {
          headers.set('content-type', 'application/json')
          body = JSON.stringify(payload)
        }
        if (ifMatch) headers.set('if-match', ifMatch)

        const response = await fetchImpl(`${baseUrl}${path}`, {
          method,
          headers,
          body,
          signal: controller.signal,
        })
        if (!response.ok) throw jpadErrorForStatus(response.status)

        const responseText = await response.text()
        let data: unknown = null
        if (responseText) {
          try {
            data = JSON.parse(responseText)
          } catch {
            throw new JpadRequestError('JPAD returned invalid JSON.', 'JPAD_SCHEMA_PERMANENT')
          }
        }
        return {
          output: JSON.stringify({
            status: response.status,
            etag: response.headers.get('etag'),
            data,
          }),
          status: response.status,
          etag: response.headers.get('etag'),
          data,
        }
      })()
      // Some compiled fetch runtimes can resolve headers and then leave
      // response.text() pending even after their AbortSignal fires. Race the
      // whole request/body operation against the controller so the tool's
      // advertised timeout remains authoritative at the runtime boundary.
      const abortPromise = new Promise<never>((_resolve, reject) => {
        const onAbort = () => {
          reject(timedOut
            ? new JpadRequestError(
                `JPAD request timed out after ${timeoutMs}ms.`,
                'JPAD_TIMEOUT_TRANSIENT',
              )
            : getAbortError(contextSignal, 'JPAD request aborted'))
        }
        if (controller.signal.aborted) {
          onAbort()
          return
        }
        controller.signal.addEventListener('abort', onAbort, { once: true })
        removeControllerAbortListener = () => {
          controller.signal.removeEventListener('abort', onAbort)
        }
      })
      return await Promise.race([requestPromise, abortPromise])
    } catch (error) {
      if (error instanceof JpadRequestError) throw error
      if (timedOut) {
        throw new JpadRequestError(
          `JPAD request timed out after ${timeoutMs}ms.`,
          'JPAD_TIMEOUT_TRANSIENT',
        )
      }
      if (isAbortError(error) || contextSignal?.aborted) {
        throw getAbortError(contextSignal, 'JPAD request aborted')
      }
      throw new JpadRequestError(
        `JPAD request failed: ${error instanceof Error ? error.message : String(error)}`,
        'JPAD_UNAVAILABLE_TRANSIENT',
      )
    } finally {
      clearTimeout(timeout)
      removeControllerAbortListener()
      contextSignal?.removeEventListener('abort', abortFromContext)
    }
  }

  function readTool(
    name: string,
    description: string,
    inputSchema: Record<string, unknown>,
    path: (input: Record<string, unknown>) => string,
    canonicalReadUrlTemplates: () => readonly string[],
    onSuccess?: (
      input: Record<string, unknown>,
      response: JpadApiResponse,
      context: ToolExecutionContext | undefined,
    ) => void,
  ): ToolDefinitionRuntime {
    return {
      name,
      description,
      resumeSafety: 'replay-safe',
      scheduling: { mode: 'parallel-safe', resource: 'jpad' },
      canonicalReadUrlTemplates,
      inputSchema,
      async execute(input, context): Promise<ToolResult> {
        const start = Date.now()
        try {
          const response = await request('GET', path(input), context?.signal)
          onSuccess?.(input, response, context)
          return { output: response.output, status: 'success', durationMs: Date.now() - start }
        } catch (error) {
          if (error instanceof JpadRequestError) return resultError(error, start)
          throw error
        }
      },
    }
  }

  const noInputSchema = { type: 'object', properties: {}, additionalProperties: false }
  const workspaces = readTool(
    'jpad.workspaces',
    'List JPAD workspaces visible to the configured personal API token. Use this to disambiguate a publish destination before asking the user to confirm it.',
    noInputSchema,
    () => '/workspaces?limit=200',
    () => [`${configuredBaseUrl()}/workspaces`],
  )
  const pages = readTool(
    'jpad.pages.list',
    'List pages in one JPAD workspace so an existing destination can be found before creating a duplicate.',
    {
      type: 'object',
      additionalProperties: false,
      properties: { workspaceId: { type: 'string' } },
      required: ['workspaceId'],
    },
    (input) => `/workspaces/${encodeURIComponent(requiredString(input.workspaceId, 'workspaceId'))}/pages?limit=500`,
    () => [`${configuredBaseUrl()}/workspaces/{workspaceId}/pages`],
  )
  const getPage: ToolDefinitionRuntime = {
    ...readTool(
      'jpad.pages.get',
      'Read one JPAD page and its ETag. A successful same-session read records one fresh update attempt; pass the exact returned ETag as ifMatch so concurrent edits are never overwritten.',
      {
        type: 'object',
        additionalProperties: false,
        properties: { pageId: { type: 'string' } },
        required: ['pageId'],
      },
      (input) => `/pages/${encodeURIComponent(requiredString(input.pageId, 'pageId'))}`,
      () => [`${configuredBaseUrl()}/pages/{pageId}`],
      (input, response, context) => {
        recordPageRead(
          context,
          requiredString(input.pageId, 'pageId'),
          response.etag,
        )
      },
    ),
    researchVerification: 'direct-source',
  }

  const createPage: ToolDefinitionRuntime = {
    name: 'jpad.pages.create',
    description:
      'Publish a new Markdown page to JPAD. External write: call only after listing/disambiguating the target and the user has confirmed the workspace/parent, title, and final Markdown. Set confirmPublish=true to attest that confirmation. Omit parentId for a top-level page.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        workspaceId: { type: 'string' },
        parentId: { type: 'string' },
        title: { type: 'string', minLength: 1, maxLength: 200 },
        content: { type: 'string', description: 'Final Markdown, at most 1 MiB UTF-8' },
        confirmPublish: { type: 'boolean', description: 'True only after user confirmation' },
      },
      required: ['workspaceId', 'title', 'content', 'confirmPublish'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      try {
        requirePublishConfirmation(input)
        const workspaceId = requiredString(input.workspaceId, 'workspaceId')
        const parentId = typeof input.parentId === 'string' && input.parentId.trim()
          ? input.parentId.trim()
          : null
        const response = await request(
          'POST',
          `/workspaces/${encodeURIComponent(workspaceId)}/pages`,
          context?.signal,
          {
            title: requiredString(input.title, 'title', 200),
            content: markdownContent(input.content),
            parentId,
          },
        )
        return {
          output: response.output,
          status: 'success',
          durationMs: Date.now() - start,
          metadata: {
            [JPAD_PUBLICATION_METADATA_KEY]: publicationMetadata({
              operation: 'create',
              response,
              workspaceId,
            }),
          },
        }
      } catch (error) {
        if (error instanceof JpadRequestError) return resultError(error, start)
        throw error
      }
    },
  }

  const updatePage: ToolDefinitionRuntime = {
    name: 'jpad.pages.update',
    description:
      'Update an existing JPAD page with optimistic concurrency. External write: read first in this session, reconcile the current page, show the destination and final Markdown to the user, then set confirmPublish=true and pass the exact returned ETag as ifMatch. The read evidence is consumed by one update attempt, so read again before any retry.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        pageId: { type: 'string' },
        title: { type: 'string', minLength: 1, maxLength: 200 },
        content: { type: 'string', description: 'Final Markdown, at most 1 MiB UTF-8' },
        ifMatch: {
          type: 'string',
          description: 'Exact ETag from the latest same-session jpad.pages.get; one update attempt consumes it',
        },
        confirmPublish: { type: 'boolean', description: 'True only after user confirmation' },
      },
      required: ['pageId', 'content', 'ifMatch', 'confirmPublish'],
    },
    async execute(input, context): Promise<ToolResult> {
      const start = Date.now()
      try {
        requirePublishConfirmation(input)
        const pageId = requiredString(input.pageId, 'pageId')
        const ifMatch = requiredString(input.ifMatch, 'ifMatch')
        const payload: Record<string, unknown> = { content: markdownContent(input.content) }
        if (input.title !== undefined) payload.title = requiredString(input.title, 'title', 200)
        consumeFreshPageRead(context, pageId, ifMatch)
        const response = await request(
          'PATCH',
          `/pages/${encodeURIComponent(pageId)}`,
          context?.signal,
          payload,
          ifMatch,
        )
        return {
          output: response.output,
          status: 'success',
          durationMs: Date.now() - start,
          metadata: {
            [JPAD_PUBLICATION_METADATA_KEY]: publicationMetadata({
              operation: 'update',
              response,
              requestedPageId: pageId,
            }),
          },
        }
      } catch (error) {
        if (error instanceof JpadRequestError) return resultError(error, start)
        throw error
      }
    },
  }

  return [workspaces, pages, getPage, createPage, updatePage]
}
