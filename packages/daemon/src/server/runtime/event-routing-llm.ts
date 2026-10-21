import { createHash } from 'node:crypto'
import { realpathSync } from 'node:fs'
import { realpath, stat } from 'node:fs/promises'
import { dirname, isAbsolute, relative, resolve } from 'node:path'
import type { LlmTurnRequest, LlmTurnResult, LlmTurnRunner } from '../../event-routing/llm-runner.js'
import { EVENT_ROUTING_SESSION_TAG, EVENT_ROUTING_SURFACE } from '../../event-routing/origin.js'
import type { ToolRegistry } from '../../tools/registry.js'
import { CHAT_TOOL_NAMES_MAX } from '../routes/chat-schema.js'

export { EVENT_ROUTING_SESSION_TAG, EVENT_ROUTING_SURFACE }

const MAX_TAG_LENGTH = 64
const CANCEL_RETRY_INTERVAL_MS = 250
const CANCEL_RETRY_BUDGET_MS = 10_000

export interface InjectLlmRunnerDeps {
  inject: (opts: {
    method: 'POST'
    url: string
    headers: Record<string, string>
    payload: Record<string, unknown>
  }) => Promise<{ statusCode: number; body: string }>
  authToken: () => string | null | undefined
  cancelRun: (sessionId: string) => Promise<unknown>
  setSessionTitle: (sessionId: string, title: string) => Promise<unknown>
  toolCeiling: (scope: UntrustedTurnScope) => string[]
  egressAllowlistActive: () => boolean
  resolveProjectWorkspace: (projectId: string) => Promise<string | undefined>
  workspaceRootProblem: (root: string) => string | undefined
}

export function realpathOrSelf(path: string): string {
  try {
    return realpathSync(path)
  } catch {
    return path
  }
}

function isSameOrInside(candidate: string, parent: string): boolean {
  const path = relative(resolve(parent), resolve(candidate))
  return path === '' || (!path.startsWith('..') && !isAbsolute(path))
}

export function untrustedWorkspaceRootProblem(
  root: string,
  scope: { home: string; dataDir: string },
): string | undefined {
  const normalized = resolve(root)
  if (dirname(normalized) === normalized) return 'is the filesystem root'
  if (normalized === resolve(scope.home)) return 'is the home directory'
  if (isSameOrInside(scope.home, normalized)) return 'contains the home directory'
  if (isSameOrInside(normalized, scope.dataDir)) return 'overlaps the daemon data directory'
  if (isSameOrInside(scope.dataDir, normalized)) return 'contains the daemon data directory'
  return undefined
}

export async function projectWorkspaceDirectory(workingDirectory: string | undefined): Promise<string | undefined> {
  const directory = workingDirectory?.trim()
  if (!directory || !isAbsolute(directory)) return undefined
  try {
    const real = await realpath(directory)
    return (await stat(real)).isDirectory() ? real : undefined
  } catch {
    return undefined
  }
}

export interface UntrustedTurnScope {
  workspace: boolean
  network: boolean
}

export function untrustedTurnToolNames(
  registry: Pick<ToolRegistry, 'list' | 'securityDescriptor'>,
  scope: UntrustedTurnScope,
): string[] {
  return registry.list()
    .map((tool) => tool.name)
    .filter((name) => {
      const descriptor = registry.securityDescriptor(name)
      if (descriptor.effect !== 'observe' || descriptor.writesArtifact) return false
      if (descriptor.reach === 'workspace') return scope.workspace
      if (descriptor.reach === 'network') return scope.network && descriptor.egressGuard === 'allowlist'
      return false
    })
    .sort()
}

function parseBody(body: string): { content?: unknown; code?: unknown } {
  try {
    const parsed = JSON.parse(body) as { data?: { content?: unknown }; error?: { code?: unknown } }
    return { content: parsed.data?.content, code: parsed.error?.code }
  } catch {
    return {}
  }
}

function errorCode(code: unknown): string {
  return typeof code === 'string' && /^[A-Z0-9_]{1,64}$/.test(code) ? code : 'UNKNOWN'
}

function isRetryableStatus(statusCode: number): boolean {
  return statusCode === 503 || statusCode === 429 || statusCode === 500 || statusCode === 502 || statusCode === 504
}

function ruleTag(ruleId: string): string {
  const tag = `event-rule:${ruleId}`
  if (tag.length <= MAX_TAG_LENGTH) return tag
  const prefix = 'event-rule:'
  const hash = createHash('sha256').update(ruleId).digest('hex').slice(0, MAX_TAG_LENGTH - prefix.length)
  return `${prefix}${hash}`
}

export function createInjectLlmTurnRunner(deps: InjectLlmRunnerDeps): LlmTurnRunner {
  return (request: LlmTurnRequest): Promise<LlmTurnResult> => {
    if (request.bot && request.network && !deps.egressAllowlistActive()) {
      return Promise.resolve(terminal('network access requires security.egressAllowlist'))
    }
    const toolNames = request.bot
      ? deps.toolCeiling({ workspace: Boolean(request.projectId), network: Boolean(request.network) })
      : []
    if (toolNames.length > CHAT_TOOL_NAMES_MAX) {
      return Promise.resolve(terminal(
        `untrusted-turn tool ceiling has ${toolNames.length} tools, over the chat limit of ${CHAT_TOOL_NAMES_MAX}`,
      ))
    }
    const payload: Record<string, unknown> = {
      message: request.message,
      sessionId: request.sessionId,
      autonomy: 'readonly',
      ...(request.bot ? { bot: request.bot } : { mode: 'instant' }),
      intentRouting: { enabled: false },
      toolNames,
      localContext: 'none',
      tags: [EVENT_ROUTING_SESSION_TAG, ruleTag(request.ruleId)],
    }
    if (!request.projectId) return runTurn(deps, request, payload)
    const projectId = request.projectId
    const startedAt = Date.now()
    return boundedByRun(request, deps.resolveProjectWorkspace(projectId)).then((resolution) => {
      if ('result' in resolution) return resolution.result
      const workspaceRoot = resolution.value
      if (!workspaceRoot) return terminal(`project ${projectId} has no workspace directory`)
      const problem = deps.workspaceRootProblem(workspaceRoot)
      if (problem) return terminal(`project ${projectId} workspace ${problem}`)
      const remainingMs = request.timeoutMs - (Date.now() - startedAt)
      if (remainingMs <= 0) return timeoutResult(request)
      return runTurn(deps, request, { ...payload, projectId, workspaceRoot }, remainingMs)
    })
  }
}

function timeoutResult(request: LlmTurnRequest): LlmTurnResult {
  return terminal(`LLM run timed out after ${request.timeoutMs}ms`)
}

function abortResult(request: LlmTurnRequest): LlmTurnResult {
  const reason = request.signal.reason as { name?: string } | undefined
  const isTimeout = typeof reason === 'object' && reason?.name === 'TimeoutError'
  return isTimeout ? timeoutResult(request) : { ok: false, retryable: true, error: 'aborted' }
}

function boundedByRun<T>(
  request: LlmTurnRequest,
  step: Promise<T>,
): Promise<{ value: T | undefined } | { result: LlmTurnResult }> {
  if (request.signal.aborted) return Promise.resolve({ result: abortResult(request) })
  return new Promise((resolveStep) => {
    const finish = (outcome: { value: T | undefined } | { result: LlmTurnResult }) => {
      clearTimeout(timer)
      request.signal.removeEventListener('abort', onAbort)
      resolveStep(outcome)
    }
    const onAbort = () => finish({ result: abortResult(request) })
    const timer = setTimeout(() => finish({ result: timeoutResult(request) }), request.timeoutMs)
    request.signal.addEventListener('abort', onAbort, { once: true })
    step.then((value) => finish({ value }), () => finish({ value: undefined }))
  })
}

function terminal(error: string): LlmTurnResult {
  return { ok: false, retryable: false, error }
}

function runTurn(
  deps: InjectLlmRunnerDeps,
  request: LlmTurnRequest,
  payload: Record<string, unknown>,
  budgetMs: number = request.timeoutMs,
): Promise<LlmTurnResult> {
  const headers: Record<string, string> = {}
  const token = deps.authToken()
  if (token) headers.authorization = `Bearer ${token}`
  headers['x-sepilotd-surface'] = EVENT_ROUTING_SURFACE
  return new Promise<LlmTurnResult>((resolve) => {
    let settled = false
    let injectSettled = false

    const settle = (result: LlmTurnResult) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      request.signal.removeEventListener('abort', onAbort)
      resolve(result)
    }

    const retryCancel = () => {
      const deadline = Date.now() + CANCEL_RETRY_BUDGET_MS
      const attempt = () => {
        if (injectSettled) return
        void deps.cancelRun(request.sessionId)
          .catch(() => false)
          .then((cancelled) => {
            if (cancelled || injectSettled) return
            if (Date.now() < deadline) setTimeout(attempt, CANCEL_RETRY_INTERVAL_MS)
          })
      }
      attempt()
    }

    const onAbort = () => {
      settle(abortResult(request))
      retryCancel()
    }

    const timer = setTimeout(() => {
      settle(timeoutResult(request))
      retryCancel()
    }, budgetMs)

    if (request.signal.aborted) {
      onAbort()
      return
    }
    request.signal.addEventListener('abort', onAbort, { once: true })

    deps.inject({ method: 'POST', url: '/api/v1/chat', headers, payload }).then(
      (response) => {
        injectSettled = true
        if (settled) return
        const body = parseBody(response.body)
        if (isRetryableStatus(response.statusCode)) {
          settle({ ok: false, retryable: true, error: `chat unavailable (${response.statusCode} ${errorCode(body.code)})` })
          return
        }
        if (response.statusCode >= 400) {
          settle({ ok: false, retryable: false, error: `chat rejected (${response.statusCode} ${errorCode(body.code)})` })
          return
        }
        if (typeof body.content !== 'string' || !body.content.trim()) {
          settle({ ok: false, retryable: false, error: 'LLM returned no text' })
          return
        }
        const text = body.content.trim()
        settle({ ok: true, text })
        void deps.setSessionTitle(request.sessionId, request.title).catch(() => undefined)
      },
      (error) => {
        injectSettled = true
        if (settled) return
        settle({
          ok: false,
          retryable: true,
          error: error instanceof Error ? error.message.slice(0, 300) : 'chat request failed',
        })
      },
    )
  })
}
