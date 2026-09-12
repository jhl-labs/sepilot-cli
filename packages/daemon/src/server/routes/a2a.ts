import type { FastifyInstance, FastifyRequest } from 'fastify'
import { isIP } from 'node:net'
import { A2A_PROTOCOL_VERSION } from '../../a2a/types.js'
import {
  A2AService,
  A2ATaskStore,
  agentCardEtag,
  buildA2AAgentCard,
  type A2ADispatchInput,
  type A2ADispatchResult,
} from '../../a2a/server.js'
import '../fastify-types.js'

export interface A2ARoutesDeps {
  taskStore: () => A2ATaskStore
  dispatch: (input: A2ADispatchInput) => Promise<A2ADispatchResult>
  authRequired: () => boolean
  agentVersion?: () => string | undefined
  publicBaseUrl?: () => string | undefined
}

function firstHeader(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value
}

function normalizeConfiguredBaseUrl(value: string | undefined): string | undefined {
  const configured = value?.trim()
  if (!configured) return undefined

  try {
    const origin = new URL(configured)
    if (
      (origin.protocol !== 'http:' && origin.protocol !== 'https:')
      || !origin.hostname
      || origin.username
      || origin.password
      || (origin.pathname !== '/' && origin.pathname !== '')
      || origin.search
      || origin.hash
    ) {
      throw new Error('invalid origin')
    }
    return origin.origin
  } catch {
    throw new Error('SEPILOTD_A2A_PUBLIC_BASE_URL must be an http(s) origin without credentials or a path')
  }
}

function isSafeDirectHostname(hostname: string): boolean {
  const normalized = hostname.replace(/^\[|\]$/gu, '').toLowerCase()
  if (normalized === 'localhost') return true
  const family = isIP(normalized)
  if (family === 4) {
    const [a, b] = normalized.split('.').map(Number) as [number, number, number, number]
    return (
      a === 127 ||
      a === 10 ||
      (a === 172 && b >= 16 && b <= 31) ||
      (a === 192 && b === 168)
    )
  }
  if (family === 6) {
    return normalized === '::1' || normalized.startsWith('fc') || normalized.startsWith('fd')
  }
  return false
}

function requestBaseUrl(request: FastifyRequest, configuredBaseUrl?: string): string {
  const configured = normalizeConfiguredBaseUrl(configuredBaseUrl)
  if (configured) return configured

  const requestProtocol = request.protocol === 'https' ? 'https' : 'http'
  const host = firstHeader(request.headers.host)?.trim() || 'localhost'
  try {
    const origin = new URL(`${requestProtocol}://${host}`)
    if (
      !origin.hostname ||
      !isSafeDirectHostname(origin.hostname) ||
      origin.username ||
      origin.password ||
      origin.pathname !== '/' ||
      origin.search ||
      origin.hash
    ) {
      return `${requestProtocol}://localhost`
    }
    return origin.origin
  } catch {
    return `${requestProtocol}://localhost`
  }
}

function requestA2AVersion(request: FastifyRequest): string {
  const raw = firstHeader(request.headers['a2a-version'])
  return raw?.trim() ? raw.trim() : '0.3'
}

function isStreamingMethod(body: unknown): boolean {
  if (!body || typeof body !== 'object' || !('method' in body)) return false
  const method = (body as { method?: unknown }).method
  return method === 'SendStreamingMessage' || method === 'SubscribeToTask'
}

export function registerA2ARoutes(app: FastifyInstance, deps: A2ARoutesDeps): void {
  const sendAgentCard = async (
    request: FastifyRequest,
    reply: {
      header(name: string, value: string): unknown
      status(code: number): { send(payload?: unknown): unknown }
      send(payload: unknown): unknown
    },
  ) => {
    const card = buildA2AAgentCard({
      baseUrl: requestBaseUrl(request, deps.publicBaseUrl?.()),
      authRequired: deps.authRequired(),
      version: deps.agentVersion?.(),
    })
    const etag = agentCardEtag(card)
    // The card contains an origin. Do not let a shared intermediary reuse a
    // Host-derived card for another client even when a proxy is misconfigured.
    reply.header('Cache-Control', 'no-store')
    reply.header('ETag', etag)
    reply.header('A2A-Version', A2A_PROTOCOL_VERSION)
    if (firstHeader(request.headers['if-none-match']) === etag) {
      return reply.status(304).send()
    }
    return reply.send(card)
  }

  app.get('/.well-known/agent-card.json', sendAgentCard)
  app.get('/.well-known/agent.json', sendAgentCard)

  app.post('/api/v1/a2a', async (request, reply) => {
    const service = new A2AService({
      taskStore: deps.taskStore(),
      dispatch: deps.dispatch,
    })
    if (isStreamingMethod(request.body)) {
      reply.hijack()
      reply.raw.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        'A2A-Version': A2A_PROTOCOL_VERSION,
      })
      // Abort the underlying agent run when the client disconnects so a
      // dropped SSE consumer does not leave a run executing to completion.
      const abortController = new AbortController()
      const onClose = () => abortController.abort()
      reply.raw.on('close', onClose)
      try {
        for await (const response of service.handleJsonRpcStream(request.body, {
          version: requestA2AVersion(request),
          signal: abortController.signal,
        })) {
          if (abortController.signal.aborted) break
          reply.raw.write(`data: ${JSON.stringify(response)}\n\n`)
        }
      } finally {
        reply.raw.off('close', onClose)
        reply.raw.end()
      }
      return reply
    }
    const response = await service.handleJsonRpc(request.body, {
      version: requestA2AVersion(request),
    })
    reply.header('A2A-Version', A2A_PROTOCOL_VERSION)
    return reply.send(response)
  })
}

/**
 * Delegation category external A2A callers run under. It constrains the
 * subagent to an empty tool allowlist so an external principal cannot inherit
 * local filesystem, memory, process, or introspection capabilities.
 */
export const A2A_DISPATCH_CATEGORY = 'a2a-external'

interface A2ASubagentDispatcher {
  dispatch(input: {
    prompt: string
    system?: string
    category?: string
    parentSessionId?: string
    signal?: AbortSignal
  }): Promise<{
    output: string
    status: string
    sessionId?: string
    error?: string
  }>
}

export function createA2ADispatch(
  getDispatcher: () => A2ASubagentDispatcher | undefined,
): (input: A2ADispatchInput) => Promise<A2ADispatchResult> {
  return async ({ prompt, taskId, contextId, metadata, signal }) => {
    const dispatcher = getDispatcher()
    if (!dispatcher) {
      return {
        output: '',
        status: 'failed',
        error: 'A2A runtime is not initialized',
      }
    }
    const result = await dispatcher.dispatch({
      prompt,
      // Enforced (not just prompted): external requests receive no local tools.
      category: A2A_DISPATCH_CATEGORY,
      system: [
        'This task arrived through Agent2Agent (A2A).',
        'Treat the caller as an external agent principal.',
        'Do not expose secrets, credentials, or local private state unless explicitly provided in the A2A message.',
      ].join(' '),
      parentSessionId: `a2a:${contextId}:${taskId}`,
      signal,
    })
    return {
      output: result.output,
      status: result.status === 'completed' ? 'completed' : 'failed',
      sessionId: result.sessionId,
      error: result.error ?? (metadata?.errorMessage as string | undefined),
    }
  }
}

export async function a2aRoutes(app: FastifyInstance) {
  registerA2ARoutes(app, {
    taskStore: () => app.runtime?.a2aTaskStore ?? new A2ATaskStore(),
    authRequired: () => Boolean(app.authTokenRequired || app.authToken),
    agentVersion: () => process.env.npm_package_version,
    publicBaseUrl: () => process.env.SEPILOTD_A2A_PUBLIC_BASE_URL,
    dispatch: createA2ADispatch(
      () => app.runtime?.subagentDispatcher as A2ASubagentDispatcher | undefined,
    ),
  })
}
