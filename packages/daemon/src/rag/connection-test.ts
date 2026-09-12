import type { SepilotdConfig } from '../config/schema.js'
import { buildMemoryHttpHeaders } from '../memory/vector-backend.js'

export interface RagConnectionTestResult {
  ok: boolean
  target: 'vector-backend' | 'rerank'
  backend?: string
  url?: string
  status?: number
  durationMs: number
  message: string
}

function elapsedMs(startedAt: number): number {
  return Math.max(0, Math.round(performance.now() - startedAt))
}

function basicAuthHeader(username?: string, password?: string): string | undefined {
  if (!username || !password) return undefined
  return `Basic ${Buffer.from(`${username}:${password}`).toString('base64')}`
}

async function requestHealth({
  auth,
  body,
  headers,
  method = 'GET',
  target,
  timeoutMs = 10_000,
  url,
}: {
  auth?: {
    apiKey?: string
    authorization?: string
  }
  body?: unknown
  headers?: Record<string, string>
  method?: string
  target: RagConnectionTestResult['target']
  timeoutMs?: number
  url: string
}): Promise<RagConnectionTestResult> {
  const startedAt = performance.now()
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const requestHeaders = {
    ...(headers ?? {}),
  }
  if (auth?.authorization && !Object.keys(requestHeaders).some((key) => key.toLowerCase() === 'authorization')) {
    requestHeaders.authorization = auth.authorization
  }
  if (auth?.apiKey && !Object.keys(requestHeaders).some((key) => key.toLowerCase() === 'api-key')) {
    requestHeaders['api-key'] = auth.apiKey
  }
  if (body !== undefined && !Object.keys(requestHeaders).some((key) => key.toLowerCase() === 'content-type')) {
    requestHeaders['content-type'] = 'application/json'
  }

  try {
    const response = await fetch(url, {
      method,
      headers: requestHeaders,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    })
    const text = await response.text().catch(() => '')
    return {
      ok: response.ok,
      target,
      url,
      status: response.status,
      durationMs: elapsedMs(startedAt),
      message: response.ok
        ? `HTTP ${response.status}`
        : `HTTP ${response.status}${text ? `: ${text.slice(0, 240)}` : ''}`,
    }
  } catch (error) {
    return {
      ok: false,
      target,
      url,
      durationMs: elapsedMs(startedAt),
      message: error instanceof Error ? error.message : String(error),
    }
  } finally {
    clearTimeout(timer)
  }
}

function localResult(
  backend: string,
  ok: boolean,
  message: string,
): RagConnectionTestResult {
  return {
    ok,
    target: 'vector-backend',
    backend,
    durationMs: 0,
    message,
  }
}

export async function testRagVectorBackendConnection(
  config: SepilotdConfig | undefined,
  semanticStatus?: { backendAvailable?: boolean; lastError?: string; status?: string },
): Promise<RagConnectionTestResult> {
  const memory = config?.memory
  const backend = memory?.vectorBackend ?? 'auto'
  if (!memory) {
    return localResult(backend, false, 'Runtime config is unavailable.')
  }

  if (backend === 'auto' || backend === 'sqlite-vec' || backend === 'sqlite-scan') {
    const ok = semanticStatus?.backendAvailable !== false
    return localResult(
      backend,
      ok,
      ok
        ? `Local vector backend is ${semanticStatus?.status ?? 'available'}.`
        : semanticStatus?.lastError ?? 'Local vector backend is unavailable.',
    )
  }

  if (backend === 'qdrant') {
    const cfg = memory.qdrant
    if (!cfg?.url) return localResult(backend, false, 'memory.qdrant.url is not configured.')
    return {
      ...await requestHealth({
        auth: { apiKey: cfg.apiKey },
        target: 'vector-backend',
        url: new URL('/collections', cfg.url).toString(),
      }),
      backend,
    }
  }

  if (backend === 'opensearch') {
    const cfg = memory.opensearch
    if (!cfg?.url) return localResult(backend, false, 'memory.opensearch.url is not configured.')
    return {
      ...await requestHealth({
        auth: {
          authorization: cfg.apiKey
            ? `Bearer ${cfg.apiKey}`
            : basicAuthHeader(cfg.username, cfg.password),
        },
        target: 'vector-backend',
        url: new URL('/_cluster/health', cfg.url).toString(),
      }),
      backend,
    }
  }

  if (backend === 'elasticsearch') {
    const cfg = memory.elasticsearch
    if (!cfg?.url) return localResult(backend, false, 'memory.elasticsearch.url is not configured.')
    return {
      ...await requestHealth({
        auth: {
          authorization: cfg.apiKey
            ? `ApiKey ${cfg.apiKey}`
            : basicAuthHeader(cfg.username, cfg.password),
        },
        target: 'vector-backend',
        url: new URL('/_cluster/health', cfg.url).toString(),
      }),
      backend,
    }
  }

  if (backend === 'meilisearch') {
    const cfg = memory.meilisearch
    if (!cfg?.url) return localResult(backend, false, 'memory.meilisearch.url is not configured.')
    return {
      ...await requestHealth({
        auth: {
          authorization: cfg.apiKey ? `Bearer ${cfg.apiKey}` : undefined,
        },
        target: 'vector-backend',
        url: new URL('/health', cfg.url).toString(),
      }),
      backend,
    }
  }

  const cfg = memory.customApi
  if (!cfg?.url) return localResult(backend, false, 'memory.customApi.url is not configured.')
  return {
    ...await requestHealth({
      headers: buildMemoryHttpHeaders({
        apiKey: cfg.apiKey,
        auth: cfg.auth,
        headers: cfg.headers,
      }),
      target: 'vector-backend',
      timeoutMs: cfg.timeoutMs,
      url: new URL(cfg.healthPath ?? '/health', cfg.url).toString(),
    }),
    backend,
  }
}

export async function testRagRerankConnection(
  config: SepilotdConfig | undefined,
): Promise<RagConnectionTestResult> {
  const rerank = config?.memory.rag?.rerank
  if (!rerank?.enabled) {
    return {
      ok: true,
      target: 'rerank',
      backend: rerank?.provider ?? 'local',
      durationMs: 0,
      message: 'Rerank is disabled.',
    }
  }
  if (rerank.provider !== 'custom-api') {
    return {
      ok: true,
      target: 'rerank',
      backend: rerank.provider,
      durationMs: 0,
      message: 'Local rerank does not require a remote connection.',
    }
  }
  if (!rerank.endpoint) {
    return {
      ok: false,
      target: 'rerank',
      backend: 'custom-api',
      durationMs: 0,
      message: 'memory.rag.rerank.endpoint is not configured.',
    }
  }

  const headers = buildMemoryHttpHeaders({
    apiKey: rerank.apiKey,
    auth: rerank.auth,
    contentType: rerank.healthPath ? undefined : 'application/json',
    headers: rerank.headers,
  })
  const healthUrl = rerank.healthPath
    ? new URL(rerank.healthPath, rerank.endpoint).toString()
    : rerank.endpoint

  return {
    ...await requestHealth({
      body: rerank.healthPath
        ? undefined
        : {
            query: 'health check',
            model: rerank.model,
            hits: [],
          },
      headers,
      method: rerank.healthPath ? 'GET' : 'POST',
      target: 'rerank',
      timeoutMs: rerank.timeoutMs,
      url: healthUrl,
    }),
    backend: 'custom-api',
  }
}
