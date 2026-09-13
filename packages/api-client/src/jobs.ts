/**
 * Standalone factory client for the daemon's `/api/v1/jobs/*` routes.
 *
 * This is a lightweight, fetch-injectable client used by surfaces (e.g.
 * the CLI batch facade) that want a small surface area without pulling
 * in the full `DaemonClient`. Tests inject `fetch`; production code
 * passes the real `globalThis.fetch`.
 *
 * URL convention matches the rest of the api-client: every path is
 * prefixed with `/api/v1` so it lines up with how the daemon registers
 * routes (`registerJobsRoutes(..., { prefix: '/api/v1' })`).
 */

import { ApiHttpClient } from './http.js'

const API_PREFIX = '/api/v1'

export interface JobsBatchSubmitRequest {
  items: unknown[]
  concurrency: number
  failureMode: 'continue' | 'abort'
  preserveOrder: boolean
}

export interface JobsBatchSubmitResult {
  jobId: string
  total: number
  status: string
  createdAt: number
}

export interface JobSnapshot {
  activity?: Array<{ idx: number; status: string; sessionId: string; phase: string; toolName?: string; approvalRequestId?: string; updatedAt: number }>
  id: string
  kind?: string
  status: string
  total: number
  succeeded: number
  failed: number
  canceled: number
  error?: string | null
}

export interface JobItem {
  idx: number
  status: string
  result: unknown
  error: string | null
}

export interface JobItemsPage {
  jobId: string
  status: string
  items: JobItem[]
}

export interface JobsClient {
  list(options?: { status?: string; kind?: string; limit?: number; offset?: number }): Promise<{ jobs: JobSnapshot[]; nextOffset: number | null }>
  submitBatch(req: JobsBatchSubmitRequest): Promise<JobsBatchSubmitResult>
  get(jobId: string): Promise<JobSnapshot>
  getItems(jobId: string, since: number): Promise<JobItemsPage>
  cancel(jobId: string): Promise<void>
}

export interface JobsClientDeps {
  origin: string
  token: string | null
  fetch?: typeof fetch
}

function trimOrigin(origin: string): string {
  return origin.endsWith('/') ? origin.slice(0, -1) : origin
}

function authHeader(token: string | null): Record<string, string> {
  return token ? { authorization: `Bearer ${token}` } : {}
}

export function createJobsClient(deps: JobsClientDeps): JobsClient {
  const origin = trimOrigin(deps.origin)
  const auth = authHeader(deps.token)
  const jsonHeaders = { 'content-type': 'application/json', ...auth }
  const client = new ApiHttpClient({
    baseUrl: origin,
    defaultHeaders: auth,
    fetch: deps.fetch,
  })

  return {
    async list(options = {}) {
      const query = new URLSearchParams()
      for (const [key, value] of Object.entries(options)) if (value !== undefined) query.set(key, String(value))
      return client.request(`${API_PREFIX}/jobs?${query}`, { headers: auth })
    },
    async submitBatch(req) {
      return client.request<JobsBatchSubmitResult>(`${API_PREFIX}/jobs/batch`, {
        method: 'POST',
        headers: jsonHeaders,
        body: req,
      })
    },

    async get(jobId) {
      return client.request<JobSnapshot>(
        `${API_PREFIX}/jobs/${encodeURIComponent(jobId)}`,
        { headers: auth },
      )
    },

    async getItems(jobId, since) {
      return client.request<JobItemsPage>(
        `${API_PREFIX}/jobs/${encodeURIComponent(jobId)}/items?since=${since}`,
        { headers: auth },
      )
    },

    async cancel(jobId) {
      await client.request(
        `${API_PREFIX}/jobs/${encodeURIComponent(jobId)}`,
        { method: 'DELETE', headers: auth },
      )
    },
  }
}
