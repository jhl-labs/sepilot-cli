/**
 * Standalone factory client for the daemon's `/api/v1/migration/*`
 * routes. Mirrors the shape of `createJobsClient`: a small, fetch-
 * injectable surface intended for the CLI `migrate` facade.
 *
 * URL convention matches the rest of the api-client: every path is
 * prefixed with `/api/v1` so it lines up with how the daemon
 * registers routes (`registerMigrationRoutes(..., { prefix: '/api/v1' })`).
 */

import { ApiHttpClient } from './http.js'

const API_PREFIX = '/api/v1'

export interface MigrationRunRequest {
  sourcePath: string
  steps?: string[]
  exclude?: string[]
  dryRun?: boolean
  conflict?: 'skip' | 'overwrite'
}

export interface MigrationRunResult {
  migrationId: string
  status: string
  dryRun: boolean
  steps: { name: string; status: string }[]
}

export interface MigrationStepError {
  path: string
  error: string
}

export interface MigrationStepProgress {
  name: string
  status: string
  copied: number
  skipped: number
  errors: MigrationStepError[]
}

export interface MigrationSnapshot {
  id: string
  status: string
  sourcePath: string
  steps: MigrationStepProgress[]
}

export interface MigrationReport {
  migrationId: string
  status: string
  sourcePath: string
  dryRun: boolean
  perStep: MigrationStepProgress[]
  summary: { copied: number; skipped: number; errors: number }
}

export interface MigrationClient {
  run(req: MigrationRunRequest): Promise<MigrationRunResult>
  get(id: string): Promise<MigrationSnapshot>
  getReport(id: string): Promise<MigrationReport>
  cancel(id: string): Promise<void>
}

export interface MigrationClientDeps {
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

export function createMigrationClient(
  deps: MigrationClientDeps,
): MigrationClient {
  const origin = trimOrigin(deps.origin)
  const auth = authHeader(deps.token)
  const jsonHeaders = { 'content-type': 'application/json', ...auth }
  const client = new ApiHttpClient({
    baseUrl: origin,
    defaultHeaders: auth,
    fetch: deps.fetch,
  })

  return {
    async run(req) {
      return client.request<MigrationRunResult>(`${API_PREFIX}/migration/run`, {
        method: 'POST',
        headers: jsonHeaders,
        body: req,
      })
    },

    async get(id) {
      return client.request<MigrationSnapshot>(
        `${API_PREFIX}/migration/${encodeURIComponent(id)}`,
        { headers: auth },
      )
    },

    async getReport(id) {
      return client.request<MigrationReport>(
        `${API_PREFIX}/migration/${encodeURIComponent(id)}/report`,
        { headers: auth },
      )
    },

    async cancel(id) {
      await client.request(
        `${API_PREFIX}/migration/${encodeURIComponent(id)}`,
        { method: 'DELETE', headers: auth },
      )
    },
  }
}
