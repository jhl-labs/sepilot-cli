import {
  createJobsClient,
  createMigrationClient,
  DaemonClient as SharedDaemonClient,
  type ChatOptions,
  type DaemonChatRequestControlOptions,
  type DaemonChatResult,
  type DaemonMemoryAuditEntry,
  type DaemonMemoryAuditOptions,
  type DaemonMemoryLifecycleOptions,
  type DaemonMemoryLifecycleStatus,
  type DaemonMemoryMaintenanceInput,
  type DaemonMemoryMaintenanceResult,
  type DaemonMemoryScopes,
  type DaemonMemoryScopeTransferInput,
  type DaemonMemoryScopeTransferResult,
  type DaemonMemorySecurityAuditResult,
  type JobsClient,
  type MigrationClient,
} from '@sepilotd/api-client'
import { loadDaemonToken, resolveDaemonBaseUrl } from './token.js'
import { getCliSyncChatFetch } from '../utils/sync-chat-transport.js'

import { memoryScopeFor } from './memory-scope.js'
export { resetCliDaemonScopeCache } from './memory-scope.js'

export interface CliMemorySemanticStatus {
  status: 'disabled' | 'ready' | 'backfilling' | 'degraded' | 'reindex_required'
  configuredProviderId?: string
  configuredModel?: string
  indexedProviderId?: string
  indexedModel?: string
  dimensions?: number
  pendingCount: number
  failedCount: number
  vecAvailable: boolean
  vectorBackend: 'sqlite-vec' | 'sqlite-scan' | 'qdrant' | 'opensearch' | 'elasticsearch' | 'meilisearch' | 'custom-api'
  backendAvailable: boolean
  lastError?: string
}

export interface CliMemoryReindexResult {
  started: boolean
  status: CliMemorySemanticStatus
}

export type CliMemoryAuditEntry = DaemonMemoryAuditEntry
export type CliMemoryAuditOptions = DaemonMemoryAuditOptions
export type CliMemoryLifecycleOptions = DaemonMemoryLifecycleOptions
export type CliMemoryLifecycleStatus = DaemonMemoryLifecycleStatus
export type CliMemoryMaintenanceInput = DaemonMemoryMaintenanceInput
export type CliMemoryMaintenanceResult = DaemonMemoryMaintenanceResult
export type CliMemoryScopes = DaemonMemoryScopes
export type CliMemoryScopeTransferInput = DaemonMemoryScopeTransferInput
export type CliMemoryScopeTransferResult = DaemonMemoryScopeTransferResult
export type CliMemorySecurityAuditResult = DaemonMemorySecurityAuditResult

interface CliFileMemorySection {
  title: string
  content: string
}

export interface CliFileMemorySnapshot {
  memoryPath: string
  todayNotePath: string
  yesterdayNotePath: string
  longTermMemory: string
  todayNote: string
  yesterdayNote: string
  sections: CliFileMemorySection[]
}

export interface CliFileMemorySectionUpdateResult extends CliFileMemorySection {
  deleted: boolean
}

export class DaemonClient extends SharedDaemonClient {
  public readonly jobs: JobsClient
  public readonly migration: MigrationClient
  /** Same-origin token for callers that need to build their own typed client. */
  public readonly token: string | null

  constructor(baseUrl?: string) {
    const resolvedBaseUrl = resolveDaemonBaseUrl(baseUrl)
    const token = loadDaemonToken()
    // memoryScope is lazy/cached so the daemon can isolate per-user
    // memory the same way desktop/web already do. Surface QA Round 5
    // X02 flagged the cli as the only surface not advertising
    // X-Memory-Scope-User-Id.
    super({
      baseUrl: resolvedBaseUrl,
      token,
      surface: 'cli',
      memoryScope: resolvedBaseUrl ? memoryScopeFor(resolvedBaseUrl, token) : undefined,
    })
    this.token = token
    // Standalone factory clients reuse the same origin/token. The
    // shared `DaemonClient` constructor already coerces undefined
    // `baseUrl` into the bundled default; mirror that for the
    // factories so the jobs/migration clients hit the same daemon.
    this.jobs = createJobsClient({ origin: this.baseUrl, token })
    this.migration = createMigrationClient({ origin: this.baseUrl, token })
  }

  override chat(
    message: string,
    sessionId?: string,
    options?: ChatOptions,
    request?: DaemonChatRequestControlOptions,
  ): Promise<DaemonChatResult> {
    const timeoutMs = request?.timeoutMs
    if (!timeoutMs || request.fetch) {
      return super.chat(message, sessionId, options, request)
    }
    return super.chat(message, sessionId, options, {
      ...request,
      fetch: getCliSyncChatFetch(timeoutMs),
    })
  }

  async memoryStatus(): Promise<CliMemorySemanticStatus> {
    const response = await this.request<{ data: CliMemorySemanticStatus }>(
      '/api/v1/memory/status',
      { method: 'GET' },
    )
    return response.data
  }

  async reindexMemory(): Promise<CliMemoryReindexResult> {
    const response = await this.request<{ data: CliMemoryReindexResult }>(
      '/api/v1/memory/reindex',
      { method: 'POST' },
    )
    return response.data
  }

  async memoryLifecycle(
    options?: CliMemoryLifecycleOptions,
  ): Promise<CliMemoryLifecycleStatus> {
    const params = new URLSearchParams()
    if (options?.staleAfterDays != null) {
      params.set('staleAfterDays', String(options.staleAfterDays))
    }
    if (options?.lowImportance != null) {
      params.set('lowImportance', String(options.lowImportance))
    }
    const query = params.toString()
    const response = await this.request<{ data: CliMemoryLifecycleStatus }>(
      query ? `/api/v1/memory/lifecycle?${query}` : '/api/v1/memory/lifecycle',
      { method: 'GET' },
    )
    return response.data
  }

  async memoryAudit(options?: CliMemoryAuditOptions): Promise<CliMemoryAuditEntry[]> {
    const params = new URLSearchParams()
    if (options?.memoryId) params.set('memoryId', options.memoryId)
    if (options?.limit != null) params.set('limit', String(options.limit))
    const query = params.toString()
    const response = await this.request<{ data: CliMemoryAuditEntry[] }>(
      query ? `/api/v1/memory/audit?${query}` : '/api/v1/memory/audit',
      { method: 'GET' },
    )
    return response.data
  }

  async runMemoryMaintenance(
    input: CliMemoryMaintenanceInput = {},
  ): Promise<CliMemoryMaintenanceResult> {
    const response = await this.request<{ data: CliMemoryMaintenanceResult }>(
      '/api/v1/memory/maintenance',
      { method: 'POST', body: input },
    )
    return response.data
  }

  async fileMemory(): Promise<CliFileMemorySnapshot> {
    const response = await this.request<{ data: CliFileMemorySnapshot }>(
      '/api/v1/memory/file',
      { method: 'GET' },
    )
    return response.data
  }

  async updateFileMemorySection(
    sectionTitle: string,
    content: string,
  ): Promise<CliFileMemorySectionUpdateResult> {
    const response = await this.request<{ data: CliFileMemorySectionUpdateResult }>(
      `/api/v1/memory/file/sections/${encodeURIComponent(sectionTitle)}`,
      { method: 'PUT', body: { content } },
    )
    return response.data
  }

  async deleteFileMemorySection(sectionTitle: string): Promise<{ deleted: boolean }> {
    const response = await this.request<{ data: { deleted: boolean } }>(
      `/api/v1/memory/file/sections/${encodeURIComponent(sectionTitle)}`,
      { method: 'DELETE' },
    )
    return response.data
  }
}
export { DEFAULT_DAEMON_BASE_URL } from '@sepilotd/api-client'
