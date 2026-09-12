import { Buffer } from 'node:buffer'
import { randomUUID } from 'node:crypto'
import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { bindCapability } from '../server/capabilities/bind.js'
import {
  buildSseResponseHeaders,
  registerSseDisconnectHandler,
} from '../server/sse-response.js'
import { createTeamDocsRepo } from './repo.js'
import {
  TeamDocsConfigInput,
  type TeamDocsConfig,
  type TeamDocsDocument,
} from './schema.js'
import { safeIdSchema } from '../utils/safe-id.js'

const TEAM_DOCS_BACKGROUND_TICK_MS = 1000
const REDACTED_TOKEN = '***redacted***'

interface GitHubContentItem {
  type: 'file' | 'dir'
  path: string
  sha: string | null
  size: number
  download_url?: string | null
  content?: string
  encoding?: string
}

interface SyncDocument {
  path: string
  sha: string | null
  size: number
  content: string
}

type TeamDocsWatchPayload =
  | {
      type: 'snapshot'
      items: TeamDocsConfig[]
      documentsByConfigId: Record<string, TeamDocsDocument[]>
    }
  | { type: 'heartbeat'; timestamp: string }

const docExtPattern = /\.(md|mdx|txt|rst|adoc|markdown)$/i

function publicConfig(config: TeamDocsConfig | null): TeamDocsConfig | null {
  if (!config) return null
  return {
    ...config,
    token: REDACTED_TOKEN,
  }
}

function publicConfigs(configs: TeamDocsConfig[]): TeamDocsConfig[] {
  return configs.map((config) => publicConfig(config) as TeamDocsConfig)
}

function publicActionResult<T extends { config: TeamDocsConfig | null }>(
  result: T,
): T {
  return {
    ...result,
    config: publicConfig(result.config),
  }
}

function repoPath(config: TeamDocsConfig): string {
  return `/repos/${encodeURIComponent(config.owner)}/${encodeURIComponent(config.repo)}`
}

function normalizeDocsPath(value: string): string {
  return value
    .replace(/\\/g, '/')
    .split('/')
    .filter((segment) => segment.length > 0)
    .join('/')
}

function encodePath(value: string): string {
  return normalizeDocsPath(value)
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/')
}

function githubApiBaseUrl(config: TeamDocsConfig): string {
  if (config.serverType === 'ghes') {
    const trimmed = config.ghesUrl.trim().replace(/\/+$/, '')
    if (!trimmed) throw new Error('GHES URL is required')
    return trimmed.endsWith('/api/v3') ? trimmed : `${trimmed}/api/v3`
  }
  return 'https://api.github.com'
}

function githubHeaders(config: TeamDocsConfig): HeadersInit {
  return {
    accept: 'application/vnd.github+json',
    authorization: `Bearer ${config.token}`,
    'user-agent': 'sepilotd-team-docs',
    'x-github-api-version': '2022-11-28',
  }
}

async function githubJson<T>(
  config: TeamDocsConfig,
  path: string,
): Promise<T> {
  const response = await fetch(`${githubApiBaseUrl(config)}${path}`, {
    headers: githubHeaders(config),
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(
      `GitHub request failed (${response.status}): ${text || response.statusText}`,
    )
  }
  return (await response.json()) as T
}

async function githubText(
  config: TeamDocsConfig,
  url: string,
): Promise<string> {
  const response = await fetch(url, {
    headers: githubHeaders(config),
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(
      `GitHub file fetch failed (${response.status}): ${text || response.statusText}`,
    )
  }
  return response.text()
}

async function listContentsRecursive(
  config: TeamDocsConfig,
  targetPath: string,
): Promise<GitHubContentItem[]> {
  const normalizedPath = normalizeDocsPath(targetPath)
  const suffix = normalizedPath ? `/${encodePath(normalizedPath)}` : ''
  const payload = await githubJson<GitHubContentItem | GitHubContentItem[]>(
    config,
    `${repoPath(config)}/contents${suffix}?ref=${encodeURIComponent(config.branch)}`,
  )

  if (!Array.isArray(payload)) return [payload]

  const items: GitHubContentItem[] = []
  for (const item of payload) {
    if (item.type === 'dir') {
      items.push(...(await listContentsRecursive(config, item.path)))
      continue
    }
    items.push(item)
  }
  return items
}

async function fetchDocumentContent(
  config: TeamDocsConfig,
  file: GitHubContentItem,
): Promise<string> {
  if (file.download_url) {
    return githubText(config, file.download_url)
  }

  const payload = await githubJson<GitHubContentItem>(
    config,
    `${repoPath(config)}/contents/${encodePath(file.path)}?ref=${encodeURIComponent(config.branch)}`,
  )

  if (payload.encoding === 'base64' && typeof payload.content === 'string') {
    return Buffer.from(payload.content.replace(/\n/g, ''), 'base64').toString(
      'utf-8',
    )
  }

  if (typeof payload.content === 'string') {
    return payload.content
  }

  throw new Error(`Unsupported document payload for ${file.path}`)
}

async function verifyConnection(config: TeamDocsConfig): Promise<string> {
  const repo = await githubJson<{ full_name: string }>(
    config,
    `${repoPath(config)}`,
  )
  await githubJson(
    config,
    `${repoPath(config)}/branches/${encodeURIComponent(config.branch)}`,
  )
  return `${repo.full_name} · ${config.branch}`
}

async function syncConfig(config: TeamDocsConfig): Promise<SyncDocument[]> {
  const candidates = await listContentsRecursive(config, config.docsPath)
  const files = candidates.filter(
    (candidate) =>
      candidate.type === 'file' && docExtPattern.test(candidate.path),
  )

  const documents: SyncDocument[] = []
  for (const file of files) {
    documents.push({
      path: file.path,
      sha: file.sha ?? null,
      size: file.size ?? 0,
      content: await fetchDocumentContent(config, file),
    })
  }
  return documents
}

async function syncTeamDocsConfig(
  repo: ReturnType<typeof createTeamDocsRepo>,
  config: TeamDocsConfig,
) {
  try {
    const documents = await syncConfig(config)
    repo.replaceDocuments(config.id, documents)
    return {
      success: true,
      message: `${documents.length} documents synced`,
      config: repo.markSyncResult({
        id: config.id,
        status: 'success',
        error: null,
        syncedDocuments: documents.length,
      }),
    }
  } catch (error) {
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Sync failed',
      config: repo.markSyncResult({
        id: config.id,
        status: 'error',
        error: error instanceof Error ? error.message : 'Sync failed',
      }),
    }
  }
}

function shouldAutoSync(config: TeamDocsConfig): boolean {
  if (!config.enabled || !config.autoSync) return false
  if (config.lastSyncAt === null) return true
  return Date.now() - config.lastSyncAt >= config.syncInterval * 60_000
}

export async function registerTeamDocsRoutes(
  app: FastifyInstance,
): Promise<void> {
  const repo = createTeamDocsRepo()
  const watchSubscribers = new Set<(payload: TeamDocsWatchPayload) => void>()
  const syncTasks = new Map<
    string,
    Promise<Awaited<ReturnType<typeof syncTeamDocsConfig>>>
  >()

  function buildWatchSnapshot(): TeamDocsWatchPayload {
    const items = repo.list()
    const documentsByConfigId: Record<string, TeamDocsDocument[]> = {}
    for (const item of items) {
      documentsByConfigId[item.id] = repo.listDocuments(item.id, 100)
    }
    return {
      type: 'snapshot',
      items: publicConfigs(items),
      documentsByConfigId,
    }
  }

  function publishWatchSnapshot(): void {
    if (watchSubscribers.size === 0) return
    const payload = buildWatchSnapshot()
    for (const subscriber of watchSubscribers) subscriber(payload)
  }

  async function runConfigSync(
    config: TeamDocsConfig,
    options?: { background?: boolean },
  ): Promise<Awaited<ReturnType<typeof syncTeamDocsConfig>> | null> {
    if (options?.background && !shouldAutoSync(config)) {
      return null
    }

    const existing = syncTasks.get(config.id)
    if (existing) {
      return existing
    }

    const task = syncTeamDocsConfig(repo, config)
    syncTasks.set(config.id, task)
    try {
      const result = await task
      publishWatchSnapshot()
      return result
    } finally {
      if (syncTasks.get(config.id) === task) {
        syncTasks.delete(config.id)
      }
    }
  }

  async function syncDueConfigs(): Promise<void> {
    for (const config of repo.list()) {
      if (!shouldAutoSync(config)) continue
      await runConfigSync(config, { background: true })
    }
  }

  const timer = setInterval(() => {
    void syncDueConfigs()
  }, TEAM_DOCS_BACKGROUND_TICK_MS)
  app.addHook('onClose', async () => {
    clearInterval(timer)
    await Promise.allSettled([...syncTasks.values()])
  })

  await bindCapability(
    app,
    {
      name: 'team-docs',
      version: '1',
      methods: [
        { method: 'GET', path: '/team-docs' },
        { method: 'GET', path: '/team-docs/watch' },
        { method: 'POST', path: '/team-docs' },
        { method: 'DELETE', path: '/team-docs/:id' },
        { method: 'GET', path: '/team-docs/:id/documents' },
        { method: 'GET', path: '/team-docs/:id/document' },
        { method: 'POST', path: '/team-docs/:id/test-connection' },
        { method: 'POST', path: '/team-docs/:id/sync' },
        { method: 'POST', path: '/team-docs/sync-all' },
      ],
    },
    async (a) => {
      a.get('/team-docs', async () => publicConfigs(repo.list()))

      a.get('/team-docs/watch', async (req, reply) => {
        reply.hijack()
        reply.raw.writeHead(200, buildSseResponseHeaders(req, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
          'X-Request-ID': req.requestId ?? randomUUID(),
        }))

        let closed = false
        const send = (payload: TeamDocsWatchPayload) => {
          if (closed) return
          reply.raw.write(
            `event: team-docs\ndata: ${JSON.stringify(payload)}\n\n`,
          )
        }
        const heartbeat = setInterval(() => {
          send({ type: 'heartbeat', timestamp: new Date().toISOString() })
        }, 15_000)
        heartbeat.unref?.()
        const close = () => {
          if (closed) return
          closed = true
          watchSubscribers.delete(send)
          clearInterval(heartbeat)
          if (!reply.raw.destroyed && !reply.raw.writableEnded) {
            reply.raw.end()
          }
        }

        watchSubscribers.add(send)
        registerSseDisconnectHandler(req, reply, close)
        send(buildWatchSnapshot())
      })

      a.post('/team-docs', async (req, reply) => {
        const parsed = TeamDocsConfigInput.safeParse(req.body)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        const existing = parsed.data.id ? repo.get(parsed.data.id) : null
        const token = parsed.data.token === REDACTED_TOKEN
          ? existing?.token ?? ''
          : parsed.data.token
        if (!token.trim()) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: 'GitHub token is required',
            retriable: false,
          })
          return reply
        }
        const config = repo.upsert({ ...parsed.data, token })
        publishWatchSnapshot()
        return publicConfig(config)
      })

      a.delete('/team-docs/:id', async (req, reply) => {
        const parsed = z.object({ id: safeIdSchema }).safeParse(req.params)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        repo.remove(parsed.data.id)
        publishWatchSnapshot()
        return { ok: true }
      })

      a.get('/team-docs/:id/documents', async (req, reply) => {
        const parsed = z.object({ id: safeIdSchema }).safeParse(req.params)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        return repo.listDocuments(parsed.data.id, 100)
      })

      a.get('/team-docs/:id/document', async (req, reply) => {
        const params = z.object({ id: safeIdSchema }).safeParse(req.params)
        const query = z.object({ path: z.string().min(1) }).safeParse(req.query)
        if (!params.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: params.error.message,
            retriable: false,
          })
          return reply
        }
        if (!query.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: query.error.message,
            retriable: false,
          })
          return reply
        }
        const document = repo.readDocument(params.data.id, query.data.path)
        if (!document) {
          void reply.status(404).send({
            code: 'NOT_FOUND',
            message: 'Team docs document not found',
            retriable: false,
          })
          return reply
        }
        return document
      })

      a.post('/team-docs/:id/test-connection', async (req, reply) => {
        const parsed = z.object({ id: safeIdSchema }).safeParse(req.params)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        const config = repo.get(parsed.data.id)
        if (!config) {
          void reply.status(404).send({
            code: 'NOT_FOUND',
            message: 'Team docs config not found',
            retriable: false,
          })
          return reply
        }
        try {
          const target = await verifyConnection(config)
          const nextConfig = repo.markTested(config.id)
          publishWatchSnapshot()
          return publicActionResult({
            success: true,
            message: `Connected to ${target}`,
            config: nextConfig,
          })
        } catch (error) {
          const nextConfig = repo.markTested(config.id)
          publishWatchSnapshot()
          return publicActionResult({
            success: false,
            message:
              error instanceof Error ? error.message : 'Connection test failed',
            config: nextConfig,
          })
        }
      })

      a.post('/team-docs/:id/sync', async (req, reply) => {
        const parsed = z.object({ id: safeIdSchema }).safeParse(req.params)
        if (!parsed.success) {
          void reply.status(400).send({
            code: 'INVALID_REQUEST',
            message: parsed.error.message,
            retriable: false,
          })
          return reply
        }
        const config = repo.get(parsed.data.id)
        if (!config) {
          void reply.status(404).send({
            code: 'NOT_FOUND',
            message: 'Team docs config not found',
            retriable: false,
          })
          return reply
        }

        const result = await runConfigSync(config)
        return result ? publicActionResult(result) : result
      })

      a.post('/team-docs/sync-all', async () => {
        const configs = repo.list().filter((candidate) => candidate.enabled)
        let succeeded = 0
        let failed = 0

        for (const config of configs) {
          const result = await runConfigSync(config)
          if (result?.success) {
            succeeded += 1
          } else {
            failed += 1
          }
        }

        return {
          success: failed === 0,
          total: configs.length,
          succeeded,
          failed,
          items: publicConfigs(repo.list()),
        }
      })
    },
  )
}

export const __testables = {
  fetchDocumentContent,
  githubApiBaseUrl,
  githubHeaders,
  listContentsRecursive,
  normalizeDocsPath,
  repoPath,
  shouldAutoSync,
  syncConfig,
  verifyConnection,
}
